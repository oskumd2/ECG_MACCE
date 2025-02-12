#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np
import os
import sys
import joblib
import pickle
from model_code import *
from model.blocks import FinalModel
from scipy import signal
from scipy.stats import zscore
from scipy.optimize import differential_evolution
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
import wfdb
from tqdm import tqdm
import warnings
import pandas as pd
import random
from skmultilearn.model_selection import iterative_train_test_split
from datetime import datetime
import copy
from pathlib import Path
from evaluate_model import load_weights #, compute_challenge_metric

from lxml import etree as ET
import base64

np.random.seed(0)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
bsize = 16
model_path = None

TRAIN_DATA_CSV = 'C:/rsrch/240923_ecg_ich/data_labels_M_N.csv'
TRAIN_DATA_FOLDER = 'C:/rsrch/240801_ecg_mace/data/'
################################################################################
#
# Training function
#
################################################################################


def find_thresholds(filename, model_directory):
    with open(filename, 'rb') as handle:
        models = pickle.load(handle)
        train_files = pickle.load(handle)
        valid_files = pickle.load(handle)
        classes = pickle.load(handle)
        lossweights = pickle.load(handle)

    results = pd.DataFrame(models)
    results.drop(columns=['model'], inplace=True)

    model_idx = np.argmax(results[:]['valid_auprc'])
    t = results.iloc[model_idx]['valid_targets']
    y = results.iloc[model_idx]['valid_outputs']

    fpr, tpr, thr = roc_curve(y_true=t[:, 1], y_score=y[:, 1])
    idx = np.argmax(tpr - fpr)
    
    select4deployment(models[model_idx]['model'], thresholds=thr[idx],
                      classes=classes, info='', model_directory=model_directory)


def select4deployment(state_dict, thresholds, classes, info, model_directory):
    select4deployment.calls += 1
    name = Path(model_directory, f'MODEL_{select4deployment.calls}.pickle') 
    with open(name, 'wb') as handle:
        model = FinalModel(num_classes=2) # 26
        model.load_state_dict(state_dict)
        model.cpu()
        model.eval()

        pickle.dump({'state_dict': model.state_dict(),
                     'classes': classes,
                     'thresholds': thresholds,
                     'info': info}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


class mytqdm(tqdm):
    def __init__(self, dataset):
        super(mytqdm, self).__init__(dataset, ncols=0)
        self.alpha = 0.99
        self._val = None

    def set_postfix(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.data.cpu().numpy()
        if self._val is None:
            self._val = loss
        else:
            self._val = self.alpha*self._val + (1-self.alpha)*loss
        super(mytqdm, self).set_postfix({'loss': self._val})
    
class dataset:
    classes = [0,1]
    normal_class = 0

    def __init__(self, header_files):
        df = pd.read_csv(TRAIN_DATA_CSV)

        self.files =[]
        self.gender=[]
        self.age=[]

    
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h

            expected_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6','III', 'aVR', 'aVL', 'aVF']
            x = ET.parse(TRAIN_DATA_FOLDER+h).iter('LeadData')
            ilead = 0
            ecg_waveforms = [""] * 8
            for c in x:
                # 500 hz * 10초 * 8 채널
                nsamp = int(c.find('LeadSampleCountTotal').text)
                if nsamp != 5000:
                    continue
                lead = c.find('LeadID').text
                if lead not in expected_leads:
                    break
                buf = base64.b64decode(c.find('WaveFormData').text)
                data = np.frombuffer(buf, '<i2', nsamp) # nsamp
                ecg_waveforms[expected_leads.index(lead)]=data
                ilead += 1
            try:
                vals = np.array(ecg_waveforms, dtype=np.float32)
            except:
                continue
            tmp['fs']=500
            tmp['record'] = np.append(vals, [vals[1]-vals[0], -(vals[1]+vals[0])*0.5, vals[0]-vals[1]*0.5, vals[1]-vals[0]*0.5], axis=0)   
            tmp['leads'] = expected_leads
            tmp['dx'] = df.loc[df['filename'] == h,'label'].values[0]
            tmp['target'] = np.zeros((2,))
            if tmp['dx'] in dataset.classes:
                idx = dataset.classes.index(tmp['dx'])
                tmp['target'][idx] = 1
            tmp['age'] = df.loc[df['filename'] == h,'age'].values[0] # !CONCAT!
            tmp['gender'] = df.loc[df['filename'] == h,'gender'].values[0] # !CONCAT!
            self.files.append(tmp)
        self.files = pd.DataFrame(self.files)        
        self.sample = True
        self.num_leads = 12 #None
        # set filter parameters
        self.b, self.a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    def summary(self, output):
        if output == 'pandas':
            return pd.Series(np.stack(self.files['target'].to_list(), axis=0).sum(axis=0), index=dataset.classes)
        if output == 'numpy':
            return np.stack(self.files['target'].to_list(), axis=0).sum(axis=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fs = self.files.iloc[item]['fs']
        target = self.files.iloc[item]['target']
        leads = self.files.iloc[item]['leads']
        data = self.files.iloc[item]['record']
        age = self.files.iloc[item]['age'] # !CONCAT!
        gender= self.files.iloc[item]['gender'] # !CONCAT!

        # expand to 12 lead setup if original signal has less channels
        data, lead_indicator = expand_leads(data, input_leads=leads)
        data = np.nan_to_num(data)

        # resample to 500hz
        if fs == float(1000):
            data = signal.resample_poly(
                data, up=1, down=2, axis=-1)  # to 500Hz
            fs = 500
        elif fs == float(500):
            pass
        else:
            data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
            fs = 500

        data = signal.filtfilt(self.b, self.a, data)

        if self.sample:
            fs = int(fs)
            # random sample signal if len > 8192 samples
            if data.shape[-1] >= 8192:
                idx = data.shape[-1] - 8192-1
                idx = np.random.randint(idx)
                data = data[:, idx:idx + 8192]

        mu = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True)
        # std = np.nanstd(data.flatten())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = (data - mu) / std
        data = np.nan_to_num(data)

        # random choose number of leads to keep
        data, lead_indicator = lead_exctractor.get(
            data, self.num_leads, lead_indicator)

        return data, target, lead_indicator, age, gender # !CONCAT!


class dataset_mimic:
    classes = [0,1]
    normal_class = 0

    def __init__(self, header_files):
        df = pd.read_csv('mimic_data_labels_2.csv')

        self.files =[]
        self.gender=[]
        self.age=[]

    
        for h in tqdm(header_files):
            tmp = dict()
            tmp['header'] = h

            expected_leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6','III', 'aVR', 'aVL', 'aVF']
            record = wfdb.rdrecord('C:/rsrch/240801_ecg_mace/mimic_ecg_data/' + str(h)).p_signal
            record = np.transpose(record)
            ecg_waveforms = [""] * 12
            for id in range(len(expected_leads)):
                lead = wfdb.rdrecord('C:/rsrch/240801_ecg_mace/mimic_ecg_data/'+h).sig_name[id]
                ecg_waveforms[expected_leads.index(lead)]=record[id]            

            tmp['fs']=500
            tmp['record'] = np.array(ecg_waveforms)    
            tmp['leads'] = expected_leads
            tmp['dx'] = df.loc[df['path'] == h,'label'].values[0]
            tmp['target'] = np.zeros((2,))
            if tmp['dx'] in dataset.classes:
                idx = dataset.classes.index(tmp['dx'])
                tmp['target'][idx] = 1
            tmp['age'] = df.loc[df['path'] == h,'age'].values[0] # !CONCAT!
            tmp['gender'] = df.loc[df['path'] == h,'gender'].values[0] # !CONCAT!
            self.files.append(tmp)
        self.files = pd.DataFrame(self.files)        
        self.sample = True
        self.num_leads = 12 #None
        # set filter parameters
        self.b, self.a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    def summary(self, output):
        if output == 'pandas':
            return pd.Series(np.stack(self.files['target'].to_list(), axis=0).sum(axis=0), index=dataset.classes)
        if output == 'numpy':
            return np.stack(self.files['target'].to_list(), axis=0).sum(axis=0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fs = self.files.iloc[item]['fs']
        target = self.files.iloc[item]['target']
        leads = self.files.iloc[item]['leads']
        data = self.files.iloc[item]['record']
        age = self.files.iloc[item]['age'] # !CONCAT!
        gender= self.files.iloc[item]['gender'] # !CONCAT!

        # expand to 12 lead setup if original signal has less channels
        data, lead_indicator = expand_leads(data, input_leads=leads)
        data = np.nan_to_num(data)

        # resample to 500hz
        if fs == float(1000):
            data = signal.resample_poly(
                data, up=1, down=2, axis=-1)  # to 500Hz
            fs = 500
        elif fs == float(500):
            pass
        else:
            data = signal.resample(data, int(data.shape[1] * 500 / fs), axis=1)
            fs = 500

        data = signal.filtfilt(self.b, self.a, data)

        if self.sample:
            fs = int(fs)
            # random sample signal if len > 8192 samples
            if data.shape[-1] >= 8192:
                idx = data.shape[-1] - 8192-1
                idx = np.random.randint(idx)
                data = data[:, idx:idx + 8192]

        mu = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True)
        # std = np.nanstd(data.flatten())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = (data - mu) / std
        data = np.nan_to_num(data)

        # random choose number of leads to keep
        data, lead_indicator = lead_exctractor.get(
            data, self.num_leads, lead_indicator)

        return data, target, lead_indicator, age, gender # !CONCAT!

class lead_exctractor:
    """
    used to select specific leads or random choice of configurations

    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: I, II, III, aVR, aVL, aVF
    Four leads: I, II, III, V2
    Three leads: I, II, V2
    Two leads: I, II

    """
    L2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    L3 = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L4 = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    L6 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    L8 = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    L12 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    @staticmethod
    def get(x, num_leads, lead_indicator):
        if num_leads == None:
            # random choice output
            num_leads = random.choice([12, 8, 6, 4, 3, 2])

        if num_leads == 12:
            # Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            return x, lead_indicator * lead_exctractor.L12

        if num_leads == 8:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L8.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L8

        if num_leads == 6:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L6.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L6

        if num_leads == 4:
            # Six leads: I, II, III, V2
            x = x * lead_exctractor.L4.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L4

        if num_leads == 3:
            # Three leads: I, II, V2
            x = x * lead_exctractor.L3.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L3

        if num_leads == 2:
            # Two leads: II, V5
            x = x * lead_exctractor.L2.reshape(12, 1)
            return x, lead_indicator * lead_exctractor.L2
        raise Exception("invalid-leads-number")


def collate(batch):
    ch = batch[0][0].shape[0]
    # maxL = max([b[0].shape[-1] for b in batch])
    maxL = 8192
    X = np.zeros((len(batch), ch, maxL))
    for i in range(len(batch)):
        X[i, :, -batch[i][0].shape[-1]:] = batch[i][0]
    t = np.array([b[1] for b in batch])
    l = np.concatenate([b[2].reshape(1, 12) for b in batch], axis=0)
    a = np.array([b[3] for b in batch]) # !CONCAT!
    g = np.array([b[4] for b in batch]) # !CONCAT!

    X = torch.from_numpy(X)
    t = torch.from_numpy(t)
    l = torch.from_numpy(l)
    a = torch.from_numpy(a) # !CONCAT!
    g = torch.from_numpy(g) # !CONCAT!
    return X, t, l,a,g


def valid_part(model, dataset):
    targets = []
    outputs = []
    #weights_file = 'weights.csv'
    #sinus_rhythm = set(['426783006'])
    #classes, weights = load_weights(weights_file)
    model.eval()

    with torch.no_grad():
        for i, (x, t, l,a,g) in enumerate(tqdm(dataset)):   # !CONCAT!
            x = x.unsqueeze(2).float().to(DEVICE)
            
            t = t.to(DEVICE)
            l = l.float().to(DEVICE)
            a = a.unsqueeze(1).float().to(DEVICE) # !CONCAT!
            g = g.unsqueeze(1).float().to(DEVICE) # !CONCAT!
            
            y = model(x, l,a,g)[0]  # !CONCAT!
            # p = torch.sigmoid(y)

            targets.append(t.data.cpu().numpy())
            outputs.append(y.data.cpu().numpy())
    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    auroc = roc_auc_score(targets, outputs) #average_precision_score(y_true=targets, y_score=outputs)
    #challenge_metric = compute_challenge_metric(
    #    weights, targets, outputs, classes, sinus_rhythm)

    return auroc, targets, outputs


def train_part(model, dataset, loss, opt):
    targets = []
    outputs = []
    model.train()

    with mytqdm(dataset) as pbar:
        for i, (x, t, l,a,g) in enumerate(pbar):  # !CONCAT!
            opt.zero_grad()
            x = x.unsqueeze(2).float().to(DEVICE)
            t = t.to(DEVICE)
            l = l.float().to(DEVICE)
            a = a.unsqueeze(1).float().to(DEVICE) # !CONCAT!
            g = g.unsqueeze(1).float().to(DEVICE) # !CONCAT!

            
            y = model(x, l,a,g)[0]  # !CONCAT!
            # p = torch.sigmoid(y)
      #      M = chloss(t, p)
      #      N = loss(input=p, target=t)
      #      Q = torch.mean(-4*p*(p-1))
      #      J = N - M + Q
            J = -torch.mean(t * F.logsigmoid(y) + (1 - t)
                            * F.logsigmoid(-y) * 0.1)
            J.backward()
            pbar.set_postfix(np.array([J.data.cpu().numpy(),
                                       ]))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            opt.step()
            targets.append(t.data.cpu().numpy())
            outputs.append(y.data.cpu().numpy())
        targets = np.concatenate(targets, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        auroc = roc_auc_score(targets, outputs) #average_precision_score(y_true=targets, y_score=outputs)

    return auroc

def training_code(data_directory, model_directory):
    select4deployment.calls = 0
    _training_code(data_directory, model_directory, str(0))
    #_training_code(data_directory, model_directory, str(1))
    #_training_code(data_directory, model_directory, str(2))

def _training_code(data_directory, model_directory, ensamble_ID):
    # Find header and recording files.
    print('Finding header and recording files...')

    df = pd.read_csv(TRAIN_DATA_CSV)
    unique_patient_ids = df['hid'].unique()
    random.shuffle(unique_patient_ids)
    num_train_patients = int(len(unique_patient_ids) * (1-0.2))
    # Split the unique 'PatientID's into train and test sets
    train_patients = unique_patient_ids[:num_train_patients]
    test_patients = unique_patient_ids[num_train_patients:]

    train = dataset(header_files=df.loc[df['hid'].isin(train_patients),'filename'].to_list())
    train.num_leads = 12
    train.sample = True
    valid = dataset(header_files=df.loc[df['hid'].isin(test_patients),'filename'].to_list())
    valid.num_leads = 12
    valid.sample = False

    valid.files.reset_index(drop=True, inplace=True)
    # negative to positive ratio
    loss_weight = (len(train) - train.summary(output='numpy')) / \
        train.summary(output='numpy')

    # to be saved in resulting model pickle
    train_files = train.files['header'].to_list()
    train_files = [k.split('/')[-1] for k in train_files]
    valid_files = valid.files['header'].to_list()
    valid_files = [k.split('/')[-1] for k in valid_files]

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    train = DataLoader(dataset=train,
                       batch_size=bsize,
                       shuffle=True,
                       num_workers=8,
                       collate_fn=collate,
                       pin_memory=True,
                       drop_last=False)

    valid = DataLoader(dataset=valid,
                       batch_size=bsize,
                       shuffle=False,
                       num_workers=8,
                       collate_fn=collate,
                       pin_memory=True,
                       drop_last=False)

    model = FinalModel(num_classes=2).to(DEVICE) #26
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print('Model Loaded!')

    lossBCE = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(loss_weight).to(DEVICE))
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.1)
    OUTPUT = []
    EPOCHS = 5 # orig: 100
    for epoch in range(EPOCHS):
        print(
            f"============================[{epoch}]============================")
        train_auprc = train_part(model, train, lossBCE, opt)
        print(train_auprc)

        valid_auprc, valid_targets, valid_outputs = valid_part(model, valid) #challenge_metric
        print(valid_auprc)

        OUTPUT.append({'epoch': epoch,
                       'model': copy.deepcopy(model).cpu().state_dict(),
                       'train_auprc': train_auprc,
                       'valid_auprc': valid_auprc,
                       'valid_targets': valid_targets,
                       'valid_outputs': valid_outputs, #'val_challenge_metric': challenge_metric
                       }) 
        scheduler.step()
        name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
        with open(name, 'wb') as handle:
            pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(dataset.classes, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(loss_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

    name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
    with open(name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset.classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(loss_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

    find_thresholds(name, model_directory)


# Generic function for loading a model.
def _load_model(model_directory, id):
    filename = Path(model_directory, f'MODEL_{id}.pickle')
    model = {}
    with open(filename, 'rb') as handle:
        input = pickle.load(handle)

    model['classifier'] = FinalModel(num_classes=2).to(DEVICE)
    model['classifier'].load_state_dict(input['state_dict'])
    model['classifier'].eval()
    model['thresholds'] = input['thresholds']
    model['classes'] = input['classes']
    return model


def load_model(model_directory, leads):

    model = {}
    model['1'] = _load_model(model_directory, 1)
   # model['2'] = _load_model(model_directory, 2)
   # model['3'] = _load_model(model_directory, 3)
    return model


################################################################################
#
# Running trained model functions
#
################################################################################


def expand_leads(recording, input_leads):
    output = np.zeros((12, recording.shape[1]))
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    twelve_leads = [k.lower() for k in twelve_leads]

    input_leads = [k.lower() for k in input_leads]
    output_leads = np.zeros((12,))
    for i, k in enumerate(input_leads):
        idx = twelve_leads.index(k)
        output[idx, :] = recording[i, :]
        output_leads[idx] = 1
    return output, output_leads


def zeropad(x):
    y = np.zeros((12, 8192))
    if x.shape[1] < 8192:
        y[:, -x.shape[1]:] = x
    else:
        y = x[:, :8192]
    return y


def preprocessing(recording, leads, fs):
    b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    if fs == 1000:
        recording = signal.resample_poly(
            recording, up=1, down=2, axis=-1)  # to 500Hz
        fs = 500
    elif fs == 500:
        pass
    else:
        recording = signal.resample(recording, int(
            recording.shape[1] * 500 / fs), axis=1)
        print(f'RESAMPLING FROM {fs} TO 500')
        fs = 500

    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)
    #recording = zeropad(recording)
    #recording = torch.from_numpy(recording).view(
    #    1, 12, 1, -1).float().to(DEVICE)
    leads = torch.from_numpy(leads).float().view(1, 12).to(DEVICE)
    return recording, leads


# Generic function for running a trained model.
def run_model(model, header, recording,age,gender):
    # load lead names form file
    input_leads = np.ones(12)
    #recording, leads = expand_leads(recording, input_leads)
    recording, leads = preprocessing(recording, input_leads, fs=500)
    age = torch.tensor(age).view(1,1).float().to(DEVICE) # !CONCAT!
    gender = torch.tensor(gender).view(1,1).float().to(DEVICE) # !CONCAT!

    classes = model['1']['classes']

    out_labels = np.zeros((3, 2))
    for i, (key, mod) in enumerate(model.items()):
        thresholds = mod['thresholds']
        classifier = mod['classifier']
        
        q = classifier(recording, leads,age,gender)[0]  # !CONCAT!
        features = classifier(recording, leads,age,gender)[1]   # !CONCAT!

        # p = torch.sigmoid(_probabilities)
        q = q.data[0, :].cpu().numpy()

        # Predict labels and probabilities.
        labels = q >= thresholds
        out_labels[i, :] = labels
    labels = np.sum(out_labels, axis=0)
    labels = np.array(labels, dtype=int)
    return classes, labels, q, features

################################################################################
#
# Other functions
#
################################################################################
