import os, shutil, random, time, base64, pickle
from collections import Counter
from glob import glob
from tqdm import tqdm
from PIL import Image
from lxml import etree as ET
#import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score, roc_auc_score,RocCurveDisplay, roc_curve, auc
from scipy import signal
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.calibration import calibration_curve, CalibrationDisplay

import seaborn

def draw_roc_curve(y_test, y_test_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                            estimator_name='Model')
    display.plot()
    plt.show()    

def draw_confusion_matrix(y_test, y_pred_test):
    fontsize = 20
    contingency_table = pd.crosstab(y_test, y_pred_test)
    contingency_table
    contingency_table.columns = ['Normal', 'MACCE']
    contingency_table.columns.name = 'Label'
    contingency_table.index = ['Normal', 'MACCE']
    contingency_table.index.name = 'Prediction'
    contingency_percentage = contingency_table / len(y_test) * 100
    plt.figure(figsize=(8, 6))
    ax = seaborn.heatmap(contingency_table, annot=False, fmt="d", cmap='Blues', annot_kws={"size": 16})
    for i in range(contingency_table.shape[0]):
        for j in range(contingency_table.shape[1]):
            if i == 0 and j == 0:
                color = 'white'
            else:
                color = 'black'
            
            count = contingency_table.iloc[i, j]
            percent = contingency_percentage.iloc[i, j]
            text = f"{count:,}\n({percent:.1f}%)"
            ax.text(j+0.5, i+0.5, text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=fontsize, color=color)
    ax.set_xlabel('Prediction', fontsize=fontsize)
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    plt.ylabel('Label', fontsize=fontsize)
    plt.show()

def draw_y_test_proba(y_test_proba):
    plt.hist(y_test_proba, bins=50, edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Test probability histogram')
    plt.show()

def draw_calibration_plot(prob_true, prob_pred , y_test_proba, n_bins, ax_end):
    hist, bin_edges = np.histogram(prob_true, bins=n_bins, range=(0, 1))
    bin_std = []
    for i in range(n_bins):
        bin_data = y_test_proba[(y_test_proba >= bin_edges[i]) & (y_test_proba < bin_edges[i+1])]
        bin_std.append(np.std(bin_data))
    
    bin_std = [value for value in bin_std if not np.isnan(value)]
    fig, ax = plt.subplots()
    ax.errorbar(prob_pred, prob_true , fmt='o', color='black',yerr=bin_std, markersize=3) #yerr=bin_std
    ax.plot([0, ax_end], [0, ax_end], "k:", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title("Calibration Curve")
    ax.legend()
    plt.show()