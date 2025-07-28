import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, auc, RocCurveDisplay, confusion_matrix
import seaborn
warnings.filterwarnings("ignore")
from scipy.stats import norm

def brier_score_loss(y_true, y_pred):
    """
    Calculate Brier score loss between true labels and predicted probabilities
    Lower values indicate better calibrated predictions (0 is perfect)
    """
    return np.mean((y_true - y_pred) ** 2)

def integrated_calibration_index(y_true, y_pred, n_bins=100):
    """
    Calculate integrated calibration index (ICI)
    Lower values indicate better calibration (0 is perfect)
    """
    # Sort predictions and corresponding true values
    sort_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sort_idx]
    y_true_sorted = y_true[sort_idx]
    
    # Calculate calibration curve using moving average
    window = len(y_true) // n_bins
    if window < 1:
        window = 1
    
    calibration_curve = np.array([
        np.mean(y_true_sorted[max(0, i-window):min(len(y_true), i+window)])
        for i in range(len(y_true))
    ])
    
    # Calculate absolute difference between predictions and calibration curve
    ici = np.mean(np.abs(y_pred_sorted - calibration_curve))
    return ici

def normal_ci(y_true, y_pred, y_pred_binary, alpha=0.05):
    """Calculate normal-distribution-based confidence intervals for various metrics."""
    n = len(y_true)
    z = norm.ppf(1 - alpha / 2)

    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Calculate metrics
    metrics = {
        'AUROC': roc_auc_score(y_true, y_pred),
        'AUPRC': average_precision_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred_binary),
        'Accuracy': accuracy_score(y_true, y_pred_binary),
        'Precision': precision_score(y_true, y_pred_binary),
        'Recall': recall_score(y_true, y_pred_binary),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # True Negative Rate
        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # True Positive Rate (same as Recall)
        'Brier': brier_score_loss(y_true, y_pred),
        'ICI': integrated_calibration_index(y_true, y_pred)
    }

    ci = {}
    # Accuracy, Precision, Recall, F1, Specificity, Sensitivity: binomial proportion SE
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'Specificity', 'Sensitivity']:
        p = metrics[metric]
        se = np.sqrt(p * (1 - p) / n)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        ci[metric] = (lower, upper)

    # AUROC and AUPRC: use normal approximation
    for metric in ['AUROC', 'AUPRC']:
        p = metrics[metric]
        se = np.sqrt(p * (1 - p) / n)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)
        ci[metric] = (lower, upper)

    # Brier score: standard error of the mean
    brier = metrics['Brier']
    se_brier = np.std((y_true - y_pred) ** 2, ddof=1) / np.sqrt(n)
    ci['Brier'] = (max(0, brier - z * se_brier), min(1, brier + z * se_brier))

    # ICI: standard error of the mean
    ici = metrics['ICI']
    se_ici = np.std(np.abs(y_pred - y_true), ddof=1) / np.sqrt(n)
    ci['ICI'] = (max(0, ici - z * se_ici), min(1, ici + z * se_ici))

    return ci, (tn, fp, fn, tp)

def draw_model_evaluation_plots(y_test, y_test_proba, y_pred_test, n_bins=10, draw =True):
    AUROC = roc_auc_score(y_test, y_test_proba)
    AUPRC = average_precision_score(y_test, y_test_proba)
    F1_score = f1_score(y_test, y_pred_test)
    brier_score = brier_score_loss(y_test, y_test_proba)
    ici = integrated_calibration_index(y_test, y_test_proba)
    ci_metrics, (tn, fp, fn, tp) = normal_ci(y_test, y_test_proba, y_pred_test)
    
    # Calculate Precision, Specificity, Sensitivity from confusion matrix
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f'AUROC {AUROC:.3f} (95% CI: {ci_metrics["AUROC"][0]:.3f}-{ci_metrics["AUROC"][1]:.3f})')
    print(f'AUPRC {AUPRC:.3f} (95% CI: {ci_metrics["AUPRC"][0]:.3f}-{ci_metrics["AUPRC"][1]:.3f})')
    print(f'F1 Score {F1_score:.3f} (95% CI: {ci_metrics["F1"][0]:.3f}-{ci_metrics["F1"][1]:.3f})')
    print(f'Test Accuracy {(accuracy_score(y_test, y_pred_test)):.3f} (95% CI: {ci_metrics["Accuracy"][0]:.3f}-{ci_metrics["Accuracy"][1]:.3f})')
    print(f'Precision {precision:.3f} (95% CI: {ci_metrics["Precision"][0]:.3f}-{ci_metrics["Precision"][1]:.3f})')
    print(f'Specificity {specificity:.3f} (95% CI: {ci_metrics["Specificity"][0]:.3f}-{ci_metrics["Specificity"][1]:.3f})')
    print(f'Sensitivity {sensitivity:.3f} (95% CI: {ci_metrics["Sensitivity"][0]:.3f}-{ci_metrics["Sensitivity"][1]:.3f})')
    print(f'Brier Score {brier_score:.3f} (95% CI: {ci_metrics["Brier"][0]:.3f}-{ci_metrics["Brier"][1]:.3f})')
    print(f'ICI {ici:.3f} (95% CI: {ci_metrics["ICI"][0]:.3f}-{ci_metrics["ICI"][1]:.3f})')
    print(f'Confusion Matrix:')
    print(f'TN: {tn}, FP: {fp}')
    print(f'FN: {fn}, TP: {tp}')
    
    if not draw:
        return

    """
    Draw four evaluation plots in a 1x4 layout:
    1. Calibration plot with error bars
    2. ROC curve (with 95% CI in shade)
    3. Confusion matrix
    4. Probability histogram
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), tight_layout=True)
    
    # Ensure all plots have the same height and size
    for ax in axes:
        ax.set_box_aspect(1)
    
    # 1. Calibration plot with error bars
    plt.sca(axes[0])
    
    # Create bins and find probabilities in each bin
    LIM = 0.1
    n_bins = 10
    bins = np.linspace(0, LIM, n_bins + 1)
    binids = np.digitize(y_test_proba, bins) - 1
    
    bin_true = np.zeros(n_bins)
    bin_pred = np.zeros(n_bins)
    bin_errors = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)
    
    # Calculate statistics for each bin
    for bin_idx in range(n_bins):
        bin_mask = binids == bin_idx
        if np.sum(bin_mask) > 0:  # Check if bin has any samples
            bin_sizes[bin_idx] = np.sum(bin_mask)
            bin_true[bin_idx] = np.mean(y_test[bin_mask])
            bin_pred[bin_idx] = np.mean(y_test_proba[bin_mask])
            # Calculate standard error for the bin
            bin_errors[bin_idx] = np.std(y_test[bin_mask]) / np.sqrt(bin_sizes[bin_idx])
    
    # Plot calibration
    plt.plot([0, LIM], [0, LIM], 'k:', label='Perfectly calibrated')
    plt.errorbar(bin_pred, bin_true, yerr=bin_errors, 
                fmt='o', color='red', ecolor='gray', 
                capsize=3, capthick=1, markersize=4,
                label='Model calibration')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, LIM)
    plt.ylim(0, LIM)
    
    # 2. ROC curve with 95% CI in shade
    plt.sca(axes[1])
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}, 95% CI: {ci_metrics["AUROC"][0]:.3f}-{ci_metrics["AUROC"][1]:.3f})')
    plt.plot([0, 1], [0, 1], 'k:', lw=2)

    # --- 95% CI for ROC curve using bootstrap ---
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    bootstrapped_tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            continue
        fpr_boot, tpr_boot, _ = roc_curve(y_test[indices], y_test_proba[indices])
        # Interpolate tpr
        tpr_interp = np.interp(mean_fpr, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        bootstrapped_tprs.append(tpr_interp)
    bootstrapped_tprs = np.array(bootstrapped_tprs)
    tpr_mean = np.mean(bootstrapped_tprs, axis=0)
    tpr_std = np.std(bootstrapped_tprs, axis=0)
    tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
    # Plot the 95% CI as a shaded area
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='b', alpha=0.2)
    # Plot the mean ROC curve (over bootstrap)
    #plt.plot(mean_fpr, tpr_mean, color='b', lw=2, alpha=0.7)
    #plt.plot(fpr, tpr, lw=2, color='r', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # 3. Confusion matrix
    plt.sca(axes[2])
    
    fontsize = 12
    contingency_table = pd.crosstab(y_test, y_pred_test)
    contingency_table.columns = ['Normal', 'MACCE']
    contingency_table.columns.name = 'Label'
    contingency_table.index = ['Normal', 'MACCE']
    contingency_table.index.name = 'Prediction'
    contingency_percentage = contingency_table / len(y_test) * 100
    
    ax = seaborn.heatmap(contingency_table, annot=False, fmt="d", cmap='Blues', annot_kws={"size": 12}, ax=axes[2])
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
    
    # 4. Probability histogram
    plt.sca(axes[3])
    
    plt.hist(y_test_proba, bins=50, edgecolor='black')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    
    plt.show()