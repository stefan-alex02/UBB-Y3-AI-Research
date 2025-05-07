import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Any, Optional
import warnings

from utils import logger


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        class_names: List[str],
        average: str = 'macro'  # 'macro' for unweighted mean, 'weighted' for weighted by support
) -> Dict[str, Any]:
    """
    Calculates various classification metrics, including per-class results.

    Args:
        y_true (np.ndarray): True labels (integer indices).
        y_pred (np.ndarray): Predicted labels (integer indices).
        y_prob (np.ndarray): Predicted probabilities (shape: n_samples, n_classes).
        class_names (List[str]): List of class names corresponding to indices.
        average (str): Averaging strategy for multiclass metrics ('macro' or 'weighted').

    Returns:
        Dict[str, Any]: A dictionary containing overall and per-class metrics.
                        Includes accuracy, precision, recall (sensitivity), F1-score,
                        specificity, AUC, AUPRC.
    """
    num_classes = len(class_names)
    results = {
        'overall': {},
        'per_class': {name: {} for name in class_names}
    }

    # --- Overall Metrics ---
    results['overall']['accuracy'] = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, F1 for overall (macro/weighted average)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    results['overall'][f'precision_{average}'] = precision
    results['overall'][f'recall_{average}'] = recall  # Recall is Sensitivity
    results['overall'][f'f1_score_{average}'] = f1

    # Binarize labels for AUC/AUPRC calculation
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Overall AUC / AUPRC (needs careful handling for multiclass)
    # Use one-vs-rest ('ovr') approach with specified averaging
    try:
        # Ensure y_prob has shape (n_samples, n_classes) even for binary case
        if num_classes == 2 and y_prob.ndim == 1:
            # If binary and probabilities are for the positive class only
            y_prob_auc = np.vstack([1 - y_prob, y_prob]).T
        else:
            y_prob_auc = y_prob

        if y_true_bin.shape[
            1] == 1 and num_classes > 1:  # Special case after label_binarize if only one class present in y_true
            logger.warning(f"⚠️ Only one class present in y_true. AUC/AUPRC calculation might be trivial.")
            # Cannot compute multiclass AUC/AUPRC reliably here. Set to NaN or skip.
            results['overall'][f'auc_{average}'] = float('nan')
            results['overall'][f'auprc_{average}'] = float('nan')
        elif num_classes == 2:  # Binary case AUC/AUPRC
            results['overall'][f'auc'] = roc_auc_score(y_true, y_prob_auc[:, 1])
            results['overall'][f'auprc'] = average_precision_score(y_true, y_prob_auc[:, 1])
        elif num_classes > 2:  # Multiclass case
            results['overall'][f'auc_ovr_{average}'] = roc_auc_score(
                y_true_bin, y_prob_auc, average=average, multi_class='ovr'
            )
            # AUPRC doesn't directly support multiclass in sklearn like AUC
            # We compute per-class AUPRC below and can average manually if needed
            # Calculate macro AUPRC manually
            per_class_auprc = []
            valid_classes_for_auprc = 0
            for i in range(num_classes):
                # Check if class has positive samples
                if np.sum(y_true_bin[:, i]) > 0:
                    auprc_cls = average_precision_score(y_true_bin[:, i], y_prob_auc[:, i])
                    per_class_auprc.append(auprc_cls)
                    valid_classes_for_auprc += 1
                else:
                    per_class_auprc.append(float('nan'))  # Or 0, or skip? NaN seems appropriate
            if valid_classes_for_auprc > 0:
                results['overall'][f'auprc_ovr_{average}'] = np.nanmean(per_class_auprc)  # Macro average
            else:
                results['overall'][f'auprc_ovr_{average}'] = float('nan')

    except ValueError as e:
        logger.warning(f"⚠️ Could not calculate AUC/AUPRC. Reason: {e}")
        results['overall'][f'auc_{average}'] = float('nan')
        results['overall'][f'auprc_{average}'] = float('nan')
        if num_classes > 2:
            results['overall'][f'auc_ovr_{average}'] = float('nan')
            results['overall'][f'auprc_ovr_{average}'] = float('nan')

    # --- Per-Class Metrics ---
    # Calculate precision, recall, F1 per class
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    # Calculate specificity per class using confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    # Specificity = TN / (TN + FP)
    tn = cm.sum(axis=1) - np.diag(
        cm)  # Sum across row (true class) gives TP+FN, subtract TP (diag) -> FN? No, this logic is wrong.
    # Let's recalculate CM components per class (One-vs-Rest)
    specificity_pc = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)  # Total - (TP + FN + FP)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_pc.append(specificity)

    # AUC and AUPRC per class
    auc_pc = []
    auprc_pc = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings for classes with no positive samples
        for i in range(num_classes):
            # Check if class has both positive and negative samples in y_true_bin
            n_pos = np.sum(y_true_bin[:, i])
            n_neg = len(y_true_bin[:, i]) - n_pos

            if n_pos > 0 and n_neg > 0:  # Need both for AUC/AUPRC
                try:
                    auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                    auprc_val = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    auc_pc.append(auc_val)
                    auprc_pc.append(auprc_val)
                except Exception as e:
                    logger.warning(f"⚠️ Could not calculate AUC/AUPRC for class '{class_names[i]}'. Reason: {e}")
                    auc_pc.append(float('nan'))
                    auprc_pc.append(float('nan'))
            else:
                # logger.debug(f"Skipping AUC/AUPRC for class '{class_names[i]}' due to lack of positive or negative samples.")
                auc_pc.append(float('nan'))  # Not meaningful if only one class present
                auprc_pc.append(float('nan'))

    # Store per-class metrics
    for i, name in enumerate(class_names):
        results['per_class'][name]['precision'] = precision_pc[i]
        results['per_class'][name]['recall'] = recall_pc[i]  # Sensitivity
        results['per_class'][name]['f1_score'] = f1_pc[i]
        results['per_class'][name]['specificity'] = specificity_pc[i]
        results['per_class'][name]['support'] = int(support_pc[i])  # Number of true instances
        results['per_class'][name]['auc'] = auc_pc[i] if i < len(auc_pc) else float('nan')
        results['per_class'][name]['auprc'] = auprc_pc[i] if i < len(auprc_pc) else float('nan')

    # Add confusion matrix to results (optional)
    results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization

    return results


def format_metrics_log(metrics: Dict[str, Any], class_names: List[str]) -> str:
    """Formats the metrics dictionary into a readable string for logging."""
    log_str = "\n--- Evaluation Metrics ---\n"
    log_str += f"Overall Accuracy: {metrics['overall'].get('accuracy', 'N/A'):.4f}\n"
    for avg_type in ['macro', 'weighted']:
        if f'precision_{avg_type}' in metrics['overall']:
            log_str += f"Overall Precision ({avg_type}): {metrics['overall'][f'precision_{avg_type}']:.4f}\n"
            log_str += f"Overall Recall ({avg_type}):    {metrics['overall'][f'recall_{avg_type}']:.4f}\n"
            log_str += f"Overall F1-Score ({avg_type}):  {metrics['overall'][f'f1_score_{avg_type}']:.4f}\n"
        if f'auc_ovr_{avg_type}' in metrics['overall']:
            log_str += f"Overall AUC (OvR {avg_type}):   {metrics['overall'][f'auc_ovr_{avg_type}']:.4f}\n"
        if f'auprc_ovr_{avg_type}' in metrics['overall']:
            log_str += f"Overall AUPRC (OvR {avg_type}): {metrics['overall'][f'auprc_ovr_{avg_type}']:.4f}\n"
    if 'auc' in metrics['overall']:  # Binary case
        log_str += f"Overall AUC:              {metrics['overall']['auc']:.4f}\n"
        log_str += f"Overall AUPRC:            {metrics['overall']['auprc']:.4f}\n"

    log_str += "\nPer-Class Metrics:\n"
    header = f"{'Class':<15} | {'Acc.':<7} | {'Prec.':<7} | {'Recall':<7} | {'Spec.':<7} | {'F1':<7} | {'AUC':<7} | {'AUPRC':<7} | Support\n"
    log_str += header
    log_str += "-" * len(header) + "\n"

    num_classes = len(class_names)
    # Calculate per-class accuracy from CM
    cm = np.array(metrics.get('confusion_matrix', np.zeros((num_classes, num_classes))))
    per_class_acc = []
    if cm.sum() > 0:  # Avoid division by zero if CM is all zeros
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            acc = (tp + tn) / cm.sum() if cm.sum() > 0 else 0.0
            per_class_acc.append(acc)
    else:
        per_class_acc = [0.0] * num_classes

    for i, name in enumerate(class_names):
        pc_metrics = metrics['per_class'][name]
        acc_pc = per_class_acc[i] if i < len(per_class_acc) else float('nan')
        log_str += (
            f"{name:<15} | "
            f"{acc_pc:<7.4f} | "
            f"{pc_metrics.get('precision', float('nan')):<7.4f} | "
            f"{pc_metrics.get('recall', float('nan')):<7.4f} | "
            f"{pc_metrics.get('specificity', float('nan')):<7.4f} | "
            f"{pc_metrics.get('f1_score', float('nan')):<7.4f} | "
            f"{pc_metrics.get('auc', float('nan')):<7.4f} | "
            f"{pc_metrics.get('auprc', float('nan')):<7.4f} | "
            f"{pc_metrics.get('support', 'N/A')}\n"
        )
    if 'confusion_matrix' in metrics:
        log_str += "\nConfusion Matrix:\n"
        log_str += "Predicted ->\n"
        label_text = "True \\ Pred"  # Define outside the f-string
        header = f"{label_text:<12}" + "".join([f"{c:<10}" for c in class_names]) + "\n"
        log_str += header
        log_str += "-" * len(header) + "\n"
        for i, row in enumerate(metrics['confusion_matrix']):
            row_str = f"{class_names[i]:<12}" + "".join([f"{x:<10}" for x in row]) + "\n"
            log_str += row_str

    log_str += "-------------------------\n"
    return log_str
