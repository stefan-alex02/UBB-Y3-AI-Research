import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def evaluate_model_metrics(model, test_loader, class_names, device, logger=None):
    """Evaluates a model on the test dataset, computing various classification metrics and plots."""
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    k = len(class_names)  # Number of classes

    # Compute class-wise metrics
    precision = np.zeros(k)
    recall = np.zeros(k)
    specificity = np.zeros(k)
    f1 = np.zeros(k)
    auc_values = np.zeros(k)
    auprc_values = np.zeros(k)

    for i in range(k):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - (tp + fp + fn)

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        auc_values[i] = (recall[i] + specificity[i]) / 2
        auprc_values[i] = (precision[i] + recall[i]) / 2

    # Compute overall metrics
    accuracy = np.trace(cm) / np.sum(cm)
    overall_precision = np.mean(precision)
    overall_recall = np.mean(recall)
    overall_specificity = np.mean(specificity)
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (
                                                                                                                overall_precision + overall_recall) > 0 else 0
    overall_auc = (overall_recall + overall_specificity) / 2
    overall_auprc = (overall_precision + overall_recall) / 2

    # Print metrics
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClass-wise Metrics:")
    print("Class | Precision | Recall | Specificity | F1 | AUC | AUPRC")
    for i in range(k):
        print(
            f"{class_names[i]:5} | {precision[i]:.4f} | {recall[i]:.4f} | {specificity[i]:.4f} | {f1[i]:.4f} | {auc_values[i]:.4f} | {auprc_values[i]:.4f}")

    print("\nOverall Metrics:")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, Specificity: {overall_specificity:.4f}, F1: {overall_f1:.4f}, AUC: {overall_auc:.4f}, AUPRC: {overall_auprc:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    for i in range(k):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc(fpr, tpr):.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

    # Plot Precision-Recall curves for each class
    plt.figure(figsize=(8, 6))
    for i in range(k):
        precision_vals, recall_vals, _ = precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(recall_vals, precision_vals,
                 label=f"{class_names[i]} (AUPRC = {auc(recall_vals, precision_vals):.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "specificity": overall_specificity,
        "f1": overall_f1,
        "auc": overall_auc,
        "auprc": overall_auprc,
        "class_metrics": {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "auc": auc_values,
            "auprc": auprc_values
        }
    }
