"""Binary concept metrics: precision / recall / F1 / ROC-AUC of the predicted
concept probabilities against the binary GT concept vectors, globally and per
class, plus a binarization-threshold sweep."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from cbm_eval.coco import COCO_CLASSES



def evaluate_binary_concepts(pred_probs, gt_vectors, class_ids, num_classes=80, threshold=0.5):
    """
    Evaluates concept predictions against a binary ground truth.
    Calculates metrics globally and per-class.
    """
    # 1. Binarize predictions based on threshold
    binary_preds = (pred_probs >= threshold).astype(int)

    # Global metrics: all concepts of all instances pooled together.
    flat_preds = pred_probs.flatten()
    flat_binary_preds = binary_preds.flatten()
    flat_gts = gt_vectors.flatten()

    global_precision = precision_score(flat_gts, flat_binary_preds, zero_division=0)
    global_recall = recall_score(flat_gts, flat_binary_preds, zero_division=0)
    global_f1 = f1_score(flat_gts, flat_binary_preds, zero_division=0)

    try:
        global_auc = roc_auc_score(flat_gts, flat_preds)
    except ValueError:
        global_auc = float('nan') # Fails if only one class is present in GT

    print(f"\nBinarization threshold: {threshold}")
    print("\nGlobal Metrics")
    t = PrettyTable([" ", "Precision", "Recall", "F1", "AUC"])
    t.add_row(["GLOBAL", round(global_precision, 4), round(global_recall, 4), round(global_f1, 4), round(global_auc, 4)])
    print(t)

    # Per-class metrics (pooled over the instances of each class).
    print("\nPer-Class Metrics")
    class_metrics = {}

    t = PrettyTable(["Class", "Precision", "Recall", "F1", "AUC"])

    for c_id in range(num_classes):
        class_mask = class_ids == c_id
        if not np.any(class_mask):
            continue

        class_preds = pred_probs[class_mask].flatten()
        class_binary_preds = binary_preds[class_mask].flatten()
        class_gts = gt_vectors[class_mask].flatten()

        c_precision = precision_score(class_gts, class_binary_preds, zero_division=0)
        c_recall = recall_score(class_gts, class_binary_preds, zero_division=0)
        c_f1 = f1_score(class_gts, class_binary_preds, zero_division=0)

        try:
            c_auc = roc_auc_score(class_gts, class_preds)
        except ValueError:
            c_auc = float('nan')

        class_metrics[c_id] = {'precision': c_precision, 'recall': c_recall, 'f1': c_f1, 'auc': c_auc}

        if len(class_metrics) <= 10:  # print only the first 10 classes
            t.add_row([COCO_CLASSES[c_id], round(c_precision, 4), round(c_recall, 4), round(c_f1, 4), round(c_auc, 4)])

    print(t)
    print("...\n")

    return {'global': {'precision': global_precision, 'recall': global_recall, 'f1': global_f1, 'auc': global_auc}, 'per_class': class_metrics}

def plot_binary_metrics(metrics_dict, save_dir):
    """Generates and saves plots for the binary metrics."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ==========================================
    # 1. GLOBAL METRICS PLOTS (Saved to root model dir)
    # ==========================================
    global_m = metrics_dict['global']
    labels = ['Precision', 'Recall', 'F1 Score', 'AUC']
    values = [global_m['precision'], global_m['recall'], global_m['f1'], global_m['auc']]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=labels, y=values, hue=labels, palette="viridis", legend=False)

    plt.ylim(0, 1.1)
    plt.title('Global Binary Metrics', fontsize=14, pad=15)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'binary_global_metrics.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. PER-CLASS F1 SCORE (Top 40 Classes, saved to root dir)
    # ==========================================
    class_metrics = metrics_dict['per_class']
    if not class_metrics: return

    # Sort classes by F1 score
    sorted_classes = sorted(class_metrics.items(), key=lambda item: item[1]['f1'], reverse=True)
    top_40 = sorted_classes[:40]

    c_names = [COCO_CLASSES[c_id] if c_id < len(COCO_CLASSES) else str(c_id) for c_id, _ in top_40]
    c_f1s = [m['f1'] for _, m in top_40]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=c_names, y=c_f1s, palette="mako")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.title('Per-Class F1 Scores (Top 40 Classes)', fontsize=14)
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'binary_per_class_f1.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. INDIVIDUAL PER-CLASS METRICS (Saved to class dirs)
    # ==========================================
    for c_id, metrics in class_metrics.items():
        class_name_clean = COCO_CLASSES[c_id].replace(" ", "_") if c_id < len(COCO_CLASSES) else f"class_{c_id}"

        # Define correct nested path for the class
        class_save_dir = os.path.join(save_dir, 'metrics_by_class', class_name_clean)
        os.makedirs(class_save_dir, exist_ok=True)

        # Plot class-specific binary metrics
        c_labels = ['Precision', 'Recall', 'F1 Score']
        c_values = [metrics['precision'], metrics['recall'], metrics['f1']]

        # Add AUC only if it's valid (not NaN)
        if not np.isnan(metrics['auc']):
            c_labels.append('AUC')
            c_values.append(metrics['auc'])

        plt.figure(figsize=(6, 4))
        ax = sns.barplot(x=c_labels, y=c_values, hue=c_labels, palette="mako", legend=False)

        plt.ylim(0, 1.1)
        plt.title(f'{COCO_CLASSES[c_id].capitalize()} Binary Metrics', fontsize=12)
        for i, v in enumerate(c_values):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(class_save_dir, f'binary_{class_name_clean}_metrics.png'), dpi=300)
        plt.close()


def plot_threshold_curves(pred_probs, gt_vectors, save_dir, steps=50):
    """Plots Precision, Recall, and F1 against varying binarization thresholds."""
    print("\nGenerating Threshold Trade-off Curves...")
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    thresholds = np.linspace(0.01, 0.99, steps)
    precisions = []
    recalls = []
    f1s = []

    flat_gts = gt_vectors.flatten()
    flat_probs = pred_probs.flatten()

    for t in thresholds:
        bin_preds = (flat_probs >= t).astype(int)

        # zero_division=0 prevents warnings at extreme thresholds
        p = precision_score(flat_gts, bin_preds, zero_division=0)
        r = recall_score(flat_gts, bin_preds, zero_division=0)
        f = f1_score(flat_gts, bin_preds, zero_division=0)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', color='#1f77b4', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', color='#ff7f0e', linewidth=2)
    plt.plot(thresholds, f1s, label='F1 Score', color='#2ca02c', linewidth=2, linestyle='--')

    # Find and highlight the threshold that maximizes F1
    best_idx = np.argmax(f1s)
    best_t = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    plt.plot(best_t, best_f1, marker='o', markersize=8, color='red')
    plt.text(best_t + 0.02, best_f1, f"max F1: {best_f1:.2f}\nthreshold: {best_t:.2f}", fontweight='bold')

    plt.title('Global Concept Metrics vs. Binarization Threshold', fontsize=15)
    plt.xlabel('Binarization Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()

    # This correctly saves to the root of the model directory
    plt.savefig(os.path.join(save_dir, 'binary_threshold_curves.png'), dpi=300)
    plt.close()
