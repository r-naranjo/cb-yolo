"""Concept frequency analysis for the per-class (400-concept) vocabulary: GT vs.
predicted activation frequencies (global long-tail + per-class bars of the 5
concepts owned by each class) and GT vs. predicted instance counts per class."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import pearsonr

from cbm_eval.coco import CAT_TO_IDX, COCO_CLASSES
from cbm_eval.concepts import read_concept_names



def analyze_concept_distributions(pred_probs, gt_vectors, class_ids, concepts_json_path, threshold=0.5, save_dir=None):
    print("\n--- Concept Distribution & Frequency Study ---")

    all_concept_names = read_concept_names(concepts_json_path)

    binary_preds = (pred_probs >= threshold).astype(int)
    num_concepts = gt_vectors.shape[1]

    # --- 1. GLOBAL DISTRIBUTION ANALYSIS ---
    gt_global_freq = np.sum(gt_vectors, axis=0)
    pred_global_freq = np.sum(binary_preds, axis=0)

    correlation, p_value = pearsonr(gt_global_freq, pred_global_freq)
    print(f"Global GT vs Pred Frequency Correlation: {correlation:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

        # Detailed Long-Tail Distribution Plot
        plt.figure(figsize=(24, 10)) # Very wide to accommodate 400 text labels

        # Sort by GT frequency (most to least)
        sort_idx = np.argsort(-gt_global_freq)
        sorted_gt = gt_global_freq[sort_idx]
        sorted_pred = pred_global_freq[sort_idx]
        sorted_names = [all_concept_names[i] for i in sort_idx]

        plt.fill_between(range(num_concepts), sorted_gt, alpha=0.3, color='#2ca02c', label='GT Area')
        plt.plot(range(num_concepts), sorted_gt, color='#2ca02c', linewidth=2, label='Ground Truth')
        plt.plot(range(num_concepts), sorted_pred, color='#d62728', alpha=0.8, linewidth=1.5, label='Predictions')

        plt.title(f'Global Concept Long-Tail Distribution (Correlation: {correlation:.3f})', fontsize=18)
        plt.xlabel('Concepts', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)

        # Set all 400 concept names vertically on the X-axis
        plt.xticks(range(num_concepts), sorted_names, rotation=90, fontsize=6)
        plt.xlim(0, num_concepts - 1)
        plt.legend(fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'distribution_global_frequencies.png'), dpi=300)
        plt.close()

    # --- 2. PER-CLASS DISTRIBUTION ANALYSIS ---
    unique_classes = np.unique(class_ids)

    if save_dir:
        for c_id in unique_classes:
            class_mask = class_ids == c_id

            # The 5 indices that officially belong to this class
            concept_start = c_id * 5
            concept_indices = list(range(concept_start, concept_start + 5))
            class_concept_names = all_concept_names[concept_start:concept_start + 5]

            # Extract frequencies just for these 5 concepts
            gt_class_freq = np.sum(gt_vectors[class_mask][:, concept_indices], axis=0)
            pred_class_freq = np.sum(binary_preds[class_mask][:, concept_indices], axis=0)

            class_name_clean = COCO_CLASSES[c_id].replace(" ", "_") if c_id < len(COCO_CLASSES) else f"class_{c_id}"
            class_dir = os.path.join(save_dir, class_name_clean)
            os.makedirs(class_dir, exist_ok=True)

            # Plot the 5 official concepts for this class
            plt.figure(figsize=(8, 6))
            x = np.arange(5)
            width = 0.35

            plt.bar(x - width/2, gt_class_freq, width, label='Ground Truth', color='#2ca02c')
            plt.bar(x + width/2, pred_class_freq, width, label='Predictions', color='#d62728')

            plt.title(f'{COCO_CLASSES[c_id].capitalize()} - Official Concept Frequencies (N={np.sum(class_mask)})', fontsize=14)

            # Clean up long concept names so they fit nicely
            wrapped_labels = [label.replace(' ', '\n') if len(label) > 15 else label for label in class_concept_names]
            plt.xticks(x, wrapped_labels, rotation=0, fontsize=10)

            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()

            plt.savefig(os.path.join(class_dir, f'distribution_{class_name_clean}_frequencies.png'), dpi=300)
            plt.close()

    return gt_global_freq, pred_global_freq


def plot_class_instance_counts(gt_json_path, coco_results_list, coco_classes, save_dir, top_k=40):
    """
    Plots the count of instances per class for Ground Truth vs Predictions.
    Filters and sorts by the most frequent classes to keep the plot readable.
    """
    print("\nGenerating Class Instance Count Plot...")
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Count Ground Truth instances
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    gt_counts = Counter()
    for ann in gt_data['annotations']:
        class_idx = CAT_TO_IDX[ann['category_id']]

        gt_counts[class_idx] += 1

    # 2. Count Predicted instances
    pred_counts = Counter()
    for res in coco_results_list:
        class_idx = CAT_TO_IDX[res['category_id']]

        pred_counts[class_idx] += 1

    # 3. Identify classes that actually appear in either GT or Preds
    active_classes = set(gt_counts.keys()).union(set(pred_counts.keys()))

    # Sort active classes by their Ground Truth frequency (descending)
    sorted_classes = sorted(list(active_classes), key=lambda x: gt_counts[x], reverse=True)

    # Limit to top_k classes so the x-axis doesn't get unreadable (e.g., top 40)
    top_classes = sorted_classes[:top_k]

    labels = [coco_classes[i] if i < len(coco_classes) else str(i) for i in top_classes]
    gt_values = [gt_counts[i] for i in top_classes]
    pred_values = [pred_counts[i] for i in top_classes]

    # 4. Generate the Grouped Bar Plot
    x = np.arange(len(labels))

    width = 0.28

    plt.figure(figsize=(32, 8))

    plt.bar(x - width/2, gt_values, width,
            label='Ground Truth',
            color='blue')

    plt.bar(x + width/2, pred_values, width,
            label='Predictions',
            color='red')

    plt.ylabel('Number of Instances', fontsize=14)
    plt.title(f'Instance Counts per Class: Ground Truth vs Predictions (Top {len(top_classes)} Classes)', fontsize=18)

    plt.xticks(x, labels, rotation=60, ha='right', fontsize=9)

    plt.legend(fontsize=13)

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, 'detection_instance_counts.png'), dpi=500)
    plt.close()
