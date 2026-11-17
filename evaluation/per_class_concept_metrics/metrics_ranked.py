"""Ranked concept metrics: the model's top-5 concepts vs. the GT top-5 salient
concepts (exact-set match, rank MAE, exact order at top-1/2/3/5), globally and
per class."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

from cbm_eval.coco import COCO_CLASSES


def evaluate_ranked_concepts(pred_logits, gt_vectors, class_ids, num_classes=80):
    """
    Evaluates predictions by looking at the Exact Set Match (Top 5)
    and the Rank Error (Mean Absolute Error of positions),
    along with exact matches for the Top 1, 2, 3, and 5 positions.
    """
    total_instances = len(pred_logits)

    # Global variables
    global_intersections = []
    global_exact_set_matches = 0
    global_exact_top1_matches = 0
    global_exact_top2_matches = 0
    global_exact_top3_matches = 0
    global_exact_top5_matches = 0
    global_rank_errors = []

    # Per-class variables
    class_metrics = {c: {
        'intersections': [], 'exact_sets': 0,
        'exact_top1': 0, 'exact_top2': 0, 'exact_top3': 0, 'exact_top5': 0,
        'rank_errors': [], 'count': 0
    } for c in range(num_classes)}

    for i in range(total_instances):
        c_id = class_ids[i]
        logits = pred_logits[i]
        gt = gt_vectors[i]

        # 1. Extract the 5 real concepts and their correct order
        true_concept_indices = np.where(gt > 0)[0]
        if len(true_concept_indices) == 0:
            continue

        true_order_indices = true_concept_indices[np.argsort(gt[true_concept_indices])]

        # 2. Extract the predicted Top 5
        # argsort sorts ascending, so we have to invert values in
        # order to extract the most activated 5 easier
        predicted_ranking_full = np.argsort(-logits)
        pred_top5_indices = predicted_ranking_full[:5]

        # --- METRIC 1: SET MATCH (Retrieval) ---
        set_true = set(true_order_indices)
        set_pred = set(pred_top5_indices)
        intersection_size = len(set_true.intersection(set_pred))

        is_exact_set = (intersection_size == len(set_true))

        # --- METRIC 2: STRICT ORDER (Only if the Exact Set was matched) ---
        is_exact_top1 = False
        is_exact_top2 = False
        is_exact_top3 = False
        is_exact_top5 = False

        if is_exact_set:
            pred_list = list(pred_top5_indices)
            true_list = list(true_order_indices)

            if pred_list[0] == true_list[0]:
                is_exact_top1 = True
            if pred_list[:2] == true_list[:2]:
                is_exact_top2 = True
            if pred_list[:3] == true_list[:3]:
                is_exact_top3 = True
            if pred_list == true_list:
                is_exact_top5 = True

        # --- METRIC 3: RANK ERROR (MAE) ---
        instance_rank_errors = []
        for true_rank_pos, concept_idx in enumerate(true_order_indices):
            pred_rank_pos = np.where(predicted_ranking_full == concept_idx)[0][0]
            error = abs(true_rank_pos - pred_rank_pos)
            instance_rank_errors.append(error)

        mean_rank_error = np.mean(instance_rank_errors) if instance_rank_errors else 0

        # Save global results
        global_intersections.append(intersection_size)
        if is_exact_set: global_exact_set_matches += 1
        if is_exact_top1: global_exact_top1_matches += 1
        if is_exact_top2: global_exact_top2_matches += 1
        if is_exact_top3: global_exact_top3_matches += 1
        if is_exact_top5: global_exact_top5_matches += 1
        global_rank_errors.append(mean_rank_error)

        # Save per-class results
        class_metrics[c_id]['count'] += 1
        class_metrics[c_id]['intersections'].append(intersection_size)
        if is_exact_set: class_metrics[c_id]['exact_sets'] += 1
        if is_exact_top1: class_metrics[c_id]['exact_top1'] += 1
        if is_exact_top2: class_metrics[c_id]['exact_top2'] += 1
        if is_exact_top3: class_metrics[c_id]['exact_top3'] += 1
        if is_exact_top5: class_metrics[c_id]['exact_top5'] += 1
        class_metrics[c_id]['rank_errors'].append(mean_rank_error)

    # --- CALCULATE GLOBALS ---
    avg_global_intersection = np.mean(global_intersections) if global_intersections else 0
    global_set_match_pct = (global_exact_set_matches / total_instances) * 100 if total_instances > 0 else 0
    avg_global_rank_error = np.mean(global_rank_errors) if global_rank_errors else 0

    # Calculate order percentages BASED ON Exact Set Matches
    denom_global = global_exact_set_matches if global_exact_set_matches > 0 else 1 # Avoid div by zero
    pct_global_top1 = (global_exact_top1_matches / denom_global) * 100 if global_exact_set_matches > 0 else 0
    pct_global_top2 = (global_exact_top2_matches / denom_global) * 100 if global_exact_set_matches > 0 else 0
    pct_global_top3 = (global_exact_top3_matches / denom_global) * 100 if global_exact_set_matches > 0 else 0
    pct_global_top5 = (global_exact_top5_matches / denom_global) * 100 if global_exact_set_matches > 0 else 0

    # --- PRINT GLOBAL METRICS ---
    print("\nGlobal Metrics")

    t_global = PrettyTable([
        " ", "Avg Intersection", "Exact Set Match", "MAE Error",
        "Exact TOP1 Match", "Exact TOP2 Match", "Exact TOP3 Match", "Exact TOP5 Match"
    ])
    t_global.add_row([
        "GLOBAL",
        f"{avg_global_intersection:.2f} / 5",
        f"{global_exact_set_matches}/{total_instances} ({global_set_match_pct:.2f}%)",
        f"{avg_global_rank_error:.2f}",
        f"{pct_global_top1:.2f}%",
        f"{pct_global_top2:.2f}%",
        f"{pct_global_top3:.2f}%",
        f"{pct_global_top5:.2f}%"
    ])
    print(t_global)

    # --- PRINT PER-CLASS METRICS ---
    print("\nPer-Class Metrics (First 40 Classes)")

    t_class = PrettyTable([
        "Class", "Count", "Avg Intersection", "Exact Set (%)", "MAE Error",
        "Exact TOP1 (%)", "Exact TOP2 (%)", "Exact TOP3 (%)", "Exact TOP5 (%)"
    ])

    # Iterate strictly through the first 40 classes (IDs 0 to 39)
    for c_id in range(min(40, num_classes)):
        metrics = class_metrics[c_id]

        # Only add to the table if the class actually appeared in the evaluation
        if metrics['count'] > 0:
            count = metrics['count']
            avg_inter = np.mean(metrics['intersections'])
            set_match_pct = (metrics['exact_sets'] / count) * 100
            err = np.mean(metrics['rank_errors'])

            # Calculate class order percentages based on Exact Set Matches
            set_matches = metrics['exact_sets']
            denom_cls = set_matches if set_matches > 0 else 1

            pct_cls_top1 = (metrics['exact_top1'] / denom_cls) * 100 if set_matches > 0 else 0
            pct_cls_top2 = (metrics['exact_top2'] / denom_cls) * 100 if set_matches > 0 else 0
            pct_cls_top3 = (metrics['exact_top3'] / denom_cls) * 100 if set_matches > 0 else 0
            pct_cls_top5 = (metrics['exact_top5'] / denom_cls) * 100 if set_matches > 0 else 0

            class_name = COCO_CLASSES[c_id] if c_id < len(COCO_CLASSES) else str(c_id)

            t_class.add_row([
                class_name, count, f"{avg_inter:.2f}", f"{set_match_pct:.2f}%", f"{err:.2f}",
                f"{pct_cls_top1:.2f}%", f"{pct_cls_top2:.2f}%", f"{pct_cls_top3:.2f}%", f"{pct_cls_top5:.2f}%"
            ])

    print(t_class)
    if num_classes > 40:
        print("... (Showing first 40 classes only)\n")

    return {
        'global': {
            'avg_intersection': avg_global_intersection,
            'set_match_pct': global_set_match_pct,
            'rank_error': avg_global_rank_error,
            'top1_match_pct': pct_global_top1,
            'top2_match_pct': pct_global_top2,
            'top3_match_pct': pct_global_top3,
            'top5_match_pct': pct_global_top5
        },
        'per_class': class_metrics
    }


def plot_ranked_metrics(metrics_dict, save_dir):
    """Generates and saves plots for the ranked metrics (Global and Per-Class)."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    global_m = metrics_dict['global']

    # ==========================================
    # 1. GLOBAL METRICS PLOT
    # ==========================================
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    # Scale intersection to percentage for easy comparison (x / 5 * 100)
    inter_pct = (global_m['avg_intersection'] / 5.0) * 100

    labels = [
        'Avg Intersection (%)', 'Exact Set Match (%)',
        'Exact Top 1 (%)', 'Exact Top 2 (%)',
        'Exact Top 3 (%)', 'Exact Top 5 (%)'
    ]
    values = [
        inter_pct, global_m['set_match_pct'],
        global_m['top1_match_pct'], global_m['top2_match_pct'],
        global_m['top3_match_pct'], global_m['top5_match_pct']
    ]

    ax1 = sns.barplot(x=labels, y=values, palette="rocket")
    plt.ylim(0, 100)
    plt.xticks(rotation=25, ha='right')
    plt.title('Global Ranked Matching Metrics')
    for i, v in enumerate(values):
        ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # Mean Rank Error (MAE)
    plt.subplot(1, 2, 2)
    ax2 = sns.barplot(x=['Mean Rank Error (MAE)'], y=[global_m['rank_error']], palette="dark:salmon")
    plt.title('Global Rank Position Error')
    plt.ylabel('Positions (Lower is better)')
    ax2.text(0, global_m['rank_error'] + 0.1, f"{global_m['rank_error']:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ranked_global_metrics.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. PER-CLASS METRICS PLOTS
    # ==========================================
    class_metrics = metrics_dict.get('per_class', {})
    if class_metrics:
        for c_id, metrics in class_metrics.items():
            count = metrics['count']
            if count > 0:
                class_name = COCO_CLASSES[c_id] if c_id < len(COCO_CLASSES) else str(c_id)
                safe_class_name = class_name.replace(" ", "_")

                # 1. Define correct nested path for the class
                class_save_dir = os.path.join(save_dir, 'metrics_by_class', safe_class_name)
                os.makedirs(class_save_dir, exist_ok=True)

                # Calculate percentages for the class
                avg_inter_pct = (np.mean(metrics['intersections']) / 5.0) * 100
                set_match_pct = (metrics['exact_sets'] / count) * 100
                err = np.mean(metrics['rank_errors'])

                set_matches = metrics['exact_sets']
                denom_cls = set_matches if set_matches > 0 else 1

                pct_cls_top1 = (metrics['exact_top1'] / denom_cls) * 100 if set_matches > 0 else 0
                pct_cls_top2 = (metrics['exact_top2'] / denom_cls) * 100 if set_matches > 0 else 0
                pct_cls_top3 = (metrics['exact_top3'] / denom_cls) * 100 if set_matches > 0 else 0
                pct_cls_top5 = (metrics['exact_top5'] / denom_cls) * 100 if set_matches > 0 else 0

                # Plot
                plt.figure(figsize=(14, 6))

                plt.subplot(1, 2, 1)
                c_values = [avg_inter_pct, set_match_pct, pct_cls_top1, pct_cls_top2, pct_cls_top3, pct_cls_top5]

                ax1 = sns.barplot(x=labels, y=c_values, palette="mako")
                plt.ylim(0, 100)
                plt.xticks(rotation=25, ha='right')
                plt.title(f'Ranked Metrics: {class_name.capitalize()} (N={count})')
                for i, v in enumerate(c_values):
                    ax1.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

                plt.subplot(1, 2, 2)
                ax2 = sns.barplot(x=['Mean Rank Error (MAE)'], y=[err], palette="flare")
                plt.title(f'Rank Error: {class_name.capitalize()}')
                plt.ylabel('Positions (Lower is better)')
                ax2.text(0, err + 0.1, f"{err:.2f}", ha='center', fontweight='bold')

                plt.tight_layout()

                # 2. Save inside the class-specific directory
                plt.savefig(os.path.join(class_save_dir, f'ranked_{safe_class_name}_metrics.png'), dpi=300)
                plt.close()
