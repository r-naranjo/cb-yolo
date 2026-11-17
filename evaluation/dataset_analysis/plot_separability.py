#!/usr/bin/env python3
"""Separability plots (dataset only, see separability.py).

Subcommands:
  sweep      mSeparability and mean concepts/class vs. activation filter
             (0..100 % in 5 % steps) -> separability_vs_filter.png
  heatmaps   (classes x concepts) heatmaps of the % of a class's annotations
             in which each concept is active, masked below each threshold
             -> concept_heatmap_<t>.png, plus a bar plot of the active
             concepts of one class -> class_barplot_<class>.png
  clustered  binary (classes x concepts) matrix of the filtered scenario,
             both axes hierarchically clustered (Jaccard distance, Ward
             linkage) -> clustered_heatmap_<t>.png
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import CAT_TO_IDX, COCO_CLASSES, NUM_COCO_CLASSES  # noqa: E402
from cbm_eval.concepts import read_concept_names  # noqa: E402

from separability import build_scenario_from_json, filter_scenario, separability_metrics  # noqa: E402


def plot_sweep(scenario, appearance_matrix, class_counts, output_dir, plt):
    import matplotlib.ticker as mticker

    thresholds = np.arange(0.0, 1.05, 0.05)
    num_concepts = appearance_matrix.shape[0]
    separabilities, concepts_per_class = [], []
    for t in thresholds:
        _, m_cpc, m_s = separability_metrics(
            filter_scenario(scenario, appearance_matrix, class_counts, t), num_concepts)
        separabilities.append(m_s)
        concepts_per_class.append(m_cpc)

    labels = [f"{int(round(t * 100))}%" for t in thresholds]
    color_sep, color_len = "#2563EB", "#DC2626"
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(labels, separabilities, color=color_sep, linewidth=2.5, marker="o", markersize=5,
             label="mSeparability")
    ax1.set_xlabel("Activation threshold (% of annotations per class)", fontsize=11)
    ax1.set_ylabel("Mean separability (mS)", color=color_sep, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_sep)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax2 = ax1.twinx()
    ax2.plot(labels, concepts_per_class, color=color_len, linewidth=2.5, marker="s", markersize=5,
             linestyle="--", label="Mean concepts / class")
    ax2.set_ylabel("Mean concepts per class (mCpC)", color=color_len, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_len)
    for y in (0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        ax1.axhline(y=y, color=color_sep, linewidth=0.5, alpha=0.2)
    for y in (20, 40, 50, 60, 80, 100, 120):
        ax2.axhline(y=y, color=color_len, linewidth=0.5, alpha=0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9, framealpha=0.85)
    ax1.set_xticks(labels[::max(1, len(labels) // 10)])
    ax1.tick_params(axis="x", rotation=45)
    plt.title("Mean separability and mean concepts per class vs. activation filter", fontsize=13, pad=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "separability_vs_filter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_heatmaps(appearance_matrix, class_counts, concept_names, thresholds_pct, barplot_class,
                  output_dir, plt):
    import seaborn as sns

    counts = np.asarray(class_counts, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        percent = np.where(appearance_matrix != 0, appearance_matrix * 100 / counts[None, :], 0.0)
    matrix = percent.T  # (classes, concepts)
    class_labels = list(COCO_CLASSES)

    for t in thresholds_pct:
        fig = plt.figure(figsize=(24, 16))
        ax = sns.heatmap(matrix, cmap="plasma", mask=(matrix < t) | (matrix == 0), vmin=0, vmax=100,
                         cbar_kws={"ticks": [0, 20, 40, 60, 80, 100]},
                         xticklabels=concept_names, yticklabels=class_labels,
                         linewidths=0.1, linecolor="gray")
        ax.set_title(f"Percentage of concept appearances per class (>= {t}%)", fontsize=18)
        ax.set_xlabel("Concepts", fontsize=14)
        ax.set_ylabel("Classes", fontsize=14)
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, f"concept_heatmap_{t:g}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")

    if barplot_class is None:
        return
    cls_idx = class_labels.index(barplot_class)
    values = matrix[cls_idx, :]
    active = np.where(values > 0)[0]
    names = [concept_names[i] for i in active]
    fig = plt.figure(figsize=(24, 10))
    ax = fig.add_subplot(111)
    ax.bar(names, values[active], color="royalblue", edgecolor="black", alpha=0.8)
    for frac, color in ((0.25, "darkorange"), (0.50, "red"), (0.75, "darkred")):
        ax.axhline(y=frac * 100, color=color, linestyle="--", linewidth=2.5,
                   label=f"{int(frac * 100)}% filter ({class_counts[cls_idx] * frac:.1f} annotations)")
    ax.set_title(f"Active concepts for class '{barplot_class}'", fontsize=18)
    ax.set_ylabel("% of the class's annotations", fontsize=14)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_xlim(-1, len(names))
    ax.set_ylim(0, max(values.max(), 75) * 1.25)
    ax.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"class_barplot_{barplot_class.replace(' ', '_')}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_clustered(scenario, appearance_matrix, class_counts, categories, concept_names, thresholds_pct,
                   output_dir, plt):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    num_concepts = appearance_matrix.shape[0]
    for t in thresholds_pct:
        filtered = filter_scenario(scenario, appearance_matrix, class_counts, t / 100.0)
        cat_ids = sorted(filtered)
        M = np.zeros((len(cat_ids), num_concepts), dtype=float)
        for row, cat_id in enumerate(cat_ids):
            M[row, list(filtered[cat_id])] = 1.0
        active = M.sum(axis=0) > 0
        M = M[:, active]
        if M.shape[1] < 2:
            print(f"threshold {t}%: fewer than 2 active concepts, skipped")
            continue
        concept_labels = [concept_names[i] for i in np.where(active)[0]]
        row_order = dendrogram(linkage(pdist(M, metric="jaccard"), method="ward"), no_plot=True)["leaves"]
        col_order = dendrogram(linkage(pdist(M.T, metric="jaccard"), method="ward"), no_plot=True)["leaves"]

        fig, ax = plt.subplots(figsize=(28, 16))
        im = ax.imshow(M[np.ix_(row_order, col_order)], aspect="auto", cmap="Blues", interpolation="none")
        plt.colorbar(im, ax=ax, label="Concept present")
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([cat_id_to_name.get(cat_ids[i], str(cat_ids[i])) for i in row_order], fontsize=8)
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels([concept_labels[i] for i in col_order], fontsize=5, rotation=90)
        ax.set_xlabel("Concepts (clustered)")
        ax.set_ylabel("Classes (clustered)")
        ax.set_title(f"Clustered class/concept matrix - {t:g}% filter")
        plt.tight_layout()
        path = os.path.join(output_dir, f"clustered_heatmap_{t:g}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plot", choices=["sweep", "heatmaps", "clustered"])
    parser.add_argument("--annotations", required=True,
                        help="COCO-style JSON with a binary concept_vector per annotation")
    parser.add_argument("--concepts", default=None,
                        help="Concept-name file (required for heatmaps/clustered labels)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="Thresholds in percent (default heatmaps: 0 5 ... 75; clustered: 0 5 ... 75)")
    parser.add_argument("--barplot_class", default="person",
                        help="heatmaps: class whose active concepts are bar-plotted ('' to skip)")
    parser.add_argument("--output_dir", default="outputs/separability")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    scenario, appearance_matrix, class_counts, categories = build_scenario_from_json(args.annotations)
    num_concepts = appearance_matrix.shape[0]
    concept_names = (read_concept_names(args.concepts) if args.concepts
                     else [f"concept_{i}" for i in range(num_concepts)])
    if len(concept_names) != num_concepts:
        sys.exit(f"{args.concepts} has {len(concept_names)} names, annotations have {num_concepts} concepts")
    thresholds = args.thresholds if args.thresholds is not None else list(range(0, 80, 5))
    assert appearance_matrix.shape[1] == NUM_COCO_CLASSES and set(scenario) <= set(CAT_TO_IDX)

    if args.plot == "sweep":
        plot_sweep(scenario, appearance_matrix, class_counts, args.output_dir, plt)
    elif args.plot == "heatmaps":
        plot_heatmaps(appearance_matrix, class_counts, concept_names, thresholds,
                      args.barplot_class or None, args.output_dir, plt)
    else:
        plot_clustered(scenario, appearance_matrix, class_counts, categories, concept_names, thresholds,
                       args.output_dir, plt)


if __name__ == "__main__":
    main()
