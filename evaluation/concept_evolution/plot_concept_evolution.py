#!/usr/bin/env python3
"""
plot_concept_evolution.py
--------------------------

Reads the JSON produced by compute_concept_evolution.py and generates
four plots plus a summary CSV.

Plots produced
--------------
01_mean_auc_ap.png
    Mean AUC-ROC and Mean AP over epochs, with IQR shaded band.
    IQR tells you whether learning is uniform or concentrated in a few
    easy concepts.
    If AUC rises but AP does not, the model learns to rank concepts but
    still predicts near-zero probabilities for many positives.

02_concept_coverage.png
    Bar chart: # concepts with AUC ≥ 0.70 (learned, green) and
    AUC < 0.60 (near-random, red) per epoch.  You want green to grow
    and red to shrink monotonically.

03_per_concept_heatmap.png
    Heatmap of per-concept AUC across epochs (concepts × epochs).
    Sorted by final AUC so that the best-learned concepts are at the top.
    Dark grey = not enough positive samples to compute AUC.
    This is the most diagnostic plot: it shows which concepts are learned
    early, which need many epochs, and which never learn.

04_auc_vs_ap_final.png
    Scatter of AUC vs AP at the final checkpoint.
    Concepts far below the diagonal have decent ranking (high AUC) but
    poor precision (low AP) — typical sign of class-imbalance collapse
    where the model predicts near-zero for that concept too often.
    The 8 worst-gap concepts are annotated.

Usage:
    python evaluation/concept_evolution/plot_concept_evolution.py \
        --results outputs/concept_evolution.json \
        --output_dir outputs/concept_evolution_plots
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plots for compute_concept_evolution.py results")
    p.add_argument("--results", required=True,
                   help="JSON file produced by compute_concept_evolution.py")
    p.add_argument("--output_dir", default="outputs/concept_evolution_plots")
    p.add_argument("--auc_learned", type=float, default=0.70,
                   help="AUC threshold above which a concept is 'learned'")
    p.add_argument("--auc_random", type=float, default=0.60,
                   help="AUC threshold below which a concept is 'near-random'")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str):
    with open(path) as f:
        data = json.load(f)

    concept_names = data["concept_names"]
    epochs        = data["epochs"]               # list of ints
    match_counts  = data["match_counts"]         # list of ints
    K             = len(concept_names)
    E             = len(epochs)

    # Rebuild (E, K) matrices — None → NaN
    auc_matrix = np.full((E, K), np.nan)
    ap_matrix  = np.full((E, K), np.nan)
    rec_matrix = np.full((E, K), np.nan)

    for i, m in enumerate(data["metrics"]):
        for k, v in enumerate(m["auc"]):
            if v is not None:
                auc_matrix[i, k] = v
        for k, v in enumerate(m["ap"]):
            if v is not None:
                ap_matrix[i, k] = v
        if "recall" in m:
            for k, v in enumerate(m["recall"]):
                if v is not None:
                    rec_matrix[i, k] = v

    return epochs, match_counts, concept_names, auc_matrix, ap_matrix, rec_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nanpercentile_rows(mat, q):
    out = np.full(mat.shape[0], np.nan)
    for i, row in enumerate(mat):
        v = row[~np.isnan(row)]
        if len(v):
            out[i] = np.percentile(v, q)
    return out

def _nanmean_rows(mat):
    out = np.full(mat.shape[0], np.nan)
    for i, row in enumerate(mat):
        v = row[~np.isnan(row)]
        if len(v):
            out[i] = v.mean()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(epochs, match_counts, concept_names,
             auc_matrix, ap_matrix, rec_matrix, output_dir: Path,
             auc_learned: float, auc_random: float):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_arr = np.array(epochs)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_auc = _nanmean_rows(auc_matrix)
    p25_auc  = _nanpercentile_rows(auc_matrix, 25)
    p75_auc  = _nanpercentile_rows(auc_matrix, 75)

    mean_ap  = _nanmean_rows(ap_matrix)
    p25_ap   = _nanpercentile_rows(ap_matrix, 25)
    p75_ap   = _nanpercentile_rows(ap_matrix, 75)

    mean_rec = _nanmean_rows(rec_matrix)
    p25_rec  = _nanpercentile_rows(rec_matrix, 25)
    p75_rec  = _nanpercentile_rows(rec_matrix, 75)

    n_valid   = np.array([np.sum(~np.isnan(row)) for row in auc_matrix])
    n_learned = np.array([np.sum(row[~np.isnan(row)] >= auc_learned) for row in auc_matrix])
    n_random  = np.array([np.sum(row[~np.isnan(row)] <  auc_random)  for row in auc_matrix])

    # ── 01 Mean AUC + AP + Recall ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Concept prediction quality over training\n"
                 "(anchor-matched instances only)", fontsize=13, fontweight="bold")

    plot_configs = [
        (axes[0, 0], mean_auc, p25_auc, p75_auc, "Mean AUC-ROC",           "#2196F3"),
        (axes[0, 1], mean_ap,  p25_ap,  p75_ap,  "Mean Average Precision", "#4CAF50"),
        (axes[1, 0], mean_rec, p25_rec, p75_rec, "Mean Recall (at 0.5)",   "#9C27B0"),
    ]

    # Hide the empty 4th subplot
    axes[1, 1].axis("off")

    for ax, mean, p25, p75, label, color in plot_configs:
        ax.plot(epochs_arr, mean, color=color, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(epochs_arr, p25, p75, color=color, alpha=0.18, label="IQR (25–75 %)")
        ax.axhline(0.5,         color="grey",   ls="--", lw=1, label="Baseline (0.5)")
        if "AUC" in label:
            ax.axhline(auc_learned, color="orange", ls=":",  lw=1,
                       label=f"'Learned' threshold ({auc_learned})")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(label,   fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Secondary x-axis: matched instance count attached to top-left plot
    ax2 = axes[0, 0].twiny()
    ax2.set_xlim(axes[0, 0].get_xlim())
    ax2.set_xticks(epochs_arr)
    ax2.set_xticklabels([f"{c}" for c in match_counts], fontsize=6, rotation=45)
    ax2.set_xlabel("Matched instances", fontsize=8, color="grey")

    plt.tight_layout()
    out1 = output_dir / "01_mean_auc_ap.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out1}")

    # ── 02 Concept coverage ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    w = max(1, (epochs_arr[1] - epochs_arr[0]) * 0.35) if len(epochs_arr) > 1 else 3
    ax.bar(epochs_arr - w/2, n_learned, width=w, color="#4CAF50", alpha=0.85,
           label=f"AUC ≥ {auc_learned}  (learned)")
    ax.bar(epochs_arr + w/2, n_random,  width=w, color="#F44336", alpha=0.85,
           label=f"AUC < {auc_random}  (near-random)")
    ax.plot(epochs_arr, n_valid, color="steelblue", lw=1.5, ls="--",
            marker=".", label="Measurable concepts (enough positives)")
    ax.set_xlabel("Epoch",              fontsize=11)
    ax.set_ylabel("Number of concepts", fontsize=11)
    ax.set_title("Concept coverage over training", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = output_dir / "02_concept_coverage.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out2}")

    # ── 03 Per-concept heatmap ────────────────────────────────────────────────
    measurable = ~np.all(np.isnan(auc_matrix), axis=0)
    hm_data    = auc_matrix[:, measurable].T          # (K_valid, E)
    hm_names   = [concept_names[k] for k in np.where(measurable)[0]]

    # Sort by mean AUC over last 3 checkpoints (or all if fewer)
    n_tail   = min(3, hm_data.shape[1])
    tail_auc = np.nanmean(hm_data[:, -n_tail:], axis=1)
    order    = np.argsort(tail_auc)[::-1]
    hm_data  = hm_data[order]
    hm_names = [hm_names[i] for i in order]

    K_vis  = hm_data.shape[0]
    fig_h  = max(8, K_vis * 0.22)
    fig_w  = max(10, len(epochs) * 0.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#2e2e2e")
    im = ax.imshow(hm_data, aspect="auto", cmap=cmap,
                   vmin=0.5, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AUC-ROC", fraction=0.02, pad=0.01)

    ax.set_xticks(range(len(epochs_arr)))
    ax.set_xticklabels([str(e) for e in epochs_arr],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K_vis))
    ax.set_yticklabels(hm_names, fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_title(
        "Per-concept AUC-ROC over training\n"
        "(sorted by final AUC ↑ ;  dark grey = not enough positive samples)",
        fontweight="bold",
    )
    plt.tight_layout()
    out3 = output_dir / "03_per_concept_heatmap.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out3}")

    # ── 04 AUC vs AP scatter at final checkpoint ─────────────────────────────
    last_auc = auc_matrix[-1]
    last_ap  = ap_matrix[-1]
    valid    = ~np.isnan(last_auc) & ~np.isnan(last_ap)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(last_auc[valid], last_ap[valid],
                    c=(last_ap - last_auc)[valid],
                    cmap="coolwarm", vmin=-0.35, vmax=0.35,
                    alpha=0.75, s=45, edgecolors="none")
    plt.colorbar(sc, ax=ax,
                 label="AP − AUC  (negative = potential prediction collapse)")

    # Annotate 8 worst-gap concepts (high AUC, low AP)
    gap      = (last_auc - last_ap)
    gap_copy = np.where(valid, gap, -np.inf)
    worst8   = np.argsort(gap_copy)[::-1][:8]
    for k in worst8:
        if valid[k]:
            ax.annotate(concept_names[k], (last_auc[k], last_ap[k]),
                        fontsize=6.5, alpha=0.85,
                        xytext=(5, 2), textcoords="offset points")

    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", lw=1, alpha=0.4, label="AUC = AP")
    ax.set_xlabel("AUC-ROC  (last checkpoint)", fontsize=11)
    ax.set_ylabel("Average Precision  (last checkpoint)", fontsize=11)
    ax.set_title(
        f"AUC vs AP — epoch {epochs[-1]}\n"
        "(concepts far below diagonal: ranks well but precision collapses)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out4 = output_dir / "04_auc_vs_ap_final.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {out4}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(epochs, auc_matrix, ap_matrix, rec_matrix, concept_names, output_dir: Path):
    import csv
    out = output_dir / "concept_metrics_by_epoch.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["concept"]
        for e in epochs:
            header += [f"auc_ep{e}", f"ap_ep{e}", f"recall_ep{e}"]
        w.writerow(header)
        for k, name in enumerate(concept_names):
            row = [name]
            for i in range(len(epochs)):
                auc = auc_matrix[i, k]
                ap  = ap_matrix[i, k]
                rec = rec_matrix[i, k]
                row += [
                    "" if np.isnan(auc) else round(float(auc), 4),
                    "" if np.isnan(ap)  else round(float(ap),  4),
                    "" if np.isnan(rec) else round(float(rec), 4),
                ]
            w.writerow(row)
    print(f"[CSV]  {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(epochs, concept_names, auc_matrix, ap_matrix, rec_matrix,
                  auc_learned, auc_random):
    final_auc = auc_matrix[-1]
    final_ap  = ap_matrix[-1]
    final_rec = rec_matrix[-1]

    learned   = int(np.sum(final_auc[~np.isnan(final_auc)] >= auc_learned))
    random_   = int(np.sum(final_auc[~np.isnan(final_auc)] <  auc_random))

    print("\n" + "="*62)
    print("  FINAL CHECKPOINT SUMMARY")
    print("="*62)
    print(f"  Epoch                      : {epochs[-1]}")
    print(f"  Mean AUC                   : {np.nanmean(final_auc):.4f}")
    print(f"  Mean AP                    : {np.nanmean(final_ap):.4f}")
    print(f"  Mean Recall                : {np.nanmean(final_rec):.4f}")
    print(f"  Concepts AUC ≥ {auc_learned}         : {learned}")
    print(f"  Concepts AUC <  {auc_random}         : {random_}")

    measurable = [(k, final_auc[k], final_ap[k], final_rec[k])
                  for k in range(len(concept_names)) if not np.isnan(final_auc[k])]

    print("\n  Top-10 best concepts (final AUC):")
    for k, auc, ap, rec in sorted(measurable, key=lambda x: -x[1])[:10]:
        print(f"    {concept_names[k]:<38}  AUC={auc:.4f}  AP={ap:.4f}  Rec={rec:.4f}")

    print("\n  Top-10 worst concepts (final AUC, measurable only):")
    for k, auc, ap, rec in sorted(measurable, key=lambda x: x[1])[:10]:
        print(f"    {concept_names[k]:<38}  AUC={auc:.4f}  AP={ap:.4f}  Rec={rec:.4f}")

    print("\n  Top-8 largest AUC−AP gap (potential collapse):")
    gap_list = [(k, final_auc[k], final_ap[k], final_auc[k]-final_ap[k])
                for k, auc, ap, rec in measurable if not np.isnan(final_ap[k])]
    for k, auc, ap, g in sorted(gap_list, key=lambda x: -x[3])[:8]:
        print(f"    {concept_names[k]:<38}  AUC={auc:.4f}  AP={ap:.4f}  gap={g:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    print(f"[INFO] Loading results from {args.results}")
    epochs, match_counts, concept_names, auc_matrix, ap_matrix, rec_matrix = \
        load_results(args.results)

    print(f"[INFO] {len(epochs)} epochs: {epochs}")
    print(f"[INFO] {len(concept_names)} concepts")

    print("[INFO] Generating plots...")
    plot_all(epochs, match_counts, concept_names,
             auc_matrix, ap_matrix, rec_matrix, output_dir,
             args.auc_learned, args.auc_random)

    save_csv(epochs, auc_matrix, ap_matrix, rec_matrix, concept_names, output_dir)
    print_summary(epochs, concept_names, auc_matrix, ap_matrix, rec_matrix,
                  args.auc_learned, args.auc_random)


if __name__ == "__main__":
    main()
