#!/usr/bin/env python3
"""Minimum number of concept corrections needed to flip a misclassification.

Subset: GT instances that are
  * well localized   - best-overlapping detection (any class, per-class NMS)
                       has IoU >= --iou_thresh,
  * misclassified    - that detection's class != GT class,
  * high confidence  - detection score (obj * cls) >= --conf_thresh.

For each instance, an oracle replaces concept probabilities with their GT
binary values. Because concept_to_cls is linear (cls = W p + b), the margin
between the wrongly predicted class and the GT class,

    margin(p) = (W[pred] - W[gt]) @ p + const,

changes by an independent amount per corrected concept j:

    delta_j = (W[pred, j] - W[gt, j]) * (gt_j - p_j).

The smallest set that makes the margin negative is therefore the prefix of
concepts sorted by delta_j ascending (closed form, no search). Each minimal
set is then verified against the true arg-max over all classes
(`flip_confirmed_by_argmax`), since beating the predicted class does not by
itself guarantee beating every other class. Instances that cannot be flipped
even with all K concepts corrected are counted as `num_unflippable`.

Reported: distribution (mean/median/percentiles, histogram, CDF) of the
minimum number of corrections and P(min_corrections <= k).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import COCO_CLASSES  # noqa: E402
from cbm_eval.concepts import load_concept_names  # noqa: E402
from cbm_eval.io import load_coco_annotations, write_json  # noqa: E402
from cbm_eval.linear_head import cls_logits_from_concept_probs, extract_concept_to_cls_weights  # noqa: E402
from cbm_eval.metrics import sigmoid  # noqa: E402
from cbm_eval.model import add_model_args, load_predictor  # noqa: E402

from concept_deactivation import collect_matches  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(
        description="CB-YOLOX minimum concept corrections to flip a misclassification")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True,
                        help="COCO-style annotation JSON with binary concept_vector per annotation "
                             "(source of the oracle concept values)")
    add_model_args(parser)
    parser.add_argument("--concepts", default=None, help="Concept-name file (labels only)")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="GT<->detection IoU threshold")
    parser.add_argument("--conf_thresh", type=float, default=0.5,
                        help="Minimum detection score for a misclassification to enter the subset "
                             "(not the same as --conf, the post-processing threshold)")
    parser.add_argument("--report_at_k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="Report P(min_corrections <= k) for these k")
    parser.add_argument("--log_concepts_cap", type=int, default=15,
                        help="Max corrected-concept names written per instance to intervention_log.json")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output_dir", default="outputs/concept_intervention_accuracy")
    return parser


def minimum_corrections(probs, gt_concepts, gt_classes, pred_classes, scale_idx, scale_weights):
    """Closed-form minimum number of oracle concept corrections per instance.

    Returns (min_corrections (N,) with -1 = unflippable, has_flip (N,),
    flip_confirmed (N,), order (N, K) correction order, class logits before
    correction (N, C))."""
    n, k = probs.shape
    rows = np.arange(n)
    cls_before = cls_logits_from_concept_probs(probs, scale_idx, scale_weights)
    margin_before = cls_before[rows, pred_classes] - cls_before[rows, gt_classes]

    delta = np.zeros((n, k), dtype=np.float64)
    for s, (W, _) in enumerate(scale_weights):
        mask = scale_idx == s
        if not np.any(mask):
            continue
        delta[mask] = (W[pred_classes[mask]] - W[gt_classes[mask]]) * (gt_concepts[mask] - probs[mask])

    order = np.argsort(delta, axis=1)  # most margin-reducing first
    margin_after = margin_before[:, None] + np.cumsum(np.take_along_axis(delta, order, axis=1), axis=1)
    crossed = margin_after < 0
    has_flip = crossed.any(axis=1)
    min_corrections = np.where(has_flip, np.argmax(crossed, axis=1) + 1, -1)

    # Verify each minimal set against the arg-max over all classes.
    rank_position = np.empty_like(order)
    rank_position[rows[:, None], order] = np.arange(k)[None, :]
    include = (rank_position < min_corrections[:, None]) & has_flip[:, None]
    corrected = np.where(include, gt_concepts, probs)
    argmax_after = np.argmax(cls_logits_from_concept_probs(corrected, scale_idx, scale_weights), axis=1)
    flip_confirmed = has_flip & (argmax_after == gt_classes)
    return min_corrections, has_flip, flip_confirmed, order, cls_before


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prettytable import PrettyTable

    os.makedirs(args.output_dir, exist_ok=True)
    predictor, model, _ = load_predictor(args)
    K = predictor.num_concepts
    concept_names = load_concept_names(args.concepts, K)
    scale_weights = extract_concept_to_cls_weights(model)

    images_meta, ann_dict = load_coco_annotations(args.json_binary)
    if args.limit:
        images_meta = images_meta[:args.limit]

    records = collect_matches(args, predictor, images_meta, ann_dict)
    print(f"GT instances matched at IoU >= {args.iou_thresh} (any predicted class): {len(records)}")
    if not records:
        print("No matched instances - check --iou_thresh, paths or the checkpoint.")
        return

    subset = [r for r in records if r["pred_class"] != r["gt_class"] and r["score"] >= args.conf_thresh]
    N = len(subset)
    print(f"Misclassified, high-confidence (score >= {args.conf_thresh}) instances: {N}")
    if not subset:
        print("Empty subset - relax --conf_thresh/--iou_thresh.")
        return

    probs = sigmoid(np.stack([r["logits"] for r in subset])).astype(np.float32)
    gt_classes = np.array([r["gt_class"] for r in subset], dtype=np.int64)
    pred_classes = np.array([r["pred_class"] for r in subset], dtype=np.int64)
    scale_idx = np.array([r["scale_idx"] for r in subset], dtype=np.int64)
    gt_concepts = np.stack([r["gt_concepts_bin"] for r in subset])

    min_corr, has_flip, flip_confirmed, order, cls_before = minimum_corrections(
        probs, gt_concepts, gt_classes, pred_classes, scale_idx, scale_weights)

    mismatch = int(np.sum(np.argmax(cls_before, axis=1) != pred_classes))
    if mismatch:
        print(f"[WARN] {mismatch}/{N} instances: top-1 class recomputed from concept_to_cls differs from "
              f"the detection's class (a few are expected from fp16 rounding; many indicate a bug).")

    log = []
    for i, r in enumerate(subset):
        d = int(min_corr[i])
        if d > 0:
            names = [concept_names[c] for c in order[i, :min(d, args.log_concepts_cap)].tolist()]
            truncated = d > args.log_concepts_cap
        else:
            names, truncated = [], False
        log.append({
            "image_id": int(r["image_id"]),
            "anchor_idx": r["anchor_idx"],
            "iou": r["iou"],
            "score": r["score"],
            "gt_class": COCO_CLASSES[r["gt_class"]],
            "pred_class": COCO_CLASSES[r["pred_class"]],
            "min_corrections_to_flip": d if d > 0 else None,
            "corrected_concepts": names,
            "corrected_concepts_truncated": truncated,
            "flip_confirmed_by_argmax": bool(flip_confirmed[i]) if d > 0 else None,
        })
    write_json(os.path.join(args.output_dir, "intervention_log.json"), log)

    counts = min_corr[has_flip].astype(np.float64)
    num_unflippable = int(N - has_flip.sum())

    def pct(p):
        return float(np.percentile(counts, p)) if counts.size else None

    report_ks = sorted(set(args.report_at_k))
    cdf_at_k = {str(k): float(np.mean(has_flip & (min_corr <= k))) for k in report_ks}
    summary = {
        "num_matched_instances": len(records),
        "num_subset_instances": N,
        "iou_thresh": args.iou_thresh,
        "conf_thresh": args.conf_thresh,
        "num_concepts": K,
        "num_flippable": int(has_flip.sum()),
        "num_unflippable": num_unflippable,
        "fraction_unflippable": float(num_unflippable / N),
        "min_corrections_mean": float(counts.mean()) if counts.size else None,
        "min_corrections_median": pct(50),
        "min_corrections_p25": pct(25),
        "min_corrections_p75": pct(75),
        "min_corrections_p90": pct(90),
        "min_corrections_max": float(counts.max()) if counts.size else None,
        "fraction_flippable_within_k": cdf_at_k,
        "margin_flip_not_confirmed_by_argmax": int((has_flip & ~flip_confirmed).sum()),
    }
    write_json(os.path.join(args.output_dir, "intervention_accuracy_summary.json"), summary)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    if counts.size:
        ax1.hist(counts, bins=np.arange(0.5, int(counts.max()) + 1.5, 1), color="tab:blue", edgecolor="white")
        ax1.axvline(summary["min_corrections_median"], color="tab:red", linestyle="--",
                    label=f"median = {summary['min_corrections_median']:.1f}")
        ax1.legend()
        xs = np.arange(1, K + 1)
        ax2.step(xs, [np.mean(has_flip & (min_corr <= d)) for d in xs], where="post", color="tab:blue")
        ax2.set_xlim(1, min(K, max(20, int(summary["min_corrections_p90"] or 20) + 5)))
    ax1.set_xlabel("Minimum concept corrections to flip")
    ax1.set_ylabel("# instances")
    ax1.set_title(f"Distribution ({num_unflippable}/{N} unflippable, not shown)")
    ax1.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("k (concepts corrected)")
    ax2.set_ylabel("P(min_corrections <= k)")
    ax2.set_title("Cumulative flippability")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "intervention_accuracy.png"), dpi=150)
    plt.close()

    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    table = PrettyTable(["Statistic", "Value"])
    table.add_row(["Subset size", N])
    table.add_row(["Flippable within K", f"{summary['num_flippable']} ({100 * summary['num_flippable'] / N:.1f}%)"])
    table.add_row(["Unflippable within K", f"{num_unflippable} ({100 * num_unflippable / N:.1f}%)"])
    table.add_row(["Mean min corrections", fmt(summary["min_corrections_mean"], ".2f")])
    table.add_row(["Median min corrections", fmt(summary["min_corrections_median"], ".1f")])
    table.add_row(["p25 / p75 / p90", " / ".join(fmt(summary[f"min_corrections_p{p}"], ".1f")
                                                 for p in (25, 75, 90))])
    print(table)
    cdf_table = PrettyTable(["k"] + [str(k) for k in report_ks])
    cdf_table.add_row(["P(min_corrections <= k)"] + [f"{cdf_at_k[str(k)]:.3f}" for k in report_ks])
    print(cdf_table)
    print(f"Margin flips not confirmed by the arg-max check: {summary['margin_flip_not_confirmed_by_argmax']}")


if __name__ == "__main__":
    main(make_parser().parse_args())
