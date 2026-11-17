#!/usr/bin/env python3
"""Accuracy loss under concept deactivation (knee / half-collapse / plateau).

Faithfulness test for the sequential concept bottleneck X -g-> C -f-> Y: does
the classifier f actually depend on the concept vector C?

For every GT instance matched to a detection (IoU >= --iou_thresh, any
predicted class, per-class NMS):

1. take the anchor's concept probabilities p = sigmoid(concept_logits);
2. rank the K concepts by their influence on the GT-class logit,
   |W_s[gt] * (p - 0.5)| (W_s = concept_to_cls weights of the anchor's FPN
   level), most influential first;
3. for d = 0..max_k, set the d most influential concepts to 0.5
   (= sigmoid(0), "uninformative"), recompute the class logits with
   W_s p + b_s (exact, see cbm_eval/linear_head.py) and measure top-1
   classification accuracy.

The accuracy loss at d is acc(0) - acc(d). Three operating points are read
off the curve:
  * knee           first d with loss > --knee_thresh;
  * half_collapse  first d with loss >= 0.5 * loss(max_k);
  * plateau        first d from which the remaining drop to the end of the
                   curve is <= --plateau_tol.

Only one inference pass is needed. Accuracy is measured at the output of
concept_to_cls for the matched anchor, not after another NMS pass.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.concepts import load_concept_names  # noqa: E402
from cbm_eval.io import load_coco_annotations, write_csv, write_json  # noqa: E402
from cbm_eval.linear_head import (  # noqa: E402
    cls_logits_from_concept_probs, extract_concept_to_cls_weights, topk_hits,
)
from cbm_eval.matching import detections_to_boxes, match_gt_any_class  # noqa: E402
from cbm_eval.metrics import sigmoid  # noqa: E402
from cbm_eval.model import (  # noqa: E402
    add_model_args, anchor_info, empty_cuda_cache, load_predictor, run_inference,
)

NEUTRAL_PROB = 0.5  # sigmoid(0.0)


def make_parser():
    parser = argparse.ArgumentParser(description="CB-YOLOX accuracy loss under concept deactivation")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True,
                        help="COCO-style annotation JSON with binary concept_vector per annotation "
                             "(the one the checkpoint was validated on)")
    add_model_args(parser)
    parser.add_argument("--concepts", default=None, help="Concept-name file (labels only)")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="GT<->detection IoU threshold")
    parser.add_argument("--max_k", type=int, default=None,
                        help="End of the deactivation curve (default: K, all concepts)")
    parser.add_argument("--knee_thresh", type=float, default=0.01,
                        help="Knee = first d with accuracy loss > this (0.01 = 1 point)")
    parser.add_argument("--plateau_tol", type=float, default=0.01,
                        help="Plateau = first d from which the remaining drop is <= this")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output_dir", default="outputs/concept_deactivation")
    return parser


def deactivation_curve(concept_logits, gt_classes, scale_idx, scale_weights, max_k):
    """Top-1 accuracy after deactivating the d most influential concepts,
    d = 0..max_k. Returns (accuracy (max_k + 1,), concept order (N, K))."""
    n, k = concept_logits.shape
    probs = sigmoid(concept_logits).astype(np.float32)
    rows = np.arange(n)[:, None]

    contributions = np.zeros((n, k), dtype=np.float32)
    for s, (W, _) in enumerate(scale_weights):
        mask = scale_idx == s
        if not np.any(mask):
            continue
        contributions[mask] = W[gt_classes[mask]] * (probs[mask] - NEUTRAL_PROB)
    order = np.argsort(-np.abs(contributions), axis=1)

    accuracy = np.zeros(max_k + 1, dtype=np.float64)
    for d in range(max_k + 1):
        if d == 0:
            modified = probs
        else:
            modified = probs.copy()
            modified[rows, order[:, :d]] = NEUTRAL_PROB
        cls_logits = cls_logits_from_concept_probs(modified, scale_idx, scale_weights)
        accuracy[d] = topk_hits(cls_logits, gt_classes, ks=(1,))[1].mean()
    return accuracy, order


def find_operating_points(accuracy, knee_thresh=0.01, plateau_tol=0.01):
    """{'knee', 'half_collapse', 'plateau'} -> d (int) or None."""
    loss = accuracy[0] - accuracy
    total_loss = loss[-1]

    def first(mask):
        idx = np.flatnonzero(mask)
        return int(idx[0]) if idx.size else None

    knee = first(loss > knee_thresh)
    half = first(loss >= 0.5 * total_loss) if total_loss > 0 else None
    tail_min = np.minimum.accumulate(accuracy[::-1])[::-1]
    plateau = first(accuracy - tail_min <= plateau_tol)
    return {"knee": knee, "half_collapse": half, "plateau": plateau}


def collect_matches(args, predictor, images_meta, ann_dict):
    """Single inference pass; one record per matched GT instance."""
    records = []
    for img_meta in tqdm(images_meta, desc="Inference"):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        anns = ann_dict.get(img_meta["id"], [])
        if not anns or not os.path.exists(img_path):
            continue
        outputs, img_info, concept_logits, scale_info = run_inference(
            predictor, img_path, class_agnostic=False)
        boxes, classes, scores = detections_to_boxes(outputs, img_info)
        if boxes is not None and concept_logits is not None:
            anchor_idxs, scale_map = anchor_info(scale_info)
            for m in match_gt_any_class(anns, boxes, classes, scores, anchor_idxs, scale_map,
                                        args.iou_thresh):
                m["image_id"] = img_meta["id"]
                m["logits"] = concept_logits[m["anchor_idx"]].copy()
                records.append(m)
        empty_cuda_cache(args.device)
    return records


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prettytable import PrettyTable

    os.makedirs(args.output_dir, exist_ok=True)
    predictor, model, _ = load_predictor(args)
    K = predictor.num_concepts
    max_k = K if args.max_k is None else max(1, min(args.max_k, K))
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

    logits = np.stack([r["logits"] for r in records])
    gt_classes = np.array([r["gt_class"] for r in records], dtype=np.int64)
    scale_idx = np.array([r["scale_idx"] for r in records], dtype=np.int64)

    accuracy, order = deactivation_curve(logits, gt_classes, scale_idx, scale_weights, max_k)
    baseline = accuracy[0]
    points = find_operating_points(accuracy, args.knee_thresh, args.plateau_tol)
    total_loss = float(baseline - accuracy[-1])

    write_csv(os.path.join(args.output_dir, "deactivation_curve.csv"),
              ["num_deactivated", "accuracy", "accuracy_loss"],
              [[d, accuracy[d], baseline - accuracy[d]] for d in range(max_k + 1)])

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(max_k + 1), accuracy, marker="o", markersize=3, color="tab:blue")
    for (name, d), color in zip(points.items(), ("tab:red", "tab:orange", "tab:green")):
        if d is None:
            continue
        plt.scatter([d], [accuracy[d]], color=color, zorder=3)
        plt.axvline(d, color=color, linestyle="--", alpha=0.5, label=f"{name.replace('_', '-')} (d={d})")
    plt.legend()
    plt.xlabel("Number of most influential concepts deactivated")
    plt.ylabel("Classification accuracy (top-1)")
    plt.title("Accuracy loss under concept deactivation")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "deactivation_curve.png"), dpi=150)
    plt.close()

    # How often each concept is the most influential one of a matched instance.
    top1_counts = np.bincount(order[:, 0], minlength=K)
    write_csv(os.path.join(args.output_dir, "most_influential_concept_counts.csv"),
              ["concept", "count"],
              [[concept_names[i], int(top1_counts[i])] for i in np.argsort(-top1_counts)])

    summary = {
        "num_matched_instances": len(records),
        "num_concepts": K,
        "iou_thresh": args.iou_thresh,
        "baseline_accuracy": float(baseline),
        "max_deactivated": max_k,
        "total_accuracy_loss": total_loss,
        "knee_thresh": args.knee_thresh,
        "plateau_tol": args.plateau_tol,
    }
    for name, d in points.items():
        summary[name] = None if d is None else {
            "num_concepts_deactivated": d,
            "accuracy": float(accuracy[d]),
            "accuracy_loss": float(baseline - accuracy[d]),
        }
    write_json(os.path.join(args.output_dir, "deactivation_summary.json"), summary)

    table = PrettyTable(["Operating point", "# concepts deactivated", "Accuracy", "Accuracy loss"])
    for name, d in points.items():
        label = name.replace("_", "-").capitalize()
        if d is None:
            table.add_row([label, "n/a", "n/a", "n/a"])
        else:
            table.add_row([label, d, f"{accuracy[d]:.4f}", f"{baseline - accuracy[d]:+.4f}"])
    print(f"\nBaseline accuracy: {baseline:.4f}  accuracy at d={max_k}: {accuracy[-1]:.4f}  "
          f"total loss: {total_loss:.4f}")
    print(table)


if __name__ == "__main__":
    main(make_parser().parse_args())
