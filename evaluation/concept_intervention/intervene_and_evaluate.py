#!/usr/bin/env python3
"""Concept-level error analysis and mass test-time intervention.

0. Class x concept activation matrix: mean sigmoid(concept_logits) of the
   matched anchor, per GT class (GT instances matched to their best-IoU
   detection of any class, IoU >= --iou_thresh).
1. Count well-localized but misclassified instances.
2. Of those, count instances whose top --top_k concepts (by logit) are
   disjoint from the GT-active concepts (concept_vector > 0).
3. For each of them build an intervention: set the top-k wrong concepts to
   --deactivate_value (raw logit, 0.0 = sigmoid 0.5) and give the top-k
   concepts most characteristic of the GT class (row of the step-0 matrix,
   excluding the wrong ones) the logit values the wrong concepts had.
4. Run a second inference pass with all interventions applied through
   head.set_interventions() and compare COCO bbox mAP before vs. after.

Inference uses CBMPredictor.inference (class-agnostic NMS) with
--conf 0.25 / --nms 0.45 by default; boxes are mapped back to original-image
pixels before matching and COCO evaluation.
"""

import argparse
import os
import sys
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import COCO_CLASSES  # noqa: E402
from cbm_eval.concepts import load_concept_names  # noqa: E402
from cbm_eval.io import AnsiStrippingFile, load_coco_annotations, write_csv, write_json  # noqa: E402
from cbm_eval.matching import detections_to_boxes, detections_to_coco, match_gt_any_class  # noqa: E402
from cbm_eval.metrics import COCO_STAT_NAMES, run_coco_eval, sigmoid  # noqa: E402
from cbm_eval.model import add_model_args, anchor_info, empty_cuda_cache, load_predictor, run_inference  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(
        description="CB-YOLOX concept-intervention diagnosis and mass test-time correction")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True,
                        help="COCO-style annotation JSON with binary concept_vector per annotation; "
                             "also the box/category GT for COCOeval")
    add_model_args(parser, conf_default=0.25, nms_default=0.45)
    parser.add_argument("--concepts", default=None, help="Concept-name file (labels only)")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="GT<->detection IoU threshold")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top concepts to inspect/replace")
    parser.add_argument("--deactivate_value", type=float, default=0.0,
                        help="Raw concept logit written into deactivated concepts")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output_dir", default="outputs/concept_intervention")
    parser.add_argument("--log_file", default=None,
                        help="Optional file for the verbose per-detection inference report")
    return parser


def build_interventions(records, concept_class_matrix, top_k, deactivate_value, concept_names):
    """Steps 1-3. Returns (n_misclassified, n_disjoint, {image_id: [[anchor, concepts, values]]}, log)."""
    misclassified = [r for r in records if r["pred_class"] != r["gt_class"]]
    disjoint = []
    for r in misclassified:
        top_wrong = np.argsort(-r["logits"])[:top_k]
        gt_active = set(np.where(r["gt_concepts_bin"] > 0)[0].tolist())
        if not set(top_wrong.tolist()) & gt_active:
            r["top_wrong"] = top_wrong
            disjoint.append(r)

    by_image = defaultdict(list)
    log = []
    for r in disjoint:
        class_row = np.nan_to_num(concept_class_matrix[r["gt_class"]], nan=-np.inf)
        wrong = set(r["top_wrong"].tolist())
        replacement = [int(c) for c in np.argsort(-class_row) if c not in wrong][:top_k]
        if len(replacement) < top_k:
            continue
        wrong_values = r["logits"][r["top_wrong"]].tolist()
        concepts = r["top_wrong"].tolist() + replacement
        values = [deactivate_value] * top_k + wrong_values
        by_image[r["image_id"]].append([r["anchor_idx"], concepts, values])
        log.append({
            "image_id": int(r["image_id"]),
            "anchor_idx": r["anchor_idx"],
            "gt_class": COCO_CLASSES[r["gt_class"]],
            "pred_class": COCO_CLASSES[r["pred_class"]],
            "iou": r["iou"],
            "deactivated_concepts": [concept_names[c] for c in r["top_wrong"].tolist()],
            "activated_concepts": [concept_names[c] for c in replacement],
            "transferred_logit_values": wrong_values,
        })
    return len(misclassified), len(disjoint), by_image, log


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from prettytable import PrettyTable

    os.makedirs(args.output_dir, exist_ok=True)
    predictor, _, _ = load_predictor(args)
    head = predictor.model.head
    K = predictor.num_concepts
    num_classes = len(COCO_CLASSES)
    concept_names = load_concept_names(args.concepts, K)

    images_meta, ann_dict = load_coco_annotations(args.json_binary)
    if args.limit:
        images_meta = images_meta[:args.limit]
    images_meta = [m for m in images_meta
                   if ann_dict.get(m["id"]) and os.path.exists(os.path.join(args.img_dir, m["file_name"]))]

    log_stream = AnsiStrippingFile(args.log_file) if args.log_file else None
    verbose = log_stream is not None

    def infer(img_meta, tag):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        with redirect_stdout(log_stream) if verbose else nullcontext():
            if verbose:
                print(f"\n{'=' * 50}\n[{tag}] {img_meta['file_name']}\n{'=' * 50}")
            return run_inference(predictor, img_path, class_agnostic=True, verbose=verbose)

    # Pass 1: baseline detections + matched records.
    records, coco_before = [], []
    for img_meta in tqdm(images_meta, desc="Pass 1/2 (baseline)"):
        outputs, img_info, concept_logits, scale_info = infer(img_meta, "PASS 1")
        boxes, classes, scores = detections_to_boxes(outputs, img_info)
        coco_before.extend(detections_to_coco(boxes, classes, scores, img_meta["id"]))
        if boxes is not None and concept_logits is not None:
            anchor_idxs, scale_map = anchor_info(scale_info)
            for m in match_gt_any_class(ann_dict[img_meta["id"]], boxes, classes, scores,
                                        anchor_idxs, scale_map, args.iou_thresh):
                m["image_id"] = img_meta["id"]
                m["logits"] = concept_logits[m["anchor_idx"]].copy()
                records.append(m)
        empty_cuda_cache(args.device)

    print(f"\nGT instances matched at IoU >= {args.iou_thresh} (any predicted class): {len(records)}")
    if not records:
        print("No matched instances - check --iou_thresh, paths or the checkpoint.")
        return
    assert len(records[0]["gt_concepts_bin"]) == K, (
        f"GT concept vector length ({len(records[0]['gt_concepts_bin'])}) != model num_concepts ({K})")

    # Step 0: class x concept activation matrix.
    sum_probs = np.zeros((num_classes, K), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for r in records:
        sum_probs[r["gt_class"]] += sigmoid(r["logits"])
        counts[r["gt_class"]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        concept_class_matrix = sum_probs / counts[:, None]

    write_csv(os.path.join(args.output_dir, "concept_class_matrix.csv"), ["class"] + concept_names,
              [[cls] + row.tolist() for cls, row in zip(COCO_CLASSES, concept_class_matrix)])
    plt.figure(figsize=(min(max(12, K * 0.25), 60), min(max(10, num_classes * 0.22), 40)))
    sns.heatmap(concept_class_matrix, cmap="viridis", vmin=0, vmax=1,
                xticklabels=concept_names, yticklabels=list(COCO_CLASSES),
                annot=(K <= 40), fmt=".2f", cbar_kws={"label": "mean sigmoid activation"})
    plt.title("Mean concept activation per class (validation set)")
    plt.xlabel("Concept")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "concept_class_matrix_heatmap.png"), dpi=150)
    plt.close()

    # Steps 1-3.
    n_misclassified, n_disjoint, interventions, intervention_log = build_interventions(
        records, concept_class_matrix, args.top_k, args.deactivate_value, concept_names)
    write_json(os.path.join(args.output_dir, "intervention_log.json"), intervention_log)

    # Pass 2: same images with the interventions applied.
    coco_after = []
    for img_meta in tqdm(images_meta, desc="Pass 2/2 (intervened)"):
        image_interventions = interventions.get(img_meta["id"])
        if image_interventions:
            head.set_interventions(image_interventions)
        try:
            outputs, img_info, _, _ = infer(img_meta, "PASS 2")
        finally:
            head.clear_interventions()
        coco_after.extend(detections_to_coco(*detections_to_boxes(outputs, img_info), img_meta["id"]))
        empty_cuda_cache(args.device)
    if log_stream is not None:
        log_stream.close()

    # Step 4: mAP before vs. after.
    stats_before = run_coco_eval(args.json_binary, coco_before, label="before intervention")
    stats_after = run_coco_eval(args.json_binary, coco_after, label="after intervention")

    summary = {
        "num_matched_instances": len(records),
        "iou_thresh": args.iou_thresh,
        "top_k": args.top_k,
        "step1_misclassified_despite_iou": n_misclassified,
        "step2_top_k_concepts_disjoint_from_gt": n_disjoint,
        "num_interventions_applied": len(intervention_log),
        "num_images_intervened": len(interventions),
    }
    table = PrettyTable(["Step", "Count"])
    table.add_row(["GT instances matched (IoU >= thresh, any class)", len(records)])
    table.add_row(["  ...misclassified despite good IoU", n_misclassified])
    table.add_row([f"  ...of which top-{args.top_k} concepts disjoint from GT concepts", n_disjoint])
    table.add_row(["Instances receiving an intervention", len(intervention_log)])
    table.add_row(["Images with at least one intervention", len(interventions)])
    print(table)

    if stats_before is not None and stats_after is not None:
        summary["coco_stat_names"] = COCO_STAT_NAMES
        summary["coco_stats_before"] = stats_before.tolist()
        summary["coco_stats_after"] = stats_after.tolist()
        table = PrettyTable(["Metric", "Before", "After", "Delta"])
        for name, b, a in zip(COCO_STAT_NAMES, stats_before, stats_after):
            table.add_row([name, f"{b:.4f}", f"{a:.4f}", f"{a - b:+.4f}"])
        print(table)
    write_json(os.path.join(args.output_dir, "intervention_summary.json"), summary)


if __name__ == "__main__":
    main(make_parser().parse_args())
