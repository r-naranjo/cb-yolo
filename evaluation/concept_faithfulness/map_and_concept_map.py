#!/usr/bin/env python3
"""Detection mAP + concept mAP of one checkpoint, in one inference pass.

Detection mAP: pycocotools bbox evaluation with --json_binary as GT, boxes
mapped back to original-image pixels and per-class NMS (as the official YOLOX
evaluator). Keep --conf/--nms at the experiment defaults (exp.test_conf,
exp.nmsthre) to obtain numbers comparable with training-time validation.

Concept mAP: for every GT instance whose best same-class detection has
IoU >= --iou_thresh, compare sigmoid(concept_logits) of that anchor with the
GT binary concept vector. Per concept (if it has >= --auc_min_pos positives):
ROC-AUC, Average Precision and Recall@0.5; concept mAP = mean AP over the
measurable concepts.
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
from cbm_eval.matching import detections_to_boxes, detections_to_coco, match_gt_same_class  # noqa: E402
from cbm_eval.metrics import (  # noqa: E402
    COCO_STAT_NAMES, mean_of_valid, per_concept_metrics, run_coco_eval, sigmoid,
)
from cbm_eval.model import add_model_args, anchor_info, empty_cuda_cache, load_predictor, run_inference  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(description="CB-YOLOX detection mAP + concept mAP")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True,
                        help="COCO-style annotation JSON with binary concept_vector per annotation; "
                             "also the box/category GT for COCOeval")
    add_model_args(parser)
    parser.add_argument("--concepts", default=None, help="Concept-name file (labels only)")
    parser.add_argument("--iou_thresh", type=float, default=0.5,
                        help="Same-class GT<->detection IoU threshold (concept mAP only)")
    parser.add_argument("--auc_min_pos", type=int, default=10,
                        help="Minimum positive matched instances for a concept to be scored")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output_dir", default="outputs/map_and_concept_map")
    return parser


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prettytable import PrettyTable

    os.makedirs(args.output_dir, exist_ok=True)
    predictor, _, _ = load_predictor(args)
    K = predictor.num_concepts
    concept_names = load_concept_names(args.concepts, K)

    images_meta, ann_dict = load_coco_annotations(args.json_binary)
    if args.limit:
        images_meta = images_meta[:args.limit]

    coco_results, all_probs, all_targets = [], [], []
    for img_meta in tqdm(images_meta, desc="Inference"):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        anns = ann_dict.get(img_meta["id"], [])
        if not anns or not os.path.exists(img_path):
            continue
        outputs, img_info, concept_logits, scale_info = run_inference(
            predictor, img_path, class_agnostic=False)
        boxes, classes, scores = detections_to_boxes(outputs, img_info)
        coco_results.extend(detections_to_coco(boxes, classes, scores, img_meta["id"]))
        if boxes is not None and concept_logits is not None:
            anchor_idxs, _ = anchor_info(scale_info)
            for m in match_gt_same_class(anns, boxes, classes, anchor_idxs, args.iou_thresh):
                all_probs.append(sigmoid(concept_logits[m["anchor_idx"]]).copy())
                all_targets.append(np.asarray(m["concept_vector"], dtype=np.float32))
        empty_cuda_cache(args.device)

    coco_stats = run_coco_eval(args.json_binary, coco_results)

    print(f"\nSame-class GT<->detection matches (IoU >= {args.iou_thresh}): {len(all_probs)}")
    concept_summary = None
    if all_probs:
        metrics = per_concept_metrics(np.array(all_probs, dtype=np.float32),
                                      np.array(all_targets, dtype=np.float32), args.auc_min_pos)

        def cell(v):
            return "" if v is None else v

        write_csv(os.path.join(args.output_dir, "per_concept_metrics.csv"),
                  ["concept", "auc", "ap", "recall", "n_pos"],
                  [[name, cell(metrics["auc"][i]), cell(metrics["ap"][i]), cell(metrics["recall"][i]),
                    metrics["n_pos"][i]] for i, name in enumerate(concept_names)])

        concept_map, n_measurable = mean_of_valid(metrics["ap"])
        concept_summary = {
            "num_matched_instances": len(all_probs),
            "num_concepts": K,
            "num_concepts_measurable": n_measurable,
            "auc_min_pos": args.auc_min_pos,
            "concept_mAP": concept_map,
            "concept_mean_auc": mean_of_valid(metrics["auc"])[0],
            "concept_mean_recall": mean_of_valid(metrics["recall"])[0],
        }

        valid_aps = [v for v in metrics["ap"] if v is not None]
        if valid_aps:
            plt.figure(figsize=(7, 5))
            plt.hist(valid_aps, bins=20, color="tab:blue", edgecolor="white")
            plt.axvline(concept_map, color="tab:red", linestyle="--", label=f"mean AP = {concept_map:.4f}")
            plt.xlabel("Per-concept Average Precision")
            plt.ylabel("# concepts")
            plt.title(f"Concept AP distribution ({n_measurable}/{K} concepts measurable)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "concept_ap_distribution.png"), dpi=150)
            plt.close()
    else:
        print("No same-class matches - concept mAP not computed.")

    write_json(os.path.join(args.output_dir, "map_report_summary.json"), {
        "iou_thresh": args.iou_thresh,
        "coco_stat_names": COCO_STAT_NAMES,
        "coco_stats": coco_stats.tolist() if coco_stats is not None else None,
        "concept_metrics": concept_summary,
    })

    if coco_stats is not None:
        table = PrettyTable(["Metric", "Value"])
        for name, val in zip(COCO_STAT_NAMES, coco_stats):
            table.add_row([name, f"{val:.4f}"])
        print(table)
    if concept_summary is not None:
        table = PrettyTable(["Concept metric", "Value"])
        table.add_row(["Concept mAP (mean AP)", f"{concept_summary['concept_mAP']:.4f}"])
        table.add_row(["Concept mean AUC", f"{concept_summary['concept_mean_auc']:.4f}"])
        table.add_row(["Concept mean Recall@0.5", f"{concept_summary['concept_mean_recall']:.4f}"])
        table.add_row(["Concepts measurable", f"{n_measurable}/{K}"])
        print(table)


if __name__ == "__main__":
    main(make_parser().parse_args())
