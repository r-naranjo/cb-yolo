#!/usr/bin/env python3
"""Single-checkpoint report for the per-class concept vocabulary (80 x 5 = 400).

Needs two index-aligned annotation JSONs of the same split:
  --json_binary  concept_vector = 0/1 over the 400 concepts
  --json_ranked  concept_vector = rank (1..5) of the 5 most salient concepts,
                 0 elsewhere

Reports:
  1. COCO bbox metrics (pycocotools) + GT vs. predicted instances per class;
  2. binary concept metrics (metrics_binary.py);
  3. ranked top-5 concept metrics (metrics_ranked.py);
  4. concept frequency analysis (metrics_distribution.py).

Concept predictions come from the anchor of the best same-class detection of
each GT box (IoU >= --iou_thresh); unmatched GT boxes are skipped. Inference
uses CBMPredictor.inference (class-agnostic NMS), --conf 0.25 / --nms 0.45.
"""

import argparse
import os
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from cbm_eval.coco import COCO_CLASSES  # noqa: E402
from cbm_eval.io import AnsiStrippingFile, load_coco_annotations, write_json  # noqa: E402
from cbm_eval.matching import detections_to_boxes, detections_to_coco, match_gt_same_class  # noqa: E402
from cbm_eval.metrics import COCO_STAT_NAMES, run_coco_eval, sigmoid_torch  # noqa: E402
from cbm_eval.model import add_model_args, anchor_info, empty_cuda_cache, load_predictor, run_inference  # noqa: E402

from metrics_binary import evaluate_binary_concepts, plot_binary_metrics, plot_threshold_curves  # noqa: E402
from metrics_distribution import analyze_concept_distributions, plot_class_instance_counts  # noqa: E402
from metrics_ranked import evaluate_ranked_concepts, plot_ranked_metrics  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(description="CB-YOLOX per-class (400-concept) evaluation report")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True, help="Binary concept annotation JSON")
    parser.add_argument("--json_ranked", required=True, help="Ranked (top-5) concept annotation JSON")
    parser.add_argument("--concepts", required=True,
                        help="Per-class concept names: per_class_descriptors_top5.json or per_class_400_list.json")
    add_model_args(parser, conf_default=0.25, nms_default=0.45)
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="GT<->detection IoU threshold")
    parser.add_argument("--legacy_box_scale", action="store_true",
                        help="Use network-space (not rescaled) detection boxes for matching and COCOeval, "
                             "reproducing the original implementation (see README)")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output_dir", default="outputs/per_class_concept_metrics")
    parser.add_argument("--log_file", default=None,
                        help="Optional file for the verbose per-detection inference report")
    return parser


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    predictor, _, _ = load_predictor(args)

    images_meta, ann_dict_bin = load_coco_annotations(args.json_binary)
    _, ann_dict_rank = load_coco_annotations(args.json_ranked)
    if args.limit:
        images_meta = images_meta[:args.limit]

    log_stream = AnsiStrippingFile(args.log_file) if args.log_file else None
    classes_all, probs_all, gt_bin_all, gt_rank_all, coco_results = [], [], [], [], []

    for img_meta in tqdm(images_meta, desc="Inference"):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        if not os.path.exists(img_path):
            continue
        anns_bin = ann_dict_bin.get(img_meta["id"], [])
        anns_rank = ann_dict_rank.get(img_meta["id"], [])
        if not anns_bin or not anns_rank:
            continue
        pairs = list(zip(anns_bin, anns_rank))
        for a_bin, a_rank in pairs:
            assert a_bin["id"] == a_rank["id"], "Annotation ids differ between binary and ranked JSONs"

        with redirect_stdout(log_stream) if log_stream else nullcontext():
            if log_stream:
                print(f"\n{'=' * 50}\nImage: {img_meta['file_name']}\n{'=' * 50}")
            outputs, img_info, concept_logits, scale_info = run_inference(
                predictor, img_path, class_agnostic=True, verbose=log_stream is not None)

        boxes, classes, scores = detections_to_boxes(outputs, img_info, rescale=not args.legacy_box_scale)
        coco_results.extend(detections_to_coco(boxes, classes, scores, img_meta["id"]))

        if concept_logits is not None and boxes is not None:
            probs = sigmoid_torch(concept_logits)
            anchor_idxs, _ = anchor_info(scale_info)
            for m in match_gt_same_class([b for b, _ in pairs], boxes, classes, anchor_idxs, args.iou_thresh):
                probs_all.append(probs[m["anchor_idx"]].copy())
                gt_bin_all.append(m["concept_vector"])
                gt_rank_all.append(pairs[m["gt_index"]][1]["concept_vector"])
                classes_all.append(m["class_id"])
        empty_cuda_cache(args.device)

    if log_stream:
        log_stream.close()

    probs_all = np.array(probs_all)
    gt_bin_all = np.array(gt_bin_all)
    gt_rank_all = np.array(gt_rank_all)
    classes_all = np.array(classes_all)
    print(f"\nMatched instances: {len(probs_all)}")
    if len(probs_all) == 0:
        print("No GT instance matched a detection - check --iou_thresh, paths or the checkpoint.")
        return

    summary = {"num_matched_instances": int(len(probs_all)), "iou_thresh": args.iou_thresh,
               "legacy_box_scale": args.legacy_box_scale, "coco_stat_names": COCO_STAT_NAMES}
    if coco_results:
        stats = run_coco_eval(args.json_binary, coco_results)
        summary["coco_stats"] = stats.tolist()
        plot_class_instance_counts(args.json_binary, coco_results, COCO_CLASSES, args.output_dir, top_k=80)
    else:
        print("The model produced no detections.")

    print("\n" + "=" * 50 + "\nBinary ground truth\n" + "=" * 50)
    binary = evaluate_binary_concepts(probs_all, gt_bin_all, classes_all, threshold=0.5)
    plot_threshold_curves(probs_all, gt_bin_all, args.output_dir)
    plot_binary_metrics(binary, args.output_dir)

    print("\n" + "=" * 50 + "\nRanked ground truth\n" + "=" * 50)
    ranked = evaluate_ranked_concepts(probs_all, gt_rank_all, classes_all)
    plot_ranked_metrics(ranked, args.output_dir)

    print("\n" + "=" * 50 + "\nConcept frequency distribution\n" + "=" * 50)
    gt_freq, pred_freq = analyze_concept_distributions(probs_all, gt_bin_all, classes_all, args.concepts,
                                                       threshold=0.5, save_dir=args.output_dir)

    summary["binary_global"] = {k: float(v) for k, v in binary["global"].items()}
    summary["ranked_global"] = {k: float(v) for k, v in ranked["global"].items()}
    summary["frequency_pearson_r"] = float(pearsonr(gt_freq, pred_freq)[0])
    write_json(os.path.join(args.output_dir, "summary.json"), summary)


if __name__ == "__main__":
    main(make_parser().parse_args())
