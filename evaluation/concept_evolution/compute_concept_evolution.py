#!/usr/bin/env python3
"""Concept-prediction quality across training epochs.

Sweeps every `epoch_<N>_ckpt.pth` in --ckpt_dir. For each checkpoint: run
inference on the validation images, match every GT instance to its best
same-class detection (IoU >= --iou_thresh), take sigmoid(concept_logits) of
that detection's anchor and compute per-concept ROC-AUC, AP and Recall@0.5
against the binary GT concept vector (concepts with fewer than --auc_min_pos
positives are reported as null for that epoch).

Output JSON (input of plot_concept_evolution.py):
    {"concept_names": [...], "epochs": [...],
     "metrics": [{"auc": [K], "ap": [K], "recall": [K], "n_pos": [K]}, ...],
     "match_counts": [...]}
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.concepts import read_concept_names  # noqa: E402
from cbm_eval.io import load_coco_annotations, write_json  # noqa: E402
from cbm_eval.matching import detections_to_boxes, match_gt_same_class  # noqa: E402
from cbm_eval.metrics import mean_of_valid, per_concept_metrics, sigmoid_torch  # noqa: E402
from cbm_eval.model import add_model_args, anchor_info, empty_cuda_cache, load_predictor, run_inference  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(description="Per-concept AUC/AP/recall over training checkpoints")
    parser.add_argument("--ckpt_dir", required=True, help="Directory with epoch_<N>_ckpt.pth files")
    parser.add_argument("--img_dir", required=True, help="Validation image directory")
    parser.add_argument("--json_binary", required=True,
                        help="COCO-style annotation JSON with binary concept_vector per annotation")
    parser.add_argument("--concepts", required=True, help="Concept-name file (defines K and labels)")
    add_model_args(parser, conf_default=0.25, nms_default=0.45, ckpt=False)
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="GT<->detection IoU threshold")
    parser.add_argument("--auc_min_pos", type=int, default=10,
                        help="Minimum positive matched instances for a concept to be scored")
    parser.add_argument("--legacy_box_scale", action="store_true",
                        help="Match GT boxes against network-space (not rescaled) detection boxes, "
                             "reproducing the original implementation (see README)")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images")
    parser.add_argument("--output", default="outputs/concept_evolution.json", help="Output JSON")
    return parser


def discover_checkpoints(ckpt_dir):
    pattern = re.compile(r"epoch_(\d+)_ckpt\.pth$")
    ckpts = sorted((int(m.group(1)), f) for f in Path(ckpt_dir).glob("*.pth")
                   if (m := pattern.match(f.name)))
    if not ckpts:
        sys.exit(f"No epoch_<N>_ckpt.pth files found in {ckpt_dir}")
    print(f"Found {len(ckpts)} checkpoints: epochs {ckpts[0][0]} -> {ckpts[-1][0]}")
    return ckpts


def evaluate_checkpoint(args, ckpt_path, images_meta, ann_dict):
    """Returns (probs (N, K), targets (N, K)) of the matched instances."""
    predictor, _, _ = load_predictor(args, ckpt_path=str(ckpt_path), require_concepts=True)
    all_probs, all_targets = [], []
    for img_meta in tqdm(images_meta, desc=ckpt_path.name, ncols=80):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        anns = ann_dict.get(img_meta["id"], [])
        if not anns or not os.path.exists(img_path):
            continue
        outputs, img_info, concept_logits, scale_info = run_inference(predictor, img_path, class_agnostic=True)
        if concept_logits is None or outputs[0] is None:
            continue
        probs = sigmoid_torch(concept_logits)
        boxes, classes, _ = detections_to_boxes(outputs, img_info, rescale=not args.legacy_box_scale)
        anchor_idxs, _ = anchor_info(scale_info)
        for m in match_gt_same_class(anns, boxes, classes, anchor_idxs, args.iou_thresh):
            all_probs.append(probs[m["anchor_idx"]].copy())
            all_targets.append(np.array(m["concept_vector"], dtype=np.float32))
        empty_cuda_cache(args.device)
    del predictor
    empty_cuda_cache(args.device)
    return all_probs, all_targets


def main(args):
    concept_names = read_concept_names(args.concepts)
    print(f"{len(concept_names)} concepts")
    ckpts = discover_checkpoints(args.ckpt_dir)

    images_meta, ann_dict = load_coco_annotations(args.json_binary)
    if args.limit:
        images_meta = images_meta[:args.limit]

    results = {"concept_names": concept_names, "epochs": [], "metrics": [], "match_counts": []}
    for epoch, ckpt_path in ckpts:
        print(f"\n=== epoch {epoch}: {ckpt_path.name} ===")
        probs, targets = evaluate_checkpoint(args, ckpt_path, images_meta, ann_dict)
        print(f"Matched instances: {len(probs)}")
        if not probs:
            print("[WARN] No matched instances - skipping epoch.")
            continue
        metrics = per_concept_metrics(np.array(probs, dtype=np.float32),
                                      np.array(targets, dtype=np.float32), args.auc_min_pos)
        mean_auc, n_valid = mean_of_valid(metrics["auc"])
        print(f"Mean AUC {mean_auc:.4f}  mean AP {mean_of_valid(metrics['ap'])[0]:.4f}  "
              f"mean recall {mean_of_valid(metrics['recall'])[0]:.4f}  "
              f"measurable {n_valid}/{len(concept_names)}")
        results["epochs"].append(epoch)
        results["metrics"].append(metrics)
        results["match_counts"].append(len(probs))

    write_json(args.output, results, indent=None)


if __name__ == "__main__":
    main(make_parser().parse_args())
