#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For each successful trial of a random-search results JSON (written by
grid_search_concepts.py), evaluate the trial's checkpoint on concept
prediction and add three fields to the trial entry:

    val_concept_mAP      mean average precision over measurable concepts
    val_concept_mAUC     mean ROC-AUC over measurable concepts
    val_concept_mRecall  mean recall (probability threshold 0.5)

Each ground-truth object is matched to the highest-IoU detection of its class
(IoU >= --iou-thresh); the concept probabilities of that detection's anchor
are compared to the object's `concept_vector`. A concept is measurable when it
has at least --auc-min-pos positives (and one negative) among the matched
objects. Trials that already have the fields are skipped unless --force.
The JSON is updated after every trial.

Example (from the cbyolox/ directory):
    python tools/grid_search/compute_concept_maps_search.py \
        --results YOLOX_outputs/random_search/results.json \
        -f exps/cb_yolox/cb_yolox_s_scratch.py \
        --img-dir datasets/COCO/val2017 \
        --json-file datasets/COCO/annotations/instances_val2017.json \
        --device gpu --fp16 --limit 500

Checkpoint per trial: best_ckpt.pth, else latest_ckpt.pth, else the highest
epoch_<N>_ckpt.pth found under the trial directory.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from grid_search_concepts import save_results, trial_dir_name

# COCO category id -> contiguous class index (0..79)
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
CAT_TO_IDX = {cat_id: i for i, cat_id in enumerate(COCO_CATEGORY_IDS)}


def parse_args():
    p = argparse.ArgumentParser(description="Add concept metrics to random-search results")
    p.add_argument("--results", required=True, help="JSON written by grid_search_concepts.py")
    p.add_argument("-f", "--exp-file", required=True, help="experiment file used for the search")
    p.add_argument("--img-dir", required=True, help="validation images directory (val2017/)")
    p.add_argument("--json-file", required=True,
                   help="validation annotation JSON with `concept_vector` per annotation")
    p.add_argument("--trials-dir", default=None,
                   help="root of the trial directories (default: dir of --results)")
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="max validation images per trial")
    p.add_argument("--auc-min-pos", type=int, default=10,
                   help="min positives for a concept to be measurable")
    p.add_argument("--force", action="store_true", help="recompute existing concept metrics")
    return p.parse_args()


def find_trial_dir(trial, trials_dir):
    """Trial directory as named by grid_search_concepts.py (or by index prefix)."""
    trial_dir = Path(trials_dir) / trial_dir_name(
        trial["trial_idx"], trial["concept_loss_weight"], trial["sigma_"]
    )
    if trial_dir.is_dir():
        return trial_dir
    prefix = f"random_trial_{trial['trial_idx']:03d}_"
    for d in sorted(Path(trials_dir).iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def find_checkpoint(trial_dir):
    for name in ("best_ckpt.pth", "latest_ckpt.pth"):
        found = sorted(trial_dir.rglob(name))
        if found:
            return found[0]
    pattern = re.compile(r"epoch_(\d+)_ckpt\.pth$")
    ckpts = [(int(m.group(1)), f) for f in trial_dir.rglob("*.pth")
             if (m := pattern.match(f.name))]
    return max(ckpts)[1] if ckpts else None


def load_predictor(exp_file, ckpt_path, args):
    from yolox.data.datasets import COCO_CLASSES
    from yolox.exp import get_exp
    from yolox.utils.cbm_predictor import CBMPredictor

    exp = get_exp(str(exp_file), None)
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return CBMPredictor(model, exp, COCO_CLASSES, device=args.device, fp16=args.fp16)


def calculate_iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


def match_predictions_to_gt(gt_anns, outputs, scale_info, ratio, iou_thresh):
    """Match each GT object to the best same-class detection (original image coords)."""
    matched = []
    if outputs[0] is None:
        return matched
    preds = outputs[0].detach().cpu().float().numpy()
    preds[:, :4] /= ratio  # resized frame -> original image
    final_anchor_idxs = scale_info[0]["final_anchor_idxs"].tolist()

    for gt in gt_anns:
        if "concept_vector" not in gt:
            continue
        x, y, w, h = gt["bbox"]
        gt_xyxy = [x, y, x + w, y + h]
        gt_class = CAT_TO_IDX[gt["category_id"]]

        best_iou, best_idx = 0.0, -1
        for i, pred in enumerate(preds):
            if int(pred[6]) == gt_class:
                iou = calculate_iou(gt_xyxy, pred[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
        if best_iou >= iou_thresh:
            matched.append((gt["concept_vector"], final_anchor_idxs[best_idx]))
    return matched


def per_concept_metrics(probs, targets, auc_min_pos, threshold=0.5):
    aucs, aps, recs = [], [], []
    for k in range(probs.shape[1]):
        gt, pr = targets[:, k], probs[:, k]
        npos = int(gt.sum())
        if npos >= auc_min_pos and len(gt) - npos > 0:
            aucs.append(float(roc_auc_score(gt, pr)))
            aps.append(float(average_precision_score(gt, pr)))
            recs.append(float(recall_score(gt, (pr > threshold).astype(int), zero_division=0)))
    return aucs, aps, recs


def evaluate_checkpoint(predictor, images_meta, ann_dict, args):
    from yolox.utils.cbm_predictor import flatten_concept_outputs

    all_probs, all_targets = [], []
    for img_meta in tqdm(images_meta, desc="  evaluating", ncols=70):
        img_path = os.path.join(args.img_dir, img_meta["file_name"])
        gt_anns = ann_dict.get(img_meta["id"], [])
        if not gt_anns or not os.path.exists(img_path):
            continue

        outputs, img_info, concept_outputs, scale_info = predictor.inference(img_path)
        if concept_outputs is None or outputs[0] is None:
            continue
        concept_probs = torch.sigmoid(
            flatten_concept_outputs(concept_outputs)[0].detach().cpu().float()
        ).numpy()  # (n_anchors_all, K)

        for concept_vector, anchor_idx in match_predictions_to_gt(
            gt_anns, outputs, scale_info, img_info["ratio"], args.iou_thresh
        ):
            all_probs.append(concept_probs[anchor_idx])
            all_targets.append(np.asarray(concept_vector, dtype=np.float32))

    result = {
        "val_concept_mAP": None,
        "val_concept_mAUC": None,
        "val_concept_mRecall": None,
        "concept_matched_instances": len(all_probs),
        "concept_measurable_concepts": 0,
    }
    if not all_probs:
        print("  no matched instances, concept metrics unavailable")
        return result

    aucs, aps, recs = per_concept_metrics(
        np.stack(all_probs), np.stack(all_targets), args.auc_min_pos
    )
    if aps:
        result.update(
            val_concept_mAP=float(np.mean(aps)),
            val_concept_mAUC=float(np.mean(aucs)),
            val_concept_mRecall=float(np.mean(recs)),
            concept_measurable_concepts=len(aps),
        )
    print(f"  {result}")
    return result


def main():
    args = parse_args()
    with open(args.results) as f:
        results = json.load(f)
    trials_dir = args.trials_dir or os.path.dirname(os.path.abspath(args.results))

    with open(args.json_file) as f:
        raw = json.load(f)
    images_meta = raw["images"][:args.limit] if args.limit else raw["images"]
    ann_dict = defaultdict(list)
    for ann in raw["annotations"]:
        ann_dict[ann["image_id"]].append(ann)

    for trial in results:
        if trial.get("status") != "OK":
            continue
        idx = trial["trial_idx"]
        if not args.force and "val_concept_mAP" in trial:
            print(f"[SKIP] trial {idx}: concept metrics already present")
            continue
        print(f"\n[TRIAL {idx}] concept_loss_weight={trial['concept_loss_weight']} "
              f"sigma={trial['sigma_']}")

        trial_dir = find_trial_dir(trial, trials_dir)
        ckpt_path = find_checkpoint(trial_dir) if trial_dir else None
        if ckpt_path is None:
            print(f"[WARNING] no checkpoint for trial {idx}, skipping")
            continue
        print(f"  checkpoint: {ckpt_path}")

        predictor = load_predictor(args.exp_file, ckpt_path, args)
        trial.update(evaluate_checkpoint(predictor, images_meta, ann_dict, args))
        del predictor
        torch.cuda.empty_cache()
        save_results(results, args.results)

    print(f"\nDone. Concept metrics added to {args.results}")


if __name__ == "__main__":
    main()
