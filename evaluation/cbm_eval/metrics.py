"""Metric helpers shared by several scripts."""

import json
import os
import tempfile

import numpy as np

COCO_STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl",
                   "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


def sigmoid(x):
    """NumPy sigmoid (keeps the input dtype)."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_torch(x):
    """torch.sigmoid on a float32 NumPy array (bit-identical to applying
    torch.sigmoid to the original float32 tensor)."""
    import torch
    return torch.sigmoid(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))).numpy()


def per_concept_metrics(probs, targets, auc_min_pos, threshold=0.5):
    """Per-concept ROC-AUC, Average Precision and Recall@threshold.

    probs, targets: (N, K) arrays. A concept is scored only if it has at least
    `auc_min_pos` positives and at least one negative; otherwise its entries
    are None. "Concept mAP" is the mean of the non-None APs.
    Returns {'auc': [K], 'ap': [K], 'recall': [K], 'n_pos': [K]}.
    """
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

    num_concepts = probs.shape[1]
    aucs, aps, recs, n_pos = [], [], [], []
    for k in range(num_concepts):
        gt = targets[:, k]
        pr = probs[:, k]
        pr_bin = (pr > threshold).astype(int)
        npos = int(gt.sum())
        nneg = int(len(gt) - npos)
        n_pos.append(npos)
        if npos >= auc_min_pos and nneg > 0:
            try:
                aucs.append(float(roc_auc_score(gt, pr)))
                aps.append(float(average_precision_score(gt, pr)))
                recs.append(float(recall_score(gt, pr_bin, zero_division=0)))
            except Exception:  # e.g. NaN probabilities; concept counted as unmeasurable
                aucs.append(None)
                aps.append(None)
                recs.append(None)
        else:
            aucs.append(None)
            aps.append(None)
            recs.append(None)
    return {"auc": aucs, "ap": aps, "recall": recs, "n_pos": n_pos}


def mean_of_valid(values):
    valid = [v for v in values if v is not None]
    return (float(np.mean(valid)) if valid else float("nan")), len(valid)


def run_coco_eval(gt_json_path, coco_results, label=""):
    """Standard pycocotools bbox evaluation. Returns the 12 COCOeval stats
    (see COCO_STAT_NAMES) or None if there are no detections."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    print(f"\n{'=' * 60}\nCOCO bbox evaluation{(' - ' + label) if label else ''}\n{'=' * 60}")
    if not coco_results:
        print("No detections to evaluate.")
        return None
    coco_gt = COCO(gt_json_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(coco_results, tmp)
        tmp_path = tmp.name
    try:
        coco_dt = coco_gt.loadRes(tmp_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    finally:
        os.remove(tmp_path)
    return coco_eval.stats
