"""Binarize the raw CLIP concept vectors written by ``main_coco.py --process label_instances``.

Pipeline (class statistics are computed once, on the raw input vectors):

1. Class statistics: per class, the mean activation of every concept
   (ignoring -1 entries) and the 3 concepts with the highest mean (top-3).
2. Exactly 5 positives per instance (instances containing -1 are skipped):
   more than 5 values > 0 -> keep the 5 largest; fewer than 5 -> fill zero
   entries with the class mean, in decreasing class-mean order, until 5.
3. Binarization: instances containing -1 (Rule 3) -> class top-3 set to 1;
   otherwise concept i is 1 iff activation[i] > class_mean[i].
4. Instances whose binary vector is all zeros get their class top-3 set to 1.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def to_python(obj):
    """Recursively convert NumPy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    return obj


def compute_class_means_and_top3(annotations):
    class_vecs = defaultdict(list)
    for ann in annotations:
        class_vecs[ann["category_id"]].append(np.array(ann["concept_vector"], dtype=float))

    class_means, class_top3 = {}, {}
    for cid, vecs in class_vecs.items():
        stacked = np.vstack(vecs)
        mask = (stacked != -1)
        sums = np.where(mask, stacked, 0.0).sum(axis=0)
        counts = mask.sum(axis=0)
        means = np.divide(sums, np.where(counts == 0, 1, counts))
        means[counts == 0] = 0.0
        class_means[cid] = means
        class_top3[cid] = list(np.argsort(-means)[:3])
    return class_means, class_top3


def enforce_five_nonzero(vec, class_mean):
    """Return ``(vector, changed)`` with exactly 5 positive entries (see module docstring)."""
    arr = vec.copy()
    if np.any(arr == -1):
        return arr, False

    pos_mask = arr > 0
    n_pos = pos_mask.sum()
    changed = False
    if n_pos > 5:
        pos_idx = np.where(pos_mask)[0]
        keep = set(pos_idx[np.argsort(-arr[pos_idx])[:5]])
        for i in pos_idx:
            if i not in keep:
                arr[i] = 0.0
                changed = True
    elif n_pos < 5:
        for idx in np.argsort(-class_mean):
            if n_pos >= 5:
                break
            if arr[idx] == 0.0:
                arr[idx] = float(class_mean[idx])
                n_pos += 1
                changed = True
    return arr, changed


def binarize_instance(vec, class_mean, top3):
    if np.any(vec == -1):
        binvec = np.zeros_like(vec, dtype=int)
        for i in top3:
            binvec[i] = 1
        return binvec
    return (vec > class_mean).astype(int)


def fix_all_zero_binary(vec, top3):
    arr = vec.copy()
    if arr.sum() == 0:
        for i in top3:
            arr[i] = 1
    return arr


def binarize(data):
    """Binarize ``data['annotations']`` in place; return a summary dict."""
    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError("No annotations in file.")
    vec_len = len(annotations[0]["concept_vector"])
    class_means, class_top3 = compute_class_means_and_top3(annotations)

    enforce_count = 0
    for ann in annotations:
        updated, changed = enforce_five_nonzero(np.array(ann["concept_vector"], dtype=float), class_means[ann["category_id"]])
        if changed:
            enforce_count += 1
            ann["concept_vector"] = updated.tolist()

    for ann in annotations:
        cid = ann["category_id"]
        binvec = binarize_instance(np.array(ann["concept_vector"], dtype=float), class_means[cid], class_top3[cid])
        ann["concept_vector"] = fix_all_zero_binary(binvec, class_top3[cid]).tolist()

    return to_python({
        "num_annotations": len(annotations),
        "vec_len": vec_len,
        "num_classes": len(class_means),
        "enforced_nonzero_fixes": enforce_count,
        "binarized_instances": len(annotations),
        "class_top3": {str(cid): class_top3[cid] for cid in list(class_top3)[:50]},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("input_json", help="Raw concept-vector JSON (main_coco.py label_instances).")
    parser.add_argument("output_json", help="Binarized output JSON.")
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        data = json.load(f)
    summary = binarize(data)
    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
