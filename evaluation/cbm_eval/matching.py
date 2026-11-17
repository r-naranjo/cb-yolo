"""GT <-> detection matching.

Every concept metric needs to know which anchor's concept prediction belongs
to which ground-truth instance. All scripts do it the same way:

1. run inference and post-process (NMS) the detections;
2. bring the detection boxes back to original-image pixels (divide by
   `img_info["ratio"]`, clip to the image);
3. pair each GT box with its highest-IoU detection (same-class only, or any
   class when misclassifications must stay visible), accepting the pair if
   IoU >= iou_thresh;
4. use that detection's entry in `scale_info[0]["final_anchor_idxs"]` to
   index the flattened (n_anchors, K) concept-logit tensor.

Unmatched GT instances are skipped.
"""

import numpy as np

from .coco import CAT_TO_IDX, IDX_TO_CAT


def iou_xyxy(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


def gt_xyxy(ann):
    x, y, w, h = ann["bbox"]
    return [x, y, x + w, y + h]


def detections_to_boxes(outputs, img_info, rescale=True):
    """Post-processed detections of one image -> (boxes_xyxy, classes, scores).

    With `rescale=True` boxes are mapped from the letterboxed network-input
    space back to original-image pixels (divided by img_info["ratio"] and
    clipped), as the official YOLOX COCO evaluator does. `rescale=False`
    returns the raw network-space boxes (legacy behaviour of the per-class
    report and the concept-evolution sweep, see README).
    Returns (None, None, None) if there are no detections.
    """
    if outputs[0] is None:
        return None, None, None
    preds = outputs[0].detach().cpu().numpy()
    if rescale:
        ratio = img_info["ratio"]
        boxes = preds[:, :4] / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_info["width"])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_info["height"])
    else:
        boxes = preds[:, :4]
    classes = preds[:, 6].astype(int)
    scores = preds[:, 4] * preds[:, 5]
    return boxes, classes, scores


def detections_to_coco(boxes, classes, scores, image_id):
    """COCOeval result dicts ([x, y, w, h], COCO category ids)."""
    if boxes is None:
        return []
    return [{
        "image_id": image_id,
        "category_id": IDX_TO_CAT[int(cls)],
        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        "score": float(score),
    } for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes)]


def match_gt_same_class(gt_anns, boxes, classes, final_anchor_idxs, iou_thresh):
    """For every GT box, the highest-IoU detection *of the same class*.

    Used for concept-quality metrics: a detection with the wrong class is not
    a meaningful sample of concept-prediction quality.
    Returns dicts with gt_index (position in gt_anns), class_id, anchor_idx
    and the raw `concept_vector` of the annotation.
    """
    matches = []
    if boxes is None:
        return matches
    for gt_index, gt in enumerate(gt_anns):
        box_gt = gt_xyxy(gt)
        gt_class = CAT_TO_IDX[gt["category_id"]]
        best_iou, best_idx = 0.0, -1
        for i, box in enumerate(boxes):
            if int(classes[i]) != gt_class:
                continue
            iou = iou_xyxy(box_gt, box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx == -1 or best_iou < iou_thresh:
            continue
        matches.append({
            "gt_index": gt_index,
            "class_id": gt_class,
            "anchor_idx": final_anchor_idxs[best_idx],
            "concept_vector": gt["concept_vector"],
        })
    return matches


def match_gt_any_class(gt_anns, boxes, classes, scores, final_anchor_idxs, scale_map, iou_thresh):
    """For every GT box, the highest-IoU detection *regardless of its class*.

    Used by the intervention/faithfulness analyses, which need to see
    well-localized but misclassified instances (same-class matching would
    hide them by construction).
    """
    matches = []
    if boxes is None:
        return matches
    for gt in gt_anns:
        box_gt = gt_xyxy(gt)
        gt_class = CAT_TO_IDX[gt["category_id"]]
        best_iou, best_idx = 0.0, -1
        for i, box in enumerate(boxes):
            iou = iou_xyxy(box_gt, box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx == -1 or best_iou < iou_thresh:
            continue
        matches.append({
            "gt_id": gt["id"],
            "gt_class": gt_class,
            "pred_class": int(classes[best_idx]),
            "iou": float(best_iou),
            "score": float(scores[best_idx]),
            "anchor_idx": int(final_anchor_idxs[best_idx]),
            "scale_idx": int(scale_map[best_idx]),
            "gt_concepts_bin": np.asarray(gt["concept_vector"], dtype=np.float32),
        })
    return matches
