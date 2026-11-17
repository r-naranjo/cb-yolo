#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "postprocess_with_scale_info",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def postprocess_with_scale_info(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    Same filtering + NMS as :func:`postprocess`, but also reports which anchor
    and FPN level produced each final detection.

    The level of an anchor is derived from its flat index assuming a 640x640
    input with strides 8/16/32 (anchors 0-6399 -> level 0, 6400-7999 -> level 1,
    8000+ -> level 2).

    Returns:
        output (list): per image, a (N, 7) tensor
            [x1, y1, x2, y2, obj_conf, class_conf, class_pred] or None.
        scale_info (list): per image, None or a dict with
            - "scale_map": (N,) int8 tensor, FPN level of each detection
            - "contributions": list of dicts per detection with
              "primary_scale", "all_contributing_scales" (levels of all
              pre-NMS candidates with IoU > nms_thre with the detection,
              itself included) and "num_suppressed_detections"
            - "final_anchor_idxs": (N,) long tensor, flat anchor index of each
              detection (index into the concatenated anchors / concept maps)
            - "anchor_contributions": list of dicts per detection with
              "primary_anchor_idx" and "all_contributing_anchor_idxs"
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    scale_info = []

    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            scale_info.append(None)
            continue

        # original anchor indices and FPN level of every anchor
        anchors_idx = torch.arange(image_pred.shape[0], device=image_pred.device, dtype=torch.long)
        scale_map = torch.zeros_like(anchors_idx, dtype=torch.int8)
        scale_map = torch.where(anchors_idx < 6400, 0, scale_map)
        scale_map = torch.where((anchors_idx >= 6400) & (anchors_idx < 8000), 1, scale_map)
        scale_map = torch.where(anchors_idx >= 8000, 2, scale_map)

        # confidence filter
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        scale_map_conf = scale_map[conf_mask]
        anchor_idx_conf = anchors_idx[conf_mask]

        if not detections.size(0):
            scale_info.append(None)
            continue

        scores_for_nms = detections[:, 4] * detections[:, 5]
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(detections[:, :4], scores_for_nms, nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4], scores_for_nms, detections[:, 6], nms_thre
            )

        final_detections = detections[nms_out_index]
        final_scale_map = scale_map_conf[nms_out_index]
        final_anchor_idx = anchor_idx_conf[nms_out_index]

        # for each kept detection, the levels / anchors of all candidates it suppressed
        scale_contributions = []
        anchor_contributions = []
        for j, nms_idx in enumerate(nms_out_index):
            current_det = detections[nms_idx]
            ious = bboxes_iou(current_det[:4].unsqueeze(0), detections[:, :4])
            suppressed_by_this = ious.squeeze() > nms_thre

            scale_contributions.append({
                "primary_scale": final_scale_map[j].item(),
                "all_contributing_scales": scale_map_conf[suppressed_by_this].unique().tolist(),
                "num_suppressed_detections": int(suppressed_by_this.sum().item()),
            })
            anchor_contributions.append({
                "primary_anchor_idx": int(final_anchor_idx[j].item()),
                "all_contributing_anchor_idxs": anchor_idx_conf[suppressed_by_this].unique().tolist(),
            })

        output[i] = final_detections
        scale_info.append({
            "scale_map": final_scale_map,
            "contributions": scale_contributions,
            "final_anchor_idxs": final_anchor_idx,
            "anchor_contributions": anchor_contributions,
        })

    return output, scale_info


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes
