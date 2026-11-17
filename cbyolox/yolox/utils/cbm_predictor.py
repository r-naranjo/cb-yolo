#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Single-image inference for CB-YOLOX (and plain YOLOX) models that also returns
the concept logits and, for every detection, the anchor / FPN level it comes
from (so its concept vector can be looked up or intervened on).
"""

import json
import os
import time

import cv2
import torch
from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES

from .boxes import postprocess_with_scale_info
from .visualize import vis

__all__ = ["CBMPredictor", "flatten_concept_outputs", "load_concept_names"]


def flatten_concept_outputs(concept_outputs):
    """
    Concatenate per-level concept logits [(B, K, H, W), ...] into
    (B, n_anchors_all, K), in the same anchor order as the model outputs.
    Returns None if ``concept_outputs`` is None.
    """
    if concept_outputs is None:
        return None
    return torch.cat(
        [c.view(c.shape[0], c.shape[1], -1).permute(0, 2, 1) for c in concept_outputs],
        dim=1,
    )


def load_concept_names(path):
    """
    Load concept names from:
      - a .txt file with one concept per line;
      - a JSON list of names;
      - a JSON ``{"concepts": [...]}`` or ``{"concepts": {group: [...], ...}}``
        (groups are concatenated in file order);
      - a JSON per-class descriptor file ``{class_name: [descriptors], ...}``
        (concatenated in file order, names become ``"class: descriptor"``).
    """
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\r\n") for line in f]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "concepts" in data:
        data = data["concepts"]
        if isinstance(data, dict):
            return [name for group in data.values() for name in group]
        return list(data)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        return [f"{cls}: {desc}" for cls, descs in data.items() for desc in descs]
    raise ValueError(f"Unsupported concept names file: {path}")


class CBMPredictor:
    """
    Args:
        model: YOLOX model (eval mode, weights loaded).
        exp: experiment; provides num_classes, test_conf, nmsthre, test_size.
        cls_names: class names.
        decoder: optional output decoder (for models with decode_in_inference=False).
        device: "cpu" or "gpu".
        fp16: run in half precision (gpu only).
        legacy: legacy ValTransform normalisation.
        concept_names: optional list of K concept names used in the verbose report.
    """

    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        concept_names=None,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.num_concepts = getattr(model.head, "num_concepts", None)
        self.concept_names = concept_names
        # number of concepts listed per detection in the verbose report
        self.topk_concepts = 20

    def inference(self, img, verbose=False, info_folder=None, image_name=None):
        """
        Args:
            img: image path or BGR ndarray.
            verbose: print a per-detection report (score, anchor, FPN level,
                top concepts). If ``info_folder`` is also given, the report is
                written to ``info_folder/<image_name stem>.txt``.
            info_folder, image_name: where to write the report.

        Returns:
            outputs: list (batch of 1) of (N, 7) tensors
                [x1, y1, x2, y2, obj_conf, class_conf, class_pred] in the
                resized (test_size) frame -- divide by img_info["ratio"] for
                original-image coordinates -- or None. Class-agnostic NMS.
            img_info: dict with id, file_name, height, width, raw_img, ratio.
            concept_outputs: list of raw concept logits (B, K, H, W) per FPN
                level, or None for a plain YOLOX model.
            scale_info: per-image scale / anchor information from
                :func:`yolox.utils.postprocess_with_scale_info`.
        """
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            if hasattr(self.model.head, "concept_outputs"):
                self.model.head.concept_outputs = None
            outputs = self.model(img)
            concept_outputs = getattr(self.model.head, "concept_outputs", None)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())

            raw_outputs = outputs.clone() if verbose else None
            # postprocess_with_scale_info modifies `outputs` in place
            outputs, scale_info = postprocess_with_scale_info(
                outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
            )
            if verbose:
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
                self.report(
                    raw_outputs,
                    outputs,
                    flatten_concept_outputs(concept_outputs),
                    scale_info,
                    info_folder=info_folder,
                    image_name=image_name,
                )

        return outputs, img_info, concept_outputs, scale_info

    def concept_name(self, idx):
        if self.concept_names is not None and 0 <= idx < len(self.concept_names):
            return self.concept_names[idx]
        return f"c{idx}"

    def report(self, raw_outputs, outputs, concept_logits, scale_info,
               info_folder=None, image_name=None):
        """Per-detection report: scores, anchor, FPN level and top concepts."""
        lines = []
        append = lines.append

        _, num_anchors, num_features = raw_outputs.shape
        append(f"Raw outputs: {num_anchors} anchors x {num_features} "
               f"(4 bbox + 1 obj + {num_features - 5} classes)")
        if concept_logits is not None:
            append(f"Concept logits: {tuple(concept_logits.shape)}")
        else:
            append("Concept logits: None (plain YOLOX model)")

        dets = outputs[0]
        if dets is None:
            append("No detections above the thresholds.")
        else:
            info = scale_info[0]
            primary_scales = info["scale_map"].tolist()
            anchors = info["final_anchor_idxs"].tolist()
            for i, det in enumerate(dets):
                x1, y1, x2, y2, obj_score, cls_score, cls_id = det.tolist()
                cls_id = int(cls_id)
                append("")
                append(f"det {i}: {self.cls_names[cls_id]}  "
                       f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) (resized frame)")
                append(f"    score = obj {obj_score:.3f} x cls {cls_score:.3f} = "
                       f"{obj_score * cls_score:.3f}")
                append(f"    anchor {anchors[i]}, FPN level {primary_scales[i]}, "
                       f"contributing levels {info['contributions'][i]['all_contributing_scales']}")
                if concept_logits is None:
                    continue
                logits = concept_logits[0, anchors[i]].detach().cpu().float()
                probs = torch.sigmoid(logits)
                top_p, top_idx = torch.topk(probs, k=min(self.topk_concepts, probs.numel()))
                append(f"    top {len(top_idx)} concepts:")
                append(f"    {'idx':>5}  {'logit':>9}  {'prob':>7}  concept")
                for p, idx in zip(top_p.tolist(), top_idx.tolist()):
                    append(f"    {idx:>5}  {logits[idx].item():9.4f}  {p:7.4f}  "
                           f"{self.concept_name(idx)}")

        text = "\n".join(lines)
        print(text)
        if info_folder is not None:
            os.makedirs(info_folder, exist_ok=True)
            stem = (
                os.path.splitext(os.path.basename(image_name))[0]
                if image_name else time.strftime("%Y_%m_%d_%H_%M_%S")
            )
            info_file = os.path.join(info_folder, f"{stem}.txt")
            with open(info_file, "w", encoding="utf-8") as f:
                f.write(text + "\n")
            logger.info(f"Saved detection report to {info_file}")

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        # back to original-image coordinates (does not modify `output`)
        bboxes = output[:, 0:4] / ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img.copy(), bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res
