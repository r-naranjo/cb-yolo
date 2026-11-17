#!/usr/bin/env python3
"""Qualitative side-by-side comparison: plain YOLOX vs. CB-YOLOX.

For each model size, both variants run on the same sample images:
  * CB-YOLOX: the experiment/checkpoint as trained;
  * YOLOX (plain head): same experiment with num_concepts=None, weights from
    --normal_ckpt_template (a checkpoint trained with the plain head) or, if
    not given, loaded non-strictly from the CB-YOLOX checkpoint (only useful
    if its cls_preds were trained).
Inference uses CBMPredictor.inference (class-agnostic NMS).

Outputs (under --output_dir):
    samples.txt                      list of images used
    <size>/YOLOX-<S>/<img>.jpg       plain-head detections
    <size>/CB-YOLOX-<S>/<img>.jpg    CB-YOLOX detections (#i tags)
    <size>/comparison/<img>.jpg      YOLOX | CB-YOLOX | top concepts per detection
    <size>/detections.json           detections (+ concept activations for CB-YOLOX)
    <size>/summary.csv               per-image detection counts and agreement
    overall_summary.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import COCO_CLASSES  # noqa: E402
from cbm_eval.concepts import read_concept_names  # noqa: E402
from cbm_eval.matching import detections_to_boxes  # noqa: E402
from cbm_eval.metrics import sigmoid  # noqa: E402
from cbm_eval.model import anchor_info, load_model, run_inference  # noqa: E402

DEFAULT_SIZES = ["s", "m", "l"]
DEFAULT_EXP_FILE_TEMPLATE = "cbyolox/exps/cb_yolox/cb_yolox_{size}.py"
DEFAULT_CKPT_TEMPLATE = "YOLOX_outputs/cb_yolox_{size}/best_ckpt.pth"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_parser():
    parser = argparse.ArgumentParser(description="Plain YOLOX vs. CB-YOLOX qualitative comparison")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, choices=["s", "m", "l"])
    parser.add_argument("--exp_file_template", default=DEFAULT_EXP_FILE_TEMPLATE,
                        help="Experiment file with a {size} placeholder")
    parser.add_argument("--ckpt_template", default=DEFAULT_CKPT_TEMPLATE,
                        help="CB-YOLOX checkpoint with a {size} placeholder")
    parser.add_argument("--normal_ckpt_template", default=None,
                        help="Checkpoint of a plain-head YOLOX ({size} placeholder); see module docstring")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir", help="Directory to sample images from")
    src.add_argument("--images", nargs="+", help="Explicit list of image paths")
    parser.add_argument("--num_samples", type=int, default=20, help="Images sampled from --image_dir")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--fp16", action="store_true", help="Half precision (GPU only)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--match_iou", type=float, default=0.5,
                        help="IoU for counting a detection as matched by the other model")
    parser.add_argument("--top_k_concepts", type=int, default=3, help="Concepts shown per detection")
    parser.add_argument("--concepts", default=None, help="Concept-name file (default: c<idx>)")
    parser.add_argument("--panel_width", type=int, default=480, help="Width of the concept panel (px)")
    parser.add_argument("--output_dir", default="outputs/qualitative")
    return parser


def predict(predictor, img):
    """((boxes, scores, cls_ids), concept probabilities (n_det, K) or None)."""
    outputs, img_info, concept_logits, scale_info = run_inference(predictor, img, class_agnostic=True)
    boxes, classes, scores = detections_to_boxes(outputs, img_info)
    if boxes is None:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)), None
    det = (boxes.astype(np.float32), scores.astype(np.float32), classes.astype(np.int64))
    concepts = None
    if concept_logits is not None:
        anchor_idxs, _ = anchor_info(scale_info)
        concepts = sigmoid(concept_logits[anchor_idxs])
    return det, concepts


def model_label(variant, size):
    """Display name: normal -> YOLOX-{SIZE}, cbm -> CB-YOLOX-{SIZE}."""
    return f"{'CB-YOLOX' if variant == 'cbm' else 'YOLOX'}-{size.upper()}"


def select_samples(args):
    if args.images:
        paths = list(args.images)
    else:
        paths = sorted(os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                       if f.lower().endswith(IMG_EXTS))
        if not paths:
            raise RuntimeError(f"No images found in {args.image_dir}")
        rng = random.Random(args.seed)
        if args.num_samples < len(paths):
            paths = sorted(rng.sample(paths, args.num_samples))
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}")
    return paths


def top_concepts(vec, names, k):
    order = np.argsort(-vec)[:k]
    return [(names[i], float(vec[i])) for i in order]


def draw_indices(img, boxes):
    """Draw a '#i' tag on each box so it can be matched to the concept panel."""
    for i, b in enumerate(boxes):
        x0, y0 = int(b[0]), int(b[1])
        (tw, th), _ = cv2.getTextSize(f"#{i}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x0, y0), (x0 + tw + 6, y0 + th + 6), (0, 0, 0), -1)
        cv2.putText(img, f"#{i}", (x0 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def concept_panel(height, width, det, concepts, names, class_names, k):
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    boxes, scores, cls_ids = det
    if concepts is None:
        cv2.putText(panel, "no concepts available", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return panel
    y, lh = 20, 16
    for i in range(len(boxes)):
        need = lh * (1 + k) + 6
        if y + need > height:
            cv2.putText(panel, f"... +{len(boxes) - i} more (see detections.json)", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 160), 1, cv2.LINE_AA)
            break
        c = int(cls_ids[i])
        cname = class_names[c] if c < len(class_names) else str(c)
        cv2.putText(panel, f"#{i} {cname} {scores[i]:.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        y += lh
        for nm, v in top_concepts(concepts[i], names, k):
            bar_w = int(v * 120)
            cv2.rectangle(panel, (width - 135, y - 9), (width - 135 + bar_w, y - 1), (60, 140, 230), -1)
            cv2.putText(panel, f"{nm[:34]} {v:.2f}", (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (60, 60, 60), 1, cv2.LINE_AA)
            y += lh
        y += 6
    return panel


def draw(img, det, conf, class_names):
    from yolox.utils import vis

    boxes, scores, cls_ids = det
    if len(boxes) == 0:
        return img.copy()
    return vis(img.copy(), boxes, scores, cls_ids, conf, class_names)


def add_title(img, text, height=36):
    bar = np.full((height, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def agreement(det_ref, det_other, iou_thr):
    """Fraction of reference detections that have a same-class detection in
    `det_other` with IoU >= iou_thr."""
    b1, _, c1 = det_ref
    b2, _, c2 = det_other
    if len(b1) == 0:
        return float("nan")
    if len(b2) == 0:
        return 0.0
    iou = box_iou(torch.from_numpy(b1), torch.from_numpy(b2)).numpy()
    same_cls = c1[:, None] == c2[None, :]
    return float(((iou >= iou_thr) & same_cls).any(axis=1).mean())


def det_to_json(det, class_names, concepts=None, concept_names=None, k=3):
    boxes, scores, cls_ids = det
    out = [{
        "bbox_xyxy": [round(float(v), 2) for v in b],
        "score": round(float(s), 4),
        "class_id": int(c),
        "class_name": class_names[int(c)] if int(c) < len(class_names) else str(int(c)),
    } for b, s, c in zip(boxes, scores, cls_ids)]
    if concepts is not None:
        for d, vec in zip(out, concepts):
            d["top_concepts"] = [{"name": n, "activation": round(v, 4)}
                                 for n, v in top_concepts(vec, concept_names, k)]
            d["concept_activations"] = [round(float(v), 4) for v in vec]
    return out


def main(args):
    from yolox.utils.cbm_predictor import CBMPredictor

    device = args.device
    if device == "gpu" and not torch.cuda.is_available():
        print("[WARN] CUDA not available - using cpu.")
        device = "cpu"
    fp16 = args.fp16 and device == "gpu"

    os.makedirs(args.output_dir, exist_ok=True)
    samples = select_samples(args)
    with open(os.path.join(args.output_dir, "samples.txt"), "w") as f:
        f.write("\n".join(samples) + "\n")
    print(f"Using {len(samples)} sample images (identical for all models).")
    images = {p: cv2.imread(p) for p in samples}
    bad = [p for p, im in images.items() if im is None]
    if bad:
        raise RuntimeError(f"Could not read images: {bad}")
    class_names = list(COCO_CLASSES)

    overall = []
    for size in args.sizes:
        exp_file = args.exp_file_template.format(size=size)
        ckpt_path = args.ckpt_template.format(size=size)
        if not os.path.isfile(exp_file) or not os.path.isfile(ckpt_path):
            print(f"[WARN] exp file or checkpoint missing for size={size} - skipped.")
            continue
        size_dir = os.path.join(args.output_dir, size)
        ln, lc = model_label("normal", size), model_label("cbm", size)
        for sub in (ln, lc, "comparison"):
            os.makedirs(os.path.join(size_dir, sub), exist_ok=True)

        dets, concs, concept_names = {"normal": {}, "cbm": {}}, {}, None
        for use_cbm, variant in [(False, "normal"), (True, "cbm")]:
            print(f"\n=== {model_label(variant, size)} ===")
            variant_ckpt = ckpt_path
            if not use_cbm:
                if args.normal_ckpt_template:
                    variant_ckpt = args.normal_ckpt_template.format(size=size)
                else:
                    print("  [WARN] --normal_ckpt_template not given: plain head loaded from the "
                          "CB-YOLOX checkpoint.")
            model, exp = load_model(exp_file, variant_ckpt, device=device, fp16=fp16, conf=args.conf,
                                    nms=args.nms, cbm=use_cbm, strict=use_cbm)
            predictor = CBMPredictor(model, exp, cls_names=class_names, device=device, fp16=fp16)
            for p in samples:
                dets[variant][p], conc = predict(predictor, images[p])
                if use_cbm:
                    concs[p] = conc
            if use_cbm:
                k = predictor.num_concepts or 0
                concept_names = (read_concept_names(args.concepts) if args.concepts
                                 else [f"c{i}" for i in range(k)])
                if len(concept_names) != k:
                    print(f"  [WARN] {len(concept_names)} concept names for {k} model concepts")
                    concept_names = (concept_names + [f"c{i}" for i in range(len(concept_names), k)])[:k]
            del model, predictor
            if device == "gpu":
                torch.cuda.empty_cache()

        json_out, rows = {}, []
        for p in samples:
            name = os.path.splitext(os.path.basename(p))[0]
            img = images[p]
            vis_n = draw(img, dets["normal"][p], args.conf, class_names)
            vis_c = draw_indices(draw(img, dets["cbm"][p], args.conf, class_names), dets["cbm"][p][0])
            cv2.imwrite(os.path.join(size_dir, ln, f"{name}.jpg"), vis_n)
            cv2.imwrite(os.path.join(size_dir, lc, f"{name}.jpg"), vis_c)
            panel = concept_panel(vis_c.shape[0], args.panel_width, dets["cbm"][p], concs.get(p),
                                  concept_names, class_names, args.top_k_concepts)
            side = np.hstack([
                add_title(vis_n, f"{ln}  ({len(dets['normal'][p][0])} det)"),
                add_title(vis_c, f"{lc}  ({len(dets['cbm'][p][0])} det)"),
                add_title(panel, f"{lc} concepts (top {args.top_k_concepts})"),
            ])
            cv2.imwrite(os.path.join(size_dir, "comparison", f"{name}.jpg"), side)
            json_out[p] = {
                ln: det_to_json(dets["normal"][p], class_names),
                lc: det_to_json(dets["cbm"][p], class_names, concs.get(p), concept_names, args.top_k_concepts),
            }
            rows.append((name, len(dets["normal"][p][0]), len(dets["cbm"][p][0]),
                         agreement(dets["normal"][p], dets["cbm"][p], args.match_iou),
                         agreement(dets["cbm"][p], dets["normal"][p], args.match_iou)))

        with open(os.path.join(size_dir, "detections.json"), "w") as f:
            json.dump(json_out, f, indent=2)
        with open(os.path.join(size_dir, "summary.csv"), "w") as f:
            f.write(f"image,n_{ln},n_{lc},{ln}_matched_by_{lc},{lc}_matched_by_{ln}\n")
            for r in rows:
                f.write(",".join(str(v) for v in r) + "\n")

        def mean_valid(i):
            vals = [r[i] for r in rows if not np.isnan(r[i])]
            return float(np.mean(vals)) if vals else float("nan")

        overall.append({"size": size, "normal_label": ln, "cbm_label": lc,
                        "avg_normal": float(np.mean([r[1] for r in rows])),
                        "avg_cbm": float(np.mean([r[2] for r in rows])),
                        "agree_nc": mean_valid(3), "agree_cn": mean_valid(4)})
        print(f"Saved results for size={size} to {size_dir}")

    if not overall:
        print("No results - check --sizes/--exp_file_template/--ckpt_template.")
        return
    with open(os.path.join(args.output_dir, "overall_summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": overall}, f, indent=2)
    for r in overall:
        print(f"{r['normal_label']} / {r['cbm_label']}: avg det {r['avg_normal']:.2f} / {r['avg_cbm']:.2f}, "
              f"{r['normal_label']} matched by {r['cbm_label']}: {r['agree_nc']:.1%}, "
              f"{r['cbm_label']} matched by {r['normal_label']}: {r['agree_cn']:.1%}")


if __name__ == "__main__":
    main(make_parser().parse_args())
