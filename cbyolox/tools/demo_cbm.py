#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
CB-YOLOX image demo: detections, optional per-detection concept report and
optional test-time concept interventions.

Examples:
    # detections + concept report for one image
    python tools/demo_cbm.py -f exps/cb_yolox/cb_yolox_s.py -c cb_yolox_s.pth \
        --path assets/dog.jpg --report --save_result \
        --concept_names ../annotation/concepts/shared_143.json

    # set concepts 0-4 of anchor 8150 to logit -10, and concept 12 of every
    # detected anchor to +10; saves the original and the intervened result
    python tools/demo_cbm.py -f exps/cb_yolox/cb_yolox_s.py -c cb_yolox_s.pth \
        --path assets/dog.jpg --save_result \
        --intervention 8150:0,1,2,3,4:-10 --intervention det:12:10
"""

import argparse
import json
import os
import time

import cv2
import torch
from loguru import logger

from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.utils.cbm_predictor import CBMPredictor, load_concept_names

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("CB-YOLOX demo")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint")
    parser.add_argument("--path", default="./assets/dog.jpg", help="image or directory of images")
    parser.add_argument("--save_result", action="store_true",
                        help="save the visualised detections under <output_dir>/<exp>/vis_res")
    parser.add_argument("--report", action="store_true",
                        help="print a per-detection report (anchor, FPN level, top concepts); "
                             "written next to the saved images with --save_result")
    parser.add_argument("--topk", type=int, default=20, help="concepts listed per detection")
    parser.add_argument("--concept_names", default=None, type=str,
                        help="concept names: .txt (one per line), JSON list, "
                             "JSON {'concepts': [...] or {group: [...]}}, or per-class "
                             "descriptor JSON {class: [descriptors]}")
    parser.add_argument("--intervention", action="append", default=[],
                        metavar="ANCHOR:IDS:VALUES",
                        help="concept intervention, e.g. 8150:0,1,2:-10 (VALUES is one "
                             "logit or one per id). ANCHOR 'det' means every detection of "
                             "the non-intervened pass. Can be repeated.")
    parser.add_argument("--interventions_json", default=None, type=str,
                        help="JSON list of [anchor_idx, [concept_ids], [values]]")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true",
                        help="To be compatible with older versions")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true",
                        help="Fuse conv and bn for testing.")
    return parser


def get_image_list(path):
    if not os.path.isdir(path):
        return [path]
    image_names = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if os.path.splitext(apath)[1].lower() in IMAGE_EXT:
                image_names.append(apath)
    return sorted(image_names)


def parse_interventions(specs, json_file=None):
    """Return a list of [anchor ("det" or int), [concept ids], [values]]."""
    interventions = []
    for spec in specs:
        try:
            anchor, ids, values = spec.split(":")
            ids = [int(i) for i in ids.split(",")]
            values = [float(v) for v in values.split(",")]
        except ValueError:
            raise ValueError(f"Bad --intervention {spec!r}, expected ANCHOR:IDS:VALUES")
        anchor = anchor if anchor == "det" else int(anchor)
        interventions.append([anchor, ids, values])
    if json_file:
        with open(json_file, "r", encoding="utf-8") as f:
            interventions.extend(json.load(f))
    return interventions


def resolve_interventions(interventions, outputs, scale_info):
    """Expand anchor 'det' to the anchors of the detections in `outputs`."""
    det_anchors = []
    if outputs[0] is not None:
        det_anchors = scale_info[0]["final_anchor_idxs"].tolist()
    resolved = []
    for anchor, ids, values in interventions:
        anchors = det_anchors if anchor == "det" else [anchor]
        resolved.extend([a, ids, values] for a in anchors)
    return resolved


def run_one(predictor, image, report, out_dir, stem, save_result, tag=""):
    info_folder = out_dir if (report and save_result) else None
    outputs, img_info, _, scale_info = predictor.inference(
        image, verbose=report, info_folder=info_folder, image_name=stem + tag
    )
    if save_result:
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        save_file = os.path.join(out_dir, stem + tag + ".jpg")
        logger.info(f"Saving detection result in {save_file}")
        cv2.imwrite(save_file, result_image)
    return outputs, img_info, scale_info


def image_demo(predictor, vis_folder, path, save_result, report, interventions):
    out_dir = None
    if save_result:
        out_dir = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S"))
        os.makedirs(out_dir, exist_ok=True)

    head = predictor.model.head
    for image_name in get_image_list(path):
        stem = os.path.splitext(os.path.basename(image_name))[0]
        head.clear_interventions()
        outputs, img_info, scale_info = run_one(
            predictor, image_name, report, out_dir, stem, save_result
        )
        if not interventions:
            continue

        resolved = resolve_interventions(interventions, outputs, scale_info)
        if not resolved:
            logger.info(f"{image_name}: no anchors to intervene on")
            continue
        logger.info(f"{image_name}: applying {len(resolved)} interventions")
        if report:
            print("\n=== with interventions ===")
        head.set_interventions(resolved)
        try:
            run_one(predictor, img_info["raw_img"], report, out_dir, stem, save_result,
                    tag="_intervened")
        finally:
            head.clear_interventions()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = args.ckpt or os.path.join(file_name, "best_ckpt.pth")
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    concept_names = load_concept_names(args.concept_names) if args.concept_names else None
    num_concepts = getattr(model.head, "num_concepts", None)
    if concept_names is not None and len(concept_names) != num_concepts:
        logger.warning(
            f"{len(concept_names)} concept names for a model with {num_concepts} concepts"
        )

    interventions = parse_interventions(args.intervention, args.interventions_json)
    if interventions and num_concepts is None:
        raise ValueError("Interventions need a model with a concept bottleneck.")

    predictor = CBMPredictor(
        model, exp, COCO_CLASSES, device=args.device, fp16=args.fp16, legacy=args.legacy,
        concept_names=concept_names,
    )
    predictor.topk_concepts = args.topk
    image_demo(predictor, vis_folder, args.path, args.save_result, args.report, interventions)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
