"""Experiment / checkpoint loading and inference.

All `yolox` imports are local to the functions so that `--help` and the
pure helpers work without the model package installed.

yolox APIs used:
    yolox.exp.get_exp
    yolox.utils.cbm_predictor.CBMPredictor
    yolox.utils.postprocess_with_scale_info
    yolox.utils.load_ckpt               (non-strict loading of the plain variant)
    model.head.clear_interventions(), model.head.concept_outputs
"""

import argparse
import os

import torch

from .coco import COCO_CLASSES


def add_model_args(parser, conf_default=None, nms_default=None, fp16_default=True, ckpt=True):
    """Standard -f/-c/--device/--fp16/--conf/--nms arguments.

    conf/nms default to None = keep the experiment file's own
    exp.test_conf / exp.nmsthre. ckpt=False omits -c (multi-checkpoint scripts).
    """
    g = parser.add_argument_group("model")
    g.add_argument("-f", "--exp_file", type=str, required=True,
                   help="Experiment file, e.g. cbyolox/exps/cb_yolox/cb_yolox_s.py")
    if ckpt:
        g.add_argument("-c", "--ckpt", type=str, required=True, help="Checkpoint (.pth) to evaluate")
    g.add_argument("-n", "--name", type=str, default=None, help="Experiment name (unused when -f is given)")
    g.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    g.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=fp16_default,
                   help="Half precision on GPU (ignored on CPU)")
    conf_help = "exp.test_conf" if conf_default is None else str(conf_default)
    nms_help = "exp.nmsthre" if nms_default is None else str(nms_default)
    g.add_argument("--conf", type=float, default=conf_default,
                   help=f"Detection confidence threshold (default: {conf_help})")
    g.add_argument("--nms", type=float, default=nms_default,
                   help=f"NMS IoU threshold (default: {nms_help})")
    return parser


def build_exp(exp_file, name=None, cbm=True, conf=None, nms=None):
    """get_exp() + threshold overrides. `cbm=False` makes get_model() build the
    plain YOLOX head (num_concepts=None) from the same experiment file."""
    from yolox.exp import get_exp

    exp = get_exp(exp_file, name)
    if not cbm:
        exp.num_concepts = None
        if hasattr(exp, "use_cbm"):
            exp.use_cbm = False
    if conf is not None:
        exp.test_conf = conf
    if nms is not None:
        exp.nmsthre = nms
    return exp


def load_weights(model, ckpt_path, strict=True):
    """Load `ckpt_path` into `model`. strict=False matches tensors by name and
    shape (yolox.utils.load_ckpt), e.g. to build the plain head from a CB-YOLOX
    checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if strict:
        model.load_state_dict(state)
        return model
    from yolox.utils import load_ckpt
    return load_ckpt(model, state)


def load_model(exp_file, ckpt_path, name=None, device="gpu", fp16=True, conf=None, nms=None,
               cbm=True, strict=True):
    """Build the model of `exp_file`, move it to the device, load weights.
    Returns (model, exp) with the model in eval mode and no interventions."""
    exp = build_exp(exp_file, name, cbm=cbm, conf=conf, nms=nms)
    model = exp.get_model()
    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()
    model.eval()
    print(f"Loading checkpoint from: {ckpt_path}")
    model = load_weights(model, ckpt_path, strict=strict)
    model.head.clear_interventions()
    return model, exp


def load_predictor(args, ckpt_path=None, require_concepts=True, cbm=True, strict=True):
    """Model + CBMPredictor from the standard model arguments (checkpoint
    `ckpt_path`, default args.ckpt). Returns (predictor, model, exp)."""
    from yolox.utils.cbm_predictor import CBMPredictor

    print(f"Loading experiment file: {args.exp_file}")
    ckpt_path = args.ckpt if ckpt_path is None else ckpt_path
    model, exp = load_model(args.exp_file, ckpt_path, args.name, args.device, args.fp16,
                            args.conf, args.nms, cbm=cbm, strict=strict)
    print(f"Confidence threshold: {exp.test_conf} "
          f"({'--conf' if args.conf is not None else 'exp default'}), "
          f"NMS threshold: {exp.nmsthre} ({'--nms' if args.nms is not None else 'exp default'})")
    predictor = CBMPredictor(model, exp, cls_names=COCO_CLASSES, device=args.device, fp16=args.fp16)
    if require_concepts and predictor.num_concepts is None:
        raise ValueError("This checkpoint has no concept head (num_concepts is None); "
                         "this analysis only applies to CB-YOLOX models.")
    return predictor, model, exp


def flatten_concept_outputs(concept_outputs):
    """Per-level raw concept logits [(1, K, H, W), ...] -> (n_anchors, K)
    float32 array, in the anchor order used by `final_anchor_idxs` and by
    `set_interventions()`. None if the model has no concept head."""
    if concept_outputs is None:
        return None
    per_level = []
    for c in concept_outputs:
        b, k, h, w = c.shape
        per_level.append(c.view(b, k, h * w).permute(0, 2, 1))
    return torch.cat(per_level, dim=1)[0].detach().cpu().float().numpy()


def run_inference(predictor, img_path, class_agnostic=True, verbose=False):
    """One forward pass on a single image.

    class_agnostic=True uses CBMPredictor.inference() (class-agnostic NMS, as
    in the demo). class_agnostic=False runs the same preprocessing and forward
    pass but standard per-class NMS, which is what the official YOLOX COCO
    evaluator uses; prefer it whenever detection mAP or the set of matched
    instances matters.

    Returns (outputs, img_info, concept_logits (n_anchors, K) | None, scale_info).
    """
    if class_agnostic:
        outputs, img_info, concept_outputs, scale_info = predictor.inference(img_path, verbose=verbose)
        return outputs, img_info, flatten_concept_outputs(concept_outputs), scale_info

    from yolox.utils import postprocess_with_scale_info

    import cv2
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img_info = {"id": 0, "file_name": os.path.basename(img_path), "height": height,
                "width": width, "raw_img": img,
                "ratio": min(predictor.test_size[0] / height, predictor.test_size[1] / width)}

    tensor, _ = predictor.preproc(img, None, predictor.test_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).float()
    if predictor.device == "gpu":
        tensor = tensor.cuda()
        if predictor.fp16:
            tensor = tensor.half()

    head = predictor.model.head
    with torch.no_grad():
        if hasattr(head, "concept_outputs"):
            head.concept_outputs = None
        raw_outputs = predictor.model(tensor)
        concept_outputs = getattr(head, "concept_outputs", None)
        if predictor.decoder is not None:
            raw_outputs = predictor.decoder(raw_outputs, dtype=raw_outputs.type())
        outputs, scale_info = postprocess_with_scale_info(
            raw_outputs, predictor.num_classes, predictor.confthre, predictor.nmsthre,
            class_agnostic=False,
        )
    return outputs, img_info, flatten_concept_outputs(concept_outputs), scale_info


def anchor_info(scale_info):
    """(final_anchor_idxs, scale_map) of the kept detections as Python lists."""
    info = scale_info[0]
    return info["final_anchor_idxs"].tolist(), info["scale_map"].tolist()


def empty_cuda_cache(device):
    if device == "gpu":
        torch.cuda.empty_cache()
