# CB-YOLOX

CB-YOLOX is YOLOX with a sequential Concept Bottleneck Model (CBM) in the
classification branch of the detection head. Every anchor first predicts K
human-interpretable concepts (for example `has_wheels_spoked`, `has_beak`) and
the class scores are computed only from those concepts. Concepts can be read
out per detection and edited at test time (interventions).

This directory is a fork of [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
at commit `6ddff48`. The Python package is still called `yolox`, and
everything that is not listed under [Changes from upstream YOLOX](#changes-from-upstream-yolox)
works as in YOLOX.

## Architecture

For each FPN level k (strides 8, 16 and 32), the YOLOX decoupled head
(`yolox/models/yolo_head.py`) computes:

```
x_k      = stems[k](fpn_feature_k)
cls_feat = cls_convs[k](x_k)
c        = concept_preds[k](cls_feat)                # g: 1x1 conv -> K concept logits
y        = concept_to_cls[k](sigmoid(c))             # f: 1x1 conv -> 80 class logits
reg, obj = reg_preds[k](reg_convs[k](x_k)), obj_preds[k](reg_convs[k](x_k))
```

`cls_preds` is still built, so checkpoints keep the same keys, but the CBM path
does not use it. With `num_concepts=None` the head is the vanilla YOLOX head.

**Training loss.** Every term is normalised by the number of foreground anchors:

```
L = 5 * L_iou + L_obj + L_cls + L_l1 + w_c * L_concept
```

`L_concept` compares the concept logits of each foreground anchor with the
concept vector of the ground-truth box it is assigned to. It is the
asymmetric loss (ASL, `timm.loss.AsymmetricLossMultiLabel` with gamma_neg=4,
gamma_pos=1 and clip=0.05; `concept_loss_type="asl"`) or a binary
cross-entropy (`"bce"`). The weight `w_c` is set by `concept_loss_weight`. All
the released experiments use 1.0. The `YOLOXHead` constructor default is
1.269, the value found by the random search. In CB-YOLOX the L1 term is active
from the first epoch (`l1_loss_from_start`).

**Concept-aware SimOTA.** The cost of assigning anchor j to ground-truth box i
is:

```
c_ij = L_cls_ij + 3 * L_iou_ij + sigma * L_concept_ij      (+ YOLOX centre-prior constraint)
```

`L_concept_ij` is the BCE between the concept logits of anchor j and the
concept vector of box i, summed over the K concepts. The term is used only
when `use_concepts_in_simota=True`. Sigma is set by `simota_concept_weight`
(default 0.6148) and the IoU weight by the head argument `simota_iou_weight`
(default 3.0).

**Interventions.** `model.head.set_interventions([[anchor_idx, [concept_ids], [values]], ...])`
overwrites the listed concept logits (the values before the sigmoid) of the
given anchors in every later forward pass. `model.head.clear_interventions()`
removes them. `anchor_idx` is the flat anchor index over the three levels:
0-6399 is stride 8, 6400-7999 is stride 16 and 8000-8399 is stride 32 for a
640x640 input. `yolox.utils.postprocess_with_scale_info` returns this index
for every detection. After each forward pass, `model.head.concept_outputs`
holds the raw concept logits as a list of `(B, K, H, W)` tensors, one per
level.

## Installation

```bash
cd cbyolox
pip install -r requirements.txt
pip install -v -e .
```

CB-YOLOX adds two dependencies to YOLOX:

- `timm>=1.0`, for the ASL concept loss.
- `albumentations==2.0.8`

The grid-search helper tools also need `scikit-learn`, `scipy` and `plotly`.

## Data

Use MS-COCO 2017 images, with COCO-format annotation JSONs in which every
annotation has an extra `concept_vector` field: a list of K values in {0, 1}.
The JSONs are produced by the concept annotation pipeline in `../annotation`
(Qwen3-VL: K=143 shared concepts; CLIP: K=400, 80 classes x 5 descriptors).
An optional top-level `concept_names` list also sets K. Put the files in the
YOLOX layout:

```
cbyolox/datasets/COCO/
    train2017/  val2017/                       # COCO images
    annotations/instances_train2017.json       # concept-annotated train JSON
    annotations/instances_val2017.json         # concept-annotated val JSON
```

The JSONs used for the final models are rebuilt from the released concept
packs with `annotation/release/merge_concepts.py` (see `annotation/README.md`).
You can create symlinks with these names, or keep the original names and set
`train_ann` / `val_ann` (file names inside `annotations/`) and `data_dir` in
the experiment, or pass them on the command line (see below). Training stops
with an error if the train JSON does not provide `num_concepts` concepts per
object. Evaluation only uses the boxes, so a plain COCO val JSON also works
for mAP.

`COCODataset` loads the concept vector of each object into the label columns
`5:5+K`. The data pipeline (mosaic, mixup, Albumentations, flip and
multi-scale rescaling) carries these columns along with the boxes.

## Experiments

| Experiment file | What it trains |
|---|---|
| `exps/cb_yolox/cb_yolox_s.py`, `cb_yolox_m.py`, `cb_yolox_l.py` | Main models. Starting from a trained YOLOX-S/M/L, trains only the classification branch (`cls_convs`, `cls_preds`, `concept_preds`, `concept_to_cls`). Backbone, FPN, `stems` and the box and objectness branches stay frozen, and their BatchNorm layers are kept in eval mode. 300 epochs, no mosaic/mixup, HSV + flip + Albumentations (rotate +-5 deg, elastic, channel dropout), multi-scale 448-832, K=143, w_c=1, ASL, concept SimOTA. |
| `exps/cb_yolox/cb_yolox_s_unfreeze_ep50.py` | Same as `cb_yolox_s.py` for epochs 0-49, then the whole model is trained from epoch 50 (the optimizer is rebuilt). Uses vanilla SimOTA. |
| `exps/cb_yolox/cb_yolox_s_scratch.py` | CB-YOLOX-S trained end-to-end with the standard YOLOX schedule and augmentations (mosaic, mixup) plus Albumentations. |

The shared logic is in `yolox/exp/cb_yolox_exp.py` (`CBYOLOXExp`,
`CBYOLOXFinetuneExp`) and `yolox/core/cb_finetune_trainer.py`. The concept
options are attributes of every experiment (`yolox/exp/yolox_base.py`):
`num_concepts`, `use_cbm`, `concept_loss_weight`, `concept_loss_type`,
`use_concepts_in_simota` and `simota_concept_weight`.
The released fine-tuned models come in two variants:

- `use_concepts_in_simota True` (default): "frozen head + concept SimOTA".
- `use_concepts_in_simota False`: "frozen head", with the vanilla SimOTA cost.

## Training

Download the official YOLOX weights from the YOLOX releases page
(https://github.com/Megvii-BaseDetection/YOLOX/releases, files `yolox_s.pth`,
`yolox_m.pth` and `yolox_l.pth`). Then run:

```bash
cd cbyolox
# concept SimOTA variant
python tools/train.py -f exps/cb_yolox/cb_yolox_s.py -d 1 -b 16 --fp16 -c yolox_s.pth

# vanilla SimOTA variant (separate output folder)
python tools/train.py -f exps/cb_yolox/cb_yolox_s.py -d 1 -b 16 --fp16 -c yolox_s.pth \
    -expn cb_yolox_s_vanilla_simota use_concepts_in_simota False
```

`-c` loads the YOLOX checkpoint and skips layers that do not exist in it
(`concept_preds`, `concept_to_cls`), which start from random initialisation.
`pretrained_ckpt <path>` passed as an experiment option has the same effect.
Any experiment attribute can be overridden at the end of the command, for
example `train_ann my_train.json val_ann my_val.json num_concepts 400 concept_loss_type bce`.
Outputs go to `YOLOX_outputs/<exp_name>/`.

## Evaluation

COCO box mAP, as in YOLOX:

```bash
python tools/eval.py -f exps/cb_yolox/cb_yolox_s.py -c YOLOX_outputs/cb_yolox_s/best_ckpt.pth -b 8 -d 1 --conf 0.001 --fp16 --fuse
```

The concept-level evaluation and analysis scripts are in `../evaluation`.

## Demo and Python API

```bash
python tools/demo_cbm.py -f exps/cb_yolox/cb_yolox_s.py -c cb_yolox_s.pth \
    --path assets/dog.jpg --conf 0.3 --report --save_result \
    --concept_names ../annotation/concepts/shared_143.json
```

- `--report` prints, for every detection, its score, anchor index, FPN level
  and top-k concepts (`--topk`). With `--save_result` the report is also
  written next to the images.
- `--path` takes an image or a directory.
- `--concept_names` accepts a text file (one name per line), a JSON list,
  `{"concepts": [...]}`, `{"concepts": {group: [...]}}` or a per-class
  descriptor JSON `{class: [descriptors]}` (for the 400-concept CLIP models).
- `--intervention ANCHOR:IDS:VALUES` (repeatable) or
  `--interventions_json file.json` (a list of `[anchor, [ids], [values]]`)
  runs a second, intervened pass. Its results are saved with the suffix
  `_intervened`. `ANCHOR` can be `det`, which applies the intervention to the
  anchors of all detections of the first pass. Example:
  `--intervention det:12,40:-10`.

`tools/demo.py` is the unmodified YOLOX demo.

In Python:

```python
import torch
from yolox.exp import get_exp
from yolox.utils.cbm_predictor import CBMPredictor, flatten_concept_outputs

exp = get_exp("exps/cb_yolox/cb_yolox_s.py")
model = exp.get_model().eval()
model.load_state_dict(torch.load("cb_yolox_s.pth", map_location="cpu")["model"])
predictor = CBMPredictor(model, exp, device="cpu")
outputs, img_info, concept_outputs, scale_info = predictor.inference("assets/dog.jpg")
# outputs[0]: (N, 7) [x1, y1, x2, y2, obj, cls_conf, cls] in the resized frame
#             (divide boxes by img_info["ratio"]); class-agnostic NMS
# scale_info[0]["final_anchor_idxs"]: anchor index of each detection
concepts = flatten_concept_outputs(concept_outputs)[0, scale_info[0]["final_anchor_idxs"]]
```

## Hyper-parameter search

`tools/grid_search/` contains a random search over `concept_loss_weight` and
`simota_concept_weight`. It trains each configuration for a few epochs on one
GPU and records the losses and the final val AP. Two more scripts add concept
metrics to the results and plot them:

```bash
python tools/grid_search/grid_search_concepts.py -f exps/cb_yolox/cb_yolox_s_scratch.py \
    -b 16 --epochs 10 --n_trials 20 --fp16 --train_ann instances_train2017_subset.json \
    --output YOLOX_outputs/random_search/results.json
python tools/grid_search/compute_concept_maps_search.py --results YOLOX_outputs/random_search/results.json \
    -f exps/cb_yolox/cb_yolox_s_scratch.py --img-dir datasets/COCO/val2017 \
    --json-file datasets/COCO/annotations/instances_val2017.json --limit 500
python tools/grid_search/plot_grid_search.py -i YOLOX_outputs/random_search/results.json
```

## Changes from upstream YOLOX

- `yolox/models/yolo_head.py`: the CBM classification branch (`concept_preds`,
  `concept_to_cls`), the ASL/BCE concept loss, the concept term in the SimOTA
  cost, interventions and `concept_outputs`. New constructor arguments:
  `num_concepts`, `concept_loss_weight`, `concept_loss_type`, `asl_*`,
  `use_concepts_in_simota`, `simota_iou_weight` and `simota_concept_weight`.
- `yolox/models/yolox.py`: logs `concept_loss` during training.
- `yolox/exp/yolox_base.py`: the concept attributes, passed to the head. Adds
  `get_train_transform()` and a check that the dataset has K concepts.
  `preprocess()` now rescales only the box columns during multi-scale
  training.
- `yolox/exp/cb_yolox_exp.py` and `yolox/core/cb_finetune_trainer.py`: the
  CB-YOLOX base experiments and the trainer for freezing.
- `yolox/data/datasets/coco.py`: loads `concept_vector` into the label columns
  `5:5+K`.
- `yolox/data/data_augment.py`: `TrainTransform` and `ValTransform` carry the
  concept columns. Adds `AlbumentationsTransform` and `ComposedTrainTransform`.
- `yolox/data/datasets/mosaicdetection.py`: mosaic and mixup keep the concept
  columns.
- `yolox/utils/boxes.py`: `postprocess_with_scale_info`, which returns the
  anchor index and FPN level of each detection.
- `yolox/utils/cbm_predictor.py`, `tools/demo_cbm.py`, `tools/grid_search/`
  and `exps/cb_yolox/`: new files.
- Removed: `hubconf.py`, `tools/export_torchscript.py`, `tools/trt.py`,
  `tools/visualize_assign.py`, and the upstream `docs/`, `demo/` and `tests/`.

## License

Apache License 2.0 (see `LICENSE`). This code is derived from YOLOX,
Copyright (c) Megvii Inc.
