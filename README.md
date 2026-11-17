# CB-YOLOX: Concept Bottleneck YOLOX

Code for CB-YOLOX, an object detector whose classification branch is a
sequential **concept bottleneck**. For every anchor the head predicts K
human-interpretable concepts, and the class scores are a linear function of
those concepts:

```
features -> cls_convs -> g (1x1 conv) -> concept logits (K) -> sigmoid -> f (1x1 conv) -> class logits (80)
```

Box regression and objectness are standard YOLOX. Because the class
prediction goes only through the concepts, every detection can be explained
by its concept activations and corrected at test time by editing them
(concept interventions).

The repository holds the code for three steps:

| Folder | Contents |
|---|---|
| [`cbyolox/`](cbyolox/) | The model. It is a fork of [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) that adds the concept-bottleneck head, a concept loss (ASL or BCE), optional concept-aware SimOTA assignment, loading of concept labels, the CB-YOLOX-S/M/L experiments, a concept-aware demo and predictor, and a hyper-parameter search. |
| [`annotation/`](annotation/) | Concept ground truth for MS-COCO. Every instance is labelled with a concept vector, either by Qwen3-VL-8B-Instruct (a shared 143-concept vocabulary, or a per-class one of 80 x 5 descriptors) or by CLIP-descriptor similarity (per-class). The folder also has the concept vocabularies and dataset tools. |
| [`evaluation/`](evaluation/) | Detection mAP and concept mAP, concept faithfulness (deactivation, minimum corrections), mass test-time interventions, concept quality over training, per-class concept metrics, inference speed, a qualitative comparison, and dataset-level separability and Sankey analyses. |

Each folder has its own README with installation steps and exact commands.

## Pipeline

1. **Annotate** COCO 2017 train/val with concept vectors (`annotation/`).
   This gives standard COCO `instances_*.json` files in which every
   annotation has a `concept_vector` field.
2. **Train** CB-YOLOX (`cbyolox/`). The final models start from the official
   pretrained YOLOX-S/M/L checkpoints and fine-tune only the classification
   branch and the concept bottleneck (`exps/cb_yolox/cb_yolox_{s,m,l}.py`).
3. **Evaluate** (`evaluation/`) with the same exp file, checkpoint and
   annotation JSON used for training and validation.

## Quick start

```bash
pip install -e cbyolox            # installs the `yolox` package (CB-YOLOX fork)

# training (annotation JSONs placed as datasets/COCO/annotations/instances_{train,val}2017.json)
cd cbyolox
python -m yolox.tools.train -f exps/cb_yolox/cb_yolox_s.py -d 1 -b 16 -c /path/to/yolox_s.pth

# demo with a per-detection concept report
python tools/demo_cbm.py -f exps/cb_yolox/cb_yolox_s.py -c /path/to/cb_yolox_s.pth \
    --path assets/dog.jpg --concept_names ../annotation/concepts/shared_143.json --report
```

See `cbyolox/README.md` for every option, including the demo flags.

## Not included

COCO images and run outputs are not part of the repository. The trained
CB-YOLOX-S/M/L checkpoints and the concept labels of the final models are
released separately, the labels as compact concept packs (about 8.5 MB)
together with the vocabulary. Rebuild the training JSONs from them with
`annotation/release/merge_concepts.py` (see `annotation/README.md`).

## License and attribution

- `cbyolox/` is derived from Megvii YOLOX (Apache-2.0); see `cbyolox/LICENSE`.
- `annotation/clip/` is adapted from Menon & Vondrick, *Visual Classification
  via Description from Large Language Models* (ICLR 2023); see
  `annotation/clip/NOTICE`.
- Training augmentations use `albumentations` 2.0.8 (MIT).
