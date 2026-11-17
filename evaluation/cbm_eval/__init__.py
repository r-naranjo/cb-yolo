"""Shared helpers for the CB-YOLOX evaluation scripts.

Modules:
    coco        COCO category-id <-> contiguous class-index mapping, class names
    concepts    concept-name loading (shared 143 vocabulary / per-class 400 vocabulary)
    io          annotation loading, JSON/CSV writers, bootstrap helper
    matching    GT <-> detection matching through `final_anchor_idxs`
    metrics     per-concept AUC/AP/recall, COCO bbox evaluation
    model       experiment/checkpoint loading, CBMPredictor construction, inference
    linear_head closed-form re-evaluation of the concept_to_cls (f) layer

Only `model` imports `yolox` (lazily, inside functions), so the dataset-only
scripts and the pure metric helpers work without the model package.
"""
