# Evaluation

Evaluation and analysis scripts for CB-YOLOX checkpoints (YOLOX whose
classification branch is a sequential concept bottleneck:
`cls_convs -> concept_preds (K logits) -> sigmoid -> concept_to_cls -> class logits`)
and for the concept annotations themselves.

## Analyses

| Script | Needs model | What it reports |
|---|---|---|
| `concept_faithfulness/map_and_concept_map.py` | yes | COCO bbox mAP + per-concept AUC/AP/Recall@0.5 and concept mAP |
| `concept_faithfulness/concept_deactivation.py` | yes | Classification accuracy as the d most influential concepts are neutralised (knee / half-collapse / plateau) |
| `concept_faithfulness/concept_intervention_accuracy.py` | yes | Minimum number of oracle concept corrections that flips a confident misclassification (distribution) |
| `concept_intervention/intervene_and_evaluate.py` | yes | Class x concept activation matrix, concept-mismatch diagnosis, mass test-time intervention with mAP before/after |
| `concept_evolution/compute_concept_evolution.py` | yes (epoch checkpoints) | Per-concept AUC/AP/recall for every `epoch_<N>_ckpt.pth` of a run |
| `concept_evolution/plot_concept_evolution.py` | no | Plots + CSV from the evolution JSON |
| `per_class_concept_metrics/evaluate.py` | yes | Report for the per-class 400-concept vocabulary: bbox mAP, binary, ranked (top-5) and frequency metrics |
| `inference_speed/benchmark_inference_speed.py` | yes (synthetic input) | Latency of plain YOLOX vs. CB-YOLOX head, S/M/L, CPU/GPU |
| `qualitative/compare_yolox_cbyolox.py` | yes | Side-by-side detections of plain YOLOX vs. CB-YOLOX with top concepts per box |
| `dataset_analysis/separability.py` | no | Concept/class separability of an annotation file (mOverlap, mConceptsPerClass, mSeparability) |
| `dataset_analysis/plot_separability.py` | no | Separability vs. filter sweep, class x concept heatmaps, clustered heatmaps |
| `dataset_analysis/plot_concept_sankey.py` | no | Interactive concept -> category Sankey diagrams (HTML) |

`cbm_eval/` is the shared helper package (COCO id mapping, concept-name
loading, model/checkpoint loading, inference, GT <-> detection matching,
metrics). Hyper-parameter sweeps over the concept loss/assignment weights are
run with `cbyolox/tools/grid_search/`.

## Install

```bash
pip install -e cbyolox                 # provides the `yolox` package
pip install scikit-learn seaborn prettytable tqdm pycocotools plotly
```

`plotly` is only needed for the Sankey diagrams. Run every script from the
repository root as `python evaluation/<folder>/<script>.py ...` (each
script adds `evaluation/` to `sys.path` to import `cbm_eval`; no other path
setup is needed). All model-based scripts take `-f <exp file> -c <ckpt>`,
`--device {gpu,cpu}` (default `gpu`) and `--fp16/--no-fp16` (default on, GPU
only); `--limit N` evaluates only the first N images (smoke test).

## Shared conventions

- **COCO ids.** COCO category ids (1..90 with gaps) are converted to YOLOX
  class indices (0..79) with `cbm_eval.coco.CAT_TO_IDX` / `IDX_TO_CAT`, never
  with `category_id - 1`.
- **Annotations.** Concept ground truth is a COCO-style JSON where every
  annotation has a `concept_vector`. `--json_binary` (0/1 over the K
  concepts) is also the box/category GT for COCOeval. Only the per-class
  report additionally needs `--json_ranked` (rank 1..5 of the 5 most salient
  concepts of the annotation, 0 elsewhere), index-aligned with the binary
  file (same annotation ids in the same order). Only images that have
  annotations in the JSON are evaluated.
- **Use the annotation file the checkpoint was trained/validated on.**
  Different annotation variants (e.g. raw vs. post-processed concept
  vectors) are different ground truths with the same K; mixing them silently
  changes every concept metric.
- **Anchor matching.** Inference keeps, for every post-NMS detection, the
  index of its anchor (`scale_info[0]["final_anchor_idxs"]`) and FPN level
  (`scale_map`). Detection boxes are mapped back to original-image pixels
  (divided by `img_info["ratio"]`, clipped). Each GT box is matched to its
  highest-IoU detection with IoU >= `--iou_thresh` (default 0.5):
  same-class only for concept-quality metrics, any class for the
  intervention/faithfulness analyses (they need to see misclassified but
  well-localized objects). The anchor index selects the row of the flattened
  `(n_anchors, K)` concept-logit tensor. Unmatched GT boxes are skipped.
- **NMS and thresholds.** `map_and_concept_map.py`, `concept_deactivation.py`
  and `concept_intervention_accuracy.py` use per-class NMS (as the official
  YOLOX COCO evaluator) and default `--conf/--nms` to the experiment's
  `test_conf`/`nmsthre` (0.01/0.65); keep these defaults for mAP comparable
  with training-time validation. The other model-based scripts use
  `CBMPredictor.inference` (class-agnostic NMS) and default to
  `--conf 0.25 --nms 0.45`.
- **Concept names** (`--concepts`) come from `annotation/concepts/`:
  `shared_143.json` (flat list, also `shared_143_grouped.json`,
  `shared_143_chunked.json`) for the shared 143-concept vocabulary,
  `per_class_descriptors_top5.json` or `per_class_400_list.json` for the
  per-class 400-concept vocabulary (class c owns concepts 5c..5c+4). A bare
  JSON list or a `.txt` with one name per line also works. Where names are
  only labels, a length mismatch with the model falls back to `concept_<i>`.

## Usage

Examples use YOLOX-S; replace `s` by `m`/`l` for the other sizes. Paths are
examples (`datasets/COCO` layout of YOLOX, checkpoints in `YOLOX_outputs/`).

```bash
EXP=cbyolox/exps/cb_yolox/cb_yolox_s.py
CKPT=YOLOX_outputs/cb_yolox_s/best_ckpt.pth
IMG=datasets/COCO/val2017
VAL=datasets/COCO/annotations/instances_val2017.json   # with concept_vector
CONCEPTS=annotation/concepts/shared_143.json
```

### Detection mAP + concept mAP

```bash
python evaluation/concept_faithfulness/map_and_concept_map.py \
    -f $EXP -c $CKPT --img_dir $IMG --json_binary $VAL --concepts $CONCEPTS \
    --output_dir outputs/map_s
```

Outputs: `map_report_summary.json` (12 COCOeval stats; `concept_mAP`,
`concept_mean_auc`, `concept_mean_recall`, number of measurable concepts),
`per_concept_metrics.csv` (`concept, auc, ap, recall, n_pos`; empty if the
concept has fewer than `--auc_min_pos`, default 10, positives),
`concept_ap_distribution.png`.

### Accuracy under concept deactivation

For each matched instance (any class) the K concepts are ranked by their
influence on the GT-class logit, `|W_s[gt] * (p - 0.5)|`, and the d most
influential ones are set to 0.5 (= sigmoid(0)); class logits are recomputed
exactly from the `concept_to_cls` 1x1-conv weights (a per-anchor linear map),
so one inference pass suffices. Operating points: knee (first d with loss >
`--knee_thresh`), half-collapse (first d with loss >= half of the loss at
`--max_k`, default K) and plateau (first d after which at most
`--plateau_tol` accuracy is still lost).

```bash
python evaluation/concept_faithfulness/concept_deactivation.py \
    -f $EXP -c $CKPT --img_dir $IMG --json_binary $VAL --concepts $CONCEPTS \
    --output_dir outputs/deactivation_s
```

Outputs: `deactivation_curve.csv/.png` (accuracy and loss for d = 0..max_k),
`deactivation_summary.json` (baseline accuracy, total loss, the three
operating points), `most_influential_concept_counts.csv` (how often each
concept is the most influential one).

### Minimum concept corrections to flip a misclassification

Subset: matched (any class) instances with the wrong class and detection
score >= `--conf_thresh` (default 0.5). An oracle replaces concept
probabilities with their GT binary values; because `concept_to_cls` is
linear, the smallest correcting set is the prefix of concepts sorted by
their individual margin reduction (closed form). Each minimal set is checked
against the arg-max over all classes; instances not flippable with all K
concepts are counted separately.

```bash
python evaluation/concept_faithfulness/concept_intervention_accuracy.py \
    -f $EXP -c $CKPT --img_dir $IMG --json_binary $VAL --concepts $CONCEPTS \
    --output_dir outputs/intervention_accuracy_s
```

Outputs: `intervention_accuracy_summary.json` (subset size,
flippable/unflippable counts, mean/median/p25/p75/p90/max of the minimum,
`P(min <= k)` for `--report_at_k`, default 1 3 5 10, and the number of
flips not confirmed by the arg-max check), `intervention_log.json` (one
entry per instance), `intervention_accuracy.png` (histogram + CDF).

These two faithfulness numbers are measured at the output of
`concept_to_cls` for the matched anchor (no second NMS pass), so they are
not directly comparable with the mAP deltas of the next analysis.

### Concept intervention with mAP before/after

1. Class x concept matrix: mean `sigmoid(concept_logits)` of matched
   instances (any class) per GT class.
2. Misclassified instances whose top `--top_k` (default 3) concepts are
   disjoint from their GT-active concepts.
3. For those, the top-k concepts are set to `--deactivate_value` (raw
   logit, default 0.0) and the top-k concepts of the GT-class row of the
   matrix receive the original logits of the deactivated ones
   (`head.set_interventions`).
4. A second inference pass with all interventions applied; COCOeval before
   and after.

```bash
python evaluation/concept_intervention/intervene_and_evaluate.py \
    -f $EXP -c $CKPT --img_dir $IMG --json_binary $VAL --concepts $CONCEPTS \
    --output_dir outputs/intervention_s
```

Outputs: `concept_class_matrix.csv` / `_heatmap.png`, `intervention_log.json`,
`intervention_summary.json` (step counts, COCOeval stats before/after).
`--log_file FILE` writes the verbose per-detection report of both passes.

### Concept quality over training

```bash
python evaluation/concept_evolution/compute_concept_evolution.py \
    -f $EXP --ckpt_dir YOLOX_outputs/cb_yolox_s --img_dir $IMG --json_binary $VAL \
    --concepts $CONCEPTS --output outputs/concept_evolution_s.json
python evaluation/concept_evolution/plot_concept_evolution.py \
    --results outputs/concept_evolution_s.json --output_dir outputs/concept_evolution_s
```

The sweep needs the per-epoch checkpoints `epoch_<N>_ckpt.pth` (written when `exp.save_history_ckpt` is True, the default).
Plots: `01_mean_auc_ap.png` (mean AUC/AP/recall with IQR per epoch),
`02_concept_coverage.png` (concepts with AUC >= `--auc_learned` 0.70 vs.
< `--auc_random` 0.60), `03_per_concept_heatmap.png` (AUC per concept and
epoch), `04_auc_vs_ap_final.png`; plus `concept_metrics_by_epoch.csv`.

### Per-class (400-concept) report

```bash
python evaluation/per_class_concept_metrics/evaluate.py \
    -f <exp with num_concepts=400> -c <ckpt> --img_dir $IMG \
    --json_binary <binary json> --json_ranked <ranked json> \
    --concepts annotation/concepts/per_class_descriptors_top5.json \
    --output_dir outputs/per_class
```

Binary metrics: precision/recall/F1 (threshold 0.5) and ROC-AUC over all
(instance, concept) pairs, globally and per class, plus a threshold sweep.
Ranked metrics (top-5 predicted vs. GT ranked concepts): average
intersection, exact-set match rate, rank MAE, exact order at top-1/2/3/5
(as % of exact-set matches). Frequency analysis: Pearson correlation of GT
vs. predicted activation counts per concept, per-class bar plots of the 5
concepts owned by the class. Outputs: `summary.json`, global plots in the
output directory, per-class plots in `metrics_by_class/<class>/` and
`<class>/`, `detection_instance_counts.png`.

### Inference speed

```bash
python evaluation/inference_speed/benchmark_inference_speed.py \
    --exp_file_template "cbyolox/exps/cb_yolox/cb_yolox_{size}.py" \
    --ckpt_template "YOLOX_outputs/cb_yolox_{size}/best_ckpt.pth" \
    --sizes s m l --devices cpu gpu --output_dir outputs/inference_speed
```

Both variants are built from the same experiment and checkpoint: `cbm` as
trained, `normal` with `num_concepts=None` (plain `cls_preds` head, weights
loaded by name/shape). Only `model(x)` (backbone, FPN, head, box decoding)
on a random input is timed, both variants interleaved; Conv+BN fusion,
channels-last (GPU) and TF32 are configurable. Outputs:
`benchmark_results.csv/.json` (mean/std/p50/p95 ms, FPS per size, device,
precision, variant), `cbm_overhead_summary.csv`, `benchmark_latency_<device>.png`.

### Qualitative comparison

```bash
python evaluation/qualitative/compare_yolox_cbyolox.py \
    --exp_file_template "cbyolox/exps/cb_yolox/cb_yolox_{size}.py" \
    --ckpt_template "YOLOX_outputs/cb_yolox_{size}/best_ckpt.pth" \
    --image_dir $IMG --num_samples 20 --concepts $CONCEPTS --output_dir outputs/qualitative
```

The plain-YOLOX variant uses `--normal_ckpt_template` if given (a checkpoint
trained with the plain head), otherwise the plain `cls_preds` weights stored
in the CB-YOLOX checkpoint. Outputs per size: annotated images of both
variants, `comparison/` side-by-side images with the top
`--top_k_concepts` concepts per detection, `detections.json`, `summary.csv`.

### Dataset analysis (no model)

```bash
python evaluation/dataset_analysis/separability.py \
    --annotations datasets/COCO/annotations/instances_train2017.json --threshold 0.25 0.5 0.75
python evaluation/dataset_analysis/plot_separability.py sweep \
    --annotations datasets/COCO/annotations/instances_train2017.json --output_dir outputs/separability
python evaluation/dataset_analysis/plot_separability.py heatmaps \
    --annotations datasets/COCO/annotations/instances_train2017.json --concepts $CONCEPTS \
    --barplot_class person --output_dir outputs/separability
python evaluation/dataset_analysis/plot_concept_sankey.py \
    --annotations $VAL --concepts $CONCEPTS --min_flow 5 --output outputs/sankey/concept_sankey
```

Separability: for every class keep the concepts active in at least
`threshold` of its annotations; `mOverlap` = mean over all K concepts of
(number of classes using the concept - 1), `mConceptsPerClass` = mean
concepts per class, `mSeparability = 1 - mOverlap / (mConceptsPerClass - 1)`
(1 = every concept belongs to a single class; undefined if
`mConceptsPerClass <= 1`). Concepts used by no class count as -1 in
`mOverlap`. `plot_separability.py clustered` draws the filtered class x
concept matrix with both axes hierarchically clustered.

Sankey: edge weight = annotations of the category with the concept active;
an edge is kept if it covers at least `--min_flow` % of the category's
annotations. Writes `<output>_all.html`, `<output>_supercategories.html` and
one `<output>_<supercategory>.html` per supercategory.

## Notes on the original implementation

The per-class report and the concept-evolution sweep originally matched GT
boxes against detection boxes in the resized network-input frame (and the
per-class report also exported those boxes to COCOeval) without dividing by
`img_info["ratio"]`. This is wrong for every image whose longer side is not
640 px. Both scripts now rescale like all other scripts; pass
`--legacy_box_scale` to reproduce numbers produced with the original code.

## File map

```
evaluation/
  cbm_eval/                  shared helpers (coco, concepts, io, matching, metrics, model, linear_head)
  concept_faithfulness/      map_and_concept_map.py, concept_deactivation.py, concept_intervention_accuracy.py
  concept_intervention/      intervene_and_evaluate.py
  concept_evolution/         compute_concept_evolution.py, plot_concept_evolution.py
  per_class_concept_metrics/ evaluate.py, metrics_binary.py, metrics_ranked.py, metrics_distribution.py
  inference_speed/           benchmark_inference_speed.py
  qualitative/               compare_yolox_cbyolox.py
  dataset_analysis/          separability.py, plot_separability.py, plot_concept_sankey.py
```
