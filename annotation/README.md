# Concept annotation of MS-COCO

Scripts that produce the concept ground truth used to train and evaluate
CB-YOLOX. Every COCO object instance (annotation) gets a `concept_vector`,
obtained either from the Qwen3-VL-8B-Instruct vision-language model (main
pipeline) or from CLIP similarity to GPT-written class descriptors.

The concept labels used for the final CB-YOLOX models are released
separately as compact *concept packs* (about 8 MB for train2017), together
with the vocabulary. `release/merge_concepts.py` combines a pack with the
official COCO 2017 annotations to rebuild the exact training JSON; see
[Released labels](#released-labels).

## Layout

```
annotation/
├── concepts/            concept vocabularies (see below)
├── common/              shared code: COCO class list, JSON / vocabulary helpers
├── qwen/
│   ├── annotate_shared.py          Qwen labelling, shared 143-concept vocabulary (main pipeline)
│   ├── annotate_per_class.py       Qwen labelling, per-class 80 x 5 = 400 vocabulary
│   ├── merge_shards.py             merge the N job shards of annotate_shared.py
│   ├── postprocess_concepts.py     zero / conditionally zero / restore concept columns
│   ├── validate.py                 integrity checks (vector shape/values, 2014->2017 redistribution)
│   ├── visualize.py                visual QA figures (--mode shared | per_class)
│   ├── redistribute_2014_to_2017.py  move 2014-labelled annotations onto the 2017 split
│   └── qwen_utils.py               model loading, generation, cropping, parsing, GPU/job splitting
├── clip/                CLIP + GPT-descriptor labelling (fork of Menon & Vondrick, see clip/NOTICE)
│   ├── generate_descriptors.py     GPT descriptors per class
│   ├── main_coco.py                CLI: evaluate | label_instances | calc_stats | visualize | misclassification
│   ├── load_coco.py                COCO crop dataset, scoring and labelling functions
│   ├── loading_helpers.py          descriptor loading / prompt formatting
│   └── binarize_concepts.py        raw CLIP similarities -> binary concept_vector
├── dataset_tools/
│   ├── split_train_val.py          split a COCO JSON into train/val by image
│   ├── make_balanced_subset.py     class-balanced X% subset
│   ├── count_instances.py          image / instance counts, per-class table and bar plot
│   └── concept_activation_stats.py average / median number of active concepts per class
├── release/
│   ├── concept_pack.py             concept-pack format (export / merge / verify)
│   ├── export_concepts.py          annotated COCO JSON -> concept pack (.npz), verified
│   └── merge_concepts.py           official COCO JSON + concept pack -> annotated JSON
└── requirements.txt
```

Every script is run directly (`python qwen/annotate_shared.py ...`); no
installation of this folder is needed. Commands below are run from
`annotation/`.

## Released labels

The released data contains only what this work adds to COCO:

| File | Content |
|---|---|
| `qwen_shared143_train2017.npz` | concept labels of all 860,001 train2017 annotations (Qwen3-VL-8B-Instruct, shared 143-concept vocabulary, post-processed) |
| `qwen_shared143_val2017.npz` | same for the 36,781 val2017 annotations |
| `vocabulary_shared143.json` | concept names in index order, the prompt chunks used for labelling, semantic groups, and the post-processing applied |

These are the labels the final CB-YOLOX-S/M/L models were trained and
validated on. To rebuild the training JSONs, download the official
`annotations_trainval2017.zip` from cocodataset.org and run:

```bash
python release/merge_concepts.py --coco annotations/instances_train2017.json \
    --pack qwen_shared143_train2017.npz --output ../cbyolox/datasets/COCO/annotations/instances_train2017.json
python release/merge_concepts.py --coco annotations/instances_val2017.json \
    --pack qwen_shared143_val2017.npz --output ../cbyolox/datasets/COCO/annotations/instances_val2017.json
```

The result is identical to the JSON files used for training. The merge
needs about 10 GB of RAM for train2017.

A concept pack (`release/concept_pack.py`) is a `.npz` file with the
annotation ids in training order, a mask of labelled annotations, the concept
vectors (bit-packed when binary) and a JSON metadata string. To export a new
pack from an annotated JSON:

```bash
python release/export_concepts.py --input annotated_train2017.json \
    --coco annotations/instances_train2017.json --output my_pack.npz
```

With `--coco`, the export rebuilds the JSON from the official file and fails
if any annotation, concept vector or used image field differs. Other
metadata, such as `info` or `coco_url`, is reported and replaced by the
official COCO 2017 values in the merge.

## Concept vocabularies

| File | Content | Used by |
|---|---|---|
| `concepts/shared_143.json` | `{"concepts": [143 names]}`, flat list; defines the concept index (0..142) | models with `num_concepts=143`, evaluation, `postprocess_concepts.py` |
| `concepts/shared_143_chunked.json` | same 143 concepts in the same order, split into 10 chunks (`chunk0..chunk9`) | `annotate_shared.py` (one prompt per chunk) |
| `concepts/shared_143_grouped.json` | same 143 concepts in the same order, in 6 semantic groups (morphology, object_parts, texture_and_material, shape_and_geometry, patterns_and_markings, spatial_and_visual_context) | reference / prompt experiments |
| `concepts/per_class_descriptors_10.json` | `{class: [10 descriptors]}` for the 80 COCO classes, as returned by GPT | input to the top-5 selection |
| `concepts/per_class_descriptors_top5.json` | `{class: [5 descriptors]}`: the 400-concept vocabulary, selected from the 10 (for 40 of the 80 classes at least one descriptor was edited or merged by hand) | `annotate_per_class.py`, `clip/main_coco.py` |
| `concepts/per_class_400_list.json` | `{"concepts": [400 strings]}`: the top-5 descriptors flattened in COCO class order (slot `5*c + j`); 368 distinct strings, since some descriptors repeat across classes | evaluation scripts that need one name per slot |

**Shared vocabulary (143).** Concepts shared by all classes (e.g. `has_legs`,
`texture_furry`, `is_being_held_by_hand`). Every concept is evaluated for
every instance, so vectors are dense binary vectors of length 143.

**Per-class vocabulary (400).** 5 descriptors for each of the 80 classes.
Class `c` (its position in `common/coco_classes.py`, the official COCO
category order) owns slots `[5c, 5c+5)`; only the slots of the instance's
own class can be non-zero.

### Renamed files

The vocabulary files were renamed for the release; contents are unchanged
(byte-identical). Configs and checkpoints may still mention the old names:

| Old name | New name |
|---|---|
| `coco_concepts.json` | `concepts/shared_143.json` |
| `coco_concepts_cat.json` | `concepts/shared_143_grouped.json` |
| `coco_concepts_chuncked.json` | `concepts/shared_143_chunked.json` |
| per-class descriptors, 10 per class | `concepts/per_class_descriptors_10.json` |
| per-class descriptors, top 5 per class | `concepts/per_class_descriptors_top5.json` |
| per-class descriptors, top 5, flattened list | `concepts/per_class_400_list.json` |

## Output format

Each output is the input COCO `instances_*.json` (`images`, `annotations`,
`categories`, ...) unchanged except that every annotation gets

```json
"concept_vector": [0, 1, 0, ...]
```

a list of length K (143 or 400). Final files are binary (0/1). The
per-class Qwen labeller can also write ranks (`--bin_rank rank`, values 1..5
in the class slots), and `clip/main_coco.py --process label_instances` writes
raw CLIP similarities (with `-1` markers) that must be binarized with
`clip/binarize_concepts.py`. CLIP outputs additionally contain a top-level
`concept_names` list (one string per slot).

The released annotation files were named as follows:

| File | Vocabulary | Produced by |
|---|---|---|
| `final_{train,val}2017_binary_annotations_with_global_concepts.json` | shared 143 | `annotate_shared.py` + `merge_shards.py` |
| `final_{train,val}2017_binary_annotations_with_global_concepts_post_processed.json` | shared 143 | + `postprocess_concepts.py --preset released` (used for training/evaluation) |
| `final_train2017_binary_annotations_with_global_concepts_post_processed_subset.json` | shared 143 | + `make_balanced_subset.py --percentage 10` (quick experiments only) |
| `final_{train,val}2014_{binary,ranked}_annotations_with_concepts.json` | per-class 400 | `annotate_per_class.py` on COCO 2014 |
| `final_{train,val}2017_{binary,ranked}_annotations_with_concepts.json` | per-class 400 | `redistribute_2014_to_2017.py` from the 2014 files |

## Installation

```bash
pip install torch torchvision   # pick the build for your CUDA version
pip install -r requirements.txt
```

`flash-attn` is only needed for the Qwen labellers and `openai` only for
`generate_descriptors.py`.

## Hardware notes (Qwen)

- Qwen3-VL-8B-Instruct is loaded in bfloat16 with FlashAttention-2, one full
  copy per GPU (about 17 GB of weights; the released labels were produced on
  80 GB H100s with `--batch_size 64`). Use `--num_gpus` to set the number of
  local GPUs; each runs one worker process on a contiguous slice of the data.
- The weights (~17 GB) are downloaded from the Hugging Face Hub on first use;
  set `HF_HOME` to a persistent cache directory shared by all jobs, e.g.
  `export HF_HOME=/path/to/hf_cache`.
- Labelling train2017 (~860k instances) with the shared vocabulary means 10
  generate calls per instance; split it over several jobs with
  `--split_total N --split_idx i`.

## Qwen: shared 143-concept vocabulary (main pipeline)

1. Label each split. Each job processes one contiguous slice of the
   annotations (sorted by `image_id`) and writes
   `<out_dir>/<annotations_basename>_part<i+1>of<N>_concepts.json`:

   ```bash
   for i in 0 1; do
     python qwen/annotate_shared.py \
         --images_dir datasets/COCO/train2017 \
         --annotations datasets/COCO/annotations/instances_train2017.json \
         --descriptors concepts/shared_143_chunked.json \
         --out_dir outputs --num_gpus 2 --batch_size 64 \
         --split_total 2 --split_idx $i      # one job per i, can run in parallel
   done
   ```

   For each crop (downscaled to at most 256 px per side, `--downscale_res 0`
   disables it) and each of the 10 chunks, Qwen is asked which of the
   chunk's concepts are visible; a concept is set to 1 if its exact name
   appears as a whole token in the answer. Degenerate boxes and unreadable
   images get an all-zero vector; if generation fails for a batch, the
   concepts of that chunk stay 0 (reported as failed in the summary).
   Generation is greedy (`do_sample=False`, `max_new_tokens=200`).

2. Merge the shards (in part order):

   ```bash
   python qwen/merge_shards.py \
       --parts outputs/instances_train2017_part1of2_concepts.json outputs/instances_train2017_part2of2_concepts.json \
       --output outputs/final_train2017_binary_annotations_with_global_concepts.json
   ```

3. Post-process (the released `_post_processed` files):

   ```bash
   python qwen/postprocess_concepts.py \
       --input  outputs/final_train2017_binary_annotations_with_global_concepts.json \
       --output outputs/final_train2017_binary_annotations_with_global_concepts_post_processed.json \
       --preset released
   ```

   `--preset released` sets to 0, for every instance, the concepts
   `color_multicolored` (113), `color_gradient` (114), `shadow_is_visible`
   (141) and `light_source_from_above` (142), and sets `has_legs` (0) to 0
   wherever `has_legs_furniture` (46) is 1. No other column is modified.
   This was verified column by column against the released train2017 and
   val2017 files. Generic use: `--zero C...`, `--zero-if TARGET:COND...`,
   `--restore C... --restore-from ORIGINAL.json`, with concept names
   (resolved through `--concepts concepts/shared_143.json`) or indices.

4. Validate and inspect:

   ```bash
   python qwen/validate.py vectors --json outputs/final_val2017_..._post_processed.json \
       --scheme shared_binary --num_concepts 143
   python qwen/visualize.py --mode shared --json outputs/final_val2017_..._post_processed.json \
       --images_dir datasets/COCO/val2017 --concepts concepts/shared_143.json --out_dir figs --num 5
   ```

## Qwen: per-class 400-concept vocabulary

```bash
python qwen/annotate_per_class.py \
    --images_dir datasets/COCO/train2014 \
    --annotations datasets/COCO/annotations/instances_train2014.json \
    --descriptors concepts/per_class_descriptors_top5.json \
    --bin_rank bin --num_gpus 2 \
    --output outputs/final_train2014_binary_annotations_with_concepts.json
```

One prompt per instance about the 5 descriptors of its class; the answer
(`[1, 0, 1, 1, 0]`, or a permutation of 1..5 with `--bin_rank rank`) fills
the class slots. Degenerate boxes get zeros (`bin`) or `[1, 2, 3, 4, 5]`
(`rank`). Annotations whose class is not in the descriptor file, whose image
cannot be read or whose batch fails are written without `concept_vector`
(behaviour of the original run, kept for reproducibility; check with
`validate.py`).

The released per-class files were labelled on COCO 2014 and moved onto the
2017 split:

```bash
python qwen/redistribute_2014_to_2017.py \
    --train2014 outputs/final_train2014_binary_annotations_with_concepts.json \
    --val2014   outputs/final_val2014_binary_annotations_with_concepts.json \
    --instances_train2017 datasets/COCO/annotations/instances_train2017.json \
    --instances_val2017   datasets/COCO/annotations/instances_val2017.json \
    --out_train outputs/final_train2017_binary_annotations_with_concepts.json \
    --out_val   outputs/final_val2017_binary_annotations_with_concepts.json
python qwen/validate.py vectors --json outputs/final_val2017_binary_annotations_with_concepts.json \
    --scheme per_class_binary --num_concepts 400        # per_class_ranked for --bin_rank rank files
python qwen/visualize.py --mode per_class --json outputs/final_val2017_binary_annotations_with_concepts.json \
    --images_dir datasets/COCO/val2017 --concepts concepts/per_class_descriptors_top5.json
```

`validate.py redistribution --train2014 ... --val2014 ... --train2017 ... --val2017 ...`
compares the pooled 2014 inputs and 2017 outputs by id (images whose id is in
neither 2017 split are dropped by the redistribution, and image `file_name`s
are rewritten to the 2017 `<id>.jpg` form, so these checks report the
differences).

## CLIP labelling (per-class vocabulary)

Adapted from Menon & Vondrick, *Visual Classification via Description from
Large Language Models*, ICLR 2023
([code](https://github.com/sachit-menon/classify_by_description_release));
see `clip/NOTICE`. Each crop is scored against the CLIP text embeddings of
`"a photo of <class>, which has <descriptor>."` prompts.

1. (Optional) generate descriptors; the key is read from the environment only:

   ```bash
   export OPENAI_API_KEY=...
   python clip/generate_descriptors.py --output outputs/descriptors_10.json
   ```

2. Raw concept vectors:

   ```bash
   python clip/main_coco.py --process label_instances \
       --descriptor_file concepts/per_class_descriptors_top5.json \
       --images_path datasets/COCO/train2017 \
       --annotations_path datasets/COCO/annotations/instances_train2017.json \
       --model_size ViT-B/32 --batch_size 64 \
       --output outputs/instances_train2017_clip_raw.json
   ```

   Classes are ranked by the mean descriptor similarity. If the true class is
   in the top 5 (Rule 1: top-1, Rule 2: top-5), its 5 slots get the raw
   similarities of its descriptors; otherwise (Rule 3) they are set to `-1`.

3. Binarize:

   ```bash
   python clip/binarize_concepts.py outputs/instances_train2017_clip_raw.json \
       outputs/instances_train2017_clip_binary.json
   ```

   Per class, the mean activation of each concept (ignoring `-1`) and the
   top-3 concepts are computed. Non-Rule-3 instances are forced to exactly 5
   positives and binarized with `activation > class mean`; Rule-3 instances
   get the class top-3; all-zero results also get the class top-3.

Other processes: `--process evaluate` (top-1/top-5 accuracy of
descriptor-based vs. plain CLIP), `calc_stats` (per-class Rule 1/2/3 table
and plot in `--out_dir`), `visualize --indices 0 10 ...` (crops and
explanations), `misclassification` (figure of a misclassified instance).

## Dataset tools

```bash
python dataset_tools/count_instances.py --json FILE.json [--per_class] [--plot counts.png]
python dataset_tools/concept_activation_stats.py --json FILE.json
python dataset_tools/make_balanced_subset.py --input FILE.json --output SUBSET.json --percentage 10 [--seed 0]
python dataset_tools/split_train_val.py --json instances_all.json --out_dir annotations --train_frac 0.7 --seed 42
```

`make_balanced_subset.py` selects whole images, rarest class first, until
each class reaches the target share of its instances. The released subset
was built unseeded and cannot be regenerated bit-exactly.
