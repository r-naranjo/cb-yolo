"""Label COCO instances with the shared 143-concept vocabulary using Qwen3-VL-8B-Instruct.

For every annotation the image is cropped to its bbox (optionally downscaled)
and, for each concept chunk of the vocabulary file, Qwen is asked which of the
chunk's concepts are visible. Answers are parsed with ``concept_present`` and
written as a binary ``concept_vector`` (length = number of concepts) into the
annotation. Output is the input COCO JSON with ``concept_vector`` added.

The released annotations were produced with
``--descriptors concepts/shared_143_chunked.json --downscale_res 256``
(train2017 / val2017, 2 shards each). ``--downscale_res 0`` disables the crop
downscaling (earlier, non-downscaled variant of the pipeline).

Work is split twice: ``--split_total/--split_idx`` selects a contiguous slice
of the (image_id-sorted) annotations for this job, and the slice is further
divided across ``--num_gpus`` local GPUs. Each job writes
``<out_dir>/<annotations_basename>_part<split_idx+1>of<split_total>_concepts.json``;
combine the shards with ``merge_shards.py``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_utils import (  # noqa: E402
    coco_lookups, concept_present, crop_bbox, generate, is_degenerate_bbox, load_json, load_qwen,
    pop_status_flags, print_summary, run_on_gpus, save_json, shard_slice, split_for_gpus, user_message,
)

GENERATION_KWARGS = dict(max_new_tokens=200, do_sample=False, temperature=None, top_p=None, top_k=None)


def build_prompt(cat_name, chunk_concepts):
    k = len(chunk_concepts)
    concepts_text = "\n".join([f"- {c}" for c in chunk_concepts])
    return (
        f"Analyze this cropped image of a '{cat_name}'. Review the following {k} concepts:\n"
        f"{concepts_text}\n\n"
        "TASK:\n"
        "Identify which of these exact concepts are visually present in the image.\n\n"
        "RULES:\n"
        f"1. STRICTLY VISUAL: Use ONLY the visual information in the image. Do NOT infer, guess, or rely on general knowledge about what a '{cat_name}' usually has. If it is not clearly in the pixels, it is absent.\n"
        "2. UNAMBIGUOUS EVIDENCE: If there is clear visual evidence (even if partially occluded), include it. If the feature is ambiguous, low-resolution, completely hidden, or you are uncertain, EXCLUDE it.\n"
        "3. EXACT MATCH: Only output concepts that exactly match the names provided in the list above.\n\n"
        "OUTPUT FORMAT:\n"
        "Respond ONLY with a single comma-separated list of the concepts that are present. Do not include any conversational text, explanations, or markdown. If none of the concepts apply, output exactly and only: none"
    )


def load_chunks(path):
    """Read a ``{"concepts": {name: [concepts...]}}`` file.

    Returns ``(chunks, global_concepts)``; the global index of a concept is its
    position in the concatenation of the chunks in file order.
    """
    data = load_json(path)["concepts"]
    if isinstance(data, list):  # flat file: one single chunk
        data = {"all": data}
    chunks, global_concepts = [], []
    for concepts in data.values():
        chunks.append(concepts)
        global_concepts.extend(concepts)
    return chunks, global_concepts


def parse_chunk_answer(text, chunk_concepts, concept_to_idx, vector):
    """Set ``vector[idx] = 1`` for every concept of the chunk named in ``text``."""
    for concept in chunk_concepts:
        if concept_present(concept, text):
            vector[concept_to_idx[concept]] = 1
    return vector


class CocoCropDataset:
    """Yields ``(crop or None, ann_id, category_name, is_invalid)`` per annotation."""

    def __init__(self, annotations, images_dir, cat_id_to_name, img_id_to_filename, downscale_res=256):
        self.annotations = annotations
        self.images_dir = images_dir
        self.cat_id_to_name = cat_id_to_name
        self.img_id_to_filename = img_id_to_filename
        self.downscale_res = downscale_res

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        from PIL import Image

        ann = self.annotations[idx]
        cat_name = self.cat_id_to_name.get(ann['category_id'], "object")
        filename = self.img_id_to_filename.get(ann['image_id'])

        if is_degenerate_bbox(ann['bbox']):
            return None, ann['id'], cat_name, True

        img_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            cropped_image = crop_bbox(image, ann['bbox'])
            if self.downscale_res:
                # Keeps aspect ratio; only shrinks crops larger than downscale_res.
                cropped_image.thumbnail((self.downscale_res, self.downscale_res), Image.Resampling.LANCZOS)
        except Exception:
            # Unreadable image: handled like a degenerate box (all-zero vector).
            return None, ann['id'], cat_name, True

        return cropped_image, ann['id'], cat_name, False


def collate_fn(batch):
    images, ann_ids, cat_names, invalid = [], [], [], []
    for image, ann_id, cat_name, is_invalid in batch:
        images.append(image)
        ann_ids.append(ann_id)
        cat_names.append(cat_name)
        invalid.append(is_invalid)
    return images, ann_ids, cat_names, invalid


def gpu_worker(gpu_id, worker_annotations, tmp_path, args, chunks, concept_to_idx, total_k,
               cat_id_to_name, img_id_to_filename):
    from torch.utils.data import DataLoader

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Qwen3-VL on {device}...")
    model, processor = load_qwen(device)

    dataset = CocoCropDataset(worker_annotations, args.images_dir, cat_id_to_name, img_id_to_filename,
                              args.downscale_res)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    print(f"[GPU {gpu_id}] Processing {len(worker_annotations)} instances...")
    ann_dict = {ann['id']: ann for ann in worker_annotations}
    processed_count = 0

    for images, ann_ids, cat_names, invalid in dataloader:
        for i, ann_id in enumerate(ann_ids):
            ann_dict[ann_id]['concept_vector'] = [0] * total_k
            if invalid[i]:
                ann_dict[ann_id]['invalid_bbox_handled'] = True
            else:
                ann_dict[ann_id]['successfully_labeled'] = True

        valid_idx = [i for i, inv in enumerate(invalid) if not inv]
        if valid_idx:
            for chunk_idx, chunk_concepts in enumerate(chunks):
                messages = [user_message(images[i], build_prompt(cat_names[i], chunk_concepts)) for i in valid_idx]
                try:
                    texts = generate(model, processor, messages, **GENERATION_KWARGS)
                    for txt, i in zip(texts, valid_idx):
                        parse_chunk_answer(txt, chunk_concepts, concept_to_idx, ann_dict[ann_ids[i]]['concept_vector'])
                except Exception as e:
                    # The concepts of a failed chunk stay 0; the instance is counted as failed.
                    print(f"[GPU {gpu_id}] Batch inference failed on chunk {chunk_idx}: {e}")
                    for i in valid_idx:
                        ann_dict[ann_ids[i]]['successfully_labeled'] = False

        processed_count += len(ann_ids)
        if processed_count % 500 == 0 or processed_count == len(worker_annotations):
            print(f"[GPU {gpu_id}] Progress: {processed_count}/{len(worker_annotations)}")

    save_json(list(ann_dict.values()), tmp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_dir", required=True, help="COCO image folder of the split (e.g. datasets/COCO/train2017).")
    parser.add_argument("--annotations", required=True, help="COCO instances JSON of the split.")
    parser.add_argument("--descriptors", required=True,
                        help="Chunked concept vocabulary, e.g. concepts/shared_143_chunked.json.")
    parser.add_argument("--out_dir", default="outputs", help="Output folder (default: outputs).")
    parser.add_argument("--num_gpus", type=int, default=2, help="Local GPUs used by this job (default: 2).")
    parser.add_argument("--batch_size", type=int, default=32, help="Instances per generate() call (default: 32).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU (default: 4).")
    parser.add_argument("--downscale_res", type=int, default=256,
                        help="Shrink crops so that max(w, h) <= this value; 0 disables (default: 256).")
    parser.add_argument("--split_total", type=int, default=1, help="Number of jobs the split is divided into.")
    parser.add_argument("--split_idx", type=int, default=0, help="0-based slice processed by this job.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    chunks, global_concepts = load_chunks(args.descriptors)
    total_k = len(global_concepts)
    concept_to_idx = {c: i for i, c in enumerate(global_concepts)}
    print(f"Loaded {len(chunks)} chunks containing {total_k} concepts.")

    coco_data = load_json(args.annotations)
    cat_id_to_name, img_id_to_filename = coco_lookups(coco_data)
    all_annotations = coco_data.get('annotations', [])
    all_annotations.sort(key=lambda x: x['image_id'])

    annotations, start, end = shard_slice(all_annotations, args.split_total, args.split_idx)
    if args.split_total > 1:
        print(f"Job {args.split_idx + 1}/{args.split_total}: annotations [{start}, {end})")

    gpu_chunks = split_for_gpus(annotations, args.num_gpus)
    print(f"Processing {len(annotations)} instances on {args.num_gpus} GPUs.")
    start_time = time.perf_counter()

    tmp_paths = [os.path.join(args.out_dir, f"tmp_worker_{g}_part{args.split_idx}.json") for g in range(args.num_gpus)]
    merged = run_on_gpus(gpu_worker, gpu_chunks,
                         (args, chunks, concept_to_idx, total_k, cat_id_to_name, img_id_to_filename), tmp_paths)
    labeled, invalid = pop_status_flags(merged)
    coco_data['annotations'] = merged

    base_name = os.path.basename(args.annotations).split('.')[0]
    output_path = os.path.join(args.out_dir, f"{base_name}_part{args.split_idx + 1}of{args.split_total}_concepts.json")
    save_json(coco_data, output_path, indent=4)
    print_summary(len(merged), labeled, invalid, start_time, output_path)


if __name__ == "__main__":
    main()
