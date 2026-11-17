"""Label COCO instances with the per-class vocabulary (80 classes x 5 descriptors) using Qwen3-VL-8B-Instruct.

For every annotation the image is cropped to its bbox and Qwen is asked about
the 5 descriptors of the instance's own class only. The answer (a 5-element
list) is written into slots ``[5*c, 5*c + 5)`` of a 400-long ``concept_vector``,
where ``c`` is the class position in ``common.coco_classes.COCO_CLASSES``; all
other slots are 0.

``--bin_rank bin``: 0/1 presence of each descriptor.
``--bin_rank rank``: permutation of 1..5 (1 = most important for the class decision).

Degenerate boxes (int width or height <= 0) get zeros (``bin``) or
``[1, 2, 3, 4, 5]`` (``rank``) in their class slots. Annotations whose class
is missing from the descriptor file, whose image cannot be read, or whose
batch fails during generation are written *without* a ``concept_vector``
field (behaviour of the original labeller, kept for reproducibility).

The released per-class annotations were produced with
``--descriptors concepts/per_class_descriptors_top5.json`` on COCO 2014 and
then re-bucketed onto the 2017 split with ``redistribute_2014_to_2017.py``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_utils import (  # noqa: E402
    coco_lookups, crop_bbox, extract_vector, generate, is_degenerate_bbox, load_json, load_qwen,
    pop_status_flags, print_summary, run_on_gpus, save_json, split_for_gpus, user_message,
)
from common.coco_classes import COCO_CLASSES, DESCRIPTORS_PER_CLASS  # noqa: E402

VECTOR_LEN = len(COCO_CLASSES) * DESCRIPTORS_PER_CLASS  # 400
GENERATION_KWARGS = dict(max_new_tokens=20, do_sample=False)


def build_prompt(desc, bin_rank_mode):
    """Prompt for the 5 descriptors ``desc`` of one class (``bin`` or ``rank`` mode)."""
    if bin_rank_mode == "bin":
        return (
            f"You are shown a single cropped image (attached). Analyze the image and decide whether the following 5 features are visibly present.\n\n"
            f"1. {desc[0]}\n2. {desc[1]}\n3. {desc[2]}\n4. {desc[3]}\n5. {desc[4]}\n\n"
            "Rules (read carefully):\n"
            "- Use ONLY the visual information in the image. Do NOT infer, guess, or rely on general knowledge. If a feature is not clearly visible in the pixels, it must be considered absent.\n"
            "- If there is any unambiguous visual evidence (even partial or occluded) that the feature appears, mark it as present (1). If the evidence is ambiguous, low-resolution, or not visible, mark it as absent (0).\n"
            "- If you are uncertain, choose 0.\n\n"
            "Output requirement: Respond with exactly one line containing only a Python list of five integers (0 or 1), corresponding in order to the five descriptors above. The output must match a five-element binary list format (for example: [1, 0, 1, 1, 0]). Do NOT include any other text, explanation, labels, or punctuation.\n\n"
            "Example exact output: [1, 0, 1, 1, 0]"
        )
    if bin_rank_mode == "rank":
        return (
            f"You are shown a single cropped image (attached). Analyze the image and rank the following 5 features by how important each feature would be when deciding the image's class.\n\n"
            f"1. {desc[0]}\n2. {desc[1]}\n3. {desc[2]}\n4. {desc[3]}\n5. {desc[4]}\n\n"
            "Ranking rules (read carefully):\n"
            "- Use ONLY the visual information in the image. Do NOT infer, guess, or rely on general knowledge. If a feature is not clearly visible in the pixels, treat it as less important.\n"
            "- Rank values: 1 = most important for the class decision, 5 = least important.\n"
            "- Produce a permutation of the integers 1 through 5 (each integer 1,2,3,4,5 must appear exactly once). No duplicates, no missing numbers.\n"
            "- If two or more features appear equally visible, break ties using visual salience (visible area first, then contrast/clarity, then centrality). If still ambiguous, choose an arbitrary consistent order but keep the permutation constraint.\n"
            "- If you are uncertain about a feature's visibility, assign it a lower importance (a larger rank number, closer to 5).\n\n"
            "Output requirement: Respond with exactly one line containing only a Python list of five integers (a permutation of 1..5) corresponding in order to the five descriptors above. The output must contain nothing else (no text, labels, explanation). Example exact output: [2, 4, 1, 3, 5]\n\n"
            "Example exact output: [2, 4, 1, 3, 5]"
        )
    raise ValueError(f"Unknown mode {bin_rank_mode!r} (expected 'bin' or 'rank').")


def class_slot(cat_name):
    return COCO_CLASSES.index(cat_name) * DESCRIPTORS_PER_CLASS


class CocoCropDataset:
    """Yields ``(message or None, ann_id, concept_vector, cat_name, is_invalid_bbox)``."""

    def __init__(self, annotations, images_dir, descriptors_data, cat_id_to_name, img_id_to_filename, bin_rank_mode):
        self.annotations = annotations
        self.images_dir = images_dir
        self.descriptors_data = descriptors_data
        self.cat_id_to_name = cat_id_to_name
        self.img_id_to_filename = img_id_to_filename
        self.bin_rank_mode = bin_rank_mode

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        from PIL import Image

        ann = self.annotations[idx]
        cat_name = self.cat_id_to_name.get(ann['category_id'])
        filename = self.img_id_to_filename.get(ann['image_id'])
        concept_vector = [0] * VECTOR_LEN

        if cat_name not in self.descriptors_data or cat_name not in COCO_CLASSES:
            return None, ann['id'], concept_vector, cat_name, False

        start_pos = class_slot(cat_name)

        if is_degenerate_bbox(ann['bbox']):
            if self.bin_rank_mode == "bin":
                concept_vector[start_pos:start_pos + 5] = [0, 0, 0, 0, 0]
            elif self.bin_rank_mode == "rank":
                concept_vector[start_pos:start_pos + 5] = [1, 2, 3, 4, 5]
            return None, ann['id'], concept_vector, cat_name, True

        img_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None, ann['id'], concept_vector, cat_name, False

        cropped_image = crop_bbox(image, ann['bbox'])
        prompt_text = build_prompt(self.descriptors_data[cat_name], self.bin_rank_mode)
        return user_message(cropped_image, prompt_text), ann['id'], concept_vector, cat_name, False


def collate_fn(batch):
    messages, ann_ids, vectors, cat_names, invalid, has_message = [], [], [], [], [], []
    for message, ann_id, concept_vector, cat_name, is_invalid in batch:
        if message is not None:
            messages.append(message)
        has_message.append(message is not None)
        ann_ids.append(ann_id)
        vectors.append(concept_vector)
        cat_names.append(cat_name)
        invalid.append(is_invalid)
    return messages, ann_ids, vectors, cat_names, invalid, has_message


def gpu_worker(gpu_id, worker_annotations, tmp_path, args, descriptors_data, cat_id_to_name, img_id_to_filename):
    from torch.utils.data import DataLoader

    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading Qwen3-VL on {device}...")
    model, processor = load_qwen(device)

    dataset = CocoCropDataset(worker_annotations, args.images_dir, descriptors_data, cat_id_to_name,
                              img_id_to_filename, args.bin_rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, prefetch_factor=2 if args.num_workers > 0 else None)

    print(f"[GPU {gpu_id}] Processing {len(worker_annotations)} instances...")
    ann_dict = {ann['id']: ann for ann in worker_annotations}
    processed_count = 0

    for messages, ann_ids, vectors, cat_names, invalid, has_message in dataloader:
        output_texts = []
        batch_success = True
        if messages:
            try:
                output_texts = generate(model, processor, messages, **GENERATION_KWARGS)
            except Exception as e:
                print(f"[GPU {gpu_id}] Batch inference failed: {e}")
                batch_success = False

        msg_idx = 0
        for i, ann_id in enumerate(ann_ids):
            if invalid[i]:
                ann_dict[ann_id]['concept_vector'] = vectors[i]
                ann_dict[ann_id]['invalid_bbox_handled'] = True
            elif has_message[i]:
                if batch_success and msg_idx < len(output_texts):
                    start_pos = class_slot(cat_names[i])
                    concept_vector = vectors[i]
                    concept_vector[start_pos:start_pos + 5] = extract_vector(output_texts[msg_idx].strip())
                    ann_dict[ann_id]['concept_vector'] = concept_vector
                    ann_dict[ann_id]['successfully_labeled'] = True
                msg_idx += 1

        processed_count += len(ann_ids)
        if processed_count % 100 == 0 or processed_count == len(worker_annotations):
            print(f"[GPU {gpu_id}] Progress: {processed_count}/{len(worker_annotations)}")

    save_json(list(ann_dict.values()), tmp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_dir", required=True, help="COCO image folder of the split.")
    parser.add_argument("--annotations", required=True, help="COCO instances JSON of the split.")
    parser.add_argument("--descriptors", required=True,
                        help="Per-class descriptors {class: [5 strings]}, e.g. concepts/per_class_descriptors_top5.json.")
    parser.add_argument("--bin_rank", required=True, choices=["bin", "rank"], help="Binary or ranked labels.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--num_instances", type=int, default=None, help="Only label the first N annotations (default: all).")
    parser.add_argument("--num_gpus", type=int, default=2, help="Local GPUs (default: 2).")
    parser.add_argument("--batch_size", type=int, default=128, help="Instances per generate() call (default: 128).")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers per GPU (default: 8).")
    args = parser.parse_args()

    descriptors_data = load_json(args.descriptors)
    coco_data = load_json(args.annotations, verbose=True)
    cat_id_to_name, img_id_to_filename = coco_lookups(coco_data)

    all_annotations = coco_data.get('annotations', [])
    if args.num_instances is not None:
        all_annotations = all_annotations[:args.num_instances]
    # Group crops of the same image together (better file-cache hits).
    all_annotations.sort(key=lambda x: x['image_id'])

    gpu_chunks = split_for_gpus(all_annotations, args.num_gpus)
    print(f"Processing {len(all_annotations)} instances on {args.num_gpus} GPUs.")
    start_time = time.perf_counter()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.output))[0]
    tmp_paths = [os.path.join(out_dir, f"tmp_{stem}_worker_{g}.json") for g in range(args.num_gpus)]
    merged = run_on_gpus(gpu_worker, gpu_chunks, (args, descriptors_data, cat_id_to_name, img_id_to_filename),
                         tmp_paths)
    labeled, invalid = pop_status_flags(merged)
    coco_data['annotations'] = merged

    save_json(coco_data, args.output, indent=4)
    print_summary(len(merged), labeled, invalid, start_time, args.output)


if __name__ == "__main__":
    main()
