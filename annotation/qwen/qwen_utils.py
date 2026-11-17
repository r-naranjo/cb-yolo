"""Shared helpers for the Qwen3-VL concept annotators.

Everything that defines the released labels (model id, dtype/attention,
prompts, generation parameters and output parsing) lives in the annotator
scripts or here and must not be changed if you want to reproduce them.

Heavy dependencies (torch, transformers) are imported lazily so that the
``--help`` of every script works without them.
"""

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_json, save_json  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_qwen(device):
    """Load Qwen3-VL-8B-Instruct in bfloat16 with FlashAttention-2 on ``device``.

    The tokenizer pads on the left, which is required for batched generation.
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def generate(model, processor, messages_batch, **generate_kwargs):
    """Run batched chat generation and return the decoded completions only."""
    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True, padding=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, **generate_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def user_message(image, text):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def coco_lookups(coco_data):
    """Return ``(category_id -> name, image_id -> file_name)``."""
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    img_id_to_filename = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}
    return cat_id_to_name, img_id_to_filename


def is_degenerate_bbox(bbox):
    """True for boxes whose integer width or height is <= 0."""
    _, _, w, h = bbox
    return int(w) <= 0 or int(h) <= 0


def crop_bbox(image, bbox):
    """Crop a PIL image to a COCO ``[x, y, w, h]`` box (no clamping)."""
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


# --------------------------------------------------------------------------- #
# Parsing of model outputs
# --------------------------------------------------------------------------- #
def concept_present(concept, text):
    """True if ``concept`` occurs in ``text`` as a whole token.

    A match must not be preceded or followed by a letter or ``_``, so
    ``has_legs`` does not match inside ``has_legs_furniture``.
    """
    pattern = r'(?<![a-zA-Z_])' + re.escape(concept) + r'(?![a-zA-Z_])'
    return bool(re.search(pattern, text))


def extract_vector(text):
    """Parse the first ``[a, b, c, d, e]`` list of digits 0-5 in ``text``.

    Returns ``[0, 0, 0, 0, 0]`` if no such list is found.
    """
    import ast

    match = re.search(r'\[[0-5](?:,\s*[0-5]){4}\]', text)
    if match:
        try:
            return ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            pass
    return [0, 0, 0, 0, 0]


# --------------------------------------------------------------------------- #
# Work splitting
# --------------------------------------------------------------------------- #
def shard_slice(items, split_total, split_idx):
    """Return the ``split_idx``-th of ``split_total`` contiguous slices.

    The last slice takes the remainder. Used to spread one split over several
    independent jobs (``--split_total`` / ``--split_idx``).
    """
    if split_total <= 1:
        return items, 0, len(items)
    chunk_size = len(items) // split_total
    start = split_idx * chunk_size
    end = start + chunk_size if split_idx < split_total - 1 else len(items)
    return items[start:end], start, end


def split_for_gpus(items, num_gpus):
    """Split ``items`` into ``num_gpus`` contiguous chunks; the last gets the remainder."""
    chunk_size = len(items) // num_gpus
    chunks = [items[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus)]
    if len(items) % num_gpus != 0:
        chunks[-1].extend(items[num_gpus * chunk_size:])
    return chunks


def run_on_gpus(worker_fn, chunks, extra_args, tmp_paths):
    """Run ``worker_fn(gpu_id, chunk, tmp_path, *extra_args)`` in one spawned process per GPU.

    Each worker dumps its annotations to ``tmp_paths[gpu_id]``. The results
    are concatenated in GPU order, the temporary files removed, and the
    merged list of annotations returned.
    """
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = mp.Process(target=worker_fn, args=(gpu_id, chunk, tmp_paths[gpu_id], *extra_args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    merged = []
    for path in tmp_paths:
        merged.extend(load_json(path))
        os.remove(path)
    return merged


def pop_status_flags(annotations):
    """Remove the per-annotation bookkeeping flags; return ``(n_labeled, n_invalid)``."""
    labeled = invalid = 0
    for ann in annotations:
        if ann.pop("successfully_labeled", False):
            labeled += 1
        if ann.pop("invalid_bbox_handled", False):
            invalid += 1
    return labeled, invalid


def print_summary(total, labeled, invalid, start_time, output_path):
    denom = total or 1
    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY")
    print("=" * 50)
    print(f"Total instances processed:  {total}")
    print(f"Successfully labeled:       {labeled} ({labeled / denom * 100:.2f}%)")
    print(f"Invalid bboxes auto-filled: {invalid} ({invalid / denom * 100:.2f}%)")
    print(f"Skipped / failed:           {total - labeled - invalid}")
    print(f"Total execution time:       {time.perf_counter() - start_time:.2f} s")
    print(f"Output: {output_path}")
    print("=" * 50)


__all__ = [
    "MODEL_ID", "load_qwen", "generate", "user_message", "coco_lookups", "is_degenerate_bbox",
    "crop_bbox", "concept_present", "extract_vector", "shard_slice", "split_for_gpus",
    "run_on_gpus", "pop_status_flags", "print_summary", "load_json", "save_json",
]
