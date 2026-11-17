"""Concept-name loading.

Supported files (see annotation/concepts/):
  * shared_143.json                   {"concepts": [143 names]}
  * shared_143_grouped.json           {"concepts": {group: [names], ...}}
  * shared_143_chunked.json           {"concepts": {chunk: [names], ...}}
  * per_class_400_list.json           {"concepts": [400 names]}
  * per_class_descriptors_top5.json   {class_name: [5 names], ...}
  * a bare JSON list of names, or a .txt file with one name per line.

Grouped/chunked files are flattened in file order, which matches the index
order of the flat list. Per-class dicts are flattened in COCO class order
(class c owns concept indices 5c..5c+4 in the 400-concept vocabulary).
"""

import json

from .coco import COCO_CLASSES


def _flatten_groups(groups):
    names = []
    for value in groups.values():
        if not isinstance(value, list):
            return None
        names.extend(value)
    return names


def read_concept_names(path, per_class_size=5):
    """Return the flat, index-ordered list of concept names stored in `path`."""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n\r") for line in f]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return list(data)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: unsupported concept-name format")
    if "concepts" in data:
        concepts = data["concepts"]
        if isinstance(concepts, list):
            return list(concepts)
        if isinstance(concepts, dict):
            names = _flatten_groups(concepts)
            if names is not None:
                return names
        raise ValueError(f"{path}: unsupported 'concepts' entry")
    if any(cls in data for cls in COCO_CLASSES):
        # Per-class descriptor dict: flatten in COCO class order.
        names = []
        for cls in COCO_CLASSES:
            names.extend(data.get(cls, [f"{cls}_concept_{i}" for i in range(per_class_size)]))
        return names
    raise ValueError(f"{path}: unsupported concept-name format")


def load_concept_names(path, num_concepts):
    """Concept names for labelling outputs.

    Falls back to 'concept_<i>' (with a warning) if `path` is not given or its
    length does not match `num_concepts`.
    """
    generic = [f"concept_{i}" for i in range(num_concepts)]
    if not path:
        return generic
    names = read_concept_names(path)
    if len(names) != num_concepts:
        print(f"[WARN] {path} has {len(names)} names but the model has {num_concepts} "
              f"concepts; using generic names.")
        return generic
    return names
