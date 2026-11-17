"""Small JSON / concept-vocabulary helpers shared by the annotation scripts."""

import json
import os


def load_json(path, verbose=False):
    if verbose:
        print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path, indent=None, ensure_ascii=True, verbose=False):
    """Write ``data`` to ``path`` (creating the parent directory if needed)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if verbose:
        print(f"Saving {path}...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def flatten_concepts(data):
    """Return the flat, ordered concept list of a vocabulary file.

    Accepts every layout used in ``annotation/concepts``:
      - ``{"concepts": [c0, c1, ...]}``                (shared_143.json, per_class_400_list.json)
      - ``{"concepts": {"group": [c...], ...}}``       (shared_143_grouped.json, shared_143_chunked.json)
      - ``{"class name": [d0, ..., d4], ...}``         (per_class_descriptors_*.json)
      - a bare list.
    Groups are concatenated in file order, which defines the concept index.
    """
    if isinstance(data, dict) and "concepts" in data:
        data = data["concepts"]
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict):
        out = []
        for values in data.values():
            out.extend(values)
        return out
    raise ValueError("Unrecognised concept vocabulary format.")


def load_concept_list(path):
    return flatten_concepts(load_json(path))
