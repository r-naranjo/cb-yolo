"""Annotation loading and small output helpers."""

import csv
import json
import os
import re
from collections import defaultdict


def load_coco_annotations(json_path):
    """Load a COCO-style JSON. Returns (images, {image_id: [annotations]})."""
    print(f"Loading annotations from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    ann_dict = defaultdict(list)
    for ann in data["annotations"]:
        ann_dict[ann["image_id"]].append(ann)
    return data["images"], ann_dict


def write_json(path, obj, indent=2):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)
    print(f"Saved {path}")


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved {path}")


class AnsiStrippingFile:
    """File-like object for `contextlib.redirect_stdout`: writes text to
    `filename` with ANSI colour codes removed (used for the optional
    per-detection inference log)."""

    _ansi = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

    def __init__(self, filename):
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, text):
        self.file.write(self._ansi.sub("", text))

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
