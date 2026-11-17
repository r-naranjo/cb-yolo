#!/usr/bin/env python3
"""Concept/class separability of an annotated dataset (no model involved).

How exclusively do concepts identify a single class? For every class, the
set of concepts that are active in at least `threshold` of its annotations
is kept (filtered "scenario"). Then

    mOverlap          = mean over all K concepts of (#classes using it - 1)
    mConceptsPerClass = mean size of the per-class concept sets
    mSeparability     = 1 - mOverlap / (mConceptsPerClass - 1)

mSeparability = 1 means every concept belongs to exactly one class; lower
values mean concepts are shared across classes. It is undefined when
mConceptsPerClass <= 1.

Library functions are also used by plot_separability.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import CAT_TO_IDX, NUM_COCO_CLASSES  # noqa: E402


def build_scenario_from_json(instances_json_path, num_concepts=None):
    """Parse the annotation JSON once.

    Returns:
        scenario:          {category_id: set(active concept indices)}
        appearance_matrix: (K, 80) number of annotations of each class in which
                           each concept is active
        class_counts:      list of 80 annotation counts per class
        categories:        the JSON's "categories" list
    K defaults to the length of the first annotation's concept_vector.
    """
    with open(instances_json_path, "r") as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    if num_concepts is None:
        num_concepts = len(annotations[0]["concept_vector"]) if annotations else 0

    scenario = {category["id"]: set() for category in data.get("categories", [])}
    appearance_matrix = np.zeros((num_concepts, NUM_COCO_CLASSES), float)
    class_counts = [0] * NUM_COCO_CLASSES

    for ann in annotations:
        cat_id = ann["category_id"]
        cat_index = CAT_TO_IDX[int(cat_id)]
        class_counts[cat_index] += 1
        for concept_idx, value in enumerate(ann["concept_vector"]):
            if value > 0:
                scenario[cat_id].add(concept_idx)
                appearance_matrix[concept_idx][cat_index] += 1

    return scenario, appearance_matrix, class_counts, data.get("categories", [])


def filter_scenario(scenario, appearance_matrix, class_counts, threshold_fraction):
    """Keep a concept for a class only if it is active in at least
    threshold_fraction * (#annotations of the class). Returns {cat_id: [concepts]}."""
    filtered = {}
    for cat_id, concepts in scenario.items():
        cat_index = CAT_TO_IDX[int(cat_id)]
        threshold = class_counts[cat_index] * threshold_fraction
        filtered[cat_id] = [c for c in concepts if appearance_matrix[c][cat_index] >= threshold]
    return filtered


def mean_overlap(scenario, num_concepts):
    repetitions = np.zeros(num_concepts)
    for concepts in scenario.values():
        for concept in concepts:
            repetitions[concept] += 1
    return np.average(repetitions - 1)


def mean_concepts_per_class(scenario):
    return np.average([len(concepts) for concepts in scenario.values()])


def mean_separability(m_overlap, m_concepts_per_class):
    return 1 - m_overlap / (m_concepts_per_class - 1)


def separability_metrics(scenario, num_concepts):
    """(mOverlap, mConceptsPerClass, mSeparability or nan if undefined)."""
    m_cpc = mean_concepts_per_class(scenario)
    m_o = mean_overlap(scenario, num_concepts)
    m_s = mean_separability(m_o, m_cpc) if m_cpc > 1 else float("nan")
    return m_o, m_cpc, m_s


def main():
    parser = argparse.ArgumentParser(description="Concept/class separability of an annotated dataset")
    parser.add_argument("--annotations", required=True,
                        help="COCO-style JSON with a binary concept_vector per annotation")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.75],
                        help="Activation-frequency filter(s), fraction of a class's annotations")
    parser.add_argument("--output", default=None, help="Optional JSON file for the results")
    args = parser.parse_args()

    scenario, appearance_matrix, class_counts, _ = build_scenario_from_json(args.annotations)
    num_concepts = appearance_matrix.shape[0]
    results = []
    for t in args.threshold:
        m_o, m_cpc, m_s = separability_metrics(
            filter_scenario(scenario, appearance_matrix, class_counts, t), num_concepts)
        results.append({"threshold": t, "mOverlap": float(m_o), "mConceptsPerClass": float(m_cpc),
                        "mSeparability": float(m_s)})
        print(f"threshold {t * 100:.0f}%: concepts={num_concepts} classes={len(scenario)} "
              f"mOverlap={m_o:.4f} mConceptsPerClass={m_cpc:.4f} mSeparability={m_s:.4f}")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
