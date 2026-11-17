"""Average and median number of active concepts (sum of ``concept_vector``) per class."""

import argparse
import json
import statistics
from collections import defaultdict


def activations_per_class(data):
    """Return ``{class_name: {'average': float, 'median': float}}``."""
    category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    counts_per_class = defaultdict(list)
    for ann in data.get('annotations', []):
        counts_per_class[ann.get('category_id')].append(sum(ann.get('concept_vector', [])))
    return {
        category_map.get(cat_id, f"Unknown ID {cat_id}"): {
            'average': sum(counts) / len(counts),
            'median': statistics.median(counts),
        }
        for cat_id, counts in counts_per_class.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Concept-labelled COCO JSON (binary vectors).")
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        results = activations_per_class(json.load(f))
    print(f"{'Class Name':<20} | {'Average':<10} | {'Median':<10}")
    print("-" * 46)
    for name, m in sorted(results.items(), key=lambda item: item[1]['average'], reverse=True):
        print(f"{name:<20} | {m['average']:<10.2f} | {m['median']:<10.2f}")


if __name__ == "__main__":
    main()
