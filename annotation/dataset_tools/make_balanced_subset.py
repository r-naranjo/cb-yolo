"""Build a class-balanced X% subset of a (concept-labelled) COCO JSON.

Whole images are selected greedily, rarest category first, until every
category reaches ``percentage``% of its original instance count; all
annotations of a selected image are kept. Prints an original-vs-subset class
ratio table. The released
``final_train2017_..._post_processed_subset.json`` was built with
``--percentage 10`` (unseeded, so it cannot be regenerated bit-exactly).
"""

import argparse
import json
import random
from collections import defaultdict


def create_balanced_subset(data, percentage, rng=random):
    """Return ``(new_data, cat_counts, current_counts, sorted_cats)``."""
    image_to_anns = defaultdict(list)
    cat_counts = defaultdict(int)
    for ann in data.get('annotations', []):
        image_to_anns[ann['image_id']].append(ann)
        cat_counts[ann['category_id']] += 1

    target_counts = {cat: count * percentage / 100.0 for cat, count in cat_counts.items()}

    cat_to_images = defaultdict(set)
    for image_id, anns in image_to_anns.items():
        for ann in anns:
            cat_to_images[ann['category_id']].add(image_id)

    sorted_cats = sorted(cat_counts.keys(), key=lambda c: cat_counts[c])
    selected_image_ids = set()
    current_counts = defaultdict(int)
    for cat in sorted_cats:
        available_images = list(cat_to_images[cat] - selected_image_ids)
        rng.shuffle(available_images)
        for img_id in available_images:
            if current_counts[cat] >= target_counts[cat]:
                break
            selected_image_ids.add(img_id)
            for ann in image_to_anns[img_id]:
                current_counts[ann['category_id']] += 1

    new_data = {
        "images": [img for img in data.get('images', []) if img['id'] in selected_image_ids],
        "annotations": [ann for ann in data.get('annotations', []) if ann['image_id'] in selected_image_ids],
        "categories": data.get('categories', []),
    }
    for key in data.keys():
        if key not in new_data:
            new_data[key] = data[key]
    return new_data, cat_counts, current_counts, sorted_cats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="Input COCO JSON.")
    parser.add_argument("--output", required=True, help="Output subset JSON.")
    parser.add_argument("--percentage", type=float, default=10, help="Target %% of instances per class (default: 10).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: unseeded).")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)
    rng = random.Random(args.seed) if args.seed is not None else random
    new_data, cat_counts, current_counts, sorted_cats = create_balanced_subset(data, args.percentage, rng)
    with open(args.output, 'w') as f:
        json.dump(new_data, f)

    total_orig = len(data.get('annotations', []))
    total_new = len(new_data['annotations'])
    print(f"\nImages: {len(data.get('images', []))} -> {len(new_data['images'])}")
    print(f"Instances: {total_orig} -> {total_new}")
    print(f"\n{'Category ID':<12} | {'Orig. inst.':<12} | {'New inst.':<10} | {'Orig %':<10} | {'New %':<10}")
    print("-" * 64)
    for cat in sorted_cats:
        orig_ratio = 100 * cat_counts[cat] / total_orig if total_orig else 0
        new_ratio = 100 * current_counts[cat] / total_new if total_new else 0
        print(f"{cat:<12} | {cat_counts[cat]:<12} | {current_counts[cat]:<10} | {orig_ratio:<10.4f} | {new_ratio:<10.4f}")


if __name__ == "__main__":
    main()
