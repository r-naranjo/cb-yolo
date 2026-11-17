"""Split one COCO JSON into train/val by image.

All annotations of an image stay in the same split. The split approximates a
given fraction of *instances* in train (default 0.7) and tries to have every
category present in both splits.

Outputs (in --out_dir):
  instances_train2017.json, instances_val2017.json   (names configurable)
  class_counts_train_val.csv
  train_images.txt, val_images.txt

Usage:
    python dataset_tools/split_train_val.py --json datasets/COCO/annotations/instances_all2017.json \
        --out_dir datasets/COCO/annotations --train_frac 0.7 --seed 42
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict, Counter
from pathlib import Path
import csv
from copy import deepcopy
from typing import Dict, Set, Any

# ----------------------------
# I/O
# ----------------------------
def load_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ----------------------------
# Grouping
# ----------------------------
def group_annotations_by_image(annotations):
    img2anns = defaultdict(list)
    for ann in annotations:
        img2anns[ann['image_id']].append(ann)
    return img2anns

def build_image_infos(images, img2anns):
    """
    Return ``image_id -> {image, anns, count, cats}``.
    """
    info = {}
    for img in images:
        iid = img['id']
        anns = img2anns.get(iid, [])
        cats = set(a['category_id'] for a in anns if 'category_id' in a)
        info[iid] = {
            'image': img,
            'anns': anns,
            'count': len(anns),
            'cats': cats
        }
    return info

def categories_map(categories):
    """Map category id -> name."""
    id2name = {}
    for c in categories:
        id2name[c['id']] = c.get('name', str(c['id']))
    return id2name

# ----------------------------
# Assignment heuristics
# ----------------------------
def initial_category_coverage_assignment(image_infos: Dict[int, dict],
                                         cat2images: Dict[int, Set[int]],
                                         train_set: Set[int],
                                         val_set: Set[int],
                                         assigned: Set[int]) -> None:
    """
    Put at least one image of every category (rarest first) in each split when possible.
    Adding an image to one split removes it from the other.
    """
    cats_by_rarity = sorted(cat2images.keys(), key=lambda c: len(cat2images[c]) if cat2images[c] else 9999)
    for cat in cats_by_rarity:
        imgs = list(cat2images.get(cat, []))
        if not imgs:
            continue

        def cat_in_set(cat_id, s):
            for iid in s:
                if cat_id in image_infos[iid]['cats']:
                    return True
            return False

        # ensure presence in train
        if not cat_in_set(cat, train_set):
            candidates = [iid for iid in imgs if iid not in assigned]
            if not candidates:
                candidates = [iid for iid in imgs if iid in val_set]
            if candidates:
                chosen = min(candidates, key=lambda x: image_infos[x]['count'])
                train_set.add(chosen)
                val_set.discard(chosen)   # exclusivity
                assigned.add(chosen)

        # ensure presence in val
        if not cat_in_set(cat, val_set):
            candidates = [iid for iid in imgs if iid not in assigned]
            if not candidates:
                candidates = [iid for iid in imgs if iid in train_set]
            if candidates:
                chosen = min(candidates, key=lambda x: image_infos[x]['count'])
                val_set.add(chosen)
                train_set.discard(chosen)  # exclusivity
                assigned.add(chosen)

def fill_remaining_by_target(image_infos: Dict[int, dict],
                             assigned: Set[int],
                             train_set: Set[int],
                             val_set: Set[int],
                             target_train_instances: int) -> int:
    """
    Assign the remaining images (largest first) to train until target_train_instances
    is reached, the rest to val. Returns the number of train instances.
    """
    unassigned = [iid for iid in image_infos.keys() if iid not in assigned]
    unassigned_sorted = sorted(unassigned, key=lambda x: image_infos[x]['count'], reverse=True)

    def current_train_instances():
        return sum(image_infos[i]['count'] for i in train_set)

    train_instances = current_train_instances()
    for iid in unassigned_sorted:
        if train_instances < target_train_instances:
            train_set.add(iid)
            val_set.discard(iid)
            assigned.add(iid)
            train_instances += image_infos[iid]['count']
        else:
            val_set.add(iid)
            train_set.discard(iid)
            assigned.add(iid)
    return train_instances

def try_fix_missing_categories_by_moving(image_infos: Dict[int, dict],
                                         cat2images: Dict[int, Set[int]],
                                         train_set: Set[int],
                                         val_set: Set[int],
                                         target_train_instances: int) -> None:
    """
    Move small images so that every category appears in both splits, then
    rebalance towards target_train_instances by moving the smallest images.
    """
    def train_instances():
        return sum(image_infos[i]['count'] for i in train_set)

    def cat_presence_in(cat, s):
        for iid in s:
            if cat in image_infos[iid]['cats']:
                return True
        return False

    cats = list(cat2images.keys())
    for cat in cats:
        in_train = cat_presence_in(cat, train_set)
        in_val = cat_presence_in(cat, val_set)
        if not in_val:
            cand = [iid for iid in train_set if cat in image_infos[iid]['cats']]
            if cand:
                chosen = min(cand, key=lambda x: image_infos[x]['count'])
                train_set.discard(chosen)
                val_set.add(chosen)
        if not in_train:
            cand = [iid for iid in val_set if cat in image_infos[iid]['cats']]
            if cand:
                chosen = min(cand, key=lambda x: image_infos[x]['count'])
                val_set.discard(chosen)
                train_set.add(chosen)

    cur_train = train_instances()
    if cur_train < target_train_instances:
        vals_sorted = sorted(list(val_set), key=lambda x: image_infos[x]['count'])
        for iid in vals_sorted:
            if cur_train >= target_train_instances:
                break
            val_set.remove(iid)
            train_set.add(iid)
            cur_train += image_infos[iid]['count']
    elif cur_train > target_train_instances:
        train_sorted_small = sorted(list(train_set), key=lambda x: image_infos[x]['count'])
        for iid in train_sorted_small:
            if cur_train <= target_train_instances:
                break
            train_set.remove(iid)
            val_set.add(iid)
            cur_train -= image_infos[iid]['count']

# ----------------------------
# Outputs and counters
# ----------------------------
def build_output_jsons(original_json: Dict[str, Any],
                       image_infos: Dict[int, dict],
                       train_set: Set[int],
                       val_set: Set[int]) -> (Dict[str, Any], Dict[str, Any]):
    """
    Return two JSONs with every top-level key of the original and the
    split-specific 'images' / 'annotations' lists.
    """
    base = deepcopy(original_json)

    anns_train = []
    anns_val = []
    images_train = []
    images_val = []

    train_ids = set(train_set)
    val_ids = set(val_set)

    for iid, info in image_infos.items():
        if iid in train_ids:
            images_train.append(info['image'])
            anns_train.extend(info['anns'])
        elif iid in val_ids:
            images_val.append(info['image'])
            anns_val.extend(info['anns'])

    json_train = {k: deepcopy(v) for k, v in base.items() if k not in ('images', 'annotations')}
    json_val = {k: deepcopy(v) for k, v in base.items() if k not in ('images', 'annotations')}

    json_train['images'] = images_train
    json_train['annotations'] = anns_train

    json_val['images'] = images_val
    json_val['annotations'] = anns_val

    return json_train, json_val

def compute_class_counts_per_split(image_infos: Dict[int, dict],
                                   train_set: Set[int],
                                   val_set: Set[int]):
    train_counts = Counter()
    val_counts = Counter()
    for iid in train_set:
        for a in image_infos[iid]['anns']:
            cid = a.get('category_id')
            if cid is not None:
                train_counts[cid] += 1
    for iid in val_set:
        for a in image_infos[iid]['anns']:
            cid = a.get('category_id')
            if cid is not None:
                val_counts[cid] += 1
    return train_counts, val_counts

def save_class_counts_csv(id2name, train_counts, val_counts, out_csv_path: Path):
    all_cats = sorted(set(list(id2name.keys()) + list(train_counts.keys()) + list(val_counts.keys())))
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category_id', 'category_name', 'train_count', 'val_count'])
        for cid in all_cats:
            name = id2name.get(cid, str(cid))
            t = train_counts.get(cid, 0)
            v = val_counts.get(cid, 0)
            writer.writerow([cid, name, t, v])

def print_class_counts_table(id2name, train_counts, val_counts):
    all_cats = sorted(set(list(id2name.keys()) + list(train_counts.keys()) + list(val_counts.keys())))
    print('\nInstances per class (train / val):')
    print(f'{"id":>4s} {"name":40s} {"train":>8s} {"val":>8s}')
    print('-' * 70)
    for cid in all_cats:
        name = id2name.get(cid, str(cid))
        t = train_counts.get(cid, 0)
        v = val_counts.get(cid, 0)
        print(f'{cid:4d} {name:40.40s} {t:8d} {v:8d}')

# ----------------------------
# Main
# ----------------------------
def main(args):
    random.seed(args.seed)

    data = load_json(Path(args.json))
    annotations = data.get('annotations', [])
    images = data.get('images', [])
    categories = data.get('categories', [])

    if not images:
        print('ERROR: the JSON has no "images" key.')
        sys.exit(1)

    img2anns = group_annotations_by_image(annotations)
    image_infos = build_image_infos(images, img2anns)
    id2name = categories_map(categories)

    total_instances = len(annotations)
    target_train = int(round(args.train_frac * total_instances))
    print(f'Total instances: {total_instances}. Train target ~= {target_train} ({args.train_frac*100:.1f}%)')

    # category -> images
    cat2images = defaultdict(set)
    for iid, info in image_infos.items():
        for c in info['cats']:
            cat2images[c].add(iid)

    assigned = set()
    train_set = set()
    val_set = set()

    # 1) Initial coverage of rare categories
    initial_category_coverage_assignment(image_infos, cat2images, train_set, val_set, assigned)

    # 2) Assign the rest towards the target
    _ = fill_remaining_by_target(image_infos, assigned, train_set, val_set, target_train)

    # 3) Fix categories missing from a split by moving small images
    try_fix_missing_categories_by_moving(image_infos, cat2images, train_set, val_set, target_train)

    # 4) Remove residual overlaps (should not happen)
    overlap = train_set & val_set
    if overlap:
        print(f'WARNING: {len(overlap)} images were in both splits; removing them from val.')
        for iid in list(overlap):
            val_set.discard(iid)

    # 5) Image lists (file_name, or id if missing)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list = [image_infos[i]['image'].get('file_name', str(i)) for i in sorted(train_set)]
    val_list   = [image_infos[i]['image'].get('file_name', str(i)) for i in sorted(val_set)]

    (out_dir / 'train_images.txt').write_text('\n'.join(train_list), encoding='utf-8')
    (out_dir / 'val_images.txt').write_text('\n'.join(val_list), encoding='utf-8')
    print(f'Image lists saved: {out_dir/"train_images.txt"}, {out_dir/"val_images.txt"}')

    # 6) Per-class counts
    train_counts, val_counts = compute_class_counts_per_split(image_infos, train_set, val_set)
    final_train_instances = sum(train_counts.values())
    final_val_instances = sum(val_counts.values())

    print(f'\nFinal instances -> train: {final_train_instances}, val: {final_val_instances}, ratio: {final_train_instances/total_instances:.3f}')

    if final_train_instances + final_val_instances != total_instances:
        print('\nWARNING: train + val differs from the total number of instances.')
        print(f'   total={total_instances} | train+val={final_train_instances + final_val_instances}')
        all_assigned_images = set(train_set) | set(val_set)
        ann_image_ids = set(a['image_id'] for a in annotations)
        missing_from_assignment = ann_image_ids - all_assigned_images
        if missing_from_assignment:
            print(f'   {len(missing_from_assignment)} image_id(s) with annotations are in no split (images/annotations mismatch). Examples: {sorted(list(missing_from_assignment))[:20]}')
    else:
        print('Check OK: train + val == total.')

    # 7) Write the split JSONs
    json_train, json_val = build_output_jsons(data, image_infos, train_set, val_set)

    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name
    save_json(json_train, train_path)
    save_json(json_val, val_path)
    print(f'\nSaved:\n - {train_path}\n - {val_path}')

    # 8) Per-class count table
    csv_path = out_dir / 'class_counts_train_val.csv'
    save_class_counts_csv(id2name, train_counts, val_counts, csv_path)
    print(f'Count table saved to: {csv_path}')
    print_class_counts_table(id2name, train_counts, val_counts)

    # 9) Report categories missing from a split
    missing_train = [c for c in id2name.keys() if train_counts.get(c, 0) == 0]
    missing_val = [c for c in id2name.keys() if val_counts.get(c, 0) == 0]
    if missing_train:
        print('\nWARNING: these classes do NOT appear in train:')
        for c in missing_train:
            print(f' - {c}: {id2name.get(c)}')
    if missing_val:
        print('\nWARNING: these classes do NOT appear in val:')
        for c in missing_val:
            print(f' - {c}: {id2name.get(c)}')

    print('\nDone.')

# ----------------------------
# CLI
# ----------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Split a COCO JSON into train/val by image (all annotations of an image stay together).')
    p.add_argument('--json', required=True, help='Path to COCO JSON input')
    p.add_argument('--out_dir', default='out_split', help='Output folder for split jsons and csv')
    p.add_argument('--train_frac', type=float, default=0.7, help='Fraction of instances to put in train (default 0.7)')
    p.add_argument('--train_name', default='instances_train2017.json', help='Filename for train json')
    p.add_argument('--val_name', default='instances_val2017.json', help='Filename for val json')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = p.parse_args()
    main(args)
