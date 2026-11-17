"""Re-bucket concept-labelled COCO-2014 train+val annotations onto the COCO-2017 train/val split.

COCO 2014 and 2017 contain the same image pool, partitioned differently. The
2014 train and val JSONs are pooled, each image's ``file_name`` is reduced to
the bare ``<coco_id>.jpg`` (e.g. ``COCO_train2014_000000557627.jpg`` ->
``000000557627.jpg``, the 2017 naming), and every image (with its annotations)
is assigned to train or val according to the image ids of the official
``instances_{train,val}2017.json``. Images in neither 2017 split are dropped.

Used to obtain the 2017 per-class (400-concept) files from the 2014 output of
``annotate_per_class.py``. All other top-level keys (``categories``,
``concept_names``, ...) are copied from the respective 2014 input.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_json, save_json  # noqa: E402


def clean_file_name(file_name):
    """``.../COCO_train2014_000000557627.jpg`` -> ``000000557627.jpg``."""
    return file_name.split('/')[-1].split('_')[-1]


def redistribute(train2014, val2014, train2017_ids, val2017_ids):
    """Return ``(new_train, new_val, dropped_images, dropped_annotations)``.

    Image dicts of the 2014 inputs are modified in place (``file_name`` cleaned).
    """
    all_images = train2014['images'] + val2014['images']
    all_annotations = train2014['annotations'] + val2014['annotations']

    new_train = {k: v for k, v in train2014.items() if k not in ('images', 'annotations')}
    new_train['images'], new_train['annotations'] = [], []
    new_val = {k: v for k, v in val2014.items() if k not in ('images', 'annotations')}
    new_val['images'], new_val['annotations'] = [], []

    destination = {}  # 2014 image id -> 'train' | 'val'
    dropped_images = 0
    for img in all_images:
        img['file_name'] = clean_file_name(img['file_name'])
        true_coco_id = int(img['file_name'].split('.')[0])
        if true_coco_id in train2017_ids:
            destination[img['id']] = 'train'
            new_train['images'].append(img)
        elif true_coco_id in val2017_ids:
            destination[img['id']] = 'val'
            new_val['images'].append(img)
        else:
            dropped_images += 1

    dropped_annotations = 0
    for ann in all_annotations:
        dest = destination.get(ann['image_id'])
        if dest == 'train':
            new_train['annotations'].append(ann)
        elif dest == 'val':
            new_val['annotations'].append(ann)
        else:
            dropped_annotations += 1
    return new_train, new_val, dropped_images, dropped_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train2014", required=True, help="Concept-labelled train2014 JSON.")
    parser.add_argument("--val2014", required=True, help="Concept-labelled val2014 JSON.")
    parser.add_argument("--instances_train2017", required=True, help="Official instances_train2017.json.")
    parser.add_argument("--instances_val2017", required=True, help="Official instances_val2017.json.")
    parser.add_argument("--out_train", required=True, help="Output train2017 JSON.")
    parser.add_argument("--out_val", required=True, help="Output val2017 JSON.")
    args = parser.parse_args()

    train2017_ids = {int(img['id']) for img in load_json(args.instances_train2017, True)['images']}
    val2017_ids = {int(img['id']) for img in load_json(args.instances_val2017, True)['images']}
    train2014 = load_json(args.train2014, True)
    val2014 = load_json(args.val2014, True)
    n_pooled = len(train2014['images']) + len(val2014['images'])

    new_train, new_val, dropped_images, dropped_annotations = redistribute(
        train2014, val2014, train2017_ids, val2017_ids)

    save_json(new_train, args.out_train, verbose=True)
    save_json(new_val, args.out_val, verbose=True)

    print("\n--- DISTRIBUTION SUMMARY ---")
    print(f"Pooled 2014 images:  {n_pooled}")
    print(f"To 2017 train:       {len(new_train['images'])}")
    print(f"To 2017 val:         {len(new_val['images'])}")
    print(f"Dropped images:      {dropped_images}")
    print(f"Dropped annotations: {dropped_annotations}")


if __name__ == "__main__":
    main()
