"""Integrity checks for concept-annotated COCO JSONs.

Sub-commands:

``vectors``  Check every annotation of one file has a ``concept_vector`` of the
             same length (``--num_concepts`` if given) and values allowed by
             ``--scheme``:
               shared_binary       values in {0, 1}                       (143-concept vocabulary)
               per_class_binary    values in {0, 1}, non-zeros only in the 5 slots of the instance's class
               per_class_ranked    exactly K-5 zeros per vector, non-zeros in [1, 5]
               any                 only presence / length checks

``redistribution``  Check that the 2017 files written by
             ``redistribute_2014_to_2017.py`` contain exactly the images and
             annotations of the pooled 2014 files (by id, order-independent).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.coco_classes import COCO_CLASSES, DESCRIPTORS_PER_CLASS  # noqa: E402
from common.io import load_json  # noqa: E402


def check_vectors(data, scheme="any", num_concepts=None, max_report=5):
    """Return a list of ``(annotation_id, reason)`` for invalid annotations (empty = OK)."""
    import numpy as np

    annotations = data.get('annotations', [])
    missing = [a.get('id') for a in annotations if 'concept_vector' not in a]
    if missing:
        return [(i, "missing concept_vector") for i in missing]
    lengths = {len(a['concept_vector']) for a in annotations}
    if len(lengths) > 1:
        return [(None, f"inconsistent concept_vector lengths: {sorted(lengths)}")]
    if not annotations:
        return []
    k = lengths.pop()
    if num_concepts is not None and k != num_concepts:
        return [(None, f"concept_vector length {k} != expected {num_concepts}")]

    vectors = np.array([a['concept_vector'] for a in annotations], dtype=np.float32)
    ids = [a.get('id') for a in annotations]
    bad = np.zeros(len(annotations), dtype=bool)
    reasons = {}

    def flag(mask, why):
        for i in np.where(mask & ~bad)[0]:
            reasons[i] = why
        bad[:] = bad | mask

    if scheme in ("shared_binary", "per_class_binary"):
        flag(~np.isin(vectors, (0, 1)).all(axis=1), "values outside {0, 1}")
    if scheme == "per_class_binary":
        cat_names = {c['id']: c['name'] for c in data.get('categories', [])}
        allowed = np.zeros_like(vectors, dtype=bool)
        for row, ann in enumerate(annotations):
            name = cat_names.get(ann['category_id'])
            if name in COCO_CLASSES:
                s = COCO_CLASSES.index(name) * DESCRIPTORS_PER_CLASS
                allowed[row, s:s + DESCRIPTORS_PER_CLASS] = True
        flag(((vectors != 0) & ~allowed).any(axis=1), "non-zero outside the class slots")
    if scheme == "per_class_ranked":
        zero_counts = (vectors == 0).sum(axis=1)
        flag(zero_counts != k - DESCRIPTORS_PER_CLASS, f"zero count != {k - DESCRIPTORS_PER_CLASS}")
        flag(((vectors != 0) & ((vectors < 1) | (vectors > 5))).any(axis=1), "non-zero values outside [1, 5]")

    failed = np.where(bad)[0]
    return [(ids[i], reasons[i]) for i in failed]


def cmd_vectors(args):
    start = time.time()
    data = load_json(args.json, verbose=True)
    failures = check_vectors(data, args.scheme, args.num_concepts)
    n = len(data.get('annotations', []))
    if not failures:
        print(f"OK: all {n} annotations comply ({args.scheme}). {time.time() - start:.1f} s")
        return 0
    print(f"FAILED: {len(failures)} invalid annotations out of {n}.")
    for ann_id, why in failures[:args.max_report]:
        print(f"  annotation {ann_id}: {why}")
    return 1


def compare_redistribution(t14, v14, t17, v17):
    """Return a dict of boolean checks comparing pooled 2014 vs pooled 2017 data."""
    images_2014 = {img['id']: img for img in t14['images'] + v14['images']}
    anns_2014 = {ann['id']: ann for ann in t14['annotations'] + v14['annotations']}
    images_2017 = {img['id']: img for img in t17['images'] + v17['images']}
    anns_2017 = {ann['id']: ann for ann in t17['annotations'] + v17['annotations']}
    return {
        "top_level_keys_match": set(t14.keys()) == set(t17.keys()),
        "image_count_match": len(images_2014) == len(images_2017),
        "annotation_count_match": len(anns_2014) == len(anns_2017),
        "images_identical": all(images_2014[i] == images_2017.get(i) for i in images_2014),
        "annotations_identical": all(anns_2014[i] == anns_2017.get(i) for i in anns_2014),
        "categories_match": t14['categories'] == t17['categories'],
        "_counts": (len(images_2014), len(images_2017), len(anns_2014), len(anns_2017)),
    }


def cmd_redistribution(args):
    t14, v14 = load_json(args.train2014, True), load_json(args.val2014, True)
    t17, v17 = load_json(args.train2017, True), load_json(args.val2017, True)
    res = compare_redistribution(t14, v14, t17, v17)
    n_img14, n_img17, n_ann14, n_ann17 = res.pop("_counts")
    print(f"images 2014/2017: {n_img14}/{n_img17}   annotations 2014/2017: {n_ann14}/{n_ann17}")
    for key, ok in res.items():
        print(f"  {key:24s}: {ok}")
    # Note: redistribute_2014_to_2017.py rewrites image file names, so
    # images_identical is only expected to be True when the 2014 inputs
    # already used bare "<id>.jpg" names.
    return 0 if all(res.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("vectors", help="Check concept_vector shape / values of one file.")
    p.add_argument("--json", required=True)
    p.add_argument("--scheme", default="any", choices=["any", "shared_binary", "per_class_binary", "per_class_ranked"])
    p.add_argument("--num_concepts", type=int, default=None, help="Expected vector length (143 or 400).")
    p.add_argument("--max_report", type=int, default=5)
    p.set_defaults(func=cmd_vectors)

    p = sub.add_parser("redistribution", help="Compare 2014 inputs and 2017 outputs of the redistribution.")
    p.add_argument("--train2014", required=True)
    p.add_argument("--val2014", required=True)
    p.add_argument("--train2017", required=True)
    p.add_argument("--val2017", required=True)
    p.set_defaults(func=cmd_redistribution)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
