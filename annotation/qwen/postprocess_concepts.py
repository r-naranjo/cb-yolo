"""Post-process the ``concept_vector`` columns of a labelled COCO JSON.

Operations (applied in this order to every annotation):

1. ``--restore-from ORIG.json --restore C [C ...]``: copy columns ``C`` from the
   annotation with the same ``id`` in ``ORIG.json``.
2. ``--zero C [C ...]``: set columns ``C`` to 0.
3. ``--zero-if T:COND [T:COND ...]``: set column ``T`` to 0 where column ``COND`` is 1.

Columns can be given by name (resolved with ``--concepts``) or by 0-based index.

``--preset released`` reproduces the ``*_post_processed.json`` files used for
the released CB-YOLOX models (shared 143-concept vocabulary,
``concepts/shared_143.json``):

    --zero color_multicolored color_gradient shadow_is_visible light_source_from_above   (113 114 141 142)
    --zero-if has_legs:has_legs_furniture                                                 (0 if 46)

History: the original one-off script additionally zeroed ``color_solid`` (112)
and ``edges_sharp`` (98) and later restored both from the unprocessed JSON, so
their net change is nil; the preset therefore leaves them untouched. This was
checked column by column against the released val2017 and train2017 files.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_concept_list, load_json, save_json  # noqa: E402

DEFAULT_CONCEPTS = Path(__file__).resolve().parents[1] / "concepts" / "shared_143.json"

PRESETS = {
    "released": {
        "zero": ["color_multicolored", "color_gradient", "shadow_is_visible", "light_source_from_above"],
        "zero_if": [("has_legs", "has_legs_furniture")],
    },
}


def resolve(column, concepts):
    """Concept name or index string -> integer index."""
    if isinstance(column, int):
        return column
    if column.isdigit():
        return int(column)
    if concepts is None:
        raise SystemExit(f"Concept name {column!r} given but no --concepts file to resolve it.")
    try:
        return concepts.index(column)
    except ValueError:
        raise SystemExit(f"Unknown concept {column!r}.")


def postprocess(data, zero=(), zero_if=(), restore=(), original=None):
    """Apply the operations in place on ``data`` (a COCO dict). Columns are integer indices.

    Returns a dict of counters (values changed per operation, missing ids).
    """
    stats = {"restored": 0, "zeroed": 0, "zeroed_if": 0, "missing_in_original": 0}
    orig_by_id = {}
    if restore:
        orig_by_id = {ann["id"]: ann for ann in original["annotations"]}

    for ann in data["annotations"]:
        vec = ann["concept_vector"]
        if restore:
            src = orig_by_id.get(ann["id"])
            if src is None:
                stats["missing_in_original"] += 1
            else:
                for idx in restore:
                    if vec[idx] != src["concept_vector"][idx]:
                        stats["restored"] += 1
                    vec[idx] = src["concept_vector"][idx]
        for idx in zero:
            if vec[idx] != 0:
                stats["zeroed"] += 1
            vec[idx] = 0
        for target, cond in zero_if:
            if vec[cond] == 1 and vec[target] != 0:
                vec[target] = 0
                stats["zeroed_if"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--input", required=True, help="Labelled COCO JSON.")
    parser.add_argument("--output", required=True, help="Output JSON (may equal --input).")
    parser.add_argument("--concepts", default=None, help="Vocabulary file used to resolve concept names "
                        "(e.g. concepts/shared_143.json).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Named set of operations.")
    parser.add_argument("--zero", nargs="+", default=[], metavar="C", help="Columns set to 0.")
    parser.add_argument("--zero-if", dest="zero_if", nargs="+", default=[], metavar="T:COND",
                        help="Set column T to 0 where column COND is 1.")
    parser.add_argument("--restore", nargs="+", default=[], metavar="C", help="Columns copied from --restore-from.")
    parser.add_argument("--restore-from", dest="restore_from", default=None, help="Reference JSON for --restore.")
    args = parser.parse_args()

    zero, zero_if = list(args.zero), [tuple(x.split(":", 1)) for x in args.zero_if]
    if any(len(t) != 2 for t in zero_if):
        raise SystemExit("--zero-if expects TARGET:CONDITION pairs.")
    if args.preset:
        zero = PRESETS[args.preset]["zero"] + zero
        zero_if = PRESETS[args.preset]["zero_if"] + zero_if
    if bool(args.restore) != bool(args.restore_from):
        raise SystemExit("--restore and --restore-from must be used together.")

    if args.preset and not args.concepts:
        args.concepts = str(DEFAULT_CONCEPTS)
    concepts = load_concept_list(args.concepts) if args.concepts else None
    zero_idx = [resolve(c, concepts) for c in zero]
    zero_if_idx = [(resolve(t, concepts), resolve(c, concepts)) for t, c in zero_if]
    restore_idx = [resolve(c, concepts) for c in args.restore]
    print(f"zero: {zero_idx}  zero_if (target, cond): {zero_if_idx}  restore: {restore_idx}")

    data = load_json(args.input, verbose=True)
    original = load_json(args.restore_from, verbose=True) if args.restore_from else None
    stats = postprocess(data, zero_idx, zero_if_idx, restore_idx, original)
    print(f"Changed values: {stats}")
    if stats["missing_in_original"]:
        print(f"WARNING: {stats['missing_in_original']} annotation ids not found in {args.restore_from}.")

    save_json(data, args.output, indent=2, ensure_ascii=False, verbose=True)


if __name__ == "__main__":
    main()
