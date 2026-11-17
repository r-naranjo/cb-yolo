"""Export the concept labels of an annotated COCO JSON to a compact concept pack.

The pack keeps only what the annotation adds to COCO (one concept vector per
annotation, plus the annotation order); everything else is taken from the
official COCO file when the JSON is rebuilt with ``merge_concepts.py``.

With ``--coco`` (recommended) the export is verified: the JSON is rebuilt
from the official file and compared with the input. The export fails if any
annotation (box, category, concept vector, order) or any image field the code
uses (id, file_name, width, height) differs. Other metadata differences (e.g.
``info`` or ``coco_url``) are reported but allowed: they do not affect
training or evaluation, and the merge uses the official 2017 values.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_json  # noqa: E402
from release.concept_pack import load_pack, pack, save_pack, verify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="annotated COCO JSON (with concept_vector)")
    parser.add_argument("--output", required=True, help="output concept pack (.npz)")
    parser.add_argument("--coco", help="official COCO instances JSON of the same split, used to verify")
    parser.add_argument("--description", default="", help="free-text description stored in the pack")
    parser.add_argument("--vocabulary", default="", help="vocabulary file name stored in the pack")
    args = parser.parse_args()

    annotated = load_json(args.input, verbose=True)
    arrays = pack(annotated, description=args.description, vocabulary=args.vocabulary)
    save_pack(args.output, arrays)
    n, m = len(arrays["annotation_ids"]), int(arrays["labelled"].sum())
    print(f"Wrote {args.output}: {n} annotations, {m} labelled, K={int(arrays['num_concepts'])}, "
          f"encoding={arrays['encoding']}, {os.path.getsize(args.output) / 1e6:.1f} MB")

    if not args.coco:
        print("Not verified (no --coco given).")
        return
    coco = load_json(args.coco, verbose=True)
    findings = verify(annotated, coco, load_pack(args.output))
    for key, value in findings.items():
        print(f"  {key}: {value}")
    required = ("concepts_equal", "annotations_equal", "image_fields_used_equal",
                "categories_equal", "extra_keys_stored")
    if not all(findings[k] for k in required):
        os.remove(args.output)
        sys.exit("Verification FAILED: the pack does not reproduce the input; output removed.")
    print("Verification passed.")


if __name__ == "__main__":
    main()
