"""Rebuild an annotated COCO JSON from the official annotations and a concept pack.

Example (shared 143-concept vocabulary, COCO 2017 train):

    python annotation/release/merge_concepts.py \\
        --coco annotations/instances_train2017.json \\
        --pack concepts_shared143_train2017.npz \\
        --output datasets/COCO/annotations/instances_train2017.json

The output is the official JSON with one ``concept_vector`` per labelled
annotation, annotations in the order used for training, ready for CB-YOLOX.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_json, save_json  # noqa: E402
from release.concept_pack import load_pack, merge  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coco", required=True, help="official COCO instances JSON of the split")
    parser.add_argument("--pack", required=True, help="concept pack (.npz)")
    parser.add_argument("--output", required=True, help="output annotated JSON")
    args = parser.parse_args()

    arrays = load_pack(args.pack)
    meta = arrays["meta"]
    print(f"Pack: K={arrays['num_concepts']}, {int(arrays['labelled'].sum())} labelled annotations"
          + (f", {meta['description']}" if meta.get("description") else ""))
    out = merge(load_json(args.coco, verbose=True), arrays)
    save_json(out, args.output, verbose=True)
    print("Done.")


if __name__ == "__main__":
    main()
