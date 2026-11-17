"""Merge the per-job shards written by ``annotate_shared.py`` into one COCO JSON.

The first shard provides every top-level key (``images``, ``categories``, ...;
all shards carry the full copies); the ``annotations`` lists of all shards are
concatenated in the order given. Pass the shards in part order
(``part1ofN ... partNofN``) to reproduce the released files.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import load_json, save_json  # noqa: E402


def merge_shards(shards):
    """Return the merged dict; ``shards`` is a list of loaded COCO dicts."""
    merged = shards[0]
    annotations = []
    for shard in shards:
        annotations.extend(shard.get('annotations', []))
    merged['annotations'] = annotations
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", nargs="+", required=True, help="Shard JSONs, in part order.")
    parser.add_argument("--output", required=True, help="Merged output JSON.")
    args = parser.parse_args()

    start_time = time.time()
    shards = []
    for path in args.parts:
        shards.append(load_json(path, verbose=True))
        print(f"  {len(shards[-1].get('annotations', []))} annotations")

    n_images = {len(s.get('images', [])) for s in shards}
    if len(n_images) > 1:
        print(f"WARNING: shards have different 'images' lists (sizes {sorted(n_images)}); using the first one.")

    merged = merge_shards(shards)
    ids = [a['id'] for a in merged['annotations']]
    if len(ids) != len(set(ids)):
        print(f"WARNING: {len(ids) - len(set(ids))} duplicated annotation ids after merging.")
    print(f"Total merged annotations: {len(ids)}")

    save_json(merged, args.output, indent=4, verbose=True)
    print(f"Merged in {time.time() - start_time:.2f} s.")


if __name__ == "__main__":
    main()
