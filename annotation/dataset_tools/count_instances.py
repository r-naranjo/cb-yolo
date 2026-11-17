"""Count images and instances of a COCO JSON, optionally per category with a bar plot.

Per-category output follows the JSON's ``categories`` list (sorted by id);
the bar plot has reference lines at 5/10/20 instances to flag rare classes.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.coco_classes import COCO_CLASSES  # noqa: E402


def category_names(data):
    """Ordered ``[(category_id, name)]``: from the JSON, else COCO_CLASSES with ids 1..80."""
    cats = data.get('categories', [])
    if cats:
        return [(c['id'], c.get('name', str(c['id']))) for c in sorted(cats, key=lambda c: c['id'])]
    return list(enumerate(COCO_CLASSES, start=1))


def count_per_category(annotations):
    return Counter(ann.get('category_id') for ann in annotations)


def plot_counts(labels, values, out_path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel('Number of instances')
    ax.set_title('Instances per category')
    for y, color in ((20, 'yellow'), (10, 'orange'), (5, 'red')):
        ax.axhline(y=y, color=color, linestyle='--', linewidth=1, label=f'{y} instances')
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Bar plot saved to: {out_path}')
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--json', required=True, help='COCO JSON.')
    parser.add_argument('--per_class', action='store_true', help='Print the per-category table and top-10.')
    parser.add_argument('--plot', default=None, help='Save a per-category bar plot (PNG) to this path.')
    parser.add_argument('--show', action='store_true', help='Also show the plot interactively.')
    args = parser.parse_args()

    start = time.time()
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = data.get('annotations', [])
    print(f"Images:    {len(data.get('images', [])):,}")
    print(f"Instances: {len(annotations):,}")
    print(f"({time.time() - start:.1f} s)")

    if not (args.per_class or args.plot):
        return
    names = category_names(data)
    counts = count_per_category(annotations)
    if args.per_class:
        print('\nInstances per category:')
        for cid, name in names:
            print(f'{cid:3d}. {name:30s} {counts.get(cid, 0)}')
        known = {cid for cid, _ in names}
        for cid in sorted((c for c in counts if c not in known), key=str):
            print(f'  extra category id {cid}: {counts[cid]}')
        print('\nTop 10 categories:')
        id2name = dict(names)
        for i, (cid, cnt) in enumerate(counts.most_common(10), start=1):
            print(f'{i:2d}. {id2name.get(cid, cid)!s:30s} {cnt}')
    if args.plot:
        plot_counts([n for _, n in names], [counts.get(cid, 0) for cid, _ in names], args.plot, args.show)


if __name__ == '__main__':
    main()
