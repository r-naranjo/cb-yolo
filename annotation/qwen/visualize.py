"""Visual QA: render random labelled instances (image + bbox, crop, concepts) to PNG files.

``--mode shared``    vectors over a shared vocabulary (e.g. concepts/shared_143.json);
                     lists the active concepts (value > 0).
``--mode per_class`` 400-long per-class vectors (concepts/per_class_descriptors_top5.json);
                     shows the 5 descriptors of the instance's class in green
                     (value > 0) or red (0).
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.coco_classes import COCO_CLASSES, DESCRIPTORS_PER_CLASS  # noqa: E402
from common.io import flatten_concepts, load_json  # noqa: E402


def _base_figure(img, crop_img, bbox, title, height):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    x, y, w, h = bbox
    fig, axs = plt.subplots(1, 3, figsize=(18, height))
    fig.suptitle(title, fontsize=16)
    axs[0].imshow(img)
    axs[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', facecolor='none'))
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    axs[1].imshow(crop_img)
    axs[1].set_title("Cropped Instance")
    axs[1].axis('off')
    axs[2].axis('off')
    return fig, axs


def draw_shared(img, crop_img, ann, class_name, concepts, out_path):
    import matplotlib.pyplot as plt

    active = [(c, v) for c, v in zip(concepts, ann['concept_vector']) if v > 0]
    active.sort(key=lambda x: x[1], reverse=True)
    fig, axs = _base_figure(img, crop_img, ann['bbox'], f"Concepts: {class_name.capitalize()}",
                            max(6, 0.35 * max(1, len(active))))
    axs[2].set_title(f"Active concepts ({len(active)} found)")
    if active:
        text = "\n".join(f"{c}: {v:.2f}" for c, v in active)
        fontsize = max(6, min(11, int(180 / max(1, len(active)))))
        axs[2].text(0.02, 0.98, text, color='green', fontsize=fontsize, weight='bold', va='top', ha='left',
                    transform=axs[2].transAxes)
    else:
        axs[2].text(0.5, 0.5, "No active concepts", color='red', fontsize=12, weight='bold', va='center',
                    ha='center', transform=axs[2].transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def draw_per_class(img, crop_img, ann, class_name, descriptors, out_path):
    import matplotlib.pyplot as plt

    start = COCO_CLASSES.index(class_name) * DESCRIPTORS_PER_CLASS
    activations = ann['concept_vector'][start:start + DESCRIPTORS_PER_CLASS]
    fig, axs = _base_figure(img, crop_img, ann['bbox'], f"Instance Analysis: {class_name.capitalize()}", 5)
    axs[2].set_title("Descriptors Activation")
    y_pos = 0.8
    for desc, val in zip(descriptors[class_name], activations):
        axs[2].text(0.05, y_pos, f"{desc} ({val})", color='green' if val > 0 else 'red', fontsize=13,
                    weight='bold', va='center', ha='left', wrap=True)
        y_pos -= 0.15
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--mode", required=True, choices=["shared", "per_class"])
    parser.add_argument("--json", required=True, help="Labelled COCO JSON.")
    parser.add_argument("--images_dir", required=True, help="Image folder of the split.")
    parser.add_argument("--concepts", required=True, help="Vocabulary file matching the JSON.")
    parser.add_argument("--out_dir", default="figs", help="Output folder (default: figs).")
    parser.add_argument("--num", type=int, default=5, help="Number of random instances (default: 5).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: unseeded).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = load_json(args.concepts)
    concepts = flatten_concepts(vocab) if args.mode == "shared" else None
    coco_data = load_json(args.json, verbose=True)
    images_map = {img['id']: img['file_name'] for img in coco_data['images']}
    categories_map = {cat['id']: cat['name'] for cat in coco_data['categories']}

    candidates = [a for a in coco_data['annotations']
                  if len(a.get('bbox', [])) == 4 and a['bbox'][2] >= 1 and a['bbox'][3] >= 1 and 'concept_vector' in a]
    rng = random.Random(args.seed)
    for i, ann in enumerate(rng.sample(candidates, min(args.num, len(candidates)))):
        file_name = images_map.get(ann['image_id'])
        class_name = categories_map.get(ann['category_id'])
        if not file_name or not class_name or (args.mode == "per_class" and class_name not in vocab):
            print(f"Skipping annotation {ann['id']}: missing image/category mapping.")
            continue
        img_path = os.path.join(args.images_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Skipping annotation {ann['id']}: image not found at {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        x, y, w, h = ann['bbox']
        crop_img = img.crop((x, y, x + w, y + h))
        out_path = os.path.join(args.out_dir, f"{args.mode}_instance_{i + 1}.png")
        if args.mode == "shared":
            draw_shared(img, crop_img, ann, class_name, concepts, out_path)
        else:
            draw_per_class(img, crop_img, ann, class_name, vocab, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
