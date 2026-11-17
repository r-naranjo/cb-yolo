"""COCO object-crop adaptation of the descriptor-based CLIP classifier.

Adapted from ``load.py`` of Menon & Vondrick, "Visual Classification via
Description from Large Language Models" (ICLR 2023),
https://github.com/sachit-menon/classify_by_description_release (see NOTICE).

Every COCO annotation is cropped to its bbox and embedded with CLIP; each
class is scored by aggregating (mean by default) the cosine similarities
between the crop and the class's descriptor prompts.
"""

import json
import math
import os
import pathlib
import random
import sys
import textwrap
from collections import OrderedDict, defaultdict
from pathlib import Path

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loading_helpers import load_gpt_descriptions, wordify  # noqa: E402
from common.coco_classes import COCO_CLASSES, DESCRIPTORS_PER_CLASS  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class COCODataset(Dataset):
    """COCO object crops. Items are ``(crop_tensor, class_index, annotation_id)``.

    Class indices follow the canonical COCO order restricted to the classes
    present in the annotation file. Boxes are clipped to the image and
    degenerate boxes are skipped.
    """

    def __init__(self, root_dir, annotation_file, transform=None, categories_to_include=None,
                 max_samples_per_class=None):
        self.root_dir = pathlib.Path(root_dir)
        self.transform = transform

        print(f"Loading COCO annotations from {annotation_file}...")
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
        self.category_name_to_id = {cat['name']: cat['id'] for cat in data['categories']}
        self.image_id_to_info = {img['id']: img for img in data['images']}

        if categories_to_include:
            valid_category_ids = [self.category_name_to_id[name] for name in categories_to_include
                                  if name in self.category_name_to_id]
        else:
            valid_category_ids = list(self.category_id_to_name.keys())

        # Classes present in the annotations, in canonical COCO order
        # (sorted alphabetically if none of them is a COCO class).
        present = []
        for ann in data['annotations']:
            if ann['category_id'] in valid_category_ids:
                name = self.category_id_to_name[ann['category_id']]
                if name not in present:
                    present.append(name)
        ordered = [c for c in COCO_CLASSES if c in present]
        self.classes = ordered if ordered else sorted(present)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        class_counts = {cls: 0 for cls in self.classes}
        for ann in data['annotations']:
            cat_id = ann['category_id']
            if cat_id not in valid_category_ids:
                continue
            cat_name = self.category_id_to_name[cat_id]
            if max_samples_per_class is not None and class_counts[cat_name] >= max_samples_per_class:
                continue
            image_id = ann['image_id']
            if image_id not in self.image_id_to_info:
                continue

            img_info = self.image_id_to_info[image_id]
            x1, y1, w, h = ann['bbox']
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_info['width'], x1 + w)
            y2 = min(img_info['height'], y1 + h)
            if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
                continue

            self.samples.append({
                'annotation_id': ann['id'],
                'image_id': image_id,
                'image_file': img_info['file_name'],
                'bbox': [x1, y1, x2, y2],
                'category_id': cat_id,
                'category_name': cat_name,
                'label': self.class_to_idx[cat_name],
            })
            class_counts[cat_name] += 1

        print(f"Created COCO dataset with {len(self.samples)} samples across {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def _image_path(self, sample):
        image_path = self.root_dir / sample['image_file']
        if not image_path.exists():
            for ext in ['.jpg', '.png', '.jpeg']:
                alt_path = image_path.with_suffix(ext)
                if alt_path.exists():
                    return alt_path
        return image_path

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._image_path(sample)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
            return image, sample['label'], sample['annotation_id']

        try:
            cropped_image = image.crop(tuple(sample['bbox']))
        except Exception as e:
            print(f"Error cropping image {image_path} with bbox {sample['bbox']}: {e}")
            cropped_image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            cropped_image = self.transform(cropped_image)
        return cropped_image, sample['label'], sample['annotation_id']

    def get_class_name(self, class_idx):
        return self.classes[class_idx]

    def get_class_index(self, class_name):
        return self.class_to_idx[class_name]


def _transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def aggregate_similarity(similarity_matrix_chunk, aggregation_method='mean'):
    if aggregation_method == 'max':
        return similarity_matrix_chunk.max(dim=1)[0]
    elif aggregation_method == 'sum':
        return similarity_matrix_chunk.sum(dim=1)
    elif aggregation_method == 'mean':
        return similarity_matrix_chunk.mean(dim=1)
    raise ValueError("Unknown aggregate_similarity")


def compute_description_encodings(model, gpt_descriptions, device):
    """Normalised CLIP text embeddings of every descriptor prompt, per class."""
    description_encodings = OrderedDict()
    for k, v in gpt_descriptions.items():
        tokens = clip.tokenize(v).to(device)
        description_encodings[k] = F.normalize(model.encode_text(tokens))
    return description_encodings


def compute_label_encodings(model, label_to_classname, hparams, device):
    """Normalised CLIP text embeddings of the bare class names (plain-CLIP baseline)."""
    labels = [hparams['label_before_text'] + wordify(name) + hparams['label_after_text'] for name in label_to_classname]
    label_tokens = clip.tokenize(labels).to(device)
    return F.normalize(model.encode_text(label_tokens))


def setup_coco_evaluation(descriptor_file, coco_images_path, coco_annotations_path, model_size="ViT-B/32",
                          batch_size=64, max_samples_per_class=None, categories_to_include=None, seed=1,
                          num_workers=4):
    """Build hparams, dataset, (shuffled) dataloader and descriptor prompts."""
    hparams = {
        'model_size': model_size,
        'dataset': 'coco',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'category_name_inclusion': 'prepend',
        'apply_descriptor_modification': True,
        'verbose': False,
        'image_size': {'ViT-L/14@336px': 336, 'RN50x4': 288, 'RN50x16': 384, 'RN50x64': 448}.get(model_size, 224),
        'before_text': "a photo of ",
        'label_before_text': "",
        'between_text': ', ',
        'after_text': '.',
        'label_after_text': '.',
        'unmodify': True,
        'seed': seed,
        'descriptor_fname': descriptor_file,
    }

    tfms = _transform(hparams['image_size'])
    dataset = COCODataset(root_dir=coco_images_path, annotation_file=coco_annotations_path, transform=tfms,
                          categories_to_include=categories_to_include, max_samples_per_class=max_samples_per_class)

    gpt_descriptions, unmodify_dict = load_gpt_descriptions(hparams, dataset.classes)
    label_to_classname = list(gpt_descriptions.keys())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return hparams, dataset, dataloader, gpt_descriptions, unmodify_dict, label_to_classname, tfms


def _descriptor_list(raw_descrs):
    if isinstance(raw_descrs, str):
        return [s for s in raw_descrs.splitlines() if s.strip()] or [raw_descrs]
    return list(raw_descrs)


def _ordered_loader(dataset, hparams):
    return DataLoader(dataset, batch_size=hparams.get("batch_size", 128), shuffle=False,
                      num_workers=hparams.get("num_workers", 4), pin_memory=True)


# --------------------------------------------------------------------------- #
# Qualitative inspection
# --------------------------------------------------------------------------- #
def _get_cropped_pil(dataset, sample):
    image_path = dataset._image_path(sample)
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return Image.new('RGB', (224, 224), color='black')
    try:
        return img.crop(tuple(sample['bbox']))
    except Exception as e:
        print(f"Error cropping {image_path} bbox {sample['bbox']}: {e}")
        return Image.new('RGB', (224, 224), color='black')


def show_from_coco_indices(indices, dataset, model, gpt_descriptions, unmodify_dict, hparams, out_dir,
                           device=None, tfms=None, description_encodings=None, label_encodings=None,
                           aggregation='mean', topk=8):
    """Save the crop of each dataset index and print plain-CLIP vs descriptor-based predictions."""
    model_device = device or hparams.get('device', 'cpu')
    model.to(model_device)
    model.eval()
    tfms = tfms if tfms is not None else dataset.transform
    label_names = list(gpt_descriptions.keys())
    os.makedirs(out_dir, exist_ok=True)

    def _aggregate(sim_vec):
        if aggregation == 'mean':
            return float(sim_vec.mean().item())
        elif aggregation == 'max':
            return float(sim_vec.max().item())
        elif aggregation == 'sum':
            return float(sim_vec.sum().item())
        raise ValueError("Unknown aggregation mode")

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(f"Index {idx} out of range (len={len(dataset)})")
            continue

        sample = dataset.samples[idx]
        crop_pil = _get_cropped_pil(dataset, sample)

        fig = plt.figure(figsize=(4, 4))
        plt.imshow(crop_pil)
        plt.axis('off')
        plt.title(f"Index {idx} - {sample['image_file']} bbox={sample['bbox']}")
        plt.savefig(os.path.join(out_dir, f"sample_{idx}.png"))
        plt.close(fig)

        true_label_name = dataset.get_class_name(sample['label'])
        print(f"Index: {idx}")
        print(f"Image file: {sample['image_file']} | bbox: {sample['bbox']}")
        print(f"True label: {true_label_name} (idx={sample['label']})")

        with torch.no_grad():
            image_feat = F.normalize(model.encode_image(tfms(crop_pil).unsqueeze(0).to(model_device)), dim=1)
            label_sims = (label_encodings.to(model_device) @ image_feat.squeeze().unsqueeze(-1)).squeeze()
            best_clip_idx = int(label_sims.argmax().item())
        print(f"Predicted label (CLIP class names): {label_names[best_clip_idx]} (idx={best_clip_idx}), "
              f"similarity={float(label_sims[best_clip_idx].item()):.4f}")

        class_scores, per_class_sims = [], {}
        for class_name, descr_enc in description_encodings.items():
            sims = (descr_enc.to(model_device) @ image_feat.squeeze()).detach().cpu().view(-1)
            per_class_sims[class_name] = sims
            class_scores.append((class_name, _aggregate(sims)))
        best_descr_name, best_descr_score = max(class_scores, key=lambda x: x[1])
        print(f"Predicted label (descriptor-based): {best_descr_name}, aggregated_score={100. * best_descr_score:.2f}")
        print(f"Total similarity to true label '{true_label_name}': {100. * dict(class_scores)[true_label_name]:.2f}")

        def _print_top_descriptors(class_name):
            sims = per_class_sims[class_name]
            order = torch.argsort(sims, descending=True)
            descr_list = _descriptor_list(gpt_descriptions[class_name])
            print(f"\nTop {topk} descriptors for class '{class_name}':")
            for i in range(min(topk, len(order))):
                di = int(order[i].item())
                text = descr_list[di] if di < len(descr_list) else f"(descriptor {di})"
                text = unmodify_dict.get(class_name, {}).get(text, text) if unmodify_dict else text
                print(f"  {i + 1:2d}. {text}\t{100. * float(sims[di].item()):.2f}")

        _print_top_descriptors(best_descr_name)
        if true_label_name != best_descr_name:
            print("\n-- True-label top descriptors --")
            _print_top_descriptors(true_label_name)
        print("\n" + "=" * 80 + "\n")


def visualize_random_misclassified_instance(dataset, model, gpt_descriptions, description_encodings, hparams,
                                            out_dir, unmodify_dict=None, device=None, aggregation='mean',
                                            n_samples=5000, batch_size=64, seed=None, topk_classes=5,
                                            topk_descriptors=5):
    """Sample instances and save a figure for the first misclassified one.

    Figure: original image | crop | true class, predicted class and the
    top-k predicted classes with their top descriptors.
    """
    os.makedirs(out_dir, exist_ok=True)
    model_device = device or hparams.get("device", "cpu")
    model.to(model_device)
    model.eval()

    if seed is None:
        seed = hparams.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ds_len = len(dataset)
    if ds_len == 0:
        raise ValueError("Dataset is empty.")
    indices = np.random.choice(np.arange(ds_len), size=n_samples, replace=n_samples > ds_len)
    label_names = list(gpt_descriptions.keys())

    def _aggregate(sim_tensor):
        if aggregation == 'mean':
            return sim_tensor.mean(dim=0)
        elif aggregation == 'max':
            return sim_tensor.max(dim=0)[0]
        elif aggregation == 'sum':
            return sim_tensor.sum(dim=0)
        raise ValueError("Unknown aggregation")

    def _top_descriptors(class_name, image_feat):
        sims = (description_encodings[class_name].to(model_device) @ image_feat.squeeze(0)).detach().cpu().float()
        vals, idxs = torch.topk(sims, k=min(topk_descriptors, sims.numel()), largest=True)
        descr_list = _descriptor_list(gpt_descriptions[class_name])
        pairs = []
        for v, i in zip(vals.tolist(), idxs.tolist()):
            text = descr_list[i] if i < len(descr_list) else f"(descriptor {i})"
            if unmodify_dict:
                text = unmodify_dict.get(class_name, {}).get(text, text)
            pairs.append((text, v))
        return pairs

    for b in range(math.ceil(n_samples / batch_size)):
        batch_idxs = indices[b * batch_size:min((b + 1) * batch_size, n_samples)]
        items = [dataset[int(idx)] for idx in batch_idxs]
        img_batch = torch.stack([it[0] for it in items], dim=0).to(model_device)
        labels = torch.tensor([int(it[1]) for it in items], dtype=torch.long, device=model_device)

        with torch.no_grad():
            feats = F.normalize(model.encode_image(img_batch), dim=1)
            class_scores = torch.empty((feats.shape[0], len(label_names)), device=model_device)
            for ci, cname in enumerate(label_names):
                class_scores[:, ci] = _aggregate(description_encodings[cname].to(model_device) @ feats.t())
            preds = class_scores.argmax(dim=1)

        mis_mask = preds != labels
        if not mis_mask.any():
            continue

        local_idx = mis_mask.nonzero(as_tuple=True)[0][0].item()
        global_idx = int(batch_idxs[local_idx])
        sample = dataset.samples[global_idx]
        original = Image.open(dataset._image_path(sample)).convert("RGB")
        crop = original.crop(tuple(sample['bbox']))
        image_feat = feats[local_idx:local_idx + 1, :]

        agg_scores_img = class_scores[local_idx].cpu().float()
        k_classes = min(topk_classes, agg_scores_img.numel())
        topk_vals, topk_idxs = torch.topk(agg_scores_img, k=k_classes, largest=True)
        topk_info = [(label_names[int(ci)], float(cs), _top_descriptors(label_names[int(ci)], image_feat))
                     for cs, ci in zip(topk_vals.tolist(), topk_idxs.tolist())]

        pred_idx = int(preds[local_idx].item())
        pred_class_name = label_names[pred_idx]
        pred_descrs = next((d for c, _, d in topk_info if c == pred_class_name), None)
        if pred_descrs is None:
            pred_descrs = _top_descriptors(pred_class_name, image_feat)
        true_label_idx = int(labels[local_idx].item())
        true_class_name = dataset.get_class_name(true_label_idx)

        lines = [f"Index: {global_idx}", "",
                 f"True class: {true_class_name} (idx={true_label_idx})",
                 f"Predicted (descriptor-based): {pred_class_name} (idx={pred_idx})", "",
                 f"Top-{min(topk_descriptors, len(pred_descrs))} descriptors for predicted class:"]
        lines += [f"  {i}. {t} - {100. * s:.2f}" for i, (t, s) in enumerate(pred_descrs, start=1)]
        lines += ["", f"Top-{k_classes} predicted classes and their top-{topk_descriptors} descriptors:"]
        for rank_pos, (cname, cscore, descr_pairs) in enumerate(topk_info, start=1):
            lines.append(f"{rank_pos}. {cname} - score {100. * cscore:.2f}")
            lines += [f"     {j}. {dt} - {100. * ds:.2f}" for j, (dt, ds) in enumerate(descr_pairs, start=1)]
            lines.append("")
        wrapped = []
        for ln in lines:
            wrapped.extend(textwrap.wrap(ln, width=60) if ln else [""])

        fig = plt.figure(constrained_layout=False, figsize=(18, 8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.4], wspace=0.05)
        for col, (im, title) in enumerate([(original, "Original Image"), (crop, "Cropped Region")]):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        ax2.text(0.01, 0.99, "\n".join(wrapped), va='top', ha='left', fontsize=10, family='monospace')

        out_path = os.path.join(out_dir, f"misclassified_{global_idx}.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved misclassified instance: {out_path}")
        return out_path

    print("No misclassified instances found in sampled indices.")
    return None


# --------------------------------------------------------------------------- #
# Concept labelling (Rule 1 / 2 / 3)
# --------------------------------------------------------------------------- #
def build_descriptor_layout(gpt_descriptions, unmodify_dict, n_per_class=DESCRIPTORS_PER_CLASS):
    """Canonical 80 x ``n_per_class`` slot layout.

    Class ``COCO_CLASSES[i]`` owns slots ``[i*n, i*n + n)`` in the order of
    its descriptors. Missing classes / descriptors get placeholder names so
    the layout is always 400 long. Returns
    ``(global_descriptor_mapping[(class, prompt)] -> idx, reverse_mapping[idx] -> (class, descriptor))``.
    """
    global_descriptor_mapping, reverse_descriptor_mapping = {}, {}
    counter = 0
    for class_name in COCO_CLASSES:
        raw_descrs = gpt_descriptions.get(class_name, None)
        if raw_descrs is None:
            descr_list = [f"__MISSING__{class_name}__{k}" for k in range(n_per_class)]
        else:
            if isinstance(raw_descrs, str):
                descr_list = [s.strip() for s in raw_descrs.splitlines() if s.strip()] or [raw_descrs.strip()]
            else:
                descr_list = [s.strip() for s in raw_descrs]
            for k in range(len(descr_list), n_per_class):
                descr_list.append(f"__PAD__{class_name}__{k}")
        for descr in descr_list[:n_per_class]:
            global_descriptor_mapping[(class_name, descr.strip())] = counter
            original = unmodify_dict.get(descr, descr) if unmodify_dict else descr
            reverse_descriptor_mapping[counter] = (class_name, original)
            counter += 1
    return global_descriptor_mapping, reverse_descriptor_mapping


def compute_instance_concept_vectors(dataset, model, gpt_descriptions, description_encodings, unmodify_dict,
                                     hparams, device=None, aggregation='mean', topk_classes=5, topk_descriptors=5):
    """Raw (continuous) 400-long concept vector per annotation.

    For each crop the classes are ranked by aggregated descriptor similarity.
      Rule 1/2 (true class is top-1 / within top-``topk_classes``): the true
        class's slots receive the CLIP similarities of its top
        ``topk_descriptors`` descriptors.
      Rule 3 (true class not in the top-k): the true class's first
        ``topk_descriptors`` slots are set to -1.
    All other slots are 0. Returns ``(vectors_by_annotation_id, reverse_descriptor_mapping)``.
    """
    model_device = device or hparams.get("device", "cpu")
    model.to(model_device)
    model.eval()

    global_descriptor_mapping, reverse_descriptor_mapping = build_descriptor_layout(gpt_descriptions, unmodify_dict)
    total_descriptors = len(reverse_descriptor_mapping)
    print(f"Global descriptor layout: {total_descriptors} slots")

    concept_vectors_dict = {}
    n_nonzero = 0
    with torch.no_grad():
        for images, labels, annotation_ids in tqdm(_ordered_loader(dataset, hparams), desc="Computing concept vectors"):
            annotation_ids = [int(x) for x in (annotation_ids.cpu().numpy() if isinstance(annotation_ids, torch.Tensor)
                                               else annotation_ids)]
            image_feats = F.normalize(model.encode_image(images.to(model_device)), dim=1)
            true_class_names = [dataset.get_class_name(int(ll)) for ll in labels.cpu().numpy()]

            for i in range(images.shape[0]):
                true_class_name = true_class_names[i]
                img_feat = image_feats[i:i + 1]

                all_descriptor_scores, class_scores = {}, {}
                for class_name, descr_enc in description_encodings.items():
                    sims = (descr_enc.to(model_device) @ img_feat.squeeze()).detach().cpu().float()
                    all_descriptor_scores[class_name] = sims
                    if aggregation == 'mean':
                        class_scores[class_name] = float(sims.mean().item())
                    elif aggregation == 'max':
                        class_scores[class_name] = float(sims.max().item())
                    elif aggregation == 'sum':
                        class_scores[class_name] = float(sims.sum().item())
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")

                sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_classes = [c for c, _ in sorted_classes[:topk_classes]]
                in_top_k = true_class_name in top_k_classes  # includes Rule 1 (top-1)

                concept_vector = np.zeros(total_descriptors, dtype=float)
                descr_list = _descriptor_list(gpt_descriptions.get(true_class_name, []))
                if in_top_k:
                    true_sims = all_descriptor_scores.get(true_class_name,
                                                          torch.zeros((DESCRIPTORS_PER_CLASS,), dtype=torch.float32))
                    for di in torch.argsort(true_sims, descending=True)[:topk_descriptors].tolist():
                        descr_text = descr_list[di].strip() if di < len(descr_list) else f"__PAD__{true_class_name}__{di}"
                        gidx = global_descriptor_mapping.get((true_class_name, descr_text))
                        if gidx is not None:
                            concept_vector[gidx] = float(true_sims[di].item())
                        else:
                            print(f"[WARN] global idx not found for ({true_class_name}, {descr_text})")
                else:
                    descrs = descr_list[:topk_descriptors] if descr_list else \
                        [f"__PAD__{true_class_name}__{k}" for k in range(topk_descriptors)]
                    for descr in descrs:
                        gidx = global_descriptor_mapping.get((true_class_name, descr.strip()))
                        if gidx is not None:
                            concept_vector[gidx] = -1.0
                        else:
                            print(f"[WARN] global idx not found for ({true_class_name}, {descr.strip()})")

                concept_vectors_dict[annotation_ids[i]] = concept_vector.tolist()
                n_nonzero += int(np.count_nonzero(concept_vector) > 0)

    print(f"Instances: {len(concept_vectors_dict)}, with non-zero vector: {n_nonzero}")
    return concept_vectors_dict, reverse_descriptor_mapping


def save_concept_vectors_to_json(coco_annotations_path, concept_vectors_dict, descriptor_mapping, output_path):
    """Write the input COCO JSON plus ``concept_names`` and a ``concept_vector`` per annotation.

    Annotations skipped by the dataset (degenerate boxes) get an all-zero vector.
    """
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)

    concept_names = [""] * len(descriptor_mapping)
    for idx, (class_name, descriptor) in descriptor_mapping.items():
        concept_names[idx] = f"{class_name}: {descriptor}"
    coco_data['concept_names'] = concept_names

    for ann in coco_data['annotations']:
        ann['concept_vector'] = concept_vectors_dict.get(ann['id'], [0.0] * len(concept_names))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"Saved concept vectors to {output_path}")
    print(f"Total annotations: {len(coco_data['annotations'])}, concept vector dimension: {len(concept_names)}")
    return output_path


def calculate_inference_statistics(dataset, model, description_encodings, hparams, out_dir, device=None,
                                   aggregation='mean', topk_classes=5):
    """Per-class share of Rule 1 (top-1 correct), Rule 2 (top-k) and Rule 3 (wrong).

    Prints a table and writes ``<out_dir>/rules_by_class.{txt,png}``.
    """
    model_device = device or hparams.get("device", "cpu")
    model.to(model_device)
    model.eval()
    stats = defaultdict(lambda: {'r1': 0, 'r2': 0, 'r3': 0, 'total': 0})
    class_names = list(description_encodings.keys())

    print(f"Computing statistics using aggregation: {aggregation}...")
    with torch.no_grad():
        for images, labels, _ in tqdm(_ordered_loader(dataset, hparams), desc="Calculating statistics"):
            batch_size = images.shape[0]
            image_feats = F.normalize(model.encode_image(images.to(model_device)), dim=1)
            true_class_names = [dataset.get_class_name(int(ll)) for ll in labels.cpu().numpy()]

            batch_scores = torch.zeros((batch_size, len(class_names)), device=model_device)
            for c_idx, class_name in enumerate(class_names):
                sims = image_feats @ description_encodings[class_name].to(model_device).T
                if aggregation == 'max':
                    batch_scores[:, c_idx] = sims.max(dim=1)[0]
                elif aggregation == 'sum':
                    batch_scores[:, c_idx] = sims.sum(dim=1)
                else:
                    batch_scores[:, c_idx] = sims.mean(dim=1)

            topk_indices = torch.topk(batch_scores, k=topk_classes, dim=1)[1].cpu().numpy()
            for i in range(batch_size):
                true_name = true_class_names[i]
                stats[true_name]['total'] += 1
                if true_name not in class_names:
                    stats[true_name]['r3'] += 1
                    continue
                true_idx = class_names.index(true_name)
                if topk_indices[i][0] == true_idx:
                    stats[true_name]['r1'] += 1
                elif true_idx in topk_indices[i]:
                    stats[true_name]['r2'] += 1
                else:
                    stats[true_name]['r3'] += 1

    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, "rules_by_class.txt")
    png_path = os.path.join(out_dir, "rules_by_class.png")

    sorted_keys = [c for c in COCO_CLASSES if c in stats] + sorted(c for c in stats if c not in COCO_CLASSES)
    plot_labels, p_r1_list, p_r2_list, p_r3_list = [], [], [], []
    header = (f"{'ID':<4} {'Class Name':<20} | {'Rule 1 (Correct)':<18} | {'Rule 2 (Top 5)':<18} | "
              f"{'Rule 3 (Wrong)':<18} | {'Total':<6}")
    with open(txt_path, 'w') as f:
        print(header)
        print("-" * len(header))
        f.write(header + "\n" + "-" * len(header) + "\n")
        for cls_name in sorted_keys:
            s = stats[cls_name]
            total = s['total']
            if total == 0:
                continue
            p_r1, p_r2, p_r3 = (100 * s[r] / total for r in ('r1', 'r2', 'r3'))
            id_str = f"c{COCO_CLASSES.index(cls_name)}" if cls_name in COCO_CLASSES else "?"
            r1_str = f"{s['r1']:<4} ({p_r1:>5.1f}%)"
            r2_str = f"{s['r2']:<4} ({p_r2:>5.1f}%)"
            r3_str = f"{s['r3']:<4} ({p_r3:>5.1f}%)"
            line = f"{id_str:<4} {cls_name:<20} | {r1_str:<18} | {r2_str:<18} | {r3_str:<18} | {total:<6}"
            print(line)
            f.write(line + "\n")
            plot_labels.append(cls_name)
            p_r1_list.append(p_r1)
            p_r2_list.append(p_r2)
            p_r3_list.append(p_r3)
    print(f"Saved table to {txt_path}")

    if plot_labels:
        plt.figure(figsize=(max(12, len(plot_labels) * 0.3), 8))
        x = np.arange(len(plot_labels))
        width = 0.25
        plt.bar(x - width, p_r1_list, width, label='Rule 1 (Correct)', color='green', alpha=0.7)
        plt.bar(x, p_r2_list, width, label='Rule 2 (Top 5)', color='#FFD700', alpha=0.8)
        plt.bar(x + width, p_r3_list, width, label='Rule 3 (Wrong)', color='red', alpha=0.7)
        plt.xlabel('Classes', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        plt.title(f'Inference Rules by Class (Aggregation: {aggregation})', fontsize=14, fontweight='bold')
        plt.xticks(x, plot_labels, rotation=90, fontsize=9)
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"Saved plot to {png_path}")
    return stats
