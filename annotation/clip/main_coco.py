"""Descriptor-based CLIP scoring of COCO object crops (evaluation and concept labelling).

Processes:
  evaluate           Top-1/Top-5 accuracy, descriptor-based vs. plain CLIP on class names.
  label_instances    Raw 400-long concept vectors (Rule 1/2: CLIP similarities of the
                     true class's descriptors; Rule 3: -1) -> --output. Binarize the
                     result with binarize_concepts.py.
  calc_stats         Per-class Rule 1/2/3 percentages -> <out_dir>/rules_by_class.{txt,png}.
  visualize          Crops + explanations for the dataset indices given with --indices.
  misclassification  Figure of the first misclassified instance among random samples.

Adapted from main.py of Menon & Vondrick (ICLR 2023), see NOTICE.
"""

import argparse
import os
import sys
from pathlib import Path

CLIP_MODELS = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
PROCESSES = ['evaluate', 'visualize', 'misclassification', 'label_instances', 'calc_stats']


def run_evaluate(model, dataloader, description_encodings, label_encodings, device):
    import torch
    import torchmetrics
    from torch.nn import functional as F
    from tqdm import tqdm

    from load_coco import aggregate_similarity

    lang_top1 = torchmetrics.Accuracy().to(device)
    lang_top5 = torchmetrics.Accuracy(top_k=5).to(device)
    clip_top1 = torchmetrics.Accuracy().to(device)
    clip_top5 = torchmetrics.Accuracy(top_k=5).to(device)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            image_encodings = F.normalize(model.encode_image(images))

            image_labels_similarity = image_encodings @ label_encodings.T
            clip_top1(image_labels_similarity, labels)
            clip_top5(image_labels_similarity, labels)

            cumulative = torch.stack([aggregate_similarity(image_encodings @ v.T)
                                      for v in description_encodings.values()], dim=1)
            lang_top1(cumulative.softmax(dim=-1), labels)
            lang_top5(cumulative.softmax(dim=-1), labels)

    results = {
        "Description-based Top-1": 100 * lang_top1.compute().item(),
        "Description-based Top-5": 100 * lang_top5.compute().item(),
        "CLIP-Standard Top-1": 100 * clip_top1.compute().item(),
        "CLIP-Standard Top-5": 100 * clip_top5.compute().item(),
    }
    print("\n" + "=" * 60 + "\nCOCO evaluation results\n" + "=" * 60)
    for key, value in results.items():
        print(f"{key}: {value:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument('--descriptor_file', required=True,
                        help='Per-class descriptors JSON, e.g. ../concepts/per_class_descriptors_top5.json')
    parser.add_argument('--images_path', required=True, help='COCO image folder of the split')
    parser.add_argument('--annotations_path', required=True, help='COCO instances JSON of the split')
    parser.add_argument('--process', default='evaluate', choices=PROCESSES)
    parser.add_argument('--output', default=None,
                        help='label_instances output JSON (default: <out_dir>/<annotations_basename>_concept_vectors.json)')
    parser.add_argument('--out_dir', default='clip_outputs', help='Folder for figures/tables (default: clip_outputs)')
    parser.add_argument('--indices', type=int, nargs='+', default=None, help='Dataset indices for --process visualize')
    parser.add_argument('--model_size', default='ViT-B/32', choices=CLIP_MODELS, help='CLIP backbone')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples per class (default: all)')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.process == 'visualize' and not args.indices:
        parser.error("--process visualize requires --indices")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import clip
    import torch

    from load_coco import (calculate_inference_statistics, compute_description_encodings,
                           compute_instance_concept_vectors, compute_label_encodings, save_concept_vectors_to_json,
                           setup_coco_evaluation, show_from_coco_indices, visualize_random_misclassified_instance)
    from loading_helpers import seed_everything

    hparams, dataset, dataloader, gpt_descriptions, unmodify_dict, label_to_classname, tfms = setup_coco_evaluation(
        descriptor_file=args.descriptor_file, coco_images_path=args.images_path,
        coco_annotations_path=args.annotations_path, model_size=args.model_size, batch_size=args.batch_size,
        max_samples_per_class=args.max_samples, seed=args.seed, num_workers=args.num_workers)
    seed_everything(hparams['seed'])

    print(f"Loading CLIP model ({args.model_size})...")
    device = torch.device(hparams['device'])
    model, _ = clip.load(args.model_size, device=device, jit=False)
    model.eval()
    model.requires_grad_(False)

    with torch.no_grad():
        description_encodings = compute_description_encodings(model, gpt_descriptions, device)
        label_encodings = compute_label_encodings(model, label_to_classname, hparams, device)

    if args.process == "evaluate":
        run_evaluate(model, dataloader, description_encodings, label_encodings, device)

    elif args.process == "visualize":
        show_from_coco_indices(args.indices, dataset, model, gpt_descriptions, unmodify_dict, hparams,
                               out_dir=os.path.join(args.out_dir, "samples"), device=device, tfms=tfms,
                               description_encodings=description_encodings, label_encodings=label_encodings,
                               aggregation='mean', topk=6)

    elif args.process == "misclassification":
        visualize_random_misclassified_instance(dataset, model, gpt_descriptions, description_encodings, hparams,
                                                out_dir=os.path.join(args.out_dir, "bad_instances"),
                                                unmodify_dict=unmodify_dict, device=hparams['device'],
                                                aggregation='mean', n_samples=500, batch_size=128, seed=143,
                                                topk_descriptors=5)

    elif args.process == "label_instances":
        concept_vectors, descriptor_mapping = compute_instance_concept_vectors(
            dataset, model, gpt_descriptions, description_encodings, unmodify_dict, hparams, device=device,
            aggregation='mean', topk_classes=5, topk_descriptors=5)
        split_name = os.path.splitext(os.path.basename(args.annotations_path))[0]
        output = args.output or os.path.join(args.out_dir, f"{split_name}_concept_vectors.json")
        save_concept_vectors_to_json(args.annotations_path, concept_vectors, descriptor_mapping, output)

    elif args.process == "calc_stats":
        calculate_inference_statistics(dataset, model, description_encodings, hparams, out_dir=args.out_dir,
                                       device=device, aggregation='mean', topk_classes=5)


if __name__ == "__main__":
    main()
