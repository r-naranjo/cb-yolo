#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random search over two CB-YOLOX hyper-parameters:
  - concept_loss_weight   (w_c): weight of the concept loss in the total loss
  - simota_concept_weight (sigma): weight of the concept cost in SimOTA

Each trial trains the given experiment for a few epochs (single GPU, in this
process) with the sampled values and records the training losses and the
final validation COCO AP50:95. Results are appended to a JSON file after every
trial, so an interrupted search can be continued with --resume.

Example (from the cbyolox/ directory; typically on a training subset):
    python tools/grid_search/grid_search_concepts.py \
        -f exps/cb_yolox/cb_yolox_s_scratch.py -b 16 --epochs 10 --n_trials 20 \
        --train_ann instances_train2017_subset.json \
        --output YOLOX_outputs/random_search/results.json

Trial i is written to <trials_dir>/random_trial_<iii>_clw<w>_sigma<s>/
(default trials_dir: the directory of --output).
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Random search for CB-YOLOX hyper-parameters")
    p.add_argument("-f", "--exp_file", required=True, help="experiment file")
    p.add_argument("-b", "--batch", type=int, default=16, help="batch size per trial")
    p.add_argument("-c", "--ckpt", default=None,
                   help="checkpoint to initialise every trial (as `-c` in tools/train.py)")
    p.add_argument("--epochs", type=int, default=5, help="training epochs per trial")
    p.add_argument("--n_trials", type=int, default=20, help="number of sampled configurations")
    p.add_argument("--seed", type=int, default=42, help="seed for sampling configurations")
    p.add_argument("--output", required=True, help="results JSON (updated after every trial)")
    p.add_argument("--trials_dir", default=None,
                   help="root of the per-trial output directories (default: dir of --output)")
    p.add_argument("--train_ann", default=None,
                   help="override exp.train_ann (e.g. a subset annotation file)")
    p.add_argument("--resume", action="store_true",
                   help="skip configurations already present in --output")
    p.add_argument("--fp16", action="store_true", help="mixed precision training")
    p.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb", "mlflow"])

    p.add_argument("--clw_range", nargs=2, type=float, default=[0.1, 5.0], metavar=("MIN", "MAX"),
                   help="search range of concept_loss_weight")
    p.add_argument("--clw_log", action="store_true", help="sample concept_loss_weight in log-space")
    p.add_argument("--sigma_range", nargs=2, type=float, default=[0.1, 3.0], metavar=("MIN", "MAX"),
                   help="search range of simota_concept_weight")
    p.add_argument("--sigma_log", action="store_true",
                   help="sample simota_concept_weight in log-space")
    return p.parse_args()


def sample(lo, hi, log):
    if log:
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    return random.uniform(lo, hi)


def build_random_configs(n, clw_range, clw_log, sigma_range, sigma_log, seed):
    """Draw `n` independent (concept_loss_weight, sigma) pairs."""
    random.seed(seed)
    configs = []
    for _ in range(n):
        clw = sample(*clw_range, clw_log)
        sigma = sample(*sigma_range, sigma_log)
        configs.append((round(clw, 6), round(sigma, 6)))
    return configs


def trial_dir_name(trial_idx, clw, sigma):
    tag = f"clw{clw}_sigma{sigma}".replace(".", "p")
    return f"random_trial_{trial_idx:03d}_{tag}"


def load_results(path):
    if os.path.isfile(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_results(results, path):
    """Atomically overwrite the JSON results file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, path)


def extract_metrics(trainer):
    metrics = {}
    for key in ("total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss", "concept_loss"):
        if key in trainer.meter:
            metrics[f"train_{key}"] = float(trainer.meter[key].global_avg)
    metrics["val_mAP50_95"] = float(trainer.best_ap)
    return metrics


def run_single_trial(args, trial_idx, clw, sigma, trials_dir):
    from yolox.exp import get_exp

    exp = get_exp(args.exp_file, None)
    if exp.num_concepts is None:
        raise ValueError("The experiment has no concept bottleneck (num_concepts is None).")
    if args.train_ann:
        exp.train_ann = args.train_ann
    exp.max_epoch = args.epochs
    exp.warmup_epochs = min(1, args.epochs)
    exp.eval_interval = args.epochs  # evaluate once, after the last epoch
    exp.save_history_ckpt = False
    exp.concept_loss_weight = clw
    exp.simota_concept_weight = sigma
    exp.use_concepts_in_simota = True
    exp.output_dir = trials_dir

    trainer_args = argparse.Namespace(
        experiment_name=trial_dir_name(trial_idx, clw, sigma),
        exp_file=args.exp_file,
        batch_size=args.batch,
        fp16=args.fp16,
        resume=False,
        ckpt=args.ckpt,
        start_epoch=None,
        cache=None,
        occupy=False,
        logger=args.logger,
        num_machines=1,
        machine_rank=0,
        dist_backend="nccl",
        dist_url=None,
        devices=1,
        opts=[],
    )
    trainer = exp.get_trainer(trainer_args)

    t0 = time.time()
    result = {"trial_idx": trial_idx, "concept_loss_weight": clw, "sigma_": sigma}
    try:
        trainer.train()
    except Exception as exc:
        print(f"[ERROR] Trial {trial_idx} failed: {exc}")
        result.update(status="FAILED", error=str(exc), elapsed_s=round(time.time() - t0, 1))
        return result
    result.update(status="OK", elapsed_s=round(time.time() - t0, 1), **extract_metrics(trainer))
    return result


def main():
    args = parse_args()
    trials_dir = args.trials_dir or os.path.dirname(os.path.abspath(args.output))
    Path(trials_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    configs = build_random_configs(
        args.n_trials, tuple(args.clw_range), args.clw_log,
        tuple(args.sigma_range), args.sigma_log, args.seed,
    )
    results = load_results(args.output) if args.resume else []
    done = {(r["concept_loss_weight"], r["sigma_"]) for r in results}

    for trial_idx, (clw, sigma) in enumerate(configs):
        print(f"\n[TRIAL {trial_idx + 1}/{len(configs)}] concept_loss_weight={clw} sigma={sigma}")
        if (clw, sigma) in done:
            print("already computed, skipping")
            continue
        result = run_single_trial(args, trial_idx, clw, sigma, trials_dir)
        results.append(result)
        save_results(results, args.output)
        print(f"[RESULT] {result}")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
