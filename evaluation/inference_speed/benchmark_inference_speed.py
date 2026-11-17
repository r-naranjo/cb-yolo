#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Inference latency of plain YOLOX vs. CB-YOLOX (S/M/L, CPU/GPU).

Both variants are built from the same CB-YOLOX checkpoint and experiment
file: the "cbm" variant is the model as trained; the "normal" variant is the
same experiment with num_concepts=None (plain 1x1 cls_preds head), loaded
non-strictly from the same checkpoint (concept_preds/concept_to_cls are
skipped). Latency only depends on shapes and ops, so this isolates the
compute added by the concept bottleneck.

Timed: `model(x)` in eval mode (backbone + FPN + head, including box
decoding) on a synthetic input; NMS is excluded.

Methodology:
- both variants of a (size, device, precision) configuration are warmed up
  and timed interleaved on the same input, alternating which goes first;
- torch.backends.cudnn.benchmark = True;
- Conv+BN fusion (yolox.utils.fuse_model) unless --no_fuse;
- channels_last on GPU unless --no_channels_last;
- TF32 set explicitly (--tf32, recorded in the results);
- CPU threads: --num_threads, else the size of the process CPU affinity mask;
- the median (p50) latency is the headline statistic.

CPU runs are always fp32; GPU runs use each of --precisions.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from prettytable import PrettyTable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.model import build_exp, load_weights  # noqa: E402

DEFAULT_SIZES = ["s", "m", "l"]
DEFAULT_EXP_FILE_TEMPLATE = "cbyolox/exps/cb_yolox/cb_yolox_{size}.py"
DEFAULT_CKPT_TEMPLATE = "YOLOX_outputs/cb_yolox_{size}/best_ckpt.pth"

VARIANTS = [(False, "normal"), (True, "cbm")]


def make_parser():
    parser = argparse.ArgumentParser(description="CB-YOLOX vs. plain YOLOX inference speed benchmark")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES, choices=["s", "m", "l"],
                        help="Which model sizes to benchmark")
    parser.add_argument("--exp_file_template", type=str, default=DEFAULT_EXP_FILE_TEMPLATE,
                        help="Python str.format() template with a {size} placeholder")
    parser.add_argument("--ckpt_template", type=str, default=DEFAULT_CKPT_TEMPLATE,
                        help="Python str.format() template with a {size} placeholder")

    parser.add_argument("--devices", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"],
                        help="Devices to benchmark; 'gpu' is skipped with a warning if CUDA is unavailable")
    parser.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"],
                        help="Precisions to test on GPU (CPU is always fp32 regardless of this flag)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per forward pass (1 = single-image real-time latency)")
    parser.add_argument("--test_size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"),
                        help="Input spatial size (must match the checkpoints' own exp.test_size)")

    parser.add_argument("--warmup", type=int, default=100, help="Untimed warmup passes per variant (GPU)")
    parser.add_argument("--iters", type=int, default=300, help="Timed passes per variant (GPU)")
    parser.add_argument("--cpu_warmup", type=int, default=10, help="Untimed warmup passes per variant (CPU)")
    parser.add_argument("--cpu_iters", type=int, default=50, help="Timed passes per variant (CPU)")

    parser.add_argument("--num_threads", type=int, default=None,
                        help="CPU threads for torch (default: size of the process CPU affinity mask)")
    parser.add_argument("--tf32", choices=["on", "off"], default="on",
                        help="Allow TF32 for cuDNN convs and CUDA matmuls (PyTorch default for convs is on)")
    parser.add_argument("--no_fuse", action="store_true", help="Disable Conv+BN fusion")
    parser.add_argument("--no_channels_last", action="store_true",
                        help="Keep NCHW memory format on GPU instead of channels_last")

    parser.add_argument("--output_dir", type=str, default="outputs/inference_speed")
    return parser


def resolve_num_threads(requested):
    if requested is not None:
        return requested
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return torch.get_num_threads()


def build_model(exp_file, ckpt_path, use_cbm, device, fp16, fuse, channels_last):
    """Fresh experiment + model per call (exp.get_model() caches its model).
    use_cbm=False builds the plain head and loads the checkpoint non-strictly."""
    from yolox.utils import fuse_model

    exp = build_exp(exp_file, cbm=use_cbm)
    model = exp.get_model()
    model.eval()
    model.head.clear_interventions()
    model = load_weights(model, ckpt_path, strict=use_cbm)
    if fuse:
        model = fuse_model(model)
    model.eval()

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
    return model


def make_input(device, fp16, batch_size, test_size, channels_last):
    h, w = test_size
    dtype = torch.float16 if (device == "gpu" and fp16) else torch.float32
    x = torch.randn(batch_size, 3, h, w, dtype=dtype)
    if device == "gpu":
        x = x.cuda()
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
    return x


def summarize(times_ms, batch_size):
    times_ms = np.asarray(times_ms, dtype=np.float64)
    return {
        'mean_ms': float(times_ms.mean()),
        'std_ms': float(times_ms.std()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'fps': float(batch_size * 1000.0 / np.percentile(times_ms, 50)),
    }


def time_interleaved(models, x, device, warmup, iters, batch_size):
    """Warm up all variants, then time them interleaved on the same input,
    alternating which variant goes first on each iteration so neither one
    systematically gets the 'cold' slot."""
    names = list(models.keys())
    is_gpu = (device == "gpu")

    def sync():
        if is_gpu:
            torch.cuda.synchronize()

    times = {n: [] for n in names}
    with torch.no_grad():
        for _ in range(warmup):
            for n in names:
                _ = models[n](x)
        sync()

        for i in range(iters):
            order = names if i % 2 == 0 else names[::-1]
            for n in order:
                sync()
                t0 = time.perf_counter()
                _ = models[n](x)
                sync()
                t1 = time.perf_counter()
                times[n].append((t1 - t0) * 1000.0)

    return {n: summarize(times[n], batch_size) for n in names}


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)

    devices = list(args.devices)
    if "gpu" in devices and not torch.cuda.is_available():
        print("[WARN] --devices includes 'gpu' but CUDA is not available on this node - skipping gpu.")
        devices = [d for d in devices if d != "gpu"]
    if not devices:
        raise RuntimeError("No usable devices left to benchmark.")

    # ------------------------------------------------------------------
    # Global backend settings (recorded alongside every result row).
    # ------------------------------------------------------------------
    num_threads = resolve_num_threads(args.num_threads)
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = True
    tf32 = (args.tf32 == "on")
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    fuse = not args.no_fuse
    channels_last = not args.no_channels_last

    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}  TF32: {tf32}  "
          f"fuse: {fuse}  channels_last (GPU): {channels_last}")
    if "gpu" in devices:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    for size in args.sizes:
        exp_file = args.exp_file_template.format(size=size)
        ckpt_path = args.ckpt_template.format(size=size)
        if not os.path.isfile(exp_file):
            print(f"[WARN] {exp_file} not found - skipping size={size}.")
            continue
        if not os.path.isfile(ckpt_path):
            print(f"[WARN] {ckpt_path} not found - skipping size={size}.")
            continue

        for device in devices:
            precisions = args.precisions if device == "gpu" else ["fp32"]
            warmup = args.warmup if device == "gpu" else args.cpu_warmup
            iters = args.iters if device == "gpu" else args.cpu_iters
            cl = channels_last and device == "gpu"

            for precision in precisions:
                fp16 = (precision == "fp16")
                print(f"\n=== size={size} device={device} precision={precision} "
                      f"(warmup={warmup}, iters={iters}, interleaved) ===")

                models = {variant: build_model(exp_file, ckpt_path, use_cbm, device, fp16, fuse, cl)
                          for use_cbm, variant in VARIANTS}
                x = make_input(device, fp16, args.batch_size, args.test_size, cl)
                stats_by_variant = time_interleaved(models, x, device, warmup, iters, args.batch_size)

                for _, variant in VARIANTS:
                    stats = stats_by_variant[variant]
                    stats.update({
                        'size': size, 'device': device, 'precision': precision,
                        'variant': variant, 'batch_size': args.batch_size,
                        'iters': iters, 'warmup': warmup,
                        'num_threads': num_threads,
                        'tf32': tf32 if device == "gpu" else None,
                        'fused': fuse, 'channels_last': cl,
                    })
                    results.append(stats)
                    print(f"  {variant:>6}: p50={stats['p50_ms']:.3f}ms  mean={stats['mean_ms']:.3f}ms  "
                          f"std={stats['std_ms']:.3f}ms  p95={stats['p95_ms']:.3f}ms  fps(p50)={stats['fps']:.2f}")

                del models, x
                if device == "gpu":
                    torch.cuda.empty_cache()

    if not results:
        print("No results - check --sizes/--exp_file_template/--ckpt_template.")
        return

    # ------------------------------------------------------------------
    # Save raw results.
    # ------------------------------------------------------------------
    fields = ["size", "device", "precision", "variant", "batch_size", "iters", "warmup",
              "num_threads", "tf32", "fused", "channels_last",
              "mean_ms", "std_ms", "p50_ms", "p95_ms", "fps"]
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    with open(csv_path, 'w') as f:
        f.write(",".join(fields) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in fields) + "\n")
    print(f"\nSaved raw results to {csv_path}")

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw results to {json_path}")

    # ------------------------------------------------------------------
    # Overhead summary (p50-based): cbm vs normal per (size, device, precision).
    # ------------------------------------------------------------------
    by_key = {}
    for r in results:
        key = (r['size'], r['device'], r['precision'])
        by_key.setdefault(key, {})[r['variant']] = r

    size_order = {s: i for i, s in enumerate(DEFAULT_SIZES)}
    overhead_rows = []
    for (size, device, precision), variants in sorted(
            by_key.items(), key=lambda kv: (size_order[kv[0][0]], kv[0][1], kv[0][2])):
        if 'normal' in variants and 'cbm' in variants:
            n, c = variants['normal'], variants['cbm']
            overhead_rows.append({
                'size': size, 'device': device, 'precision': precision,
                'normal_p50_ms': n['p50_ms'], 'cbm_p50_ms': c['p50_ms'],
                'overhead_p50_ms': c['p50_ms'] - n['p50_ms'],
                'overhead_p50_pct': 100.0 * (c['p50_ms'] - n['p50_ms']) / n['p50_ms'],
                'normal_mean_ms': n['mean_ms'], 'cbm_mean_ms': c['mean_ms'],
                'overhead_mean_ms': c['mean_ms'] - n['mean_ms'],
                'normal_fps': n['fps'], 'cbm_fps': c['fps'],
            })

    ofields = ["size", "device", "precision", "normal_p50_ms", "cbm_p50_ms", "overhead_p50_ms",
               "overhead_p50_pct", "normal_mean_ms", "cbm_mean_ms", "overhead_mean_ms",
               "normal_fps", "cbm_fps"]
    overhead_path = os.path.join(args.output_dir, "cbm_overhead_summary.csv")
    with open(overhead_path, 'w') as f:
        f.write(",".join(ofields) + "\n")
        for r in overhead_rows:
            f.write(",".join(str(r[k]) for k in ofields) + "\n")
    print(f"Saved CBM overhead summary to {overhead_path}")

    # ------------------------------------------------------------------
    # Plot: grouped bars (p50 latency), one figure per device.
    # ------------------------------------------------------------------
    for device in devices:
        device_results = [r for r in results if r['device'] == device]
        if not device_results:
            continue
        precisions = sorted(set(r['precision'] for r in device_results), reverse=True)  # fp32, fp16
        sizes = [s for s in args.sizes if any(r['size'] == s for r in device_results)]

        fig, ax = plt.subplots(figsize=(2.2 * len(sizes) + 2, 5))
        n_groups = len(precisions) * 2
        width = 0.8 / max(n_groups, 1)
        x = np.arange(len(sizes))

        group_i = 0
        for precision in precisions:
            for variant, color in [('normal', 'tab:blue'), ('cbm', 'tab:orange')]:
                ys = []
                for size in sizes:
                    match = [r for r in device_results
                             if r['size'] == size and r['precision'] == precision and r['variant'] == variant]
                    ys.append(match[0]['p50_ms'] if match else np.nan)
                offset = (group_i - (n_groups - 1) / 2) * width
                ax.bar(x + offset, ys, width, label=f"{variant} ({precision})",
                       color=color, alpha=(0.55 if precision == 'fp32' else 1.0))
                group_i += 1

        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in sizes])
        ax.set_xlabel("Model size")
        ax.set_ylabel(f"Median latency (ms/image, batch={args.batch_size})")
        ax.set_title(f"Inference latency - {device.upper()}")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"benchmark_latency_{device}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot to {plot_path}")

    # ------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY - Inference speed (normal vs. CBM)")
    print("=" * 60)
    t = PrettyTable(["Size", "Device", "Precision", "Variant", "p50 (ms)", "Mean (ms)", "Std (ms)", "FPS (p50)"])
    for r in results:
        t.add_row([r['size'].upper(), r['device'], r['precision'], r['variant'],
                   f"{r['p50_ms']:.3f}", f"{r['mean_ms']:.3f}", f"{r['std_ms']:.3f}", f"{r['fps']:.2f}"])
    print(t)

    print("\nCBM overhead (cbm - normal, p50, same size/device/precision):")
    t2 = PrettyTable(["Size", "Device", "Precision", "Normal p50 (ms)", "CBM p50 (ms)", "Overhead", "Overhead %"])
    for r in overhead_rows:
        t2.add_row([r['size'].upper(), r['device'], r['precision'],
                    f"{r['normal_p50_ms']:.3f}", f"{r['cbm_p50_ms']:.3f}",
                    f"{r['overhead_p50_ms']:+.3f} ms", f"{r['overhead_p50_pct']:+.2f}%"])
    print(t2)


if __name__ == "__main__":
    main(make_parser().parse_args())

