#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_grid_search.py
===================
Reads the incremental JSON produced by grid_search_concepts.py and generates
an **interactive 3D HTML page** (Plotly) that shows:

  1. Surface plot   – mAP interpolated over concept_loss_weight × sigma_,
                      so you can visualise the 2-D performance landscape.
  2. Scatter markers – one per explored (clw, sigma_) pair, plotted at their
                       actual mAP height on top of the surface.
  3. A summary table – top-10 configurations by the chosen metric.

Usage:
    python plot_grid_search.py \
        --input   YOLOX_outputs/random_search/results.json \
        --metric  val_mAP50_95 \
        --output  YOLOX_outputs/random_search/plot.html

If --metric is not found in the data, the script falls back to train_total_loss
(lower-is-better, indicated in the plot title).
"""

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-D interactive surface plot of YOLOX random-search results"
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the JSON results file produced by grid_search_concepts.py",
    )
    p.add_argument(
        "-m", "--metric",
        default="val_mAP50_95",
        help="Metric to colour / optimise (default: val_mAP50_95)",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output HTML file path (default: <input stem>_plot.html)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str):
    with open(path) as f:
        records = json.load(f)

    ok = [r for r in records if r.get("status") == "OK"]
    if not ok:
        raise ValueError(f"No successful trials found in {path}")
    return ok


def pick_metric(records: list, preferred: str) -> str:
    """Return `preferred` if at least one record has it, else fall back."""
    if any(preferred in r for r in records):
        return preferred
    fallbacks = ["train_total_loss", "train_iou_loss", "train_cls_loss"]
    for fb in fallbacks:
        if any(fb in r for r in records):
            print(f"[WARNING] Metric '{preferred}' not found; falling back to '{fb}'")
            return fb
    hparams = {
        "trial_idx", "concept_loss_weight", "sigma_",
        "status", "elapsed_s", "error",
    }
    for key in records[0]:
        if key not in hparams and isinstance(records[0][key], (int, float)):
            print(f"[WARNING] Falling back to first available metric: '{key}'")
            return key
    raise ValueError("No usable metric found in results JSON.")


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def build_surface(xs, ys, zvals, n_grid=60):
    """
    Given scattered (x, y, z) triplets, return a dense grid (Xi, Yi, Zi)
    suitable for go.Surface.  Uses scipy.interpolate.griddata when available,
    otherwise falls back to nearest-neighbour.

    xs  = concept_loss_weight values
    ys  = sigma_ values
    zvals = metric values
    """
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    try:
        from scipy.interpolate import griddata
        points = np.column_stack([xs, ys])
        n_pts = len(xs)

        # Cubic needs Delaunay triangulation: requires ≥4 non-collinear points.
        # Linear needs a valid triangulation too: requires ≥3 non-collinear points.
        # Nearest always works.  Degrade gracefully.
        def _is_collinear(pts):
            """Return True if all points lie on a single line."""
            if len(pts) < 3:
                return True
            v = pts[1:] - pts[0]
            cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
            return np.allclose(cross, 0, atol=1e-12)

        collinear = _is_collinear(points)

        if n_pts >= 4 and not collinear:
            method = "cubic"
        elif n_pts >= 3 and not collinear:
            method = "linear"
        else:
            method = "nearest"

        print(f"[INFO] Surface interpolation method: {method} ({n_pts} points)")
        Zi = griddata(points, zvals, (Xi, Yi), method=method)

        # Fill remaining NaNs (extrapolation holes) stepping down through methods
        nan_mask = np.isnan(Zi)
        if nan_mask.any() and method == "cubic":
            Zi_lin = griddata(points, zvals, (Xi, Yi), method="linear")
            Zi[nan_mask] = Zi_lin[nan_mask]
            nan_mask = np.isnan(Zi)
        if nan_mask.any():
            Zi_nn = griddata(points, zvals, (Xi, Yi), method="nearest")
            Zi[nan_mask] = Zi_nn[nan_mask]

    except ImportError:
        print(
            "[WARNING] scipy not available; using nearest-neighbour interpolation "
            "for surface"
        )
        Zi = np.full_like(Xi, float("nan"), dtype=float)
        for i, yv in enumerate(yi):
            for j, xv in enumerate(xi):
                dists = [
                    (abs(a - xv) + abs(b - yv), z)
                    for a, b, z in zip(xs, ys, zvals)
                ]
                dists.sort()
                Zi[i, j] = dists[0][1]

    return Xi, Yi, Zi


# ---------------------------------------------------------------------------
# Plotly HTML generation
# ---------------------------------------------------------------------------

def _make_surface_figure(
    records: list,
    clw: np.ndarray,
    sigma: np.ndarray,
    metric_vals: np.ndarray,
    metric: str,
    go,
):
    """
    Build a single Plotly Figure with an interpolated surface + scatter overlay
    for the given metric values.  Reused for both the detection mAP figure and
    the concept mAP figure.
    """
    higher_is_better = "mAP" in metric or "ap" in metric.lower() or "auc" in metric.lower()
    valid   = ~np.isnan(metric_vals)
    clw_v   = clw[valid]
    sigma_v = sigma[valid]
    mval_v  = metric_vals[valid]

    best_idx_local = int(
        np.nanargmax(metric_vals) if higher_is_better else np.nanargmin(metric_vals)
    )

    colorscale = "RdYlGn" if higher_is_better else "RdYlGn_r"

    Xi, Yi, Zi = build_surface(clw_v, sigma_v, mval_v, n_grid=60)

    surface = go.Surface(
        x=Xi, y=Yi, z=Zi,
        colorscale=colorscale,
        opacity=0.75,
        showscale=True,
        colorbar=dict(title=dict(text=metric, side="right"), thickness=18, len=0.7),
        name="Interpolated surface",
        hovertemplate=(
            "concept_loss_weight=%{x:.4f}<br>"
            "sigma_=%{y:.4f}<br>"
            f"{metric}=%{{z:.4f}}<extra></extra>"
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
    )

    hover_text = [
        (
            f"concept_loss_weight={r['concept_loss_weight']}<br>"
            f"sigma_={r['sigma_']}<br>"
            f"{metric}={r.get(metric, 'N/A'):.4f}<br>"
            f"elapsed={r.get('elapsed_s', 'N/A')}s"
        )
        for r, v in zip(records, valid)
        if v
    ]
    scatter = go.Scatter3d(
        x=clw_v, y=sigma_v, z=mval_v,
        mode="markers",
        marker=dict(
            size=6,
            color=mval_v,
            colorscale=colorscale,
            cmin=float(np.nanmin(metric_vals)),
            cmax=float(np.nanmax(metric_vals)),
            opacity=1.0,
            line=dict(width=1, color="white"),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="Observed trials",
    )

    best_marker = go.Scatter3d(
        x=[clw[best_idx_local]],
        y=[sigma[best_idx_local]],
        z=[metric_vals[best_idx_local]],
        mode="markers+text",
        marker=dict(size=12, color="red", symbol="diamond", opacity=1.0),
        text=["Best"],
        textposition="top center",
        textfont=dict(size=13, color="red"),
        name=f"Best ({metric}={metric_vals[best_idx_local]:.4f})",
        hovertemplate=(
            f"concept_loss_weight={clw[best_idx_local]}<br>"
            f"sigma_={sigma[best_idx_local]}<br>"
            f"{metric}={metric_vals[best_idx_local]:.4f}<extra></extra>"
        ),
    )

    fig = go.Figure(data=[surface, scatter, best_marker])
    fig.update_layout(
        title=dict(
            text=(
                f"YOLOX-CBM Random Search – {metric} Surface<br>"
                f"<sup>Axes: concept_loss_weight × sigma_  |  "
                f"Colour = {metric}  |  "
                f"{'Higher' if higher_is_better else 'Lower'} is better</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="concept_loss_weight",
            yaxis_title="sigma_",
            zaxis_title=metric,
            xaxis=dict(backgroundcolor="rgb(230,230,250)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230,250,230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(250,230,230)", gridcolor="white"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=100, b=0),
        height=720,
    )
    return fig, best_idx_local


def build_html(records: list, metric: str, output_path: str):
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for plot generation.  "
            "Install it with:  pip install plotly"
        )

    # ------------------------------------------------------------------
    # Extract common arrays
    # ------------------------------------------------------------------
    clw   = np.array([r["concept_loss_weight"] for r in records], dtype=float)
    sigma = np.array([r["sigma_"]              for r in records], dtype=float)

    # ------------------------------------------------------------------
    # Detection mAP figure
    # ------------------------------------------------------------------
    det_vals = np.array(
        [r.get(metric, float("nan")) for r in records], dtype=float
    )
    if not (~np.isnan(det_vals)).any():
        raise ValueError(f"All metric values for '{metric}' are NaN.")

    higher_is_better_det = "mAP" in metric or "ap" in metric.lower()
    best_det_idx = int(
        np.nanargmax(det_vals) if higher_is_better_det else np.nanargmin(det_vals)
    )
    best_det_record = records[best_det_idx]

    print(
        f"\n[BEST TRIAL– {metric}]  "
        f"concept_loss_weight={best_det_record['concept_loss_weight']}  "
        f"sigma_={best_det_record['sigma_']}  "
        f"{metric}={det_vals[best_det_idx]:.4f}"
    )

    fig_det, _ = _make_surface_figure(records, clw, sigma, det_vals, metric, go)

    # ------------------------------------------------------------------
    # Concept mAP figure  (optional – only if data exists)
    # ------------------------------------------------------------------
    concept_metric = "val_concept_mAP"
    concept_vals = np.array(
        [r.get(concept_metric, float("nan")) for r in records], dtype=float
    )
    has_concept_data = (~np.isnan(concept_vals)).any()

    html_concept_section = ""
    best_concept_record  = None
    best_concept_val     = float("nan")

    if has_concept_data:
        best_concept_idx    = int(np.nanargmax(concept_vals))
        best_concept_record = records[best_concept_idx]
        best_concept_val    = concept_vals[best_concept_idx]

        print(
            f"[BEST TRIAL – {concept_metric}]  "
            f"concept_loss_weight={best_concept_record['concept_loss_weight']}  "
            f"sigma_={best_concept_record['sigma_']}  "
            f"{concept_metric}={best_concept_val:.4f}"
        )

        fig_concept, _ = _make_surface_figure(
            records, clw, sigma, concept_vals, concept_metric, go
        )
        html_concept_fig = fig_concept.to_html(full_html=False, include_plotlyjs=False)

        best_concept_clw_str = f"{best_concept_record['concept_loss_weight']:.4f}"
        best_concept_sig_str = f"{best_concept_record['sigma_']:.4f}"
        best_concept_val_str = f"{best_concept_val:.4f}"

        html_concept_section = f"""
  <div class="stats-row" style="margin-top:0;">
    <div class="stat-box best-box">
      <div class="val">{best_concept_val_str}</div>
      <div class="lbl">Best {concept_metric}</div>
    </div>
    <div class="stat-box best-box">
      <div class="val">{best_concept_clw_str}</div>
      <div class="lbl">Best concept_loss_weight</div>
    </div>
    <div class="stat-box best-box">
      <div class="val">{best_concept_sig_str}</div>
      <div class="lbl">Best sigma_</div>
    </div>
  </div>

  <div class="card">
    <h2>Concept mAP Surface – concept_loss_weight × sigma_</h2>
    <p style="color:#555; font-size:0.9em;">
      Mean Average Precision averaged across all measurable concepts
      (those with ≥ 10 positive instances in the matched set).
      Computed by <code>compute_concept_maps_search.py</code> on the last
      checkpoint of each trial.
      Rotate, zoom, and hover for details.
    </p>
    {html_concept_fig}
  </div>
"""
    else:
        html_concept_section = """
  <div class="card" style="border-left: 4px solid #f0ad4e;">
    <h2>Concept mAP Surface – not yet available</h2>
    <p style="color:#888; font-size:0.9em;">
      Run <code>compute_concept_maps_search.py</code> to add concept metrics
      to the results JSON, then regenerate this plot.
    </p>
  </div>
"""

    # ------------------------------------------------------------------
    # Summary table (top-10 by detection mAP, also shows concept mAP)
    # ------------------------------------------------------------------
    det_valid      = ~np.isnan(det_vals)
    sorted_idxs    = np.argsort(det_vals[det_valid])
    if higher_is_better_det:
        sorted_idxs = sorted_idxs[::-1]
    top_k          = min(10, len(sorted_idxs))
    valid_idxs     = np.where(det_valid)[0]
    top_records    = [records[valid_idxs[i]] for i in sorted_idxs[:top_k]]

    # Build table columns conditionally
    col_headers = [
        "<b>Rank</b>",
        "<b>concept_loss_weight</b>",
        "<b>sigma_</b>",
        f"<b>{metric}</b>",
    ]
    col_values = [
        list(range(1, top_k + 1)),
        [f"{r['concept_loss_weight']:.6f}" for r in top_records],
        [f"{r['sigma_']:.6f}"              for r in top_records],
        [f"{r.get(metric, 'N/A'):.4f}"     for r in top_records],
    ]
    if has_concept_data:
        col_headers += ["<b>val_concept_mAP</b>", "<b>val_concept_mAUC</b>"]
        col_values  += [
            [
                f"{r['val_concept_mAP']:.4f}"
                if r.get("val_concept_mAP") is not None else "–"
                for r in top_records
            ],
            [
                f"{r['val_concept_mAUC']:.4f}"
                if r.get("val_concept_mAUC") is not None else "–"
                for r in top_records
            ],
        ]
    col_headers.append("<b>elapsed (s)</b>")
    col_values.append([r.get("elapsed_s", "N/A") for r in top_records])

    n_cols = len(col_headers)
    table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=col_headers,
                    fill_color="#4C72B0",
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=col_values,
                    fill_color=[
                        [
                            "#ffd700" if i == 0 else ("#e8e8e8" if i % 2 == 0 else "white")
                            for i in range(top_k)
                        ]
                    ] * n_cols,
                    font=dict(size=11),
                    align="center",
                ),
            )
        ]
    )
    table.update_layout(
        title=dict(text=f"Top-{top_k} Trials by {metric}", x=0.5),
        margin=dict(l=0, r=0, t=60, b=0),
        height=40 * (top_k + 2),
    )

    # ------------------------------------------------------------------
    # Combine into a single HTML file
    # ------------------------------------------------------------------
    html_det_surface = fig_det.to_html(full_html=False, include_plotlyjs="cdn")
    html_table       = table.to_html(full_html=False, include_plotlyjs=False)

    n_trials = len(records)
    n_ok     = sum(1 for r in records if r.get("status") == "OK")
    n_failed = n_trials - n_ok

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>YOLOX-CBM Random Search Results</title>
  <style>
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f5f7fa;
      color: #222;
      margin: 0;
      padding: 0 20px 40px;
    }}
    h1 {{
      text-align: center;
      color: #2c3e50;
      padding-top: 30px;
    }}
    .subtitle {{
      text-align: center;
      color: #555;
      font-size: 0.95em;
      margin-bottom: 30px;
    }}
    .card {{
      background: white;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.10);
      padding: 20px;
      margin-bottom: 30px;
    }}
    .stats-row {{
      display: flex;
      gap: 20px;
      justify-content: center;
      flex-wrap: wrap;
      margin-bottom: 30px;
    }}
    .stat-box {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      padding: 16px 28px;
      text-align: center;
      min-width: 160px;
    }}
    .stat-box .val {{
      font-size: 2em;
      font-weight: bold;
      color: #4C72B0;
    }}
    .stat-box .lbl {{
      font-size: 0.82em;
      color: #888;
      margin-top: 4px;
    }}
    .best-box .val {{ color: #27ae60; }}
    h2 {{ color: #2c3e50; border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; }}
    .section-label {{
      text-align: center;
      font-size: 0.8em;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #4C72B0;
      margin: 30px 0 8px;
    }}
  </style>
</head>
<body>
  <h1>YOLOX-CBM Hyper-parameter Random Search</h1>
  <p class="subtitle">
    Parameters tuned: <code>concept_loss_weight</code> ×
    <code>sigma_</code>
    &nbsp;|&nbsp; Detection metric: <strong>{metric}</strong>
  </p>

  <div class="section-label">Detection performance</div>
  <div class="stats-row">
    <div class="stat-box">
      <div class="val">{n_trials}</div>
      <div class="lbl">Total trials</div>
    </div>
    <div class="stat-box">
      <div class="val">{n_ok}</div>
      <div class="lbl">Successful</div>
    </div>
    <div class="stat-box">
      <div class="val">{n_failed}</div>
      <div class="lbl">Failed</div>
    </div>
    <div class="stat-box best-box">
      <div class="val">{det_vals[best_det_idx]:.4f}</div>
      <div class="lbl">Best {metric}</div>
    </div>
    <div class="stat-box best-box">
      <div class="val">{best_det_record['concept_loss_weight']:.4f}</div>
      <div class="lbl">Best concept_loss_weight</div>
    </div>
    <div class="stat-box best-box">
      <div class="val">{best_det_record['sigma_']:.4f}</div>
      <div class="lbl">Best sigma_</div>
    </div>
  </div>

  <div class="card">
    <h2>Detection mAP Surface – concept_loss_weight × sigma_</h2>
    <p style="color:#555; font-size:0.9em;">
      The surface is interpolated (cubic spline) from the random-search observations.
      Coloured dots show the actual evaluated points.
      The red diamond marks the best configuration.
      Rotate, zoom, and hover for details.
    </p>
    {html_det_surface}
  </div>

  <div class="section-label">Concept performance</div>
  {html_concept_section}

  <div class="card">
    <h2>Top-{top_k} Configurations (ranked by {metric})</h2>
    {html_table}
  </div>

  <p style="text-align:center; color:#aaa; font-size:0.8em; margin-top:20px;">
    Generated by plot_grid_search.py &nbsp;|&nbsp; YOLOX-CBM project
  </p>
</body>
</html>
"""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"\n[INFO] Interactive HTML plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print(f"[INFO] Loading results from: {args.input}")
    records = load_data(args.input)
    print(f"[INFO] Loaded {len(records)} successful trial(s)")

    metric = pick_metric(records, args.metric)
    output = args.output or os.path.splitext(args.input)[0] + "_plot.html"
    build_html(records, metric, output)


if __name__ == "__main__":
    main()
