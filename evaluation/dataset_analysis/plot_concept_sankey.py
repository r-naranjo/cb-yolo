#!/usr/bin/env python3
"""Sankey diagrams of concept -> category flows in an annotated dataset.

Reads a COCO-style annotation JSON with a binary `concept_vector` per
annotation and writes interactive Plotly HTML files (no model involved):

  <output>_all.html              top --top_concepts concepts -> all classes
  <output>_supercategories.html  all concepts -> supercategories
  <output>_<supercategory>.html  all concepts -> classes of that supercategory
                                 (every supercategory, or --supercategories)

Flow weight = number of annotations of the category in which the concept is
active. An edge is kept only if
    activations(concept, category) / annotations(category) * 100 >= --min_flow.
In the supercategory diagrams all concepts are listed; concepts without any
surviving edge are drawn in grey.

Requires plotly (pip install plotly).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # evaluation/ (cbm_eval)
from cbm_eval.coco import COCO_CATEGORIES  # noqa: E402
from cbm_eval.concepts import read_concept_names  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sankey diagrams: concept -> COCO category (HTML)")
    p.add_argument("--annotations", required=True,
                   help="COCO-style JSON with a binary concept_vector per annotation")
    p.add_argument("--concepts", required=True, help="Concept-name file (e.g. shared_143.json)")
    p.add_argument("--output", default="outputs/sankey/concept_sankey",
                   help="Output path prefix, no extension")
    p.add_argument("--top_concepts", type=int, default=143,
                   help="All-classes diagram: keep the N concepts with the largest total flow")
    p.add_argument("--min_flow", type=float, default=10.0,
                   help="Minimum activation rate (%%) of a concept within a category for the edge to be kept")
    p.add_argument("--supercategories", nargs="*", default=None,
                   help="Only draw per-supercategory diagrams for these (default: all)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_concepts(path: str):
    concepts = read_concept_names(path)
    print(f"[INFO] {len(concepts)} concepts loaded")
    return concepts


def load_annotations(ann_path: str):
    """
    Returns:
        annotations  list of annotation dicts
        cat_id_to_name   dict  cat_id → name
        cat_id_to_super  dict  cat_id → supercategory
    """
    print(f"[INFO] Loading annotations from {ann_path} ...")
    with open(ann_path) as f:
        data = json.load(f)

    annotations = data["annotations"]
    print(f"[INFO] {len(annotations)} annotations found")

    # Start from the built-in map, then override with whatever is in the file
    cat_id_to_name  = {cid: info[0] for cid, info in COCO_CATEGORIES.items()}
    cat_id_to_super = {cid: info[1] for cid, info in COCO_CATEGORIES.items()}

    if "categories" in data:
        for c in data["categories"]:
            cat_id_to_name[c["id"]] = c["name"]
            if "supercategory" in c:
                cat_id_to_super[c["id"]] = c["supercategory"]

    return annotations, cat_id_to_name, cat_id_to_super


# ─────────────────────────────────────────────────────────────────────────────
# Flow computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_flows(annotations, concepts, cat_id_to_name, cat_whitelist=None):
    """
    Returns:
        flows        dict[(concept_idx, cat_name)] → count
        cat_names    set of category names present in result
        cat_counts   dict[cat_name] → total annotation count (denominator for %)
    """
    flows = defaultdict(int)
    cat_counts = defaultdict(int)
    cat_names_seen = set()
    skipped = 0

    for ann in annotations:
        cat_id = ann.get("category_id")
        cv     = ann.get("concept_vector")

        if cat_id is None or cv is None:
            skipped += 1
            continue

        cat_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")

        if cat_whitelist and cat_name not in cat_whitelist:
            continue

        cat_names_seen.add(cat_name)
        cat_counts[cat_name] += 1

        for ci, val in enumerate(cv):
            if ci >= len(concepts):
                break
            if val:
                flows[(ci, cat_name)] += 1

    if skipped:
        print(f"[WARN] {skipped} annotations skipped (missing category_id or concept_vector)")
    print(f"[INFO] {len(cat_names_seen)} categories · {len(flows)} raw edges computed")
    return flows, cat_names_seen, cat_counts


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_flows(flows, cat_counts, concepts, top_concepts, min_flow_pct):
    """
    Keep only edges where:
        activations(concept, category) / total_instances(category) * 100 >= min_flow_pct

    Then retain only the top-N concepts by total (percentage-weighted) flow.
    """
    flows = {
        (ci, cat): cnt
        for (ci, cat), cnt in flows.items()
        if cat_counts.get(cat, 0) > 0
        and (cnt / cat_counts[cat] * 100) >= min_flow_pct
    }

    concept_totals = defaultdict(int)
    for (ci, _), count in flows.items():
        concept_totals[ci] += count

    top_ci = set(
        sorted(concept_totals, key=concept_totals.get, reverse=True)[:top_concepts]
    )
    flows = {(ci, cat): cnt for (ci, cat), cnt in flows.items() if ci in top_ci}

    print(f"[INFO] After filtering: {len(top_ci)} concepts · {len(flows)} edges")
    return flows, top_ci


# ─────────────────────────────────────────────────────────────────────────────
# Color helpers
# ─────────────────────────────────────────────────────────────────────────────

def concept_color(name: str) -> str:
    n = name.lower()
    if any(n.startswith(p) for p in (
        "has_legs","has_arm","has_paw","has_hoo","has_claw","has_wing",
        "has_fin","has_tail","has_long","has_trunk","has_eyes","has_ears",
        "has_snout","has_beak","has_horns","has_whis","has_mane","has_furry",
    )):
        return "rgba(83, 74, 183, 0.7)"    # purple  – animal anatomy

    if any(n.startswith(p) for p in (
        "has_wheel","has_handle","has_steer","has_wind","has_prop","has_jet",
        "has_licen","has_head","has_tail_l","has_exhaust","has_pedal",
        "has_flat","has_seatb","has_armr","has_grab","has_lid","has_blade",
        "has_brist","has_digit","has_phys","has_knob","has_spout","has_rim",
        "has_zipper","has_should","has_hinge","has_legs_f","has_keyb",
        "has_antenna","has_bell",
    )):
        return "rgba(15, 110, 86, 0.7)"    # teal    – object parts

    if n.startswith("texture_"):
        return "rgba(153, 60, 29, 0.7)"    # coral   – textures

    if any(n.startswith(p) for p in ("shape_","edges_","is_thin","is_thick","is_taper")):
        return "rgba(24, 95, 165, 0.7)"    # blue    – shape / geometry

    if any(n.startswith(p) for p in ("pattern_","color_")):
        return "rgba(153, 53, 86, 0.7)"    # pink    – pattern / color

    if any(n.startswith(p) for p in (
        "is_","has_text","has_sym","has_logo","has_red","has_circ",
        "has_conc","has_speck","has_vein",
    )):
        return "rgba(133, 79, 11, 0.7)"    # amber   – state / context

    if any(n.startswith(p) for p in (
        "appears_","orientation_","shadow","light_",
    )):
        return "rgba(59, 109, 17, 0.7)"    # green   – scene context

    return "rgba(95, 94, 90, 0.7)"         # gray    – other


_CAT_PALETTE = [
    "#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0",
    "#00BCD4","#FF5722","#607D8B","#795548","#F44336",
    "#3F51B5","#009688","#FFC107","#8BC34A","#673AB7",
]


# ─────────────────────────────────────────────────────────────────────────────
# Build Plotly Sankey
# ─────────────────────────────────────────────────────────────────────────────

def build_sankey(flows, concepts, top_ci, cat_names_seen, top_concepts,
                 title_suffix="All classes", show_all_concepts=False):
    """
    Build a Plotly Sankey figure.

    When show_all_concepts=True (used for supercategory diagrams):
      - ALL concepts appear on the left in JSON order, regardless of whether
        they have any flow to the filtered categories.
      - Concepts with no connections are shown in a dimmed grey so the viewer
        can immediately spot which concepts are irrelevant for that supercategory.
      - top_concepts / top_ci are ignored for node construction (but flows are
        still pre-filtered by min_flow before this function is called).
    When show_all_concepts=False (used for the all-classes diagram):
      - Only the top-N concepts with actual flow are shown (original behaviour).
    """
    import plotly.graph_objects as go

    cat_list = sorted(cat_names_seen)

    if show_all_concepts:
        # All concepts in JSON order (index 0 … N-1)
        concept_indices = list(range(len(concepts)))
    else:
        concept_indices = sorted(top_ci)

    concept_node_idx = {ci: i for i, ci in enumerate(concept_indices)}
    cat_node_idx     = {cat: len(concept_indices) + i for i, cat in enumerate(cat_list)}

    # Determine which concepts actually have at least one surviving edge
    connected_ci = {ci for (ci, _) in flows}

    # Node labels & colors
    node_labels = []
    node_colors = []
    for ci in concept_indices:
        node_labels.append(concepts[ci])
        if show_all_concepts and ci not in connected_ci:
            # Dimmed grey for concepts with no connection to this supercategory
            node_colors.append("rgba(180, 180, 180, 0.35)")
        else:
            node_colors.append(concept_color(concepts[ci]))

    node_labels += cat_list
    node_colors += [_CAT_PALETTE[i % len(_CAT_PALETTE)] for i in range(len(cat_list))]

    sources, targets, values, link_colors, link_labels = [], [], [], [], []
    for (ci, cat), count in sorted(flows.items(), key=lambda x: -x[1]):
        if ci not in concept_node_idx or cat not in cat_node_idx:
            continue
        sources.append(concept_node_idx[ci])
        targets.append(cat_node_idx[cat])
        values.append(count)
        link_colors.append(concept_color(concepts[ci]).replace("0.7)", "0.25)"))
        link_labels.append(f"{concepts[ci]} → {cat}: {count:,}")

    n_cats     = len(cat_list)
    n_concepts = len(concept_indices)
    n_connected = len(connected_ci)
    height     = max(600, max(n_cats, n_concepts) * 28 + 120)

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad       = 12,
            thickness = 14,
            line      = dict(color="rgba(0,0,0,0.1)", width=0.5),
            label     = node_labels,
            color     = node_colors,
            hovertemplate="%{label}<br>Total flow: %{value:,}<extra></extra>",
        ),
        link=dict(
            source      = sources,
            target      = targets,
            value       = values,
            color       = link_colors,
            label       = link_labels,
            hovertemplate="%{label}<extra></extra>",
        ),
    ))

    if show_all_concepts:
        subtitle = (
            f"{n_concepts} concepts ({n_connected} connected, "
            f"{n_concepts - n_connected} unconnected · shown in grey) · "
            f"{n_cats} categories · {len(sources):,} edges"
        )
    else:
        subtitle = (
            f"Top {top_concepts} concepts by flow weight · "
            f"{n_concepts} concepts · {n_cats} categories · "
            f"{len(sources):,} edges"
        )

    fig.update_layout(
        title=dict(
            text=(
                f"Concept → Category Sankey — {title_suffix}<br>"
                f"<sup>{subtitle}</sup>"
            ),
            font=dict(size=16),
            x=0.5, xanchor="center",
        ),
        font=dict(family="sans-serif", size=11),
        height=height,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
    )

    return fig



# ─────────────────────────────────────────────────────────────────────────────
# Build supercategory-level Sankey  (concept → supercategory)
# ─────────────────────────────────────────────────────────────────────────────

def build_sankey_supercategories(
    flows_all, cat_counts_all, cat_name_to_super,
    concepts, top_concepts, min_flow_pct,
):
    """
    Aggregate category-level flows up to supercategory level, apply the
    percentage-based min_flow filter (denominator = total instances in the
    supercategory), then build a Sankey with ALL concepts on the left in JSON
    order (dimmed grey when unconnected) and supercategories on the right.
    """
    # Total instances per supercategory (sum over its member categories)
    super_counts = defaultdict(int)
    for cat, total in cat_counts_all.items():
        sup = cat_name_to_super.get(cat)
        if sup:
            super_counts[sup] += total

    # Aggregate flows: (concept_idx, supercategory) → count
    super_flows = defaultdict(int)
    for (ci, cat), cnt in flows_all.items():
        sup = cat_name_to_super.get(cat)
        if sup:
            super_flows[(ci, sup)] += cnt

    # Percentage filter: keep edge if activations / supercategory_instances >= min%
    super_flows = {
        (ci, sup): cnt
        for (ci, sup), cnt in super_flows.items()
        if super_counts.get(sup, 0) > 0
        and (cnt / super_counts[sup] * 100) >= min_flow_pct
    }

    supercats_seen = {sup for (_, sup) in super_flows}

    if not super_flows:
        return None

    # Reuse build_sankey: pass supercategory names as "cat_names_seen" and
    # show_all_concepts=True so every concept appears in JSON order.
    return build_sankey(
        super_flows, concepts, set(), supercats_seen,
        top_concepts,
        title_suffix="All supercategories (macro view)",
        show_all_concepts=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Save HTML
# ─────────────────────────────────────────────────────────────────────────────

def save_html(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"[SAVE] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    concepts = load_concepts(args.concepts)
    annotations, cat_id_to_name, cat_id_to_super = load_annotations(args.annotations)

    prefix = Path(args.output)

    # ── 1. Diagram: All concepts → All classes ────────────────────────────────
    print("\n[INFO] === Generating diagram: All concepts → All classes ===")
    flows_all, cats_all, cat_counts_all = compute_flows(annotations, concepts, cat_id_to_name)

    if not flows_all:
        sys.exit("[ERROR] No flows computed — check your JSON paths.")

    flows_all, top_ci_all = filter_flows(
        flows_all, cat_counts_all, concepts, args.top_concepts, args.min_flow
    )
    if flows_all:
        fig = build_sankey(
            flows_all, concepts, top_ci_all, cats_all,
            args.top_concepts, title_suffix="All classes"
        )
        save_html(fig, prefix.with_name(prefix.name + "_all.html"))
    else:
        print("[WARN] All-classes diagram: all edges filtered out — skipping.")

    # ── 1b. Diagram: All concepts → All supercategories (macro) ─────────────────
    print("\n[INFO] === Generating diagram: All concepts → All supercategories ===")
    # cat_name_to_super is needed here; build it before the supercategory loop
    cat_name_to_super = {
        cat_id_to_name[cid]: super_
        for cid, super_ in cat_id_to_super.items()
        if cid in cat_id_to_name
    }
    fig_macro = build_sankey_supercategories(
        flows_all  if flows_all else {},   # may be empty if filtered to nothing
        cat_counts_all,
        cat_name_to_super,
        concepts, args.top_concepts, args.min_flow,
    )
    if fig_macro is not None:
        save_html(fig_macro, prefix.with_name(prefix.name + "_supercategories.html"))
    else:
        print("[WARN] Supercategory macro diagram: no edges survived filtering — skipping.")

    # ── 2. Determine supercategories to plot ──────────────────────────────────
    all_supercats = sorted({
        cat_name_to_super[cat]
        for cat in cats_all
        if cat in cat_name_to_super
    })

    if args.supercategories:
        requested = set(args.supercategories)
        unknown   = requested - set(all_supercats)
        if unknown:
            print(f"[WARN] Unknown supercategories (will be skipped): {sorted(unknown)}")
        supercats_to_plot = [s for s in all_supercats if s in requested]
    else:
        supercats_to_plot = all_supercats

    print(f"\n[INFO] Supercategories to plot: {supercats_to_plot}")

    # ── 3. One diagram per supercategory ──────────────────────────────────────
    for supercat in supercats_to_plot:
        print(f"\n[INFO] === Generating diagram: All concepts → {supercat} ===")

        # Category whitelist for this supercategory
        cat_whitelist = {
            name for name, sup in cat_name_to_super.items() if sup == supercat
        }

        flows_sc, cats_sc, cat_counts_sc = compute_flows(
            annotations, concepts, cat_id_to_name, cat_whitelist=cat_whitelist
        )

        if not flows_sc:
            print(f"[WARN] No flows for supercategory '{supercat}' — skipping.")
            continue

        # Apply only percentage-based min_flow edge pruning; do NOT restrict which
        # concept nodes appear — all concepts must be shown in JSON order.
        flows_sc = {
            (ci, cat): cnt
            for (ci, cat), cnt in flows_sc.items()
            if cat_counts_sc.get(cat, 0) > 0
            and (cnt / cat_counts_sc[cat] * 100) >= args.min_flow
        }

        if not flows_sc:
            print(f"[WARN] All edges filtered for '{supercat}' — skipping.")
            continue

        # top_ci is unused when show_all_concepts=True, pass empty set
        fig = build_sankey(
            flows_sc, concepts, set(), cats_sc,
            args.top_concepts,
            title_suffix=f"Supercategory: {supercat}",
            show_all_concepts=True,
        )
        safe_name = supercat.replace(" ", "_").lower()
        save_html(fig, prefix.with_name(f"{prefix.name}_{safe_name}.html"))

    print("\n[DONE]")


if __name__ == "__main__":
    main()

