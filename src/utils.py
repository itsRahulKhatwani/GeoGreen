"""
GeoGreen Revolution — Utilities Module
========================================
Visualization, map generation, and report saving functions.

UPGRADED: Added ML land-cover map, NDWI map, comparison map,
fused classification map, and enhanced summary report with
model metadata and academic documentation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config import (
    RULEBASED_LAND_CLASS_LABELS, RULEBASED_LAND_CLASS_COLORS,
    ML_LAND_CLASS_LABELS, ML_LAND_CLASS_COLORS,
    OUTPUT_DIR, OUTPUT_DPI, FIGURE_SIZE,
    LAND_CLASS_LABELS, LAND_CLASS_COLORS,
    MODEL_INFO, REGION_NAME,
)


def ensure_output_dir():
    """Create the output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


# ──────────────────────────────────────────────────────────────
# NDVI MAP
# ──────────────────────────────────────────────────────────────
def save_ndvi_map(ndvi, output_path=None):
    """
    Generate and save a color-coded NDVI map.
    Red = low NDVI (barren), Green = high NDVI (vegetated).
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ndvi_map.png")

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="NDVI Value")
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("NDVI Map — Vegetation Health Index", fontsize=16, fontweight="bold")
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    for val, label in [(0.0, "Water"), (0.12, "Built-up"), (0.25, "Sparse"), (0.45, "Dense")]:
        cbar.ax.axhline(y=val, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 NDVI map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# NDWI MAP (NEW)
# ──────────────────────────────────────────────────────────────
def save_ndwi_map(ndwi, output_path=None):
    """
    Generate and save a color-coded NDWI (water index) map.
    Blue = high NDWI (water), Brown = low NDWI (dry land).
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ndwi_map.png")

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)

    im = ax.imshow(ndwi, cmap="RdYlBu", vmin=-0.5, vmax=0.5)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="NDWI Value")
    cbar.ax.tick_params(labelsize=10)

    # Mark water threshold
    cbar.ax.axhline(y=0.3, color="red", linewidth=1.5, linestyle="--")

    ax.set_title("NDWI Map — Water Detection Index", fontsize=16, fontweight="bold")
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 NDWI map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# RULE-BASED CLASSIFICATION MAP
# ──────────────────────────────────────────────────────────────
def save_classification_map(classification, output_path=None):
    """
    Generate and save a RULE-BASED classification map (NDVI thresholds).
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "classification_map_rulebased.png")

    colors = [RULEBASED_LAND_CLASS_COLORS[i] for i in sorted(RULEBASED_LAND_CLASS_COLORS.keys())]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    im = ax.imshow(classification, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")

    patches = [
        mpatches.Patch(color=RULEBASED_LAND_CLASS_COLORS[k],
                       label=RULEBASED_LAND_CLASS_LABELS[k])
        for k in sorted(RULEBASED_LAND_CLASS_LABELS.keys())
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.9,
              title="Land Classes (Rule-Based)", title_fontsize=11)

    ax.set_title("Land Classification — Rule-Based (NDVI Thresholds)",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Rule-based classification map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# ML LAND-COVER MAP (NEW)
# ──────────────────────────────────────────────────────────────
def save_ml_landcover_map(ml_classification, output_path=None):
    """
    Generate and save the ML-based (ESA WorldCover) land-cover map.
    This is the PRIMARY classification output of the upgraded system.
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "ml_landcover_map.png")

    # Build custom colormap for 8 classes (0–7)
    colors = [ML_LAND_CLASS_COLORS[i] for i in range(8)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    im = ax.imshow(ml_classification, cmap=cmap, vmin=-0.5, vmax=7.5,
                   interpolation="nearest")

    # Create legend (skip class 0 if no pixels)
    patches = []
    for k in sorted(ML_LAND_CLASS_LABELS.keys()):
        if k == 0 and np.count_nonzero(ml_classification == 0) == 0:
            continue
        patches.append(
            mpatches.Patch(color=ML_LAND_CLASS_COLORS[k],
                           label=ML_LAND_CLASS_LABELS[k])
        )
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9,
              title="Land Cover (ML — ESA WorldCover)", title_fontsize=10)

    ax.set_title(
        "AI Land-Cover Classification — ESA WorldCover (Pretrained DL Model)",
        fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    # Add model info as text
    info_text = (
        f"Model: {MODEL_INFO['name']}\n"
        f"Algorithm: {MODEL_INFO['algorithm']}\n"
        f"Accuracy: {MODEL_INFO['overall_accuracy']}"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 ML land-cover map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# FUSED CLASSIFICATION MAP (NEW)
# ──────────────────────────────────────────────────────────────
def save_fused_map(fused_classification, output_path=None):
    """
    Generate and save the FUSED land-cover map
    (ML + NDVI/NDWI refinement).
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "fused_landcover_map.png")

    colors = [ML_LAND_CLASS_COLORS[i] for i in range(8)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    im = ax.imshow(fused_classification, cmap=cmap, vmin=-0.5, vmax=7.5,
                   interpolation="nearest")

    patches = []
    for k in sorted(ML_LAND_CLASS_LABELS.keys()):
        if k == 0 and np.count_nonzero(fused_classification == 0) == 0:
            continue
        patches.append(
            mpatches.Patch(color=ML_LAND_CLASS_COLORS[k],
                           label=ML_LAND_CLASS_LABELS[k])
        )
    ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.9,
              title="Fused Land Cover (ML + Indices)", title_fontsize=10)

    ax.set_title(
        "Fused Land-Cover Map — ML Classification + Spectral Index Refinement",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Fused land-cover map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# RECOMMENDATION MAP
# ──────────────────────────────────────────────────────────────
def save_recommendation_map(classification, zone_df, grid_size=50, output_path=None):
    """
    Generate a recommendation overlay map showing intervention zones.
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "recommendation_map.png")

    h, w = classification.shape
    priority_map = np.full((h, w), fill_value=0, dtype=np.int8)

    priority_values = {"None": 0, "Low": 1, "Medium": 2, "High": 3}

    for _, row in zone_df.iterrows():
        r = int(row["row_start"])
        c = int(row["col_start"])
        r_end = min(r + grid_size, h)
        c_end = min(c + grid_size, w)
        pri = priority_values.get(row.get("priority", "None"), 0)
        priority_map[r:r_end, c:c_end] = pri

    colors_pri = [
        (0.9, 0.9, 0.9, 0.3),    # None — very light grey
        (0.56, 0.78, 0.34, 0.7),  # Low — green
        (1.0, 0.85, 0.0, 0.8),    # Medium — amber
        (0.9, 0.2, 0.2, 0.9),     # High — red
    ]
    cmap_pri = ListedColormap(colors_pri)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    ax.imshow(classification, cmap="Greys_r", alpha=0.3)
    im = ax.imshow(priority_map, cmap=cmap_pri, vmin=-0.5, vmax=3.5,
                   alpha=0.7, interpolation="nearest")

    patches = [
        mpatches.Patch(color=colors_pri[0], label="No Intervention"),
        mpatches.Patch(color=colors_pri[1], label="Low Priority"),
        mpatches.Patch(color=colors_pri[2], label="Medium Priority"),
        mpatches.Patch(color=colors_pri[3], label="High Priority — Immediate Action"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10, framealpha=0.9,
              title="Greening Intervention Priority", title_fontsize=11)

    ax.set_title("Recommended Greening Interventions — Priority Map",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Recommendation map saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# RGB COMPOSITE
# ──────────────────────────────────────────────────────────────
def save_rgb_composite(rgb_image, output_path=None):
    """Save the RGB composite image for visual reference."""
    if rgb_image is None:
        return

    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "rgb_composite.png")

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE)
    ax.imshow(rgb_image)
    ax.set_title(f"True Color Composite (RGB) — {REGION_NAME}",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Column (pixels)", fontsize=11)
    ax.set_ylabel("Row (pixels)", fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 RGB composite saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# REPORTS
# ──────────────────────────────────────────────────────────────
def save_report(zone_df, output_path=None):
    """Save the full recommendation report as CSV."""
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "recommendations.csv")

    zone_df.to_csv(output_path, index=False)
    print(f"  💾 Report saved: {output_path}")


def save_summary_statistics(classification, ndvi, zone_df, output_path=None,
                            ml_classification=None, ndwi=None,
                            comparison_stats=None, use_ml=False):
    """
    Save a comprehensive human-readable summary report.

    UPGRADED: Includes ML model info, comparison stats, and
    academic documentation section.
    """
    ensure_output_dir()
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "summary_statistics.txt")

    total = classification.size
    class_labels = ML_LAND_CLASS_LABELS if use_ml else RULEBASED_LAND_CLASS_LABELS
    target_class = ml_classification if (use_ml and ml_classification is not None) else classification

    lines = [
        "=" * 65,
        "  🌿 GeoGreen Revolution — AI-Powered Analysis Report",
        "=" * 65,
        "",
        f"  Pilot Region     : {REGION_NAME}",
        f"  Image dimensions : {classification.shape[1]} x {classification.shape[0]} pixels",
        f"  Total pixels     : {total:,}",
        f"  NDVI range       : {np.nanmin(ndvi):.4f} to {np.nanmax(ndvi):.4f}",
        f"  NDVI mean        : {np.nanmean(ndvi):.4f}",
    ]

    if ndwi is not None:
        lines.append(f"  NDWI range       : {np.nanmin(ndwi):.4f} to {np.nanmax(ndwi):.4f}")
        water_pct = np.count_nonzero(ndwi > 0.3) / total * 100
        lines.append(f"  Water (NDWI>0.3) : {water_pct:.1f}%")

    # ── ML Model Information ──────────────────────────────────
    if use_ml:
        lines.extend([
            "",
            "  ─── AI / ML MODEL INFORMATION ───",
            f"  Model Name       : {MODEL_INFO['name']}",
            f"  Algorithm        : {MODEL_INFO['algorithm']}",
            f"  Resolution       : {MODEL_INFO['resolution']}",
            f"  Overall Accuracy : {MODEL_INFO['overall_accuracy']}",
            f"  Source           : {MODEL_INFO['source']}",
            f"  License          : {MODEL_INFO['license']}",
            "",
            "  CLASSIFICATION METHOD: ML (Pretrained Deep Learning Model)",
            "  The land-cover map is produced by ESA's pretrained model",
            "  trained on Sentinel-1/2 data using a U-Net + Random Forest",
            "  ensemble. We perform inference by loading this model output",
            "  and refining it with NDVI/NDWI spectral indices.",
        ])

    # ── Land Cover Distribution ───────────────────────────────
    lines.extend([
        "",
        "  ─── LAND COVER DISTRIBUTION ───",
    ])

    for class_id, label in class_labels.items():
        if class_id == 0 and not use_ml:
            count = np.count_nonzero(target_class == class_id)
        else:
            count = np.count_nonzero(target_class == class_id)
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        if pct > 0.05:
            lines.append(f"    {label:25s}: {pct:5.1f}%  {bar}")

    # ── Recommendation Summary ────────────────────────────────
    lines.extend([
        "",
        "  ─── RECOMMENDATION SUMMARY ───",
    ])

    if "recommendation" in zone_df.columns:
        rec_counts = zone_df["recommendation"].value_counts()
        for rec, count in rec_counts.items():
            lines.append(f"    {rec}: {count} zones")

    if "priority" in zone_df.columns:
        lines.extend([
            "",
            "  ─── PRIORITY BREAKDOWN ───",
        ])
        pri_counts = zone_df["priority"].value_counts()
        for pri, count in pri_counts.items():
            lines.append(f"    {pri:10s}: {count} zones")

    # ── ML vs Rule-Based Comparison ───────────────────────────
    if comparison_stats:
        lines.extend([
            "",
            "  ─── ML vs RULE-BASED COMPARISON ───",
            f"    Vegetation agreement : {comparison_stats.get('vegetation_agreement_pct', 'N/A')}%",
            f"    Bare land agreement  : {comparison_stats.get('bare_land_agreement_pct', 'N/A')}%",
            f"    Water agreement      : {comparison_stats.get('water_agreement_pct', 'N/A')}%",
            "",
            f"    ML Advantage: {comparison_stats.get('ml_advantage', 'N/A')}",
        ])

    # ── Academic Clarity ──────────────────────────────────────
    lines.extend([
        "",
        "  ─── METHODOLOGY CLARITY (for Academic Review) ───",
        "",
        "  AI / ML COMPONENT:",
        "    • ESA WorldCover pretrained deep learning model output",
        "    • Pixel-wise land-cover classification (7 classes)",
        "    • Refined using NDVI and NDWI spectral indices",
        "",
        "  RULE-BASED COMPONENT:",
        "    • NDVI-threshold classification (fallback/comparison)",
        "    • Climate-based feasibility assessment",
        "    • Decision-tree recommendation engine",
        "",
        "  LIMITATIONS:",
        "    • WorldCover model accuracy varies by region (~75% global)",
        "    • Climate data is aggregated (not pixel-level raster)",
        "    • Zone-based analysis uses fixed grid (not administrative boundaries)",
        "",
        "  FUTURE SCOPE:",
        "    • Fine-tune with local training data for higher accuracy",
        "    • Integrate pixel-level climate rasters (WorldClim)",
        "    • Add time-series change detection",
        "    • Deploy as web-based decision support dashboard",
        "",
    ])

    lines.extend([
        "=" * 65,
        "  Report generated by GeoGreen Revolution (AI-Upgraded)",
        "  " + MODEL_INFO.get("citation", ""),
        "=" * 65,
    ])

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"  💾 Summary statistics saved: {output_path}")
    print()
    print(text)
