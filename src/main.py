"""
GeoGreen Revolution — Main Pipeline (AI-Upgraded)
====================================================
Entry point for the AI-powered geospatial analysis pipeline.

UPGRADED PIPELINE:
  Load → Preprocess → ML Inference → Indices → Fuse → Recommend → Output

Two modes:
  1. WITH WorldCover (ML mode): Full AI-powered pipeline
  2. WITHOUT WorldCover (fallback): Rule-based NDVI classification only

Usage:
    python src/main.py
    python src/main.py --worldcover data/worldcover/worldcover_sehore.tif
    python src/main.py --satellite data/satellite/sehore.tif --climate data/climate/sehore_climate.csv
"""

import os
import sys
import argparse
import time

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SATELLITE_DIR, CLIMATE_DIR, WORLDCOVER_DIR, OUTPUT_DIR,
    SATELLITE_FILE, CLIMATE_FILE, WORLDCOVER_FILE,
    REGION_NAME, MODEL_INFO,
)
from preprocessing import (
    load_satellite_image, normalize_bands,
    load_climate_data, create_rgb_composite, auto_detect_file,
)
from indices import compute_ndvi, compute_ndwi, compute_vegetation_health
from analysis import (
    classify_land_rulebased, compute_zone_statistics,
    assign_climate_to_zones, generate_recommendations,
    generate_recommendations_ml,
)
from ml_inference import (
    perform_inference, refine_with_indices,
    compare_ml_vs_rulebased, get_model_metadata,
)
from utils import (
    save_ndvi_map, save_ndwi_map, save_classification_map,
    save_ml_landcover_map, save_fused_map,
    save_recommendation_map, save_rgb_composite,
    save_report, save_summary_statistics, ensure_output_dir,
)


def print_banner():
    """Print the project banner."""
    print()
    print("=" * 65)
    print("  🌿 GeoGreen Revolution")
    print("  AI-Powered Geospatial Decision Support System")
    print("  ──────────────────────────────────────────────")
    print(f"  Pilot Region : {REGION_NAME}")
    print(f"  ML Model     : {MODEL_INFO['name']}")
    print("=" * 65)
    print()


def resolve_file_path(directory, config_file, extension, label):
    """
    Resolve the file path: use config override, or auto-detect.
    """
    if config_file:
        path = os.path.join(directory, config_file)
        if os.path.exists(path):
            return path
        elif os.path.exists(config_file):
            return config_file

    # Auto-detect
    path = auto_detect_file(directory, extension)
    if path:
        print(f"  📁 Auto-detected {label}: {os.path.basename(path)}")
        return path

    return None


def main(satellite_path=None, climate_path=None, worldcover_path=None,
         grid_size=50):
    """
    Run the complete GeoGreen Revolution AI-powered analysis pipeline.

    Parameters
    ----------
    satellite_path : str, optional
        Path to the satellite GeoTIFF. Auto-detected if None.
    climate_path : str, optional
        Path to the climate CSV. Auto-detected if None.
    worldcover_path : str, optional
        Path to the ESA WorldCover GeoTIFF. Auto-detected if None.
    grid_size : int
        Grid cell size for zone statistics (pixels).
    """
    print_banner()
    start_time = time.time()

    # ──────────────────────────────────────────────────────────
    # STEP 1: Resolve Input Files
    # ──────────────────────────────────────────────────────────
    print("📌 Step 1: Locating input files...")

    if satellite_path is None:
        satellite_path = resolve_file_path(
            SATELLITE_DIR, SATELLITE_FILE, ".tif", "satellite image"
        )
    if satellite_path is None:
        print("  ❌ ERROR: No satellite image found!")
        print(f"     Place a .tif file in: {SATELLITE_DIR}")
        print("     Or run: python generate_dummy_data.py")
        sys.exit(1)

    if climate_path is None:
        climate_path = resolve_file_path(
            CLIMATE_DIR, CLIMATE_FILE, ".csv", "climate data"
        )
    if climate_path is None:
        print("  ❌ ERROR: No climate data file found!")
        print(f"     Place a .csv file in: {CLIMATE_DIR}")
        print("     Or run: python generate_dummy_data.py")
        sys.exit(1)

    # WorldCover is optional — enables ML mode
    has_worldcover = False
    if worldcover_path is None:
        worldcover_path = resolve_file_path(
            WORLDCOVER_DIR, WORLDCOVER_FILE, ".tif", "WorldCover map"
        )
    if worldcover_path and os.path.exists(worldcover_path):
        has_worldcover = True
        print(f"  🤖 ML MODE: WorldCover map found — full AI pipeline enabled")
    else:
        print(f"  ℹ️  FALLBACK MODE: No WorldCover map — using rule-based classification")
        print(f"     To enable ML mode, place a WorldCover .tif in: {WORLDCOVER_DIR}")

    print()

    # ──────────────────────────────────────────────────────────
    # STEP 2: Load & Preprocess Satellite Data
    # ──────────────────────────────────────────────────────────
    print("📌 Step 2: Loading and preprocessing satellite data...")

    satellite_data = load_satellite_image(satellite_path)
    bands = normalize_bands(satellite_data["bands"])
    climate_df = load_climate_data(climate_path)
    sat_shape = satellite_data["shape"]
    sat_profile = satellite_data["profile"]
    print()

    # ──────────────────────────────────────────────────────────
    # STEP 3: Compute Spectral Indices (NDVI + NDWI)
    # ──────────────────────────────────────────────────────────
    print("📌 Step 3: Computing spectral indices...")

    nir = bands.get("nir")
    red = bands.get("red")
    green = bands.get("green")

    if nir is None or red is None:
        print("  ❌ ERROR: NIR and Red bands are required for NDVI.")
        sys.exit(1)

    ndvi = compute_ndvi(nir, red)

    ndwi = None
    if green is not None and nir is not None:
        ndwi = compute_ndwi(green, nir)
    print()

    # ──────────────────────────────────────────────────────────
    # STEP 4: ML Land-Cover Inference (or Rule-Based Fallback)
    # ──────────────────────────────────────────────────────────
    ml_classification = None
    fused_classification = None
    comparison_stats = None

    if has_worldcover:
        print("📌 Step 4: AI/ML Land-Cover Inference (ESA WorldCover)...")
        print("  ──────────────────────────────────────────────")
        print("  This step uses a PRETRAINED DEEP LEARNING MODEL")
        print("  trained by ESA on Sentinel-1/2 satellite imagery.")
        print("  We perform inference by loading the model output")
        print("  and post-processing it for our analysis.")
        print("  ──────────────────────────────────────────────")

        # Perform ML inference
        ml_result = perform_inference(
            worldcover_path,
            target_shape=sat_shape,
            target_profile=sat_profile,
        )
        ml_classification = ml_result["ml_classification"]

        # Refine ML classification with spectral indices
        print()
        print("  🔬 Refining ML classification with spectral indices...")
        fused_classification = refine_with_indices(
            ml_classification, ndvi, ndwi
        )
        print()
    else:
        print("📌 Step 4: Rule-Based Land Classification (NDVI Thresholds)...")

    # Always compute rule-based classification (for comparison or as primary)
    rulebased_classification = classify_land_rulebased(ndvi)
    print()

    # Compare ML vs rule-based if both available
    if ml_classification is not None:
        print("📌 Step 4b: Comparing ML vs Rule-Based Classification...")
        comparison_stats = compare_ml_vs_rulebased(
            ml_classification, rulebased_classification
        )
        print()

    # Determine primary classification for downstream analysis
    primary_classification = fused_classification if fused_classification is not None \
        else rulebased_classification
    use_ml = fused_classification is not None

    # ──────────────────────────────────────────────────────────
    # STEP 5: Zone Statistics & Climate Merge
    # ──────────────────────────────────────────────────────────
    print("📌 Step 5: Computing zone statistics...")

    zone_df = compute_zone_statistics(
        primary_classification, ndvi,
        grid_size=grid_size,
        use_ml_classes=use_ml,
        ndwi=ndwi,
    )
    zone_df = assign_climate_to_zones(zone_df, climate_df)
    print()

    # ──────────────────────────────────────────────────────────
    # STEP 6: Generate Recommendations
    # ──────────────────────────────────────────────────────────
    print("📌 Step 6: Generating greening recommendations...")

    if use_ml:
        zone_df = generate_recommendations_ml(zone_df)
    else:
        zone_df = generate_recommendations(zone_df)
    print()

    # Compute vegetation health
    veg_health = compute_vegetation_health(
        ndvi, ml_classification if use_ml else None
    )
    print()

    # ──────────────────────────────────────────────────────────
    # STEP 7: Save All Outputs
    # ──────────────────────────────────────────────────────────
    print("📌 Step 7: Saving outputs...")
    print()

    ensure_output_dir()

    # Always save: NDVI, RGB, rule-based classification
    save_ndvi_map(ndvi)
    save_classification_map(rulebased_classification)

    # Save NDWI map
    if ndwi is not None:
        save_ndwi_map(ndwi)

    # Save ML-specific maps
    if ml_classification is not None:
        save_ml_landcover_map(ml_classification)

    if fused_classification is not None:
        save_fused_map(fused_classification, ml_classification=ml_classification)

    # Recommendation map (uses primary classification)
    save_recommendation_map(primary_classification, zone_df, grid_size=grid_size)

    # RGB Composite
    rgb = create_rgb_composite(bands)
    save_rgb_composite(rgb)

    # Reports
    save_report(zone_df)
    save_summary_statistics(
        rulebased_classification, ndvi, zone_df,
        ml_classification=ml_classification,
        ndwi=ndwi,
        comparison_stats=comparison_stats,
        use_ml=use_ml,
    )

    # ──────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("=" * 65)
    print(f"  ✅ Analysis complete! (Time: {elapsed:.1f} seconds)")
    print(f"  📂 Results saved to: {OUTPUT_DIR}")
    if use_ml:
        print(f"  🤖 Mode: AI/ML (ESA WorldCover + NDVI/NDWI refinement)")
    else:
        print(f"  📊 Mode: Rule-Based (NDVI thresholds)")
    print()
    print("  Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    📄 {f:40s} ({size_kb:.0f} KB)")
    print()
    print("=" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoGreen Revolution — AI-Powered Geospatial Analysis Pipeline"
    )
    parser.add_argument(
        "--satellite", type=str, default=None,
        help="Path to satellite GeoTIFF image"
    )
    parser.add_argument(
        "--climate", type=str, default=None,
        help="Path to climate data CSV"
    )
    parser.add_argument(
        "--worldcover", type=str, default=None,
        help="Path to ESA WorldCover GeoTIFF (enables ML mode)"
    )
    parser.add_argument(
        "--grid-size", type=int, default=50,
        help="Grid cell size for zone analysis (pixels, default: 50)"
    )

    args = parser.parse_args()
    main(
        satellite_path=args.satellite,
        climate_path=args.climate,
        worldcover_path=args.worldcover,
        grid_size=args.grid_size,
    )
