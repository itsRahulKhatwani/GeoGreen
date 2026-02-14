"""
GeoGreen Revolution — Spectral Indices Module
===============================================
Computes vegetation and spectral indices from satellite imagery.

Indices computed:
  - NDVI (Normalized Difference Vegetation Index)
  - NDWI (Normalized Difference Water Index)
  - NDBI (Normalized Difference Built-up Index) — optional

UPGRADED: NDWI now integrated into main pipeline for water detection
and ML classification refinement.
"""

import numpy as np


def compute_ndvi(nir_band, red_band):
    """
    Compute the Normalized Difference Vegetation Index (NDVI).

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    NDVI ranges from -1 to +1:
        < 0.0   : Water
        0.0–0.12: Built-up / Barren
        0.12–0.25: Sparse vegetation
        0.25–0.45: Moderate vegetation
        > 0.45  : Dense vegetation

    Parameters
    ----------
    nir_band : numpy.ndarray
        Near-Infrared band (Band 8 for Sentinel-2), values 0–1.
    red_band : numpy.ndarray
        Red band (Band 4 for Sentinel-2), values 0–1.

    Returns
    -------
    numpy.ndarray
        NDVI values in the range [-1, 1].
    """
    # Avoid division by zero
    denominator = nir_band + red_band
    denominator = np.where(denominator == 0, np.nan, denominator)

    ndvi = (nir_band - red_band) / denominator

    # Clip to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Replace NaN with 0 (no-data areas)
    ndvi = np.nan_to_num(ndvi, nan=0.0)

    # Statistics
    valid_pixels = np.count_nonzero(~np.isnan(denominator))
    print(f"  ✅ NDVI computed successfully")
    print(f"     Min: {np.nanmin(ndvi):.4f}, Max: {np.nanmax(ndvi):.4f}, "
          f"Mean: {np.nanmean(ndvi):.4f}")
    print(f"     Valid pixels: {valid_pixels:,}")

    return ndvi


def compute_ndwi(green_band, nir_band):
    """
    Compute the Normalized Difference Water Index (NDWI).

    Formula (McFeeters, 1996):
        NDWI = (Green - NIR) / (Green + NIR)

    NDWI is used to:
      - Identify water bodies (NDWI > 0.3)
      - Refine ML land-cover classification
      - Distinguish irrigated vs rainfed cropland

    Parameters
    ----------
    green_band : numpy.ndarray
        Green band (Band 3 for Sentinel-2), values 0–1.
    nir_band : numpy.ndarray
        Near-Infrared band (Band 8 for Sentinel-2), values 0–1.

    Returns
    -------
    numpy.ndarray
        NDWI values.
    """
    denominator = green_band + nir_band
    denominator = np.where(denominator == 0, np.nan, denominator)

    ndwi = (green_band - nir_band) / denominator
    ndwi = np.nan_to_num(ndwi, nan=0.0)

    print(f"  ✅ NDWI computed (Water Index)")
    print(f"     Min: {np.nanmin(ndwi):.4f}, Max: {np.nanmax(ndwi):.4f}, "
          f"Mean: {np.nanmean(ndwi):.4f}")

    # Report water pixel count
    water_pixels = np.count_nonzero(ndwi > 0.3)
    total = ndwi.size
    print(f"     Water pixels (NDWI > 0.3): {water_pixels:,} ({water_pixels/total*100:.1f}%)")

    return ndwi


def compute_vegetation_health(ndvi, ml_classification=None):
    """
    Compute a vegetation health score per-pixel, factoring in
    land-cover context from ML classification.

    Health score (0–100):
      - Based on NDVI normalized to 0–100 for vegetated areas
      - Non-vegetated areas get score 0
      - Used for identifying degraded cropland and stressed forests

    Parameters
    ----------
    ndvi : numpy.ndarray
        NDVI values.
    ml_classification : numpy.ndarray, optional
        ML land-cover classification (for context-aware scoring).

    Returns
    -------
    numpy.ndarray
        Vegetation health score (0–100).
    """
    # Normalize NDVI to 0–100 score
    health = np.clip((ndvi + 0.1) / 0.8 * 100, 0, 100).astype(np.float32)

    # If ML classification available, zero out non-vegetated areas
    if ml_classification is not None:
        non_veg_mask = (ml_classification == 5) | \
                       (ml_classification == 6) | \
                       (ml_classification == 7)
        health[non_veg_mask] = 0.0

    mean_health = np.mean(health[health > 0]) if np.any(health > 0) else 0.0
    print(f"  ✅ Vegetation health score computed")
    print(f"     Mean health (vegetated areas): {mean_health:.1f}/100")

    return health
