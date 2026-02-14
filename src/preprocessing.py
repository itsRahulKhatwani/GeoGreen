"""
GeoGreen Revolution — Preprocessing Module
============================================
Handles loading satellite imagery (GeoTIFF), ESA WorldCover maps,
and climate data (CSV). Normalizes pixel values and extracts
individual spectral bands.

UPGRADED: Added WorldCover loading, AOI clipping, and raster alignment.
"""

import os
import numpy as np
import rasterio
import pandas as pd

from config import BAND_ORDER, REGION_BBOX


def load_satellite_image(filepath):
    """
    Load a multi-band GeoTIFF satellite image.

    Parameters
    ----------
    filepath : str
        Path to the GeoTIFF file.

    Returns
    -------
    dict with keys:
        'bands'     : dict of numpy arrays keyed by band name ('red', 'nir', etc.)
        'profile'   : rasterio profile (metadata — CRS, transform, dimensions)
        'shape'     : tuple (height, width)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Satellite image not found: {filepath}")

    with rasterio.open(filepath) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

        bands = {}
        for band_name, band_index in BAND_ORDER.items():
            if band_index <= src.count:
                band_data = src.read(band_index).astype(np.float32)
                bands[band_name] = band_data
            else:
                print(f"  ⚠ Warning: Band '{band_name}' (index {band_index}) not found. "
                      f"Image has {src.count} bands.")

    print(f"  ✅ Loaded satellite image: {os.path.basename(filepath)}")
    print(f"     Dimensions: {width} x {height} pixels")
    print(f"     Bands loaded: {list(bands.keys())}")

    return {
        "bands": bands,
        "profile": profile,
        "shape": (height, width),
    }


def normalize_bands(bands_dict):
    """
    Normalize band values to the 0–1 range.

    Sentinel-2 L2A reflectance values are typically 0–10000.
    If values exceed 1.0, we divide by 10000.
    If already in 0–1, no change is made.

    Parameters
    ----------
    bands_dict : dict
        Dictionary of band_name -> numpy array.

    Returns
    -------
    dict : Normalized band arrays.
    """
    normalized = {}
    for name, data in bands_dict.items():
        if np.nanmax(data) > 1.0:
            # Sentinel-2 reflectance scaling: values are 0–10000
            normalized[name] = np.clip(data / 10000.0, 0.0, 1.0)
            print(f"     Normalized '{name}' band (max was {np.nanmax(data):.0f} → scaled to 0–1)")
        else:
            normalized[name] = np.clip(data, 0.0, 1.0)
    return normalized


def load_climate_data(filepath):
    """
    Load climate data from a CSV file.

    Expected CSV columns:
        - zone_id       : Identifier for the zone/grid cell
        - rainfall_mm   : Annual rainfall in millimeters
        - temperature_c : Average summer temperature in Celsius
        - soil_type     : Soil type (e.g., alluvial, laterite, sandy, clay, black)

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Climate data file not found: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = ["zone_id", "rainfall_mm", "temperature_c", "soil_type"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Climate CSV is missing required columns: {missing}")

    print(f"  ✅ Loaded climate data: {os.path.basename(filepath)}")
    print(f"     Zones: {len(df)}, Columns: {list(df.columns)}")

    return df


def create_rgb_composite(bands_dict):
    """
    Create an RGB composite image from individual bands for visualization.

    Parameters
    ----------
    bands_dict : dict
        Normalized band arrays.

    Returns
    -------
    numpy.ndarray : RGB image of shape (H, W, 3), values 0–1.
    """
    red = bands_dict.get("red")
    green = bands_dict.get("green")
    blue = bands_dict.get("blue")

    if red is None or green is None or blue is None:
        print("  ⚠ Cannot create RGB composite — missing bands.")
        return None

    # Stack and apply a simple contrast stretch
    rgb = np.stack([red, green, blue], axis=-1)

    # Percentile-based contrast stretch for better visualization
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)

    return rgb


def auto_detect_file(directory, extension):
    """
    Auto-detect a file in a directory by extension.
    Returns the first matching file path, or None.
    """
    if not os.path.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if fname.lower().endswith(extension):
            return os.path.join(directory, fname)
    return None


def clip_raster_to_aoi(data, profile, bbox=None):
    """
    Clip a raster array to the area of interest (AOI) bounding box.

    If the raster is already within the AOI or bbox is None,
    returns the data unchanged.

    Parameters
    ----------
    data : numpy.ndarray
        Raster data (2D).
    profile : dict
        Rasterio profile with 'transform' key.
    bbox : dict, optional
        Bounding box with keys: west, south, east, north.
        Defaults to REGION_BBOX from config.

    Returns
    -------
    tuple : (clipped_data, clipped_profile)
    """
    if bbox is None:
        bbox = REGION_BBOX

    if "transform" not in profile:
        return data, profile

    transform = profile["transform"]

    # Convert geographic coordinates to pixel coordinates
    # transform: (a, b, c, d, e, f) where c,f = origin x,y
    inv_transform = ~transform
    col_start, row_start = inv_transform * (bbox["west"], bbox["north"])
    col_end, row_end = inv_transform * (bbox["east"], bbox["south"])

    # Ensure integer pixel indices
    row_start = max(0, int(row_start))
    row_end = min(data.shape[0], int(row_end) + 1)
    col_start = max(0, int(col_start))
    col_end = min(data.shape[1], int(col_end) + 1)

    if row_start >= row_end or col_start >= col_end:
        print("  ⚠ AOI clip resulted in empty region, returning original data.")
        return data, profile

    clipped = data[row_start:row_end, col_start:col_end]

    # Update profile
    new_transform = rasterio.transform.from_bounds(
        bbox["west"], bbox["south"], bbox["east"], bbox["north"],
        clipped.shape[1], clipped.shape[0]
    )
    clipped_profile = profile.copy()
    clipped_profile.update({
        "height": clipped.shape[0],
        "width": clipped.shape[1],
        "transform": new_transform,
    })

    print(f"     Clipped raster from {data.shape} to {clipped.shape}")
    return clipped, clipped_profile


def align_worldcover_to_satellite(wc_data, wc_profile, sat_shape, sat_profile):
    """
    Align WorldCover raster to match the satellite image dimensions.

    Uses nearest-neighbor resampling to preserve categorical labels.

    Parameters
    ----------
    wc_data : numpy.ndarray
        WorldCover raster (2D).
    wc_profile : dict
        WorldCover rasterio profile.
    sat_shape : tuple
        Target (height, width) from satellite image.
    sat_profile : dict
        Satellite image rasterio profile.

    Returns
    -------
    numpy.ndarray : Aligned WorldCover raster matching sat_shape.
    """
    if wc_data.shape == sat_shape:
        return wc_data

    # Simple nearest-neighbor resample
    row_idx = np.linspace(0, wc_data.shape[0] - 1, sat_shape[0]).astype(int)
    col_idx = np.linspace(0, wc_data.shape[1] - 1, sat_shape[1]).astype(int)
    aligned = wc_data[np.ix_(row_idx, col_idx)]

    print(f"     Aligned WorldCover from {wc_data.shape} → {sat_shape}")
    return aligned
