"""
GeoGreen Revolution — Data Generator for Testing
===================================================
Creates synthetic satellite imagery, WorldCover classification,
and climate data for the Sehore district pilot region.

This allows you to TEST THE ENTIRE PIPELINE IMMEDIATELY
without downloading any real data.

IMPORTANT: For the final demo/report, use REAL satellite data.
           See download_data.py for real data instructions.

Usage:
    python generate_dummy_data.py
"""

import os
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠ rasterio not installed. Install with: pip install rasterio")

# ── Sehore District Region Parameters ────────────────────────
# Bounding box for Sehore district, Madhya Pradesh
WEST, SOUTH, EAST, NORTH = 76.90, 23.05, 77.20, 23.25
IMG_WIDTH, IMG_HEIGHT = 500, 500


def generate_synthetic_satellite_image():
    """
    Generate a 4-band synthetic satellite image simulating the
    Sehore district landscape with realistic land cover patterns.

    Sehore district features:
      - Narmada river tributaries (water bodies)
      - Agricultural cropland (dominant land use)
      - Sehore town (built-up area)
      - Forest patches (near Jamonia Dam)
      - Barren/fallow land

    Returns
    -------
    tuple : (blue, green, red, nir) arrays, shape (H, W), values 0–10000
    """
    np.random.seed(42)
    h, w = IMG_HEIGHT, IMG_WIDTH

    blue  = np.zeros((h, w), dtype=np.float32)
    green = np.zeros((h, w), dtype=np.float32)
    red   = np.zeros((h, w), dtype=np.float32)
    nir   = np.zeros((h, w), dtype=np.float32)

    # ── Water bodies: River flowing through the region ────────
    for i in range(h):
        river_center = int(w * 0.35 + 30 * np.sin(i / 40.0))
        river_width = 12 + int(4 * np.sin(i / 25.0))
        r_start = max(0, river_center - river_width)
        r_end = min(w, river_center + river_width)
        n = r_end - r_start
        if n > 0:
            blue[i, r_start:r_end]  = 1200 + np.random.normal(0, 60, n)
            green[i, r_start:r_end] = 1000 + np.random.normal(0, 50, n)
            red[i, r_start:r_end]   = 600  + np.random.normal(0, 40, n)
            nir[i, r_start:r_end]   = 250  + np.random.normal(0, 30, n)

    # ── Jamonia Dam (water body, top-right) ───────────────────
    dam_r, dam_c = 40, 350
    dam_h, dam_w = 60, 100
    for i in range(dam_h):
        for j in range(dam_w):
            dist = np.sqrt((i - dam_h/2)**2 + (j - dam_w/2)**2)
            if dist < min(dam_h, dam_w) / 2:
                ri, ci = dam_r + i, dam_c + j
                if 0 <= ri < h and 0 <= ci < w:
                    blue[ri, ci]  = 1300 + np.random.normal(0, 50)
                    green[ri, ci] = 1100 + np.random.normal(0, 40)
                    red[ri, ci]   = 500  + np.random.normal(0, 30)
                    nir[ri, ci]   = 200  + np.random.normal(0, 20)

    # ── Sehore Town (built-up, center-left) ───────────────────
    urban_r, urban_c = 180, 60
    urban_h, urban_w = 120, 100
    noise = np.random.normal(0, 100, (urban_h, urban_w))
    blue[urban_r:urban_r+urban_h, urban_c:urban_c+urban_w]  = 1500 + noise
    green[urban_r:urban_r+urban_h, urban_c:urban_c+urban_w] = 1450 + noise
    red[urban_r:urban_r+urban_h, urban_c:urban_c+urban_w]   = 1650 + noise * 1.1
    nir[urban_r:urban_r+urban_h, urban_c:urban_c+urban_w]   = 1900 + noise * 0.9

    # ── Agricultural Fields (dominant — multiple patches) ─────
    agri_patches = [
        (10, 10, 150, 140),     # Top-left fields
        (320, 20, 150, 160),    # Bottom-left fields
        (200, 250, 130, 180),   # Center-right fields
    ]
    for ar, ac, ah, aw in agri_patches:
        noise = np.random.normal(0, 80, (ah, aw))
        stripe = np.sin(np.arange(aw) / 7.0) * 250
        blue[ar:ar+ah, ac:ac+aw]  = 800 + noise * 0.5
        green[ar:ar+ah, ac:ac+aw] = 1700 + noise + stripe
        red[ar:ar+ah, ac:ac+aw]   = 900 + noise * 0.6
        nir[ar:ar+ah, ac:ac+aw]   = 4200 + noise * 1.8 + stripe

    # ── Forest Patches (near Jamonia Dam area) ────────────────
    forest_patches = [
        (30, 250, 100, 90),     # Forest near dam
        (380, 360, 100, 120),   # Southern forest
    ]
    for fr, fc, fh, fw in forest_patches:
        noise = np.random.normal(0, 90, (fh, fw))
        texture_x = np.sin(np.arange(fw) / 10.0) * 200
        texture_y = np.cos(np.arange(fh) / 12.0) * 180
        canopy = texture_x[None, :] + texture_y[:, None]
        blue[fr:fr+fh, fc:fc+fw]  = 450 + noise * 0.4
        green[fr:fr+fh, fc:fc+fw] = 1400 + noise * 0.8 + canopy * 0.5
        red[fr:fr+fh, fc:fc+fw]   = 550 + noise * 0.4
        nir[fr:fr+fh, fc:fc+fw]   = 5800 + noise * 2.0 + canopy

    # ── Barren / Fallow Land ──────────────────────────────────
    barren_r, barren_c = 170, 350
    barren_h, barren_w = 100, 130
    noise = np.random.normal(0, 110, (barren_h, barren_w))
    blue[barren_r:barren_r+barren_h, barren_c:barren_c+barren_w]  = 1350 + noise
    green[barren_r:barren_r+barren_h, barren_c:barren_c+barren_w] = 1450 + noise
    red[barren_r:barren_r+barren_h, barren_c:barren_c+barren_w]   = 1950 + noise * 1.2
    nir[barren_r:barren_r+barren_h, barren_c:barren_c+barren_w]   = 2100 + noise * 0.7

    # ── Background: sparse scrubland / grassland ──────────────
    mask_empty = (blue == 0) & (green == 0) & (red == 0) & (nir == 0)
    bg_count = np.count_nonzero(mask_empty)
    if bg_count > 0:
        blue[mask_empty]  = 900  + np.random.normal(0, 80, bg_count)
        green[mask_empty] = 1300 + np.random.normal(0, 90, bg_count)
        red[mask_empty]   = 1100 + np.random.normal(0, 100, bg_count)
        nir[mask_empty]   = 3000 + np.random.normal(0, 200, bg_count)

    # Clip to valid Sentinel-2 range
    blue  = np.clip(blue, 0, 10000)
    green = np.clip(green, 0, 10000)
    red   = np.clip(red, 0, 10000)
    nir   = np.clip(nir, 0, 10000)

    return blue, green, red, nir


def generate_synthetic_worldcover():
    """
    Generate a synthetic ESA WorldCover classification for the
    Sehore region. Uses the same land-cover layout as the satellite image.

    WorldCover class values:
      10 = Tree cover
      20 = Shrubland
      30 = Grassland
      40 = Cropland
      50 = Built-up
      60 = Bare/sparse
      80 = Water

    Returns
    -------
    numpy.ndarray : WorldCover classification, shape (H, W), dtype uint8
    """
    np.random.seed(42)
    h, w = IMG_HEIGHT, IMG_WIDTH

    # Start with grassland/shrubland as default
    wc = np.full((h, w), fill_value=30, dtype=np.uint8)  # Grassland default

    # ── Water bodies ──────────────────────────────────────────
    for i in range(h):
        river_center = int(w * 0.35 + 30 * np.sin(i / 40.0))
        river_width = 12 + int(4 * np.sin(i / 25.0))
        r_start = max(0, river_center - river_width)
        r_end = min(w, river_center + river_width)
        wc[i, r_start:r_end] = 80  # Water

    # Jamonia Dam
    dam_r, dam_c = 40, 350
    dam_h, dam_w = 60, 100
    for i in range(dam_h):
        for j in range(dam_w):
            dist = np.sqrt((i - dam_h/2)**2 + (j - dam_w/2)**2)
            if dist < min(dam_h, dam_w) / 2:
                ri, ci = dam_r + i, dam_c + j
                if 0 <= ri < h and 0 <= ci < w:
                    wc[ri, ci] = 80  # Water

    # ── Built-up (Sehore town) ────────────────────────────────
    wc[180:300, 60:160] = 50   # Built-up

    # ── Cropland ──────────────────────────────────────────────
    wc[10:160, 10:150] = 40    # Agricultural fields
    wc[320:470, 20:180] = 40
    wc[200:330, 250:430] = 40

    # ── Forest ────────────────────────────────────────────────
    wc[30:130, 250:340] = 10   # Tree cover near dam
    wc[380:480, 360:480] = 10  # Southern forest

    # ── Shrubland (transition zones) ──────────────────────────
    wc[130:180, 10:150] = 20   # Between cropland and town
    wc[300:330, 20:180] = 20   # Between town and lower cropland

    # ── Bare/sparse land ──────────────────────────────────────
    wc[170:270, 350:480] = 60  # Barren area

    # Add some noise/variability within regions
    noise_mask = np.random.random((h, w)) < 0.03  # 3% of pixels get randomized
    noise_values = np.random.choice([10, 20, 30, 40], size=(h, w))
    wc[noise_mask] = noise_values[noise_mask]

    return wc


def generate_sehore_climate_csv():
    """
    Generate climate data CSV for Sehore district.

    Based on real IMD / WorldClim data for Sehore:
      - Annual rainfall: ~1000-1200 mm (monsoon-dominated)
      - Summer temperature: 36-42°C
      - Soil: Black cotton (vertisol), alluvial

    Returns
    -------
    str : CSV content
    """
    csv_content = """zone_id,rainfall_mm,temperature_c,soil_type
0,1050,38,black_cotton
1,980,39,alluvial
2,1120,37,black_cotton
3,950,40,laterite
4,1200,36,alluvial
5,1080,38,black_cotton
6,900,41,sandy_loam
7,1150,37,alluvial
8,1050,39,black_cotton
9,1000,38,alluvial
"""
    return csv_content


def main():
    """Generate all sample data files for the Sehore district pilot."""
    print()
    print("=" * 65)
    print("  🌿 GeoGreen Revolution — Sample Data Generator")
    print("  Pilot Region: Sehore District, Madhya Pradesh")
    print("=" * 65)
    print()

    project_root = os.path.dirname(os.path.abspath(__file__))

    # Create directories
    sat_dir = os.path.join(project_root, "data", "satellite")
    clim_dir = os.path.join(project_root, "data", "climate")
    wc_dir = os.path.join(project_root, "data", "worldcover")
    os.makedirs(sat_dir, exist_ok=True)
    os.makedirs(clim_dir, exist_ok=True)
    os.makedirs(wc_dir, exist_ok=True)

    # ── 1. Generate Satellite Image ───────────────────────────
    print("📌 Generating synthetic Sentinel-2 satellite image...")
    blue, green, red, nir = generate_synthetic_satellite_image()

    sat_path = os.path.join(sat_dir, "sehore_sentinel2.tif")

    if HAS_RASTERIO:
        transform = from_bounds(WEST, SOUTH, EAST, NORTH, IMG_WIDTH, IMG_HEIGHT)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT,
            "count": 4,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        with rasterio.open(sat_path, "w", **profile) as dst:
            dst.write(blue,  1)
            dst.write(green, 2)
            dst.write(red,   3)
            dst.write(nir,   4)

        print(f"  ✅ Satellite image saved: {sat_path}")
        print(f"     Format: GeoTIFF, 4 bands (B/G/R/NIR), {IMG_WIDTH}x{IMG_HEIGHT} pixels")
        print(f"     CRS: EPSG:4326 (WGS84)")
        print(f"     Region: Sehore District, MP ({WEST}°E, {SOUTH}°N - {EAST}°E, {NORTH}°N)")
    else:
        npy_path = sat_path.replace(".tif", ".npy")
        np.save(npy_path, np.stack([blue, green, red, nir]))
        print(f"  ✅ Satellite data saved as numpy: {npy_path}")
        print(f"     ⚠ Install rasterio for GeoTIFF support: pip install rasterio")

    # ── 2. Generate WorldCover Classification ─────────────────
    print()
    print("📌 Generating synthetic ESA WorldCover classification...")
    worldcover = generate_synthetic_worldcover()

    wc_path = os.path.join(wc_dir, "worldcover_sehore.tif")

    if HAS_RASTERIO:
        transform = from_bounds(WEST, SOUTH, EAST, NORTH, IMG_WIDTH, IMG_HEIGHT)

        wc_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": IMG_WIDTH,
            "height": IMG_HEIGHT,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        with rasterio.open(wc_path, "w", **wc_profile) as dst:
            dst.write(worldcover, 1)

        print(f"  ✅ WorldCover map saved: {wc_path}")
        print(f"     Format: GeoTIFF, single-band, {IMG_WIDTH}x{IMG_HEIGHT} pixels")

        # Print class distribution
        unique, counts = np.unique(worldcover, return_counts=True)
        wc_labels = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop",
                     50: "Built-up", 60: "Bare", 80: "Water"}
        for val, cnt in zip(unique, counts):
            pct = cnt / worldcover.size * 100
            label = wc_labels.get(val, f"Class {val}")
            print(f"     {label:10s} (ESA {val}): {pct:5.1f}%")
    else:
        npy_path = wc_path.replace(".tif", ".npy")
        np.save(npy_path, worldcover)
        print(f"  ✅ WorldCover data saved as numpy: {npy_path}")

    # ── 3. Generate Climate Data ──────────────────────────────
    print()
    print("📌 Generating Sehore district climate data...")
    csv_content = generate_sehore_climate_csv()
    clim_path = os.path.join(clim_dir, "sehore_climate.csv")

    with open(clim_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    print(f"  ✅ Climate data saved: {clim_path}")
    print(f"     Zones: 10, Variables: rainfall, temperature, soil type")
    print(f"     Region: Sehore District, MP (based on IMD averages)")

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  ✅ All sample data generated successfully!")
    print()
    print("  Files created:")
    print(f"    📡 {sat_path}")
    print(f"    🤖 {wc_path}")
    print(f"    🌤️  {clim_path}")
    print()
    print("  Next step: Run the AI pipeline:")
    print("    python src/main.py")
    print()
    print("  The pipeline will auto-detect all three data files")
    print("  and run in FULL ML MODE (WorldCover + NDVI/NDWI).")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
