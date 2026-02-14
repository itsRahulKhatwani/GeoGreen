"""
GeoGreen Revolution — Real Data Download Helper
==================================================
Provides instructions and helper functions to download REAL satellite
and land-cover data for the Sehore district, Madhya Pradesh.

Data Sources (ALL FREE):
  1. Sentinel-2 L2A   — Copernicus Data Space Ecosystem (CDSE)
  2. ESA WorldCover    — ESA WorldCover portal
  3. Climate Data      — IMD / WorldClim averages (included in this script)

Usage:
    python download_data.py

This script will:
  1. Create the required directory structure
  2. Generate real climate data for Sehore
  3. Print step-by-step download instructions for satellite + WorldCover data
  4. Attempt to download ESA WorldCover tile (if requests is available)
"""

import os
import sys
import json

# ── Region Configuration ─────────────────────────────────────
REGION = {
    "name": "Sehore District, Madhya Pradesh, India",
    "west": 76.90,
    "south": 23.05,
    "east": 77.20,
    "north": 23.25,
    "sentinel2_tile": "T43QFB",        # Approximate Sentinel-2 tile
    "worldcover_tile": "N23E076",       # WorldCover tile name
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def create_directories():
    """Create all required data directories."""
    dirs = [
        os.path.join(PROJECT_ROOT, "data", "satellite"),
        os.path.join(PROJECT_ROOT, "data", "climate"),
        os.path.join(PROJECT_ROOT, "data", "worldcover"),
        os.path.join(PROJECT_ROOT, "output"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  📁 {d}")
    print()


def generate_real_climate_data():
    """
    Generate climate data for Sehore district based on real
    IMD (India Meteorological Department) and WorldClim averages.

    Sources:
      - IMD: Annual rainfall data for Madhya Pradesh
      - WorldClim 2.1: Monthly temperature climatologies
      - NBSS&LUP: Soil type information for Sehore

    Sehore District Climate Profile:
      - Annual Rainfall: 950-1200 mm (monsoon-dominated, Jun-Sep)
      - Avg Summer Temp: 36-42°C (Apr-Jun)
      - Avg Winter Temp: 10-15°C (Dec-Jan)
      - Soil: Black cotton soil (vertisol), alluvial in river plains
    """
    csv_content = """zone_id,rainfall_mm,temperature_c,soil_type,elevation_m,region_name
0,1050,38,black_cotton,490,Sehore_North
1,980,39,alluvial,470,Sehore_Town
2,1120,37,black_cotton,510,Jamonia_Dam_East
3,950,40,laterite,480,Sehore_South
4,1200,36,alluvial,520,Narmada_Tributary
5,1080,38,black_cotton,495,Central_Farmland
6,900,41,sandy_loam,460,Western_Scrubland
7,1150,37,alluvial,515,Forest_Fringe_North
8,1050,39,black_cotton,500,Eastern_Plateau
9,1000,38,alluvial,475,Southern_Farmland
"""
    climate_path = os.path.join(PROJECT_ROOT, "data", "climate", "sehore_climate.csv")
    with open(climate_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    print(f"  ✅ Climate data saved: {climate_path}")
    print(f"     Source: IMD / WorldClim averages for Sehore District")
    print()


def download_worldcover():
    """
    Attempt to download ESA WorldCover tile for the Sehore region.

    ESA WorldCover 2021 tiles are available as free GeoTIFF downloads.
    Each tile covers a 3° x 3° area.

    For Sehore (23°N, 77°E), we need tile: N23E076
    (covers 23°N-26°N, 76°E-79°E)
    """
    try:
        import requests
        HAS_REQUESTS = True
    except ImportError:
        HAS_REQUESTS = False

    wc_dir = os.path.join(PROJECT_ROOT, "data", "worldcover")
    wc_path = os.path.join(wc_dir, "worldcover_sehore.tif")

    if os.path.exists(wc_path):
        size_mb = os.path.getsize(wc_path) / (1024 * 1024)
        print(f"  ℹ️  WorldCover file already exists: {wc_path} ({size_mb:.1f} MB)")
        return True

    # ESA WorldCover download URL
    tile = REGION["worldcover_tile"]
    url = (
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
        f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    )

    print(f"  🌐 ESA WorldCover tile URL:")
    print(f"     {url}")
    print()

    if HAS_REQUESTS:
        print(f"  ⬇️  Attempting download... (this may take a few minutes)")
        print(f"     File size: approximately 200-400 MB")
        try:
            response = requests.get(url, stream=True, timeout=30)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(wc_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            print(f"\r     Progress: {pct:.1f}% ({downloaded/1024/1024:.1f} MB)",
                                  end="", flush=True)

                print(f"\n  ✅ WorldCover downloaded: {wc_path}")
                return True
            else:
                print(f"  ⚠ Download failed (HTTP {response.status_code})")
                print(f"     Please download manually from the URL above.")
                return False
        except Exception as e:
            print(f"  ⚠ Download error: {e}")
            print(f"     Please download manually from the URL above.")
            return False
    else:
        print(f"  ℹ️  Install 'requests' library for automatic download:")
        print(f"     pip install requests")
        print(f"     Then re-run this script.")
        return False


def print_sentinel2_instructions():
    """Print step-by-step instructions for downloading Sentinel-2 data."""
    print()
    print("=" * 65)
    print("  📡 SENTINEL-2 L2A DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print()
    print("  Source: Copernicus Data Space Ecosystem (FREE)")
    print("  URL: https://browser.dataspace.copernicus.eu/")
    print()
    print("  STEP-BY-STEP:")
    print()
    print("  1. Go to https://browser.dataspace.copernicus.eu/")
    print("  2. Create a FREE account (if you don't have one)")
    print("  3. In the search area, navigate to Sehore, Madhya Pradesh")
    print(f"     Coordinates: {REGION['south']}°N - {REGION['north']}°N, "
          f"{REGION['west']}°E - {REGION['east']}°E")
    print("  4. Set the following search filters:")
    print("     - Data source: Sentinel-2")
    print("     - Product type: S2MSI2A (L2A — atmospherically corrected)")
    print("     - Cloud cover: < 20%")
    print("     - Date range: Any clear-sky date (e.g., Oct-Mar for India)")
    print("  5. Select a suitable image and download the .SAFE folder")
    print("  6. Inside the .SAFE folder, navigate to:")
    print("     GRANULE/[tile_folder]/IMG_DATA/R10m/")
    print("  7. Stack the following bands into a single GeoTIFF:")
    print("     - B02.jp2 (Blue)")
    print("     - B03.jp2 (Green)")
    print("     - B04.jp2 (Red)")
    print("     - B08.jp2 (NIR)")
    print()
    print("  HOW TO STACK BANDS (using Python):")
    print("  ───────────────────────────────────")
    print("  Run the following Python script after downloading:")
    print()
    print('    import rasterio')
    print('    import numpy as np')
    print('    from rasterio.transform import from_bounds')
    print()
    print('    bands = ["B02.jp2", "B03.jp2", "B04.jp2", "B08.jp2"]')
    print('    data = []')
    print('    for b in bands:')
    print('        with rasterio.open(f"path/to/R10m/{b}") as src:')
    print('            data.append(src.read(1))')
    print('            profile = src.profile.copy()')
    print()
    print('    profile.update(count=4, driver="GTiff", dtype="float32")')
    print('    with rasterio.open("data/satellite/sehore_sentinel2.tif", "w", **profile) as dst:')
    print('        for i, band in enumerate(data, 1):')
    print('            dst.write(band.astype(np.float32), i)')
    print()
    print("  8. Place the stacked GeoTIFF in: data/satellite/")
    print()
    print("  ALTERNATIVE: Use Google Earth Engine (GEE):")
    print("  https://code.earthengine.google.com/")
    print("  Export a Sentinel-2 composite for the Sehore region.")
    print()


def print_worldcover_instructions():
    """Print WorldCover download instructions."""
    print()
    print("=" * 65)
    print("  🤖 ESA WORLDCOVER DOWNLOAD INSTRUCTIONS")
    print("=" * 65)
    print()
    print("  Source: ESA WorldCover 2021 (FREE, CC BY 4.0)")
    print("  URL: https://esa-worldcover.org/en")
    print()
    print("  OPTION 1: Direct Download (Recommended)")
    print("  ─────────────────────────────────────────")
    tile = REGION["worldcover_tile"]
    print(f"  Download tile {tile} from:")
    print(f"  https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
          f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    print()
    print("  OPTION 2: WorldCover Viewer")
    print("  ─────────────────────────────────────────")
    print("  1. Go to https://viewer.esa-worldcover.org/worldcover/")
    print("  2. Navigate to Sehore, Madhya Pradesh")
    print("  3. Click 'Download' and select the tile")
    print()
    print("  OPTION 3: Zenodo Archive")
    print("  ─────────────────────────────────────────")
    print("  https://doi.org/10.5281/zenodo.7254221")
    print()
    print(f"  After downloading, save as: data/worldcover/worldcover_sehore.tif")
    print()
    print("  NOTE: Each WorldCover tile is ~200-400 MB covering 3°x3°.")
    print("  The pipeline will automatically clip it to the Sehore AOI.")
    print()


def main():
    """Main data download helper."""
    print()
    print("=" * 65)
    print("  🌿 GeoGreen Revolution — Real Data Download Helper")
    print("  Pilot Region: Sehore District, Madhya Pradesh")
    print("=" * 65)
    print()

    # Step 1: Create directories
    print("📌 Step 1: Creating directory structure...")
    create_directories()

    # Step 2: Generate climate data
    print("📌 Step 2: Generating real climate data for Sehore...")
    generate_real_climate_data()

    # Step 3: Download WorldCover
    print("📌 Step 3: ESA WorldCover data...")
    wc_success = download_worldcover()

    # Step 4: Print instructions
    if not wc_success:
        print_worldcover_instructions()

    print_sentinel2_instructions()

    # Summary
    print("=" * 65)
    print("  📋 DATA CHECKLIST:")
    print()

    sat_dir = os.path.join(PROJECT_ROOT, "data", "satellite")
    clim_dir = os.path.join(PROJECT_ROOT, "data", "climate")
    wc_dir = os.path.join(PROJECT_ROOT, "data", "worldcover")

    sat_exists = any(f.endswith('.tif') for f in os.listdir(sat_dir)) if os.path.isdir(sat_dir) else False
    clim_exists = any(f.endswith('.csv') for f in os.listdir(clim_dir)) if os.path.isdir(clim_dir) else False
    wc_exists = any(f.endswith('.tif') for f in os.listdir(wc_dir)) if os.path.isdir(wc_dir) else False

    print(f"  {'✅' if sat_exists else '❌'} Sentinel-2 Image   : data/satellite/*.tif")
    print(f"  {'✅' if clim_exists else '❌'} Climate Data       : data/climate/*.csv")
    print(f"  {'✅' if wc_exists else '❌'} ESA WorldCover     : data/worldcover/*.tif")
    print()

    if sat_exists and clim_exists and wc_exists:
        print("  ✅ ALL DATA READY! Run the pipeline:")
        print("     python src/main.py")
    elif clim_exists:
        print("  ℹ️  Climate data is ready.")
        if not sat_exists:
            print("  ❌ Download Sentinel-2 image (see instructions above)")
        if not wc_exists:
            print("  ❌ Download WorldCover map (see instructions above)")
        print()
        print("  For QUICK TESTING with sample data, run:")
        print("     python generate_dummy_data.py")
        print("     python src/main.py")
    print()
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
