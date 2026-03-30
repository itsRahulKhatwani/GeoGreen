"""
GeoGreen Revolution — Configuration File
=========================================
Central place for all paths, thresholds, and settings.
Edit values here to customize the analysis for your region.

UPGRADED: Now includes ESA WorldCover ML model configuration,
NDWI thresholds, Sehore district pilot region settings,
and expanded land-cover classes.
"""

import os

# ──────────────────────────────────────────────────────────────
# PROJECT PATHS
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SATELLITE_DIR = os.path.join(DATA_DIR, "satellite")
CLIMATE_DIR = os.path.join(DATA_DIR, "climate")
WORLDCOVER_DIR = os.path.join(DATA_DIR, "worldcover")

# Output directory (auto-created if missing)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Default file names (auto-detected if only one file exists in the folder)
SATELLITE_FILE = None   # e.g., "sehore_sentinel2.tif", or None for auto-detect
CLIMATE_FILE = None     # e.g., "sehore_climate.csv", or None for auto-detect
WORLDCOVER_FILE = None  # e.g., "worldcover_sehore.tif", or None for auto-detect

# ──────────────────────────────────────────────────────────────
# PILOT REGION — Sehore District, Madhya Pradesh, India
# Includes Sehore town and nearby Jamonia Dam region
# ──────────────────────────────────────────────────────────────
REGION_NAME = "Sehore District, Madhya Pradesh, India"
REGION_BBOX = {
    "west":  76.90,
    "south": 23.05,
    "east":  77.20,
    "north": 23.25,
}
# EPSG:4326 (WGS84) — standard for Sentinel-2 and WorldCover
REGION_CRS = "EPSG:4326"

# ──────────────────────────────────────────────────────────────
# SENTINEL-2 BAND CONFIGURATION
# For multi-band GeoTIFF stacks (bands stored in order below)
# ──────────────────────────────────────────────────────────────
BAND_ORDER = {
    "blue":  1,   # Band 2 (B02) — 490 nm  — 10m
    "green": 2,   # Band 3 (B03) — 560 nm  — 10m
    "red":   3,   # Band 4 (B04) — 665 nm  — 10m
    "nir":   4,   # Band 8 (B08) — 842 nm  — 10m (Near-Infrared)
    "swir":  5,   # Band 11 (B11) — 1610 nm — 20m (Short-Wave Infrared, optional)
}

# ──────────────────────────────────────────────────────────────
# ESA WORLDCOVER — PRETRAINED ML MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────
# ESA WorldCover 2021 is a global 10m land-cover map produced by
# a deep learning model (U-Net + Random Forest ensemble) trained
# on millions of Sentinel-1/2 labeled samples.
# Overall accuracy: ~75% globally.
# Source: https://esa-worldcover.org/
#
# Original 11 classes → Remapped to 7 project classes
# ──────────────────────────────────────────────────────────────
WORLDCOVER_CLASS_MAP = {
    # ESA_value: (project_class_id, project_label)
    10:  (1, "Tree Cover"),          # Tree cover
    20:  (2, "Shrubland"),           # Shrubland
    30:  (3, "Grassland"),           # Grassland
    40:  (4, "Cropland"),            # Cropland
    50:  (5, "Built-up"),            # Built-up
    60:  (6, "Bare / Sparse"),       # Bare / sparse vegetation
    70:  (6, "Bare / Sparse"),       # Snow and Ice → treat as bare (N/A in Sehore)
    80:  (7, "Water"),               # Permanent water bodies
    90:  (3, "Grassland"),           # Herbaceous wetland → classify as grassland
    95:  (3, "Grassland"),           # Mangroves → classify with grassland
    100: (3, "Grassland"),           # Moss and lichen → grassland
}

# ML-based land-cover class labels (from WorldCover)
ML_LAND_CLASS_LABELS = {
    0: "No Data",
    1: "Tree Cover",
    2: "Shrubland",
    3: "Grassland",
    4: "Cropland",
    5: "Built-up",
    6: "Bare / Sparse",
    7: "Water",
}

# Colors for ML land-cover map (RGBA)
ML_LAND_CLASS_COLORS = {
    0: (0.50, 0.50, 0.50, 1.0),    # No Data — Grey
    1: (0.00, 0.39, 0.00, 1.0),    # Tree Cover — Dark Green
    2: (0.55, 0.76, 0.29, 1.0),    # Shrubland — Yellow-Green
    3: (0.80, 0.93, 0.55, 1.0),    # Grassland — Light Yellow-Green
    4: (1.00, 0.78, 0.24, 1.0),    # Cropland — Golden / Amber
    5: (0.77, 0.12, 0.15, 1.0),    # Built-up — Dark Red
    6: (0.87, 0.72, 0.47, 1.0),    # Bare/Sparse — Sandy/Brown
    7: (0.18, 0.55, 0.82, 1.0),    # Water — Blue
}

# ──────────────────────────────────────────────────────────────
# NDVI CLASSIFICATION THRESHOLDS (Rule-Based — Fallback/Compare)
# Adjust these based on your region's characteristics
# ──────────────────────────────────────────────────────────────
NDVI_THRESHOLDS = {
    "water":            (-1.0, 0.0),    # NDVI < 0   → Water bodies
    "built_up":         (0.0,  0.12),   # 0.0–0.12   → Built-up / Urban
    "barren":           (0.12, 0.25),   # 0.12–0.25  → Barren / Sparse land
    "moderate_veg":     (0.25, 0.45),   # 0.25–0.45  → Moderate vegetation
    "dense_veg":        (0.45, 1.0),    # > 0.45     → Dense vegetation / Forest
}

# Rule-based class labels (kept for comparison with ML output)
RULEBASED_LAND_CLASS_LABELS = {
    0: "Water",
    1: "Built-up / Urban",
    2: "Barren / Sparse",
    3: "Moderate Vegetation",
    4: "Dense Vegetation",
}

# Colors for rule-based classification map (RGBA)
RULEBASED_LAND_CLASS_COLORS = {
    0: (0.18, 0.55, 0.82, 1.0),   # Water — Blue
    1: (0.75, 0.75, 0.75, 1.0),   # Built-up — Grey
    2: (0.87, 0.72, 0.47, 1.0),   # Barren — Sandy/Brown
    3: (0.56, 0.78, 0.34, 1.0),   # Moderate Veg — Light Green
    4: (0.13, 0.55, 0.13, 1.0),   # Dense Veg — Dark Green
}

# Legacy aliases for backward compatibility
LAND_CLASS_LABELS = RULEBASED_LAND_CLASS_LABELS
LAND_CLASS_COLORS = RULEBASED_LAND_CLASS_COLORS

# ──────────────────────────────────────────────────────────────
# NDWI THRESHOLD (Water Detection)
# NDWI = (Green - NIR) / (Green + NIR)
# ──────────────────────────────────────────────────────────────
NDWI_WATER_THRESHOLD = 0.3   # Pixels with NDWI > 0.3 are water

# ──────────────────────────────────────────────────────────────
# CLIMATE-BASED FEASIBILITY RULES (EXPANDED for ML classes)
# Rainfall in mm/year, Temperature in °C (avg summer)
# Uses ML land-cover class IDs (1–7)
# ──────────────────────────────────────────────────────────────
ML_RECOMMENDATION_RULES = [
    # (ml_class, rainfall_min, rainfall_max, temp_min, temp_max, recommendation, priority)

    # Tree Cover (class 1) — Conservation
    (1, 0,    9999, 0,  50, "Conservation: Maintain existing tree cover. No major intervention needed.", "Low"),

    # Shrubland (class 2) — Enhancement
    (2, 800,  9999, 0,  40, "Ecosystem Improvement: Upgrade to Tree Cover by planting high-rainfall native species (Teak, Sal) to improve biodiversity.", "High"),
    (2, 500,  800,  0,  42, "Agroforestry: Enhance shrub-land with structured tree-planting on boundaries to improve soil health.", "Medium"),
    (2, 0,    500,  0,  50, "Ecosystem Improvement: Enrich sparse shrubland with drought-resistant plants to prevent desertification.", "Low"),

    # Grassland (class 3) — Restoration
    (3, 800,  9999, 0,  40, "Ecological Restoration: Transition grassland to forest using native species.", "High"),
    (3, 500,  800,  0,  42, "Biodiversity Enhancement: Restore grassland boundaries to support local fauna.", "Medium"),
    (3, 0,    500,  0,  50, "Soil Preservation: Grass seeding and micro-irrigation to maintain top-soil.", "Low"),

    # Cropland (class 4) — Crop Rotation & Agroforestry
    (4, 800,  9999, 0,  40, "Agriculture: Intercross with native trees (Agroforestry) for dual-income and wind protection.", "Medium"),
    (4, 500,  800,  0,  42, "Agriculture: Adopt crop rotation using leguminous crops to naturally restore soil nitrogen.", "Medium"),
    (4, 0,    500,  0,  50, "Agriculture: Use drought-resistant crop varieties and integrate rainwater harvesting systems.", "High"),

    # Built-up (class 5) — Urban Greening & Pollution Reduction
    (5, 0,    9999, 0,  50, "Pollution Reduction: Implement vertical gardens, green rooftops, and plant native roadside trees (Neem, Peepal, Banyan) to absorb pollution and reduce the urban heat island effect.", "High"),

    # Bare / Sparse (class 6) — High priority greening
    (6, 800,  9999, 0,  40, "Ecosystem Improvement: Major afforestation drive with native species to restore bare land.", "High"),
    (6, 500,  800,  0,  42, "Land Utilization: Implement Agroforestry / Silvopasture to regenerate degraded land.", "High"),
    (6, 300,  500,  0,  45, "Ecosystem Improvement: Plant hardy drought-resistant species (Babool, Khejri) to bind soil.", "Medium"),
    (6, 0,    300,  0,  50, "Land Utilization: Micro-irrigation combined with grass seeding to prevent soil erosion.", "Low"),

    # Water (class 7) — Conservation
    (7, 0,    9999, 0,  50, "Water Body Utilization: Implement periodic desilting to increase groundwater recharge, and create riparian green buffer zones to prevent agricultural and urban runoff from polluting the water.", "High"),
]

# Legacy rule-based rules (kept for backward compatibility)
RECOMMENDATION_RULES = [
    # (land_class, rainfall_min, rainfall_max, temp_min, temp_max, recommendation, priority)
    (2, 800, 9999, 0, 40, "Afforestation (Native Species — Neem, Banyan, Peepal)", "High"),
    (2, 500, 800,  0, 42, "Agroforestry / Silvopasture", "High"),
    (2, 300, 500,  0, 45, "Drought-Resistant Plantation (Babool, Khejri)", "Medium"),
    (2, 0,   300,  0, 50, "Micro-Irrigation + Grass Seeding", "Low"),
    (1, 0,   9999, 0, 50, "Urban Greening (Rooftop Gardens, Road Plantations, Parks)", "Medium"),
    (3, 600, 9999, 0, 40, "Grassland Restoration & Biodiversity Enhancement", "Low"),
    (3, 0,   600,  0, 50, "Shrub Plantation (Low-Water Native Species)", "Low"),
]

# ──────────────────────────────────────────────────────────────
# OUTPUT SETTINGS
# ──────────────────────────────────────────────────────────────
OUTPUT_DPI = 150          # Resolution for saved PNG maps
FIGURE_SIZE = (14, 10)    # Figure size in inches (width, height)

# ──────────────────────────────────────────────────────────────
# MODEL METADATA (for academic documentation)
# ──────────────────────────────────────────────────────────────
MODEL_INFO = {
    "name": "ESA WorldCover 2021",
    "version": "v200",
    "resolution": "10m",
    "source": "https://esa-worldcover.org/",
    "training_data": "Sentinel-1 SAR + Sentinel-2 MSI, globally distributed training samples",
    "algorithm": "Deep Learning (U-Net) + Random Forest ensemble classifier",
    "overall_accuracy": "~75.6% (global), higher in agricultural and forested regions",
    "classes": 11,
    "coverage": "Global, 2021 epoch",
    "license": "CC BY 4.0",
    "citation": (
        "Zanaga, D., et al. (2022). ESA WorldCover 10m 2021 v200. "
        "Zenodo. https://doi.org/10.5281/zenodo.7254221"
    ),
}
