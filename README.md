# 🌿 GeoGreen Revolution — AI-Powered Geospatial Decision Support System

## 📌 Project Overview

India is facing declining green cover, underutilized rural land, and increasing urban heat. **GeoGreen Revolution** is an AI-powered geospatial system that uses **real satellite imagery** and a **pretrained deep learning model** to:

1. **Detect** barren and underutilized land using Sentinel-2 satellite imagery
2. **Classify** land cover using ESA WorldCover (pretrained DL model) — Tree, Shrub, Crop, Built-up, Bare, Water
3. **Analyze** vegetation health using NDVI and NDWI spectral indices
4. **Evaluate** greening feasibility using climate data (rainfall, temperature, soil)
5. **Recommend** actionable greening interventions for local authorities

**Pilot Region:** Sehore District, Madhya Pradesh, India (including Jamonia Dam region)

> **AI/ML Model Used:** ESA WorldCover 2021 — a pretrained deep learning model (U-Net + Random Forest ensemble) trained on Sentinel-1/2 satellite data. We perform **inference only** — no model training.

---

## 🧠 System Architecture - Flowchart
```
Sentinel-2 L2A Image (Real)          ESA WorldCover Map (Pretrained DL Model)
       │                                        │
       ▼                                        ▼
Band Extraction (B02-B08)            Load & Reclassify to Project Classes
       │                                        │
       ▼                                        ▼
Normalization (0–1 range)            7-Class ML Land-Cover Map
       │                                        │
       ▼                                        ▼
NDVI + NDWI Computation ──────┐     ML Classification
       │                      │            │
       ▼                      ▼            ▼
Rule-Based Classification   Fusion: ML Primary + Index Refinement
       │                                   │
       ▼                                   ▼
       └────────── Climate Data Fusion (Rainfall, Temp, Soil)
                              │
                              ▼
                    Zone-Based Spatial Analysis
                              │
                              ▼
                    Greening Recommendations
                              │
                              ▼
                    Maps + Tables + Reports
```

---

## 🤖 AI / ML Component Explained

### What AI Model Is Used?

**ESA WorldCover 2021 (v200)** — a global 10m land-cover classification map.

| Property | Details |
|----------|---------|
| **Model** | U-Net (Deep Learning) + Random Forest ensemble |
| **Training Data** | Sentinel-1 SAR + Sentinel-2 MSI imagery, millions of labeled samples |
| **Resolution** | 10 meters per pixel |
| **Accuracy** | ~75.6% overall (higher in agricultural/forested regions) |
| **Classes** | 11 land-cover types |
| **Coverage** | Global, 2021 epoch |
| **License** | CC BY 4.0 (free for all use) |
| **Source** | [esa-worldcover.org](https://esa-worldcover.org/) |

### How Is ML Inference Performed?

1. **Load**: The pretrained model output (WorldCover GeoTIFF) is loaded
2. **Reclassify**: ESA's 11 classes are mapped to 7 project classes
3. **Refine**: NDVI and NDWI indices refine the ML classification:
   - Confirm water bodies using NDWI > 0.3
   - Identify degraded cropland using very low NDVI
   - Upgrade dense vegetation based on NDVI > 0.6
4. **Fuse**: The refined ML output becomes the primary classification

### What Is Rule-Based?

- NDVI-threshold classification (fallback/comparison)
- Climate-based feasibility assessment (decision tree)
- Recommendation engine (if-then rules combining land class + rainfall + temperature)

> The system clearly separates **AI/ML inference** from **rule-based logic** throughout the code and reports.

---

## 📂 Folder Structure

```
GeoGreen Revolution/
│
├── README.md                     ← This file (User Manual)
├── requirements.txt              ← Python dependencies
├── generate_dummy_data.py        ← Creates sample data for TESTING
├── download_data.py              ← Helps download REAL data
│
├── data/                         ← All input data
│   ├── satellite/                ← Sentinel-2 GeoTIFF images
│   │   └── sehore_sentinel2.tif
│   ├── worldcover/               ← ESA WorldCover classification
│   │   └── worldcover_sehore.tif
│   └── climate/                  ← Climate data CSV
│       └── sehore_climate.csv
│
├── output/                       ← Analysis results (auto-created)
│   ├── ndvi_map.png              ← Vegetation health map
│   ├── ndwi_map.png              ← Water index map
│   ├── classification_map_rulebased.png  ← NDVI-based classification
│   ├── ml_landcover_map.png      ← AI land-cover classification ⭐
│   ├── fused_landcover_map.png   ← ML + spectral index fusion ⭐
│   ├── recommendation_map.png    ← Greening priority map ⭐
│   ├── rgb_composite.png         ← True-color satellite image
│   ├── recommendations.csv       ← Zone-wise recommendations
│   └── summary_statistics.txt    ← Comprehensive analysis report
│
└── src/                          ← Source code modules
    ├── config.py                 ← Settings, thresholds, model config
    ├── preprocessing.py          ← Data loading & normalization
    ├── indices.py                ← NDVI, NDWI computation
    ├── ml_inference.py           ← AI/ML model integration ⭐
    ├── analysis.py               ← Classification & recommendations
    ├── utils.py                  ← Visualization & report generation
    └── main.py                   ← Main pipeline entry point
```

---

## ⚙️ Installation

### Prerequisites
- **Python 3.8+** ([python.org](https://www.python.org/downloads/))
- **pip** (comes with Python)

### Step 1: Open Terminal
```bash
cd "F:\GeoGreen Revolution"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> All libraries are **free and open-source**. No paid APIs or services needed.

---

## 📥 Data Sources (All FREE)

### Option A: Quick Testing with Sample Data
```bash
python generate_dummy_data.py
```
This creates synthetic Sentinel-2, WorldCover, and climate data for **immediate testing**.

### Option B: Download Real Data (For Final Demo)

| Data | Source | Format | Link |
|------|--------|--------|------|
| Sentinel-2 L2A | Copernicus CDSE | GeoTIFF/JP2 | [browser.dataspace.copernicus.eu](https://browser.dataspace.copernicus.eu/) |
| ESA WorldCover | ESA WorldCover | GeoTIFF | [esa-worldcover.org](https://esa-worldcover.org/) |
| Climate Data | IMD / WorldClim | CSV (pre-generated) | Included in `download_data.py` |

Run the download helper for detailed instructions:
```bash
python download_data.py
```

#### Sentinel-2 Download Steps:
1. Register at [Copernicus CDSE](https://browser.dataspace.copernicus.eu/) (free)
2. Navigate to Sehore, MP (23.1°N, 77.1°E)
3. Filter: **S2MSI2A** (L2A), Cloud < 20%
4. Download and stack bands B02, B03, B04, B08 into a GeoTIFF
5. Place in `data/satellite/`

#### WorldCover Download:
1. Download tile N23E076 from [ESA WorldCover](https://esa-worldcover.org/)
2. Or direct URL: `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N23E076_Map.tif`
3. Place in `data/worldcover/`

---

## 🚀 How to Run

### Quick Demo (Sample Data — Recommended First)
```bash
# Step 1: Generate sample data (satellite + WorldCover + climate)
python generate_dummy_data.py

# Step 2: Run the AI pipeline
python src/main.py
```

### With Real Data (Full ML Mode)
```bash
# Step 1: Download real data (follow instructions)
python download_data.py

# Step 2: Place files in data/ directories

# Step 3: Run the pipeline
python src/main.py

# Or specify paths explicitly:
python src/main.py --satellite data/satellite/sehore.tif --worldcover data/worldcover/worldcover_sehore.tif --climate data/climate/sehore_climate.csv
```

### Pipeline Modes
- **ML Mode** (when WorldCover file is detected): Full AI pipeline with pretrained model
- **Fallback Mode** (no WorldCover): Rule-based NDVI classification only

---

## 📜 What Each Script Does

| Script | Type | Purpose |
|--------|------|---------|
| `generate_dummy_data.py` | Utility | Creates synthetic Sehore data for testing |
| `download_data.py` | Utility | Downloads real satellite + WorldCover data |
| `src/config.py` | Config | All settings: paths, thresholds, model config, recommendation rules |
| `src/preprocessing.py` | Core | Loads GeoTIFFs, normalizes bands, clips to AOI |
| `src/indices.py` | Core | Computes NDVI (vegetation) and NDWI (water) indices |
| `src/ml_inference.py` | **AI/ML** ⭐ | Loads ESA WorldCover pretrained model, reclassifies, refines with indices |
| `src/analysis.py` | Core | Zone statistics, climate fusion, recommendation engine |
| `src/utils.py` | Output | Map generation, report saving |
| `src/main.py` | Pipeline | Orchestrates the entire analysis pipeline |

---

## 📊 Expected Outputs

After running `python src/main.py`:

| Output File | Type | Description |
|-------------|------|-------------|
| `ndvi_map.png` | Map | Vegetation health (Red=Barren, Green=Healthy) |
| `ndwi_map.png` | Map | Water detection index (Blue=Water) |
| `classification_map_rulebased.png` | Map | NDVI-threshold classification (rule-based) |
| `ml_landcover_map.png` ⭐ | Map | AI land-cover classification (ESA WorldCover) |
| `fused_landcover_map.png` ⭐ | Map | Final fused classification (ML + indices) |
| `recommendation_map.png` ⭐ | Map | Greening intervention priority heatmap |
| `rgb_composite.png` | Map | True-color satellite image |
| `recommendations.csv` | Table | Zone-wise recommendations with priority |
| `summary_statistics.txt` | Report | Full analysis report with model info |

---

## 🔬 Land-Cover Classes (ML-Based)

| Class ID | Label | Color | Description |
|----------|-------|-------|-------------|
| 1 | Tree Cover | Dark Green | Forest and tree canopy |
| 2 | Shrubland | Yellow-Green | Low woody vegetation |
| 3 | Grassland | Light Green | Herbaceous vegetation, wetlands |
| 4 | Cropland | Golden/Amber | Agricultural fields |
| 5 | Built-up | Dark Red | Urban areas, settlements |
| 6 | Bare / Sparse | Sandy Brown | Barren or sparsely vegetated land |
| 7 | Water | Blue | Rivers, dams, reservoirs |

---

## 📋 Recommendation Logic

Recommendations combine **ML land-cover class + climate data**:

| Land Cover | Climate Condition | Recommendation | Priority |
|-----------|-------------------|----------------|----------|
| Bare/Sparse | Rainfall > 800mm, Temp < 40°C | Afforestation (Neem, Banyan, Peepal) | **High** |
| Bare/Sparse | Rainfall 500–800mm | Agroforestry / Silvopasture | **High** |
| Bare/Sparse | Rainfall 300–500mm | Drought-Resistant Plantation | Medium |
| Built-up | Any | Urban Greening (Rooftop Gardens, Parks) | Medium |
| Cropland | Low NDVI (< 0.15) | Soil Restoration + Cover Cropping | **High** |
| Cropland | Rainfall < 500mm | Water Harvesting + Drought Crops | **High** |
| Shrubland | Rainfall > 800mm | Afforestation (Upgrade to Trees) | **High** |
| Tree Cover | Any | Conservation & Biodiversity Monitoring | Low |
| Water | Any | Wetland Conservation | Low |

---

## ❌ Common Errors & How to Fix

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: 'rasterio'` | Library not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: data/satellite/...` | No input data | Run `python generate_dummy_data.py` |
| `FileNotFoundError: data/worldcover/...` | No WorldCover file | Run `python download_data.py` or use sample data |
| `ValueError: operands broadcast` | Band dimension mismatch | Ensure all bands are same resolution (10m) |
| `MemoryError` | Image too large | Crop to smaller area using QGIS (Free) |

---

## 🎤 How to Demonstrate to Evaluators

### Recommended Demo Flow (10-15 minutes):

**1. Show the AI Model (1 min)**
> "We use ESA WorldCover, a pretrained deep learning model trained by the European Space Agency on Sentinel satellite data. It classifies every 10m pixel into land-cover types."

**2. Generate and run with sample data (2 min)**
```bash
python generate_dummy_data.py
python src/main.py
```

**3. Show the outputs (5 min)**
- **NDVI Map** → "This shows vegetation health — red is barren, green is healthy."
- **ML Land-Cover Map** → "This is the AI model output — it distinguishes crops from forest from urban."
- **Fused Map** → "We refine the AI output using spectral indices for higher accuracy."
- **Recommendation Map** → "Red zones need immediate greening intervention."
- **CSV Report** → "Zone-wise actionable recommendations for authorities."

**4. Explain the pipeline (2 min)**
> "Real satellite image → preprocessing → AI inference → spectral index refinement → climate fusion → greening recommendations."

**5. Show real data (if available) (2 min)**
> "We also tested with real Sentinel-2 imagery of Sehore district, Madhya Pradesh."

### Key Points to Highlight:
- ✅ Uses **pretrained deep learning model** (ESA WorldCover)
- ✅ Performs **inference only** — no model training needed
- ✅ Combines **AI classification + spectral indices + climate data**
- ✅ All tools and data are **100% free**
- ✅ Runs on **local CPU** — no GPU or cloud needed
- ✅ Modular, clean, documented code
- ✅ Produces **actionable outputs** for local authorities

---

## 🔬 Academic Clarity

### What Is AI / ML in This Project?

| Component | Type | Method |
|-----------|------|--------|
| Land-cover classification | **AI/ML** ⭐ | ESA WorldCover pretrained DL model (U-Net + RF) |
| Spectral index refinement | **AI/ML + Rule-based** | ML primary + NDVI/NDWI confirmation rules |
| NDVI/NDWI computation | Rule-based | Formula-based spectral indices |
| Climate feasibility | Rule-based | Decision tree with thresholds |
| Recommendation engine | Rule-based | If-then rules combining land class + climate |
| Zone-based analysis | Rule-based | Grid-based spatial aggregation |

### Limitations

1. WorldCover accuracy varies by region (~75% global, may differ locally)
2. Climate data is aggregated at zone level (not pixel-level raster)
3. Zone analysis uses fixed grid, not administrative boundaries
4. Single-epoch analysis (no time-series change detection)
5. Sample data is synthetic — real data required for production use

### Future Scope

1. Fine-tune with local ground-truth data for higher accuracy
2. Integrate pixel-level WorldClim climate rasters
3. Add multi-temporal change detection (seasonal/annual)
4. Deploy as a web-based dashboard for local authorities
5. Add soil moisture indices from Sentinel-1 SAR data
6. Include district boundary shapefiles for administrative analysis

---

## 📚 References & Data Sources

1. ESA WorldCover 2021: Zanaga, D., et al. (2022). *ESA WorldCover 10m 2021 v200*. Zenodo. https://doi.org/10.5281/zenodo.7254221
2. ESA Sentinel-2 Mission: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
3. Copernicus Data Space: https://browser.dataspace.copernicus.eu/
4. WorldClim 2.1 Climate Data: https://www.worldclim.org/
5. NDVI Reference: Rouse et al. (1974) — *Monitoring vegetation systems in the Great Plains*
6. NDWI Reference: McFeeters (1996) — *The use of NDWI for delineating open water features*
7. Rasterio Documentation: https://rasterio.readthedocs.io/

---

## 📄 License

This project is for **academic/educational purposes**. All data sources and tools used are freely available under open licenses.

**ESA WorldCover** is licensed under CC BY 4.0.

---

*Built with ❤️ for a Greener India.*
