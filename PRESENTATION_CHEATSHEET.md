# GeoGreen Revolution — Presentation Cheat Sheet

Use this cheat sheet during your presentation to confidently answer questions about model accuracy, data sources, and the rationale behind your AI pipeline.

## 1. AI Land-Cover Classification Model
When asked: *"How does the AI know what is on the ground?"*

*   **Model Name:** ESA WorldCover 2021 (v100)
*   **Architecture:** Ensemble of **U-Net** (a Convolutional Neural Network perfect for image segmentation) and **Random Forest** (a decision tree algorithm). 
*   **Training Data:** Trained by the European Space Agency using multi-modal data (Sentinel-1 SAR radar + Sentinel-2 MSI optical imagery).
*   **Accuracy:** **75.6% overall global accuracy**. 
*   **Resolution:** 10 meters per pixel (classifying every single 10x10m square on the ground).
*   **Citation for your Presentation:** *Zanaga, D., et al. (2022). ESA WorldCover 10 m 2021 v100.*

## 2. Satellite Data (The Input)
When asked: *"Where did you get the images and how recent are they?"*

*   **Source:** Copernicus Sentinel-2 L2A (Level-2A means it's already atmospherically corrected; clouds and atmospheric distortions are filtered out).
*   **Resolution:** 10m/pixel (the highest freely available global optical resolution).
*   **Bands Used:** 
    *   `B02` (Blue)
    *   `B03` (Green)
    *   `B04` (Red)
    *   `B08` (Near-Infrared / NIR) — * Crucial for detecting vegetation health, because healthy leaves reflect NIR light strongly but absorb Red light.*

## 3. Spectral Indices (The "Math" Layer)
When asked: *"Why not just use the AI? Why calculate NDVI and NDWI?"*

The AI classifies the land cover, but spectral indices tell you the **health and exact physical state** of that land. Because they are mathematical formulas based on light physics, their "accuracy" is functionally **100%** (they are exact measurements of the pixel's light reflectance).

*   **NDVI** (Normalized Difference Vegetation Index): `(NIR - Red) / (NIR + Red)`
    *   **Purpose:** Measures vegetation density and health.
    *   **Scale:** `-1.0 to 1.0`. 
    *   **What it means:** `< 0` = Water/Built-up. `0 to 0.2` = Bare soil. `> 0.5` = Dense, healthy forest.
*   **NDWI** (Normalized Difference Water Index): `(Green - NIR) / (Green + NIR)`
    *   **Purpose:** Exclusively isolates water bodies (lakes, dams, rivers).
    *   **Scale:** `-1.0 to 1.0`. `> 0` generally indicates standing water. 

## 4. The Recommendation Logic
When asked: *"How does the dashboard decide which scheme to suggest?"*

The recommendation engine is a **deterministic Rule-Based Decision Tree** that fuses 3 things:
1.  **AI classification** (e.g., is it Grassland or Cropland?)
2.  **NDVI Health deficit** (how degraded is the land?)
3.  **Climate context** (e.g., Black soil? >900mm rain?)

**Recommendation Mapping Accuracies (Policy Links):**
*   **Bare/Sparse Land:** Automatically routed to the **CAMPA Fund / Green India Mission** for high-density afforestation.
*   **Water Bodies (Desilting):** Automatically routed to the **Amrit Sarovar Mission** to improve groundwater recharge.
*   **Cropland / Rural:** Routed to **PMKSY** (irrigation) or **MGNREGS** (employment generation).

## 5. Overall Impact Metrics (Sehore Pilot)
*   **Total Area Analysed:** 317 km²
*   **Total Zones Identified:** 27,906 distinct eco-zones.
*   **High-Priority Intervention Zones:** 1,001 zones (the exact locations your dashboard flags for immediate action).
*   **Capacity:** Over **10 Million trees** could plantable if all high-priority bare/grassland zones are afforested.
*   **Carbon ROI:** Offsetting **>210,000 tonnes of CO₂ annually** if implemented.

---
> [!TIP]
> **Workflow for your Supervisor Demo:**
> 1. Open `run_presentation.ipynb`
> 2. Click **Run All**
> 3. Scroll to the newly added **Step 7 cell**, which automatically launches the Streamlit Interactive Dashboard and opens your browser.
