"""
GeoGreen Revolution — Analysis Module
=======================================
Land classification, feasibility assessment, and recommendation engine.

UPGRADED:
  - ML-based classification using ESA WorldCover (pretrained DL model)
  - Rule-based NDVI classification retained as fallback/comparison
  - Fused classification combining ML + spectral indices
  - Enhanced recommendation engine with 7-class ML land-cover support
  - Zone-wise analysis with climate data fusion
"""

import numpy as np
import pandas as pd

from config import (
    NDVI_THRESHOLDS, RULEBASED_LAND_CLASS_LABELS,
    ML_LAND_CLASS_LABELS, ML_RECOMMENDATION_RULES,
    RECOMMENDATION_RULES,
)


def classify_land_rulebased(ndvi):
    """
    [RULE-BASED] Classify each pixel into a land type based on NDVI thresholds.

    This is the ORIGINAL classification method — kept as a fallback
    and for comparison with the ML-based WorldCover classification.

    Classification Map:
        0 = Water
        1 = Built-up / Urban
        2 = Barren / Sparse
        3 = Moderate Vegetation
        4 = Dense Vegetation

    Parameters
    ----------
    ndvi : numpy.ndarray
        NDVI values (2D array).

    Returns
    -------
    numpy.ndarray
        Classification map (integer array, same shape as NDVI).
    """
    classification = np.full(ndvi.shape, fill_value=2, dtype=np.int8)  # Default: Barren

    # Apply thresholds in order
    thresholds = NDVI_THRESHOLDS

    # Water: NDVI < 0
    low, high = thresholds["water"]
    classification[ndvi < high] = 0

    # Built-up: 0.0 <= NDVI < 0.12
    low, high = thresholds["built_up"]
    classification[(ndvi >= low) & (ndvi < high)] = 1

    # Barren: 0.12 <= NDVI < 0.25
    low, high = thresholds["barren"]
    classification[(ndvi >= low) & (ndvi < high)] = 2

    # Moderate Vegetation: 0.25 <= NDVI < 0.45
    low, high = thresholds["moderate_veg"]
    classification[(ndvi >= low) & (ndvi < high)] = 3

    # Dense Vegetation: NDVI >= 0.45
    low, high = thresholds["dense_veg"]
    classification[ndvi >= low] = 4

    # Print distribution
    total = classification.size
    print(f"  ✅ [Rule-Based] Land classification complete")
    for class_id, label in RULEBASED_LAND_CLASS_LABELS.items():
        count = np.count_nonzero(classification == class_id)
        pct = (count / total) * 100
        print(f"     {label:25s}: {count:>8,} pixels ({pct:5.1f}%)")

    return classification


# Keep legacy name for backward compatibility
classify_land = classify_land_rulebased


def compute_zone_statistics(classification, ndvi, grid_size=50,
                            use_ml_classes=False, ndwi=None):
    """
    Divide the image into spatial zones (grid cells) and compute
    statistics for each zone.

    Parameters
    ----------
    classification : numpy.ndarray
        Land classification map (rule-based or ML-based).
    ndvi : numpy.ndarray
        NDVI values.
    grid_size : int
        Size of each grid cell in pixels.
    use_ml_classes : bool
        If True, use ML class labels (1–7). Else use rule-based (0–4).
    ndwi : numpy.ndarray, optional
        NDWI values for additional water statistics.

    Returns
    -------
    pandas.DataFrame
        Zone-level statistics.
    """
    h, w = classification.shape
    zones = []
    zone_id = 0

    class_labels = ML_LAND_CLASS_LABELS if use_ml_classes else RULEBASED_LAND_CLASS_LABELS

    for r in range(0, h, grid_size):
        for c in range(0, w, grid_size):
            r_end = min(r + grid_size, h)
            c_end = min(c + grid_size, w)

            patch_class = classification[r:r_end, c:c_end]
            patch_ndvi = ndvi[r:r_end, c:c_end]

            total_pixels = patch_class.size
            if total_pixels == 0:
                continue

            # Dominant class
            unique, counts = np.unique(patch_class, return_counts=True)
            dominant_class = unique[np.argmax(counts)]

            # Mean NDVI
            mean_ndvi = float(np.nanmean(patch_ndvi))

            zone_data = {
                "zone_id": zone_id,
                "row_start": r,
                "col_start": c,
                "mean_ndvi": mean_ndvi,
                "dominant_class": int(dominant_class),
                "dominant_class_label": class_labels.get(int(dominant_class), "Unknown"),
            }

            if use_ml_classes:
                # ML class percentages
                zone_data["tree_cover_pct"] = round(
                    np.count_nonzero(patch_class == 1) / total_pixels * 100, 1)
                zone_data["cropland_pct"] = round(
                    np.count_nonzero(patch_class == 4) / total_pixels * 100, 1)
                zone_data["built_up_pct"] = round(
                    np.count_nonzero(patch_class == 5) / total_pixels * 100, 1)
                zone_data["bare_pct"] = round(
                    np.count_nonzero(patch_class == 6) / total_pixels * 100, 1)
                zone_data["water_pct"] = round(
                    np.count_nonzero(patch_class == 7) / total_pixels * 100, 1)
                zone_data["vegetation_pct"] = round(
                    (np.count_nonzero(patch_class == 1) +
                     np.count_nonzero(patch_class == 2) +
                     np.count_nonzero(patch_class == 3) +
                     np.count_nonzero(patch_class == 4)) / total_pixels * 100, 1)
            else:
                # Rule-based class percentages
                zone_data["barren_pct"] = round(
                    np.count_nonzero(patch_class == 2) / total_pixels * 100, 1)
                zone_data["built_up_pct"] = round(
                    np.count_nonzero(patch_class == 1) / total_pixels * 100, 1)
                zone_data["vegetation_pct"] = round(
                    (np.count_nonzero(patch_class == 3) +
                     np.count_nonzero(patch_class == 4)) / total_pixels * 100, 1)

            # NDWI stats if available
            if ndwi is not None:
                patch_ndwi = ndwi[r:r_end, c:c_end]
                zone_data["mean_ndwi"] = round(float(np.nanmean(patch_ndwi)), 4)

            zones.append(zone_data)
            zone_id += 1

    df = pd.DataFrame(zones)
    print(f"  ✅ Zone statistics computed: {len(df)} zones (grid size: {grid_size}px)")
    return df


def assign_climate_to_zones(zone_df, climate_df):
    """
    Merge climate data into zone statistics.

    If climate_df has fewer rows than zones, the climate values are
    applied uniformly (assuming a single-region pilot study).

    Parameters
    ----------
    zone_df : pandas.DataFrame
        Zone statistics from compute_zone_statistics().
    climate_df : pandas.DataFrame
        Climate data loaded from CSV.

    Returns
    -------
    pandas.DataFrame
        Zone data enriched with climate columns.
    """
    if len(climate_df) == 1:
        # Single-region: apply same climate to all zones
        for col in ["rainfall_mm", "temperature_c", "soil_type"]:
            if col in climate_df.columns:
                zone_df[col] = climate_df[col].iloc[0]
        print(f"  ✅ Climate data applied uniformly (single region)")
    elif len(climate_df) >= len(zone_df):
        # Multi-zone: merge by zone_id
        zone_df = zone_df.merge(
            climate_df[["zone_id", "rainfall_mm", "temperature_c", "soil_type"]],
            on="zone_id",
            how="left"
        )
        print(f"  ✅ Climate data merged by zone_id")
    else:
        # Fewer climate zones than image zones: cycle through
        n_climate = len(climate_df)
        for col in ["rainfall_mm", "temperature_c", "soil_type"]:
            if col in climate_df.columns:
                values = climate_df[col].values
                zone_df[col] = [values[i % n_climate] for i in range(len(zone_df))]
        print(f"  ✅ Climate data cycled across {len(zone_df)} zones ({n_climate} climate records)")

    return zone_df


def generate_recommendations_ml(zone_df):
    """
    [ML-BASED] Apply the recommendation engine using ML land-cover classes.

    Uses the 7-class ML land-cover output (from ESA WorldCover) combined
    with climate data to generate greening recommendations.

    This is the PRIMARY recommendation method in the upgraded system.

    Parameters
    ----------
    zone_df : pandas.DataFrame
        Zone data with ML classification and climate information.

    Returns
    -------
    pandas.DataFrame
        Zone data with 'recommendation' and 'priority' columns.
    """
    recommendations = []
    priorities = []

    for _, row in zone_df.iterrows():
        dom_class = row["dominant_class"]
        rainfall = row.get("rainfall_mm", 800)
        temperature = row.get("temperature_c", 35)
        mean_ndvi = row.get("mean_ndvi", 0.3)

        rec = "No Intervention Needed"
        pri = "None"

        # Apply ML-based rules
        for rule_class, rain_min, rain_max, temp_min, temp_max, rule_rec, rule_pri in ML_RECOMMENDATION_RULES:
            if (dom_class == rule_class and
                    rain_min <= rainfall < rain_max and
                    temp_min <= temperature < temp_max):
                rec = rule_rec
                pri = rule_pri
                break

        # Special case: Cropland with very low NDVI = degraded
        if dom_class == 4 and mean_ndvi < 0.15:
            rec = "Urgent: Degraded Cropland — Soil Restoration + Cover Cropping"
            pri = "High"

        recommendations.append(rec)
        priorities.append(pri)

    zone_df["recommendation"] = recommendations
    zone_df["priority"] = priorities

    # Summary
    high_priority = sum(1 for p in priorities if p == "High")
    medium_priority = sum(1 for p in priorities if p == "Medium")
    print(f"  ✅ [ML-Based] Recommendations generated for {len(zone_df)} zones")
    print(f"     High priority: {high_priority}, Medium: {medium_priority}")

    return zone_df


def generate_recommendations(zone_df):
    """
    [RULE-BASED] Apply the rule-based recommendation engine (legacy).

    For each zone, evaluate the decision rules from config.py
    to determine the best greening intervention.

    Parameters
    ----------
    zone_df : pandas.DataFrame
        Zone data with classification and climate information.

    Returns
    -------
    pandas.DataFrame
        Zone data with added 'recommendation' and 'priority' columns.
    """
    recommendations = []
    priorities = []

    for _, row in zone_df.iterrows():
        dom_class = row["dominant_class"]
        rainfall = row.get("rainfall_mm", 500)
        temperature = row.get("temperature_c", 35)

        rec = "No Intervention Needed"
        pri = "None"

        # Check if zone is already well-vegetated
        if dom_class == 4:
            rec = "Conservation & Monitoring (Already Green)"
            pri = "Low"
        elif dom_class == 0:
            rec = "Wetland Conservation / No Intervention"
            pri = "None"
        else:
            # Apply rules from config
            for rule_class, rain_min, rain_max, temp_min, temp_max, rule_rec, rule_pri in RECOMMENDATION_RULES:
                if (dom_class == rule_class and
                        rain_min <= rainfall < rain_max and
                        temp_min <= temperature < temp_max):
                    rec = rule_rec
                    pri = rule_pri
                    break

        recommendations.append(rec)
        priorities.append(pri)

    zone_df["recommendation"] = recommendations
    zone_df["priority"] = priorities

    # Summary
    high_priority = sum(1 for p in priorities if p == "High")
    medium_priority = sum(1 for p in priorities if p == "Medium")
    print(f"  ✅ [Rule-Based] Recommendations generated for {len(zone_df)} zones")
    print(f"     High priority: {high_priority}, Medium: {medium_priority}")

    return zone_df

def generate_rgb_recommendations(cv_stats, climate_data):
    """
    [CV-BASED] Generate tailored recommendations based on land cover and climate data,
    specifically answering ecosystem improvement, water bodies, urban reduction, and agriculture.
    """
    recs = []
    
    veg_pct = cv_stats.get("Vegetation", 0)
    water_pct = cv_stats.get("Water", 0)
    bare_pct = cv_stats.get("Bare/Sparse", 0)
    built_pct = cv_stats.get("Built-up", 0)
    
    rainfall = climate_data.get("annual_rainfall_mm", 0)
    temp = climate_data.get("mean_temp_c", 0)

    # 1. Ecosystem Improvement / Afforestation
    if bare_pct > 15:
        if rainfall > 800:
            recs.append({
                "priority": "High",
                "action": "Ecosystem Improvement: Afforestation",
                "reason": f"Significant bare land ({bare_pct:.1f}%). Improve the regional ecosystem by planting native species (like Teak or Sal) which thrive in good rainfall ({rainfall}mm). This will restore soil and local biodiversity.",
                "type": "ecosystem"
            })
        else:
            recs.append({
                "priority": "High",
                "action": "Ecosystem Improvement: Drought-Resistant Plantations",
                "reason": f"Bare land ({bare_pct:.1f}%) in low rainfall zone. To improve the ecosystem and prevent desertification, plant hardy species like Neem, Babool, or Khejri.",
                "type": "ecosystem"
            })

    # 2. Water Bodies Utilization
    if water_pct > 2:
        recs.append({
            "priority": "High",
            "action": "Water Body Utilization & Conservation",
            "reason": f"Water bodies detected ({water_pct:.1f}%). Best utilization: Integrate rainwater harvesting into nearby structures, implement periodic desilting to increase groundwater recharge, and create green buffer zones (planting native riparian vegetation) to prevent agricultural/urban runoff from polluting the water.",
            "type": "water"
        })
    elif water_pct > 0.1:
        recs.append({
            "priority": "Medium",
            "action": "Water Body Rejuvenation",
            "reason": "Small, possibly seasonal water bodies detected. Consider micro-irrigation setups sourcing from this, and construct check dams to ensure water availability year-round.",
            "type": "water"
        })

    # 3. Urbanization & Pollution Reduction
    if built_pct > 15:
        recs.append({
            "priority": "High",
            "action": "Pollution Reduction via Urban Greening",
            "reason": f"Highly urbanized region detected ({built_pct:.1f}%). To reduce air/noise pollution and the heat-island effect, mandate green roofs, vertical gardens on larger buildings, and plant pollution-absorbing native roadside trees (e.g., Peepal, Neem, Banyan).",
            "type": "urban"
        })

    # 4. Agricultural Land / Crop Rotation
    # If there is moderate vegetation but it's not dense (likely cropland), or barren land that might be unutilized agriculture:
    if veg_pct > 20 and veg_pct < 60:
        recs.append({
            "priority": "Medium",
            "action": "Crop Rotation & Agroforestry",
            "reason": "Agricultural or sparse vegetation detected. Best practice: Adopt crop rotations utilizing leguminous crops (like pulses) to naturally restore soil nitrogen. For unutilized tracts, practice agroforestry by planting timber trees on the farm boundaries for dual-income and wind-breaks.",
            "type": "agriculture"
        })
        
    # Default if nothing matches
    if not recs:
        recs.append({
            "priority": "Low",
            "action": "Conservation & Monitoring",
            "reason": "Diverse and highly vegetative land cover detected. Maintain current green coverage and monitor routinely. Support local biodiversity with minimal intervention.",
            "type": "general"
        })
        
    return recs


# ──────────────────────────────────────────────────────────────────────
# ZONE ENRICHMENT — Species + Scheme + Cost + Priority Score
# ──────────────────────────────────────────────────────────────────────

# Species recommendation matrix:  (soil_type, rainfall_band) → species list
_SPECIES_MAP = {
    ("black",    "high"):   ["Teak (Tectona grandis)", "Bamboo (Dendrocalamus strictus)", "Mahua (Madhuca longifolia)", "Arjun (Terminalia arjuna)"],
    ("black",    "medium"): ["Neem (Azadirachta indica)", "Mahua (Madhuca longifolia)", "Amaltas (Cassia fistula)", "Shisham (Dalbergia sissoo)"],
    ("black",    "low"):    ["Neem (Azadirachta indica)", "Khejri (Prosopis cineraria)", "Babool (Acacia nilotica)"],
    ("alluvial", "high"):   ["Teak/Sagwan (Tectona grandis)", "Peepal (Ficus religiosa)", "Arjun (Terminalia arjuna)", "Mango (Mangifera indica)"],
    ("alluvial", "medium"): ["Mango (Mangifera indica)", "Jamun (Syzygium cumini)", "Drumstick (Moringa oleifera)", "Indian Gooseberry (Amla)"],
    ("alluvial", "low"):    ["Neem (Azadirachta indica)", "Drumstick (Moringa oleifera)", "Indian Gooseberry (Amla)"],
    ("laterite", "high"):   ["Sal (Shorea robusta)", "Cashew (Anacardium occidentale)", "Bamboo (Dendrocalamus strictus)", "Kusum (Schleichera oleosa)"],
    ("laterite", "medium"): ["Neem (Azadirachta indica)", "Babool (Acacia nilotica)", "Indian Gooseberry (Amla)", "Karanj (Millettia pinnata)"],
    ("laterite", "low"):    ["Neem (Azadirachta indica)", "Babool (Acacia nilotica)", "Khejri (Prosopis cineraria)"],
    ("sandy",    "high"):   ["Neem (Azadirachta indica)", "Babool (Acacia nilotica)", "Casuarina (Casuarina equisetifolia)", "Eucalyptus"],
    ("sandy",    "medium"): ["Neem (Azadirachta indica)", "Khejri (Prosopis cineraria)", "Babool (Acacia nilotica)"],
    ("sandy",    "low"):    ["Khejri (Prosopis cineraria)", "Babool (Acacia nilotica)", "Rohida (Tecomella undulata)"],
}

# Species overrides per land-cover class (regardless of soil)
_SPECIES_BY_CLASS = {
    "Water":     ["Arjun (Terminalia arjuna) — riparian", "Kadam (Neolamarckia cadamba)", "River Acacia", "Vetiver grass (buffer zone)"],
    "Built-up":  ["Peepal (Ficus religiosa)", "Neem (Azadirachta indica)", "Banyan (Ficus benghalensis)", "Drumstick (Moringa oleifera)"],
    "Tree Cover":["Existing species — conserve", "Bamboo (understory)", "Native shrubs for understory biodiversity"],
    "Cropland":  ["Drumstick (Moringa oleifera) — boundary", "Mango (Mangifera indica) — boundary", "Indian Gooseberry (Amla)", "Subabul (Leucaena leucocephala)"],
}

# Government scheme mapping: (dominant_class_label, priority) → scheme info
_SCHEME_MAP = {
    ("Grassland",    "High"):   {"scheme": "Green India Mission (GIM) / CAMPA Fund",         "authority": "MoEFCC / State Forest Dept", "type": "Afforestation",              "rate_per_ha": 45000},
    ("Grassland",    "Medium"): {"scheme": "CAMPA Fund",                                       "authority": "State Forest Department",    "type": "Ecological Restoration",     "rate_per_ha": 32000},
    ("Shrubland",    "High"):   {"scheme": "Green India Mission (GIM)",                        "authority": "MoEFCC",                     "type": "Forest Upgradation",         "rate_per_ha": 35000},
    ("Cropland",     "High"):   {"scheme": "PMKSY (Har Khet Ko Paani) + RKVY",                "authority": "Ministry of Agriculture",    "type": "Soil Restoration + Irrigation","rate_per_ha": 28000},
    ("Cropland",     "Medium"): {"scheme": "RKVY (Rashtriya Krishi Vikas Yojana)",            "authority": "Ministry of Agriculture",    "type": "Agroforestry",               "rate_per_ha": 20000},
    ("Built-up",     "Medium"): {"scheme": "Smart Cities Mission / AMRUT 2.0",                "authority": "MoHUA",                      "type": "Urban Greening",             "rate_per_ha": 30000},
    ("Built-up",     "High"):   {"scheme": "Smart Cities Mission / AMRUT 2.0",                "authority": "MoHUA",                      "type": "Urban Greening",             "rate_per_ha": 30000},
    ("Water",        "High"):   {"scheme": "Amrit Sarovar Mission + MGNREGS",                 "authority": "Ministry of Jal Shakti",     "type": "Desilting + Buffer Zones",   "rate_per_ha": 15000},
    ("Water",        "Medium"): {"scheme": "MGNREGS / National Water Mission",                "authority": "Ministry of Jal Shakti",     "type": "Water Body Rejuvenation",    "rate_per_ha": 12000},
    ("Tree Cover",   "Low"):    {"scheme": "Joint Forest Management (JFM) / Van Dhan Yojana", "authority": "State Forest Department",    "type": "Conservation & Monitoring",  "rate_per_ha":  8000},
    ("Bare / Sparse","High"):   {"scheme": "Green India Mission (GIM) / CAMPA Fund",          "authority": "MoEFCC / State Forest Dept", "type": "Wasteland Development",      "rate_per_ha": 40000},
    ("Bare / Sparse","Medium"): {"scheme": "CAMPA Fund / National Afforestation Programme",   "authority": "State Forest Department",    "type": "Afforestation",              "rate_per_ha": 30000},
}

_DEFAULT_SCHEME = {"scheme": "State Forest Department Budget", "authority": "District Collector", "type": "General Greening", "rate_per_ha": 20000}

_PRIORITY_SCORE = {"High": 3, "Medium": 2, "Low": 1, "None": 0}


def enrich_zone_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich zone-level recommendations with:

    1. **Native species suggestions** — based on soil type, annual rainfall
       band, and dominant land-cover class (so each zone gets specific
       tree/plant names, not just a generic action).

    2. **Government scheme linkages** — maps each (land-cover, priority)
       combination to the relevant central or state government scheme
       (CAMPA, GIM, PMKSY, MGNREGS, Amrit Sarovar, etc.) along with
       the responsible authority.

    3. **Estimated implementation cost** — ₹/hectare × zone area to give
       district planners a rough budget figure.

    4. **Numeric priority score** — so the table can be sorted by urgency.

    Parameters
    ----------
    df : pd.DataFrame
        Output of generate_recommendations_ml() — must include columns:
        dominant_class_label, soil_type, rainfall_mm, priority, bare_pct,
        vegetation_pct, mean_ndvi, row_start, col_start.

    Returns
    -------
    pd.DataFrame
        Same dataframe with added columns:
        species, scheme, authority, intervention_type, rate_per_ha,
        zone_area_ha, est_cost_inr, est_cost_lakhs, priority_score.
    """
    ZONE_PIXEL_SIDE = 50        # zone grid is 50×50 pixels
    PIXEL_RESOLUTION = 10       # Sentinel-2 L2A = 10 m/pixel
    zone_area_m2 = (ZONE_PIXEL_SIDE * PIXEL_RESOLUTION) ** 2
    zone_area_ha = zone_area_m2 / 10_000  # 1 ha = 10,000 m²

    species_list, schemes, authorities, int_types = [], [], [], []
    rates, costs_inr, costs_lakh, scores = [], [], [], []

    for _, row in df.iterrows():
        dom_class   = str(row.get("dominant_class_label", "Grassland"))
        soil        = str(row.get("soil_type", "alluvial")).lower().strip()
        rainfall    = float(row.get("rainfall_mm", 800))
        priority    = str(row.get("priority", "None"))
        veg_pct     = float(row.get("vegetation_pct", 0))
        bare_pct    = float(row.get("bare_pct", 0))

        # ── 1. Species ────────────────────────────────────────────────
        if dom_class in _SPECIES_BY_CLASS:
            sp = _SPECIES_BY_CLASS[dom_class]
        else:
            # Rainfall band: high>900, medium 500–900, low<500
            rain_band = "high" if rainfall >= 900 else ("medium" if rainfall >= 500 else "low")
            sp = _SPECIES_MAP.get((soil, rain_band),
                 _SPECIES_MAP.get(("alluvial", rain_band),
                 ["Neem (Azadirachta indica)", "Peepal (Ficus religiosa)", "Babool (Acacia nilotica)"]))
        species_list.append(" · ".join(sp[:3]))      # show top-3

        # ── 2. Scheme ────────────────────────────────────────────────
        scheme_info = _SCHEME_MAP.get((dom_class, priority), _DEFAULT_SCHEME)
        schemes.append(scheme_info["scheme"])
        authorities.append(scheme_info["authority"])
        int_types.append(scheme_info["type"])
        rate = scheme_info["rate_per_ha"]
        rates.append(rate)

        # ── 3. Cost ──────────────────────────────────────────────────
        # Scale cost by how degraded the zone is (bare % contributes more)
        degradation_factor = min(1.0, (bare_pct / 100) + (1 - veg_pct / 100) * 0.5)
        effective_rate = rate * max(0.3, degradation_factor)
        cost_inr = effective_rate * zone_area_ha
        costs_inr.append(int(cost_inr))
        costs_lakh.append(round(cost_inr / 1e5, 2))

        # ── 4. Priority score ────────────────────────────────────────
        base = _PRIORITY_SCORE.get(priority, 0)
        ndvi_penalty = max(0, 0.4 - float(row.get("mean_ndvi", 0.4)))  # low NDVI = bad
        score = base + ndvi_penalty + bare_pct / 100
        scores.append(round(score, 3))

    df = df.copy()
    df["species"]           = species_list
    df["scheme"]            = schemes
    df["authority"]         = authorities
    df["intervention_type"] = int_types
    df["rate_per_ha"]       = rates
    df["zone_area_ha"]      = round(zone_area_ha, 2)
    df["est_cost_inr"]      = costs_inr
    df["est_cost_lakhs"]    = costs_lakh
    df["priority_score"]    = scores

    return df
