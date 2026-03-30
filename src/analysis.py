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
