"""
GeoGreen Revolution — Climate Data API Integration
===================================================
Fetches historical climate data (rainfall, temperature) for any
global location using the Open-Meteo Archive API (Free).

This replaces the static CSV approach with real-time API lookups,
enabling analysis for ANY coordinates extracted from images.

Usage:
    from climate_api import get_climate_data
    climate = get_climate_data(lat, lon)
"""

import requests
import datetime
import numpy as np

# Open-Meteo Archive API (Free, no key required)
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_climate_data(lat, lon, years=5):
    """
    Fetch historical climate data for the given coordinates.
    Calculates annual averages based on the last N years.

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.
    years : int, optional
        Number of past years to analyze (default 5).

    Returns
    -------
    dict
        {
            'annual_rainfall_mm': float,
            'mean_temp_c': float,
            'soil_type': str (inferred/placeholder),
            'elevation_m': float (from API),
            'is_valid': bool
        }
    """
    try:
        # Calculate date range (last N full years)
        end_date = datetime.date.today().replace(month=1, day=1) - datetime.timedelta(days=1)
        start_date = end_date.replace(year=end_date.year - years)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean,rain_sum",
            "timezone": "auto",
        }

        print(f"  ☁️  Fetching climate data for {lat:.4f}, {lon:.4f} ({years} years)...")
        response = requests.get(API_URL, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"  ❌ API Error: {response.status_code}")
            return _get_fallback_data()

        data = response.json()

        # Process daily data
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        rains = daily.get("rain_sum", [])
        elevation = data.get("elevation", 0)

        # Filter out None values
        temps = [t for t in temps if t is not None]
        rains = [r for r in rains if r is not None]

        if not temps or not rains:
            return _get_fallback_data()

        # Calculate statistics
        mean_temp = np.mean(temps)
        total_rain = np.sum(rains)
        avg_annual_rain = total_rain / years

        print(f"     ✅ Data acquired: {avg_annual_rain:.1f} mm/yr, {mean_temp:.1f}°C")

        return {
            "annual_rainfall_mm": round(avg_annual_rain, 1),
            "mean_temp_c": round(mean_temp, 1),
            "soil_type": "Unknown (Field check required)",  # API doesn't provide soil
            "elevation_m": elevation,
            "location_ok": True,
        }

    except Exception as e:
        print(f"  ⚠ Climate API Warning: {e}")
        return _get_fallback_data()


def _get_fallback_data():
    """Return safe fallback defaults."""
    return {
        "annual_rainfall_mm": 1000.0,
        "mean_temp_c": 25.0,
        "soil_type": "Unknown",
        "elevation_m": 0,
        "location_ok": False,
    }
