import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import io

# Add src to sys.path so we can import modules directly by name
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Fix for OpenMP DLL conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Backend imports (direct module names since src/ has no __init__.py)
from cv_analysis import analyze_land_cover
from climate_api import get_climate_data
from analysis import generate_rgb_recommendations

# Optional: Folium for interactive maps
try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoGreen Revolution AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Fix Streamlit Material Icons font fallback overlap bug in dataframe dropdowns */
div[role="menuitem"] span.material-symbols-rounded, 
div[role="columnheader"] span.material-symbols-rounded { 
    display: none !important; 
}

* { font-family: 'Inter', sans-serif !important; }

html, body, .stApp {
    background: #0d1117 !important;
    color: #e6edf3 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0d2818 0%, #1a4731 40%, #0d3320 70%, #091d10 100%);
    border: 1px solid #2ea043;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(46,160,67,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 8px 0;
    line-height: 1.1;
}
.hero-title span { color: #4ade80; }
.hero-subtitle {
    font-size: 1.05rem;
    color: #8b949e;
    margin: 0 0 20px 0;
    font-weight: 400;
}
.hero-problem {
    font-size: 0.95rem;
    color: #c9d1d9;
    line-height: 1.6;
    max-width: 720px;
    padding: 16px 20px;
    background: rgba(46,160,67,0.08);
    border-left: 3px solid #2ea043;
    border-radius: 0 8px 8px 0;
}

/* KPI cards */
.kpi-grid { display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }
.kpi-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1;
    min-width: 160px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2ea043; }
.kpi-label { font-size: 0.75rem; color: #8b949e; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.kpi-value { font-size: 1.65rem; font-weight: 700; color: #ffffff; line-height: 1; }
.kpi-sub { font-size: 0.75rem; color: #4ade80; margin-top: 4px; font-weight: 500; }

/* Pipeline flow */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 20px 0;
    overflow-x: auto;
}
.pipe-step {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    min-width: 110px;
    font-size: 0.78rem;
    color: #c9d1d9;
    font-weight: 500;
}
.pipe-step .icon { font-size: 1.3rem; display: block; margin-bottom: 4px; }
.pipe-step.active { border-color: #2ea043; background: rgba(46,160,67,0.1); color: #4ade80; }
.pipe-arrow {
    color: #30363d;
    font-size: 1.2rem;
    padding: 0 4px;
    user-select: none;
}

/* Section headers */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Map cards */
.map-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 16px;
}
.map-card-header {
    background: #21262d;
    padding: 12px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Stat metric cards */
.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.1s;
}
.stat-card:hover { border-color: #2ea043; transform: translateY(-1px); }
.stat-value { font-size: 1.8rem; font-weight: 800; margin-bottom: 2px; }
.stat-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; font-weight: 500; letter-spacing: 0.04em; }

/* Priority badge */
.badge-high { background: rgba(248,81,73,0.15); color: #ff6b6b; border: 1px solid rgba(248,81,73,0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; display: inline-block; }
.badge-medium { background: rgba(255,199,0,0.12); color: #f5a623; border: 1px solid rgba(255,199,0,0.25); padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; display: inline-block; }
.badge-low { background: rgba(46,160,67,0.12); color: #4ade80; border: 1px solid rgba(46,160,67,0.25); padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; display: inline-block; }

/* Rec cards */
.rec-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: #4ade80; }
.rec-title { font-size: 0.95rem; font-weight: 600; color: #e6edf3; margin: 8px 0 6px; }
.rec-reason { font-size: 0.83rem; color: #8b949e; line-height: 1.5; }

/* Impact box */
.impact-box {
    background: linear-gradient(135deg, #0d2818, #091a10);
    border: 1px solid #2ea043;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}
.impact-title { font-size: 1rem; font-weight: 700; color: #4ade80; margin: 0 0 16px; }
.impact-row { display: flex; gap: 12px; flex-wrap: wrap; }
.impact-item { flex: 1; min-width: 120px; text-align: center; padding: 12px; background: rgba(46,160,67,0.08); border-radius: 8px; }
.impact-num { font-size: 1.4rem; font-weight: 800; color: #ffffff; }
.impact-desc { font-size: 0.72rem; color: #4ade80; margin-top: 2px; }

/* Override Streamlit elements */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 0 !important;
    padding: 0 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b949e !important;
    border-radius: 6px 6px 0 0 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: #0d1117 !important;
    color: #4ade80 !important;
    border-bottom: 2px solid #2ea043 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 24px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(46,160,67,0.3) !important;
}
.stFileUploader {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
}
.stMetric {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
.stMetric [data-testid="metric-container"] { color: #e6edf3 !important; }
div[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
}
.stAlert { border-radius: 8px !important; }
hr { border-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PIXEL_SIZE_M = 10  # Sentinel-2 L2A at 10m resolution
TOTAL_PIXELS = 3_173_926  # From the last run (1946×1631)
AREA_KM2 = round(TOTAL_PIXELS * PIXEL_SIZE_M * PIXEL_SIZE_M / 1e6, 1)  # ~317 km²

# ── Helper: load output data ─────────────────────────────────────────
def load_results():
    rec_csv = os.path.join(OUTPUT_DIR, "recommendations.csv")
    summary_txt = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
    results = {"loaded": False}

    if os.path.exists(rec_csv):
        try:
            df = pd.read_csv(rec_csv)

            # ── Enrich with species / schemes / cost / score ──────────
            try:
                from analysis import enrich_zone_recommendations
                df = enrich_zone_recommendations(df)
            except Exception as enrich_err:
                pass  # graceful fallback — run without enrichment

            results["df"] = df
            results["loaded"] = True

            # Compute aggregates
            total = len(df)
            results["total_zones"] = total
            results["high_priority"] = int((df["priority"] == "High").sum())
            results["medium_priority"] = int((df["priority"] == "Medium").sum())
            results["low_priority"] = int((df["priority"] == "Low").sum())

            # Land cover distribution from dominant_class_label
            class_dist = df["dominant_class_label"].value_counts(normalize=True) * 100
            results["class_dist"] = class_dist.to_dict()

            # Mean NDVI
            results["mean_ndvi"] = round(float(df["mean_ndvi"].mean()), 3)

            # Water zones
            water_zones = df[df["dominant_class_label"] == "Water"]
            results["water_zone_count"] = len(water_zones)
            results["water_area_km2"] = round(len(water_zones) * (50 * 10 / 1000) ** 2, 2)

            # Grassland to forest potential (main restoration target)
            grass_zones = df[df["dominant_class_label"] == "Grassland"]
            results["grass_zones"] = len(grass_zones)
            bare_pct_total = df["bare_pct"].mean() if "bare_pct" in df.columns else 0
            results["bare_pct"] = round(bare_pct_total, 1)

            # Impact estimates
            restorable_zones = results["high_priority"]
            zone_area_m2 = 50 * PIXEL_SIZE_M * 50 * PIXEL_SIZE_M
            trees_possible = int(restorable_zones * zone_area_m2 / 25)  # 1 tree per 5x5m
            results["trees_possible"] = trees_possible
            results["co2_tonnes_yr"] = round(trees_possible * 21 / 1000, 0)
            results["restorable_area_km2"] = round(restorable_zones * zone_area_m2 / 1e6, 1)

            # Total estimated implementation cost (from enrichment)
            if "est_cost_inr" in df.columns:
                total_cost_inr = df.loc[df["priority"].isin(["High","Medium"]), "est_cost_inr"].sum()
                results["total_cost_crore"] = round(total_cost_inr / 1e7, 1)
            else:
                results["total_cost_crore"] = None

        except Exception as e:
            results["error"] = str(e)

    if os.path.exists(summary_txt):
        with open(summary_txt, "r", encoding="utf-8") as f:
            results["summary_text"] = f.read()

    return results


# ══════════════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🌿 <span>GeoGreen</span> Revolution AI</div>
  <div class="hero-subtitle">AI-Powered Geospatial Decision Support System · Pilot: Sehore District, Madhya Pradesh</div>
  <div class="hero-problem">
    India loses over <strong>1.5 million hectares</strong> of forest annually and vast tracts of rural land remain 
    degraded — offering no ecological or economic value. GeoGreen Revolution uses <strong>real Sentinel-2 
    satellite imagery</strong> and pretrained deep learning models to automatically detect underutilised land, 
    classify its vegetation cover, and deliver <strong>actionable greening recommendations</strong> for 
    policymakers and local authorities.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ────────────────────────────────────────────────────────
results = load_results()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Region Area Analysed</div>
        <div class="kpi-value">317 km²</div>
        <div class="kpi-sub">Sehore District, MP</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ML Model Accuracy</div>
        <div class="kpi-value">75.6%</div>
        <div class="kpi-sub">ESA WorldCover (global)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    hp = results.get("high_priority")
    hp_str = f"{hp:,}" if isinstance(hp, (int, float)) else "—"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">High Priority Zones</div>
        <div class="kpi-value">{hp_str}</div>
        <div class="kpi-sub">Needing intervention</div>
    </div>""", unsafe_allow_html=True)
with c4:
    trees = results.get("trees_possible", 0)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Trees Plantable</div>
        <div class="kpi-value">{trees//1_000_000:.1f}M</div>
        <div class="kpi-sub">Estimated potential</div>
    </div>""", unsafe_allow_html=True)
with c5:
    co2 = results.get("co2_tonnes_yr", 0)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">CO₂ Offset / Year</div>
        <div class="kpi-value">{int(co2//1000)}K+</div>
        <div class="kpi-sub">Tonnes if restored</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline Flow ─────────────────────────────────────────────────────
st.markdown("""
<p class="section-header">⚙️ How the AI Pipeline Works</p>
<div class="pipeline">
    <div class="pipe-step active"><span class="icon">🛰️</span>Sentinel-2<br>Satellite Data</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><span class="icon">🔧</span>Band<br>Extraction</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step active"><span class="icon">📊</span>NDVI + NDWI<br>Computation</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step active"><span class="icon">🤖</span>ESA WorldCover<br>ML Inference</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><span class="icon">🔬</span>Index<br>Refinement</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step"><span class="icon">🌦️</span>Climate<br>Fusion</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step active"><span class="icon">📋</span>Greening<br>Recommendations</div>
</div>
<br>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════
tab_exec, tab_pipeline, tab_results, tab_recs, tab_analyze, tab_report = st.tabs([
    "🏠 Executive Summary",
    "🚀 Run AI Pipeline",
    "📊 Scientific Results",
    "📋 Recommendations",
    "🛰️ Live Demo (Upload Image)",
    "📄 Full Report"
])

# ─────────────────────────────────────────────────────────────────────
# TAB 0: EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────
with tab_exec:
    _df = results.get("df", None)
    _loaded = results.get("loaded", False)

    # ── Headline ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:32px 0 24px;">
        <div style="font-size:0.8rem; color:#4ade80; font-weight:700; letter-spacing:.15em;
                    text-transform:uppercase; margin-bottom:10px;">Pilot Study · Sehore District, Madhya Pradesh, India</div>
        <div style="font-size:2.4rem; font-weight:800; color:#ffffff; line-height:1.15; margin-bottom:12px;">
            317 km² Analysed From Space.<br>
            <span style="color:#4ade80;">1,001 Zones Need Intervention.</span>
        </div>
        <div style="font-size:1rem; color:#8b949e; max-width:640px; margin:0 auto; line-height:1.6;">
            GeoGreen Revolution is an AI-powered geospatial decision-support system that turns 
            real Sentinel-2 satellite imagery into <strong style="color:#c9d1d9;">specific, fundable, field-ready 
            greening recommendations</strong> for government authorities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ex_col1, ex_col2 = st.columns([1.1, 1])

    with ex_col1:
        # ── Problem ─────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0a0a,#2d0f0f); border:1px solid #e6394630;
                    border-left:4px solid #e63946; border-radius:10px; padding:18px 20px; margin-bottom:16px;">
            <div style="font-size:0.75rem; color:#e63946; font-weight:700; text-transform:uppercase;
                        letter-spacing:.08em; margin-bottom:8px;">⚠️ The Problem</div>
            <div style="font-size:0.92rem; color:#e6edf3; line-height:1.6;">
                India loses <strong>1.5 million hectares</strong> of forest annually. Over <strong>30% of India's 
                land is degraded</strong>, costing ₹2.5 lakh crore/year in lost ecosystem services. Local 
                administrations have no scalable way to identify <em>where</em> to intervene or <em>which</em> 
                schemes to apply.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Our Solution ─────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0a1a0f,#0d2818); border:1px solid #2ea04330;
                    border-left:4px solid #2ea043; border-radius:10px; padding:18px 20px; margin-bottom:16px;">
            <div style="font-size:0.75rem; color:#4ade80; font-weight:700; text-transform:uppercase;
                        letter-spacing:.08em; margin-bottom:8px;">✅ Our Solution</div>
            <div style="font-size:0.92rem; color:#e6edf3; line-height:1.6;">
                A 4-step AI pipeline: <strong>Sentinel-2 imagery</strong> → <strong>ESA WorldCover ML model</strong> 
                (U-Net + Random Forest, 75.6% accuracy) → <strong>NDVI/NDWI spectral analysis</strong> → 
                <strong>Climate-fused zone recommendations</strong> with specific native species, government 
                scheme linkages, and ₹ budget estimates — all at <strong>10m resolution</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key Findings ─────────────────────────────────────────────
        st.markdown('<div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; font-weight:700; letter-spacing:.08em; margin-bottom:10px;">📌 Key Findings</div>', unsafe_allow_html=True)

        if _loaded and _df is not None:
            hp  = results["high_priority"]
            r_area = results.get("restorable_area_km2", 0)
            trees  = results.get("trees_possible", 0)
            co2    = results.get("co2_tonnes_yr", 0)
            cost   = results.get("total_cost_crore", 0)
            warea  = results.get("water_area_km2", 0)

            findings = [
                ("🔴", str(hp), "High-priority zones", "Needing immediate greening intervention", "#e63946"),
                ("🌲", f"{trees//1_000_000:.1f}M", "Trees plantable", f"Across {r_area} km² of restorable land", "#4ade80"),
                ("🌬️", f"{int(co2//1000)}K t", "CO₂ offset/year", "If all High Priority zones are restored", "#58a6ff"),
                ("💧", f"{warea} km²", "Water bodies", "Identified for Amrit Sarovar scheme protection", "#4895ef"),
                ("💰", f"₹{cost} Cr", "Estimated budget", "High+Medium zones via CAMPA/GIM/PMKSY schemes", "#f5a623"),
            ]

            for icon, val, label, desc, color in findings:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:14px; background:#161b22;
                            border:1px solid #30363d; border-radius:8px; padding:12px 16px; margin-bottom:8px;">
                    <div style="font-size:1.5rem; width:36px; text-align:center;">{icon}</div>
                    <div style="flex:1;">
                        <div style="display:flex; align-items:baseline; gap:8px;">
                            <span style="font-size:1.5rem; font-weight:800; color:{color};">{val}</span>
                            <span style="font-size:0.85rem; font-weight:600; color:#e6edf3;">{label}</span>
                        </div>
                        <div style="font-size:0.75rem; color:#8b949e;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run `python src/main.py` to load findings here.")

    with ex_col2:
        # ── 3-Phase Implementation Roadmap ───────────────────────────
        st.markdown("""
        <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; font-weight:700;
                    letter-spacing:.08em; margin-bottom:12px;">🗓️ Implementation Roadmap</div>
        """, unsafe_allow_html=True)

        phases = [
            ("Phase 1", "0–6 months", "Survey & Planning", [
                "Ground-truth top 50 High Priority zones",
                "Prepare DPR for CAMPA/GIM funding",
                "Identify species nurseries (Teak, Bamboo, Neem)",
                "Soil testing for zone-specific validation",
            ], "#e63946", "🔴"),
            ("Phase 2", "6–18 months", "Afforestation & Restoration", [
                "Plant 2–4M trees in High Priority grassland zones",
                "Desilt 14 water body zones (Amrit Sarovar Mission)",
                "Install drip irrigation for Cropland zones (PMKSY)",
                "Trigger MGNREGS work in rural zones (employment)",
            ], "#f5a623", "🟡"),
            ("Phase 3", "18+ months", "Monitor & Scale", [
                "Annual satellite re-analysis to track NDVI improvement",
                "Carbon credit registration (REDD+ / Verra VCS)",
                "Expand model to adjacent Vidisha & Raisen districts",
                "Publish validated impact data for policy advocacy",
            ], "#4ade80", "🟢"),
        ]

        for phase, timeline, title, actions, color, icon in phases:
            actions_html = "".join([f'<li style="margin-bottom:4px; color:#c9d1d9;">{a}</li>' for a in actions])
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid {color}30; border-left:4px solid {color};
                        border-radius:10px; padding:16px 18px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span style="font-size:0.75rem; color:{color}; font-weight:700; text-transform:uppercase;">{icon} {phase}</span>
                        <div style="font-size:0.95rem; font-weight:700; color:#ffffff;">{title}</div>
                    </div>
                    <div style="font-size:0.72rem; color:#8b949e; text-align:right;">{timeline}</div>
                </div>
                <ul style="margin:0; padding-left:18px; font-size:0.78rem; line-height:1.7;">
                    {actions_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # ── Methodology Credibility Box ──────────────────────────────
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #30363d; border-radius:10px;
                    padding:16px 18px; margin-top:4px;">
            <div style="font-size:0.75rem; color:#4ade80; font-weight:700; text-transform:uppercase;
                        letter-spacing:.08em; margin-bottom:10px;">🔬 Methodology & Credibility</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.78rem;">
                <div style="color:#8b949e;">📡 <strong style="color:#c9d1d9;">Satellite:</strong> Sentinel-2 L2A</div>
                <div style="color:#8b949e;">📐 <strong style="color:#c9d1d9;">Resolution:</strong> 10m/pixel</div>
                <div style="color:#8b949e;">🤖 <strong style="color:#c9d1d9;">ML Model:</strong> ESA WorldCover 2021</div>
                <div style="color:#8b949e;">✅ <strong style="color:#c9d1d9;">Accuracy:</strong> 75.6% (global)</div>
                <div style="color:#8b949e;">🌿 <strong style="color:#c9d1d9;">Indices:</strong> NDVI · NDWI</div>
                <div style="color:#8b949e;">🌦️ <strong style="color:#c9d1d9;">Climate:</strong> IMD / WorldClim</div>
                <div style="color:#8b949e; grid-column:span 2;">📄 <strong style="color:#c9d1d9;">Reference:</strong> Zanaga et al. (2022) — ESA WorldCover 10m v100. DOI: 10.5281/zenodo.7254221</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 1: RUN AI PIPELINE
# ─────────────────────────────────────────────────────────────────────
with tab_pipeline:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d2818,#091a10); border:1px solid #2ea043;
                border-radius:14px; padding:24px 28px; margin-bottom:24px;">
        <div style="font-size:1.5rem; font-weight:800; color:#ffffff; margin-bottom:6px;">🚀 Run the AI Pipeline</div>
        <div style="font-size:0.9rem; color:#8b949e; line-height:1.6;">
            Upload your <strong style="color:#c9d1d9;">Sentinel-2 satellite band files</strong> and optional
            <strong style="color:#c9d1d9;">ESA WorldCover</strong> map, then click
            <strong style="color:#4ade80;">Run</strong>. The pipeline will classify land cover,
            compute NDVI/NDWI, generate greening recommendations, and save all result maps
            — which will instantly appear in the <em>Scientific Results</em> and
            <em>Recommendations</em> tabs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    def stack_band_files(band_files_dict):
        import rasterio
        from rasterio.enums import Resampling
        valid_bands = [(label, f) for label, f in band_files_dict.items() if f is not None]
        if not valid_bands:
            raise ValueError("No band files provided.")
        tmp_band_paths = []
        for label, uploaded in valid_bands:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_band_paths.append(tmp.name)
        with rasterio.open(tmp_band_paths[0]) as ref:
            ref_shape = (ref.height, ref.width)
            ref_profile = ref.profile.copy()
        out_profile = ref_profile.copy()
        out_profile.update(count=len(valid_bands), dtype="float32")
        stacked_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        stacked_path = stacked_tmp.name
        stacked_tmp.close()
        with rasterio.open(stacked_path, "w", **out_profile) as dst:
            for i, (band_path, (label, _)) in enumerate(zip(tmp_band_paths, valid_bands), start=1):
                with rasterio.open(band_path) as src:
                    if (src.height, src.width) == ref_shape:
                        data = src.read(1).astype("float32")
                    else:
                        data = src.read(1, out_shape=(1, ref_shape[0], ref_shape[1]),
                                       resampling=Resampling.bilinear).squeeze().astype("float32")
                    dst.write(data, i)
        for p in tmp_band_paths:
            try: os.remove(p)
            except: pass
        return stacked_path

    pipe_tab1, pipe_tab2 = st.tabs(["📦 Single Stacked .tif (Easy)", "🗂️ Individual Bands B02–B11 (Advanced)"])

    # ── Option A: Single stacked TIF ─────────────────────────────────
    with pipe_tab1:
        st.markdown("""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:14px 18px; margin-bottom:20px; font-size:0.85rem; color:#8b949e;">
            Use this if you already have a <strong style="color:#c9d1d9;">pre-stacked multi-band GeoTIFF</strong>
            (e.g. the files in <code>d:/EPICS/</code> like <code>sehore_2024_01.tif</code>).
            These already contain all bands stacked together.
        </div>
        """, unsafe_allow_html=True)

        p1c1, p1c2, p1c3 = st.columns(3)
        with p1c1:
            sci_sat_file = st.file_uploader(
                "🛰️ Satellite Image (.tif) ✱",
                type=["tif","tiff"], key="sci_sat",
                help="Multi-band Sentinel-2 GeoTIFF — e.g. sehore_2024_01.tif"
            )
        with p1c2:
            sci_wc_file = st.file_uploader(
                "🌍 ESA WorldCover (.tif) — optional but enables full ML mode",
                type=["tif","tiff"], key="sci_wc",
                help="Enables ESA WorldCover land classification. Without it, rule-based NDVI is used."
            )
        with p1c3:
            sci_clim_file = st.file_uploader(
                "🌦️ Climate Data (.csv) ✱",
                type=["csv"], key="sci_clim",
                help="CSV with climate variables for the region"
            )

        # Status indicators
        s1, s2, s3 = st.columns(3)
        with s1:
            if sci_sat_file: st.success(f"✅ {sci_sat_file.name} ({sci_sat_file.size/1e6:.1f} MB)")
            else: st.warning("⚠️ Satellite .tif required")
        with s2:
            if sci_wc_file: st.success(f"✅ {sci_wc_file.name} (ML mode enabled)")
            else: st.info("ℹ️ No WorldCover — rule-based mode")
        with s3:
            if sci_clim_file: st.success(f"✅ {sci_clim_file.name}")
            else: st.warning("⚠️ Climate CSV required")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Run AI Pipeline", type="primary", key="run_single", use_container_width=True):
            if sci_sat_file is None or sci_clim_file is None:
                st.error("❌ Satellite .tif and Climate .csv are both required.")
            else:
                with st.spinner("🤖 Running AI Pipeline… this may take 1–3 minutes for large images."):
                    try:
                        from main import main as run_pipeline
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f1:
                            f1.write(sci_sat_file.getbuffer()); sat_path = f1.name
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f2:
                            f2.write(sci_clim_file.getbuffer()); clim_path = f2.name
                        wc_path = None
                        if sci_wc_file:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f3:
                                f3.write(sci_wc_file.getbuffer()); wc_path = f3.name
                        run_pipeline(satellite_path=sat_path, climate_path=clim_path, worldcover_path=wc_path)
                        st.cache_data.clear()
                        st.success("✅ Pipeline complete! Go to the **📊 Scientific Results** and **📋 Recommendations** tabs to see your results.")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
                        st.exception(e)

    # ── Option B: Individual band files ───────────────────────────────
    with pipe_tab2:
        st.markdown("""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:14px 18px; margin-bottom:20px; font-size:0.85rem; color:#8b949e;">
            Upload individual Sentinel-2 band files (e.g. <code>B02.tif</code>, <code>B03.tif</code>…).
            They will be <strong style="color:#c9d1d9;">automatically stacked</strong> before running the pipeline.
            <br>✱ = required · All other fields are optional.
        </div>
        """, unsafe_allow_html=True)

        bc1, bc2, bc3 = st.columns(3)
        bd1, bd2 = st.columns(2)
        with bc1: b02_file = st.file_uploader("B02 Blue ✱", type=["tif","tiff"], key="b02")
        with bc2: b03_file = st.file_uploader("B03 Green ✱", type=["tif","tiff"], key="b03")
        with bc3: b04_file = st.file_uploader("B04 Red ✱", type=["tif","tiff"], key="b04")
        with bd1: b08_file = st.file_uploader("B08 NIR ✱", type=["tif","tiff"], key="b08")
        with bd2: b11_file = st.file_uploader("B11 SWIR (optional)", type=["tif","tiff"], key="b11")
        be1, be2 = st.columns(2)
        with be1: bands_wc_file = st.file_uploader("WorldCover (.tif optional)", type=["tif","tiff"], key="bands_wc")
        with be2: bands_clim_file = st.file_uploader("Climate CSV ✱", type=["csv"], key="bands_clim")

        required_bands = {"B02": b02_file, "B03": b03_file, "B04": b04_file, "B08": b08_file}
        missing = [k for k, v in required_bands.items() if v is None]
        if not missing and bands_clim_file:
            st.success("✅ All required files uploaded — ready to run!")
        else:
            if missing: st.warning(f"⚠️ Still needed: **{', '.join(missing)}**")
            if bands_clim_file is None: st.warning("⚠️ Climate CSV is required")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔗 Stack Bands & Run AI Pipeline", type="primary", key="run_bands", use_container_width=True):
            if missing or bands_clim_file is None:
                st.error("❌ Missing required files. Upload all ✱ items first.")
            else:
                stacked_path = None
                with st.spinner("🔗 Stacking band files…"):
                    try:
                        stacked_path = stack_band_files({"B02":b02_file,"B03":b03_file,"B04":b04_file,"B08":b08_file,"B11":b11_file})
                        st.success("✅ Bands stacked. Running AI pipeline…")
                    except Exception as e:
                        st.error(f"Stacking failed: {e}"); stacked_path = None
                if stacked_path:
                    with st.spinner("🤖 Running AI Pipeline… this may take 1–3 minutes."):
                        try:
                            from main import main as run_pipeline
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fc:
                                fc.write(bands_clim_file.getbuffer()); clim_path = fc.name
                            wc_path = None
                            if bands_wc_file:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as fw:
                                    fw.write(bands_wc_file.getbuffer()); wc_path = fw.name
                            run_pipeline(satellite_path=stacked_path, climate_path=clim_path, worldcover_path=wc_path)
                            st.cache_data.clear()
                            st.success("✅ Pipeline complete! Go to **📊 Scientific Results** and **📋 Recommendations** tabs.")
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
                            st.exception(e)
                        finally:
                            try: os.remove(stacked_path)
                            except: pass

# ─────────────────────────────────────────────────────────────────────
# TAB 2: SCIENTIFIC RESULTS
# ─────────────────────────────────────────────────────────────────────
with tab_results:
    has_output = os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0

    if not has_output:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px;">
            <div style="font-size:3rem; margin-bottom:16px;">🛸</div>
            <div style="font-size:1.3rem; font-weight:700; color:#ffffff; margin-bottom:10px;">
                No results yet — let's generate some!
            </div>
            <div style="font-size:0.95rem; color:#8b949e; max-width:480px; margin:0 auto 24px; line-height:1.6;">
                Upload your Sentinel-2 <code>.tif</code> files in the
                <strong style="color:#4ade80;">🚀 Run AI Pipeline</strong> tab to generate
                land cover maps, NDVI analysis, and greening recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Impact Summary ──────────────────────────────────────────
        if results.get("loaded"):
            r_area = results.get("restorable_area_km2", 0)
            trees = results.get("trees_possible", 0)
            co2 = results.get("co2_tonnes_yr", 0)
            w_area = results.get("water_area_km2", 0)

            st.markdown(f"""
            <div class="impact-box">
                <div class="impact-title">🌍 Estimated Environmental Impact (if interventions are implemented)</div>
                <div class="impact-row">
                    <div class="impact-item">
                        <div class="impact-num">{r_area} km²</div>
                        <div class="impact-desc">Land available for restoration</div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-num">{trees:,}</div>
                        <div class="impact-desc">Trees that could be planted</div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-num">{int(co2):,} t</div>
                        <div class="impact-desc">CO₂ offset per year</div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-num">{w_area} km²</div>
                        <div class="impact-desc">Water bodies identified</div>
                    </div>
                    <div class="impact-item">
                        <div class="impact-num">{results['mean_ndvi']}</div>
                        <div class="impact-desc">Mean NDVI (vegetation health)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Land Cover Stats ────────────────────────────────────────
        col_metrics, col_dist = st.columns([1.1, 1])

        with col_metrics:
            st.markdown('<p class="section-header">🗺️ Output Maps</p>', unsafe_allow_html=True)

            ml_map = os.path.join(OUTPUT_DIR, "ml_landcover_map.png")
            fused_map = os.path.join(OUTPUT_DIR, "fused_landcover_map.png")
            ndvi_map = os.path.join(OUTPUT_DIR, "ndvi_map.png")
            rec_map = os.path.join(OUTPUT_DIR, "recommendation_map.png")
            rgb_map = os.path.join(OUTPUT_DIR, "rgb_composite.png")

            map_tabs = st.tabs(["🤖 ML Classification", "🔬 Fused Map", "🌱 NDVI", "⚡ Recommendations", "🛰 RGB", "🗺️ Interactive Map"])

            with map_tabs[0]:
                if os.path.exists(ml_map):
                    st.image(ml_map, caption="AI Land-Cover Classification — ESA WorldCover (U-Net + Random Forest, 10m resolution)", use_container_width=True)
                    st.caption("Classes: Tree Cover · Shrubland · Grassland · Cropland · Built-up · Bare/Sparse · Water")
                else:
                    st.info("Run `python src/main.py` to generate this map")

            with map_tabs[1]:
                if os.path.exists(fused_map):
                    st.image(fused_map, caption="Fused Classification — ML output refined with NDVI & NDWI spectral indices", use_container_width=True)
                else:
                    st.info("Run pipeline with WorldCover data to generate this map")

            with map_tabs[2]:
                if os.path.exists(ndvi_map):
                    st.image(ndvi_map, caption="NDVI Vegetation Health Map — Red = Barren · Yellow = Sparse · Green = Dense Vegetation", use_container_width=True)

            with map_tabs[3]:
                if os.path.exists(rec_map):
                    st.image(rec_map, caption="Greening Intervention Priority Map — Zone-wise priority heatmap", use_container_width=True)

            with map_tabs[4]:
                if os.path.exists(rgb_map):
                    st.image(rgb_map, caption="True-Color RGB Composite from Sentinel-2 Bands", use_container_width=True)

            with map_tabs[5]:  # Interactive Folium map
                if not FOLIUM_AVAILABLE:
                    st.warning("📦 Install folium: `pip install folium` then restart the app.")
                elif not results.get("loaded"):
                    st.info("Run the scientific pipeline first to generate zone data.")
                else:
                    _idf = results["df"].copy()

                    # ── Coordinate conversion ───────────────────────
                    # Try rasterio for real coords, fall back to approx Sehore bounds
                    _geo = {"lat_min":22.62,"lat_max":23.18,"lon_min":76.88,"lon_max":77.62,
                            "img_h":1631,"img_w":1946}
                    try:
                        import rasterio
                        from rasterio.crs import CRS as _CRS
                        from rasterio.warp import transform_bounds as _tb
                        _sat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "satellite")
                        _tifs = [f for f in os.listdir(_sat_dir) if f.endswith(".tif")] if os.path.exists(_sat_dir) else []
                        if _tifs:
                            with rasterio.open(os.path.join(_sat_dir, _tifs[0])) as _src:
                                _b = _src.bounds
                                _crs = _src.crs
                                _h, _w = _src.height, _src.width
                            if _crs and not _crs.is_geographic:
                                _l,_bo,_r,_t = _tb(_crs, _CRS.from_epsg(4326), _b.left,_b.bottom,_b.right,_b.top)
                            else:
                                _l,_bo,_r,_t = _b.left,_b.bottom,_b.right,_b.top
                            _geo = {"lat_min":_bo,"lat_max":_t,"lon_min":_l,"lon_max":_r,
                                    "img_h":_h,"img_w":_w}
                    except Exception:
                        pass

                    def _px_to_latlon(row_s, col_s):
                        lat = _geo["lat_max"] - ((row_s + 25) / _geo["img_h"]) * (_geo["lat_max"] - _geo["lat_min"])
                        lon = _geo["lon_min"] + ((col_s + 25) / _geo["img_w"]) * (_geo["lon_max"] - _geo["lon_min"])
                        return round(lat, 5), round(lon, 5)

                    # ── Build Folium map ────────────────────────────
                    _center_lat = (_geo["lat_min"] + _geo["lat_max"]) / 2
                    _center_lon = (_geo["lon_min"] + _geo["lon_max"]) / 2

                    _m = folium.Map(
                        location=[_center_lat, _center_lon], zoom_start=11,
                        tiles="CartoDB dark_matter", control_scale=True,
                        max_zoom=16, min_zoom=8
                    )

                    # Also add satellite layer toggle
                    folium.TileLayer(
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri", name="🛰️ Satellite", overlay=False, control=True
                    ).add_to(_m)
                    folium.TileLayer("CartoDB dark_matter", name="🌑 Dark Base", control=True).add_to(_m)

                    _pri_color = {"High":"#e63946","Medium":"#f5a623","Low":"#4ade80","None":"#6b7280"}
                    _pri_radius = {"High":7,"Medium":5,"Low":4,"None":3}

                    # Sample up to 800 zones to keep map snappy
                    _plot_df = _idf[_idf["priority"].isin(["High","Medium"])].head(600)
                    _plot_df = pd.concat([_plot_df, _idf[_idf["priority"]=="Low"].head(200)]).reset_index(drop=True)

                    for _, _zrow in _plot_df.iterrows():
                        if "row_start" not in _zrow or "col_start" not in _zrow:
                            break
                        _lat, _lon = _px_to_latlon(int(_zrow["row_start"]), int(_zrow["col_start"]))
                        _pri = str(_zrow.get("priority","None"))
                        _cls = str(_zrow.get("dominant_class_label","—"))
                        _ndvi = round(float(_zrow.get("mean_ndvi",0)), 3)
                        _bare = round(float(_zrow.get("bare_pct",0)), 1)
                        _sp   = str(_zrow.get("species",   "—")) if "species" in _zrow else "—"
                        _sc   = str(_zrow.get("scheme",    "—")) if "scheme"  in _zrow else "—"
                        _cost = round(float(_zrow.get("est_cost_lakhs",0)),1) if "est_cost_lakhs" in _zrow else "—"
                        _soil = str(_zrow.get("soil_type","—"))
                        _rain = str(_zrow.get("rainfall_mm","—"))

                        _popup_html = f"""
                        <div style="font-family:Arial,sans-serif; font-size:12px; min-width:220px; max-width:280px;">
                            <div style="background:#0d1117; color:#4ade80; font-weight:700; padding:8px 10px;
                                        border-radius:4px 4px 0 0; font-size:13px;">Zone {int(_zrow.get('zone_id', _zrow.name))}</div>
                            <div style="background:#161b22; padding:10px;">
                                <b style="color:#e6edf3;">Land Cover:</b> <span style="color:#c9d1d9;">{_cls}</span><br>
                                <b style="color:#e6edf3;">Priority:</b>
                                    <span style="color:{_pri_color.get(_pri,'#6b7280')}; font-weight:700;">{_pri}</span><br>
                                <b style="color:#e6edf3;">NDVI:</b> <span style="color:#c9d1d9;">{_ndvi}</span> &nbsp;
                                <b style="color:#e6edf3;">Bare:</b> <span style="color:#c9d1d9;">{_bare}%</span><br>
                                <b style="color:#e6edf3;">Soil:</b> <span style="color:#c9d1d9;">{_soil}</span> &nbsp;
                                <b style="color:#e6edf3;">Rain:</b> <span style="color:#c9d1d9;">{_rain} mm</span><br>
                                <hr style="border-color:#30363d; margin:6px 0;">
                                <b style="color:#4ade80;">🌱 Species:</b><br>
                                <span style="color:#c9d1d9; font-size:11px;">{_sp}</span><br>
                                <b style="color:#4ade80;">🏛️ Scheme:</b><br>
                                <span style="color:#c9d1d9; font-size:11px;">{_sc}</span><br>
                                <b style="color:#f5a623;">💰 Est. Cost:</b>
                                <span style="color:#c9d1d9;">₹{_cost} L</span>
                            </div>
                        </div>
                        """
                        folium.CircleMarker(
                            location=[_lat, _lon],
                            radius=_pri_radius.get(_pri, 4),
                            color=_pri_color.get(_pri, "#6b7280"),
                            fill=True, fill_color=_pri_color.get(_pri, "#6b7280"),
                            fill_opacity=0.7, weight=1,
                            popup=folium.Popup(_popup_html, max_width=300),
                            tooltip=f"{_cls} · {_pri} · NDVI {_ndvi}"
                        ).add_to(_m)

                    # Legend
                    _legend = """
                    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                                background:#161b22; border:1px solid #30363d; border-radius:8px;
                                padding:12px 16px; font-family:Arial; font-size:12px; color:#e6edf3;">
                        <div style="font-weight:700; color:#4ade80; margin-bottom:6px;">🌿 GeoGreen Priority</div>
                        <div><span style="color:#e63946;">●</span> High Priority (r=7)</div>
                        <div><span style="color:#f5a623;">●</span> Medium Priority (r=5)</div>
                        <div><span style="color:#4ade80;">●</span> Low Priority (r=4)</div>
                        <div style="margin-top:6px; font-size:10px; color:#8b949e;">Click circles for zone details</div>
                    </div>
                    """
                    _m.get_root().html.add_child(folium.Element(_legend))
                    folium.LayerControl().add_to(_m)

                    _map_html = _m._repr_html_()
                    st.components.v1.html(_map_html, height=520, scrolling=False)
                    st.caption(f"🗺️ Showing {len(_plot_df):,} zones · Click any circle for full zone details · Toggle layers in top-right")

        with col_dist:
            st.markdown('<p class="section-header">📊 Land Cover Distribution</p>', unsafe_allow_html=True)

            if results.get("loaded") and "class_dist" in results:
                class_dist = results["class_dist"]

                # Colour map for classes
                colors_map = {
                    "Tree Cover":   "#2d6a4f",
                    "Shrubland":    "#74c69d",
                    "Grassland":    "#b7e4c7",
                    "Cropland":     "#f4a261",
                    "Built-up":     "#e63946",
                    "Bare / Sparse":"#a8763e",
                    "Water":        "#4895ef",
                    "No Data":      "#6b7280",
                }

                for cls, pct in sorted(class_dist.items(), key=lambda x: -x[1]):
                    if pct < 0.1:
                        continue
                    color = colors_map.get(cls, "#8b949e")
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="font-size:0.85rem; color:#c9d1d9; font-weight:500;">{cls}</span>
                            <span style="font-size:0.85rem; color:#4ade80; font-weight:700;">{pct:.1f}%</span>
                        </div>
                        <div style="background:#21262d; border-radius:4px; height:8px; overflow:hidden;">
                            <div style="width:{min(pct,100):.1f}%; height:100%; background:{color}; border-radius:4px; transition:width 0.5s;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Priority breakdown
                st.markdown('<p class="section-header">⚡ Priority Breakdown</p>', unsafe_allow_html=True)

                hp = results.get("high_priority", 0)
                mp = results.get("medium_priority", 0)
                lp = results.get("low_priority", 0)
                total = results.get("total_zones", 1)

                st.markdown(f"""
                <div style="display:flex; gap:12px; margin-bottom:16px;">
                    <div class="stat-card" style="flex:1">
                        <div class="stat-value" style="color:#ff6b6b;">{hp:,}</div>
                        <div class="stat-label">🔴 High Priority</div>
                    </div>
                    <div class="stat-card" style="flex:1">
                        <div class="stat-value" style="color:#f5a623;">{mp:,}</div>
                        <div class="stat-label">🟡 Medium</div>
                    </div>
                    <div class="stat-card" style="flex:1">
                        <div class="stat-value" style="color:#4ade80;">{lp:,}</div>
                        <div class="stat-label">🟢 Low</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ML vs Rule-based note
                st.markdown("""
                <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; margin-top:8px;">
                    <div style="font-size:0.8rem; color:#8b949e; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">🤖 AI Model Used</div>
                    <div style="font-size:0.85rem; color:#c9d1d9; line-height:1.5;">
                        <strong style="color:#4ade80;">ESA WorldCover 2021</strong> — pretrained U-Net + Random Forest ensemble.<br>
                        Trained on Sentinel-1 SAR + Sentinel-2 MSI globally.<br>
                        Resolution: <strong>10 m/pixel</strong> · Accuracy: <strong>~75.6%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.info("Run the pipeline to see land cover statistics here.")


# ─────────────────────────────────────────────────────────────────────
# TAB 3: RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────
with tab_recs:
    if not results.get("loaded"):
        st.info("📊 Run the scientific pipeline first to generate recommendations.")
    else:
        df = results["df"]
        hp  = results["high_priority"]
        mp  = results["medium_priority"]
        lp  = results["low_priority"]
        r_area    = results.get("restorable_area_km2", 0)
        total_cost = results.get("total_cost_crore")
        has_enrichment = "species" in df.columns

        # ── Top Summary Strip ───────────────────────────────────────
        cost_str = f"₹{total_cost} Cr est. budget" if total_cost else "—"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d2818,#091a10); border:1px solid #2ea043;
                    border-radius:12px; padding:20px 24px; margin-bottom:20px;">
            <div style="display:flex; gap:32px; align-items:center; flex-wrap:wrap;">
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:.05em;">High Priority</div>
                    <div style="font-size:2rem; font-weight:800; color:#ff6b6b; line-height:1;">{hp:,}</div>
                    <div style="font-size:0.72rem; color:#8b949e;">zones · {r_area} km²</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:.05em;">Medium Priority</div>
                    <div style="font-size:2rem; font-weight:800; color:#f5a623; line-height:1;">{mp:,}</div>
                    <div style="font-size:0.72rem; color:#8b949e;">zones</div>
                </div>
                <div>
                    <div style="font-size:0.7rem; color:#8b949e; text-transform:uppercase; letter-spacing:.05em;">Low Priority</div>
                    <div style="font-size:2rem; font-weight:800; color:#4ade80; line-height:1;">{lp:,}</div>
                    <div style="font-size:0.72rem; color:#8b949e;">zones</div>
                </div>
                <div style="margin-left:auto; text-align:right;">
                    <div style="font-size:0.75rem; color:#8b949e; margin-bottom:4px;">Estimated Implementation Budget</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#ffffff;">{cost_str}</div>
                    <div style="font-size:0.7rem; color:#4ade80;">High + Medium priority zones</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3 Sub-Tabs ──────────────────────────────────────────────
        sub_cards, sub_schemes, sub_table = st.tabs([
            "🌿 Action Cards + Species",
            "🏛️ Government Schemes",
            "📊 Priority Zone Table"
        ])

        icons = {"Tree Cover":"🌲","Shrubland":"🌿","Grassland":"🌾","Cropland":"🌻",
                 "Built-up":"🏘️","Bare / Sparse":"🏜️","Water":"💧","No Data":"❓"}
        badge_map = {
            "High":   '<span class="badge-high">🔴 High</span>',
            "Medium": '<span class="badge-medium">🟡 Medium</span>',
            "Low":    '<span class="badge-low">🟢 Low</span>',
            "None":   '<span class="badge-low">➖ None</span>',
        }

        # ─── Sub-Tab 1: Action Cards ──────────────────────────────
        with sub_cards:
            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                filter_priority = st.selectbox("Priority:", ["All","High","Medium","Low"], key="rc_pri")
            with col_f2:
                filter_class = st.selectbox("Land Cover:", ["All"] + sorted(df["dominant_class_label"].unique().tolist()), key="rc_cls")

            fdf = df.copy()
            if filter_priority != "All":
                fdf = fdf[fdf["priority"] == filter_priority]
            if filter_class != "All":
                fdf = fdf[fdf["dominant_class_label"] == filter_class]

            # Aggregate unique combinations
            agg_cols = ["recommendation","priority","dominant_class_label"]
            if has_enrichment:
                agg_cols += ["species","scheme","authority","intervention_type"]

            unique_recs = (fdf.groupby(agg_cols, dropna=False)
                             .agg(zone_count=("zone_id","count") if "zone_id" in fdf.columns else ("mean_ndvi","count"),
                                  est_cost_lakhs=("est_cost_lakhs","sum") if has_enrichment else ("mean_ndvi","count"))
                             .reset_index())
            priority_order = {"High":0,"Medium":1,"Low":2,"None":3}
            unique_recs["_sort"] = unique_recs["priority"].map(priority_order).fillna(99)
            unique_recs = unique_recs.sort_values(["_sort","zone_count"], ascending=[True,False])

            st.caption(f"Showing {len(unique_recs)} unique actions across **{len(fdf):,}** filtered zones")
            st.markdown("<br>", unsafe_allow_html=True)

            for _, row in unique_recs.iterrows():
                if str(row["recommendation"]) in ("No Intervention Needed", "nan"):
                    continue
                icon  = icons.get(str(row["dominant_class_label"]), "📌")
                badge = badge_map.get(str(row["priority"]), "")
                zcount = int(row["zone_count"])
                area   = round(zcount * (50 * PIXEL_SIZE_M)**2 / 1e6, 1)
                cost_l = round(float(row.get("est_cost_lakhs", 0)), 1) if has_enrichment else None

                sp_html = ""
                sc_html = ""
                if has_enrichment:
                    sp = str(row.get("species",""))
                    sc = str(row.get("scheme",""))
                    au = str(row.get("authority",""))
                    it = str(row.get("intervention_type",""))
                    if sp:
                        sp_html = f'<div style="font-size:0.78rem; color:#4ade80; margin-top:6px;">🌱 <strong>Species:</strong> {sp}</div>'
                    if sc:
                        sc_html = f'<div style="font-size:0.77rem; color:#8b949e; margin-top:3px;">🏛️ <strong>Scheme:</strong> {sc} &nbsp;·&nbsp; {au} &nbsp;·&nbsp; <em>{it}</em></div>'

                cost_html = f'<div style="font-size:0.72rem; color:#f5a623; margin-top:4px;">₹{cost_l:.1f} L est.</div>' if cost_l else ""

                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px;">
                        <div style="flex:1; min-width:0;">
                            {badge}
                            <div class="rec-title" style="margin-top:6px;">{icon} {row['dominant_class_label']} — {str(row['recommendation'])[:100]}{'...' if len(str(row['recommendation']))>100 else ''}</div>
                            {sp_html}
                            {sc_html}
                        </div>
                        <div style="text-align:right; min-width:90px; flex-shrink:0;">
                            <div style="font-size:1.25rem; font-weight:800; color:#fff;">{zcount:,}</div>
                            <div style="font-size:0.68rem; color:#8b949e;">zones · {area} km²</div>
                            {cost_html}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Download
            st.markdown("<br>", unsafe_allow_html=True)
            dl_buf = io.BytesIO()
            with pd.ExcelWriter(dl_buf, engine="openpyxl") as w:
                fdf.to_excel(w, index=False, sheet_name="Recommendations")
            st.download_button("⬇️ Download Filtered Recommendations (Excel)", dl_buf.getvalue(),
                               "geogreen_recommendations.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_cards")

        # ─── Sub-Tab 2: Government Schemes ────────────────────────
        with sub_schemes:
            if not has_enrichment:
                st.info("Enrichment data not available. Check that analysis.py has enrich_zone_recommendations().")
            else:
                st.markdown("""
                <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                            padding:14px 18px; margin-bottom:20px; font-size:0.83rem; color:#8b949e; line-height:1.5;">
                    Each intervention zone is linked to a <strong style="color:#c9d1d9;">central or state government scheme</strong>
                    that can fund the activity. Budgets shown are standard GoI rates — actual disbursements subject to DPR approval.
                </div>
                """, unsafe_allow_html=True)

                # Aggregate by scheme
                scheme_agg = (df[df["priority"].isin(["High","Medium"])]
                              .groupby(["scheme","authority","intervention_type"])
                              .agg(zones=("zone_id","count") if "zone_id" in df.columns else ("mean_ndvi","count"),
                                   total_cost_inr=("est_cost_inr","sum"),
                                   avg_rate=("rate_per_ha","mean"),
                                   total_area_ha=("zone_area_ha","sum"))
                              .reset_index()
                              .sort_values("total_cost_inr", ascending=False))

                scheme_colors = [
                    "#2ea043","#238636","#3fb950","#1f6feb","#0969da",
                    "#f78166","#d29922","#58a6ff","#bc8cff","#79c0ff",
                ]

                for idx, (_, row) in enumerate(scheme_agg.iterrows()):
                    color = scheme_colors[idx % len(scheme_colors)]
                    zones = int(row["zones"])
                    cost_cr = round(float(row["total_cost_inr"]) / 1e7, 2)
                    area_ha = round(float(row["total_area_ha"]), 0)
                    rate    = int(row["avg_rate"])

                    st.markdown(f"""
                    <div style="background:#161b22; border:1px solid {color}33; border-left:4px solid {color};
                                border-radius:10px; padding:18px 20px; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                            <div style="flex:1; min-width:200px;">
                                <div style="font-size:1rem; font-weight:700; color:#ffffff; margin-bottom:4px;">
                                    🏛️ {row['scheme']}
                                </div>
                                <div style="font-size:0.78rem; color:#8b949e; margin-bottom:6px;">
                                    <strong style="color:#c9d1d9;">Authority:</strong> {row['authority']} &nbsp;·&nbsp;
                                    <strong style="color:#c9d1d9;">Type:</strong> {row['intervention_type']}
                                </div>
                                <div style="font-size:0.75rem; color:{color};">
                                    Standard Rate: ₹{rate:,}/hectare
                                </div>
                            </div>
                            <div style="display:flex; gap:20px; flex-shrink:0; text-align:center;">
                                <div>
                                    <div style="font-size:1.3rem; font-weight:800; color:#ffffff;">{zones:,}</div>
                                    <div style="font-size:0.68rem; color:#8b949e;">Zones</div>
                                </div>
                                <div>
                                    <div style="font-size:1.3rem; font-weight:800; color:#ffffff;">{int(area_ha):,}</div>
                                    <div style="font-size:0.68rem; color:#8b949e;">Hectares</div>
                                </div>
                                <div>
                                    <div style="font-size:1.3rem; font-weight:800; color:{color};">₹{cost_cr}</div>
                                    <div style="font-size:0.68rem; color:#8b949e;">Crore est.</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grand total
                grand_total = scheme_agg["total_cost_inr"].sum()
                st.markdown(f"""
                <div style="background:rgba(46,160,67,0.1); border:1px solid #2ea043; border-radius:10px;
                            padding:16px 20px; margin-top:8px; display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:0.9rem; font-weight:700; color:#4ade80;">💰 Total Estimated Budget (High + Medium Priority)</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#ffffff;">₹{grand_total/1e7:.1f} Crore</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("💡 Note: Costs are indicative based on GoI CAMPA/GIM/PMKSY standard rates. Actual costs subject to site DPR and state matching funds.")

                # Download scheme summary
                dl_scheme = io.BytesIO()
                with pd.ExcelWriter(dl_scheme, engine="openpyxl") as w:
                    scheme_agg.to_excel(w, index=False, sheet_name="Scheme Summary")
                st.download_button("⬇️ Download Scheme Budget Summary (Excel)", dl_scheme.getvalue(),
                                   "geogreen_scheme_budget.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="dl_schemes")

        # ─── Sub-Tab 3: Priority Zone Table ──────────────────────
        with sub_table:
            st.markdown("""
            <div style="font-size:0.85rem; color:#8b949e; margin-bottom:16px; line-height:1.5;">
                Sorted by <strong style="color:#c9d1d9;">Priority Score</strong> — a composite of priority class,
                NDVI deficit, and bare land fraction. Use this table to identify the <em>most urgent</em>
                specific zones for on-ground survey and intervention.
            </div>
            """, unsafe_allow_html=True)

            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            with t_col1:
                t_pri = st.selectbox("Priority:", ["High + Medium","High Only","All"], key="tbl_pri")
            with t_col2:
                t_cls = st.selectbox("Land Cover:", ["All"] + sorted(df["dominant_class_label"].unique().tolist()), key="tbl_cls")
            with t_col3:
                t_top = st.selectbox("Show top:", [25, 50, 100, 200, "All"], key="tbl_top")
            with t_col4:
                t_sort = st.selectbox("Sort By:", ["Priority Score", "Bare Land %", "NDVI Health", "Estimated Cost"], key="tbl_srt")

            tdf = df.copy()
            if t_pri == "High + Medium":
                tdf = tdf[tdf["priority"].isin(["High","Medium"])]
            elif t_pri == "High Only":
                tdf = tdf[tdf["priority"] == "High"]
            if t_cls != "All":
                tdf = tdf[tdf["dominant_class_label"] == t_cls]

            # Apply custom outer sorting
            if t_sort == "Priority Score" and "priority_score" in tdf.columns:
                tdf = tdf.sort_values("priority_score", ascending=False)
            elif t_sort == "NDVI Health" and "mean_ndvi" in tdf.columns:
                tdf = tdf.sort_values("mean_ndvi", ascending=True)
            elif t_sort == "Bare Land %" and "bare_pct" in tdf.columns:
                tdf = tdf.sort_values("bare_pct", ascending=False)
            elif t_sort == "Estimated Cost" and "est_cost_lakhs" in tdf.columns:
                tdf = tdf.sort_values("est_cost_lakhs", ascending=False)
            else:
                sort_map = {"High":0,"Medium":1,"Low":2,"None":3}
                tdf["_sort"] = tdf["priority"].map(sort_map).fillna(99)
                tdf = tdf.sort_values(["_sort","bare_pct"], ascending=[True,False])

            if t_top != "All":
                tdf = tdf.head(int(t_top))

            # Select display columns
            display_cols = ["zone_id","dominant_class_label","priority","soil_type",
                            "rainfall_mm","mean_ndvi","bare_pct","water_pct"]
            if has_enrichment:
                display_cols += ["species","scheme","est_cost_lakhs","priority_score"]

            display_cols = [c for c in display_cols if c in tdf.columns]
            display_df = tdf[display_cols].copy()

            # Rename for readability
            rename = {
                "zone_id": "Zone",
                "dominant_class_label": "Land Cover",
                "priority": "Priority",
                "soil_type": "Soil",
                "rainfall_mm": "Rainfall mm",
                "mean_ndvi": "NDVI",
                "bare_pct": "Bare Pct",
                "water_pct": "Water Pct",
                "species": "Recommended Species",
                "scheme": "Govt Scheme",
                "est_cost_lakhs": "Est Cost (L)",
                "priority_score": "Score",
            }
            display_df = display_df.rename(columns={k:v for k,v in rename.items() if k in display_df.columns})

            # Round floats
            for col in ["NDVI", "Bare Pct", "Water Pct", "Score"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(3)
            if "Est Cost (L)" in display_df.columns:
                display_df["Est Cost (L)"] = display_df["Est Cost (L)"].round(2)

            # Render as a clean, native Streamlit dataframe
            col_cfg = {
                "Zone": st.column_config.NumberColumn("Zone", width="small"),
                "Land Cover": st.column_config.TextColumn("Land Cover", width="medium"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "NDVI": st.column_config.NumberColumn("NDVI", format="%.3f", width="small"),
                "Bare Pct": st.column_config.NumberColumn("Bare Pct", format="%.1f", width="small"),
                "Water Pct": st.column_config.NumberColumn("Water Pct", format="%.1f", width="small"),
                "Score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
                "Est Cost (L)": st.column_config.NumberColumn("Est Cost (L)", format="%.2f", width="medium"),
                "Recommended Species": st.column_config.TextColumn("Recommended Species", width="large"),
                "Govt Scheme": st.column_config.TextColumn("Govt Scheme", width="large"),
                "Soil": st.column_config.TextColumn("Soil", width="medium"),
                "Rainfall mm": st.column_config.NumberColumn("Rainfall mm", format="%d", width="small"),
            }
            # only pass configs that exist in this df
            col_cfg = {k: v for k, v in col_cfg.items() if k in display_df.columns}

            st.dataframe(
                display_df,
                use_container_width=True,
                height=520,
                hide_index=True,
                column_config=col_cfg,
            )

            st.caption(f"Showing {len(display_df):,} zones  |  "
                       + (f"Total est. cost: Rs {tdf['est_cost_lakhs'].sum():.1f} Lakhs" if has_enrichment and "est_cost_lakhs" in tdf.columns else ""))

            # Download
            dl_tbl = io.BytesIO()
            with pd.ExcelWriter(dl_tbl, engine="openpyxl") as w:
                display_df.to_excel(w, index=False, sheet_name="Priority Zones")
            st.download_button("⬇️ Download Zone Table (Excel)", dl_tbl.getvalue(),
                               "geogreen_priority_zones.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_table")



# ─────────────────────────────────────────────────────────────────────
# TAB 3: LIVE DEMO (CV Mode)
# ─────────────────────────────────────────────────────────────────────
with tab_analyze:
    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; margin-bottom:20px;">
        <div style="font-size:0.85rem; color:#8b949e; margin-bottom:6px;">ℹ️ <strong style="color:#c9d1d9;">Live Demo Mode</strong></div>
        <div style="font-size:0.83rem; color:#8b949e; line-height:1.5;">
            Upload any satellite image (PNG, JPG, or TIF) to instantly run AI-powered land cover classification using computer vision.
            Works best with Google Earth / Google Maps screenshots or natural-colour satellite imagery.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_sidebar, col_main = st.columns([1, 2])

    with col_sidebar:
        uploaded_files = st.file_uploader(
            "Upload satellite images (PNG/JPG/TIF)",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            accept_multiple_files=True,
            key="live_upload"
        )
        st.markdown("---")
        st.subheader("📍 Coordinates (optional)")
        st.caption("Excel/CSV with columns: **file_name**, **lat**, **long**")
        coord_file = st.file_uploader("Upload coordinates", type=['xlsx','xls','csv'], key="coord_file")
        st.markdown("---")
        n_clusters = st.slider("Segmentation Clusters (K)", 3, 8, 5)

        # Legend
        st.markdown("""
        **Colour Legend:**
        <div style="font-size:0.82rem; line-height:1.8; color:#c9d1d9;">
        🟩 <strong style="color:#4ade80;">Green</strong> outline — Vegetation<br>
        🟫 <strong style="color:#cd7f32;">Brown</strong> outline — Bare/Sparse land<br>
        🟥 <strong style="color:#ff4444;">Red</strong> outline — Built-up/Urban<br>
        🟦 <strong style="color:#4895ef;">Blue</strong> outline — Water bodies
        </div>
        """, unsafe_allow_html=True)

    with col_main:
        # Parse coordinates
        def _parse_coord(val):
            s = str(val).strip()
            parts = s.split('.')
            if len(parts) == 3:
                return float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
            return float(s)

        coord_lookup = {}
        if coord_file is not None:
            try:
                if coord_file.name.endswith('.csv'):
                    coord_df = pd.read_csv(coord_file, dtype=str)
                else:
                    coord_df = pd.read_excel(coord_file, dtype=str)
                coord_df.columns = [c.strip().lower().replace(' ', '_') for c in coord_df.columns]
                fname_col = next((c for c in ['file_name','filename','image','name','file'] if c in coord_df.columns), None)
                lat_col = next((c for c in ['lat','latitude','y'] if c in coord_df.columns), None)
                lon_col = next((c for c in ['long','lon','longitude','lng','x'] if c in coord_df.columns), None)
                if fname_col and lat_col and lon_col:
                    for _, row in coord_df.iterrows():
                        coord_lookup[str(row[fname_col]).strip()] = (_parse_coord(row[lat_col]), _parse_coord(row[lon_col]))
                    st.success(f"✅ Loaded {len(coord_lookup)} coordinate entries")
            except Exception as e:
                st.error(f"Failed to read coordinates: {e}")

        if not uploaded_files:
            st.markdown("""
            <div style="border:2px dashed #30363d; border-radius:12px; padding:60px; text-align:center;">
                <div style="font-size:2rem; margin-bottom:12px;">🛰️</div>
                <div style="color:#8b949e; font-size:0.9rem;">Upload satellite images from the left panel to begin analysis</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for uploaded_file in uploaded_files:
                with st.expander(f"📷 {uploaded_file.name}", expanded=True):
                    c1, c2 = st.columns([1, 2])

                    with c1:
                        st.subheader("Input")
                        try:
                            from PIL import Image
                            image = Image.open(uploaded_file)
                            image.thumbnail((800, 800))
                            st.image(image, use_container_width=True)
                        except Exception as e:
                            st.info("Preview unavailable for this large/specialized file.")

                        ext = os.path.splitext(uploaded_file.name)[1].lower() or ".png"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        fname = uploaded_file.name
                        lat, lon = None, None
                        if fname in coord_lookup:
                            lat, lon = coord_lookup[fname]
                            st.success(f"📍 {lat:.4f}°N, {lon:.4f}°E")
                        else:
                            base = os.path.splitext(fname)[0]
                            for key in coord_lookup:
                                if os.path.splitext(key)[0] == base:
                                    lat, lon = coord_lookup[key]
                                    st.success(f"📍 {lat:.4f}°N, {lon:.4f}°E")
                                    break

                        if st.button(f"🔍 Analyze", key=f"btn_{fname}"):
                            with st.spinner("🤖 Running classification…"):
                                try:
                                    cv_stats, seg_mask, overlay = analyze_land_cover(tmp_path, n_clusters=n_clusters)
                                    climate_info = {}
                                    if lat is not None:
                                        climate_info = get_climate_data(lat, lon)
                                    if not climate_info:
                                        climate_info = {"annual_rainfall_mm": 1000, "mean_temp_c": 25.0, "location_ok": False}
                                    recs = generate_rgb_recommendations(cv_stats, climate_info)
                                    st.session_state[f"res_{fname}"] = {
                                        "cv_stats": cv_stats, "seg_mask": seg_mask, "overlay": overlay,
                                        "lat": lat, "lon": lon, "climate": climate_info, "recs": recs,
                                    }
                                except Exception as e:
                                    st.error(f"Analysis failed: {e}"); st.exception(e)
                                finally:
                                    try: os.remove(tmp_path)
                                    except: pass

                    with c2:
                        if f"res_{fname}" in st.session_state:
                            res = st.session_state[f"res_{fname}"]
                            st.subheader("🗺️ Results")

                            r_tab1, r_tab2, r_tab3 = st.tabs(["🎨 Maps", "🌦️ Climate", "📋 Actions"])

                            with r_tab1:
                                rc1, rc2 = st.columns(2)
                                with rc1:
                                    st.image(res["overlay"], caption="Precision Contour Classification", use_container_width=True)
                                with rc2:
                                    st.image(res["seg_mask"], caption="Segmentation Mask", use_container_width=True)

                                st.markdown("**Land Cover Statistics**")
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("🌿 Vegetation", f"{res['cv_stats'].get('Vegetation',0):.1f}%")
                                m2.metric("💧 Water", f"{res['cv_stats'].get('Water',0):.1f}%")
                                m3.metric("🏜️ Bare/Sparse", f"{res['cv_stats'].get('Bare/Sparse',0):.1f}%")
                                m4.metric("🏘️ Built-up", f"{res['cv_stats'].get('Built-up',0):.1f}%")

                            with r_tab2:
                                if res["lat"]:
                                    st.markdown(f"**Location:** {res['lat']:.4f}°N, {res['lon']:.4f}°E")
                                    st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}), zoom=10)
                                    clim = res["climate"]
                                    if clim.get("location_ok"):
                                        cm1, cm2 = st.columns(2)
                                        cm1.metric("Annual Rainfall", f"{clim['annual_rainfall_mm']} mm")
                                        cm2.metric("Avg Temperature", f"{clim['mean_temp_c']} °C")
                                    else:
                                        st.info("Using default climate values")
                                else:
                                    st.info("Upload a coordinates file to enable climate & map view")

                            with r_tab3:
                                for rec in res["recs"]:
                                    pc = "red" if rec["priority"] == "High" else "orange" if rec["priority"] == "Medium" else "green"
                                    st.markdown(f"**:{pc}[{rec['priority']}]** — {rec['action']}")
                                    st.caption(rec["reason"])
                                    st.markdown("---")

                                excel_buf = io.BytesIO()
                                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                                    pd.DataFrame(res["recs"]).to_excel(writer, index=False, sheet_name="Recommendations")
                                    pd.DataFrame([res["cv_stats"]]).to_excel(writer, index=False, sheet_name="Land Cover Stats")
                                st.download_button("⬇️ Download Report (Excel)", excel_buf.getvalue(),
                                                   file_name=f"report_{fname}.xlsx",
                                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ─────────────────────────────────────────────────────────────────────
# TAB 4: FULL REPORT
# ─────────────────────────────────────────────────────────────────────
with tab_report:
    if results.get("summary_text"):
        st.markdown("""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:20px; margin-bottom:20px;">
            <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase; font-weight:600; letter-spacing:0.05em; margin-bottom:8px;">📄 Auto-Generated Analysis Report</div>
            <div style="font-size:0.82rem; color:#4ade80;">Generated from real Sentinel-2 satellite data · Sehore District, Madhya Pradesh</div>
        </div>
        """, unsafe_allow_html=True)
        st.code(results["summary_text"], language=None)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            report_path = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button("⬇️ Download Summary Report (.txt)", f, "geogreen_summary.txt", "text/plain")
        with col_dl2:
            rec_csv_path = os.path.join(OUTPUT_DIR, "recommendations.csv")
            if os.path.exists(rec_csv_path) and results.get("loaded"):
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                    results["df"].to_excel(writer, index=False, sheet_name="Zone Recommendations")
                st.download_button("⬇️ Download All Zones (.xlsx)", excel_buf.getvalue(), "geogreen_all_zones.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("📄 Run the scientific pipeline (`python src/main.py`) to generate the full report.")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center; padding:24px; border-top:1px solid #21262d; margin-top:32px;">
    <div style="font-size:0.8rem; color:#8b949e;">
        GeoGreen Revolution AI · Pilot Region: Sehore District, Madhya Pradesh, India<br>
        ML Model: <a href="https://esa-worldcover.org/" target="_blank" style="color:#4ade80;">ESA WorldCover 2021</a> 
        (U-Net + Random Forest, CC BY 4.0) · Satellite Data: Copernicus Sentinel-2 L2A<br>
        <em>Built for a Greener India 🌿</em>
    </div>
</div>
""", unsafe_allow_html=True)
