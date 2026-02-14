import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys

# Add src to sys.path for internal imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Fix for OpenMP DLL conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Backend imports
from src.cv_analysis import analyze_land_cover
from src.climate_api import get_climate_data
from src.analysis import generate_rgb_recommendations

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoGreen Revolution AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #2e7d32, #43a047);
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
    border: none;
}
.stButton>button:hover { background: linear-gradient(135deg, #1b5e20, #2e7d32); }
h1, h2, h3 { color: #1b5e20; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────
st.title("🌿 GeoGreen Revolution AI Dashboard")
st.markdown("**AI-Powered Land Cover Analysis & Greening Recommendations**")
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────────────
# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Analysis Mode")
    mode = st.radio(
        "Choose Mode:",
        ["🚀 Live Demo (CV)", "🔬 Scientific Results (Mode B)"],
        index=0
    )
    st.markdown("---")

    if mode == "🚀 Live Demo (CV)":
        st.header("📂 Data Input")

        # 1. Image upload
        st.subheader("Satellite Images")
        uploaded_files = st.file_uploader(
            "Upload PNG / JPG images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )

        st.markdown("---")

        # 2. Coordinate Excel upload
        st.subheader("📍 Coordinates File")
        st.caption("Excel/CSV with columns: **file_name**, **lat**, **long**")
        coord_file = st.file_uploader(
            "Upload coordinates file",
            type=['xlsx', 'xls', 'csv'],
            key="coord_file"
        )
        
        # ... (keep existing coordinate parsing logic) ...
        def _parse_coord(val):
            """
            Parse a coordinate that may be in:
              - Decimal degrees:  23.25
              - DMS with dots:    23.14.46  (= 23° 14' 46")
            """
            s = str(val).strip()
            parts = s.split('.')
            if len(parts) == 3:
                # DD.MM.SS format
                deg = float(parts[0])
                mins = float(parts[1])
                secs = float(parts[2])
                return deg + mins / 60.0 + secs / 3600.0
            elif len(parts) <= 2:
                # Normal decimal
                return float(s)
            else:
                raise ValueError(f"Cannot parse coordinate: {s}")

        coord_lookup = {}
        if coord_file is not None:
            try:
                if coord_file.name.endswith('.csv'):
                    coord_df = pd.read_csv(coord_file, dtype=str)
                else:
                    coord_df = pd.read_excel(coord_file, dtype=str)

                # Normalize column names
                coord_df.columns = [c.strip().lower().replace(' ', '_') for c in coord_df.columns]

                # Build lookup: filename -> (lat, lon)
                fname_col = None
                for candidate in ['file_name', 'filename', 'image', 'name', 'file']:
                    if candidate in coord_df.columns:
                        fname_col = candidate
                        break

                lat_col = None
                for candidate in ['lat', 'latitude', 'y']:
                    if candidate in coord_df.columns:
                        lat_col = candidate
                        break

                lon_col = None
                for candidate in ['long', 'lon', 'longitude', 'lng', 'x']:
                    if candidate in coord_df.columns:
                        lon_col = candidate
                        break

                if fname_col and lat_col and lon_col:
                    for _, row in coord_df.iterrows():
                        fname = str(row[fname_col]).strip()
                        coord_lookup[fname] = (_parse_coord(row[lat_col]), _parse_coord(row[lon_col]))
                    st.success(f"✅ Loaded {len(coord_lookup)} coordinate entries")
                else:
                    st.error(f"Columns found: {list(coord_df.columns)}. Need: file_name, lat, long")
            except Exception as e:
                st.error(f"Failed to read coordinate file: {e}")

        st.markdown("---")

        # 3. Settings
        st.subheader("⚙️ Settings")
        n_clusters = st.slider("Segmentation Clusters (K)", 3, 8, 5)

    else:
        # Scientific Mode Sidebar
        st.info("ℹ️ **Scientific Mode** displays high-accuracy results generated by the backend pipeline.")
        st.markdown("""
        **To update results:**
        1. Run `python src/main.py`
        2. Refresh this page
        """)

    st.markdown("---")
    st.caption("GeoGreen Revolution v2.2 | AI-Powered")

# ── Main Content ────────────────────────────────────────────────────

if mode == "🚀 Live Demo (CV)":
    # ── LIVE DEMO MODE (Existing Logic) ──
    if not uploaded_files:
        st.info("👋 Upload satellite images and an optional coordinates file from the sidebar to begin.")
        st.markdown("""
        **How it works:**
        1. 🛰️ Upload your satellite images (PNG/JPG)
        2. 📍 Upload an Excel/CSV with coordinates (**file_name**, **lat**, **long**)
        3. 🤖 Click **Analyze** to run AI-powered land cover classification
        4. 📋 Get greening recommendations based on land cover + climate data

        **Legend:**
        | Color | Land Cover |
        |-------|-----------|
        | 🟩 Green outline | Vegetation |
        | 🟫 Brown outline | Barren / Sparse land |
        | 🟥 Red outline | Built-up / Urban |
        | 🟦 Blue outline | Water bodies |
        """)
    else:
        for uploaded_file in uploaded_files:
            with st.expander(f"📷 {uploaded_file.name}", expanded=True):

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Input Image")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)

                    # Save temp file for CV processing
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    # Coordinate lookup
                    lat, lon = None, None
                    fname = uploaded_file.name
                    if fname in coord_lookup:
                        lat, lon = coord_lookup[fname]
                        st.success(f"📍 Coords: {lat:.4f}, {lon:.4f}")
                    else:
                        # Try matching without extension
                        base = os.path.splitext(fname)[0]
                        for key in coord_lookup:
                            if os.path.splitext(key)[0] == base:
                                lat, lon = coord_lookup[key]
                                st.success(f"📍 Coords: {lat:.4f}, {lon:.4f}")
                                break
                        if lat is None:
                            st.warning("📍 No coordinates found for this image")

                    if st.button(f"🔍 Analyze", key=f"btn_{fname}"):
                        with st.spinner("🤖 Running land cover analysis..."):
                            try:
                                # 1. Computer Vision
                                cv_stats, seg_mask, overlay = analyze_land_cover(
                                    tmp_path, n_clusters=n_clusters
                                )

                                # 2. Climate Data
                                climate_info = {}
                                if lat is not None and lon is not None:
                                    climate_info = get_climate_data(lat, lon)

                                # 3. Recommendations
                                if not climate_info:
                                    climate_info = {
                                        "annual_rainfall_mm": 1000,
                                        "mean_temp_c": 25.0,
                                        "location_ok": False
                                    }
                                recs = generate_rgb_recommendations(cv_stats, climate_info)

                                # Store results
                                st.session_state[f"res_{fname}"] = {
                                    "cv_stats": cv_stats,
                                    "seg_mask": seg_mask,
                                    "overlay": overlay,
                                    "lat": lat,
                                    "lon": lon,
                                    "climate": climate_info,
                                    "recs": recs,
                                }
                            except Exception as e:
                                st.error(f"Analysis failed: {e}")
                                st.exception(e)
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)

                # ── Display Results ─────────────────────────────────────
                if f"res_{fname}" in st.session_state:
                    res = st.session_state[f"res_{fname}"]

                    with col2:
                        st.subheader("🗺️ Analysis Results")

                        tab1, tab2, tab3 = st.tabs([
                            "🎨 Visualization", "🌦️ Location & Climate", "📋 Recommendations"
                        ])

                        with tab1:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.image(res["overlay"],
                                         caption="Precision Contour Classification",
                                         use_container_width=True)
                            with c2:
                                st.image(res["seg_mask"],
                                         caption="Segmentation Mask",
                                         use_container_width=True)

                            # Metrics
                            st.markdown("#### Land Cover Statistics")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("🌿 Vegetation", f"{res['cv_stats'].get('Vegetation',0):.1f}%")
                            m2.metric("💧 Water", f"{res['cv_stats'].get('Water',0):.1f}%")
                            m3.metric("🏜️ Bare/Sparse", f"{res['cv_stats'].get('Bare/Sparse',0):.1f}%")
                            m4.metric("🏘️ Built-up", f"{res['cv_stats'].get('Built-up',0):.1f}%")

                        with tab2:
                            if res["lat"] and res["lon"]:
                                st.markdown(f"**Location:** {res['lat']:.4f}°N, {res['lon']:.4f}°E")
                                st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}), zoom=10)

                                clim = res["climate"]
                                if clim.get("location_ok"):
                                    st.markdown("#### 🌦️ Historical Climate (5-yr Average)")
                                    cm1, cm2 = st.columns(2)
                                    cm1.metric("Annual Rainfall", f"{clim['annual_rainfall_mm']} mm")
                                    cm2.metric("Avg Temperature", f"{clim['mean_temp_c']} °C")
                                else:
                                    st.info("Using default climate values (no API data)")
                            else:
                                st.info("Upload a coordinates file to enable climate analysis and map view.")

                        with tab3:
                            st.subheader("📋 Greening Action Plan")
                            for rec in res["recs"]:
                                pc = "red" if rec["priority"] == "High" else "orange" if rec["priority"] == "Medium" else "green"
                                st.markdown(f"**:{pc}[{rec['priority']}]** — {rec['action']}")
                                st.caption(rec["reason"])
                                st.markdown("---")

                            csv = pd.DataFrame(res["recs"]).to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Download Report CSV", csv,
                                               file_name=f"report_{fname}.csv", mime="text/csv")

else:
    # ── SCIENTIFIC MODE (New Logic) ──
    st.header("🔬 Scientific Analysis Results (Mode B)")
    st.markdown("Displays outputs from the **ESA WorldCover Deep Learning Model** pipeline.")
    
    # Path to output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    if not os.path.exists(OUTPUT_DIR) or not os.listdir(OUTPUT_DIR):
        st.warning("⚠️ No scientific results found!")
        st.markdown("""
        **How to generate results:**
        1. Open your terminal.
        2. Run: `python src/main.py`
        3. Refresh this page.
        """)
    else:
        # Check for key files
        ml_map = os.path.join(OUTPUT_DIR, "ml_landcover_map.png")
        rec_map = os.path.join(OUTPUT_DIR, "recommendation_map.png")
        fused_map = os.path.join(OUTPUT_DIR, "fused_landcover_map.png")
        ndvi_map = os.path.join(OUTPUT_DIR, "ndvi_map.png")
        report_txt = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
        rec_csv = os.path.join(OUTPUT_DIR, "recommendations.csv")

        tab_main, tab_report, tab_raw = st.tabs(["🌍 Maps & Recommendations", "📊 Full Report", "📥 Downloads"])
        
        with tab_main:
            col_a, col_b = st.columns(2)
            
            with col_a:
                if os.path.exists(ml_map):
                    st.image(ml_map, caption="AI Land-Cover Classification (ESA WorldCover)", use_container_width=True)
                else:
                    st.info("ML Map not found (Run pipeline in ML mode)")
                
                if os.path.exists(ndvi_map):
                    st.image(ndvi_map, caption="Vegetation Health (NDVI)", use_container_width=True)

            with col_b:
                if os.path.exists(rec_map):
                    st.image(rec_map, caption="Greening & Intervention Priority", use_container_width=True)
                
                if os.path.exists(fused_map):
                    st.image(fused_map, caption="Fused Classification (ML + Indices)", use_container_width=True)
        
        with tab_report:
            if os.path.exists(report_txt):
                with open(report_txt, "r", encoding="utf-8") as f:
                    report_content = f.read()
                st.text_area("Analysis Report", report_content, height=600)
            else:
                st.warning("Summary report not found.")
        
        with tab_raw:
            st.subheader("Download Results")
            if os.path.exists(rec_csv):
                with open(rec_csv, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Recommendation CSV",
                        data=f,
                        file_name="scientific_recommendations.csv",
                        mime="text/csv"
                    )
            
            if os.path.exists(report_txt):
                with open(report_txt, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Summary Report",
                        data=f,
                        file_name="summary_statistics.txt",
                        mime="text/plain"
                    )
