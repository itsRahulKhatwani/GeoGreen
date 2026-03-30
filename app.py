import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import sys
import io

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
            "Upload PNG / JPG / TIF images",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
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
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if not ext: ext = ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
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

                            # Generate Excel Report
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                pd.DataFrame(res["recs"]).to_excel(writer, index=False, sheet_name="Recommendations")
                                pd.DataFrame([res["cv_stats"]]).to_excel(writer, index=False, sheet_name="Land Cover Stats")
                            
                            st.download_button(
                                "⬇️ Download Excel Report", 
                                excel_buffer.getvalue(),
                                file_name=f"report_{fname}.xlsx", 
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

else:
    # ── SCIENTIFIC MODE (New Logic) ──
    st.header("🔬 Scientific Analysis Results (Mode B)")
    st.markdown("Run the **ESA WorldCover Deep Learning Model** pipeline natively.")

    # ── Band Stacking Helper ──────────────────────────────────────
    def stack_band_files(band_files_dict):
        """
        Stack individual single-band GeoTIFF files into one multi-band GeoTIFF.

        Parameters
        ----------
        band_files_dict : dict
            Ordered dict of label -> UploadedFile, e.g.
            {"B02 (Blue)": <file>, "B03 (Green)": <file>, ...}
            None values are skipped (optional bands).

        Returns
        -------
        str : Path to the temporary stacked GeoTIFF.
        """
        import rasterio
        from rasterio.enums import Resampling
        import numpy as np

        # Filter out None entries
        valid_bands = [(label, f) for label, f in band_files_dict.items() if f is not None]
        if not valid_bands:
            raise ValueError("No band files provided.")

        # Save each uploaded band to a temp file and open with rasterio
        tmp_band_paths = []
        for label, uploaded in valid_bands:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_band_paths.append(tmp.name)

        # Read first band to get reference shape & profile
        with rasterio.open(tmp_band_paths[0]) as ref:
            ref_shape = (ref.height, ref.width)
            ref_profile = ref.profile.copy()

        # Build output profile for N-band stack
        out_profile = ref_profile.copy()
        out_profile.update(count=len(valid_bands), dtype="float32")

        # Write stacked output
        stacked_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        stacked_path = stacked_tmp.name
        stacked_tmp.close()

        with rasterio.open(stacked_path, "w", **out_profile) as dst:
            for i, (band_path, (label, _)) in enumerate(
                zip(tmp_band_paths, valid_bands), start=1
            ):
                with rasterio.open(band_path) as src:
                    if (src.height, src.width) == ref_shape:
                        data = src.read(1).astype("float32")
                    else:
                        # Resample to match reference dimensions
                        # out_shape for single-band read must be (1, rows, cols)
                        data = src.read(
                            1,
                            out_shape=(1, ref_shape[0], ref_shape[1]),
                            resampling=Resampling.bilinear,
                        ).squeeze().astype("float32")
                    dst.write(data, i)

        # Cleanup temp individual band files
        for p in tmp_band_paths:
            try:
                os.remove(p)
            except Exception:
                pass

        return stacked_path
    # ─────────────────────────────────────────────────────────────

    with st.expander("📂 Run New Analysis", expanded=True):

        input_tab1, input_tab2 = st.tabs([
            "📦 Single Stacked .tif",
            "🗂️ Individual Bands (B02 / B03 / B04 / B08 / B11)"
        ])

        # ── Tab 1: Single stacked TIF (existing behaviour) ────────
        with input_tab1:
            st.info("Upload a pre-stacked multi-band Sentinel-2 `.tif` and a climate `.csv`.")
            col1, col2, col3 = st.columns(3)
            with col1:
                sci_sat_file = st.file_uploader(
                    "Satellite Image (Sentinel-2 .tif)",
                    type=["tif", "tiff"], key="sci_sat"
                )
            with col2:
                sci_wc_file = st.file_uploader(
                    "ESA WorldCover Map (.tif — optional)",
                    type=["tif", "tiff"], key="sci_wc"
                )
            with col3:
                sci_clim_file = st.file_uploader(
                    "Climate Data (.csv)",
                    type=["csv"], key="sci_clim"
                )

            if st.button("🚀 Run AI Pipeline", type="primary", key="run_single"):
                if sci_sat_file is None or sci_clim_file is None:
                    st.error("❌ Satellite `.tif` and Climate `.csv` are required.")
                else:
                    with st.spinner("🤖 Running AI Pipeline… Please wait."):
                        try:
                            from src.main import main as run_pipeline

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f1:
                                f1.write(sci_sat_file.getbuffer())
                                sat_path = f1.name
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f2:
                                f2.write(sci_clim_file.getbuffer())
                                clim_path = f2.name

                            wc_path = None
                            if sci_wc_file:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f3:
                                    f3.write(sci_wc_file.getbuffer())
                                    wc_path = f3.name

                            run_pipeline(
                                satellite_path=sat_path,
                                climate_path=clim_path,
                                worldcover_path=wc_path,
                                grid_size=50,
                            )
                            st.success("✅ Analysis Complete! Scroll down for results.")
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")
                            st.exception(e)

        # ── Tab 2: Individual band uploads → auto-stack ────────────
        with input_tab2:
            st.info(
                "Upload each Sentinel-2 band as a **separate single-band `.tif`** file. "
                "They will be automatically stacked into a multi-band image before the pipeline runs.\n\n"
                "**Required:** B02, B03, B04, B08 &nbsp;|&nbsp; **Optional:** B11 (SWIR)"
            )

            st.markdown("#### 📡 Sentinel-2 Band Files")
            bc1, bc2, bc3 = st.columns(3)
            bd1, bd2 = st.columns(2)

            with bc1:
                b02_file = st.file_uploader("B02 — Blue (490 nm) ✱", type=["tif", "tiff"], key="b02")
            with bc2:
                b03_file = st.file_uploader("B03 — Green (560 nm) ✱", type=["tif", "tiff"], key="b03")
            with bc3:
                b04_file = st.file_uploader("B04 — Red (665 nm) ✱", type=["tif", "tiff"], key="b04")
            with bd1:
                b08_file = st.file_uploader("B08 — NIR (842 nm) ✱", type=["tif", "tiff"], key="b08")
            with bd2:
                b11_file = st.file_uploader("B11 — SWIR (1610 nm) optional", type=["tif", "tiff"], key="b11")

            st.markdown("#### 🗺️ Supporting Files")
            be1, be2 = st.columns(2)
            with be1:
                bands_wc_file = st.file_uploader(
                    "ESA WorldCover Map (.tif — optional)",
                    type=["tif", "tiff"], key="bands_wc"
                )
            with be2:
                bands_clim_file = st.file_uploader(
                    "Climate Data (.csv) ✱",
                    type=["csv"], key="bands_clim"
                )

            # Status indicators
            required_bands = {"B02": b02_file, "B03": b03_file, "B04": b04_file, "B08": b08_file}
            missing = [k for k, v in required_bands.items() if v is None]
            if missing:
                st.warning(f"⚠️ Still needed: **{', '.join(missing)}** band(s) and Climate CSV")
            else:
                st.success("✅ All required bands uploaded — ready to stack & run!")

            if st.button("🔗 Stack Bands & Run AI Pipeline", type="primary", key="run_bands"):
                if any(v is None for v in required_bands.values()):
                    st.error(f"❌ Required bands missing: {', '.join(missing)}")
                elif bands_clim_file is None:
                    st.error("❌ Climate `.csv` is required.")
                else:
                    stacked_path = None  # ensure always defined
                    with st.spinner("🔗 Stacking bands into multi-band GeoTIFF…"):
                        try:
                            # Stack bands in order: B02, B03, B04, B08, [B11]
                            band_files_ordered = {
                                "B02 (Blue)":  b02_file,
                                "B03 (Green)": b03_file,
                                "B04 (Red)":   b04_file,
                                "B08 (NIR)":   b08_file,
                                "B11 (SWIR)":  b11_file,   # None if not uploaded → skipped
                            }
                            stacked_path = stack_band_files(band_files_ordered)

                            n_bands = 4 + (1 if b11_file else 0)
                            st.success(
                                f"✅ Stacked {n_bands} bands into a single GeoTIFF. "
                                "Running AI pipeline…"
                            )
                        except Exception as e:
                            st.error(f"Band stacking failed: {e}")
                            st.exception(e)
                            stacked_path = None

                    if stacked_path:
                        with st.spinner("🤖 Running AI Pipeline… Please wait."):
                            try:
                                from src.main import main as run_pipeline

                                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as fc:
                                    fc.write(bands_clim_file.getbuffer())
                                    clim_path = fc.name

                                wc_path = None
                                if bands_wc_file:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as fw:
                                        fw.write(bands_wc_file.getbuffer())
                                        wc_path = fw.name

                                run_pipeline(
                                    satellite_path=stacked_path,
                                    climate_path=clim_path,
                                    worldcover_path=wc_path,
                                    grid_size=50,
                                )
                                st.success("✅ Analysis Complete! Scroll down for results.")
                            except Exception as e:
                                st.error(f"Pipeline failed: {e}")
                                st.exception(e)
                            finally:
                                # Cleanup stacked temp file
                                try:
                                    os.remove(stacked_path)
                                except Exception:
                                    pass
    
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
                # Read the CSV and convert it to Excel format for download
                try:
                    df_recs = pd.read_csv(rec_csv)
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        df_recs.to_excel(writer, index=False, sheet_name="Scientific Recommendations")
                        
                    st.download_button(
                        label="⬇️ Download Recommendations (Excel)",
                        data=excel_buffer.getvalue(),
                        file_name="scientific_recommendations.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    # Fallback to CSV if conversion fails somehow
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
