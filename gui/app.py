import os
import sys
import io
import time
import json
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import folium
from streamlit_folium import st_folium

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.infer import SurveillanceInferenceEngine
from src.generate_samples import generate_all_samples

# Page Configuration
st.set_page_config(
    page_title="AEGIS-SAT | Border Surveillance AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tactical Military Dark CSS
st.markdown("""
<style>
    /* Dark Defense Radar Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Header Banner */
    .tactical-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 5px solid #00E5FF;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.08);
    }
    .tactical-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #f8fafc;
        margin: 0;
        text-transform: uppercase;
    }
    .tactical-subtitle {
        font-size: 13px;
        color: #94a3b8;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* Metric Cards */
    .hud-card {
        background: #131c2e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.4);
    }
    .hud-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .hud-val {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }

    /* Threat Status Banners */
    .threat-banner {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        border-left: 6px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Image view containers */
    .img-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 8px;
        text-align: center;
    }

    /* Custom Streamlit adjustments */
    div[data-testid="stSidebar"] {
        background-color: #090d16;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine(checkpoint_path: str, backbone: str, threshold: float, min_cluster: int, gsd: float):
    return SurveillanceInferenceEngine(
        model_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
        backbone=backbone,
        threshold=threshold,
        min_cluster_area=min_cluster,
        gsd_meters_per_pixel=gsd
    )


# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🛰️ **AEGIS CONTROL SYSTEM**")
    st.caption("AI-Powered Bi-Temporal Satellite Surveillance")
    st.markdown("---")

    app_mode = st.radio(
        "Operating Mode",
        ["Sector Simulation (Demo Presets)", "Upload Satellite Pair"],
        index=0
    )
    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    
    filter_mode = st.selectbox(
        "False-Positive Filter Mode",
        ["High Precision (Anti-Noise)", "Balanced (Default)", "High Sensitivity (Fine-grained)"],
        index=0,
        help="High Precision eliminates shadows, natural seasonal foliage shifts, and registration jitter."
    )
    
    if filter_mode == "High Precision (Anti-Noise)":
        default_thresh = 0.55
        default_min_cluster = 35
    elif filter_mode == "Balanced (Default)":
        default_thresh = 0.45
        default_min_cluster = 20
    else:
        default_thresh = 0.35
        default_min_cluster = 10

    thresh_val = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=default_thresh, step=0.05)
    min_cluster_val = st.slider("Min Cluster Filter (px)", min_value=5, max_value=120, value=default_min_cluster, step=5)
    gsd_val = st.selectbox("Ground Resolution (GSD)", [0.5, 1.0, 2.5], index=0, format_func=lambda x: f"{x} m/pixel (High-Res)")
    match_light_val = st.checkbox("Radiometric Lighting Equalization", value=True, help="Equalizes sun shadow and atmospheric illumination differences.")
    backbone_choice = st.selectbox("Model Encoder Backbone", ["resnet34", "resnet18"], index=0)

    st.markdown("---")
    checkpoint_file = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pth")
    has_ckpt = os.path.isfile(checkpoint_file)
    if has_ckpt:
        st.success("✅ Neural Checkpoint Loaded: `best_model.pth`")
    else:
        st.info("ℹ️ Using Adaptive Bi-Temporal Feature Engine")

    if st.button("🔄 Regenerate Sample Sectors"):
        generate_all_samples(output_dir=os.path.join(PROJECT_ROOT, "data", "samples"))
        st.success("Samples refreshed!")

# Load Inference Engine
engine = load_engine(
    checkpoint_path=checkpoint_file,
    backbone=backbone_choice,
    threshold=thresh_val,
    min_cluster=min_cluster_val,
    gsd=gsd_val
)
# Update dynamic properties
engine.postprocessor.min_cluster_area = min_cluster_val
engine.postprocessor.gsd = gsd_val
engine.threshold = thresh_val

# Top Header
st.markdown("""
<div class="tactical-header">
    <div class="tactical-title">🛰️ SATELLITE BORDER SURVEILLANCE SYSTEM</div>
    <div class="tactical-subtitle">DEEP SIAMESE U-NET INFRASTRUCTURE & ENCROACHMENT CHANGE DETECTION SYSTEM</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MODE 1: SECTOR SIMULATION (PRESETS)
# ==============================================================================
if app_mode == "Sector Simulation (Demo Presets)":
    # Preset scenarios / Real Dataset pairs
    preset_dir = os.path.join(PROJECT_ROOT, "data", "samples")
    if not os.path.exists(os.path.join(preset_dir, "A")):
        generate_all_samples(output_dir=preset_dir)

    # Discover all available image pairs across datasets (LEVIR-CD, S2Looking, etc.)
    sectors = []
    base_lats = [34.1205, 34.2541, 33.9850, 34.0512, 34.3100, 34.1500, 34.2200, 34.0900]
    base_lons = [74.8320, 74.9182, 74.6521, 74.7745, 75.0120, 74.8800, 74.9500, 74.7200]
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    # 1. Check data/samples/A
    dir_levir_a = os.path.join(preset_dir, "A")
    if os.path.exists(dir_levir_a):
        for idx, fname in enumerate(sorted(os.listdir(dir_levir_a))):
            if not fname.startswith('.') and os.path.splitext(fname)[1].lower() in valid_exts:
                sid = os.path.splitext(fname)[0]
                num = sid.replace("train_", "") if sid.startswith("train_") else sid
                stitle = f"[LEVIR-CD] Sector Tile #{num} ({fname})"
                lat = base_lats[idx % len(base_lats)] + (idx * 0.0005)
                lon = base_lons[idx % len(base_lons)] + (idx * 0.0005)
                sectors.append({
                    "id": sid,
                    "name": stitle,
                    "path_a": os.path.join(preset_dir, "A", fname),
                    "path_b": os.path.join(preset_dir, "B", fname),
                    "path_mask": os.path.join(preset_dir, "label", fname),
                    "lat": round(lat, 6),
                    "lon": round(lon, 6)
                })

    # 2. Check data/samples/S2Looking or data/s2looking
    s2_dirs = [os.path.join(preset_dir, "S2Looking"), os.path.join(PROJECT_ROOT, "data", "s2looking")]
    for s2_root in s2_dirs:
        s2_a = os.path.join(s2_root, "Image1") if os.path.exists(os.path.join(s2_root, "Image1")) else os.path.join(s2_root, "A")
        s2_b = os.path.join(s2_root, "Image2") if os.path.exists(os.path.join(s2_root, "Image2")) else os.path.join(s2_root, "B")
        s2_m = os.path.join(s2_root, "label") if os.path.exists(os.path.join(s2_root, "label")) else os.path.join(s2_root, "mask")
        if os.path.exists(s2_a) and os.path.exists(s2_b):
            for idx, fname in enumerate(sorted(os.listdir(s2_a))):
                if not fname.startswith('.') and os.path.splitext(fname)[1].lower() in valid_exts:
                    sid = f"s2_{os.path.splitext(fname)[0]}"
                    stitle = f"[S2Looking] Sector Tile ({fname})"
                    lat = base_lats[(idx + 3) % len(base_lats)] + (idx * 0.0004)
                    lon = base_lons[(idx + 3) % len(base_lons)] + (idx * 0.0004)
                    sectors.append({
                        "id": sid,
                        "name": stitle,
                        "path_a": os.path.join(s2_a, fname),
                        "path_b": os.path.join(s2_b, fname),
                        "path_mask": os.path.join(s2_m, fname) if os.path.exists(s2_m) else None,
                        "lat": round(lat, 6),
                        "lon": round(lon, 6)
                    })

    if not sectors:
        st.warning("No image pairs found. Please generate samples or upload imagery.")
        st.stop()

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        chosen_sector = st.selectbox(
            f"Select Surveillance Sector for Analysis ({len(sectors)} Available Pairs across LEVIR-CD & S2Looking)",
            sectors,
            format_func=lambda s: s["name"]
        )
    with col_btn:
        st.write("")
        st.write("")
        run_scan = st.button("🚀 EXECUTE SECTOR SCAN", use_container_width=True)

    img_a_path = chosen_sector["path_a"]
    img_b_path = chosen_sector["path_b"]
    mask_gt_path = chosen_sector["path_mask"]

    img_a = Image.open(img_a_path)
    img_b = Image.open(img_b_path)

    # Run Analysis
    with st.spinner("Processing satellite bi-temporal passes via Siamese U-Net..."):
        results = engine.predict(
            img_a,
            img_b,
            threshold=thresh_val,
            base_lat=chosen_sector['lat'],
            base_lon=chosen_sector['lon'],
            match_lighting=match_light_val
        )

    threat = results['threat_info']
    clusters = results['clusters']

    # --- DEFCON Threat HUD Banner ---
    st.markdown(f"""
    <div class="threat-banner" style="background-color: {threat['color']}18; border-color: {threat['color']};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 13px; font-weight: 700; color: {threat['color']}; letter-spacing: 1px;">SURVEILLANCE ASSESSMENT STATUS</span>
                <div style="font-size: 22px; font-weight: 800; color: #ffffff; margin: 2px 0;">{threat['threat_level']} — {threat['defcon']}</div>
                <div style="font-size: 14px; color: #cbd5e1; margin-top: 4px;"><strong>Recommended Action:</strong> {threat['action_recommendation']}</div>
            </div>
            <div style="text-align: right; background: {threat['color']}33; padding: 10px 18px; border-radius: 6px; border: 1px solid {threat['color']};">
                <div style="font-size: 10px; color: #e2e8f0; text-transform: uppercase;">Threat Index</div>
                <div style="font-size: 28px; font-weight: 900; color: {threat['color']};">{threat['threat_score']}/100</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric HUD Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">Sector Coordinates</div>
            <div class="hud-val" style="font-size: 16px;">{chosen_sector['lat']}° N, {chosen_sector['lon']}° E</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">Total Altered Footprint</div>
            <div class="hud-val">{threat['total_sqm_changed']} m²</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">Sector Area Alteration</div>
            <div class="hud-val">{threat['pct_change']}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-label">Detected Structural Targets</div>
            <div class="hud-val" style="color: {'#FF1744' if threat['num_clusters'] > 0 else '#00E676'};">{threat['num_clusters']} Units</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Inspection Matrix
    # Visual Inspection Matrix
    tab_view, tab_feat, tab_geo, tab_manifest = st.tabs([
        "🖼️ Satellite Imaging Matrix", 
        "📐 Feature Engineering Matrix",
        "🗺️ Tactical Sector Map", 
        "📋 Target Intelligence Manifest"
    ])

    with tab_view:
        has_gt = os.path.isfile(mask_gt_path) if mask_gt_path else False
        cols = st.columns(5 if has_gt else 4)
        with cols[0]:
            st.markdown("**1. Time T1 (Baseline)**")
            st.image(img_a, use_container_width=True)
            st.caption("Reference Pre-temporal Orbit Scan")
        with cols[1]:
            st.markdown("**2. Time T2 (Current)**")
            st.image(img_b, use_container_width=True)
            st.caption("Surveillance Post-temporal Orbit Scan")
        with cols[2]:
            st.markdown("**3. Tactical Target Overlay**")
            st.image(results['tactical_overlay'], use_container_width=True)
            st.caption("Detected Targets & Bounding Boxes")
        with cols[3]:
            st.markdown("**4. AI Heatmap**")
            st.image(results['heatmap_rgb'], use_container_width=True)
            st.caption("Feature Difference Probability")
        if has_gt:
            with cols[4]:
                st.markdown("**5. Ground Truth Mask**")
                st.image(Image.open(mask_gt_path), use_container_width=True)
                st.caption("Benchmark Change Label")

    with tab_feat:
        st.markdown("##### 📐 Remote Sensing & Spatial Feature Engineering Suite")
        st.caption("Classical multispectral indices, edge gradients, and texture entropy engineered to isolate structural anomalies from natural terrain shifts.")
        
        feats = results.get('engineered_features', {})
        if feats:
            f_cols = st.columns(4)
            with f_cols[0]:
                st.markdown("**🌿 Vegetation Loss Index (ΔVARI)**")
                st.image(feats['vis_vari_rgb'], use_container_width=True)
                st.caption("Atmospherically Resistant Veg Clearing")
            with f_cols[1]:
                st.markdown("**🏗️ Built-Up Emergence (ΔEBI)**")
                st.image(feats['vis_ebi_rgb'], use_container_width=True)
                st.caption("Engineered Built-up Concrete Index")
            with f_cols[2]:
                st.markdown("**📐 Structural Gradient (ΔSobel)**")
                st.image(feats['vis_grad_rgb'], use_container_width=True)
                st.caption("Sobel High-Frequency Edge Mag")
            with f_cols[3]:
                st.markdown("**🔬 Texture Entropy Shift**")
                st.image(feats['vis_tex_rgb'] if 'vis_tex_rgb' in feats else feats['vis_comp_rgb'], use_container_width=True)
                st.caption("Local Variance / Roughness Shift")

            st.markdown("---")
            st.markdown("###### 📊 Engineered Feature Statistical Signatures")
            m = feats.get('metrics', {})
            fm1, fm2, fm3, fm4 = st.columns(4)
            with fm1:
                st.metric("Vegetation Loss Score", f"{m.get('mean_veg_loss_index', 0.0):.4f}")
            with fm2:
                st.metric("Built-Up Emergence Score", f"{m.get('mean_builtup_emergence_index', 0.0):.4f}")
            with fm3:
                st.metric("Mean Edge Gradient", f"{m.get('mean_structural_gradient', 0.0):.1f} / 255")
            with fm4:
                st.metric("Texture Roughness Shift", f"{m.get('mean_texture_entropy_shift', 0.0):.1f} / 255")

    with tab_geo:
        st.markdown("##### 📍 Geospatial Tactical Positioning")
        m = folium.Map(
            location=[chosen_sector['lat'], chosen_sector['lon']],
            zoom_start=15,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="ESRI World Imagery Satellite"
        )

        # Sector Center Marker
        folium.Marker(
            [chosen_sector['lat'], chosen_sector['lon']],
            tooltip=chosen_sector['name'],
            popup=f"Sector: {chosen_sector['name']}<br>Threat: {threat['threat_level']}",
            icon=folium.Icon(color="red" if threat['num_clusters'] > 0 else "green", icon="crosshairs", prefix="fa")
        ).add_to(m)

        # Draw detected clusters
        for cl in clusters:
            folium.CircleMarker(
                location=[cl['lat'], cl['lon']],
                radius=10,
                popup=f"<b>{cl['id']}</b><br>Area: {cl['area_sqm']} m²<br>Centroid: {cl['lat']}, {cl['lon']}",
                color="#FF1744",
                fill=True,
                fill_color="#FF1744",
                fill_opacity=0.6
            ).add_to(m)

        st_folium(m, width=1000, height=450)

    with tab_manifest:
        if clusters:
            st.markdown("##### 📋 Detected Target Clusters & Coordinates")
            df_clusters = pd.DataFrame([
                {
                    "Target ID": c['id'],
                    "Estimated Area (m²)": c['area_sqm'],
                    "Pixel Area (px)": c['area_px'],
                    "Latitude": c['lat'],
                    "Longitude": c['lon'],
                    "Bounding Box (X, Y, W, H)": f"{c['bbox']}",
                    "Perimeter (px)": c['perimeter']
                } for c in clusters
            ])
            st.dataframe(df_clusters, use_container_width=True)

            # Export options
            csv_data = df_clusters.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Target Manifest (CSV)",
                data=csv_data,
                file_name=f"target_manifest_{chosen_sector['id']}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No unauthorized structural targets detected in this sector.")


# ==============================================================================
# MODE 2: MANUAL IMAGE UPLOAD
# ==============================================================================
else:
    st.subheader("📤 Upload Bi-Temporal Satellite Imagery")
    st.caption("Upload pre-change (Time 1) and post-change (Time 2) satellite tiles for AI change detection analysis.")

    u_col1, u_col2 = st.columns(2)
    with u_col1:
        up_a = st.file_uploader("Upload Time T1 Satellite Image (Before)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])
    with u_col2:
        up_b = st.file_uploader("Upload Time T2 Satellite Image (After)", type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])

    # Optional Georeferencing (coordinates are NOT required)
    with st.expander("📍 Optional Georeferencing (Coordinates / GPS)", expanded=False):
        st.caption("Provide sector reference coordinates for GIS mapping. If omitted, default sector coordinates are used.")
        c_lat, c_lon = st.columns(2)
        with c_lat:
            user_lat = st.number_input("Base Latitude (Optional)", value=34.0837, format="%.6f")
        with c_lon:
            user_lon = st.number_input("Base Longitude (Optional)", value=74.7973, format="%.6f")

    if up_a and up_b:
        img_a = Image.open(up_a).convert('RGB')
        img_b = Image.open(up_b).convert('RGB')

        if st.button("🚀 RUN SURVEILLANCE INFERENCE", use_container_width=True):
            with st.spinner("Analyzing uploaded bi-temporal satellite pair..."):
                results = engine.predict(
                    img_a,
                    img_b,
                    threshold=thresh_val,
                    base_lat=user_lat,
                    base_lon=user_lon,
                    match_lighting=match_light_val
                )

            threat = results['threat_info']
            clusters = results['clusters']

            # Threat Status Banner
            st.markdown(f"""
            <div class="threat-banner" style="background-color: {threat['color']}18; border-color: {threat['color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 20px; font-weight: 800; color: #ffffff;">{threat['threat_level']} — {threat['defcon']}</div>
                        <div style="font-size: 14px; color: #cbd5e1; margin-top: 4px;"><strong>Recommended Action:</strong> {threat['action_recommendation']}</div>
                    </div>
                    <div style="text-align: right; background: {threat['color']}33; padding: 10px 18px; border-radius: 6px; border: 1px solid {threat['color']};">
                        <div style="font-size: 10px; color: #e2e8f0; text-transform: uppercase;">Threat Score</div>
                        <div style="font-size: 26px; font-weight: 900; color: {threat['color']};">{threat['threat_score']}/100</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="hud-card"><div class="hud-label">Total Altered Footprint</div><div class="hud-val">{threat['total_sqm_changed']} m²</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="hud-card"><div class="hud-label">Sector Alteration</div><div class="hud-val">{threat['pct_change']}%</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="hud-card"><div class="hud-label">Detected Clusters</div><div class="hud-val">{threat['num_clusters']} Units</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="hud-card"><div class="hud-label">Max Structure Size</div><div class="hud-val">{threat['max_cluster_area_sqm']} m²</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab_u_view, tab_u_feat, tab_u_manifest = st.tabs([
                "🖼️ Satellite Imaging Matrix", 
                "📐 Feature Engineering Matrix", 
                "📋 Target Intelligence Manifest"
            ])

            with tab_u_view:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown("**1. Time T1 (Before)**")
                    st.image(img_a, use_container_width=True)
                with c2:
                    st.markdown("**2. Time T2 (After)**")
                    st.image(img_b, use_container_width=True)
                with c3:
                    st.markdown("**3. Tactical Target Overlay**")
                    st.image(results['tactical_overlay'], use_container_width=True)
                with c4:
                    st.markdown("**4. AI Probability Heatmap**")
                    st.image(results['heatmap_rgb'], use_container_width=True)

            with tab_u_feat:
                st.markdown("##### 📐 Remote Sensing & Spatial Feature Engineering Suite")
                feats_u = results.get('engineered_features', {})
                if feats_u:
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    with fc1:
                        st.markdown("**🌿 Vegetation Loss Index (ΔVARI)**")
                        st.image(feats_u['vis_vari_rgb'], use_container_width=True)
                    with fc2:
                        st.markdown("**🏗️ Built-Up Emergence (ΔEBI)**")
                        st.image(feats_u['vis_ebi_rgb'], use_container_width=True)
                    with fc3:
                        st.markdown("**📐 Structural Gradient (ΔSobel)**")
                        st.image(feats_u['vis_grad_rgb'], use_container_width=True)
                    with fc4:
                        st.markdown("**🔬 Texture Entropy Shift**")
                        st.image(feats_u['vis_comp_rgb'], use_container_width=True)

                    st.markdown("---")
                    st.markdown("###### 📊 Statistical Feature Signatures")
                    mu = feats_u.get('metrics', {})
                    um1, um2, um3, um4 = st.columns(4)
                    with um1:
                        st.metric("Vegetation Loss Score", f"{mu.get('mean_veg_loss_index', 0.0):.4f}")
                    with um2:
                        st.metric("Built-Up Emergence Score", f"{mu.get('mean_builtup_emergence_index', 0.0):.4f}")
                    with um3:
                        st.metric("Mean Edge Gradient", f"{mu.get('mean_structural_gradient', 0.0):.1f} / 255")
                    with um4:
                        st.metric("Texture Roughness Shift", f"{mu.get('mean_texture_entropy_shift', 0.0):.1f} / 255")

            with tab_u_manifest:
                if clusters:
                    st.markdown("##### 📋 Detected Target Manifest")
                    df_u = pd.DataFrame([
                        {
                            "Target ID": c['id'],
                            "Area (m²)": c['area_sqm'],
                            "Area (px)": c['area_px'],
                            "Centroid (X, Y)": f"({c['centroid'][0]}, {c['centroid'][1]})",
                            "Bounding Box (X, Y, W, H)": str(c['bbox']),
                            "Simulated Lat": c['lat'],
                            "Simulated Lon": c['lon']
                        } for c in clusters
                    ])
                    st.dataframe(df_u, use_container_width=True)

                    csv_data_u = df_u.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Export Target Manifest (CSV)",
                        data=csv_data_u,
                        file_name="uploaded_target_manifest.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ No unauthorized structural changes detected between uploaded images.")

st.markdown("---")
st.caption("AEGIS-SAT Border Surveillance AI System | Developed for Academic Research & Evaluation")
