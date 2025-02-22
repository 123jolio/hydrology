import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile
import io
import rasterio
from rasterio.transform import from_origin
import imageio
import os
from PIL import Image
from scipy.ndimage import convolve

# Set page config for a wide layout and dark theme
st.set_page_config(page_title="Hydrogeology & DEM Analysis", layout="wide", initial_sidebar_state="collapsed")

# Set matplotlib to dark background style for dark mode compatibility
plt.style.use('dark_background')

# Dark mode CSS with logo and header styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');

    /* Dark mode base styles */
    html, body, [class*="css"] {
        background-color: #1E1E1E !important;
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Header and logo container */
    .header-container {
        display: flex;
        align-items: center;
        padding: 10px;
        background-color: #2D2D2D;
        border-bottom: 1px solid #444;
    }
    .header-container img {
        width: 50px;
        margin-right: 15px;
    }
    .header-container h1 {
        font-size: 24px;
        color: #FFFFFF;
        margin: 0;
    }

    /* Ribbon toolbar */
    .ribbon {
        background-color: #2D2D2D;
        padding: 10px;
        border-bottom: 1px solid #444;
        display: flex;
        gap: 20px;
    }
    .ribbon button {
        background-color: #005A9E;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
    }

    /* Tabs styling */
    .stTabs > div {
        background-color: #2D2D2D;
        border-bottom: 2px solid #005A9E;
    }
    .stTabs button {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stTabs button:hover {
        background-color: #3A3A3A;
    }
    .stTabs .stTab--active {
        background-color: #1E1E1E;
        border-bottom: 2px solid #005A9E;
        color: #FFFFFF;
    }

    /* Expander and widget styling */
    .streamlit-expanderHeader {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    .streamlit-expanderContent {
        background-color: #1E1E1E;
    }
    .stSlider > div > div > div > div {
        background-color: #005A9E;
    }
    .stNumberInput input {
        background-color: #333333;
        color: #E0E0E0;
    }
    .stSelectbox > div > div {
        background-color: #333333;
        color: #E0E0E0;
    }

    /* Plot styling */
    .matplotlib-figure {
        background-color: #1E1E1E;
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo and title from the repository (logo.png)
st.markdown("""
    <div class="header-container">
        <img src="logo.png" alt="Logo">
        <h1>Advanced Hydrogeology & DEM Analysis</h1>
    </div>
""", unsafe_allow_html=True)

# Ribbon Toolbar
with st.container():
    st.markdown('<div class="ribbon">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_stl = st.file_uploader("Upload STL File", type=["stl"])
        uploaded_burned = st.file_uploader("Upload Burned-Area TIFF", type=["tif", "tiff"])
    with col2:
        run_button = st.button("Run Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# Define tabs
tabs = st.tabs([
    "DEM & Flow Simulation", "Slope Map", "Aspect Map", "Retention Time", "GeoTIFF Export",
    "Nutrient Leaching", "Flow Accumulation", "TWI", "Curvature", "Scenario GIFs"
])

# Georeference bounds
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# DEM & Flow Simulation Tab settings
with tabs[0]:
    st.header("DEM & Flow Simulation")
    with st.expander("Elevation Adjustments", expanded=True):
        scale = st.slider("Scale Factor", 0.1, 5.0, 1.0, 0.1, key="scale")
        offset = st.slider("Offset (m)", -100.0, 100.0, 0.0, 1.0, key="offset")
        dem_min = st.number_input("Min Elevation (m)", value=0.0, step=1.0, key="dem_min")
        dem_max = st.number_input("Max Elevation (m)", value=500.0, step=1.0, key="dem_max")
        grid_res = st.number_input("Grid Resolution", 100, 1000, 500, 50, key="grid_res")
    with st.expander("Flow & Retention"):
        rainfall = st.number_input("Rainfall (mm/hr)", value=30.0, step=1.0, key="rainfall")
        duration = st.number_input("Duration (hr)", value=2.0, step=0.1, key="duration")
        area = st.number_input("Area (ha)", value=10.0, step=0.1, key="area")
        runoff = st.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05, key="runoff")
        recession = st.number_input("Recession Rate (1/hr)", value=0.5, step=0.1, key="recession")
        sim_hours = st.number_input("Simulation Duration (hr)", value=6.0, step=0.5, key="sim_hours")
        storage = st.number_input("Storage Volume (m³)", value=5000.0, step=100.0, key="storage")
    with st.expander("Burned Area Effects"):
        burn_factor = st.slider("Runoff Increase Factor", 0.0, 2.0, 1.0, 0.1, key="burn_factor")

# Nutrient Leaching Tab settings
with tabs[5]:
    st.header("Nutrient Leaching")
    with st.expander("Nutrient Leaching Parameters", expanded=True):
        nutrient = st.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0, key="nutrient")
        retention = st.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05, key="retention")
        erosion = st.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05, key="erosion")

# Scenario GIFs Tab settings
with tabs[9]:
    st.header("Scenario GIFs")
    with st.expander("GIF Settings", expanded=True):
        gif_frames = st.number_input("GIF Frames", value=10, step=1, key="gif_frames")
        gif_fps = st.number_input("GIF FPS", value=2, step=1, key="gif_fps")

# Option to overlay burned area TIFF
st.sidebar.header("Overlay Options")
overlay_burned_option = st.sidebar.checkbox("Overlay Burned Area TIFF", value=True)
burned_transparency = st.sidebar.slider("Burned Area Overlay Transparency", 0.0, 1.0, 0.3, 0.05)

# Processing and Display Logic
if uploaded_stl and run_button:
    # Retrieve parameters from session state
    scale = st.session_state.scale
    offset = st.session_state.offset
    dem_min = st.session_state.dem_min
    dem_max = st.session_state.dem_max
    grid_res = st.session_state.grid_res
    rainfall = st.session_state.rainfall
    duration = st.session_state.duration
    area = st.session_state.area
    runoff = st.session_state.runoff
    recession = st.session_state.recession
    sim_hours = st.session_state.sim_hours
    storage = st.session_state.storage
    burn_factor = st.session_state.burn_factor
    nutrient = st.session_state.nutrient
    retention = st.session_state.retention
    erosion = st.session_state.erosion
    gif_frames = st.session_state.gif_frames
    gif_fps = st.session_state.gif_fps

    # Load and process STL file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_mesh = mesh.Mesh.from_file(tmp_stl.name)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_adj = (z_raw * scale) + offset

    # Georeference and interpolate DEM
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = bottom_bound + (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)
    xi = np.linspace(left_bound, right_bound, grid_res)
    yi = np.linspace(bottom_bound, top_bound, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # Compute derivatives
    dx = (right_bound - left_bound) / (grid_res - 1)
    dy = (top_bound - bottom_bound) / (grid_res - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters, dy_meters = dx * meters_per_deg_lon, dy * meters_per_deg_lat
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # Flow simulation (simplified)
    area_m2 = area * 10000.0
    total_rain_m = (rainfall / 1000.0) * duration
    V_runoff = total_rain_m * area_m2 * runoff
    Q_peak = V_runoff / duration
    t = np.linspace(0, sim_hours, int(sim_hours * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= duration:
            Q[i] = Q_peak * (time / duration)
        else:
            Q[i] = Q_peak * np.exp(-recession * (time - duration))
    retention_time = storage / (V_runoff / duration) if V_runoff > 0 else None

    # Nutrient leaching
    nutrient_load = nutrient * (1 - retention) * erosion * area

    # Burned area processing
    burned_mask = None
    if uploaded_burned:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
            tmp_tif.write(uploaded_burned.read())
            with rasterio.open(tmp_tif.name) as src:
                burned_img = src.read(1)
                # Here we assume a simple threshold; adjust as needed
                burned_mask = (burned_img > 150).astype(np.float32)

    # Terrain derivatives (placeholders/simplified)
    flow_acc = np.ones_like(grid_z)  # Placeholder for flow accumulation
    twi = np.log((flow_acc + 1) / (np.tan(np.radians(slope)) + 0.05))
    curvature = convolve(grid_z, np.ones((3, 3)) / 9, mode='reflect')

    # Helper: Plotting function with correct aspect ratio
    def plot_with_correct_aspect(ax, data, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, origin='lower', extent=(left_bound, right_bound, bottom_bound, top_bound), vmin=vmin, vmax=vmax)
        # Compute aspect ratio for geographical accuracy
        aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * (meters_per_deg_lat / meters_per_deg_lon)
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        return im

    # Helper: Create figure using consistent size based on spatial extents
    def create_fig_ax():
        spatial_width = right_bound - left_bound
        spatial_height = top_bound - bottom_bound
        aspect_ratio = spatial_height / spatial_width
        base_width = 8
        fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
        return fig, ax

    # Helper: Overlay burned area if enabled
    def overlay_burned(ax):
        if overlay_burned_option and (burned_mask is not None):
            ax.imshow(burned_mask, cmap="Reds", alpha=burned_transparency,
                      origin='lower', extent=(left_bound, right_bound, bottom_bound, top_bound))

    # Tab 0: DEM & Flow Simulation
    with tabs[0]:
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, grid_z, 'terrain', vmin=dem_min, vmax=dem_max)
        step = max(1, grid_res // 20)
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step],
                  -dz_dx[::step, ::step], -dz_dy[::step, ::step],
                  color='blue', scale=1e5, width=0.0025)
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 1: Slope Map
    with tabs[1]:
        with st.expander("Visualization Options"):
            slope_vmin = st.number_input("Slope Min", value=0.0, key="slope_vmin")
            slope_vmax = st.number_input("Slope Max", value=90.0, key="slope_vmax")
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno"], key="slope_cmap")
        st.subheader("Slope Map")
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, slope, slope_cmap, vmin=slope_vmin, vmax=slope_vmax)
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 2: Aspect Map
    with tabs[2]:
        with st.expander("Visualization Options"):
            aspect_vmin = st.number_input("Aspect Min", value=0.0, key="aspect_vmin")
            aspect_vmax = st.number_input("Aspect Max", value=360.0, key="aspect_vmax")
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv"], key="aspect_cmap")
        st.subheader("Aspect Map")
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, aspect, aspect_cmap, vmin=aspect_vmin, vmax=aspect_vmax)
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 3: Retention Time
    with tabs[3]:
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.write("No effective runoff → Retention time not applicable.")

    # Tab 4: GeoTIFF Export
    with tabs[4]:
        st.subheader("GeoTIFF Export")
        st.write("Export functionality to be implemented.")

    # Tab 5: Nutrient Leaching
    with tabs[5]:
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    # Tab 6: Flow Accumulation
    with tabs[6]:
        st.subheader("Flow Accumulation")
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, flow_acc, 'Blues')
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 7: TWI
    with tabs[7]:
        st.subheader("Topographic Wetness Index")
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, twi, 'RdYlBu')
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 8: Curvature
    with tabs[8]:
        st.subheader("Curvature Analysis")
        fig, ax = create_fig_ax()
        im = plot_with_correct_aspect(ax, curvature, 'Spectral')
        overlay_burned(ax)
        st.pyplot(fig)

    # Tab 9: Scenario GIFs
    with tabs[9]:
        st.write("GIF generation to be implemented.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
