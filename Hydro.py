import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile
import io
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import imageio
import os
from PIL import Image
from scipy.ndimage import convolve, zoom
from matplotlib.colors import ListedColormap
import pandas as pd
import math  # For radians conversion

# -----------------------------------------------------------------------------
# 1. Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hydrogeology & DEM Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

plt.style.use('dark_background')

# -----------------------------------------------------------------------------
# 2. Dark Mode + Custom CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Segoe+UI&display=swap');
html, body, [class*="css"] {
    background-color: #1E1E1E !important;
    color: #E0E0E0 !important;
    font-family: 'Segoe UI', sans-serif;
}
.header-container {
    display: flex;
    align-items: center;
    padding: 10px;
    background-color: #2D2D2D;
    border-bottom: 1px solid #444;
}
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
.stCheckbox > div > label {
    color: #E0E0E0;
}
.matplotlib-figure {
    background-color: #1E1E1E;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Header (logo + title) using st.image()
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("logo.png", width=250)
with col_title:
    st.markdown("<h1 style='color: #FFFFFF;'>Advanced Hydrogeology & DEM Analysis</h1>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Ribbon Toolbar
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="ribbon">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_stl = st.file_uploader("Upload STL File", type=["stl"])
        uploaded_burned = st.file_uploader("Upload Burned-Area TIFF (RGB)", type=["tif", "tiff"])
    with col2:
        run_button = st.button("Run Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "DEM & Flow Simulation", "Burned Areas", "Slope Map", "Aspect Map", 
    "Retention Time", "GeoTIFF Export", "Nutrient Leaching", 
    "Flow Accumulation", "TWI", "Curvature", "Scenario GIFs", 
    "Burned-Area Hydro Impacts", "Parameter Comparison"
])

# -----------------------------------------------------------------------------
# 6. Georeference bounding box (EPSG:4326) and constants
# -----------------------------------------------------------------------------
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509
avg_lat = (top_bound + bottom_bound) / 2.0
meters_per_deg_lon = 111320 * math.cos(math.radians(avg_lat))
meters_per_deg_lat = 111320

# -----------------------------------------------------------------------------
# Helper function for plotting with burned area overlay
# -----------------------------------------------------------------------------
def plot_with_burned_overlay(ax, data, cmap, vmin=None, vmax=None, 
                             burned_mask=None, show_burned=True, alpha=0.5):
    data = np.flipud(data)
    if burned_mask is not None and show_burned:
        burned_mask = np.flipud(burned_mask)
    
    im = ax.imshow(data, cmap=cmap, origin='upper',
                   extent=(left_bound, right_bound, bottom_bound, top_bound),
                   vmin=vmin, vmax=vmax)
    if show_burned and burned_mask is not None:
        burned_cmap = ListedColormap(['none', 'red'])
        ax.imshow(burned_mask, cmap=burned_cmap, origin='upper',
                  extent=(left_bound, right_bound, bottom_bound, top_bound),
                  alpha=alpha)
    aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound)
    aspect_ratio *= (meters_per_deg_lat / meters_per_deg_lon)
    ax.set_aspect(aspect_ratio)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    return im

# -----------------------------------------------------------------------------
# Flow Direction and Accumulation Functions
# -----------------------------------------------------------------------------
def flow_direction_d8(dem):
    """Calculate D8 flow direction based on steepest slope."""
    rows, cols = dem.shape
    flow_dir = np.zeros_like(dem, dtype=np.int8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            neighbors = [
                (dem[i-1, j-1], 1), (dem[i-1, j], 2), (dem[i-1, j+1], 3),
                (dem[i, j-1], 8),                     (dem[i, j+1], 4),
                (dem[i+1, j-1], 7), (dem[i+1, j], 6), (dem[i+1, j+1], 5)
            ]
            min_elev = min(neighbors, key=lambda x: x[0])
            if min_elev[0] < dem[i, j]:
                flow_dir[i, j] = min_elev[1]
    return flow_dir

def flow_accumulation(flow_dir):
    """Calculate flow accumulation using D8 flow directions."""
    rows, cols = flow_dir.shape
    accum = np.ones_like(flow_dir, dtype=np.float32)
    for _ in range(rows * cols):  # Iterate to propagate flow
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if flow_dir[i, j] != 0:
                    if flow_dir[i, j] == 1:
                        accum[i-1, j-1] += accum[i, j]
                    elif flow_dir[i, j] == 2:
                        accum[i-1, j] += accum[i, j]
                    elif flow_dir[i, j] == 3:
                        accum[i-1, j+1] += accum[i, j]
                    elif flow_dir[i, j] == 4:
                        accum[i, j+1] += accum[i, j]
                    elif flow_dir[i, j] == 5:
                        accum[i+1, j+1] += accum[i, j]
                    elif flow_dir[i, j] == 6:
                        accum[i+1, j] += accum[i, j]
                    elif flow_dir[i, j] == 7:
                        accum[i+1, j-1] += accum[i, j]
                    elif flow_dir[i, j] == 8:
                        accum[i, j-1] += accum[i, j]
    return accum

# -----------------------------------------------------------------------------
# 7. Parameter Inputs in "DEM & Flow Simulation" Tab
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("DEM & Flow Simulation")
    st.markdown("### Adjust the following parameters to customize your DEM and flow simulation:")
    st.markdown("**Note**: Changes to these parameters will be applied when you click 'Run Analysis'.")
    
    with st.expander("Elevation Adjustments", expanded=True):
        st.markdown("**Scale Factor**: Multiplies elevation values to adjust vertical exaggeration (0.1–5.0).")
        scale = st.slider("Scale Factor", 0.1, 5.0, 1.0, 0.1, key="scale")
        
        st.markdown("**Offset (m)**: Adds or subtracts a constant elevation (m) to shift the entire DEM.")
        offset = st.slider("Offset (m)", -100.0, 100.0, 0.0, 1.0, key="offset")
        
        st.markdown("**Min Elevation (m)**: Sets the minimum elevation for clipping (0–500 m).")
        dem_min = st.number_input("Min Elevation (m)", value=0.0, step=1.0, key="dem_min")
        
        st.markdown("**Max Elevation (m)**: Sets the maximum elevation for clipping (0–500 m).")
        dem_max = st.number_input("Max Elevation (m)", value=500.0, step=1.0, key="dem_max")
        
        st.markdown("**Grid Resolution**: Sets the number of grid cells (100–1000) for DEM interpolation.")
        grid_res = st.number_input("Grid Resolution", 100, 1000, 500, 50, key="grid_res")

    with st.expander("Flow & Retention", expanded=True):
        st.markdown("**Rainfall (mm/hr)**: Sets rainfall intensity (1–100 mm/hr).")
        rainfall = st.number_input("Rainfall (mm/hr)", value=30.0, step=1.0, key="rainfall")
        
        st.markdown("**Duration (hr)**: Sets storm duration (0.1–24 hr).")
        duration = st.number_input("Duration (hr)", value=2.0, step=0.1, key="duration")
        
        st.markdown("**Area (ha)**: Sets the watershed area (0.1–100 ha).")
        area = st.number_input("Area (ha)", value=10.0, step=0.1, key="area")
        
        st.markdown("**Runoff Coefficient**: Fraction of rainfall becoming runoff (0.0–1.0).")
        runoff = st.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05, key="runoff")
        
        st.markdown("**Recession Rate (1/hr)**: Controls how quickly flow decreases after rain (0.1–2.0).")
        recession = st.number_input("Recession Rate (1/hr)", value=0.5, step=0.1, key="recession")
        
        st.markdown("**Simulation Duration (hr)**: Sets the total simulation time (0.5–24 hr).")
        sim_hours = st.number_input("Simulation Duration (hr)", value=6.0, step=0.5, key="sim_hours")
        
        st.markdown("**Storage Volume (m³)**: Sets water storage capacity (100–10000 m³).")
        storage = st.number_input("Storage Volume (m³)", value=5000.0, step=100.0, key="storage")

    with st.expander("Burned Area Effects", expanded=True):
        st.markdown("**Runoff Increase Factor**: Multiplies runoff in burned areas (0.0–2.0).")
        burn_factor = st.slider("Runoff Increase Factor", 0.0, 2.0, 1.0, 0.1, key="burn_factor")
        
        st.markdown("**Burned Area Threshold**: Sets the pixel value threshold (0–255) for detecting burned areas.")
        burn_threshold = st.slider("Burned Area Threshold", 0, 255, 200, 1, key="burn_threshold")
        
        st.markdown("**Band for Burned Area Threshold**: Selects the color band (Red, Green, Blue) for thresholding.")
        band_to_threshold = st.selectbox("Band for Burned Area Threshold", ["Red", "Green", "Blue"], key="band_threshold")

    # Process data when "Run Analysis" is clicked
    if run_button:
        if uploaded_stl is None:
            st.error("Please upload an STL file before running the analysis.")
        else:
            with st.spinner("Running analysis..."):
                try:
                    st.session_state.processed_data = None  # Clear previous data
                    # Retrieve parameters from session state
                    scale_val = st.session_state.get('scale', 1.0)
                    offset_val = st.session_state.get('offset', 0.0)
                    dem_min_val = st.session_state.get('dem_min', 0.0)
                    dem_max_val = st.session_state.get('dem_max', 500.0)
                    grid_res_val = st.session_state.get('grid_res', 500)
                    rainfall_val = st.session_state.get('rainfall', 30.0)
                    duration_val = st.session_state.get('duration', 2.0)
                    area_val = st.session_state.get('area', 10.0)
                    runoff_val = st.session_state.get('runoff', 0.5)
                    recession_val = st.session_state.get('recession', 0.5)
                    sim_hours_val = st.session_state.get('sim_hours', 6.0)
                    storage_val = st.session_state.get('storage', 5000.0)
                    burn_factor_val = st.session_state.get('burn_factor', 1.0)
                    burn_threshold_val = st.session_state.get('burn_threshold', 200)
                    band_to_threshold = st.session_state.get('band_threshold', "Red")
                    nutrient_val = st.session_state.get('nutrient', 50.0)
                    retention_val = st.session_state.get('retention', 0.7)
                    erosion_val = st.session_state.get('erosion', 0.3)
                    gif_frames_val = st.session_state.get('gif_frames', 10)
                    gif_fps_val = st.session_state.get('gif_fps', 2)

                    # Load STL
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
                        tmp_stl.write(uploaded_stl.read())
                        stl_mesh = mesh.Mesh.from_file(tmp_stl.name)

                    vertices = stl_mesh.vectors.reshape(-1, 3)
                    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                    z_adj = (z_raw * scale_val) + offset_val

                    # Interpolate DEM
                    x_min, x_max = x_raw.min(), x_raw.max()
                    y_min, y_max = y_raw.min(), y_raw.max()
                    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
                    lat_raw = bottom_bound + (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)
                    xi = np.linspace(left_bound, right_bound, grid_res_val)
                    yi = np.linspace(bottom_bound, top_bound, grid_res_val)
                    grid_x, grid_y = np.meshgrid(xi, yi)
                    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
                    grid_z = np.clip(grid_z, dem_min_val, dem_max_val)

                    # Derivatives
                    dx = (right_bound - left_bound) / (grid_res_val - 1)
                    dy = (top_bound - bottom_bound) / (grid_res_val - 1)
                    dx_meters, dy_meters = dx * meters_per_deg_lon, dy * meters_per_deg_lat
                    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
                    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
                    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

                    # Burned area detection with reprojection if CRS is available
                    burned_mask = None
                    if uploaded_burned:
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
                                tmp_tif.write(uploaded_burned.read())
                                with rasterio.open(tmp_tif.name) as src:
                                    if src.count < 3:
                                        st.warning("The burned area TIFF must be an RGB image with 3 bands.")
                                    else:
                                        band_index = 1 if band_to_threshold == "Red" else 2 if band_to_threshold == "Green" else 3
                                        band_data = src.read(band_index)
                                        burned_mask = (band_data > burn_threshold_val).astype(np.float32)

                                        # Attempt reprojection if CRS is available
                                        src_crs = src.crs
                                        if src_crs:
                                            src_transform = src.transform
                                            target_transform = from_origin(left_bound, top_bound, dx, dy)
                                            target_crs = 'EPSG:4326'
                                            target_shape = grid_z.shape
                                            resampled_mask = np.empty(target_shape, dtype=np.float32)
                                            reproject(
                                                source=burned_mask,
                                                destination=resampled_mask,
                                                src_transform=src_transform,
                                                src_crs=src_crs,
                                                dst_transform=target_transform,
                                                dst_crs=target_crs,
                                                resampling=Resampling.nearest
                                            )
                                            burned_mask = resampled_mask
                                        else:
                                            st.warning("TIFF has no CRS. Resizing mask to match DEM shape (may be inaccurate).")
                                            zoom_factors = (grid_z.shape[0] / burned_mask.shape[0], grid_z.shape[1] / burned_mask.shape[1])
                                            burned_mask = zoom(burned_mask, zoom_factors, order=0)  # Nearest neighbor

                        except Exception as e:
                            st.error(f"Error processing burned area TIFF: {e}")
                            burned_mask = None

                    # Flow Direction and Accumulation
                    flow_dir = flow_direction_d8(grid_z)
                    flow_acc = flow_accumulation(flow_dir)

                    # Enhanced TWI Calculation
                    twi = np.log((flow_acc + 1) / (np.tan(np.radians(slope)) + 0.001))

                    # SCS-CN Method for Runoff
                    base_cn = 70  # Base curve number for unburned areas
                    burned_cn = 90  # Higher curve number for burned areas
                    cn_map = np.full_like(grid_z, base_cn, dtype=np.float32)
                    cn_map[burned_mask == 1] = burned_cn
                    S = (25400 / cn_map) - 254  # Potential maximum retention in mm
                    rainfall_depth = rainfall_val / 1000  # Rainfall depth in meters
                    runoff_depth = np.where(
                        rainfall_depth > 0.2 * S,
                        (rainfall_depth - 0.2 * S)**2 / (rainfall_depth + 0.8 * S),
                        0
                    )
                    area_per_cell_m2 = (dx_meters * dy_meters)  # Assuming grid cells are square
                    runoff_volume_spatial = runoff_depth * area_per_cell_m2

                    # Store processed data in session state
                    st.session_state.processed_data = {
                        'grid_z': grid_z,
                        'slope': slope,
                        'aspect': aspect,
                        'burned_mask': burned_mask,
                        'flow_acc': flow_acc,
                        'twi': twi,
                        'runoff_depth': runoff_depth,
                        'runoff_volume_spatial': runoff_volume_spatial,
                        'grid_x': grid_x,
                        'grid_y': grid_y,
                        'dz_dx': dz_dx,
                        'dz_dy': dz_dy,
                        'dem_min_val': dem_min_val,
                        'dem_max_val': dem_max_val,
                        'rainfall_val': rainfall_val,
                        'area_val': area_val,
                        'burn_factor_val': burn_factor_val,
                        'base_cn': base_cn,
                        'burned_cn': burned_cn
                    }
                    st.write("Analysis complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
                    st.session_state.processed_data = None

    # If data is processed, display results
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        processed_data = st.session_state.processed_data
        grid_z = processed_data['grid_z']
        slope = processed_data['slope']
        aspect = processed_data['aspect']
        burned_mask = processed_data['burned_mask']
        flow_acc = processed_data['flow_acc']
        twi = processed_data['twi']
        runoff_depth = processed_data['runoff_depth']
        runoff_volume_spatial = processed_data['runoff_volume_spatial']
        grid_x = processed_data['grid_x']
        grid_y = processed_data['grid_y']
        dz_dx = processed_data['dz_dx']
        dz_dy = processed_data['dz_dy']
        dem_min_val = processed_data['dem_min_val']
        dem_max_val = processed_data['dem_max_val']
        rainfall_val = processed_data['rainfall_val']
        area_val = processed_data['area_val']
        burn_factor_val = processed_data['burn_factor_val']
        base_cn = processed_data['base_cn']
        burned_cn = processed_data['burned_cn']

        st.header("DEM & Flow Simulation")
        st.markdown("### This tab displays the Digital Elevation Model (DEM) and flow simulation results.")

        with st.expander("Visualization Options", expanded=True):
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="dem_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="dem_alpha")

        # DEM Visualization
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, grid_z, 'terrain',
            vmin=dem_min_val, vmax=dem_max_val,
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

        # Flow Accumulation Map
        st.subheader("Flow Accumulation Map")
        fig, ax = plt.subplots()
        im = ax.imshow(np.flipud(flow_acc), cmap='Blues', origin='upper',
                       extent=(left_bound, right_bound, bottom_bound, top_bound))
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        fig.colorbar(im, ax=ax, label="Flow Accumulation")
        st.pyplot(fig)

        # TWI Map
        st.subheader("Topographic Wetness Index (TWI) Map")
        fig, ax = plt.subplots()
        im = ax.imshow(np.flipud(twi), cmap='RdYlBu', origin='upper',
                       extent=(left_bound, right_bound, bottom_bound, top_bound))
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

        # Runoff Depth Map
        st.subheader("Runoff Depth Map (m)")
        fig, ax = plt.subplots()
        im = ax.imshow(np.flipud(runoff_depth), cmap='Blues', origin='upper',
                       extent=(left_bound, right_bound, bottom_bound, top_bound))
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        fig.colorbar(im, ax=ax, label="Runoff Depth (m)")
        st.pyplot(fig)

        # Scenario Analysis in Burned-Area Hydro Impacts Tab
        with tabs[11]:
            st.header("Burned-Area Hydro Impacts")
            st.subheader("Scenario Analysis")

            burn_factor_scenario = st.slider("Burn Factor for Scenario", 0.0, 2.0, 1.0, 0.1)
            erosion_multiplier_scenario = st.slider("Erosion Multiplier for Scenario", 1.0, 5.0, 2.0, 0.1)

            # Scenario-based Runoff Potential
            runoff_potential_scenario = runoff_depth / (rainfall_val / 1000)
            runoff_potential_scenario[burned_mask == 1] *= (1 + burn_factor_scenario)
            runoff_potential_scenario = np.clip(runoff_potential_scenario, 0, 1)

            # Scenario-based Erosion Risk
            base_erosion_rate = 0.1  # Example value in tons/ha
            slope_normalized = (slope - np.min(slope)) / (np.max(slope) - np.min(slope))
            erosion_risk_scenario = base_erosion_rate * (1 + slope_normalized)
            erosion_risk_scenario[burned_mask == 1] *= erosion_multiplier_scenario

            # Display Scenario Maps
            st.subheader("Runoff Potential Map (Scenario)")
            fig, ax = plt.subplots()
            im = ax.imshow(np.flipud(runoff_potential_scenario), cmap='Blues', origin='upper',
                           extent=(left_bound, right_bound, bottom_bound, top_bound),
                           vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Runoff Potential (0-1)")
            st.pyplot(fig)

            st.subheader("Erosion Risk Map (Scenario)")
            fig, ax = plt.subplots()
            im = ax.imshow(np.flipud(erosion_risk_scenario), cmap='YlOrRd', origin='upper',
                           extent=(left_bound, right_bound, bottom_bound, top_bound))
            ax.set_aspect('equal')
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Erosion Risk (tons/ha)")
            st.pyplot(fig)
