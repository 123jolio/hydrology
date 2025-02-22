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
from scipy.ndimage import convolve
from matplotlib.colors import ListedColormap
import pysheds.grid as Grid  # Added for flow accumulation

# -----------------------------------------------------------------------------
# 1. Streamlit Page Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hydrogeology & DEM Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Use a dark background style for Matplotlib
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
.header-container img {
    width: 50px;
    margin-right: 15px;
}
.header-container h1 {
    font-size: 24px;
    color: #FFFFFF;
    margin: 0;
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
# 3. Header (logo + title)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <img src="logo.png" alt="Logo">
    <h1>Advanced Hydrogeology & DEM Analysis</h1>
</div>
""", unsafe_allow_html=True)

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
# 5. Tabs (Added "Flood Risk Map")
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "DEM & Flow Simulation", "Burned Areas", "Flood Risk Map", "Slope Map", "Aspect Map", 
    "Retention Time", "GeoTIFF Export", "Nutrient Leaching", 
    "Flow Accumulation", "TWI", "Curvature", "Scenario GIFs", 
    "Burned-Area Hydro Impacts"
])

# -----------------------------------------------------------------------------
# 6. Georeference bounding box (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# -----------------------------------------------------------------------------
# 7. Parameter Inputs in "DEM & Flow Simulation" Tab
# -----------------------------------------------------------------------------
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
        burn_threshold = st.slider("Burned Area Threshold (Red Band)", 0, 255, 240, 1, key="burn_threshold")

# -----------------------------------------------------------------------------
# 8. Nutrient Leaching Tab
# -----------------------------------------------------------------------------
with tabs[7]:  # Adjusted index due to new tab
    st.header("Nutrient Leaching")
    with st.expander("Nutrient Leaching Parameters", expanded=True):
        nutrient = st.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0, key="nutrient")
        retention = st.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05, key="retention")
        erosion = st.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05, key="erosion")

# -----------------------------------------------------------------------------
# 9. Scenario GIFs Tab
# -----------------------------------------------------------------------------
with tabs[11]:  # Adjusted index due to new tab
    st.header("Scenario GIFs")
    with st.expander("GIF Settings", expanded=True):
        gif_frames = st.number_input("GIF Frames", value=10, step=1, key="gif_frames")
        gif_fps = st.number_input("GIF FPS", value=2, step=1, key="gif_fps")

# -----------------------------------------------------------------------------
# 10. Processing Logic
# -----------------------------------------------------------------------------
if uploaded_stl and run_button:
    # Retrieve parameters
    scale_val = st.session_state.scale
    offset_val = st.session_state.offset
    dem_min_val = st.session_state.dem_min
    dem_max_val = st.session_state.dem_max
    grid_res_val = st.session_state.grid_res
    rainfall_val = st.session_state.rainfall
    duration_val = st.session_state.duration
    area_val = st.session_state.area
    runoff_val = st.session_state.runoff
    recession_val = st.session_state.recession
    sim_hours_val = st.session_state.sim_hours
    storage_val = st.session_state.storage
    burn_factor_val = st.session_state.burn_factor
    burn_threshold_val = st.session_state.burn_threshold
    nutrient_val = st.session_state.nutrient
    retention_val = st.session_state.retention
    erosion_val = st.session_state.erosion
    gif_frames_val = st.session_state.gif_frames
    gif_fps_val = st.session_state.gif_fps

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
    yi = np.linspace(bottom_bound, top_bound, grid_res_val)  # Note: Assuming custom_bound was a typo for bottom_bound
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Create DEM
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min_val, dem_max_val)

    # Derivatives
    dx = (right_bound - left_bound) / (grid_res_val - 1)
    dy = (top_bound - bottom_bound) / (grid_res_val - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters, dy_meters = dx * meters_per_deg_lon, dy * meters_per_deg_lat
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # Flow simulation (simplified)
    area_m2 = area_val * 10000.0
    total_rain_m = (rainfall_val / 1000.0) * duration_val
    V_runoff = total_rain_m * area_m2 * runoff_val
    Q_peak = V_runoff / duration_val
    t = np.linspace(0, sim_hours_val, int(sim_hours_val * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= duration_val:
            Q[i] = Q_peak * (time / duration_val)
        else:
            Q[i] = Q_peak * np.exp(-recession_val * (time - duration_val))

    # Retention time
    retention_time = storage_val / (V_runoff / duration_val) if V_runoff > 0 else None

    # Nutrient leaching
    nutrient_load = nutrient_val * (1 - retention_val) * erosion_val * area_val

    # Burned area processing with resampling
    burned_mask_resampled = None
    if uploaded_burned:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
                tmp_tif.write(uploaded_burned.read())
                with rasterio.open(tmp_tif.name) as src:
                    if src.count < 3:
                        st.warning("The burned area TIFF must be an RGB image with 3 bands.")
                    else:
                        red = src.read(1)
                        burned_mask = (red > burn_threshold_val).astype(np.float32)
                        # Define target transform for DEM grid
                        transform = from_origin(left_bound - dx/2, top_bound + dy/2, dx, -dy)
                        burned_mask_resampled = np.zeros((grid_res_val, grid_res_val), dtype=np.float32)
                        reproject(
                            source=burned_mask,
                            destination=burned_mask_resampled,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=src.crs,
                            resampling=Resampling.nearest
                        )
        except Exception as e:
            st.error(f"Error processing burned area TIFF: {e}")
            burned_mask_resampled = None

    # Runoff coefficient grid
    runoff_coeff_grid = np.full_like(grid_z, runoff_val)
    if burned_mask_resampled is not None:
        runoff_coeff_grid[burned_mask_resampled == 1] *= burn_factor_val
    else:
        st.info("No burned area data uploaded; using uniform runoff coefficient.")

    # Flow accumulation using pysheds
    grid = Grid.Grid.from_array(np.flipud(grid_z), data_name='dem', 
                                bbox=(left_bound, bottom_bound, right_bound, top_bound))
    flooded_dem = grid.fill_depressions('dem')
    inflated_dem = grid.resolve_flats(flooded_dem)
    fdir = grid.flowdir(inflated_dem)
    flow_acc = grid.accumulation(fdir)
    flow_acc_to_plot = np.flipud(flow_acc)  # Flip back for plotting consistency

    # Flood risk map
    risk_map = flow_acc * runoff_coeff_grid  # pysheds flow_acc is already in DEM orientation after flip

    # Additional terrain derivatives
    twi = np.log((flow_acc_to_plot + 1) / (np.tan(np.radians(slope)) + 0.05))
    curvature = convolve(grid_z, np.ones((3, 3)) / 9, mode='reflect')

    # Helper for plotting
    def plot_with_correct_aspect(ax, data, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, origin='lower',
                       extent=(left_bound, right_bound, bottom_bound, top_bound),
                       vmin=vmin, vmax=vmax)
        aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound)
        aspect_ratio *= (meters_per_deg_lat / meters_per_deg_lon)
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        return im

    # 11. Display in each tab
    # DEM & Flow Simulation tab
    with tabs[0]:
        st.header("DEM & Flow Simulation")
        fig, ax = plt.subplots()
        im = plot_with_correct_aspect(ax, grid_z, 'terrain', vmin=dem_min_val, vmax=dem_max_val)
        step = max(1, grid_res_val // 20)
        ax.quiver(
            grid_x[::step, ::step], grid_y[::step, ::step],
            -dz_dx[::step, ::step], -dz_dy[::step, ::step],
            color='blue', scale=1e5, width=0.0025
        )
        st.pyplot(fig)

    # Burned Areas tab
    with tabs[1]:
        st.header("Burned Areas")
        if burned_mask_resampled is not None:
            fig, ax = plt.subplots()
            cmap = ListedColormap(['red', 'black'])
            im = ax.imshow(
                burned_mask_resampled, cmap=cmap, origin='upper',
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound)
            aspect_ratio *= (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Burned', 'Non-burned'])
            st.pyplot(fig)
        else:
            st.write("No burned area data uploaded or TIFF processing failed.")

    # Flood Risk Map tab (New)
    with tabs[2]:
        st.header("Flood Risk Map")
        st.write("Areas with higher values indicate increased flood risk due to burned areas and water accumulation.")
        fig, ax = plt.subplots()
        im = plot_with_correct_aspect(ax, risk_map, 'RdYlBu')
        fig.colorbar(im, ax=ax, label="Flood Risk Index")
        st.pyplot(fig)

    # Slope Map tab
    with tabs[3]:
        st.header("Slope Map")
        with st.expander("Visualization Options"):
            slope_vmin = st.number_input("Slope Min", value=0.0, key="slope_vmin")
            slope_vmax = st.number_input("Slope Max", value=90.0, key="slope_vmax")
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno"], key="slope_cmap")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, slope, slope_cmap, vmin=slope_vmin, vmax=slope_vmax)
        st.pyplot(fig)

    # Aspect Map tab
    with tabs[4]:
        st.header("Aspect Map")
        with st.expander("Visualization Options"):
            aspect_vmin = st.number_input("Aspect Min", value=0.0, key="aspect_vmin")
            aspect_vmax = st.number_input("Aspect Max", value=360.0, key="aspect_vmax")
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv"], key="aspect_cmap")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, aspect, aspect_cmap, vmin=aspect_vmin, vmax=aspect_vmax)
        st.pyplot(fig)

    # Retention Time tab
    with tabs[5]:
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.write("No effective runoff → Retention time not applicable.")

    # GeoTIFF Export tab
    with tabs[6]:
        st.subheader("GeoTIFF Export")
        st.write("Export functionality to be implemented (placeholder).")

    # Nutrient Leaching tab
    with tabs[7]:
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    # Flow Accumulation tab
    with tabs[8]:
        st.header("Flow Accumulation")
        fig, ax = plt.subplots()
        im = plot_with_correct_aspect(ax, flow_acc_to_plot, 'Blues')
        fig.colorbar(im, ax=ax, label="Flow Accumulation (cells)")
        st.pyplot(fig)

    # TWI tab
    with tabs[9]:
        st.header("Topographic Wetness Index")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, twi, 'RdYlBu')
        st.pyplot(fig)

    # Curvature tab
    with tabs[10]:
        st.header("Curvature Analysis")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, curvature, 'Spectral')
        st.pyplot(fig)

    # Scenario GIFs tab
    with tabs[11]:
        st.write("GIF generation to be implemented (placeholder).")

    # Burned-Area Hydro Impacts Tab
    with tabs[12]:
        st.header("Burned-Area Hydro Impacts")
        st.markdown("""
        **How Burned Areas Affect Hydrogeology**  
        - **Reduced Infiltration** in burned patches → More surface runoff  
        - **Accelerated Erosion** (less vegetative cover) → Higher sediment loads  
        - **Decreased Groundwater Recharge** (if infiltration is lower)  
        - **Nutrient & Ash Loading** in runoff → Potential water quality issues  
        """)

        st.subheader("Advanced Burned-Area Parameters")
        base_infiltration = st.number_input(
            "Base Infiltration Rate (mm/hr)", value=10.0, step=1.0, min_value=0.0
        )
        infiltration_reduction = st.slider(
            "Infiltration Reduction in Burned Areas (fraction)",
            0.0, 1.0, 0.5, 0.05
        )
        base_erosion_rate = st.number_input(
            "Base Erosion Rate (tons/ha)", value=0.5, step=0.1
        )
        erosion_multiplier_burned = st.slider(
            "Erosion Multiplier in Burned Areas",
            1.0, 5.0, 2.0, 0.1
        )

        if burned_mask_resampled is not None:
            infiltration_map = np.full_like(grid_z, base_infiltration)
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask_resampled
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()

            erosion_map = np.full_like(grid_z, base_erosion_rate)
            erosion_map[burned_mask_resampled == 1] *= erosion_multiplier_burned
            total_erosion = erosion_map.sum()

            st.write(f"**Infiltration Volume (mm * cell_area):** ~{infiltration_volume_total:.2f} mm-hr equivalent")
            st.write(f"**Estimated Erosion (sum):** {total_erosion:.2f} tons")

            infiltration_ratio = (infiltration_map.mean() / base_infiltration)
            new_runoff_coefficient = runoff_val + burn_factor_val * (1.0 - infiltration_ratio)
            new_runoff_coefficient = np.clip(new_runoff_coefficient, 0.0, 1.0)
            st.write(f"**Adjusted Runoff Coefficient (approx):** {new_runoff_coefficient:.2f}")

            burned_fraction = burned_mask_resampled.mean()
            nutrient_load_burned = nutrient_load * (1.0 + burned_fraction * 0.3)
            st.write(f"**Potential Increase in Nutrient Load:** from {nutrient_load:.2f} to ~{nutrient_load_burned:.2f} kg")

            st.subheader("Infiltration Map (mm/hr)")
            fig, ax = plt.subplots()
            im = plot_with_correct_aspect(ax, infiltration_map, 'Greens')
            fig.colorbar(im, ax=ax, label="Infiltration Rate (mm/hr)")
            st.pyplot(fig)

            st.info("""
            **Interpretation**:  
            - Lower infiltration in burned areas increases surface runoff.  
            - Higher erosion in burned areas indicates potential soil loss.  
            - Nutrient loads may increase with reduced infiltration.
            """)
        else:
            st.warning("No burned area data; upload a TIFF to see advanced impacts.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
