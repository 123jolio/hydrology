import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import os
from PIL import Image
from scipy.ndimage import convolve
from matplotlib.colors import ListedColormap
import pysheds.grid as pysheds
from rasterio.crs import CRS
import plotly.graph_objects as go  # For 3D STL viewer

# -----------------------------------------------------------------------------
# 1. Page Config & Matplotlib Style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Hydrogeology & DEM Analysis", layout="wide", initial_sidebar_state="collapsed")
plt.style.use('dark_background')

# -----------------------------------------------------------------------------
# 2. Dark Mode CSS + Header Styling
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
.matplotlib-figure {
    background-color: #1E1E1E;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Header with Logo and Title
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
# 5. Define Tabs (including extra analysis tabs and 3D viewer)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "DEM & Flow Simulation", "Burned Areas", "Flood Risk Map", "Slope Map", "Aspect Map", 
    "Retention Time", "GeoTIFF Export", "Nutrient Leaching", "Flow Accumulation", 
    "TWI", "Curvature", "Scenario GIFs", "Burned-Area Hydro Impacts", "3D STL Viewer"
])

# -----------------------------------------------------------------------------
# 6. Georeference Bounds
# -----------------------------------------------------------------------------
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# -----------------------------------------------------------------------------
# 7. Parameter Inputs (in DEM & Flow Simulation Tab)
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
with tabs[7]:
    st.header("Nutrient Leaching")
    with st.expander("Nutrient Leaching Parameters", expanded=True):
        nutrient = st.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0, key="nutrient")
        retention = st.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05, key="retention")
        erosion = st.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05, key="erosion")

# -----------------------------------------------------------------------------
# 9. Scenario GIFs Tab
# -----------------------------------------------------------------------------
with tabs[11]:
    st.header("Scenario GIFs")
    with st.expander("GIF Settings", expanded=True):
        gif_frames = st.number_input("GIF Frames", value=10, step=1, key="gif_frames")
        gif_fps = st.number_input("GIF FPS", value=2, step=1, key="gif_fps")

# -----------------------------------------------------------------------------
# 10. Processing and Display Logic
# -----------------------------------------------------------------------------
if uploaded_stl and run_button:
    # Save uploaded STL bytes (for DEM processing and 3D viewer)
    stl_bytes = uploaded_stl.getvalue()
    
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

    # -------------------------------
    # Process STL for DEM generation
    # -------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(stl_bytes)
        stl_mesh = mesh.Mesh.from_file(tmp_stl.name)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_adj = (z_raw * scale_val) + offset_val

    # Interpolate DEM over a grid
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = bottom_bound + (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)
    xi = np.linspace(left_bound, right_bound, grid_res_val)
    yi = np.linspace(bottom_bound, top_bound, grid_res_val)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min_val, dem_max_val)
    # (Optionally warn/replace NaNs)
    if np.any(np.isnan(grid_z)):
        st.warning("DEM contains NaN values; replacing with minimum elevation.")
        grid_z = np.nan_to_num(grid_z, nan=dem_min_val)

    # Compute terrain derivatives
    dx = (right_bound - left_bound) / (grid_res_val - 1)
    dy = (top_bound - bottom_bound) / (grid_res_val - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters, dy_meters = dx * meters_per_deg_lon, dy * meters_per_deg_lat
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # -------------------------------
    # Flow simulation (simplified)
    # -------------------------------
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
    retention_time = storage_val / (V_runoff / duration_val) if V_runoff > 0 else None

    # -------------------------------
    # Nutrient Leaching Calculation
    # -------------------------------
    nutrient_load = nutrient_val * (1 - retention_val) * erosion_val * area_val

    # -------------------------------
    # Burned Area Processing
    # -------------------------------
    burned_mask = None
    if uploaded_burned:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
                tmp_tif.write(uploaded_burned.read())
                with rasterio.open(tmp_tif.name) as src:
                    src_crs = src.crs if src.crs is not None else CRS.from_epsg(4326)
                    if src.count < 3:
                        st.warning("The burned area TIFF must be an RGB image with 3 bands.")
                    else:
                        red = src.read(1)
                        burned_mask = (red > burn_threshold_val).astype(np.float32)
                        burned_mask[burned_mask > 0] = 1  # Force binary
                        # If shape differs from DEM, reproject
                        if burned_mask.shape != grid_z.shape:
                            st.write(f"Resampling burned mask from {burned_mask.shape} to {grid_z.shape}")
                            resampled_mask = np.zeros_like(grid_z, dtype=np.float32)
                            reproject(
                                source=burned_mask,
                                destination=resampled_mask,
                                src_transform=src.transform,
                                src_crs=src_crs,
                                dst_transform=from_origin(left_bound, top_bound, dx, dy),
                                dst_crs="EPSG:4326",
                                resampling=Resampling.nearest
                            )
                            burned_mask = resampled_mask
                            st.write(f"Burned mask resampled to {burned_mask.shape}")
        except Exception as e:
            st.error(f"Error processing burned area TIFF: {e}")
            burned_mask = None

    # Compute runoff coefficient grid
    runoff_coeff_grid = np.full_like(grid_z, runoff_val)
    if burned_mask is not None:
        runoff_coeff_grid[burned_mask == 1] *= burn_factor_val
    else:
        st.write("No burned area data uploaded; using uniform runoff coefficient.")

    # -------------------------------
    # Flow Accumulation using pysheds
    # -------------------------------
    # Use the strategy of converting DEM to float32 so that negative nodata values are representable.
    grid_z = grid_z.astype(np.float32)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_dem:
        transform = from_origin(left_bound, top_bound, dx, -dy)
        with rasterio.open(
            tmp_dem.name, 'w', driver='GTiff', height=grid_res_val, width=grid_res_val,
            count=1, dtype=grid_z.dtype, crs=CRS.from_epsg(4326), transform=transform,
            nodata=-9999.0
        ) as dst:
            dst.write(np.flipud(grid_z), 1)
        grid = pysheds.Grid.from_raster(tmp_dem.name, nodata=-9999.0)
        grid.fill_depressions(data='raster', out_name='flooded_dem')
        grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
        grid.flowdir(data='inflated_dem', out_name='fdir')
        grid.accumulation(data='fdir', out_name='flow_acc')
        flow_acc = grid.flow_acc
        flow_acc_to_plot = np.flipud(flow_acc)  # Flip for plotting consistency
    os.unlink(tmp_dem.name)

    # Flood risk map (combine flow accumulation and runoff coefficient grid)
    risk_map = flow_acc * runoff_coeff_grid

    # -------------------------------
    # Additional Terrain Derivatives
    # -------------------------------
    twi = np.log((flow_acc_to_plot + 1) / (np.tan(np.radians(slope)) + 0.05))
    curvature = convolve(grid_z, np.ones((3, 3)) / 9, mode='reflect')

    # -----------------------------------------------------------------------------
    # Helper: Plotting Function with Correct Aspect Ratio
    # -----------------------------------------------------------------------------
    def plot_with_correct_aspect(ax, data, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, origin='lower',
                       extent=(left_bound, right_bound, bottom_bound, top_bound),
                       vmin=vmin, vmax=vmax)
        aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * (meters_per_deg_lat / meters_per_deg_lon)
        ax.set_aspect(aspect_ratio)
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        return im

    # -----------------------------------------------------------------------------
    # Display Results in Tabs
    # -----------------------------------------------------------------------------
    with tabs[0]:  # DEM & Flow Simulation
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, grid_z, 'terrain', vmin=dem_min_val, vmax=dem_max_val)
        step = max(1, grid_res_val // 20)
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step],
                  -dz_dx[::step, ::step], -dz_dy[::step, ::step],
                  color='blue', scale=1e5, width=0.0025)
        st.pyplot(fig)

    with tabs[1]:  # Burned Areas
        st.header("Burned Areas")
        if burned_mask is not None:
            fig, ax = plt.subplots()
            cmap_burn = ListedColormap(['black', 'red'])
            im = ax.imshow(burned_mask, cmap=cmap_burn, origin='lower',
                           extent=(left_bound, right_bound, bottom_bound, top_bound))
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Non-burned', 'Burned'])
            st.pyplot(fig)
        else:
            st.write("No burned area data uploaded or TIFF processing failed.")

    with tabs[2]:  # Flood Risk Map
        st.header("Flood Risk Map")
        st.write("Areas with higher values indicate increased flood risk due to burned areas and water accumulation.")
        fig, ax = plt.subplots()
        im = plot_with_correct_aspect(ax, risk_map, 'RdYlBu')
        fig.colorbar(im, ax=ax, label='Flood Risk')
        st.pyplot(fig)

    with tabs[3]:  # Slope Map
        with st.expander("Visualization Options"):
            slope_vmin = st.number_input("Slope Min", value=0.0, key="slope_vmin")
            slope_vmax = st.number_input("Slope Max", value=90.0, key="slope_vmax")
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno"], key="slope_cmap")
        st.subheader("Slope Map")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, slope, slope_cmap, vmin=slope_vmin, vmax=slope_vmax)
        st.pyplot(fig)

    with tabs[4]:  # Aspect Map
        with st.expander("Visualization Options"):
            aspect_vmin = st.number_input("Aspect Min", value=0.0, key="aspect_vmin")
            aspect_vmax = st.number_input("Aspect Max", value=360.0, key="aspect_vmax")
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv"], key="aspect_cmap")
        st.subheader("Aspect Map")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, aspect, aspect_cmap, vmin=aspect_vmin, vmax=aspect_vmax)
        st.pyplot(fig)

    with tabs[5]:  # Retention Time
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.write("No effective runoff → Retention time not applicable.")

    with tabs[6]:  # GeoTIFF Export (placeholder)
        st.subheader("GeoTIFF Export")
        st.write("Export functionality to be implemented.")

    with tabs[7]:  # Nutrient Leaching
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    with tabs[8]:  # Flow Accumulation
        st.header("Flow Accumulation")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, flow_acc_to_plot, 'Blues')
        st.pyplot(fig)

    with tabs[9]:  # TWI
        st.header("Topographic Wetness Index")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, twi, 'RdYlBu')
        st.pyplot(fig)

    with tabs[10]:  # Curvature
        st.header("Curvature Analysis")
        fig, ax = plt.subplots()
        plot_with_correct_aspect(ax, curvature, 'Spectral')
        st.pyplot(fig)

    with tabs[11]:  # Scenario GIFs
        st.write("GIF generation functionality to be implemented.")

    with tabs[12]:  # Burned-Area Hydro Impacts
        st.header("Burned-Area Hydro Impacts")
        st.markdown("""
        **Hydrogeologic Impacts of Burned Areas**  
        - **Reduced Infiltration**: Burned areas may become hydrophobic, increasing runoff.  
        - **Accelerated Erosion**: Loss of vegetation can boost soil erosion.  
        - **Decreased Groundwater Recharge**: Lower infiltration may reduce aquifer recharge.  
        - **Water Quality Impacts**: Enhanced runoff can transport ash and nutrients into water bodies.
        """)
        st.subheader("Advanced Burned-Area Parameters")
        base_infiltration = st.number_input("Base Infiltration Rate (mm/hr)", value=10.0, step=1.0)
        infiltration_reduction = st.slider("Infiltration Reduction in Burned Areas (fraction)", 0.0, 1.0, 0.5, 0.05)
        base_erosion_rate = st.number_input("Base Erosion Rate (tons/ha)", value=0.5, step=0.1)
        erosion_multiplier_burned = st.slider("Erosion Multiplier in Burned Areas", 1.0, 5.0, 2.0, 0.1)
        if burned_mask is not None:
            infiltration_map = np.full_like(grid_z, base_infiltration, dtype=np.float32)
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()
            st.write(f"**Infiltration Volume:** ~{infiltration_volume_total:.2f} mm-hr equiv.")
            erosion_map = np.full_like(grid_z, base_erosion_rate, dtype=np.float32)
            erosion_map[burned_mask == 1] *= erosion_multiplier_burned
            total_erosion = erosion_map.sum()
            st.write(f"**Estimated Erosion:** {total_erosion:.2f} tons")
            infiltration_ratio = infiltration_map.mean() / base_infiltration
            new_runoff_coefficient = runoff_val + burn_factor_val * (1.0 - infiltration_ratio)
            new_runoff_coefficient = np.clip(new_runoff_coefficient, 0.0, 1.0)
            st.write(f"**Adjusted Runoff Coefficient:** {new_runoff_coefficient:.2f}")
            burned_fraction = burned_mask.mean()
            nutrient_load_burned = nutrient_load * (1.0 + burned_fraction * 0.3)
            st.write(f"**Potential Nutrient Load:** from {nutrient_load:.2f} kg to ~{nutrient_load_burned:.2f} kg")
            st.subheader("Infiltration Map (mm/hr)")
            fig, ax = plt.subplots()
            im = ax.imshow(infiltration_map, cmap='Greens', origin='lower',
                           extent=(left_bound, right_bound, bottom_bound, top_bound))
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Infiltration Rate (mm/hr)")
            st.pyplot(fig)
            st.info("""
            **Interpretation**:  
            - Burned areas show reduced infiltration, leading to increased runoff and erosion.  
            - The adjusted runoff coefficient and nutrient load indicate potential water-quality impacts.
            """)
        else:
            st.warning("No burned area detected. Please upload a valid burned-area TIFF.")

    with tabs[13]:  # 3D STL Viewer
        st.header("3D STL Viewer")
        if stl_bytes is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_3d:
                    tmp_3d.write(stl_bytes)
                    stl_path = tmp_3d.name
                stl_mesh_3d = mesh.Mesh.from_file(stl_path)
                vertices = stl_mesh_3d.vectors.reshape(-1, 3)
                n_triangles = len(stl_mesh_3d.vectors)
                faces = np.array([[3*i, 3*i+1, 3*i+2] for i in range(n_triangles)])
                fig = go.Figure(data=[
                    go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        opacity=0.5,
                        color='lightblue'
                    )
                ])
                fig.update_layout(scene=dict(aspectmode='data'))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying 3D STL: {e}")
        else:
            st.write("No STL file available.")
            
else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
