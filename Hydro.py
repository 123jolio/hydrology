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
# 5. Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "DEM & Flow Simulation", "Burned Areas", "Slope Map", "Aspect Map", 
    "Retention Time", "GeoTIFF Export", "Nutrient Leaching", 
    "Flow Accumulation", "TWI", "Curvature", "Scenario GIFs", 
    "Burned-Area Hydro Impacts"   # <--- NEW TAB FOR ADDITIONAL CALCULATIONS
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
with tabs[6]:
    st.header("Nutrient Leaching")
    with st.expander("Nutrient Leaching Parameters", expanded=True):
        nutrient = st.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0, key="nutrient")
        retention = st.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05, key="retention")
        erosion = st.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05, key="erosion")

# -----------------------------------------------------------------------------
# 9. Scenario GIFs Tab
# -----------------------------------------------------------------------------
with tabs[10]:
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
    yi = np.linspace(bottom_bound, top_bound, grid_res_val)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Create DEM
    from scipy.interpolate import griddata
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

    # Burned area (RGB TIFF)
    burned_mask = None
    if uploaded_burned:
        try:
            import rasterio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_tif:
                tmp_tif.write(uploaded_burned.read())
                with rasterio.open(tmp_tif.name) as src:
                    # If < 3 bands, not RGB
                    if src.count < 3:
                        st.warning("The burned area TIFF must be an RGB image with 3 bands.")
                    else:
                        red = src.read(1)
                        # Simple threshold on the red band
                        burned_mask = (red > burn_threshold_val).astype(np.float32)
        except Exception as e:
            st.error(f"Error processing burned area TIFF: {e}")
            burned_mask = None

    # Additional terrain derivatives
    flow_acc = np.ones_like(grid_z)  # Placeholder
    twi = np.log((flow_acc + 1) / (np.tan(np.radians(slope)) + 0.05))
    curvature = convolve(grid_z, np.ones((3, 3)) / 9, mode='reflect')

    # Helper for plotting
    def plot_with_burned_overlay(ax, data, cmap, vmin=None, vmax=None, 
                                 burned_mask=None, show_burned=True, alpha=0.5):
        im = ax.imshow(data, cmap=cmap, origin='lower',
                       extent=(left_bound, right_bound, bottom_bound, top_bound),
                       vmin=vmin, vmax=vmax)
        if show_burned and (burned_mask is not None):
            burned_cmap = ListedColormap(['none', 'red'])
            ax.imshow(burned_mask, cmap=burned_cmap, origin='lower',
                      extent=(left_bound, right_bound, bottom_bound, top_bound),
                      alpha=alpha)
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
        with st.expander("Visualization Options"):
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="dem_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="dem_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, grid_z, 'terrain',
            vmin=dem_min_val, vmax=dem_max_val,
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
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
        if burned_mask is not None:
            fig, ax = plt.subplots()
            cmap = ListedColormap(['red', 'black'])
            im = ax.imshow(
                burned_mask, cmap=cmap, origin='upper',
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

    # Slope Map tab
    with tabs[2]:
        st.header("Slope Map")
        with st.expander("Visualization Options"):
            slope_vmin = st.number_input("Slope Min", value=0.0, key="slope_vmin")
            slope_vmax = st.number_input("Slope Max", value=90.0, key="slope_vmax")
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno"], key="slope_cmap")
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="slope_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="slope_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, slope, slope_cmap, 
            vmin=slope_vmin, vmax=slope_vmax,
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

    # Aspect Map tab
    with tabs[3]:
        st.header("Aspect Map")
        with st.expander("Visualization Options"):
            aspect_vmin = st.number_input("Aspect Min", value=0.0, key="aspect_vmin")
            aspect_vmax = st.number_input("Aspect Max", value=360.0, key="aspect_vmax")
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv"], key="aspect_cmap")
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="aspect_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="aspect_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, aspect, aspect_cmap, 
            vmin=aspect_vmin, vmax=aspect_vmax,
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

    # Retention Time tab
    with tabs[4]:
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.write("No effective runoff → Retention time not applicable.")

    # GeoTIFF Export tab
    with tabs[5]:
        st.subheader("GeoTIFF Export")
        st.write("Export functionality to be implemented (placeholder).")

    # Nutrient Leaching tab
    with tabs[6]:
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    # Flow Accumulation tab
    with tabs[7]:
        st.header("Flow Accumulation")
        with st.expander("Visualization Options"):
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="flow_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="flow_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, flow_acc, 'Blues',
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

    # TWI tab
    with tabs[8]:
        st.header("Topographic Wetness Index")
        with st.expander("Visualization Options"):
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="twi_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="twi_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, twi, 'RdYlBu',
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

    # Curvature tab
    with tabs[9]:
        st.header("Curvature Analysis")
        with st.expander("Visualization Options"):
            show_burned = st.checkbox("Show Burned Areas Overlay", value=False, key="curv_burned")
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, 0.5, 0.1, key="curv_alpha")
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, curvature, 'Spectral',
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.pyplot(fig)

    # Scenario GIFs tab
    with tabs[10]:
        st.write("GIF generation to be implemented (placeholder).")

    # -----------------------------------------------------------------------------
    # 12. NEW: Burned-Area Hydro Impacts Tab
    # -----------------------------------------------------------------------------
    with tabs[11]:
        st.header("Burned-Area Hydro Impacts")

        st.markdown("""
        **How Burned Areas Affect Hydrogeology**  
        - **Reduced Infiltration** in burned patches → More surface runoff  
        - **Accelerated Erosion** (less vegetative cover) → Higher sediment loads  
        - **Decreased Groundwater Recharge** (if infiltration is lower)  
        - **Nutrient & Ash Loading** in runoff → Potential water quality issues  
        """)

        # Additional user inputs for advanced analysis
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

        # Compute infiltration map
        if burned_mask is not None:
            # Convert rainfall to mm
            infiltration_map = np.full_like(grid_z, base_infiltration)
            # Reduce infiltration in burned cells
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask
            # The infiltration_map now has lower infiltration rates where burned_mask=1

            # Potential infiltration volume over entire area
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()

            # Erosion estimate (very simplified)
            # Base erosion where burned_mask=0, multiplied by erosion_multiplier_burned where burned_mask=1
            erosion_map = np.full_like(grid_z, base_erosion_rate)
            erosion_map[burned_mask == 1] *= erosion_multiplier_burned
            total_erosion = erosion_map.sum()  # Summed over the DEM

            st.write(f"**Infiltration Volume (mm * cell_area) over the domain:** ~{infiltration_volume_total:.2f} mm-hr equivalent")
            st.write(f"**Estimated Erosion (placeholder, sum of map):** {total_erosion:.2f} tons")

            # Simple example of how infiltration might affect final runoff
            # If infiltration is reduced, more water becomes surface runoff
            # We can approximate a new effective runoff_coefficient
            # ratio of infiltration to total rainfall
            infiltration_ratio = (infiltration_map.mean() / base_infiltration)
            # We scale the original runoff_coefficient by (1 + burn_factor*(1 - infiltration_ratio)) as a placeholder
            new_runoff_coefficient = runoff_val + burn_factor_val * (1.0 - infiltration_ratio)
            new_runoff_coefficient = np.clip(new_runoff_coefficient, 0.0, 1.0)
            st.write(f"**Adjusted Runoff Coefficient** (approx): {new_runoff_coefficient:.2f}")

            # Potential water-quality placeholder
            # If infiltration is lower, more nutrients or ash are carried away
            # We can assume a fraction of nutrient_load is multiplied by the fraction of burned area
            burned_fraction = (burned_mask.mean())  # fraction of domain that is burned
            nutrient_load_burned = nutrient_load * (1.0 + burned_fraction * 0.3)  # +30% for example
            st.write(f"**Potential Increase in Nutrient Load** due to burned area: from {nutrient_load:.2f} to ~{nutrient_load_burned:.2f} kg")

            # Visualize infiltration map
            st.subheader("Infiltration Map (mm/hr)")
            fig, ax = plt.subplots()
            im = ax.imshow(
                infiltration_map,
                cmap='Greens',
                origin='lower',
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound)
            aspect_ratio *= (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Infiltration Rate (mm/hr)")
            st.pyplot(fig)

            st.info("""
            **Interpretation**:  
            - The infiltration map is reduced where burned_mask=1, simulating a hydrophobic or crusted soil.  
            - Lower infiltration → higher surface runoff → potentially higher peak flows and less groundwater recharge.
            - The simplified erosion map indicates how burned areas might accelerate soil loss.
            - Nutrient or ash loads can also increase if infiltration is reduced and runoff is higher.
            """)

        else:
            st.warning("No burned area detected or TIFF missing. Please upload a valid burned-area TIFF to see advanced impacts.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
