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
# 6. Georeference bounding box (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# -----------------------------------------------------------------------------
# 7. Parameter Inputs in "DEM & Flow Simulation" Tab
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("DEM & Flow Simulation")
    st.markdown("### Adjust the following parameters to customize your DEM and flow simulation:")
    
    with st.expander("Elevation Adjustments", expanded=True):
        st.markdown("**Scale Factor**: Multiplies elevation values to adjust vertical exaggeration (0.1–5.0). Higher values increase elevation height, affecting slope and flow patterns.")
        scale = st.slider("Scale Factor", 0.1, 5.0, 1.0, 0.1, key="scale")
        
        st.markdown("**Offset (m)**: Adds or subtracts a constant elevation (m) to shift the entire DEM. Positive values raise, negative lower the terrain, impacting flow direction.")
        offset = st.slider("Offset (m)", -100.0, 100.0, 0.0, 1.0, key="offset")
        
        st.markdown("**Min Elevation (m)**: Sets the minimum elevation for clipping (0–500 m). Use to focus on specific elevation ranges, affecting flow accumulation.")
        dem_min = st.number_input("Min Elevation (m)", value=0.0, step=1.0, key="dem_min")
        
        st.markdown("**Max Elevation (m)**: Sets the maximum elevation for clipping (0–500 m). Adjust to limit elevation range, influencing slope and water flow.")
        dem_max = st.number_input("Max Elevation (m)", value=500.0, step=1.0, key="dem_max")
        
        st.markdown("**Grid Resolution**: Sets the number of grid cells (100–1000) for DEM interpolation. Higher resolution increases detail but slows computation; adjust for balance.")
        grid_res = st.number_input("Grid Resolution", 100, 1000, 500, 50, key="grid_res")

    with st.expander("Flow & Retention", expanded=True):
        st.markdown("**Rainfall (mm/hr)**: Sets rainfall intensity (1–100 mm/hr). Higher values increase runoff and flow, affecting hydrographs and retention.")
        rainfall = st.number_input("Rainfall (mm/hr)", value=30.0, step=1.0, key="rainfall")
        
        st.markdown("**Duration (hr)**: Sets storm duration (0.1–24 hr). Longer durations increase total runoff volume, impacting peak flow and retention time.")
        duration = st.number_input("Duration (hr)", value=2.0, step=0.1, key="duration")
        
        st.markdown("**Area (ha)**: Sets the watershed area (0.1–100 ha). Larger areas increase total runoff, affecting flow volume and peak discharge.")
        area = st.number_input("Area (ha)", value=10.0, step=0.1, key="area")
        
        st.markdown("**Runoff Coefficient**: Fraction of rainfall becoming runoff (0.0–1.0). Higher values increase surface runoff, reducing infiltration; adjust to match land cover.")
        runoff = st.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05, key="runoff")
        
        st.markdown("**Recession Rate (1/hr)**: Controls how quickly flow decreases after rain (0.1–2.0). Higher values mean faster recession, affecting hydrograph shape.")
        recession = st.number_input("Recession Rate (1/hr)", value=0.5, step=0.1, key="recession")
        
        st.markdown("**Simulation Duration (hr)**: Sets the total simulation time (0.5–24 hr). Longer durations show longer hydrograph tails; adjust to capture full flow response.")
        sim_hours = st.number_input("Simulation Duration (hr)", value=6.0, step=0.5, key="sim_hours")
        
        st.markdown("**Storage Volume (m³)**: Sets water storage capacity (100–10000 m³). Higher volumes increase retention time, reducing peak flows; adjust for reservoirs or ponds.")
        storage = st.number_input("Storage Volume (m³)", value=5000.0, step=100.0, key="storage")

    with st.expander("Burned Area Effects", expanded=True):
        st.markdown("**Runoff Increase Factor**: Multiplies runoff in burned areas (0.0–2.0). Higher values increase runoff due to reduced infiltration post-fire; adjust to reflect burn severity.")
        burn_factor = st.slider("Runoff Increase Factor", 0.0, 2.0, 1.0, 0.1, key="burn_factor")
        
        st.markdown("**Burned Area Threshold**: Sets the pixel value threshold (0–255) for detecting burned areas in the selected band. Lower values detect more burned areas; adjust if maps lack variation.")
        burn_threshold = st.slider("Burned Area Threshold", 0, 255, 200, 1, key="burn_threshold")
        
        st.markdown("**Band for Burned Area Threshold**: Selects the color band (Red, Green, Blue) for thresholding burned areas. Choose based on TIFF data; Red often highlights burned areas, but Green/Blue may work better for specific images.")
        band_to_threshold = st.selectbox("Band for Burned Area Threshold", ["Red", "Green", "Blue"], key="band_threshold")

    # Process data once and store in session state
    if 'processed_data' not in st.session_state or run_button:
        try:
            st.session_state.processed_data = None
            if uploaded_stl:
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
                avg_lat = (top_bound + bottom_bound) / 2.0
                meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
                meters_per_deg_lat = 111320
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
                                    # Debug: Check the band data
                                    st.write(f"Band data max: {np.max(band_data)}, min: {np.min(band_data)}")
                                    burned_mask = (band_data > burn_threshold_val).astype(np.float32)
                                    st.write(f"Burned mask mean: {np.mean(burned_mask)}, shape: {burned_mask.shape}")

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
                                        st.write(f"Reprojected burned mask mean: {np.mean(burned_mask)}, shape: {burned_mask.shape}")
                                    else:
                                        st.warning("TIFF has no CRS. Resizing mask to match DEM shape (may be inaccurate).")
                                        zoom_factors = (grid_z.shape[0] / burned_mask.shape[0], grid_z.shape[1] / burned_mask.shape[1])
                                        burned_mask = zoom(burned_mask, zoom_factors, order=0)  # Nearest neighbor
                                        st.write(f"Resized burned mask mean: {np.mean(burned_mask)}, shape: {burned_mask.shape}")

                    except Exception as e:
                        st.error(f"Error processing burned area TIFF: {e}")
                        burned_mask = None

                # Calculate spatially varying parameters
                area_m2 = area_val * 10000.0
                total_rain_m = (rainfall_val / 1000.0) * duration_val
                if burned_mask is not None:
                    burned_fraction = np.mean(burned_mask)
                    unburned_fraction = 1 - burned_fraction
                    burned_runoff_coefficient = min(runoff_val * (1 + burn_factor_val), 1.0)
                    effective_runoff = runoff_val * unburned_fraction + burned_runoff_coefficient * burned_fraction
                    V_runoff_unburned = total_rain_m * area_m2 * unburned_fraction * runoff_val
                    V_runoff_burned = total_rain_m * area_m2 * burned_fraction * burned_runoff_coefficient
                else:
                    burned_fraction = 0
                    unburned_fraction = 1
                    effective_runoff = runoff_val
                    V_runoff_unburned = total_rain_m * area_m2 * runoff_val
                    V_runoff_burned = 0

                # Flow simulation with effective runoff
                V_runoff = total_rain_m * area_m2 * effective_runoff
                Q_peak = V_runoff / duration_val
                t = np.linspace(0, sim_hours_val, int(sim_hours_val * 60))
                Q = np.zeros_like(t)
                for i, time in enumerate(t):
                    if time <= duration_val:
                        Q[i] = Q_peak * (time / duration_val)
                    else:
                        Q[i] = Q_peak * np.exp(-recession_val * (time - duration_val))

                # Separate hydrographs
                if V_runoff > 0:
                    Q_unburned = (V_runoff_unburned / V_runoff) * Q
                    Q_burned = (V_runoff_burned / V_runoff) * Q if burned_mask is not None else np.zeros_like(t)
                else:
                    Q_unburned = np.zeros_like(t)
                    Q_burned = np.zeros_like(t)

                # Retention time
                retention_time = storage_val / (V_runoff / duration_val) if V_runoff > 0 else None

                # Nutrient leaching
                nutrient_load = nutrient_val * (1 - retention_val) * erosion_val * area_val

                # Additional terrain derivatives
                flow_acc = np.ones_like(grid_z)  # Placeholder
                twi = np.log((flow_acc + 1) / (np.tan(np.radians(slope)) + 0.05))
                curvature = convolve(grid_z, np.ones((3, 3)) / 9, mode='reflect')

                # Store processed data in session state
                st.session_state.processed_data = {
                    'grid_z': grid_z,
                    'slope': slope,
                    'aspect': aspect,
                    'burned_mask': burned_mask,
                    'flow_acc': flow_acc,
                    'twi': twi,
                    'curvature': curvature,
                    'Q': Q,
                    'Q_unburned': Q_unburned,
                    'Q_burned': Q_burned,
                    'retention_time': retention_time,
                    'nutrient_load': nutrient_load,
                    'V_runoff_unburned': V_runoff_unburned,
                    'V_runoff_burned': V_runoff_burned,
                    'V_runoff': V_runoff,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'dz_dx': dz_dx,
                    'dz_dy': dz_dy
                }
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.session_state.processed_data = None

    # If data is processed, retrieve parameters for display
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
        # Retrieve parameters from session state for display logic
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

        grid_z = st.session_state.processed_data['grid_z']
        slope = st.session_state.processed_data['slope']
        aspect = st.session_state.processed_data['aspect']
        burned_mask = st.session_state.processed_data['burned_mask']
        flow_acc = st.session_state.processed_data['flow_acc']
        twi = st.session_state.processed_data['twi']
        curvature = st.session_state.processed_data['curvature']
        Q = st.session_state.processed_data['Q']
        Q_unburned = st.session_state.processed_data['Q_unburned']
        Q_burned = st.session_state.processed_data['Q_burned']
        retention_time = st.session_state.processed_data['retention_time']
        nutrient_load = st.session_state.processed_data['nutrient_load']
        V_runoff_unburned = st.session_state.processed_data['V_runoff_unburned']
        V_runoff_burned = st.session_state.processed_data['V_runoff_burned']
        V_runoff = st.session_state.processed_data['V_runoff']
        grid_x = st.session_state.processed_data['grid_x']
        grid_y = st.session_state.processed_data['grid_y']
        dz_dx = st.session_state.processed_data['dz_dx']
        dz_dy = st.session_state.processed_data['dz_dy']

        st.header("DEM & Flow Simulation")
        st.markdown("### This tab displays the Digital Elevation Model (DEM) and flow simulation results.")
        st.markdown("**Instructions**: Adjust elevation and flow parameters to see how they impact the DEM visualization and hydrograph. If maps are uniform, check the STL file for terrain variability or adjust the scale factor and offset.")

        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the DEM for context. Useful when analyzing fire impacts on hydrology. Changes update dynamically without restarting the app.")
            if 'show_burned_dem' not in st.session_state:
                st.session_state.show_burned_dem = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_dem, key="dem_burned", on_change=lambda: st.session_state.update(show_burned_dem=not st.session_state.show_burned_dem))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make the DEM more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_dem' not in st.session_state:
                st.session_state.burn_alpha_dem = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_dem, 0.1, key="dem_alpha", on_change=lambda x: st.session_state.update(burn_alpha_dem=x))

        # Dynamic map update
        fig, ax = plt.subplots()
        plot_with_burned_overlay(
            ax, grid_z, 'terrain',
            vmin=dem_min_val, vmax=dem_max_val,
            burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
        )
        st.markdown("**DEM & Flow Visualization**: Shows the terrain elevation (m) with flow direction arrows (blue). Steeper slopes and burned areas affect flow paths; adjust parameters to see changes.")
        step = max(1, grid_res_val // 20)
        ax.quiver(
            grid_x[::step, ::step], grid_y[::step, ::step],
            -dz_dx[::step, ::step], -dz_dy[::step, ::step],
            color='blue', scale=1e5, width=0.0025
        )
        st.pyplot(fig)

        # Hydrograph plot (static for now, can be made dynamic if needed)
        st.subheader("Hydrograph")
        st.markdown("**Hydrograph**: Plots total flow (blue), unburned area flow (green), and burned area flow (red) over time (hr). Higher rainfall or runoff coefficients increase peak flows; adjust parameters to see impacts.")
        fig, ax = plt.subplots()
        ax.plot(t, Q, label="Total Flow", color='blue')
        if burned_mask is not None:
            ax.plot(t, Q_unburned, label="Unburned Area Flow", color='green')
            ax.plot(t, Q_burned, label="Burned Area Flow", color='red')
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow Rate (m³/hr)")
        ax.legend()
        st.pyplot(fig)

    # Burned Areas tab
    with tabs[1]:
        st.header("Burned Areas")
        st.markdown("### This tab shows the distribution of burned areas from the uploaded TIFF.")
        st.markdown("**Instructions**: Upload a georeferenced RGB TIFF to visualize burned areas. If no data appears, ensure the TIFF is valid, has 3 bands, and adjust the 'Burned Area Threshold' in the 'DEM & Flow Simulation' tab to detect burned regions.")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            burned_mask = st.session_state.processed_data['burned_mask']
            if burned_mask is not None:
                fig, ax = plt.subplots()
                cmap = ListedColormap(['red', 'black'])
                im = ax.imshow(
                    np.flipud(burned_mask), cmap=cmap, origin='upper',  # Flip to correct orientation
                    extent=(left_bound, right_bound, bottom_bound, top_bound)
                )
                st.markdown("**Burned Areas Map**: Red areas indicate burned regions (value=1), black areas are unburned (value=0). Adjust the threshold to capture more or fewer burned areas if the map is uniform.")
                aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound)
                aspect_ratio *= (meters_per_deg_lat / meters_per_deg_lon)
                ax.set_aspect(aspect_ratio)
                ax.set_xlabel('Longitude (°E)')
                ax.set_ylabel('Latitude (°N)')
                cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
                cbar.ax.set_yticklabels(['Unburned', 'Burned'])  # Updated for clarity
                st.pyplot(fig)
            else:
                st.write("No burned area data uploaded or TIFF processing failed.")
        else:
            st.write("No burned area data uploaded or TIFF processing failed.")

    # Slope Map tab
    with tabs[2]:
        st.header("Slope Map")
        st.markdown("### This tab displays the slope of the terrain derived from the DEM.")
        st.markdown("**Instructions**: Steeper slopes increase runoff and erosion. If the map lacks variation, check the STL file for terrain variability or adjust the scale factor and offset in 'DEM & Flow Simulation'.")
        
        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Slope Min**: Sets the minimum slope value (0–90 degrees) for visualization. Lower values focus on flatter areas; increase to highlight steeper slopes.")
            slope_vmin = st.number_input("Slope Min", value=0.0, key="slope_vmin")
            
            st.markdown("**Slope Max**: Sets the maximum slope value (0–90 degrees). Higher values show the full range of slopes; adjust to focus on specific ranges.")
            slope_vmax = st.number_input("Slope Max", value=90.0, key="slope_vmax")
            
            st.markdown("**Colormap**: Selects the color scheme (viridis, plasma, inferno) for the slope map. Choose based on preference for visualizing slope variation.")
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno"], key="slope_cmap")
            
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the slope map for context. Useful for identifying erosion risks in burned regions. Changes update dynamically.")
            if 'show_burned_slope' not in st.session_state:
                st.session_state.show_burned_slope = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_slope, key="slope_burned", on_change=lambda: st.session_state.update(show_burned_slope=not st.session_state.show_burned_slope))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make the slope map more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_slope' not in st.session_state:
                st.session_state.burn_alpha_slope = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_slope, 0.1, key="slope_alpha", on_change=lambda x: st.session_state.update(burn_alpha_slope=x))
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            slope = st.session_state.processed_data['slope']
            burned_mask = st.session_state.processed_data['burned_mask']
            fig, ax = plt.subplots()
            plot_with_burned_overlay(
                ax, slope, slope_cmap, 
                vmin=slope_vmin, vmax=slope_vmax,
                burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
            )
            st.markdown("**Slope Map**: Shows terrain slope in degrees (0–90°), with steeper areas indicating higher runoff and erosion potential. Use sliders to adjust the range and colormap for better visualization.")
            st.pyplot(fig)

    # Aspect Map tab
    with tabs[3]:
        st.header("Aspect Map")
        st.markdown("### This tab displays the aspect (direction) of the terrain derived from the DEM.")
        st.markdown("**Instructions**: Aspect indicates flow direction (0–360°). If the map lacks variation, check the STL file or adjust elevation parameters in 'DEM & Flow Simulation'.")
        
        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Aspect Min**: Sets the minimum aspect value (0–360 degrees) for visualization. Adjust to focus on specific flow directions.")
            aspect_vmin = st.number_input("Aspect Min", value=0.0, key="aspect_vmin")
            
            st.markdown("**Aspect Max**: Sets the maximum aspect value (0–360 degrees). Adjust to limit the range, highlighting specific flow directions.")
            aspect_vmax = st.number_input("Aspect Max", value=360.0, key="aspect_vmax")
            
            st.markdown("**Colormap**: Selects the color scheme (twilight, hsv) for the aspect map. Choose based on preference for visualizing flow direction.")
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv"], key="aspect_cmap")
            
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the aspect map for context. Useful for identifying flow paths in burned regions. Changes update dynamically.")
            if 'show_burned_aspect' not in st.session_state:
                st.session_state.show_burned_aspect = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_aspect, key="aspect_burned", on_change=lambda: st.session_state.update(show_burned_aspect=not st.session_state.show_burned_aspect))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make the aspect map more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_aspect' not in st.session_state:
                st.session_state.burn_alpha_aspect = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_aspect, 0.1, key="aspect_alpha", on_change=lambda x: st.session_state.update(burn_alpha_aspect=x))
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            aspect = st.session_state.processed_data['aspect']
            burned_mask = st.session_state.processed_data['burned_mask']
            fig, ax = plt.subplots()
            plot_with_burned_overlay(
                ax, aspect, aspect_cmap, 
                vmin=aspect_vmin, vmax=aspect_vmax,
                burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
            )
            st.markdown("**Aspect Map**: Shows terrain aspect in degrees (0–360°), indicating flow direction. Use sliders to adjust range and colormap for better visualization.")
            st.pyplot(fig)

    # Retention Time tab
    with tabs[4]:
        st.header("Retention Time")
        st.markdown("### This tab estimates how long water is retained in the watershed.")
        st.markdown("**Instructions**: Retention time depends on storage volume, runoff, and rainfall. If 'No effective runoff,' increase rainfall, runoff coefficient, or area in 'DEM & Flow Simulation'.")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            retention_time = st.session_state.processed_data['retention_time']
            if retention_time is not None:
                st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
                st.markdown("**Retention Time**: Indicates how long water is held before draining, based on storage volume and runoff. Higher storage or lower runoff increases retention; adjust parameters to test scenarios.")
            else:
                st.write("No effective runoff → Retention time not applicable.")

    # GeoTIFF Export tab
    with tabs[5]:
        st.header("GeoTIFF Export")
        st.markdown("### This tab is for exporting analysis results as GeoTIFF files.")
        st.markdown("**Instructions**: Export functionality is not yet implemented. Future updates will allow saving maps like infiltration, erosion, and runoff potential as georeferenced TIFFs for GIS use.")

    # Nutrient Leaching tab
    with tabs[6]:
        st.header("Nutrient Leaching")
        st.markdown("### This tab estimates nutrient leaching from soil due to erosion and runoff.")
        st.markdown("**Instructions**: Adjust soil nutrient, retention, and erosion factors. If the load seems too low or high, increase soil nutrient or erosion factor, or decrease retention to reflect post-fire conditions.")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            nutrient_load = st.session_state.processed_data['nutrient_load']
            st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    # Flow Accumulation tab
    with tabs[7]:
        st.header("Flow Accumulation")
        st.markdown("### This tab shows accumulated flow across the terrain.")
        st.markdown("**Instructions**: Flow accumulation indicates water volume downstream. If the map is uniform, ensure the DEM has varied slopes and adjust grid resolution for detail.")
        
        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the flow accumulation map. Useful for identifying flow impacts in burned regions.")
            if 'show_burned_flow' not in st.session_state:
                st.session_state.show_burned_flow = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_flow, key="flow_burned", on_change=lambda: st.session_state.update(show_burned_flow=not st.session_state.show_burned_flow))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make flow accumulation more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_flow' not in st.session_state:
                st.session_state.burn_alpha_flow = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_flow, 0.1, key="flow_alpha", on_change=lambda x: st.session_state.update(burn_alpha_flow=x))
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            flow_acc = st.session_state.processed_data['flow_acc']
            burned_mask = st.session_state.processed_data['burned_mask']
            fig, ax = plt.subplots()
            plot_with_burned_overlay(
                ax, flow_acc, 'Blues',
                burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
            )
            st.markdown("**Flow Accumulation Map**: Shows water accumulation (arbitrary units). Higher values indicate areas receiving more flow; adjust DEM parameters for variability.")
            st.pyplot(fig)

    # TWI tab
    with tabs[8]:
        st.header("Topographic Wetness Index")
        st.markdown("### This tab shows areas prone to saturation based on terrain.")
        st.markdown("**Instructions**: TWI indicates potential wetness. If uniform, check slope variability in the STL or adjust grid resolution.")
        
        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the TWI map. Useful for identifying wetness changes in burned regions.")
            if 'show_burned_twi' not in st.session_state:
                st.session_state.show_burned_twi = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_twi, key="twi_burned", on_change=lambda: st.session_state.update(show_burned_twi=not st.session_state.show_burned_twi))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make TWI more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_twi' not in st.session_state:
                st.session_state.burn_alpha_twi = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_twi, 0.1, key="twi_alpha", on_change=lambda x: st.session_state.update(burn_alpha_twi=x))
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            twi = st.session_state.processed_data['twi']
            burned_mask = st.session_state.processed_data['burned_mask']
            fig, ax = plt.subplots()
            plot_with_burned_overlay(
                ax, twi, 'RdYlBu',
                burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
            )
            st.markdown("**Topographic Wetness Index Map**: Shows wetness potential, with higher values (yellow-red) indicating wetter areas. Adjust slope and flow parameters for variability.")
            st.pyplot(fig)

    # Curvature tab
    with tabs[9]:
        st.header("Curvature Analysis")
        st.markdown("### This tab analyzes terrain curvature from the DEM.")
        st.markdown("**Instructions**: Curvature indicates terrain convexity/concavity, affecting flow. If uniform, check the STL for terrain variability or adjust resolution.")
        
        with st.expander("Visualization Options", expanded=True):
            st.markdown("**Show Burned Areas Overlay**: Toggle to overlay burned areas (red) on the curvature map. Useful for identifying curvature impacts in burned regions.")
            if 'show_burned_curv' not in st.session_state:
                st.session_state.show_burned_curv = False
            show_burned = st.checkbox("Show Burned Areas Overlay", value=st.session_state.show_burned_curv, key="curv_burned", on_change=lambda: st.session_state.update(show_burned_curv=not st.session_state.show_burned_curv))
            
            st.markdown("**Burned Areas Transparency**: Adjust transparency (0.0–1.0) of the burned overlay. Lower values make curvature more visible; higher values emphasize burned areas. Changes update dynamically.")
            if 'burn_alpha_curv' not in st.session_state:
                st.session_state.burn_alpha_curv = 0.5
            burn_alpha = st.slider("Burned Areas Transparency", 0.0, 1.0, st.session_state.burn_alpha_curv, 0.1, key="curv_alpha", on_change=lambda x: st.session_state.update(burn_alpha_curv=x))
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            curvature = st.session_state.processed_data['curvature']
            burned_mask = st.session_state.processed_data['burned_mask']
            fig, ax = plt.subplots()
            plot_with_burned_overlay(
                ax, curvature, 'Spectral',
                burned_mask=burned_mask, show_burned=show_burned, alpha=burn_alpha
            )
            st.markdown("**Curvature Map**: Shows terrain curvature (positive=convex, negative=concave). Adjust DEM parameters to enhance variability.")
            st.pyplot(fig)

    # Scenario GIFs tab
    with tabs[10]:
        st.header("Scenario GIFs")
        st.markdown("### This tab will generate animated GIFs for scenario analysis (not implemented yet).")
        st.markdown("**Instructions**: Upload files and adjust GIF settings. Future updates will enable dynamic visualization of hydrological changes over time.")

    # Burned-Area Hydro Impacts tab
    with tabs[11]:
        st.header("Burned-Area Hydro Impacts")
        st.markdown("### This tab analyzes how burned areas affect hydrology, combining DEM slope and burned areas.")
        st.markdown("**Instructions**: Upload a burned-area TIFF and adjust parameters to see impacts on infiltration, erosion, runoff, and erosion risk. If maps are blank or uniform, check the TIFF for burned areas (adjust 'Burned Area Threshold'), ensure the STL has varied slopes, and tweak parameters like 'Runoff Increase Factor' or 'Erosion Multiplier' to enhance variability.")
        
        st.markdown("""
        **How Burned Areas Affect Hydrogeology**  
        - **Reduced Infiltration** in burned patches → More surface runoff  
        - **Accelerated Erosion** (less vegetative cover) → Higher sediment loads  
        - **Decreased Groundwater Recharge** (if infiltration is lower)  
        - **Nutrient & Ash Loading** in runoff → Potential water quality issues  
        """)
        
        st.subheader("Advanced Burned-Area Parameters")
        st.markdown("**Base Infiltration Rate (mm/hr)**: Sets baseline infiltration before burn effects (0–50 mm/hr). Higher values reduce runoff; adjust to match soil conditions.")
        base_infiltration = st.number_input(
            "Base Infiltration Rate (mm/hr)", value=10.0, step=1.0, min_value=0.0
        )
        
        st.markdown("**Infiltration Reduction in Burned Areas (fraction)**: Reduces infiltration in burned areas (0.0–1.0). Higher values increase runoff; adjust to reflect burn severity.")
        infiltration_reduction = st.slider(
            "Infiltration Reduction in Burned Areas (fraction)",
            0.0, 1.0, 0.5, 0.05
        )
        
        st.markdown("**Base Erosion Rate (tons/ha)**: Sets baseline erosion rate before burn effects (0.1–2.0 tons/ha). Higher values increase sediment loss; adjust for soil type.")
        base_erosion_rate = st.number_input(
            "Base Erosion Rate (tons/ha)", value=0.5, step=0.1
        )
        
        st.markdown("**Erosion Multiplier in Burned Areas**: Increases erosion in burned areas (1.0–5.0). Higher values reflect greater soil loss post-fire; adjust to match burn severity.")
        erosion_multiplier_burned = st.slider(
            "Erosion Multiplier in Burned Areas",
            1.0, 5.0, 2.0, 0.1
        )

        if 'processed_data' in st.session_state and st.session_state.processed_data is not None and st.session_state.processed_data['burned_mask'] is not None:
            # Retrieve parameters for display logic
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

            grid_z = st.session_state.processed_data['grid_z']
            slope = st.session_state.processed_data['slope']
            burned_mask = st.session_state.processed_data['burned_mask']

            # Ensure shape compatibility
            if burned_mask.shape != grid_z.shape:
                st.error(f"Burned mask shape {burned_mask.shape} does not match DEM shape {grid_z.shape}. Adjusting mask size.")
                zoom_factors = (grid_z.shape[0] / burned_mask.shape[0], grid_z.shape[1] / burned_mask.shape[1])
                burned_mask = zoom(burned_mask, zoom_factors, order=0)  # Nearest neighbor
                st.write(f"Adjusted burned mask shape: {burned_mask.shape}, mean: {np.mean(burned_mask)}")

            # Infiltration and Erosion Maps
            infiltration_map = np.full_like(grid_z, base_infiltration)
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()

            erosion_map = np.full_like(grid_z, base_erosion_rate)
            erosion_map[burned_mask == 1] *= erosion_multiplier_burned
            area_per_cell_m2 = area_m2 / (grid_res_val * grid_res_val)
            total_erosion_unburned = np.sum(erosion_map[burned_mask == 0]) * (area_per_cell_m2 / 10000)
            total_erosion_burned = np.sum(erosion_map[burned_mask == 1]) * (area_per_cell_m2 / 10000)
            total_erosion = total_erosion_unburned + total_erosion_burned

            # Combined Effect Maps
            # Runoff Potential Map: Higher slope and burned areas increase runoff
            slope_normalized = slope / np.max(slope, initial=0.1)  # Avoid division by zero, ensure variability
            runoff_potential = runoff_val * (1 + slope_normalized)  # Base runoff scaled by normalized slope
            runoff_potential[burned_mask == 1] *= (1 + burn_factor_val)  # Increase in burned areas
            runoff_potential = np.clip(runoff_potential, 0, 1)  # Ensure within [0, 1]
            st.write(f"Runoff potential max: {np.max(runoff_potential)}, min: {np.min(runoff_potential)}")

            # Erosion Risk Map: Higher slope and burned areas increase erosion risk
            erosion_risk = base_erosion_rate * (1 + slope_normalized)  # Base erosion scaled by normalized slope
            erosion_risk[burned_mask == 1] *= erosion_multiplier_burned  # Increase in burned areas
            st.write(f"Erosion risk max: {np.max(erosion_risk)}, min: {np.min(erosion_risk)}")

            # Display statistics
            st.write(f"**Runoff from Unburned Areas:** {V_runoff_unburned:.2f} m³")
            st.write(f"**Runoff from Burned Areas:** {V_runoff_burned:.2f} m³")
            st.write(f"**Total Runoff:** {V_runoff:.2f} m³")
            st.write(f"**Erosion from Unburned Areas:** {total_erosion_unburned:.2f} tons")
            st.write(f"**Erosion from Burned Areas:** {total_erosion_burned:.2f} tons")
            st.write(f"**Total Erosion:** {total_erosion:.2f} tons (adjusted for cell area)")

            # Infiltration Map
            st.subheader("Infiltration Map (mm/hr)")
            st.markdown("**Infiltration Map**: Shows infiltration rates (mm/hr) across the terrain, with lower values in burned areas (green). Adjust 'Base Infiltration Rate' and 'Infiltration Reduction' to see changes in runoff potential.")
            fig, ax = plt.subplots()
            im = ax.imshow(
                np.flipud(infiltration_map), cmap='Greens', origin='upper',  # Flip for correct orientation
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Infiltration Rate (mm/hr)")
            st.pyplot(fig)

            # Erosion Map
            st.subheader("Erosion Map (tons/ha)")
            st.markdown("**Erosion Map**: Shows erosion rates (tons/ha), with higher values in burned and steeper areas (red-orange). Adjust 'Base Erosion Rate' and 'Erosion Multiplier' to increase variability.")
            fig, ax = plt.subplots()
            im = ax.imshow(
                np.flipud(erosion_map), cmap='OrRd', origin='upper',  # Flip for correct orientation
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Erosion Rate (tons/ha)")
            st.pyplot(fig)

            # Runoff Potential Map
            st.subheader("Runoff Potential Map (Normalized)")
            st.markdown("**Runoff Potential Map**: Shows normalized runoff potential (0–1, blue), combining slope and burned areas. Higher values (darker blue) indicate greater runoff likelihood in steeper, burned regions. If blank or uniform, adjust 'Runoff Coefficient,' 'Runoff Increase Factor,' 'Burned Area Threshold,' or check the STL for slope variability and TIFF for burned areas.")
            fig, ax = plt.subplots()
            im = ax.imshow(
                np.flipud(runoff_potential), cmap='Blues', origin='upper',  # Flip for correct orientation
                extent=(left_bound, right_bound, bottom_bound, top_bound),
                vmin=0, vmax=1
            )
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Runoff Potential (0-1)")
            st.pyplot(fig)

            # Erosion Risk Map
            st.subheader("Erosion Risk Map (tons/ha)")
            st.markdown("**Erosion Risk Map**: Shows erosion risk (tons/ha, yellow-orange-red), combining slope and burned areas. Higher values (redder) indicate greater risk in steeper, burned regions. If uniform, adjust 'Base Erosion Rate,' 'Erosion Multiplier,' 'Burned Area Threshold,' or verify STL and TIFF data.")
            fig, ax = plt.subplots()
            im = ax.imshow(
                np.flipud(erosion_risk), cmap='YlOrRd', origin='upper',  # Flip for correct orientation
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Erosion Risk (tons/ha)")
            st.pyplot(fig)

            st.info("""
            **Additional Tips for Users**:  
            - If maps are blank or uniform, check the debug outputs below for variability in slope, burned mask, and combined effects.  
            - Adjust the 'Burned Area Threshold' (lower values detect more burned areas) or switch bands (Red, Green, Blue) if burned areas aren’t detected.  
            - Increase 'Runoff Increase Factor' or 'Erosion Multiplier' to amplify effects in burned areas.  
            - Verify the STL file has varied terrain (slope) and the TIFF shows burned areas for spatial variation.  
            """)
        else:
            st.write(f"**Total Runoff:** {V_runoff:.2f} m³")
            st.warning("No burned area detected or TIFF missing. Upload a valid burned-area TIFF and adjust the 'Burned Area Threshold' to detect burned regions.")

    # Parameter Comparison Tab
    with tabs[12]:
        st.header("Parameter Comparison")
        st.markdown("### This tab compares hydrological parameters between burned and unburned areas.")
        st.markdown("**Instructions**: If no comparison appears, ensure a burned-area TIFF is uploaded and shows variation. Adjust parameters in other tabs to enhance differences.")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None and st.session_state.processed_data['burned_mask'] is not None:
            # Retrieve parameters for display logic
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

            grid_z = st.session_state.processed_data['grid_z']
            slope = st.session_state.processed_data['slope']
            aspect = st.session_state.processed_data['aspect']
            flow_acc = st.session_state.processed_data['flow_acc']
            twi = st.session_state.processed_data['twi']
            curvature = st.session_state.processed_data['curvature']
            burned_mask = st.session_state.processed_data['burned_mask']

            params = {
                "Elevation (m)": grid_z,
                "Slope (degrees)": slope,
                "Aspect (degrees)": aspect,
                "Flow Accumulation": flow_acc,
                "TWI": twi,
                "Curvature": curvature
            }
            comparison_data = {}
            for param_name, param_data in params.items():
                burned_data = param_data[burned_mask == 1]
                unburned_data = param_data[burned_mask == 0]
                if len(burned_data) > 0 and len(unburned_data) > 0:
                    comparison_data[param_name] = {
                        "Burned Mean": np.mean(burned_data),
                        "Unburned Mean": np.mean(unburned_data),
                        "Burned Median": np.median(burned_data),
                        "Unburned Median": np.median(unburned_data),
                        "Burned Std": np.std(burned_data),
                        "Unburned Std": np.std(unburned_data)
                    }
            if comparison_data:
                df = pd.DataFrame(comparison_data).T
                st.write("**Statistical Comparison of Parameters**")
                st.markdown("**Parameter Comparison Table**: Shows means, medians, and standard deviations for burned vs. unburned areas. Higher differences indicate stronger fire impacts; adjust parameters to enhance variability.")
                st.write(df)
            else:
                st.write("No data available for comparison. Ensure burned areas are detected in the TIFF.")
        else:
            st.write("No burned area data available for comparison. Upload a valid burned-area TIFF.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")

# Helper function for plotting with burned area overlay
def plot_with_burned_overlay(ax, data, cmap, vmin=None, vmax=None, 
                            burned_mask=None, show_burned=True, alpha=0.5):
    # Flip data vertically to correct orientation (north up)
    data = np.flipud(data)
    if burned_mask is not None and show_burned:
        burned_mask = np.flipud(burned_mask)
    
    im = ax.imshow(data, cmap=cmap, origin='upper',  # Changed to 'upper' for correct orientation
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
