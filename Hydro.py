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

    # If data is processed, display dynamically
    if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
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
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None and st.session_state.processed_data['burned_mask'] is not None:
            burned_mask = st.session_state.processed_data['burned_mask']
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
            
