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

# Set page configuration
st.set_page_config(
    page_title="Hydrogeology & DEM Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define tabs (example tab list; adjust as needed)
tab_names = [
    "DEM & Flow Simulation", "Slope & Aspect", "Flow Accumulation",
    "Retention Time", "Nutrient Leaching", "Burned Area Detection",
    "Terrain Derivatives", "Hydro Simulation", "STL Visualization",
    "GeoTIFF Export", "Animation", "Burned-Area Hydro Impacts"
]
tabs = st.tabs(tab_names)

# Georeference bounding box (EPSG:4326) - Example coordinates
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# DEM & Flow Simulation Tab (unchanged example)
with tabs[0]:
    st.header("DEM & Flow Simulation")
    uploaded_stl = st.file_uploader("Upload STL File", type=["stl"])
    run_button = st.button("Run Analysis")
    scale_val = st.slider("Scale Factor", 0.1, 10.0, 1.0)
    offset_val = st.slider("Offset (m)", -100.0, 100.0, 0.0)
    dem_min_val = st.slider("Min DEM Height (m)", -1000.0, 0.0, 0.0)
    dem_max_val = st.slider("Max DEM Height (m)", 0.0, 5000.0, 1000.0)
    grid_res_val = st.slider("Grid Resolution", 50, 500, 100)
    rainfall_val = st.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
    duration_val = st.slider("Storm Duration (hr)", 0.1, 24.0, 1.0)
    area_val = st.slider("Drainage Area (ha)", 1.0, 1000.0, 100.0)
    runoff_val = st.slider("Runoff Coefficient", 0.0, 1.0, 0.5)
    recession_val = st.slider("Recession Constant", 0.0, 1.0, 0.1)
    sim_hours_val = st.slider("Simulation Hours", 1, 48, 24)
    storage_val = st.slider("Storage Capacity (mm)", 0.0, 100.0, 10.0)
    burn_factor_val = st.slider("Burn Factor", 1.0, 3.0, 1.5)
    burn_threshold_val = st.slider("Burn Threshold", 0.0, 1.0, 0.3)
    nutrient_val = st.slider("Nutrient Load (kg/ha)", 0.0, 10.0, 1.0)
    retention_val = st.slider("Retention Factor", 0.0, 1.0, 0.5)
    erosion_val = st.slider("Erosion Rate (tons/ha)", 0.0, 5.0, 0.5)
    st.session_state.update({
        'scale': scale_val, 'offset': offset_val, 'dem_min': dem_min_val,
        'dem_max': dem_max_val, 'grid_res': grid_res_val, 'rainfall': rainfall_val,
        'duration': duration_val, 'area': area_val, 'runoff': runoff_val,
        'recession': recession_val, 'sim_hours': sim_hours_val, 'storage': storage_val,
        'burn_factor': burn_factor_val, 'burn_threshold': burn_threshold_val,
        'nutrient': nutrient_val, 'retention': retention_val, 'erosion': erosion_val
    })

# Processing Logic
if uploaded_stl and run_button:
    # Retrieve parameters from session state
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

    # Load STL and compute DEM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_mesh = mesh.Mesh.from_file(tmp_stl.name)

    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_adj = (z_raw * scale_val) + offset_val

    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = bottom_bound + (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)
    xi = np.linspace(left_bound, right_bound, grid_res_val)
    yi = np.linspace(bottom_bound, top_bound, grid_res_val)  # Should be bottom_bound
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

    # Example burned mask (replace with actual TIFF loading if available)
    burned_mask = np.random.rand(grid_res_val, grid_res_val) > (1 - burn_threshold_val)
    burned_mask = burned_mask.astype(int)

    # *** Burned-Area Hydro Impacts Tab (Enhanced) ***
    with tabs[11]:
        st.header("Burned-Area Hydro Impacts")

        st.markdown("""
        **How Burned Areas Affect Hydrogeology**  
        - **Reduced Infiltration**: Increases surface runoff in burned patches.  
        - **Accelerated Erosion**: Less vegetative cover leads to higher sediment loads.  
        - **Decreased Groundwater Recharge**: Lower infiltration reduces recharge potential.  
        - **Nutrient & Ash Loading**: Enhanced runoff carries more nutrients/ash, impacting water quality.  
        """)

        # User inputs for burned-area parameters
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

        if burned_mask is not None:
            # Existing infiltration map
            infiltration_map = np.full_like(grid_z, base_infiltration)
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()

            # Adjusted runoff coefficient (existing)
            infiltration_ratio = (infiltration_map.mean() / base_infiltration)
            new_runoff_coefficient = runoff_val + burn_factor_val * (1.0 - infiltration_ratio)
            new_runoff_coefficient = np.clip(new_runoff_coefficient, 0.0, 1.0)

            # Nutrient load increase (existing)
            burned_fraction = burned_mask.mean()
            nutrient_load = nutrient_val * area_val  # Example calculation
            nutrient_load_burned = nutrient_load * (1.0 + burned_fraction * 0.3)

            # NEW: Runoff Coefficient Map
            st.subheader("Runoff Coefficient Map")
            runoff_coeff_map = np.minimum(
                runoff_val * (1 + (burn_factor_val - 1) * burned_mask), 1.0
            )
            fig, ax = plt.subplots()
            im = ax.imshow(
                runoff_coeff_map, cmap='Blues', origin='lower',
                extent=(left_bound, right_bound, bottom_bound, top_bound),
                vmin=0, vmax=1
            )
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * \
                           (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Runoff Coefficient")
            st.pyplot(fig)

            # NEW: Erosion Risk Map
            st.subheader("Erosion Risk Map")
            max_slope = np.max(slope)
            erosion_risk_map = base_erosion_rate * (slope / max_slope) * \
                              (1 + (erosion_multiplier_burned - 1) * burned_mask)
            fig, ax = plt.subplots()
            im = ax.imshow(
                erosion_risk_map, cmap='OrRd', origin='lower',
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Erosion Risk (tons/ha)")
            st.pyplot(fig)

            # Interpretation text for the new maps
            st.write("""
            The **Runoff Coefficient Map** shows the spatial variation in runoff potential. 
            Higher values (darker blue) indicate areas where a larger fraction of rainfall becomes surface runoff, 
            particularly in burned regions due to reduced infiltration.

            The **Erosion Risk Map** highlights areas susceptible to soil erosion. 
            Darker red areas have higher risk, influenced by both steep slopes and burned land cover.
            """)

        else:
            st.warning("No burned area detected or TIFF missing. Upload a valid burned-area TIFF.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
