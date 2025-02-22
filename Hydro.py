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

# [Previous imports and configurations remain unchanged]

# Set page config
st.set_page_config(
    page_title="Hydrogeology & DEM Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [CSS, header, ribbon toolbar, and tab definitions remain unchanged]

# Georeference bounding box (EPSG:4326)
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# [Parameter inputs in "DEM & Flow Simulation" and other tabs remain unchanged]

# Processing Logic
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

    # [STL loading, DEM interpolation, derivatives, flow simulation, etc., remain unchanged]

    # Load STL and compute DEM (unchanged)
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
    yi = np.linspace(bottom_bound, top_bound, grid_res_val)  # Note: Should be bottom_bound, assuming typo
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min_val, dem_max_val)

    # Derivatives (unchanged)
    dx = (right_bound - left_bound) / (grid_res_val - 1)
    dy = (top_bound - bottom_bound) / (grid_res_val - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters, dy_meters = dx * meters_per_deg_lon, dy * meters_per_deg_lat
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # [Flow simulation, retention time, nutrient leaching, burned area mask, terrain derivatives, and plotting helper remain unchanged]

    # Burned-Area Hydro Impacts Tab (Enhanced)
    with tabs[11]:
        st.header("Burned-Area Hydro Impacts")

        st.markdown("""
        **How Burned Areas Affect Hydrogeology**  
        - **Reduced Infiltration**: Increases surface runoff in burned patches.  
        - **Accelerated Erosion**: Less vegetative cover leads to higher sediment loads.  
        - **Decreased Groundwater Recharge**: Lower infiltration reduces recharge potential.  
        - **Nutrient & Ash Loading**: Enhanced runoff carries more nutrients/ash, impacting water quality.  
        """)

        # User inputs
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
            # Infiltration map (existing)
            infiltration_map = np.full_like(grid_z, base_infiltration)
            infiltration_map -= infiltration_map * infiltration_reduction * burned_mask
            infiltration_volume_total = (infiltration_map * rainfall_val * duration_val).sum()

            # Existing erosion map (to be replaced)
            erosion_map = np.full_like(grid_z, base_erosion_rate)
            erosion_map[burned_mask == 1] *= erosion_multiplier_burned
            total_erosion_old = erosion_map.sum()  # Incorrect units, will correct below

            # Adjusted runoff coefficient (existing)
            infiltration_ratio = (infiltration_map.mean() / base_infiltration)
            new_runoff_coefficient = runoff_val + burn_factor_val * (1.0 - infiltration_ratio)
            new_runoff_coefficient = np.clip(new_runoff_coefficient, 0.0, 1.0)

            # Nutrient load increase (existing)
            burned_fraction = burned_mask.mean()
            nutrient_load_burned = nutrient_load * (1.0 + burned_fraction * 0.3)

            # NEW: Runoff Depth Map
            # Compute spatially varying runoff coefficient
            C_map = np.minimum(
                runoff_val * (1 + (burn_factor_val - 1) * burned_mask), 1.0
            )
            total_rainfall_depth = rainfall_val * duration_val
            runoff_depth_map = C_map * total_rainfall_depth

            # Calculate total runoff volume
            cell_area_m2 = dx_meters * dy_meters
            total_runoff_volume = np.sum(runoff_depth_map / 1000 * cell_area_m2)  # mm to m

            # NEW: Erosion Risk Map
            max_slope = np.max(slope)
            erosion_risk_map = base_erosion_rate * (slope / max_slope) * \
                              (1 + (erosion_multiplier_burned - 1) * burned_mask)
            cell_area_ha = cell_area_m2 / 10000
            total_erosion = np.sum(erosion_risk_map * cell_area_ha)

            # Display summary statistics
            st.write(f"**Infiltration Volume (mm * cell_area):** ~{infiltration_volume_total:.2f} mm-hr equivalent")
            st.write(f"**Total Runoff Volume:** {total_runoff_volume:.2f} m³")
            st.write(f"**Total Estimated Erosion:** {total_erosion:.2f} tons")
            st.write(f"**Adjusted Runoff Coefficient (approx):** {new_runoff_coefficient:.2f}")
            st.write(f"**Potential Nutrient Load Increase:** from {nutrient_load:.2f} to ~{nutrient_load_burned:.2f} kg")

            # Visualize Infiltration Map (existing)
            st.subheader("Infiltration Map (mm/hr)")
            fig, ax = plt.subplots()
            im = ax.imshow(
                infiltration_map, cmap='Greens', origin='lower',
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            aspect_ratio = (right_bound - left_bound) / (top_bound - bottom_bound) * \
                           (meters_per_deg_lat / meters_per_deg_lon)
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Infiltration Rate (mm/hr)")
            st.pyplot(fig)

            # Visualize Runoff Depth Map
            st.subheader("Runoff Depth Map (mm)")
            fig, ax = plt.subplots()
            im = ax.imshow(
                runoff_depth_map, cmap='Blues', origin='lower',
                extent=(left_bound, right_bound, bottom_bound, top_bound)
            )
            ax.set_aspect(aspect_ratio)
            ax.set_xlabel('Longitude (°E)')
            ax.set_ylabel('Latitude (°N)')
            fig.colorbar(im, ax=ax, label="Runoff Depth (mm)")
            st.pyplot(fig)

            # Visualize Erosion Risk Map
            st.subheader("Erosion Risk Map (tons/ha)")
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

            # Interpretation
            st.info("""
            **Interpretation**:  
            - **Infiltration Map**: Lower rates in burned areas (red overlay) indicate reduced water absorption, leading to more runoff.  
            - **Runoff Depth Map**: Higher values in burned regions show increased surface runoff due to lower infiltration.  
            - **Erosion Risk Map**: Combines slope and burned area effects, highlighting areas prone to soil loss, especially where vegetation is absent.  
            These maps illustrate how burned areas amplify runoff and erosion, impacting hydrology and downstream water quality.
            """)

        else:
            st.warning("No burned area detected or TIFF missing. Upload a valid burned-area TIFF.")

else:
    st.info("Please upload an STL file and click 'Run Analysis' to begin.")
