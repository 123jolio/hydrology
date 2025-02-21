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
import imageio  # for GIF creation
import matplotlib.contour as contour  # for boundary extraction

# -----------------------------------------------------------------------------
# 1. MUST be the very first Streamlit command!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 2. Inject minimal, professional CSS (dark sidebar, no gradient)
# -----------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
/* Overall background and font */
html, body {
    background-color: #f5f5f5;
    font-family: "Roboto", sans-serif;
    color: #333;
}

/* Solid dark sidebar (no gradient) */
[data-testid="stSidebar"] > div:first-child {
    background: #2a2a2a;
    color: white;
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .css-1d391kg p {
    color: white;
}

/* Remove borders on tabs */
div.stTabs > div {
    border: none;
}

/* Main Title styling */
h1 {
    text-align: center;
    font-size: 3rem;
    color: #2e7bcf;
}

/* Override Streamlit's button styling */
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border-radius: 0.5rem;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    border: none;
}

/* Download button customization */
.css-1emrehy.edgvbvh3 { 
    background-color: #2e7bcf !important; 
    color: white !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Header (logo + title)
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    pass

st.title("Advanced Hydrogeology & DEM Analysis (with Scenario GIFs)")
st.markdown("""
This application creates a DEM from an STL file, computes advanced hydrogeological maps (slope, aspect), 
simulates a runoff hydrograph, estimates retention time, nutrient leaching, and (optionally) burned-area risk.  
Each analysis has its own dedicated parameter controls.  
All outputs can be exported as georeferenced GeoTIFF files.
""")

# -----------------------------------------------------------------------------
# 4. Georeference bounding box (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

# -----------------------------------------------------------------------------
# 5. File upload: STL and optional burned-area GeoTIFF
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Upload STL file (for DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Optional: Upload burned-area GeoTIFF", type=["tif","tiff"])

# -----------------------------------------------------------------------------
# 6. Global Sidebar Parameters (for DEM and Flow/Retention/Nutrients)
# -----------------------------------------------------------------------------
st.sidebar.header("Global DEM & Elevation")
global_scale = st.sidebar.slider("Global Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
global_offset = st.sidebar.slider("Global Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0)
global_dem_min = st.sidebar.number_input("Global Min Elevation (m)", value=0.0, step=1.0)
global_dem_max = st.sidebar.number_input("Global Max Elevation (m)", value=500.0, step=1.0)
global_grid_res = st.sidebar.number_input("Global Grid Resolution", 100, 1000, 500, 50)

st.sidebar.header("Global Flow & Retention")
rainfall_intensity = st.sidebar.number_input("Rainfall (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Simulation Duration (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

st.sidebar.header("Global Nutrient Leaching")
soil_nutrient = st.sidebar.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05)

# -----------------------------------------------------------------------------
# Slope Map and Burned Risk parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Slope Map Customization")
slope_display_min = st.sidebar.number_input("Slope Display Minimum (°)", value=0.0, step=1.0)
slope_display_max = st.sidebar.number_input("Slope Display Maximum (°)", value=70.0, step=1.0)

st.sidebar.header("Burned Risk Parameters")
risk_slope_weight = st.sidebar.slider("Slope Weight", 0.0, 2.0, 1.0, 0.1)
risk_dem_weight = st.sidebar.slider("DEM Weight", 0.0, 2.0, 1.0, 0.1)
risk_burned_weight = st.sidebar.slider("Burned Area Weight", 0.0, 2.0, 1.0, 0.1)
risk_rain_weight = st.sidebar.slider("Rain Weight", 0.0, 2.0, 1.0, 0.1)

# -----------------------------------------------------------------------------
# 7. Process STL and compute DEM and related maps
# -----------------------------------------------------------------------------
risk_map = None       # Will hold risk map if burned area is provided
burned_mask = None    # For later use in burned-area analysis

if uploaded_stl is not None:
    # Save STL to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_filename = tmp_stl.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL: {e}")
        st.stop()

    # Extract vertices from STL
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]

    # Apply global elevation adjustments
    z_adj = (z_raw * global_scale) + global_offset

    # Map raw x,y to geographic coordinates using bounding box
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    # Create DEM grid
    xi = np.linspace(left_bound, right_bound, global_grid_res)
    yi = np.linspace(top_bound, bottom_bound, global_grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate to create DEM
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, global_dem_min, global_dem_max)

    # Compute grid spacing in degrees
    dx = (right_bound - left_bound) / (global_grid_res - 1)
    dy = (top_bound - bottom_bound) / (global_grid_res - 1)

    # Approximate conversion from degrees to meters
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320  # roughly constant

    dx_meters = dx * meters_per_deg_lon
    dy_meters = dy * meters_per_deg_lat

    # Compute slope and aspect
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # Affine transform for GeoTIFF export (still in EPSG:4326 coords)
    transform = from_origin(left_bound, top_bound, dx, dy)

    # Flow Simulation
    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coeff
    Q_peak = V_runoff / event_duration

    t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            # Rising limb
            Q[i] = Q_peak * (time / event_duration)
        else:
            # Recession
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    # Retention Time Calculation
    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None

    # Nutrient Leaching Calculation
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # -----------------------------------------------------------------------------
    # 8. Burned-Area Processing (if burned GeoTIFF is provided)
    # -----------------------------------------------------------------------------
    if uploaded_burned is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_burn:
            tmp_burn.write(uploaded_burned.read())
            burned_filename = tmp_burn.name
        try:
            with rasterio.open(burned_filename) as src:
                # Read the burned raster data
                burned_img = src.read()  # shape: (bands, height, width)
                src_transform = src.transform
                src_crs = src.crs
                st.write("**Debug Info: Burned Raster**")
                st.write(f"CRS: {src_crs}")
                st.write(f"Bounds: {src.bounds}")
                st.write(f"Shape: {burned_img.shape}")
                # Print min/max for each band
                for band_i in range(burned_img.shape[0]):
                    band_min = burned_img[band_i].min()
                    band_max = burned_img[band_i].max()
                    st.write(f"Band {band_i}: min={band_min}, max={band_max}")
        except Exception as e:
            st.warning(f"Error reading burned GeoTIFF: {e}")
            burned_img = None

        if burned_img is not None:
            # We handle single-band or multi-band
            if burned_img.shape[0] == 1:
                # Example: single-band classification (0 = not burned, 1 = burned)
                # Adjust threshold logic to match your data
                single_band = burned_img[0]
                # For example, treat values > 0.5 as burned
                burned_mask = (single_band > 0.5).astype(np.uint8)

            elif burned_img.shape[0] >= 3:
                # Example: RGB-based threshold
                # Adjust these thresholds to match your actual "burned color"
                red_band   = burned_img[0]
                green_band = burned_img[1]
                blue_band  = burned_img[2]

                burned_mask = np.logical_and.reduce((
                    (red_band >= 100) & (red_band <= 180),
                    (green_band >= 200) & (green_band <= 255),
                    (blue_band >= 100) & (blue_band <= 180)
                )).astype(np.uint8)
            else:
                st.warning("Unrecognized burned raster format (less than 1 band?).")
                burned_mask = None

            # If we have a valid burned_mask, reproject it
            if burned_mask is not None:
                burned_mask_resampled = np.empty(grid_z.shape, dtype=np.uint8)
                try:
                    reproject(
                        source=burned_mask,
                        destination=burned_mask_resampled,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest
                    )
                    burned_mask = burned_mask_resampled
                except Exception as e:
                    st.warning(f"Error reprojecting burned mask: {e}")
                    burned_mask = None

    # -----------------------------------------------------------------------------
    # 9. Risk Map Calculation
    # -----------------------------------------------------------------------------
    if burned_mask is not None:
        # Normalize slope and DEM
        norm_slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
        norm_dem   = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min() + 1e-9)

        # Rain factor (constant for entire domain, but you could also vary it spatially)
        norm_rain = (rainfall_intensity * event_duration) / 100.0  # arbitrary scale

        # Weighted sum for risk
        risk_map = (risk_slope_weight * norm_slope +
                    risk_dem_weight   * norm_dem +
                    risk_burned_weight * burned_mask +
                    risk_rain_weight  * norm_rain)

        # Normalize risk to [0, 1]
        risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-9)

    # -----------------------------------------------------------------------------
    # Helper: Placeholder GIF creation function
    # -----------------------------------------------------------------------------
    def create_placeholder_gif(data_array, frames=10, fps=2, scenario_name="flow"):
        images = []
        for i in range(frames):
            factor = 1 + 0.1 * i
            array_i = np.clip(data_array * factor, 0, 1e9)
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(array_i, origin='lower', cmap='hot')
            ax.set_title(f"{scenario_name.capitalize()} Frame {i+1}")
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            plt.close(fig)
            buf.seek(0)
            images.append(imageio.imread(buf))
        gif_bytes = io.BytesIO()
        imageio.mimsave(gif_bytes, images, format='GIF', fps=fps)
        gif_bytes.seek(0)
        return gif_bytes.getvalue()

    # Create placeholder GIF data
    flow_placeholder = np.clip(grid_z / (np.max(grid_z) + 1e-9), 0, 1)
    retention_placeholder = np.clip(slope / (np.max(slope) + 1e-9), 0, 1)
    nutrient_placeholder = np.clip(aspect / 360.0, 0, 1)
    risk_gif = None
    if risk_map is not None:
        risk_placeholder = np.clip(risk_map, 0, 1)
        risk_gif = create_placeholder_gif(risk_placeholder, scenario_name="risk")

    # -----------------------------------------------------------------------------
    # 10. Display results in dedicated tabs with scenario-specific parameter expanders
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "GeoTIFF Export",
        "Nutrient Leaching", "Burned Area Analysis", "Burned Risk", "Scenario GIFs"
    ])

    # ---- DEM Heatmap Tab ----
    with tab1:
        with st.expander("DEM Heatmap Parameters", expanded=False):
            dem_vmin = st.number_input("DEM Color Scale Minimum (m)", value=global_dem_min, step=1.0, key="dem_min_tab")
            dem_vmax = st.number_input("DEM Color Scale Maximum (m)", value=global_dem_max, step=1.0, key="dem_max_tab")
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax, aspect='auto')
        ax.set_title("DEM (Adjusted Elevation)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    # ---- Slope Map Tab ----
    with tab2:
        with st.expander("Slope Map Parameters", expanded=False):
            slope_cmap = st.selectbox("Select Slope Colormap", ["viridis", "plasma", "inferno", "magma"], key="slope_cmap")
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=slope_cmap,
                       vmin=slope_display_min, vmax=slope_display_max, aspect='auto')
        ax.set_title("Slope (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    # ---- Aspect Map Tab ----
    with tab3:
        with st.expander("Aspect Map Parameters", expanded=False):
            aspect_cmap = st.selectbox("Select Aspect Colormap", ["twilight", "hsv", "cool"], key="aspect_cmap")
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=aspect_cmap, aspect='auto')
        ax.set_title("Aspect (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    # ---- Burned Area Analysis Tab ----
    with tab8:
        st.subheader("Burned Area GeoTIFF & Analysis")
        if burned_mask is not None:
            # Calculate percent burned
            percent_burned = 100.0 * np.sum(burned_mask) / burned_mask.size
            st.write(f"Percent Burned Area: {percent_burned:.2f}%")
            # Display the burned mask
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(burned_mask, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='gray', aspect='auto')
            ax.set_title("Burned Area Mask")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Burned (1) / Not Burned (0)")
            # Compute boundary via contour
            cs = ax.contour(burned_mask, levels=[0.5], colors='red',
                            extent=(left_bound, right_bound, bottom_bound, top_bound))
            st.pyplot(fig)
            st.info("Red contour shows the boundary of the burned area.")
        else:
            st.info("No burned-area data available for analysis or burned_mask is None.")

    # ---- Burned Risk Tab ----
    with tab9:
        st.subheader("Enhanced Burned Risk Map")
        if risk_map is not None:
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(risk_map, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='inferno', aspect='auto')
            ax.set_title("Burned Risk Map (Weighted Factors)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Risk Score (0-1)")
            st.pyplot(fig)
        else:
            st.info("Burned risk map unavailable (no burned-area data or not calculated).")

    # ---- GeoTIFF Export Tab ----
    with tab6:
        st.subheader("Export GeoTIFFs")
        def export_geotiff(array, transform, crs="EPSG:4326"):
            memfile = io.BytesIO()
            with rasterio.io.MemoryFile() as memfile_obj:
                with memfile_obj.open(
                    driver="GTiff",
                    height=array.shape[0],
                    width=array.shape[1],
                    count=1,
                    dtype="float32",
                    crs=crs,
                    transform=transform,
                ) as dataset:
                    dataset.write(array.astype("float32"), 1)
                memfile_obj.seek(0)
                memfile.write(memfile_obj.read())
            return memfile.getvalue()

        # DEM, Slope, Aspect
        dem_tiff = export_geotiff(grid_z, transform)
        slope_tiff = export_geotiff(slope, transform)
        aspect_tiff = export_geotiff(aspect, transform)

        st.download_button("Download DEM (GeoTIFF)", dem_tiff, "DEM.tif", "image/tiff")
        st.download_button("Download Slope (GeoTIFF)", slope_tiff, "Slope.tif", "image/tiff")
        st.download_button("Download Aspect (GeoTIFF)", aspect_tiff, "Aspect.tif", "image/tiff")

        # Burned Mask
        if burned_mask is not None:
            burned_tiff = export_geotiff(burned_mask.astype("float32"), transform)
            st.download_button("Download Burned Area Mask (GeoTIFF)", burned_tiff, "BurnedArea.tif", "image/tiff")

        # Risk Map
        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Download Risk (GeoTIFF)", risk_tiff, "RiskMap.tif", "image/tiff")

    # ---- Flow Simulation Tab ----
    with tab4:
        with st.expander("Flow Simulation Parameters", expanded=False):
            flow_fps = st.number_input("Flow GIF FPS", value=2, step=1, key="flow_fps")
            flow_frames = st.number_input("Flow GIF Frames", value=10, step=1, key="flow_frames")
        st.subheader("Flow Simulation Hydrograph")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(t, Q, 'b-')
        ax.set_title("Runoff Hydrograph")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow (m³/hr)")
        st.pyplot(fig)
        st.write(f"Peak Flow: {Q_peak:.2f} m³/hr")
        st.write(f"Total Runoff Volume: {V_runoff:.2f} m³")
        st.markdown("**Flow Scenario GIF:**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(flow_frames), fps=int(flow_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")

    # ---- Retention Time Tab ----
    with tab5:
        with st.expander("Retention Time Parameters", expanded=False):
            st.write("Retention time is computed from effective runoff and storage volume.")
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.warning("No effective runoff -> Retention time not applicable.")

    # ---- Nutrient Leaching Tab ----
    with tab7:
        with st.expander("Nutrient Leaching Parameters", expanded=False):
            nutrient_scale = st.number_input("Nutrient Scale Factor", value=1.0, step=0.1, key="nutrient_scale")
        st.subheader("Nutrient Leaching")
        st.write(f"Soil Nutrient Content: {soil_nutrient} kg/ha")
        st.write(f"Vegetation Retention: {veg_retention}")
        st.write(f"Soil Erosion Factor: {erosion_factor}")
        st.write(f"Catchment Area: {catchment_area} ha")
        st.write(f"Estimated Nutrient Load: {nutrient_load * nutrient_scale:.2f} kg")

    # ---- Scenario GIFs Tab ----
    with tab10:
        with st.expander("Scenario GIF Parameters", expanded=False):
            gif_frames = st.number_input("GIF Frames", value=10, step=1, key="gif_frames")
            gif_fps = st.number_input("GIF FPS", value=2, step=1, key="gif_fps")
        st.subheader("Scenario-Based GIF Animations (Placeholders)")
        st.markdown("**Flow Scenario**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")
        st.markdown("**Retention Scenario**")
        st.image(create_placeholder_gif(retention_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="retention"), caption="Retention Scenario (Demo)")
        st.markdown("**Nutrient Scenario**")
        st.image(create_placeholder_gif(nutrient_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="nutrient"), caption="Nutrient Scenario (Demo)")
        if risk_gif is not None:
            st.markdown("**Risk Scenario**")
            st.image(create_placeholder_gif(risk_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="risk"), caption="Risk Scenario (Demo)")
else:
    st.info("Please upload an STL file to generate DEM and scenario analyses.")
