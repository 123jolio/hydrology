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
from rasterio.features import rasterize
import imageio  # for GIF creation
import zipfile
from fastkml import kml
import os
from PIL import Image

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
html, body {
    background-color: #f5f5f5;
    font-family: "Roboto", sans-serif;
    color: #333;
}
[data-testid="stSidebar"] > div:first-child {
    background: #2a2a2a;
    color: white;
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .css-1d391kg p {
    color: white;
}
div.stTabs > div {
    border: none;
}
h1 {
    text-align: center;
    font-size: 3rem;
    color: #2e7bcf;
}
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border-radius: 0.5rem;
    font-size: 1rem;
    padding: 0.5rem 1rem;
    border: none;
}
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
This application creates a DEM from an STL file and computes advanced hydrogeological maps (slope, aspect).  
It simulates flow over the terrain, estimates retention time, nutrient leaching, and assesses burned-area risk from a TIFF file.  
Additional terrain derivatives (flow accumulation, topographic wetness index, and curvature) are also computed.  
New charts now incorporate burned-area data by adjusting flow accumulation and TWI based on a user-defined runoff/infiltration multiplier.  
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
# 5. File upload: STL and burned-area data (TIFF only for burned area)
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Upload STL file (for DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Upload Burned-Area Data (TIFF only)", type=["tif", "tiff"])

# -----------------------------------------------------------------------------
# 6. Global Sidebar Parameters (for DEM, Flow/Retention/Nutrients, and Burned Adjustments)
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

st.sidebar.header("Burned Area Hydrological Adjustments")
# Slider to adjust how much burned areas affect runoff/infiltration (multiplier: 1.0 means no adjustment)
burned_runoff_multiplier = st.sidebar.slider("Burned Runoff Multiplier", 1.0, 2.0, 1.0, 0.1)

st.sidebar.header("Slope Map & Burned Risk Customization")
slope_display_min = st.sidebar.number_input("Slope Display Minimum (°)", value=0.0, step=1.0)
slope_display_max = st.sidebar.number_input("Slope Display Maximum (°)", value=70.0, step=1.0)
risk_slope_weight = st.sidebar.slider("Slope Weight", 0.0, 2.0, 1.0, 0.1)
risk_dem_weight = st.sidebar.slider("DEM Weight", 0.0, 2.0, 1.0, 0.1)
risk_burned_weight = st.sidebar.slider("Burned Area Weight", 0.0, 2.0, 1.0, 0.1)
risk_rain_weight = st.sidebar.slider("Rain Weight", 0.0, 2.0, 1.0, 0.1)

# -----------------------------------------------------------------------------
# 7. Process STL and compute DEM and related maps
# -----------------------------------------------------------------------------
risk_map = None       # Will hold risk map if burned area is provided
burned_mask = None    # For burned-area analysis
burned_img_raw = None # To store raw burned TIFF values if applicable

if uploaded_stl is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_filename = tmp_stl.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL: {e}")
        st.stop()

    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]

    z_adj = (z_raw * global_scale) + global_offset

    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    xi = np.linspace(left_bound, right_bound, global_grid_res)
    yi = np.linspace(top_bound, bottom_bound, global_grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, global_dem_min, global_dem_max)

    dx = (right_bound - left_bound) / (global_grid_res - 1)
    dy = (top_bound - bottom_bound) / (global_grid_res - 1)

    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320

    dx_meters = dx * meters_per_deg_lon
    dy_meters = dy * meters_per_deg_lat

    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    transform = from_origin(left_bound, top_bound, dx, dy)

    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coeff
    Q_peak = V_runoff / event_duration

    t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak * (time / event_duration)
        else:
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # -----------------------------------------------------------------------------
    # 8. Burned-Area Processing (TIFF only)
    # -----------------------------------------------------------------------------
    if uploaded_burned is not None:
        ext = os.path.splitext(uploaded_burned.name)[1].lower()
        if ext in [".tif", ".tiff"]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_tif:
                    tmp_tif.write(uploaded_burned.read())
                    tif_filename = tmp_tif.name
                with rasterio.open(tif_filename) as src:
                    src_crs = src.crs if src.crs is not None else "EPSG:4326"
                    burned_img_raw = src.read(1)
                    st.write("Raw burned TIFF values: min", burned_img_raw.min(), "max", burned_img_raw.max())
                    src_transform = src.transform
                    # Normalize the raw values to [0, 1]
                    burned_img_norm = (burned_img_raw - burned_img_raw.min()) / (burned_img_raw.max() - burned_img_raw.min() + 1e-9)
                    burned_mask_temp = burned_img_norm
                    burned_mask_resampled = np.empty(grid_z.shape, dtype=np.float32)
                    reproject(
                        source=burned_mask_temp,
                        destination=burned_mask_resampled,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest
                    )
                    burned_mask = burned_mask_resampled
            except Exception as e:
                st.warning(f"Error reading burned TIFF: {e}")
                burned_mask = None
        else:
            st.warning("Please upload a TIFF file for burned-area analysis.")
            burned_mask = None

    # -----------------------------------------------------------------------------
    # 9. Risk Map Calculation
    # -----------------------------------------------------------------------------
    if burned_mask is not None:
        norm_slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
        norm_dem   = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min() + 1e-9)
        norm_rain = (rainfall_intensity * event_duration) / 100.0
        risk_map = (risk_slope_weight * norm_slope +
                    risk_dem_weight   * norm_dem +
                    risk_burned_weight * burned_mask +
                    risk_rain_weight  * norm_rain)
        risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-9)

    # -----------------------------------------------------------------------------
    # 10. Additional Terrain Derivatives: Flow Accumulation, TWI, and Curvature
    # -----------------------------------------------------------------------------
    def compute_flow_accumulation(dem):
        acc = np.ones_like(dem)
        rows, cols = dem.shape
        indices = np.argsort(-dem.flatten())
        for idx in indices:
            r, c = np.unravel_index(idx, dem.shape)
            best_drop = 0
            best_neighbor = None
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        drop = dem[r, c] - dem[rr, cc]
                        if drop > best_drop:
                            best_drop = drop
                            best_neighbor = (rr, cc)
            if best_neighbor is not None:
                rr, cc = best_neighbor
                acc[rr, cc] += acc[r, c]
        return acc

    flow_acc = compute_flow_accumulation(grid_z)
    slope_radians = np.radians(slope)
    cell_area = dx_meters * dy_meters
    twi = np.log((flow_acc * cell_area) / (np.tan(slope_radians) + 1e-9))
    d2z_dx2 = np.gradient(dz_dx, dx_meters, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dy_meters, axis=0)
    curvature = d2z_dx2 + d2z_dy2

    # -----------------------------------------------------------------------------
    # 11. Helper: Placeholder GIF creation function
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

    flow_placeholder = np.clip(grid_z / (np.max(grid_z) + 1e-9), 0, 1)
    retention_placeholder = np.clip(slope / (np.max(slope) + 1e-9), 0, 1)
    nutrient_placeholder = np.clip(aspect / 360.0, 0, 1)
    risk_gif = None
    if risk_map is not None:
        risk_placeholder = np.clip(risk_map, 0, 1)
        risk_gif = create_placeholder_gif(risk_placeholder, scenario_name="risk")

    # -----------------------------------------------------------------------------
    # 12. Display results in dedicated tabs with scenario-specific parameter expanders
    # -----------------------------------------------------------------------------
    # Updated tabs list now includes new burned-adjusted charts.
    tabs = st.tabs([
        "DEM & Flow Simulation", "Slope Map", "Aspect Map",
        "Retention Time", "GeoTIFF Export",
        "Nutrient Leaching", "Burned Area Analysis", "Burned Risk",
        "Flow Accumulation", "Topographic Wetness Index", "Curvature Analysis",
        "Burned-Adjusted Flow Accumulation", "Burned-Adjusted TWI",
        "Scenario GIFs", "Raw Burned TIFF"
    ])

    # Tab 0: DEM heatmap with flow simulation overlay
    with tabs[0]:
        with st.expander("DEM & Flow Overlay Parameters", expanded=False):
            dem_vmin = st.number_input("DEM Color Scale Minimum (m)", value=global_dem_min, step=1.0, key="dem_min_tab")
            dem_vmax = st.number_input("DEM Color Scale Maximum (m)", value=global_dem_max, step=1.0, key="dem_max_tab")
        st.subheader("DEM (Adjusted Elevation) with Flow Simulation")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax, aspect='auto')
        # Overlay flow simulation using quiver arrows (downhill direction)
        step = max(1, global_grid_res // 20)
        q = ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step],
                      -dz_dx[::step, ::step], -dz_dy[::step, ::step],
                      color='blue', scale=1e5, width=0.0025)
        ax.set_title("DEM with Flow Simulation Overlay")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    with tabs[1]:
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

    with tabs[2]:
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

    with tabs[3]:
        with st.expander("Retention Time Parameters", expanded=False):
            st.write("Retention time is computed from effective runoff and storage volume.")
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.warning("No effective runoff -> Retention time not applicable.")

    with tabs[4]:
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
        dem_tiff = export_geotiff(grid_z, transform)
        slope_tiff = export_geotiff(slope, transform)
        aspect_tiff = export_geotiff(aspect, transform)
        st.download_button("Download DEM (GeoTIFF)", dem_tiff, "DEM.tif", "image/tiff")
        st.download_button("Download Slope (GeoTIFF)", slope_tiff, "Slope.tif", "image/tiff")
        st.download_button("Download Aspect (GeoTIFF)", aspect_tiff, "Aspect.tif", "image/tiff")
        if burned_mask is not None:
            burned_tiff = export_geotiff(burned_mask.astype("float32"), transform)
            st.download_button("Download Burned Area Mask (GeoTIFF)", burned_tiff, "BurnedArea.tif", "image/tiff")
        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Download Risk (GeoTIFF)", risk_tiff, "RiskMap.tif", "image/tiff")

    with tabs[5]:
        with st.expander("Nutrient Leaching Parameters", expanded=False):
            nutrient_scale = st.number_input("Nutrient Scale Factor", value=1.0, step=0.1, key="nutrient_scale")
        st.subheader("Nutrient Leaching")
        st.write(f"Soil Nutrient Content: {soil_nutrient} kg/ha")
        st.write(f"Vegetation Retention: {veg_retention}")
        st.write(f"Soil Erosion Factor: {erosion_factor}")
        st.write(f"Catchment Area: {catchment_area} ha")
        st.write(f"Estimated Nutrient Load: {nutrient_load * nutrient_scale:.2f} kg")

    with tabs[6]:
        st.subheader("Burned Area Analysis")
        if burned_mask is not None:
            percent_burned = 100.0 * np.sum(burned_mask) / burned_mask.size
            st.write(f"Percent Burned Area: {percent_burned:.2f}%")
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(burned_mask, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='gray', aspect='auto')
            ax.set_title("Burned Area Mask")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Burned (1) / Not Burned (0)")
            st.pyplot(fig)
        else:
            st.info("No burned-area data available (burned_mask is None).")

    with tabs[7]:
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

    with tabs[8]:
        st.subheader("Flow Accumulation Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(flow_acc, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Flow Accumulation (D8)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Accumulated Flow")
        st.pyplot(fig)

    with tabs[9]:
        st.subheader("Topographic Wetness Index (TWI)")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(twi, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='coolwarm', aspect='auto')
        ax.set_title("TWI")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

    with tabs[10]:
        st.subheader("Curvature Analysis")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(curvature, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='Spectral', aspect='auto')
        ax.set_title("Curvature (Laplacian)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Curvature")
        st.pyplot(fig)

    # -----------------------------------------------------------------------------
    # New Tab: Burned-Adjusted Flow Accumulation
    # -----------------------------------------------------------------------------
    with tabs[11]:
        st.subheader("Burned-Adjusted Flow Accumulation")
        # Allow user to set legend boundaries for this map
        flow_adj_lower = st.number_input("Flow Accumulation Legend Lower Bound (log scale)", 
                                         value=float(np.log1p(flow_acc).min()), step=0.1, key="flow_adj_lower")
        flow_adj_upper = st.number_input("Flow Accumulation Legend Upper Bound (log scale)", 
                                         value=float(np.log1p(flow_acc).max()), step=0.1, key="flow_adj_upper")
        # Create a burned adjustment factor:
        # If burned_mask is available, amplify flow accumulation by a factor that increases from 1 (unburned)
        # to burned_runoff_multiplier (in fully burned areas). Otherwise, leave unchanged.
        if burned_mask is not None:
            burned_adjustment = 1 + (burned_runoff_multiplier - 1) * burned_mask
        else:
            burned_adjustment = np.ones_like(flow_acc)
        adjusted_flow_acc = flow_acc * burned_adjustment
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(np.log1p(adjusted_flow_acc), extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='Blues', aspect='auto',
                       vmin=flow_adj_lower, vmax=flow_adj_upper)
        ax.set_title("Burned-Adjusted Flow Accumulation (log scale)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label='Log(Adjusted Flow Accumulation)')
        st.pyplot(fig)

    # -----------------------------------------------------------------------------
    # New Tab: Burned-Adjusted TWI
    # -----------------------------------------------------------------------------
    with tabs[12]:
        st.subheader("Burned-Adjusted Topographic Wetness Index (TWI)")
        twi_adj_lower = st.number_input("TWI Legend Lower Bound", 
                                        value=float(twi.min()), step=0.1, key="twi_adj_lower")
        twi_adj_upper = st.number_input("TWI Legend Upper Bound", 
                                        value=float(twi.max()), step=0.1, key="twi_adj_upper")
        # Use the same burned_adjustment as above
        if burned_mask is not None:
            burned_adjustment = 1 + (burned_runoff_multiplier - 1) * burned_mask
        else:
            burned_adjustment = np.ones_like(flow_acc)
        adjusted_flow_acc = flow_acc * burned_adjustment
        twi_burned_adjusted = np.log((adjusted_flow_acc * cell_area) / (np.tan(slope_radians) + 1e-9))
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(twi_burned_adjusted, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='coolwarm', aspect='auto',
                       vmin=twi_adj_lower, vmax=twi_adj_upper)
        ax.set_title("Burned-Adjusted TWI")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label='TWI')
        st.pyplot(fig)

    with tabs[13]:
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

    with tabs[14]:
        st.subheader("Raw Burned TIFF Data")
        if burned_img_raw is not None:
            # Rescale raw data for visualization
            raw_vis = (burned_img_raw - burned_img_raw.min()) / (burned_img_raw.max() - burned_img_raw.min() + 1e-9)
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(raw_vis, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='upper', cmap='gray', aspect='auto')
            ax.set_title("Raw Burned TIFF")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Normalized Value")
            st.pyplot(fig)
        else:
            st.info("No raw burned TIFF data available.")
else:
    st.info("Please upload an STL file to generate DEM and scenario analyses.")
