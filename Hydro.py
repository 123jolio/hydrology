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

# =============================================================================
# 1. Page Configuration & Custom CSS
# =============================================================================
st.set_page_config(page_title="HydroGeo Pro: Advanced DEM & Burn Analysis", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body {
    font-family: 'Roboto', sans-serif;
    background-color: #f0f2f6;
    color: #333;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #1e1e2f;
    color: #fff;
}
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .css-1d391kg p {
    color: #fff;
}
h1, h2, h3 {
    color: #1e1e2f;
}
.stButton>button {
    background-color: #1e88e5;
    color: #fff;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}
.stDownloadButton>button {
    background-color: #43a047;
    color: #fff;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. Header and Logo
# =============================================================================
try:
    st.image("logo.png", width=150)
except Exception:
    pass

st.title("HydroGeo Pro: Advanced DEM & Burned-Area Analysis")
st.markdown("""
Developed for advanced hydrogeological analyses, HydroGeo Pro creates a Digital Elevation Model (DEM) from an STL file, computes key terrain derivatives (slope, aspect, TWI, curvature), and overlays burned‐area effects.  
A composite vulnerability index integrates these factors to highlight regions at risk. All outputs are exportable as GeoTIFFs.
""")

# =============================================================================
# 3. Geographic & File Upload Parameters
# =============================================================================
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

st.sidebar.header("Input Files")
uploaded_stl = st.sidebar.file_uploader("Upload STL (for DEM)", type=["stl"])
uploaded_burned = st.sidebar.file_uploader("Upload Burned-Area TIFF", type=["tif", "tiff"])

# =============================================================================
# 4. Advanced Processing Parameters (Sidebar)
# =============================================================================
st.sidebar.header("DEM & Flow Settings")
global_scale = st.sidebar.slider("Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
global_offset = st.sidebar.slider("Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0)
global_dem_min = st.sidebar.number_input("DEM Min Elevation (m)", value=0.0, step=1.0)
global_dem_max = st.sidebar.number_input("DEM Max Elevation (m)", value=500.0, step=1.0)
global_grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, 50)

st.sidebar.header("Hydrology")
rainfall_intensity = st.sidebar.number_input("Rainfall (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Simulation Duration (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

st.sidebar.header("Nutrient & Erosion")
soil_nutrient = st.sidebar.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Erosion Factor", 0.0, 1.0, 0.3, 0.05)

st.sidebar.header("Burned-Area Effects")
burn_runoff_factor = st.sidebar.slider("Burned Runoff Factor", 0.0, 2.0, 1.0, 0.1)
burn_interp_method = st.sidebar.selectbox("Burned Mask Resampling", ["nearest", "bilinear"])

st.sidebar.header("Vulnerability Index Weights")
weight_slope = st.sidebar.slider("Slope Weight", 0.0, 1.0, 0.4, 0.05)
weight_twi = st.sidebar.slider("TWI Weight", 0.0, 1.0, 0.4, 0.05)
weight_burned = st.sidebar.slider("Burned Weight", 0.0, 1.0, 0.2, 0.05)

# =============================================================================
# 5. Modular Functions
# =============================================================================
def compute_flow_accumulation(dem):
    """Compute flow accumulation using steepest descent."""
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

def export_geotiff(array, transform, crs="EPSG:4326"):
    """Export a numpy array as a GeoTIFF."""
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

def create_placeholder_gif(data_array, frames=10, fps=2, scenario_name="flow"):
    """Create a placeholder GIF animation."""
    images = []
    for i in range(frames):
        factor = 1 + 0.1 * i
        array_i = np.clip(data_array * factor, 0, 1e9)
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(array_i, origin='lower', cmap='hot')
        ax.set_title(f"{scenario_name.capitalize()} Frame {i+1}", fontsize=10)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))
    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, images, format='GIF', fps=fps)
    gif_bytes.seek(0)
    return gif_bytes.getvalue()

def compute_vulnerability(slope, twi, burned_mask, weight_slope, weight_twi, weight_burned):
    """Compute a composite vulnerability index."""
    norm_slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
    norm_twi = (twi - twi.min()) / (twi.max() - twi.min() + 1e-9)
    norm_burned = burned_mask if burned_mask is not None else np.zeros_like(slope)
    vulnerability = (weight_slope * norm_slope + weight_twi * norm_twi + weight_burned * norm_burned)
    return vulnerability

# =============================================================================
# 6. DEM, Hydrology, & Burned-Area Processing
# =============================================================================
if uploaded_stl is not None:
    # --- Process STL to create DEM ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_filename = tmp_stl.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
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

    # --- Compute grid spacing in meters ---
    dx = (right_bound - left_bound) / (global_grid_res - 1)
    dy = (top_bound - bottom_bound) / (global_grid_res - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters = dx * meters_per_deg_lon
    dy_meters = dy * meters_per_deg_lat

    # --- Compute terrain derivatives ---
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360
    transform = from_origin(left_bound, top_bound, dx, dy)

    # --- Flow Simulation ---
    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coeff
    Q_peak = V_runoff / event_duration
    t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
    Q = np.array([Q_peak * (time/event_duration) if time<= event_duration 
                  else Q_peak * np.exp(-recession_rate*(time-event_duration)) 
                  for time in t])
    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # --- Process Burned-Area TIFF ---
    burned_mask = None
    if uploaded_burned is not None:
        ext = os.path.splitext(uploaded_burned.name)[1].lower()
        if ext in [".tif", ".tiff"]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_tif:
                    tmp_tif.write(uploaded_burned.read())
                    tif_filename = tmp_tif.name
                with rasterio.open(tif_filename) as src:
                    src_crs = src.crs if src.crs is not None else "EPSG:4326"
                    burned_img_raw = src.read()
                    src_transform = src.transform
                    if burned_img_raw.shape[0] >= 3:
                        red = burned_img_raw[0]
                        green = burned_img_raw[1]
                        blue = burned_img_raw[2]
                        burned_mask_raw = ((red > 150) & (green < 100) & (blue < 100)).astype(np.float32)
                    else:
                        band = burned_img_raw[0]
                        norm_band = (band - band.min()) / (band.max() - band.min() + 1e-9)
                        burned_mask_raw = (norm_band > 0.5).astype(np.float32)
                    burned_mask_resampled = np.empty(grid_z.shape, dtype=np.float32)
                    interp_method = Resampling.nearest if burn_interp_method=="nearest" else Resampling.bilinear
                    reproject(
                        source=burned_mask_raw,
                        destination=burned_mask_resampled,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=interp_method
                    )
                    burned_mask = burned_mask_resampled
            except Exception as e:
                st.warning(f"Error processing burned-area TIFF: {e}")
                burned_mask = None
        else:
            st.warning("Uploaded file is not a valid TIFF.")
            burned_mask = None

    # --- Additional Terrain Derivatives ---
    flow_acc = compute_flow_accumulation(grid_z)
    adjusted_flow_acc = flow_acc * (1 + burn_runoff_factor * burned_mask) if burned_mask is not None else flow_acc
    slope_radians = np.radians(slope)
    cell_area = dx_meters * dy_meters
    A_eff = adjusted_flow_acc * cell_area
    epsilon = 0.05
    twi = np.log((A_eff + 1) / (np.tan(slope_radians) + epsilon))
    laplacian_kernel = np.array([[1,  1, 1],
                                 [1, -8, 1],
                                 [1,  1, 1]]) / (dx_meters * dy_meters)
    curvature = convolve(grid_z, laplacian_kernel, mode='reflect')
    vulnerability = compute_vulnerability(slope, twi, burned_mask, weight_slope, weight_twi, weight_burned)

    # --- Placeholder GIFs for demonstration ---
    flow_placeholder = np.clip(grid_z / (np.max(grid_z) + 1e-9), 0, 1)
    retention_placeholder = np.clip(slope / (np.max(slope) + 1e-9), 0, 1)
    nutrient_placeholder = np.clip(aspect / 360.0, 0, 1)

    # =============================================================================
    # 7. Display Results: Organized Tabs with Enhanced Visuals
    # =============================================================================
    tabs = st.tabs([
        "DEM & Flow", "Slope", "Aspect", "Retention",
        "GeoTIFF Export", "Nutrient", "Flow Accumulation",
        "TWI", "Curvature", "Vulnerability", "Scenario GIFs"
    ])

    # --- Tab 0: DEM & Flow with Burned-Area Overlay ---
    with tabs[0]:
        st.subheader("DEM with Flow & Burned-Area Overlay")
        with st.expander("DEM Settings"):
            dem_vmin = st.number_input("DEM Minimum (m)", value=global_dem_min, step=1.0)
            dem_vmax = st.number_input("DEM Maximum (m)", value=global_dem_max, step=1.0)
        fig, ax = plt.subplots(figsize=(8,5))
        # Plot DEM
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax)
        # If burned mask exists, normalize it and overlay
        if burned_mask is not None:
            burned_norm = (burned_mask - burned_mask.min()) / (burned_mask.max() - burned_mask.min() + 1e-9)
            ax.imshow(burned_norm, extent=(left_bound, right_bound, bottom_bound, top_bound),
                      origin='lower', cmap='Reds', alpha=0.4)
            # Optional contour outlining burned areas
            ax.contour(burned_norm, levels=[0.5], colors='darkred',
                       extent=(left_bound, right_bound, bottom_bound, top_bound))
        # Plot flow vectors
        step = max(1, global_grid_res // 20)
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step],
                  -dz_dx[::step, ::step], -dz_dy[::step, ::step],
                  color='blue', scale=1e5, width=0.003)
        ax.set_title("DEM with Flow Vectors and Burned-Area Overlay", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    # --- Tab 1: Slope Map ---
    with tabs[1]:
        st.subheader("Slope Map")
        with st.expander("Slope Settings"):
            slope_vmin = st.number_input("Slope Minimum (°)", value=0.0, step=1.0)
            slope_vmax = st.number_input("Slope Maximum (°)", value=70.0, step=1.0)
            slope_cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma"])
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=slope_cmap, vmin=slope_vmin, vmax=slope_vmax)
        ax.set_title("Slope (°)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    # --- Tab 2: Aspect Map ---
    with tabs[2]:
        st.subheader("Aspect Map")
        with st.expander("Aspect Settings"):
            aspect_vmin = st.number_input("Aspect Minimum (°)", value=0.0, step=1.0)
            aspect_vmax = st.number_input("Aspect Maximum (°)", value=360.0, step=1.0)
            aspect_cmap = st.selectbox("Colormap", ["twilight", "hsv", "cool"])
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=aspect_cmap, vmin=aspect_vmin, vmax=aspect_vmax)
        ax.set_title("Aspect (°)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    # --- Tab 3: Retention Time ---
    with tabs[3]:
        st.subheader("Retention Time")
        st.info("Retention time is based on effective runoff and storage volume.")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: **{retention_time:.2f} hr**")
        else:
            st.warning("No effective runoff calculated; retention time not applicable.")

    # --- Tab 4: GeoTIFF Export ---
    with tabs[4]:
        st.subheader("Export GeoTIFFs")
        dem_tiff = export_geotiff(grid_z, transform)
        slope_tiff = export_geotiff(slope, transform)
        aspect_tiff = export_geotiff(aspect, transform)
        st.download_button("Download DEM", dem_tiff, "DEM.tif", "image/tiff")
        st.download_button("Download Slope", slope_tiff, "Slope.tif", "image/tiff")
        st.download_button("Download Aspect", aspect_tiff, "Aspect.tif", "image/tiff")

    # --- Tab 5: Nutrient Leaching ---
    with tabs[5]:
        st.subheader("Nutrient Leaching")
        st.write(f"**Soil Nutrient:** {soil_nutrient} kg/ha")
        st.write(f"**Vegetation Retention:** {veg_retention}")
        st.write(f"**Erosion Factor:** {erosion_factor}")
        st.write(f"**Catchment Area:** {catchment_area} ha")
        st.write(f"**Estimated Nutrient Load:** {nutrient_load:.2f} kg")

    # --- Tab 6: Flow Accumulation ---
    with tabs[6]:
        st.subheader("Flow Accumulation (Adjusted)")
        with st.expander("Flow Accumulation Settings"):
            flowacc_vmin = st.number_input("Flow Accumulation Min", value=float(np.min(adjusted_flow_acc)), step=1.0)
            flowacc_vmax = st.number_input("Flow Accumulation Max", value=float(np.max(adjusted_flow_acc)), step=1.0)
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(adjusted_flow_acc, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='viridis', vmin=flowacc_vmin, vmax=flowacc_vmax)
        ax.set_title("Flow Accumulation", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Accumulated Flow")
        st.pyplot(fig)

    # --- Tab 7: Topographic Wetness Index (TWI) ---
    with tabs[7]:
        st.subheader("Topographic Wetness Index (TWI)")
        with st.expander("TWI Settings"):
            twi_vmin = st.number_input("TWI Min", value=float(np.min(twi)), step=0.1)
            twi_vmax = st.number_input("TWI Max", value=float(np.max(twi)), step=0.1)
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(twi, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='coolwarm', vmin=twi_vmin, vmax=twi_vmax)
        ax.set_title("TWI (Adjusted for Burned Areas)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

    # --- Tab 8: Curvature Analysis ---
    with tabs[8]:
        st.subheader("Curvature Analysis")
        with st.expander("Curvature Settings"):
            curv_vmin = st.number_input("Curvature Min", value=float(np.min(curvature)), step=0.1)
            curv_vmax = st.number_input("Curvature Max", value=float(np.max(curvature)), step=0.1)
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(curvature, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='Spectral', vmin=curv_vmin, vmax=curv_vmax)
        ax.set_title("Curvature (Laplacian Convolution)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Curvature")
        st.pyplot(fig)

    # --- Tab 9: Composite Vulnerability Index ---
    with tabs[9]:
        st.subheader("Composite Vulnerability Index")
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(vulnerability, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='inferno')
        threshold = st.number_input("Vulnerability Threshold", value=float(np.percentile(vulnerability, 80)), step=0.01)
        ax.contour(vulnerability, levels=[threshold], colors='white',
                   extent=(left_bound, right_bound, bottom_bound, top_bound))
        ax.set_title("Vulnerability (Weighted Slope, TWI & Burned-Area)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Vulnerability")
        st.pyplot(fig)

    # --- Tab 10: Scenario GIF Animations ---
    with tabs[10]:
        with st.expander("GIF Settings"):
            gif_frames = st.number_input("GIF Frames", value=10, step=1)
            gif_fps = st.number_input("GIF FPS", value=2, step=1)
        st.subheader("Scenario-Based GIF Animations")
        st.markdown("**Flow Scenario**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")
        st.markdown("**Retention Scenario**")
        st.image(create_placeholder_gif(retention_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="retention"), caption="Retention Scenario (Demo)")
        st.markdown("**Nutrient Scenario**")
        st.image(create_placeholder_gif(nutrient_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="nutrient"), caption="Nutrient Scenario (Demo)")
else:
    st.info("Please upload an STL file via the sidebar to begin analysis.")
