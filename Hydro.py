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

# ------------------------------------------------------------------------------
# 1. Page Setup & Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

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

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------

def create_placeholder_gif(data_array: np.ndarray, frames: int = 10, fps: int = 2, scenario_name: str = "flow") -> bytes:
    """
    Generate a placeholder GIF animation from the provided data array.

    Parameters:
        data_array (np.ndarray): Array used for visualization.
        frames (int): Number of frames in the GIF.
        fps (int): Frames per second.
        scenario_name (str): Label for the scenario (used in frame titles).

    Returns:
        bytes: Byte content of the generated GIF.
    """
    images = []
    for i in range(frames):
        factor = 1 + 0.1 * i
        array_frame = np.clip(data_array * factor, 0, 1e9)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(array_frame, origin='lower', cmap='hot')
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

def export_geotiff(array: np.ndarray, transform, crs: str = "EPSG:4326") -> bytes:
    """
    Export a numpy array as a GeoTIFF file in memory.

    Parameters:
        array (np.ndarray): Data array to export.
        transform: Affine transform for georeferencing.
        crs (str): Coordinate Reference System.

    Returns:
        bytes: Byte content of the GeoTIFF.
    """
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

def process_stl_file(uploaded_file, global_scale, global_offset, global_dem_min, global_dem_max, global_grid_res, left_bound, top_bound, right_bound, bottom_bound):
    """
    Process the uploaded STL file to generate a DEM, slope, and aspect maps.

    Parameters:
        uploaded_file: Uploaded STL file.
        global_scale (float): Elevation scale factor.
        global_offset (float): Elevation offset.
        global_dem_min (float): Minimum elevation value.
        global_dem_max (float): Maximum elevation value.
        global_grid_res (int): DEM grid resolution.
        left_bound, top_bound, right_bound, bottom_bound (float): Geographic bounding box.

    Returns:
        tuple: (grid_z, slope, aspect, transform)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_file.read())
        stl_filename = tmp.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract vertices and apply elevation adjustments
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_adj = (z_raw * global_scale) + global_offset

    # Map raw coordinates to geographic coordinates
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    # Create the DEM grid
    xi = np.linspace(left_bound, right_bound, global_grid_res)
    yi = np.linspace(top_bound, bottom_bound, global_grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, global_dem_min, global_dem_max)

    # Compute grid spacing and affine transform for georeferencing
    dx = (right_bound - left_bound) / (global_grid_res - 1)
    dy = (top_bound - bottom_bound) / (global_grid_res - 1)
    transform = from_origin(left_bound, top_bound, dx, dy)

    # Calculate slope and aspect from the DEM
    dz_dx, dz_dy = np.gradient(grid_z, dx, dy)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    return grid_z, slope, aspect, transform

def process_burned_area(uploaded_burned, grid_shape, transform, slope):
    """
    Process the burned-area GeoTIFF to generate a risk map.

    Parameters:
        uploaded_burned: Uploaded burned-area GeoTIFF.
        grid_shape (tuple): Shape of the DEM grid.
        transform: Affine transform for reprojection.
        slope (np.ndarray): Slope array for risk calculations.

    Returns:
        np.ndarray or None: Normalized risk map, or None if processing fails.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_burned.read())
        burned_filename = tmp.name

    try:
        with rasterio.open(burned_filename) as src:
            burned_img = src.read()  # (bands, height, width)
            src_transform = src.transform
            src_crs = src.crs
    except Exception as e:
        st.warning(f"Error reading burned GeoTIFF: {e}")
        return None

    if burned_img is not None and burned_img.shape[0] >= 3:
        burned_mask = np.logical_and.reduce((
            (burned_img[0] >= 100) & (burned_img[0] <= 180),
            (burned_img[1] >= 200) & (burned_img[1] <= 255),
            (burned_img[2] >= 100) & (burned_img[2] <= 180)
        )).astype(np.uint8)
        burned_mask_resampled = np.empty(grid_shape, dtype=np.uint8)
        reproject(
            source=burned_mask,
            destination=burned_mask_resampled,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest
        )
        eps = 0.01
        risk_map = burned_mask_resampled * (1.0 / (slope + eps))
        rmin, rmax = risk_map.min(), risk_map.max()
        if rmax > rmin:
            risk_map = (risk_map - rmin) / (rmax - rmin)
        else:
            risk_map[:] = 0
        return risk_map
    return None

# ------------------------------------------------------------------------------
# 3. Header & File Upload
# ------------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    pass

st.title("Advanced Hydrogeology & DEM Analysis (with Scenario GIFs)")
st.markdown("""
This application creates a Digital Elevation Model (DEM) from an STL file and computes advanced hydrogeological maps, 
including slope and aspect. It simulates a runoff hydrograph, estimates retention time, and assesses nutrient leaching 
and, optionally, burned-area risk. Each analysis has dedicated parameter controls, and outputs can be exported as georeferenced GeoTIFF files.
""")

# Define geographic bounding box (EPSG:4326)
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

# File upload widgets
uploaded_stl = st.file_uploader("Upload STL file (for DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Optional: Upload burned-area GeoTIFF", type=["tif", "tiff"])

# ------------------------------------------------------------------------------
# 4. Sidebar Parameters
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 5. Main Processing & Analysis
# ------------------------------------------------------------------------------
if uploaded_stl is not None:
    # Generate DEM, slope, and aspect maps from the STL
    grid_z, slope, aspect, transform = process_stl_file(
        uploaded_stl, global_scale, global_offset, global_dem_min, global_dem_max,
        global_grid_res, left_bound, top_bound, right_bound, bottom_bound
    )
    
    # Flow simulation calculations
    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coeff
    Q_peak = V_runoff / event_duration if event_duration > 0 else 0.0

    t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak * (time / event_duration)
        else:
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # Process burned-area risk (if available)
    risk_map = None
    if uploaded_burned is not None:
        risk_map = process_burned_area(uploaded_burned, grid_z.shape, transform, slope)

    # Create placeholder GIF data arrays
    flow_placeholder = np.clip(grid_z / (np.max(grid_z) + 1e-9), 0, 1)
    retention_placeholder = np.clip(slope / (np.max(slope) + 1e-9), 0, 1)
    nutrient_placeholder = np.clip(aspect / 360.0, 0, 1)
    risk_placeholder = np.clip(risk_map, 0, 1) if risk_map is not None else None

    # ------------------------------------------------------------------------------
    # 6. Display Analysis Results in Tabs
    # ------------------------------------------------------------------------------
    tabs = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "GeoTIFF Export",
        "Nutrient Leaching", "Burned Risk", "Scenario GIFs"
    ])

    # --- DEM Heatmap ---
    with tabs[0]:
        with st.expander("DEM Heatmap Parameters", expanded=False):
            dem_vmin = st.number_input("DEM Color Scale Minimum (m)", value=global_dem_min, step=1.0, key="dem_min_tab")
            dem_vmax = st.number_input("DEM Color Scale Maximum (m)", value=global_dem_max, step=1.0, key="dem_max_tab")
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax, aspect='auto')
        ax.set_title("DEM (Adjusted Elevation)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    # --- Slope Map ---
    with tabs[1]:
        with st.expander("Slope Map Parameters", expanded=False):
            slope_cmap = st.selectbox("Select Slope Colormap", ["viridis", "plasma", "inferno", "magma"], key="slope_cmap")
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=slope_cmap, aspect='auto')
        ax.set_title("Slope (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    # --- Aspect Map ---
    with tabs[2]:
        with st.expander("Aspect Map Parameters", expanded=False):
            aspect_cmap = st.selectbox("Select Aspect Colormap", ["twilight", "hsv", "cool"], key="aspect_cmap")
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=aspect_cmap, aspect='auto')
        ax.set_title("Aspect (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    # --- Flow Simulation ---
    with tabs[3]:
        with st.expander("Flow Simulation Parameters", expanded=False):
            flow_fps = st.number_input("Flow GIF FPS", value=2, step=1, key="flow_fps")
            flow_frames = st.number_input("Flow GIF Frames", value=10, step=1, key="flow_frames")
        st.subheader("Flow Simulation Hydrograph")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, Q, 'b-')
        ax.set_title("Runoff Hydrograph")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow (m³/hr)")
        st.pyplot(fig)
        st.write(f"Peak Flow: {Q_peak:.2f} m³/hr")
        st.write(f"Total Runoff Volume: {V_runoff:.2f} m³")
        st.markdown("**Flow Scenario GIF:**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(flow_frames), fps=int(flow_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")

    # --- Retention Time ---
    with tabs[4]:
        with st.expander("Retention Time Parameters", expanded=False):
            st.write("Retention time is computed from effective runoff and storage volume.")
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.warning("No effective runoff → Retention time not applicable.")

    # --- GeoTIFF Export ---
    with tabs[5]:
        st.subheader("Export GeoTIFFs")
        dem_tiff = export_geotiff(grid_z, transform)
        slope_tiff = export_geotiff(slope, transform)
        aspect_tiff = export_geotiff(aspect, transform)
        st.download_button("Download DEM (GeoTIFF)", dem_tiff, "DEM.tif", "image/tiff")
        st.download_button("Download Slope (GeoTIFF)", slope_tiff, "Slope.tif", "image/tiff")
        st.download_button("Download Aspect (GeoTIFF)", aspect_tiff, "Aspect.tif", "image/tiff")
        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Download Risk (GeoTIFF)", risk_tiff, "RiskMap.tif", "image/tiff")

    # --- Nutrient Leaching ---
    with tabs[6]:
        with st.expander("Nutrient Leaching Parameters", expanded=False):
            nutrient_scale = st.number_input("Nutrient Scale Factor", value=1.0, step=0.1, key="nutrient_scale")
        st.subheader("Nutrient Leaching")
        st.write(f"Soil Nutrient Content: {soil_nutrient} kg/ha")
        st.write(f"Vegetation Retention: {veg_retention}")
        st.write(f"Soil Erosion Factor: {erosion_factor}")
        st.write(f"Catchment Area: {catchment_area} ha")
        st.write(f"Estimated Nutrient Load: {nutrient_load * nutrient_scale:.2f} kg")

    # --- Burned-Area Risk ---
    with tabs[7]:
        with st.expander("Burned-Area Risk Parameters", expanded=False):
            risk_epsilon = st.number_input("Risk Epsilon", value=0.01, step=0.001, key="risk_epsilon")
            risk_scale = st.number_input("Risk Scale Factor", value=1.0, step=0.1, key="risk_scale")
        st.subheader("Burned-Area Risk")
        if risk_map is not None:
            risk_adjusted = risk_map * risk_scale  # Apply a simple scaling adjustment
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(risk_adjusted, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='inferno', aspect='auto')
            ax.set_title("Risk Map (Burned & Accumulation)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Risk Score (0-1)")
            st.pyplot(fig)
        else:
            st.info("No burned-area data → Risk map unavailable.")

    # --- Scenario GIFs ---
    with tabs[8]:
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
        if risk_placeholder is not None:
            st.markdown("**Risk Scenario**")
            st.image(create_placeholder_gif(risk_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="risk"), caption="Risk Scenario (Demo)")
else:
    st.info("Please upload an STL file to generate DEM and scenario analyses.")
