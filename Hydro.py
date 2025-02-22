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
import os
from PIL import Image
from scipy.ndimage import convolve  # For improved curvature calculation

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
This application creates a DEM from an STL file and computes advanced hydrogeological maps (slope, aspect). It overlays a flow simulation on the DEM, estimates retention time and nutrient leaching, and computes additional terrain derivatives (flow accumulation, topographic wetness index (TWI), and curvature).  
If a burned‐area TIFF file is provided—with red regions indicating burned and white indicating non‐burned areas—its effect on the hydrological response is incorporated (reduced infiltration and increased surface runoff).  
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
# 5. File upload: STL file and burned‐area TIFF file
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Upload STL file (for DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Upload Burned-Area Data (TIFF)", type=["tif", "tiff"])

# -----------------------------------------------------------------------------
# 6. Global Processing Parameters (in the sidebar)
# -----------------------------------------------------------------------------
st.sidebar.header("Processing Parameters")
global_scale = st.sidebar.slider("Global Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
global_offset = st.sidebar.slider("Global Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0)
global_dem_min = st.sidebar.number_input("Global Min Elevation (m)", value=0.0, step=1.0)
global_dem_max = st.sidebar.number_input("Global Max Elevation (m)", value=500.0, step=1.0)
global_grid_res = st.sidebar.number_input("Global Grid Resolution", 100, 1000, 500, 50)

st.sidebar.header("Flow & Retention")
rainfall_intensity = st.sidebar.number_input("Rainfall (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Simulation Duration (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

st.sidebar.header("Nutrient Leaching")
soil_nutrient = st.sidebar.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05)

st.sidebar.header("Burned Area Effects")
burn_runoff_factor = st.sidebar.slider("Burned Runoff Increase Factor", 0.0, 2.0, 1.0, 0.1)
# Option to overlay the burned area TIFF
overlay_burned_option = st.sidebar.checkbox("Overlay Burned Area TIFF", value=True)
burned_transparency = st.sidebar.slider("Burned Area Overlay Transparency", 0.0, 1.0, 0.3, 0.05)

# -----------------------------------------------------------------------------
# Helper function to create a figure and axis using a fixed size based on the spatial extent.
# This ensures all plots have the same proportional dimensions.
def create_fig_ax():
    spatial_width = right_bound - left_bound
    spatial_height = top_bound - bottom_bound
    aspect_ratio = spatial_height / spatial_width
    # Choose a base width (in inches) and compute height accordingly.
    base_width = 8
    fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
    return fig, ax

# -----------------------------------------------------------------------------
# 7. Process STL and compute DEM and related maps
# -----------------------------------------------------------------------------
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
    # 8. Burned-Area Processing (TIFF)
    # -----------------------------------------------------------------------------
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
                    # If the image has multiple bands, assume a color image.
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
                    reproject(
                        source=burned_mask_raw,
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
    # 9. Additional Terrain Derivatives: Flow Accumulation, TWI, and Curvature
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
    if burned_mask is not None:
        adjusted_flow_acc = flow_acc * (1 + burn_runoff_factor * burned_mask)
    else:
        adjusted_flow_acc = flow_acc

    slope_radians = np.radians(slope)
    cell_area = dx_meters * dy_meters

    # Improved TWI Calculation
    A_eff = adjusted_flow_acc * cell_area
    epsilon = 0.05  # small constant to avoid division by zero
    twi = np.log((A_eff + 1) / (np.tan(slope_radians) + epsilon))

    # Improved Curvature Analysis using a Laplacian Convolution
    laplacian_kernel = np.array([[1,  1, 1],
                                 [1, -8, 1],
                                 [1,  1, 1]]) / (dx_meters * dy_meters)
    curvature = convolve(grid_z, laplacian_kernel, mode='reflect')

    # -----------------------------------------------------------------------------
    # 10. Helper: Placeholder GIF creation function
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

    # -----------------------------------------------------------------------------
    # 11. Display results in dedicated tabs with consistent sizing and optional overlay
    # -----------------------------------------------------------------------------
    tabs = st.tabs([
        "DEM & Flow Simulation", "Slope Map", "Aspect Map",
        "Retention Time", "GeoTIFF Export",
        "Nutrient Leaching", "Flow Accumulation", "Topographic Wetness Index",
        "Curvature Analysis", "Scenario GIFs"
    ])

    # Helper to overlay burned TIFF if enabled
    def overlay_burned(ax):
        if overlay_burned_option and (burned_mask is not None):
            ax.imshow(burned_mask, extent=(left_bound, right_bound, bottom_bound, top_bound),
                      origin='lower', cmap="Reds", alpha=burned_transparency)

    # Tab 0: DEM with Flow Simulation
    with tabs[0]:
        with st.expander("DEM Legend Boundaries"):
            dem_vmin = st.number_input("DEM Minimum (m)", value=global_dem_min, step=1.0)
            dem_vmax = st.number_input("DEM Maximum (m)", value=global_dem_max, step=1.0)
        st.subheader("DEM (Adjusted Elevation) with Flow Simulation")
        fig, ax = create_fig_ax()
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax)
        step = max(1, global_grid_res // 20)
        ax.quiver(grid_x[::step, ::step], grid_y[::step, ::step],
                  -dz_dx[::step, ::step], -dz_dy[::step, ::step],
                  color='blue', scale=1e5, width=0.0025)
        overlay_burned(ax)
        ax.set_title("DEM with Flow Overlay")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    # Tab 1: Slope Map
    with tabs[1]:
        with st.expander("Slope Legend Boundaries"):
            slope_vmin = st.number_input("Slope Minimum (°)", value=0.0, step=1.0)
            slope_vmax = st.number_input("Slope Maximum (°)", value=70.0, step=1.0)
        with st.expander("Colormap Options"):
            slope_cmap = st.selectbox("Select Slope Colormap", ["viridis", "plasma", "inferno", "magma"])
        st.subheader("Slope Map (Degrees)")
        fig, ax = create_fig_ax()
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=slope_cmap, vmin=slope_vmin, vmax=slope_vmax)
        overlay_burned(ax)
        ax.set_title("Slope (°)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    # Tab 2: Aspect Map
    with tabs[2]:
        with st.expander("Aspect Legend Boundaries"):
            aspect_vmin = st.number_input("Aspect Minimum (°)", value=0.0, step=1.0)
            aspect_vmax = st.number_input("Aspect Maximum (°)", value=360.0, step=1.0)
        with st.expander("Colormap Options"):
            aspect_cmap = st.selectbox("Select Aspect Colormap", ["twilight", "hsv", "cool"])
        st.subheader("Aspect Map (Degrees)")
        fig, ax = create_fig_ax()
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=aspect_cmap, vmin=aspect_vmin, vmax=aspect_vmax)
        overlay_burned(ax)
        ax.set_title("Aspect (°)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    # Tab 3: Retention Time
    with tabs[3]:
        st.subheader("Retention Time")
        st.info("Retention time is computed from effective runoff and storage volume.")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.warning("No effective runoff → Retention time not applicable.")

    # Tab 4: GeoTIFF Export
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

    # Tab 5: Nutrient Leaching
    with tabs[5]:
        st.subheader("Nutrient Leaching")
        st.write(f"Soil Nutrient Content: {soil_nutrient} kg/ha")
        st.write(f"Vegetation Retention: {veg_retention}")
        st.write(f"Soil Erosion Factor: {erosion_factor}")
        st.write(f"Catchment Area: {catchment_area} ha")
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    # Tab 6: Flow Accumulation Map
    with tabs[6]:
        with st.expander("Flow Accumulation Legend Boundaries"):
            flowacc_vmin = st.number_input("Flow Accumulation Min", value=float(np.min(adjusted_flow_acc)), step=1.0)
            flowacc_vmax = st.number_input("Flow Accumulation Max", value=float(np.max(adjusted_flow_acc)), step=1.0)
        st.subheader("Flow Accumulation Map")
        fig, ax = create_fig_ax()
        im = ax.imshow(adjusted_flow_acc, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='viridis', vmin=flowacc_vmin, vmax=flowacc_vmax)
        overlay_burned(ax)
        ax.set_title("Flow Accumulation (Adjusted for Burned Areas)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Accumulated Flow")
        st.pyplot(fig)

    # Tab 7: Topographic Wetness Index (TWI)
    with tabs[7]:
        with st.expander("TWI Legend Boundaries"):
            twi_vmin = st.number_input("TWI Minimum", value=float(np.min(twi)), step=0.1)
            twi_vmax = st.number_input("TWI Maximum", value=float(np.max(twi)), step=0.1)
        st.subheader("Topographic Wetness Index (TWI)")
        fig, ax = create_fig_ax()
        im = ax.imshow(twi, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='coolwarm', vmin=twi_vmin, vmax=twi_vmax)
        overlay_burned(ax)
        ax.set_title("TWI (Adjusted for Burned Areas)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

    # Tab 8: Curvature Analysis
    with tabs[8]:
        with st.expander("Curvature Legend Boundaries"):
            curv_vmin = st.number_input("Curvature Minimum", value=float(np.min(curvature)), step=0.1)
            curv_vmax = st.number_input("Curvature Maximum", value=float(np.max(curvature)), step=0.1)
        st.subheader("Curvature Analysis")
        fig, ax = create_fig_ax()
        im = ax.imshow(curvature, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='Spectral', vmin=curv_vmin, vmax=curv_vmax)
        overlay_burned(ax)
        ax.set_title("Curvature (Laplacian Convolution)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Curvature")
        st.pyplot(fig)

    # Tab 9: Scenario GIFs
    with tabs[9]:
        with st.expander("GIF Settings"):
            gif_frames = st.number_input("GIF Frames", value=10, step=1)
            gif_fps = st.number_input("GIF FPS", value=2, step=1)
        st.subheader("Scenario-Based GIF Animations (Placeholders)")
        st.markdown("**Flow Scenario**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")
        st.markdown("**Retention Scenario**")
        st.image(create_placeholder_gif(retention_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="retention"), caption="Retention Scenario (Demo)")
        st.markdown("**Nutrient Scenario**")
        st.image(create_placeholder_gif(nutrient_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="nutrient"), caption="Nutrient Scenario (Demo)")
else:
    st.info("Please upload an STL file to generate DEM and scenario analyses.")
