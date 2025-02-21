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
import imageio
import zipfile
from fastkml import kml
import os
from PIL import Image
import matplotlib.animation as animation
from pysheds.grid import Grid

# -----------------------------------------------------------------------------
# 1. Streamlit Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# Minimal professional CSS
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
h1 {
    text-align: center;
    font-size: 2.5rem;
    color: #2e7bcf;
}
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.title("Advanced Hydrogeology & DEM Analysis")

# -----------------------------------------------------------------------------
# 2. Cached Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def process_stl_file(file_obj, scale, offset, grid_res, bounds):
    """
    Generate DEM from STL file with georeferencing.
    Returns: (grid_z, transform)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(file_obj.read())
        stl_mesh = mesh.Mesh.from_file(tmp.name)
    
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_adj = z_raw * scale + offset
    
    left, top, right, bottom = bounds
    lon_raw = left + (x_raw - x_raw.min()) * (right - left) / (x_raw.max() - x_raw.min())
    lat_raw = top - (y_raw - y_raw.min()) * (top - bottom) / (y_raw.max() - y_raw.min())
    
    xi = np.linspace(left, right, grid_res)
    yi = np.linspace(top, bottom, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, 0, 500)  # Adjust clip range as needed
    
    dx = (right - left) / (grid_res - 1)
    dy = (top - bottom) / (grid_res - 1)
    transform = from_origin(left, top, dx, dy)
    return grid_z, transform

@st.cache_data
def compute_terrain_derivatives(grid_z, dx_meters, dy_meters):
    """
    Compute slope, aspect, and curvature from DEM.
    Returns: (slope, aspect, curvature)
    """
    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360
    d2z_dx2 = np.gradient(dz_dx, dx_meters, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dy_meters, axis=0)
    curvature = d2z_dx2 + d2z_dy2
    return slope, aspect, curvature

@st.cache_data
def compute_hydro_derivatives(grid_z, transform, cell_size_m):
    """
    Compute flow accumulation and TWI using pysheds.
    Uses a fallback method if Grid.from_array is not available.
    Returns: (flow_acc, twi)
    """
    grid = Grid()
    # Add the DEM data manually into the grid.
    grid.add_gridded_data(grid_z, data_name='dem', affine=transform, nodata=-9999, crs={'init':'epsg:4326'})
    grid.fill_depressions(data='dem', out_name='filled_dem')
    grid.flowdir(data='filled_dem', out_name='flow_dir')
    flow_acc = grid.accumulation(data='flow_dir')
    slope_rad = np.radians(np.gradient(grid_z, cell_size_m)[0])
    twi = np.log((flow_acc * cell_size_m**2) / (np.tan(np.clip(slope_rad, 0.01, np.inf)) + 1e-9))
    return flow_acc, twi

@st.cache_data
def create_gif(data_array, frames=10, fps=2, title="Scenario"):
    """
    Create animated GIF using matplotlib.animation.
    Returns: bytes of the GIF.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data_array, origin='lower', cmap='hot')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, label="Value")
    
    def update(frame):
        im.set_data(np.clip(data_array * (1 + 0.1 * frame), 0, data_array.max()))
        ax.set_title(f"{title} Frame {frame + 1}")
        return im,
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps)
    buf = io.BytesIO()
    ani.save(buf, format='gif', writer='pillow', fps=fps)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@st.cache_data
def export_geotiff(array, transform, crs="EPSG:4326", metadata=None):
    """
    Export array as a GeoTIFF with optional metadata.
    Returns: bytes of the GeoTIFF.
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
            **(metadata or {})
        ) as dataset:
            dataset.write(array.astype("float32"), 1)
        memfile.write(memfile_obj.read())
    return memfile.getvalue()

# -----------------------------------------------------------------------------
# 3. Main Application
# -----------------------------------------------------------------------------
st.title("Advanced Hydrogeology & DEM Analysis")
st.markdown("""
Analyze terrain, hydrology, and burned-area risks using STL files and optional burned-area data (KMZ, TIFF, JPG/PNG).  
Outputs are cached for performance and exported as GeoTIFFs.
""")

# Define geographic bounds (EPSG:4326)
BOUNDS = (27.906069, 36.92337189, 28.045764, 36.133509)  # (left, top, right, bottom)

# Sidebar: Option to use local files from the same folder as Hydro.py
use_local = st.sidebar.checkbox("Use local files (same folder as Hydro.py)", value=False)

if use_local:
    script_dir = os.path.dirname(__file__)
    # EXACT filenames from your provided photo
    stl_file_path = os.path.join(script_dir, "3d layout.stl")
    burned_file_path = os.path.join(script_dir, "burned areas.tif")
    
    if not os.path.isfile(stl_file_path):
        st.error(f"Local STL file not found: {stl_file_path}")
        st.stop()
    else:
        st.write(f"Using local STL: {stl_file_path}")
    with open(stl_file_path, "rb") as f:
        stl_file_obj = io.BytesIO(f.read())
    
    if not os.path.isfile(burned_file_path):
        st.warning(f"Local burned-area file not found: {burned_file_path}")
        burned_file_obj = None
    else:
        st.write(f"Using local burned-area file: {burned_file_path}")
        with open(burned_file_path, "rb") as f:
            burned_file_obj = io.BytesIO(f.read())
else:
    stl_file_obj = st.file_uploader("Upload STL File (for DEM)", type=["stl"])
    burned_file_obj = st.file_uploader("Upload Burned-Area Data (KMZ, TIFF, JPG, PNG)", type=["kmz", "tif", "tiff", "jpg", "png"])

# Sidebar Parameters
st.sidebar.header("Terrain Parameters")
scale = st.sidebar.slider("Elevation Scale", 0.1, 5.0, 1.0, help="Scale factor for STL elevation.")
offset = st.sidebar.slider("Elevation Offset (m)", -100.0, 100.0, 0.0, help="Vertical adjustment for DEM.")
grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, help="Number of grid cells per axis.")

st.sidebar.header("Hydrology Parameters")
rainfall = st.sidebar.number_input("Rainfall (mm/hr)", 0.0, 100.0, 20.0, help="Rainfall intensity.")
duration = st.sidebar.number_input("Rainfall Duration (hr)", 0.1, 24.0, 2.0, help="Duration of rainfall event.")
area_ha = st.sidebar.number_input("Catchment Area (ha)", 0.1, 1000.0, 10.0, help="Catchment area in hectares.")
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.3, help="Fraction of rainfall becoming runoff.")
recession = st.sidebar.number_input("Recession Rate (1/hr)", 0.1, 2.0, 0.5, help="Rate of flow decrease post-event.")

st.sidebar.header("Risk Parameters")
risk_slope_w = st.sidebar.slider("Slope Weight", 0.0, 2.0, 1.0, help="Weight for slope in risk calculation.")
risk_dem_w = st.sidebar.slider("DEM Weight", 0.0, 2.0, 1.0, help="Weight for elevation in risk calculation.")
risk_burn_w = st.sidebar.slider("Burned Area Weight", 0.0, 2.0, 1.0, help="Weight for burned areas in risk.")

# Input Validation
if stl_file_obj is None:
    st.error("No STL file provided. Please either enable 'Use local files' or upload an STL.")
    st.stop()

if grid_res <= 0 or rainfall < 0 or duration <= 0 or area_ha <= 0:
    st.error("Invalid input: Grid resolution, rainfall, duration, and area must be positive.")
    st.stop()

# -----------------------------------------------------------------------------
# Processing: Generate DEM from STL and compute derivatives
# -----------------------------------------------------------------------------
with st.spinner("Processing STL file..."):
    grid_z, transform = process_stl_file(stl_file_obj, scale, offset, grid_res, BOUNDS)
    dx = (BOUNDS[2] - BOUNDS[0]) / (grid_res - 1)
    dy = (BOUNDS[1] - BOUNDS[3]) / (grid_res - 1)
    avg_lat = (BOUNDS[1] + BOUNDS[3]) / 2.0
    dx_m = dx * 111320 * np.cos(np.radians(avg_lat))
    dy_m = dy * 111320
    cell_size_m = (dx_m + dy_m) / 2
    slope, aspect, curvature = compute_terrain_derivatives(grid_z, dx_m, dy_m)
    flow_acc, twi = compute_hydro_derivatives(grid_z, transform, cell_size_m)

# -----------------------------------------------------------------------------
# Processing: Burned-Area Data (if provided)
# -----------------------------------------------------------------------------
burned_mask = None
if burned_file_obj is not None:
    with st.spinner("Processing burned-area data..."):
        file_ext = os.path.splitext(getattr(burned_file_obj, "name", ""))[1].lower()
        if file_ext == ".kmz":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
                tmp.write(burned_file_obj.read())
                kmz_filename = tmp.name
            try:
                with zipfile.ZipFile(kmz_filename, 'r') as zf:
                    kml_file = [n for n in zf.namelist() if n.endswith('.kml')][0]
                    kml_data = zf.read(kml_file)
                k_obj = kml.KML()
                k_obj.from_string(kml_data)
                polygons = [p.geometry for f in k_obj.features() for p in f.features() if hasattr(p, 'geometry')]
                if polygons:
                    burned_mask = rasterize([(p, 1) for p in polygons],
                                            out_shape=grid_z.shape,
                                            transform=transform,
                                            fill=0, dtype=np.uint8)
            except Exception as e:
                st.warning(f"Error processing KMZ: {e}")
        elif file_ext in [".tif", ".tiff"]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(burned_file_obj.read())
                    tif_filename = tmp.name
                with rasterio.open(tif_filename) as src:
                    src_crs = src.crs if src.crs is not None else "EPSG:4326"
                    burned_data = src.read(1)
                    burned_mask = np.zeros_like(grid_z, dtype=np.float32)
                    reproject(
                        burned_data,
                        burned_mask,
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest
                    )
            except Exception as e:
                st.warning(f"Error reading burned TIFF: {e}")
        elif file_ext in [".jpg", ".jpeg", ".png"]:
            try:
                burned_img = imageio.imread(burned_file_obj)
                burned_mask = ((burned_img[..., 0] > 150) &
                               (burned_img[..., 1] < 100) &
                               (burned_img[..., 2] < 100)).astype(np.uint8)
                burned_mask = np.array(Image.fromarray(burned_mask).resize((grid_z.shape[1], grid_z.shape[0]), resample=Image.NEAREST))
            except Exception as e:
                st.warning(f"Error reading burned image: {e}")
        else:
            st.warning("Unsupported burned-area file format.")

# -----------------------------------------------------------------------------
# Risk Map Calculation
# -----------------------------------------------------------------------------
if burned_mask is not None:
    norm_slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
    norm_dem = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min() + 1e-9)
    risk_map = (risk_slope_w * norm_slope +
                risk_dem_w   * norm_dem +
                risk_burn_w  * burned_mask)
    risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-9)
else:
    risk_map = None

# -----------------------------------------------------------------------------
# Runoff Hydrograph Calculation
# -----------------------------------------------------------------------------
area_m2 = area_ha * 10000
V_runoff = (rainfall / 1000) * duration * area_m2 * runoff_coeff
Q_peak = V_runoff / duration
t = np.linspace(0, 6, int(6 * 60))
Q = np.where(t <= duration, Q_peak * (t / duration), Q_peak * np.exp(-recession * (t - duration)))

# -----------------------------------------------------------------------------
# Display Results in Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(["Terrain Analysis", "Hydrology", "Risk Assessment", "Export"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(grid_z, extent=BOUNDS, origin='lower', cmap='hot')
        ax.set_title("DEM")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(slope, extent=BOUNDS, origin='lower', cmap='viridis')
        ax.set_title("Slope")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(aspect, extent=BOUNDS, origin='lower', cmap='twilight')
        ax.set_title("Aspect")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(curvature, extent=BOUNDS, origin='lower', cmap='Spectral')
        ax.set_title("Curvature")
        fig.colorbar(im, ax=ax, label="Curvature")
        st.pyplot(fig)

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, Q)
        ax.set_title("Runoff Hydrograph")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow (m³/hr)")
        st.pyplot(fig)
        st.write(f"Peak Flow: {Q_peak:.2f} m³/hr")
        st.write(f"Total Runoff Volume: {V_runoff:.2f} m³")
    with col2:
        st.image(create_gif(grid_z / grid_z.max(), title="Flow Animation"), caption="Flow Animation")
    
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(flow_acc, extent=BOUNDS, origin='lower', cmap='viridis')
        ax.set_title("Flow Accumulation")
        fig.colorbar(im, ax=ax, label="Accumulated Flow")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(twi, extent=BOUNDS, origin='lower', cmap='coolwarm')
        ax.set_title("Topographic Wetness Index (TWI)")
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

with tabs[2]:
    if burned_mask is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(burned_mask, extent=BOUNDS, origin='lower', cmap='gray')
            ax.set_title("Burned Area")
            fig.colorbar(im, ax=ax, label="Burned (1)")
            st.pyplot(fig)
        with col2:
            if risk_map is not None:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(risk_map, extent=BOUNDS, origin='lower', cmap='inferno')
                ax.set_title("Risk Map")
                fig.colorbar(im, ax=ax, label="Risk Score")
                st.pyplot(fig)
                st.image(create_gif(risk_map, title="Risk Animation"), caption="Risk Animation")
    else:
        st.info("No burned-area data available to compute risk.")

with tabs[3]:
    st.subheader("Export GeoTIFFs")
    st.download_button("Download DEM GeoTIFF", export_geotiff(grid_z, transform, metadata={"units": "meters"}), "dem.tif")
    st.download_button("Download Slope GeoTIFF", export_geotiff(slope, transform, metadata={"units": "degrees"}), "slope.tif")
    st.download_button("Download Aspect GeoTIFF", export_geotiff(aspect, transform, metadata={"units": "degrees"}), "aspect.tif")
    st.download_button("Download Curvature GeoTIFF", export_geotiff(curvature, transform), "curvature.tif")
    st.download_button("Download Flow Accumulation GeoTIFF", export_geotiff(flow_acc, transform), "flow_acc.tif")
    st.download_button("Download TWI GeoTIFF", export_geotiff(twi, transform), "twi.tif")
    if burned_mask is not None:
        st.download_button("Download Burned Mask GeoTIFF", export_geotiff(burned_mask, transform), "burned_mask.tif")
        if risk_map is not None:
            st.download_button("Download Risk Map GeoTIFF", export_geotiff(risk_map, transform), "risk_map.tif")
else:
    st.info("Please upload an STL file (or use local files) to begin analysis.")
