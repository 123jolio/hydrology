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

# -----------------------------------------------------------------------------
# 1. MUST be the very first Streamlit command!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 2. Inject minimal, professional CSS (dark sidebar, no gradient, no raw CSS displayed)
# -----------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
/* Minimal, Professional UI */

/* Overall background and font */
html, body {
    background-color: #f5f5f5;
    font-family: "Roboto", sans-serif;
    color: #333;
}

/* Solid dark sidebar (no gradient) */
[data-testid="stSidebar"] > div:first-child {
    background: #2a2a2a; /* solid dark color */
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
# 3. Header with logo and title
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    st.write("Logo file not found.")

st.title("Advanced Hydrogeology & DEM Analysis")
st.markdown("""
This application generates a high-quality Digital Elevation Model (DEM) from an STL file, computes advanced hydrogeological maps (slope and aspect), simulates a runoff hydrograph, calculates a retention time estimate, and estimates potential nutrient leaching.

Optionally, you can upload a burned–area GeoTIFF (with burned areas marked as light green) to produce a risk map that estimates the potential for accumulation of burned vegetation and soil (increasing erosion risk).

All outputs can be exported as georeferenced GeoTIFF files.
""")

# -----------------------------------------------------------------------------
# 4. Georeference (bounding box) for the DEM products (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound = 27.906069      # Longitude (Top left)
top_bound = 36.92337189     # Latitude (Top left)
right_bound = 28.045764     # Longitude (Bottom right)
bottom_bound = 36.133509    # Latitude (Bottom right)

# -----------------------------------------------------------------------------
# 5. File upload: STL file and optional burned area GeoTIFF
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Upload your STL file", type=["stl"])
uploaded_burned = st.file_uploader("Optional: Upload burned area GeoTIFF (light green = burned)", type=["tif", "tiff"])

# -----------------------------------------------------------------------------
# 6. Sidebar: DEM and Elevation Adjustment
# -----------------------------------------------------------------------------
st.sidebar.header("DEM & Elevation Adjustment")
scale_factor = st.sidebar.slider("Raw Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
elevation_offset = st.sidebar.slider("Raw Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0)
st.sidebar.header("DEM Clipping")
dem_min = st.sidebar.number_input("DEM Minimum Elevation (m)", value=0.0, step=1.0)
dem_max = st.sidebar.number_input("DEM Maximum Elevation (m)", value=500.0, step=1.0)
grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, 50)

# -----------------------------------------------------------------------------
# 7. Sidebar: Flow Simulation Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Flow Simulation Parameters")
rainfall_intensity = st.sidebar.number_input("Rainfall Intensity (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Event Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coefficient = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_duration = st.sidebar.number_input("Hydrograph Simulation Duration (hr)", value=6.0, step=0.5)

# -----------------------------------------------------------------------------
# 8. Sidebar: Retention Time
# -----------------------------------------------------------------------------
st.sidebar.header("Retention Time Parameters")
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

# -----------------------------------------------------------------------------
# 9. Sidebar: Nutrient Leaching
# -----------------------------------------------------------------------------
st.sidebar.header("Nutrient Leaching Parameters")
soil_nutrient = st.sidebar.number_input("Soil Nutrient Content (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention Factor", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05)

# -----------------------------------------------------------------------------
# 10. Process STL file and generate DEM & Hydro Maps
# -----------------------------------------------------------------------------
if uploaded_stl is not None:
    # Save STL to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_stl.read())
        stl_filename = tmp_file.name

    # Load the STL mesh
    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract raw coordinates (each triangle has 3 vertices)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]

    # Adjust raw elevations
    z_adj = (z_raw * scale_factor) + elevation_offset

    # Map raw x, y to lon/lat using bounding box
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    # Create georeferenced grid
    xi = np.linspace(left_bound, right_bound, grid_res)
    yi = np.linspace(top_bound, bottom_bound, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate DEM
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # Grid spacing (degrees)
    pixel_width = (right_bound - left_bound) / (grid_res - 1)
    pixel_height = (top_bound - bottom_bound) / (grid_res - 1)

    # Slope & Aspect
    dz_dx, dz_dy = np.gradient(grid_z, pixel_width, pixel_height)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360

    # Affine transform for GeoTIFF
    transform = from_origin(left_bound, top_bound, pixel_width, pixel_height)

    # Flow Simulation
    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coefficient
    Q_peak = V_runoff / event_duration
    t = np.linspace(0, simulation_duration, int(simulation_duration * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak * (time / event_duration)
        else:
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    # Retention Time
    retention_time = V_runoff > 0 and (storage_volume / (V_runoff / event_duration)) or None

    # Nutrient Leaching
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # Optional: Load burned area and compute risk
    risk_map = None
    if uploaded_burned is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_burned:
            tmp_burned.write(uploaded_burned.read())
            burned_filename = tmp_burned.name

        try:
            with rasterio.open(burned_filename) as src:
                burned_img = src.read()  # shape: (bands, height, width)
                src_transform = src.transform
                src_crs = src.crs
        except Exception as e:
            st.warning(f"Error reading burned area GeoTIFF: {e}")
            burned_img = None

        if burned_img is not None:
            # Simple threshold for "light green" in the image (adjust as needed)
            burned_mask = np.logical_and.reduce((
                burned_img[0] >= 100, burned_img[0] <= 180,  # R
                burned_img[1] >= 200, burned_img[1] <= 255,  # G
                burned_img[2] >= 100, burned_img[2] <= 180   # B
            ))
            burned_mask = burned_mask.astype(np.uint8)

            # Resample to DEM grid
            burned_mask_resampled = np.empty(grid_z.shape, dtype=np.uint8)
            reproject(
                source=burned_mask,
                destination=burned_mask_resampled,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest
            )

            # Simple risk: burned_mask * (1 / slope)
            epsilon = 0.01
            risk_map = burned_mask_resampled * (1.0 / (slope + epsilon))
            # Normalize 0..1
            risk_min, risk_max = risk_map.min(), risk_map.max()
            if risk_max > risk_min:
                risk_map = (risk_map - risk_min) / (risk_max - risk_min)
            else:
                risk_map[:] = 0

    # Helper to export arrays as GeoTIFF
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

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "Export GeoTIFFs",
        "Nutrient Leaching", "Risk Map"
    ])

    with tab1:
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=500)
        ax.set_title("DEM Heatmap (Adjusted Elevation)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m) [0 - 500 m]")
        st.pyplot(fig)

    with tab2:
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Slope Map (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    with tab3:
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='twilight', aspect='auto')
        ax.set_title("Aspect Map (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    with tab4:
        st.subheader("Flow Simulation Hydrograph")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, Q, color="blue", lw=2)
        ax.set_title("Simulated Runoff Hydrograph")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Flow (m³/hr)")
        st.pyplot(fig)
        st.markdown(f"**Peak Flow:** {Q_peak:.2f} m³/hr")
        st.markdown(f"**Total Runoff Volume:** {V_runoff:.2f} m³")

    with tab5:
        st.subheader("Retention Time Calculation")
        if retention_time is not None:
            st.markdown(f"**Estimated Retention Time:** {retention_time:.2f} hours")
        else:
            st.warning("Effective runoff is zero; please check your input parameters.")

    with tab6:
        st.subheader("Export GeoTIFF Products")
        dem_tiff = export_geotiff(grid_z, transform)
        slope_tiff = export_geotiff(slope, transform)
        aspect_tiff = export_geotiff(aspect, transform)
        st.download_button("Download DEM GeoTIFF", dem_tiff, file_name="DEM.tif", mime="image/tiff")
        st.download_button("Download Slope GeoTIFF", slope_tiff, file_name="Slope.tif", mime="image/tiff")
        st.download_button("Download Aspect GeoTIFF", aspect_tiff, file_name="Aspect.tif", mime="image/tiff")

        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Download Risk Map GeoTIFF", risk_tiff, file_name="RiskMap.tif", mime="image/tiff")

    with tab7:
        st.subheader("Nutrient Leaching Simulation")
        st.markdown("""
        This simulation estimates the mass of soil nutrients (in kg) that could be washed out 
        from the catchment due to a rainfall event, considering vegetation retention and soil erosion.
        """)
        st.markdown(f"**Soil Nutrient Content:** {soil_nutrient} kg/ha")
        st.markdown(f"**Vegetation Retention Factor:** {veg_retention:.2f}")
        st.markdown(f"**Soil Erosion Factor:** {erosion_factor:.2f}")
        st.markdown(f"**Catchment Area:** {catchment_area} ha")
        st.markdown(f"**Estimated Nutrient Load Exported:** {nutrient_load:.2f} kg")
        if nutrient_load > 100:
            st.warning("High nutrient export may increase eutrophication risk.")
        else:
            st.success("Estimated nutrient export is moderate. Further analysis is recommended.")

    with tab8:
        st.subheader("Risk Map (Burned Areas & Erosion Accumulation)")
        if risk_map is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(risk_map, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='inferno', aspect='auto')
            ax.set_title("Risk Map: Higher = More Potential Accumulation")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Normalized Risk Score (0-1)")
            st.pyplot(fig)
        else:
            st.info("No burned area GeoTIFF provided; risk map not available.")
else:
    st.info("Please upload an STL file to begin the hydrogeological analysis.")
