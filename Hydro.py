import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile
import io
import rasterio
from rasterio.transform import from_origin

# MUST be the very first Streamlit command!
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# Custom CSS for a professional, ultra-cool look
# -----------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
/* Title styling */
h1 {
    text-align: center;
    font-size: 3rem;
    color: #2e7bcf;
}

/* Sidebar styling: gradient background and white text */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #2e7bcf, #1a5a99);
    color: white;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .css-1d391kg p {
    color: white;
}

/* Style for tabs to remove borders and add spacing */
div.stTabs > div {
    border: none;
}

/* Button styling override */
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border-radius: 0.5rem;
    font-size: 1rem;
    padding: 0.5rem 1rem;
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
# Header with logo and title
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    st.write("Logo file not found.")

st.title("Advanced Hydrogeology & DEM Analysis")
st.markdown("""
This application generates a high-quality Digital Elevation Model (DEM) from an STL file, computes advanced hydrogeological maps (slope and aspect), simulates a runoff hydrograph, calculates a retention time estimate, and estimates potential nutrient leaching.  
All outputs can be exported as georeferenced GeoTIFF files.
""")

# -----------------------------------------------------------------------------
# Georeference (bounding box) for the DEM products (EPSG:4326)
# (Coordinates extracted from the image)
# -----------------------------------------------------------------------------
left_bound = 27.906069      # Longitude (Top left)
top_bound = 36.92337189       # Latitude (Top left)
right_bound = 28.045764       # Longitude (Bottom right)
bottom_bound = 36.133509      # Latitude (Bottom right)

# -----------------------------------------------------------------------------
# File upload (STL file)
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your STL file", type=["stl"])

# -----------------------------------------------------------------------------
# Sidebar: DEM and Elevation Adjustment Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("DEM & Elevation Adjustment")
scale_factor = st.sidebar.slider("Raw Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1,
                                 help="Multiply the raw elevation values by this factor")
elevation_offset = st.sidebar.slider("Raw Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0,
                                     help="Add this offset to all raw elevation values")
st.sidebar.header("DEM Clipping")
dem_min = st.sidebar.number_input("DEM Minimum Elevation (m)", value=0.0, step=1.0)
dem_max = st.sidebar.number_input("DEM Maximum Elevation (m)", value=500.0, step=1.0)
grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, 50)

# -----------------------------------------------------------------------------
# Sidebar: Flow Simulation Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Flow Simulation Parameters")
rainfall_intensity = st.sidebar.number_input("Rainfall Intensity (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Event Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coefficient = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_duration = st.sidebar.number_input("Hydrograph Simulation Duration (hr)", value=6.0, step=0.5)

# -----------------------------------------------------------------------------
# Sidebar: Retention Time Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Retention Time Parameters")
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

# -----------------------------------------------------------------------------
# Sidebar: Nutrient Leaching Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Nutrient Leaching Parameters")
soil_nutrient = st.sidebar.number_input("Soil Nutrient Content (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention Factor", 0.0, 1.0, 0.7, 0.05,
                                  help="Fraction of soil nutrients retained by vegetation (1 = full retention)")
erosion_factor = st.sidebar.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05,
                                   help="Fraction of soil nutrients mobilized due to erosion")

# -----------------------------------------------------------------------------
# Process STL file and generate DEM & Hydro Maps
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    # Save the uploaded STL file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_file.read())
        stl_filename = tmp_file.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract vertices from the STL mesh (each triangle has 3 vertices)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]

    # Apply raw elevation adjustments (scale and offset)
    z_adj = (z_raw * scale_factor) + elevation_offset

    # --- Transform raw x,y coordinates to lon/lat ---
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    # Create a georeferenced grid using the provided bounding box
    xi = np.linspace(left_bound, right_bound, grid_res)
    yi = np.linspace(top_bound, bottom_bound, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate the adjusted elevation values onto the grid using cubic interpolation.
    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # Determine grid spacing (approximation in degrees)
    pixel_width = (right_bound - left_bound) / (grid_res - 1)
    pixel_height = (top_bound - bottom_bound) / (grid_res - 1)

    # Compute gradients to derive slope and aspect
    dz_dx, dz_dy = np.gradient(grid_z, pixel_width, pixel_height)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360

    # Define affine transform for GeoTIFF export (upper-left corner, pixel size)
    transform = from_origin(left_bound, top_bound, pixel_width, pixel_height)

    # -----------------------------------------------------------------------------
    # Flow Simulation (Advanced Hydrograph)
    # -----------------------------------------------------------------------------
    area_m2 = catchment_area * 10000.0  # Convert ha to m²
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration  # Total rainfall depth (m)
    V_runoff = total_rain_m * area_m2 * runoff_coefficient  # Effective runoff volume (m³)
    Q_peak = V_runoff / event_duration  # Approximate peak flow (m³/hr)

    t = np.linspace(0, simulation_duration, int(simulation_duration * 60))  # time in hours (minute resolution)
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak * (time / event_duration)
        else:
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    # -----------------------------------------------------------------------------
    # Retention Time Calculation (Using effective runoff)
    # -----------------------------------------------------------------------------
    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None

    # -----------------------------------------------------------------------------
    # Nutrient Leaching Simulation
    # -----------------------------------------------------------------------------
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # -----------------------------------------------------------------------------
    # Function to export arrays as GeoTIFF using rasterio
    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    # Display results in multiple tabs
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "Export GeoTIFFs", "Nutrient Leaching"
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

    with tab7:
        st.subheader("Nutrient Leaching Simulation")
        st.markdown("""
        This simulation estimates the mass of soil nutrients (in kg) that could be washed out 
        from the catchment due to a rainfall event, considering the effects of vegetation 
        retention and soil erosion.
        """)
        st.markdown(f"**Soil Nutrient Content:** {soil_nutrient} kg/ha")
        st.markdown(f"**Vegetation Retention Factor:** {veg_retention:.2f}")
        st.markdown(f"**Soil Erosion Factor:** {erosion_factor:.2f}")
        st.markdown(f"**Catchment Area:** {catchment_area} ha")
        st.markdown(f"**Estimated Nutrient Load Exported:** {nutrient_load:.2f} kg")
        if nutrient_load > 100:
            st.warning("High nutrient export may increase the risk of eutrophication in downstream water bodies.")
        else:
            st.success("Estimated nutrient export is moderate. Further analysis is recommended.")
else:
    st.info("Please upload an STL file to generate the hydrogeological maps and analyses.")
