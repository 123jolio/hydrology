import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile
import io
import rasterio
from rasterio.transform import from_origin

# --- Page Configuration & Logo ---
st.set_page_config(page_title="Hydrogeology & DEM Analysis", layout="wide")
try:
    st.image("logo.png", width=200)
except Exception:
    st.write("Logo file not found.")

st.title("Hydrogeology & DEM Analysis")
st.markdown("""
This application performs DEM generation from an STL file with adjustable elevation parameters,
computes slope and aspect maps, and provides options to export the resulting products as GeoTIFF files.
""")

# --- User-provided Georeference (Bounding Box) ---
# These coordinates are used to georeference the DEM.
# (Assumed from the extracted image coordinates)
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your STL file", type=["stl"])

# --- Sidebar: DEM & Elevation Adjustment ---
st.sidebar.header("DEM & Elevation Adjustment")
scale_factor = st.sidebar.slider("Raw Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
elevation_offset = st.sidebar.slider("Raw Elevation Offset", -100.0, 100.0, 0.0, 1.0)
st.sidebar.header("DEM Clipping")
dem_min = st.sidebar.number_input("DEM Minimum Elevation (m)", value=0.0, step=1.0)
dem_max = st.sidebar.number_input("DEM Maximum Elevation (m)", value=400.0, step=1.0)
grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, 50)

# --- Additional Hydro Tools (Flow simulation, retention, etc.) can be added here later ---

if uploaded_file is not None:
    # Save STL to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_file.read())
        stl_filename = tmp_file.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract vertices from the mesh
    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]

    # Apply raw elevation adjustments
    z_adj = (z_raw * scale_factor) + elevation_offset

    # Instead of using the raw x,y extents, we force the georeferenced extent using the provided bounding box.
    # Create grid over the bounding box.
    xi = np.linspace(left_bound, right_bound, grid_res)
    yi = np.linspace(top_bound, bottom_bound, grid_res)  # note: from top (high lat) to bottom (low lat)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate the adjusted z-values onto the grid
    # (We use the raw x,y positions from the STL, but the output grid is forced to our bounding box)
    grid_z = griddata((x_raw, y_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # --- Compute Hydrogeological Parameters ---
    # Calculate grid spacing using our defined bounds
    pixel_width = (right_bound - left_bound) / (grid_res - 1)
    pixel_height = (top_bound - bottom_bound) / (grid_res - 1)
    # Compute gradients (assumes uniform spacing)
    dz_dx, dz_dy = np.gradient(grid_z, pixel_width, pixel_height)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360

    # Define affine transform for the GeoTIFF (upper-left corner, pixel size)
    transform = from_origin(left_bound, top_bound, pixel_width, pixel_height)  # note: pixel_height positive here

    # --- Function to export a GeoTIFF into a BytesIO object ---
    def export_geotiff(array, transform, crs="EPSG:4326"):
        memfile = io.BytesIO()
        # Write single band float32 GeoTIFF
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

    # --- Visualization and Download Tabs ---
    tab1, tab2, tab3 = st.tabs(["DEM Heatmap", "Slope Map", "Aspect Map"])

    with tab1:
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='upper', cmap='hot', aspect='auto')
        ax.set_title("DEM Heatmap (Adjusted Elevation)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

        # Button to download DEM as GeoTIFF
        dem_tiff = export_geotiff(grid_z, transform)
        st.download_button("Download DEM GeoTIFF", dem_tiff, file_name="DEM.tif", mime="image/tiff")

    with tab2:
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='upper', cmap='viridis', aspect='auto')
        ax.set_title("Slope Map (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

        slope_tiff = export_geotiff(slope, transform)
        st.download_button("Download Slope GeoTIFF", slope_tiff, file_name="Slope.tif", mime="image/tiff")

    with tab3:
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='upper', cmap='twilight', aspect='auto')
        ax.set_title("Aspect Map (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

        aspect_tiff = export_geotiff(aspect, transform)
        st.download_button("Download Aspect GeoTIFF", aspect_tiff, file_name="Aspect.tif", mime="image/tiff")
else:
    st.info("Please upload an STL file to generate the hydrogeological maps and export options.")
