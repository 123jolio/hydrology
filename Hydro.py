import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile

# Configure the Streamlit page
st.set_page_config(page_title="STL Heatmap & Hydrogeology", layout="wide")
st.title("STL Heatmap & Hydrogeology Visualization")
st.markdown("""
Upload an STL file to generate a DEM heatmap based on its elevation (z‑values).  
You can adjust the raw elevation values and then constrain the final DEM to a specific range.  
Additionally, this app computes a slope map and an aspect map derived from the DEM for basic hydrogeological analysis.
""")

# File uploader widget for STL file
uploaded_file = st.file_uploader("Upload your STL file", type=["stl"])

# Sidebar controls for raw elevation adjustments
st.sidebar.header("Raw Elevation Adjustment")
scale_factor = st.sidebar.slider("Elevation Scale Factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                 help="Multiply raw elevation values by this factor")
elevation_offset = st.sidebar.slider("Elevation Offset", min_value=-100.0, max_value=100.0, value=0.0, step=1.0,
                                     help="Add this value to all raw elevation values")

# Sidebar controls for post‑interpolation DEM adjustment
st.sidebar.header("DEM Elevation Range")
dem_min = st.sidebar.number_input("DEM Minimum Elevation", value=0.0, step=1.0, help="Minimum elevation for DEM")
dem_max = st.sidebar.number_input("DEM Maximum Elevation", value=400.0, step=1.0, help="Maximum elevation for DEM")

# Sidebar widget for grid resolution
grid_res = st.sidebar.number_input("Grid resolution", min_value=100, max_value=1000, value=500, step=50)

if uploaded_file is not None:
    # Write the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_file.read())
        stl_filename = tmp_file.name

    try:
        # Load the STL mesh using numpy-stl
        mesh_data = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract vertices from the mesh (each triangle has 3 vertices)
    vertices = mesh_data.vectors.reshape(-1, 3)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Apply raw elevation adjustments
    z_adjusted = (z * scale_factor) + elevation_offset

    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate the scattered adjusted z-values onto the grid using cubic interpolation
    grid_z = griddata((x, y), z_adjusted, (grid_x, grid_y), method='cubic')

    # Apply post-interpolation clipping to constrain the DEM within the specified range
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # Calculate grid spacing (assumes uniform spacing)
    dx = (xi[-1] - xi[0]) / (grid_res - 1)
    dy = (yi[-1] - yi[0]) / (grid_res - 1)

    # Compute gradients in x and y directions
    dz_dx, dz_dy = np.gradient(grid_z, dx, dy)

    # Compute slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Compute aspect in degrees
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360  # convert to 0-360 range

    # Create tabs for visualizing results
    tab1, tab2, tab3 = st.tabs(["DEM Heatmap", "Slope Map", "Aspect Map"])

    with tab1:
        st.subheader("DEM Heatmap (Adjusted Elevation)")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        im1 = ax1.imshow(
            grid_z, 
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower', 
            cmap='hot', 
            aspect='auto'
        )
        ax1.set_title("DEM Heatmap (Adjusted Elevation)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        fig1.colorbar(im1, ax=ax1, label="Elevation (Z)")
        st.pyplot(fig1)

    with tab2:
        st.subheader("Slope Map")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        im2 = ax2.imshow(
            slope, 
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower', 
            cmap='viridis', 
            aspect='auto'
        )
        ax2.set_title("Slope Map (Degrees)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        fig2.colorbar(im2, ax=ax2, label="Slope (°)")
        st.pyplot(fig2)

    with tab3:
        st.subheader("Aspect Map")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        im3 = ax3.imshow(
            aspect, 
            extent=(x.min(), x.max(), y.min(), y.max()),
            origin='lower', 
            cmap='twilight', 
            aspect='auto'
        )
        ax3.set_title("Aspect Map (Degrees)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        fig3.colorbar(im3, ax=ax3, label="Aspect (°)")
        st.pyplot(fig3)
else:
    st.info("Please upload an STL file to generate the hydrogeological maps.")
