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
Upload an STL file to generate a heatmap based on its elevation (z‑values).  
You can adjust the elevation data in two stages:  
1. Adjust the raw elevation values (scale & offset) before interpolation.  
2. Constrain (clip) the final DEM’s elevation range to your expected bounds (default: 0–400 m).
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

    # Plotting the heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(
        grid_z, 
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin='lower', 
        cmap='hot', 
        aspect='auto'
    )
    ax.set_title("Heatmap of STL Mesh (DEM Adjusted)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(heatmap, ax=ax, label="DEM Elevation (Z)")

    st.pyplot(fig)
else:
    st.info("Please upload an STL file to generate the heatmap.")
