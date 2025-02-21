import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile

# Configure the Streamlit page
st.set_page_config(page_title="STL Heatmap Visualization", layout="wide")
st.title("STL Heatmap Visualization")
st.markdown("""
Upload an STL file to generate a heatmap based on its elevation (z‑values).  
The application extracts the vertices, interpolates the z‑values over a grid, and plots the heatmap.
""")

# File uploader widget for STL file
uploaded_file = st.file_uploader("Upload your STL file", type=["stl"])

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

    # Sidebar widget for grid resolution
    grid_res = st.sidebar.number_input("Grid resolution", min_value=100, max_value=1000, value=500, step=50)

    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Interpolate the scattered z values onto the grid using cubic interpolation
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Plotting the heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(
        grid_z, 
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin='lower', 
        cmap='hot', 
        aspect='auto'
    )
    ax.set_title("Heatmap of STL Mesh (Z-values)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(heatmap, ax=ax, label="Elevation (Z)")

    st.pyplot(fig)
else:
    st.info("Please upload an STL file to generate the heatmap.")
