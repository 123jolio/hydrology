import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from scipy.interpolate import griddata
import tempfile

# --- Page Configuration & Logo ---
st.set_page_config(page_title="Hydrogeology & DEM Analysis", layout="wide")
# Display logo (make sure 'logo.png' is in the same folder)
try:
    st.image("logo.png", width=200)
except Exception:
    st.write("Logo file not found.")

st.title("Hydrogeology & DEM Analysis")
st.markdown("""
This application performs DEM generation from an STL file (with adjustable elevation parameters),
computes slope and aspect maps, and includes additional hydrogeological tools such as a simple
flow simulation and retention time calculation.
""")

# --- STL File Upload and DEM Generation ---

uploaded_file = st.file_uploader("Upload your STL file", type=["stl"])

# Sidebar for DEM adjustment options
st.sidebar.header("DEM & Elevation Adjustment")
scale_factor = st.sidebar.slider("Raw Elevation Scale Factor", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
elevation_offset = st.sidebar.slider("Raw Elevation Offset", min_value=-100.0, max_value=100.0, value=0.0, step=1.0)

st.sidebar.header("DEM Clipping")
dem_min = st.sidebar.number_input("DEM Minimum Elevation (m)", value=0.0, step=1.0)
dem_max = st.sidebar.number_input("DEM Maximum Elevation (m)", value=400.0, step=1.0)

grid_res = st.sidebar.number_input("Grid Resolution", min_value=100, max_value=1000, value=500, step=50)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_file:
        tmp_file.write(uploaded_file.read())
        stl_filename = tmp_file.name

    try:
        mesh_data = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL file: {e}")
        st.stop()

    # Extract vertices from mesh (each triangle has 3 vertices)
    vertices = mesh_data.vectors.reshape(-1, 3)
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    # Adjust raw elevations
    z_adjusted = (z * scale_factor) + elevation_offset

    # Create a regular grid for interpolation
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((x, y), z_adjusted, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # --- Compute Hydrogeological Parameters ---
    # Compute grid spacing (assumes uniform grid)
    dx = (xi[-1] - xi[0]) / (grid_res - 1)
    dy = (yi[-1] - yi[0]) / (grid_res - 1)
    dz_dx, dz_dy = np.gradient(grid_z, dx, dy)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect = (aspect + 360) % 360

    # --- Tabs for Visualization ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DEM Heatmap", "Slope Map", "Aspect Map", "Flow Simulation", "Retention Time"])

    with tab1:
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid_z, extent=(x.min(), x.max(), y.min(), y.max()),
                       origin='lower', cmap='hot', aspect='auto')
        ax.set_title("DEM Heatmap (Adjusted Elevation)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    with tab2:
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(slope, extent=(x.min(), x.max(), y.min(), y.max()),
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Slope Map (Degrees)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    with tab3:
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(aspect, extent=(x.min(), x.max(), y.min(), y.max()),
                       origin='lower', cmap='twilight', aspect='auto')
        ax.set_title("Aspect Map (Degrees)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    # --- Additional Hydrogeology Tools ---

    with tab4:
        st.subheader("Flow Simulation")
        st.markdown("""
        This simple simulation estimates runoff response from a rainfall event.
        Adjust the rainfall intensity and event duration to see a simulated hydrograph.
        """)
        # Simulation parameters
        rainfall_intensity = st.number_input("Rainfall Intensity (mm/hr)", value=20.0, step=1.0)
        event_duration = st.number_input("Event Duration (hr)", value=2.0, step=0.5)
        recession_rate = st.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
        simulation_time = st.slider("Simulation Duration (hr)", 1, 24, 12)

        # Simple simulation: assume a peak flow followed by exponential recession
        # (This is a very simplified model; actual hydrograph modeling is more complex.)
        t = np.linspace(0, simulation_time, simulation_time * 60)  # time in minutes
        # Calculate peak runoff using a rational method: Q = C * i * A (simplified; here we use an arbitrary scaling)
        peak_flow = rainfall_intensity * 0.1  # arbitrary scaling factor
        # Exponential recession: Q(t) = Q_peak * exp(-k*t)
        flow = peak_flow * np.exp(-recession_rate * t / 60)  # recession_rate per hour, convert t to hours

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, flow, label="Flow (arbitrary units)")
        ax.set_title("Simulated Runoff Hydrograph")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Flow")
        ax.legend()
        st.pyplot(fig)

    with tab5:
        st.subheader("Retention Time Calculation")
        st.markdown("""
        Calculate the estimated retention time using a simplified approach.
        Enter the following parameters:
        - **Rainfall Depth (mm)**
        - **Catchment Area (ha)**
        - **Runoff Coefficient (0–1)**
        - **Storage Volume (m³)**
        
        Retention Time (hr) is estimated as:
        
        \\[
        \\text{Retention Time} = \\frac{\\text{Storage Volume}}{\\text{Effective Runoff}}
        \\]
        
        where \\( \\text{Effective Runoff} = \\text{Rainfall Depth (m)} \\times \\text{Catchment Area (m²)} \\times \\text{Runoff Coefficient} \\).
        """)
        rainfall_depth = st.number_input("Rainfall Depth (mm)", value=50.0, step=1.0)
        catchment_area = st.number_input("Catchment Area (ha)", value=10.0, step=0.1)
        runoff_coefficient = st.slider("Runoff Coefficient", 0.0, 1.0, 0.5, step=0.05)
        storage_volume = st.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

        # Convert rainfall depth to meters, catchment area from ha to m²
        rainfall_m = rainfall_depth / 1000.0
        area_m2 = catchment_area * 10000.0
        effective_runoff = rainfall_m * area_m2 * runoff_coefficient  # in m³

        if effective_runoff > 0:
            retention_time = storage_volume / effective_runoff  # in hours (simplified)
            st.markdown(f"**Estimated Retention Time:** {retention_time:.2f} hours")
        else:
            st.warning("Effective runoff is zero; please check your parameters.")
else:
    st.info("Please upload an STL file to begin the analysis.")
