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

# -----------------------------------------------------------------------------
# 1. MUST be the first Streamlit command!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 2. Inject minimal, professional CSS (dark sidebar, no gradient)
#    This CSS is not printed to the UI; it's just injected.
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
# 3. Header (logo + title)
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    pass  # No logo found, silently ignore

st.title("Advanced Hydrogeology & DEM Analysis (with Scenario-Based GIFs)")

st.markdown("""
This application creates a DEM from an STL file, computes advanced hydrogeological maps (slope, aspect), 
simulates a runoff hydrograph, retention time, nutrient leaching, and optionally handles a burned-area GeoTIFF.  

**New**: Each scenario (flow, retention, nutrients, risk) can produce a **GIF animation** that depends on user inputs.  
*(Animations here are illustrative placeholders—replace with your real modeling logic!)*
""")

# -----------------------------------------------------------------------------
# 4. Georeference bounding box (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

# -----------------------------------------------------------------------------
# 5. File upload: STL and optional burned-area GeoTIFF
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Upload STL file (for DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Optional: Upload burned-area GeoTIFF", type=["tif","tiff"])

# -----------------------------------------------------------------------------
# 6. Sidebar: DEM & Elevation
# -----------------------------------------------------------------------------
st.sidebar.header("DEM & Elevation")
scale_factor = st.sidebar.slider("Elevation Scale Factor", 0.1, 5.0, 1.0, 0.1)
elevation_offset = st.sidebar.slider("Elevation Offset (m)", -100.0, 100.0, 0.0, 1.0)
dem_min = st.sidebar.number_input("Min Elevation (m)", value=0.0, step=1.0)
dem_max = st.sidebar.number_input("Max Elevation (m)", value=500.0, step=1.0)
grid_res = st.sidebar.number_input("Grid Resolution", 100, 1000, 500, 50)

# -----------------------------------------------------------------------------
# 7. Sidebar: Flow & Retention
# -----------------------------------------------------------------------------
st.sidebar.header("Flow & Retention")
rainfall_intensity = st.sidebar.number_input("Rainfall (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Rainfall Duration (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Catchment Area (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Runoff Coefficient", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Recession Rate (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Simulation (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Storage Volume (m³)", value=5000.0, step=100.0)

# -----------------------------------------------------------------------------
# 8. Sidebar: Nutrient Leaching
# -----------------------------------------------------------------------------
st.sidebar.header("Nutrient Leaching")
soil_nutrient = st.sidebar.number_input("Soil Nutrient (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Vegetation Retention", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Soil Erosion Factor", 0.0, 1.0, 0.3, 0.05)

# -----------------------------------------------------------------------------
# 9. If STL is provided, generate DEM and do advanced analysis
# -----------------------------------------------------------------------------
risk_map = None  # For burned-area scenario
if uploaded_stl is not None:
    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_filename = tmp_stl.name

    # Load STL
    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Error reading STL: {e}")
        st.stop()

    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:,0]
    y_raw = vertices[:,1]
    z_raw = vertices[:,2]

    # Elevation adjust
    z_adj = (z_raw * scale_factor) + elevation_offset

    # Map x,y to lon/lat
    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min)*(right_bound-left_bound)/(x_max-x_min)
    lat_raw = top_bound - (y_raw - y_min)*(top_bound-bottom_bound)/(y_max-y_min)

    # DEM grid
    xi = np.linspace(left_bound, right_bound, grid_res)
    yi = np.linspace(top_bound, bottom_bound, grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, dem_min, dem_max)

    # Slope & aspect
    dx = (right_bound-left_bound)/(grid_res-1)
    dy = (top_bound-bottom_bound)/(grid_res-1)
    dz_dx, dz_dy = np.gradient(grid_z, dx, dy)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    # Transform for GeoTIFF
    transform = from_origin(left_bound, top_bound, dx, dy)

    # Flow simulation
    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity/1000.0)*event_duration
    V_runoff = total_rain_m*area_m2*runoff_coeff
    Q_peak = V_runoff/event_duration

    t = np.linspace(0, simulation_hours, int(simulation_hours*60))  # minute steps
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak*(time/event_duration)  # linear rise
        else:
            Q[i] = Q_peak*np.exp(-recession_rate*(time-event_duration))

    # Retention
    if V_runoff>0:
        retention_time = storage_volume/(V_runoff/event_duration)
    else:
        retention_time = None

    # Nutrient
    nutrient_load = soil_nutrient*(1-veg_retention)*erosion_factor*catchment_area

    # Optional: Burned-area risk
    if uploaded_burned is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_burn:
            tmp_burn.write(uploaded_burned.read())
            burned_filename = tmp_burn.name
        try:
            with rasterio.open(burned_filename) as src:
                burned_img = src.read()  # shape (bands, h, w)
                src_transform = src.transform
                src_crs = src.crs
        except Exception as e:
            st.warning(f"Error reading burned TIFF: {e}")
            burned_img = None

        if burned_img is not None and burned_img.shape[0]>=3:
            # threshold "light green"
            burned_mask = np.logical_and.reduce((
                burned_img[0]>=100, burned_img[0]<=180,
                burned_img[1]>=200, burned_img[1]<=255,
                burned_img[2]>=100, burned_img[2]<=180
            )).astype(np.uint8)

            # Resample
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
            eps = 0.01
            risk_map = burned_mask_resampled*(1/(slope+eps))
            rmin, rmax = risk_map.min(), risk_map.max()
            if rmax>rmin:
                risk_map = (risk_map-rmin)/(rmax-rmin)
            else:
                risk_map[:] = 0

    # -----------------------------------------------------------------------------
    # Create scenario-based GIFs (placeholder animations)
    # -----------------------------------------------------------------------------
    # We'll show how to do a simple time-lapse for each scenario.

    def create_placeholder_gif(data_array, frames=10, scenario_name="flow"):
        """
        Produces a simple placeholder GIF with random scaling of data_array.
        In a real scenario, you'd do real time-step modeling for each frame.
        """
        images = []
        for i in range(frames):
            # Example: scale data by a factor that changes with i
            factor = 1 + 0.1*i
            array_i = np.clip(data_array*factor, 0, 1e9)  # just a silly transform

            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(array_i, origin='lower', cmap='hot')
            ax.set_title(f"{scenario_name.capitalize()} Frame {i+1}")
            ax.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80)
            plt.close(fig)
            buf.seek(0)
            images.append(imageio.imread(buf))

        # combine into GIF in memory
        gif_bytes = io.BytesIO()
        imageio.mimsave(gif_bytes, images, format='GIF', fps=2)
        gif_bytes.seek(0)
        return gif_bytes.getvalue()

    # We'll create four placeholder arrays for demonstration
    # In reality, you'd compute the actual time-step results for each scenario.

    # 1) Flow scenario
    flow_placeholder = np.clip(grid_z/np.max(grid_z+1e-9), 0, 1)  # normalize
    flow_gif = create_placeholder_gif(flow_placeholder, scenario_name="flow")

    # 2) Retention scenario
    retention_placeholder = np.clip(slope/np.max(slope+1e-9), 0, 1)
    retention_gif = create_placeholder_gif(retention_placeholder, scenario_name="retention")

    # 3) Nutrient scenario
    nutrient_placeholder = np.clip(aspect/360, 0, 1)
    nutrient_gif = create_placeholder_gif(nutrient_placeholder, scenario_name="nutrient")

    # 4) Risk scenario (only if risk_map is available)
    risk_gif = None
    if risk_map is not None:
        risk_placeholder = np.clip(risk_map, 0, 1)
        risk_gif = create_placeholder_gif(risk_placeholder, scenario_name="risk")

    # -----------------------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "Nutrient Leaching",
        "Burned Risk", "GeoTIFF Export", "Scenario GIFs"
    ])

    with tab1:
        st.subheader("DEM Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(grid_z, extent=(left_bound,right_bound,bottom_bound,top_bound),
                       origin='lower', cmap='hot', vmin=0, vmax=500, aspect='auto')
        ax.set_title("DEM (Adjusted Elevation)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Elevation (m)")
        st.pyplot(fig)

    with tab2:
        st.subheader("Slope Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(slope, extent=(left_bound,right_bound,bottom_bound,top_bound),
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Slope (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Slope (°)")
        st.pyplot(fig)

    with tab3:
        st.subheader("Aspect Map")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(aspect, extent=(left_bound,right_bound,bottom_bound,top_bound),
                       origin='lower', cmap='twilight', aspect='auto')
        ax.set_title("Aspect (Degrees)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Aspect (°)")
        st.pyplot(fig)

    with tab4:
        st.subheader("Flow Simulation")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(t, Q, 'b-')
        ax.set_title("Runoff Hydrograph")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Flow (m³/hr)")
        st.pyplot(fig)
        st.write(f"Peak Flow: {Q_peak:.2f} m³/hr")
        st.write(f"Total Runoff Volume: {V_runoff:.2f} m³")

    with tab5:
        st.subheader("Retention Time")
        if retention_time is not None:
            st.write(f"Estimated Retention Time: {retention_time:.2f} hr")
        else:
            st.warning("No effective runoff -> Retention time not applicable.")

    with tab6:
        st.subheader("Nutrient Leaching")
        st.write(f"Soil Nutrient Content: {soil_nutrient} kg/ha")
        st.write(f"Vegetation Retention Factor: {veg_retention}")
        st.write(f"Soil Erosion Factor: {erosion_factor}")
        st.write(f"Catchment Area: {catchment_area} ha")
        st.write(f"Estimated Nutrient Load: {nutrient_load:.2f} kg")

    with tab7:
        st.subheader("Burned-Area Risk")
        if risk_map is not None:
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(risk_map, extent=(left_bound,right_bound,bottom_bound,top_bound),
                           origin='lower', cmap='inferno', aspect='auto')
            ax.set_title("Risk Map (Burned & Accumulation)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Risk Score (0-1)")
            st.pyplot(fig)
        else:
            st.info("No burned-area GeoTIFF or no valid data -> Risk map unavailable.")

    with tab8:
        st.subheader("Export GeoTIFFs")
        # Export DEM, slope, aspect
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

        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Download Risk (GeoTIFF)", risk_tiff, "RiskMap.tif", "image/tiff")

    with tab9:
        st.subheader("Scenario-Based GIF Animations (Placeholder)")
        st.markdown("Below are **placeholder** animations showing how you might illustrate time-step changes.")
        
        st.markdown("**Flow Scenario**")
        st.image(flow_gif, caption="Flow scenario (demo)")

        st.markdown("**Retention Scenario**")
        st.image(retention_gif, caption="Retention scenario (demo)")

        st.markdown("**Nutrient Scenario**")
        st.image(nutrient_gif, caption="Nutrient scenario (demo)")

        if risk_gif is not None:
            st.markdown("**Risk Scenario**")
            st.image(risk_gif, caption="Risk scenario (demo)")

else:
    st.info("Please upload an STL file to generate DEM and scenario analyses.")
