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

# -----------------------------------------------------------------------------
# 1. Ρύθμιση Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 2. Ενσωμάτωση CSS για σκοτεινή πλαϊνή μπάρα και καθαρό στυλ
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
h1 {
    text-align: center;
    font-size: 3rem;
    color: #2e7bcf;
}
.stButton>button {
    background-color: #2e7bcf;
    color: white;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Επικεφαλίδα
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    pass

st.title("Advanced Hydrogeology & DEM Analysis")
st.markdown("""
Αυτή η εφαρμογή δημιουργεί DEM από STL, υπολογίζει τοπογραφικούς χάρτες (κλίση, προσανατολισμό), 
προσομοιώνει ροή, διατήρηση, διάχυση θρεπτικών και κίνδυνο καμένων περιοχών.  
Υποστηρίζει KMZ, TIFF ή JPG/PNG για καμένες περιοχές και εξάγει GeoTIFF αποτελέσματα.
""")

# -----------------------------------------------------------------------------
# 4. Γεωαναφορά (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound, top_bound, right_bound, bottom_bound = 27.906069, 36.92337189, 28.045764, 36.133509

# -----------------------------------------------------------------------------
# 5. Ανέβασμα αρχείων
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Ανεβάστε STL (για DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Ανεβάστε δεδομένα καμένων (KMZ, TIFF, JPG, PNG)", type=["kmz", "tif", "tiff", "jpg", "png"])

# -----------------------------------------------------------------------------
# 6. Sidebar Παράμετροι
# -----------------------------------------------------------------------------
st.sidebar.header("Global DEM & Elevation")
global_scale = st.sidebar.slider("Κλίμακα Ύψους", 0.1, 5.0, 1.0, 0.1)
global_offset = st.sidebar.slider("Μετατόπιση Ύψους (m)", -100.0, 100.0, 0.0, 1.0)
global_dem_min = st.sidebar.number_input("Ελάχιστο Ύψος (m)", value=0.0, step=1.0)
global_dem_max = st.sidebar.number_input("Μέγιστο Ύψος (m)", value=500.0, step=1.0)
global_grid_res = st.sidebar.number_input("Ανάλυση Πλέγματος", 100, 1000, 500, 50)

st.sidebar.header("Flow & Retention")
rainfall_intensity = st.sidebar.number_input("Βροχόπτωση (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Διάρκεια Βροχόπτωσης (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Έκταση Λεκάνης (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Συντελεστής Ροής", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Ρυθμός Υποχώρησης (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Διάρκεια Προσομοίωσης (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Όγκος Αποθήκευσης (m³)", value=5000.0, step=100.0)

st.sidebar.header("Nutrient Leaching")
soil_nutrient = st.sidebar.number_input("Θρεπτικό Περιεχόμενο (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Διατήρηση Βλάστησης", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Συντελεστής Διάβρωσης", 0.0, 1.0, 0.3, 0.05)

st.sidebar.header("Slope & Risk Parameters")
slope_threshold = st.sidebar.number_input("Όριο Κλίσης για Κίνδυνο (°)", value=30.0, step=1.0)
risk_slope_weight = st.sidebar.slider("Βάρος Κλίσης", 0.0, 2.0, 1.0, 0.1)
risk_dem_weight = st.sidebar.slider("Βάρος Ύψους", 0.0, 2.0, 1.0, 0.1)
risk_burned_weight = st.sidebar.slider("Βάρος Καμένων", 0.0, 2.0, 1.0, 0.1)
risk_rain_weight = st.sidebar.slider("Βάρος Βροχόπτωσης", 0.0, 2.0, 1.0, 0.1)

st.sidebar.header("Dynamic Risk Simulation")
simulate_risk = st.sidebar.checkbox("Προσομοίωση Κινδύνου")
if simulate_risk:
    sim_rain_intensity = st.sidebar.slider("Ένταση Βροχής (mm/hr)", 0.0, 100.0, rainfall_intensity)
    sim_duration = st.sidebar.slider("Διάρκεια (hr)", 0.0, 10.0, event_duration)

# -----------------------------------------------------------------------------
# 7. Συναρτήσεις Επεξεργασίας
# -----------------------------------------------------------------------------
def process_dem(stl_file, scale, offset, grid_res):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(stl_file.read())
            stl_mesh = mesh.Mesh.from_file(tmp.name)
        vertices = stl_mesh.vectors.reshape(-1, 3)
        x_raw, y_raw, z_raw = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        z_adj = (z_raw * scale) + offset
        x_min, x_max = x_raw.min(), x_raw.max()
        y_min, y_max = y_raw.min(), y_raw.max()
        lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
        lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)
        xi = np.linspace(left_bound, right_bound, grid_res)
        yi = np.linspace(bottom_bound, top_bound, grid_res)
        grid_x, grid_y = np.meshgrid(xi, yi)
        grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='linear')
        return np.clip(grid_z, global_dem_min, global_dem_max)
    except Exception as e:
        st.error(f"Σφάλμα επεξεργασίας STL: {e}")
        return None

def compute_slope_aspect(dem, dx_meters, dy_meters):
    dz_dx, dz_dy = np.gradient(dem, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360
    return slope, aspect

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
        if best_neighbor:
            rr, cc = best_neighbor
            acc[rr, cc] += acc[r, c]
    return acc

def process_burned_data(uploaded_file, grid_shape, transform):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".kmz":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp:
            tmp.write(uploaded_file.read())
            with zipfile.ZipFile(tmp.name, 'r') as zf:
                kml_file = [n for n in zf.namelist() if n.endswith('.kml')][0]
                kml_data = zf.read(kml_file)
        k = kml.KML()
        k.from_string(kml_data)
        polygons = []
        for feature in k.features():
            for placemark in feature.features():
                if hasattr(placemark, 'geometry') and placemark.geometry:
                    geom = placemark.geometry
                    if geom.geom_type == "Polygon":
                        polygons.append(geom)
                    elif geom.geom_type == "MultiPolygon":
                        polygons.extend(list(geom.geoms))
        if polygons:
            shapes = [(poly, 1) for poly in polygons]
            return rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, dtype=np.uint8)
        return None
    elif ext in [".tif", ".tiff"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            with rasterio.open(tmp.name) as src:
                img = src.read(1)
                norm_img = (img - img.min()) / (img.max() - img.min() + 1e-9)
                resampled = np.empty(grid_shape, dtype=np.float32)
                reproject(
                    source=norm_img,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs or "EPSG:4326",
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest
                )
                return resampled
    elif ext in [".jpg", ".png"]:
        img = imageio.imread(uploaded_file)
        if img.ndim == 3:
            mask = ((img[..., 0] > 150) & (img[..., 1] < 100) & (img[..., 2] < 100)).astype(np.uint8)
        else:
            mask = (img > 128).astype(np.uint8)
        return np.array(Image.fromarray(mask).resize((grid_shape[1], grid_shape[0]), resample=Image.NEAREST))
    return None

def create_gif(data_array, frames=10, fps=2, title="Scenario"):
    images = []
    for i in range(frames):
        factor = 1 + 0.1 * i
        array_i = np.clip(data_array * factor, 0, 1e9)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(array_i, origin='lower', cmap='hot')
        ax.set_title(f"{title} Frame {i+1}")
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

def create_risk_gif(risk_map, rain_intensity, frames=10, fps=2):
    images = []
    for i in range(frames):
        rain_factor = rain_intensity * (i + 1) / frames
        dynamic_risk = risk_map * (1 + rain_factor / 100)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(dynamic_risk, origin='lower', cmap='inferno')
        ax.set_title(f"Risk Frame {i+1} (Rain: {rain_factor:.1f} mm/hr)")
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

def export_geotiff(array, transform, crs="EPSG:4326"):
    memfile = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile_obj:
        with memfile_obj.open(
            driver="GTiff", height=array.shape[0], width=array.shape[1],
            count=1, dtype="float32", crs=crs, transform=transform
        ) as dataset:
            dataset.write(array.astype("float32"), 1)
        memfile_obj.seek(0)
        memfile.write(memfile_obj.read())
    return memfile.getvalue()

# -----------------------------------------------------------------------------
# 8. Επεξεργασία και Υπολογισμοί
# -----------------------------------------------------------------------------
if uploaded_stl:
    grid_z = process_dem(uploaded_stl, global_scale, global_offset, global_grid_res)
    if grid_z is not None:
        # Υπολογισμός μετατροπής γεωγραφικών μονάδων σε μέτρα
        dx = (right_bound - left_bound) / (global_grid_res - 1)
        dy = (top_bound - bottom_bound) / (global_grid_res - 1)
        avg_lat = (top_bound + bottom_bound) / 2.0
        meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
        meters_per_deg_lat = 111320
        dx_meters = dx * meters_per_deg_lon
        dy_meters = dy * meters_per_deg_lat

        # Υπολογισμοί κλίσης, προσανατολισμού, ροής κλπ
        slope, aspect = compute_slope_aspect(grid_z, dx_meters, dy_meters)
        flow_acc = compute_flow_accumulation(grid_z)
        slope_radians = np.radians(slope)
        cell_area = dx_meters * dy_meters
        twi = np.log((flow_acc * cell_area) / (np.tan(slope_radians) + 1e-9))
        d2z_dx2 = np.gradient(np.gradient(grid_z, dx_meters, axis=1), dx_meters, axis=1)
        d2z_dy2 = np.gradient(np.gradient(grid_z, dy_meters, axis=0), dy_meters, axis=0)
        curvature = d2z_dx2 + d2z_dy2

        transform = from_origin(left_bound, top_bound, dx, dy)

        # Υδρογράφημα ροής
        area_m2 = catchment_area * 10000.0
        total_rain_m = (rainfall_intensity / 1000.0) * event_duration
        V_runoff = total_rain_m * area_m2 * runoff_coeff
        Q_peak = V_runoff / event_duration
        t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
        Q = np.where(t <= event_duration, Q_peak * (t / event_duration), 
                     Q_peak * np.exp(-recession_rate * (t - event_duration)))
        retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None
        nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

        # Επεξεργασία καμένων περιοχών
        burned_mask = None
        if uploaded_burned:
            burned_mask = process_burned_data(uploaded_burned, grid_z.shape, transform)

        # Υπολογισμός χάρτη κινδύνου
        risk_map = None
        high_slope_risk = None
        if burned_mask is not None:
            norm_slope = np.where(slope > slope_threshold, (slope / slope.max())**2, slope / slope.max())
            norm_dem = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min() + 1e-9)
            norm_rain = (rainfall_intensity * event_duration) / 100.0
            norm_flow_acc = (flow_acc - flow_acc.min()) / (flow_acc.max() - flow_acc.min() + 1e-9)
            norm_twi = (twi - twi.min()) / (twi.max() - twi.min() + 1e-9)
            risk_map = (risk_slope_weight * norm_slope +
                        risk_dem_weight * norm_dem +
                        risk_burned_weight * burned_mask +
                        risk_rain_weight * norm_rain +
                        0.5 * norm_flow_acc +
                        0.5 * norm_twi)
            risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-9)
            high_slope_burned_mask = (slope > slope_threshold) & (burned_mask > 0.5)
            high_slope_risk = risk_map * high_slope_burned_mask

        # -----------------------------------------------------------------------------
        # 9. Εμφάνιση Αποτελεσμάτων
        # -----------------------------------------------------------------------------
        tabs = st.tabs([
            "DEM", "Slope", "Aspect", "Flow", "Retention", "Nutrient", 
            "Burned", "Risk", "High Slope Risk", "Flow Acc", "TWI", "Curvature", "GIFs"
        ])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='hot', vmin=global_dem_min, vmax=global_dem_max)
            ax.set_title("DEM")
            fig.colorbar(im, label="Ύψος (m)")
            st.pyplot(fig)

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='viridis')
            ax.set_title("Κλίση")
            fig.colorbar(im, label="Κλίση (°)")
            st.pyplot(fig)

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='twilight')
            ax.set_title("Προσανατολισμός")
            fig.colorbar(im, label="Προσανατολισμός (°)")
            st.pyplot(fig)

        with tabs[3]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, Q, 'b-')
            ax.set_title("Υδρογράφημα Ροής")
            ax.set_xlabel("Χρόνος (hr)")
            ax.set_ylabel("Ροή (m³/hr)")
            st.pyplot(fig)
            st.write(f"Μέγιστη Ροή: {Q_peak:.2f} m³/hr")

        with tabs[4]:
            if retention_time:
                st.write(f"Χρόνος Διατήρησης: {retention_time:.2f} hr")
            else:
                st.warning("Ανεπαρκή δεδομένα για χρόνο διατήρησης.")

        with tabs[5]:
            st.write(f"Θρεπτικό Φορτίο: {nutrient_load:.2f} kg")

        with tabs[6]:
            if burned_mask is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(burned_mask, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                               origin='lower', cmap='gray')
                ax.set_title("Καμένες Περιοχές")
                fig.colorbar(im, label="Καμένα (1) / Μη Καμένα (0)")
                st.pyplot(fig)
                st.write(f"Ποσοστό Καμένων: {100.0 * np.sum(burned_mask) / burned_mask.size:.2f}%")
            else:
                st.info("Δεν υπάρχουν δεδομένα καμένων περιοχών.")

        with tabs[7]:
            if risk_map is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(risk_map, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                               origin='lower', cmap='inferno')
                ax.set_title("Χάρτης Κινδύνου")
                fig.colorbar(im, label="Κίνδυνος (0-1)")
                st.pyplot(fig)
            else:
                st.info("Ο χάρτης κινδύνου δεν είναι διαθέσιμος.")

        with tabs[8]:
            if high_slope_risk is not None and np.any(high_slope_risk):
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(high_slope_risk, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                               origin='lower', cmap='inferno')
                ax.set_title(f"Κίνδυνος (Κλίση > {slope_threshold}° & Καμένες)")
                fig.colorbar(im, label="Κίνδυνος (0-1)")
                st.pyplot(fig)
                high_slope_risk_mean = np.mean(risk_map[high_slope_burned_mask])
                high_slope_area = np.sum(high_slope_burned_mask) * cell_area / 10000
                st.write(f"Μέσος Κίνδυνος: {high_slope_risk_mean:.2f}")
                st.write(f"Έκταση: {high_slope_area:.2f} ha")
            else:
                st.info("Δεν υπάρχουν δεδομένα για απότομες καμένες περιοχές.")

        with tabs[9]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(flow_acc, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='viridis')
            ax.set_title("Συγκέντρωση Ροής")
            fig.colorbar(im, label="Ροή")
            st.pyplot(fig)

        with tabs[10]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(twi, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='coolwarm')
            ax.set_title("TWI")
            fig.colorbar(im, label="TWI")
            st.pyplot(fig)

        with tabs[11]:
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(curvature, extent=(left_bound, right_bound, bottom_bound, top_bound), 
                           origin='lower', cmap='Spectral')
            ax.set_title("Καμπυλότητα")
            fig.colorbar(im, label="Καμπυλότητα")
            st.pyplot(fig)

        with tabs[12]:
            st.image(create_gif(grid_z / (np.max(grid_z) + 1e-9), title="Flow"), caption="Flow GIF")
            if risk_map is not None and simulate_risk:
                dynamic_risk = risk_map * (1 + (sim_rain_intensity * sim_duration) / 100)
                st.image(create_risk_gif(dynamic_risk, sim_rain_intensity), caption="Dynamic Risk GIF")

        # Εξαγωγή GeoTIFF
        st.download_button("Λήψη DEM (GeoTIFF)", export_geotiff(grid_z, transform), "DEM.tif")
        if risk_map is not None:
            st.download_button("Λήψη Risk (GeoTIFF)", export_geotiff(risk_map, transform), "Risk.tif")
else:
    st.info("Ανεβάστε STL για να ξεκινήσει η ανάλυση.")
