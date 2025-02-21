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
import imageio  # for GIF creation
import matplotlib.contour as contour  # for boundary extraction
import zipfile
from fastkml import kml
import os
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Πρέπει να εκτελεστεί ως πρώτη εντολή στο Streamlit!
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Advanced Hydrogeology & DEM Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 2. Ενσωμάτωση ελαφριάς, επαγγελματικής CSS (σκοτεινή πλαϊνή μπάρα, χωρίς gradient)
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
div.stTabs > div {
    border: none;
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
    font-size: 1rem;
    padding: 0.5rem 1rem;
    border: none;
}
.css-1emrehy.edgvbvh3 { 
    background-color: #2e7bcf !important; 
    color: white !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Επικεφαλίδα (λογότυπο + τίτλος)
# -----------------------------------------------------------------------------
try:
    st.image("logo.png", width=200)
except Exception:
    pass

st.title("Advanced Hydrogeology & DEM Analysis (with Scenario GIFs)")
st.markdown("""
Αυτή η εφαρμογή δημιουργεί ένα DEM από ένα αρχείο STL και υπολογίζει εξελιγμένους τοπογραφικούς χάρτες (κλίση, προσανατολισμό), 
προσομοιώνει ένα υδρογράφημα ροής, εκτιμά το χρόνο διατήρησης, τη διαρροή θρεπτικών στοιχείων και τον κίνδυνο καμένων περιοχών.  
Για τις αναλύσεις καμένων περιοχών μπορείτε να ανεβάσετε είτε ένα αρχείο KMZ (με διανυσματικά πολύγωνα),  
είτε ένα γεωαναφερθέν TIFF, είτε μια εικόνα JPG/PNG (π.χ. με λευκό-κόκκινο χρωματικό διάγραμμα όπου οι καμένες περιοχές επισημαίνονται).  
Επιπλέον, υπολογίζονται και παράγωγα τοπογραφικά (συγκέντρωση ροής, δείκτης υγρασίας τοπογραφίας (TWI), ανάλυση καμπυλότητας).  
Όλα τα αποτελέσματα μπορούν να εξαχθούν ως γεωαναφερθέντα αρχεία GeoTIFF.
""")

# -----------------------------------------------------------------------------
# 4. Ορισμός ορίου γεωαναφοράς (EPSG:4326)
# -----------------------------------------------------------------------------
left_bound = 27.906069
top_bound = 36.92337189
right_bound = 28.045764
bottom_bound = 36.133509

# -----------------------------------------------------------------------------
# 5. Ανέβασμα αρχείων: STL και δεδομένων καμένων περιοχών
# -----------------------------------------------------------------------------
uploaded_stl = st.file_uploader("Ανεβάστε αρχείο STL (για DEM)", type=["stl"])
uploaded_burned = st.file_uploader("Ανεβάστε δεδομένα καμένων περιοχών (KMZ, TIFF, JPG ή PNG)", 
                                   type=["kmz", "tif", "tiff", "jpg", "png"])

# -----------------------------------------------------------------------------
# 6. Παράμετροι από το sidebar (για DEM, ροή, διατήρηση, θρεπτικά στοιχεία)
# -----------------------------------------------------------------------------
st.sidebar.header("Global DEM & Elevation")
global_scale = st.sidebar.slider("Κλίμακα Ύψους", 0.1, 5.0, 1.0, 0.1)
global_offset = st.sidebar.slider("Μετατόπιση Ύψους (m)", -100.0, 100.0, 0.0, 1.0)
global_dem_min = st.sidebar.number_input("Ελάχιστο Ύψος (m)", value=0.0, step=1.0)
global_dem_max = st.sidebar.number_input("Μέγιστο Ύψος (m)", value=500.0, step=1.0)
global_grid_res = st.sidebar.number_input("Ανάλυση Πλέγματος", 100, 1000, 500, 50)

st.sidebar.header("Global Flow & Retention")
rainfall_intensity = st.sidebar.number_input("Βροχόπτωση (mm/hr)", value=30.0, step=1.0)
event_duration = st.sidebar.number_input("Διάρκεια Βροχόπτωσης (hr)", value=2.0, step=0.1)
catchment_area = st.sidebar.number_input("Έκταση λεκάνης (ha)", value=10.0, step=0.1)
runoff_coeff = st.sidebar.slider("Συντελεστής Ροής", 0.0, 1.0, 0.5, 0.05)
recession_rate = st.sidebar.number_input("Ρυθμός Υποχώρησης (1/hr)", value=0.5, step=0.1)
simulation_hours = st.sidebar.number_input("Διάρκεια Προσομοίωσης (hr)", value=6.0, step=0.5)
storage_volume = st.sidebar.number_input("Όγκος Αποθήκευσης (m³)", value=5000.0, step=100.0)

st.sidebar.header("Global Nutrient Leaching")
soil_nutrient = st.sidebar.number_input("Θρεπτικό Περιεχόμενο (kg/ha)", value=50.0, step=1.0)
veg_retention = st.sidebar.slider("Διατήρηση Βλάστησης", 0.0, 1.0, 0.7, 0.05)
erosion_factor = st.sidebar.slider("Συντελεστής Διάβρωσης", 0.0, 1.0, 0.3, 0.05)

# -----------------------------------------------------------------------------
# Παράμετροι για Slope Map και Burned Risk
# -----------------------------------------------------------------------------
st.sidebar.header("Slope Map Customization")
slope_display_min = st.sidebar.number_input("Ελάχιστη Κλίση (°)", value=0.0, step=1.0)
slope_display_max = st.sidebar.number_input("Μέγιστη Κλίση (°)", value=70.0, step=1.0)

st.sidebar.header("Burned Risk Parameters")
risk_slope_weight = st.sidebar.slider("Βάρος Κλίσης", 0.0, 2.0, 1.0, 0.1)
risk_dem_weight = st.sidebar.slider("Βάρος Ύψους", 0.0, 2.0, 1.0, 0.1)
risk_burned_weight = st.sidebar.slider("Βάρος Καμένων Περιοχών", 0.0, 2.0, 1.0, 0.1)
risk_rain_weight = st.sidebar.slider("Βάρος Βροχόπτωσης", 0.0, 2.0, 1.0, 0.1)

# -----------------------------------------------------------------------------
# 7. Επεξεργασία STL και υπολογισμός DEM και σχετικών χαρτών
# -----------------------------------------------------------------------------
risk_map = None       # Ο χάρτης κινδύνου (risk) θα υπολογιστεί αν υπάρχουν δεδομένα καμένων περιοχών
burned_mask = None    # Η μάσκα καμένων περιοχών (από TIFF, KMZ ή εικόνα)
burned_polygons = []  # Για αποθήκευση διανυσματικών γεωμετριών (από KMZ)
burned_img_raw = None # Για αποθήκευση των ακατέργαστων τιμών του TIFF (αν υπάρχουν)

if uploaded_stl is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_stl:
        tmp_stl.write(uploaded_stl.read())
        stl_filename = tmp_stl.name

    try:
        stl_mesh = mesh.Mesh.from_file(stl_filename)
    except Exception as e:
        st.error(f"Σφάλμα ανάγνωσης STL: {e}")
        st.stop()

    vertices = stl_mesh.vectors.reshape(-1, 3)
    x_raw = vertices[:, 0]
    y_raw = vertices[:, 1]
    z_raw = vertices[:, 2]
    z_adj = (z_raw * global_scale) + global_offset

    x_min, x_max = x_raw.min(), x_raw.max()
    y_min, y_max = y_raw.min(), y_raw.max()
    lon_raw = left_bound + (x_raw - x_min) * (right_bound - left_bound) / (x_max - x_min)
    lat_raw = top_bound - (y_raw - y_min) * (top_bound - bottom_bound) / (y_max - y_min)

    xi = np.linspace(left_bound, right_bound, global_grid_res)
    yi = np.linspace(top_bound, bottom_bound, global_grid_res)
    grid_x, grid_y = np.meshgrid(xi, yi)

    grid_z = griddata((lon_raw, lat_raw), z_adj, (grid_x, grid_y), method='cubic')
    grid_z = np.clip(grid_z, global_dem_min, global_dem_max)

    dx = (right_bound - left_bound) / (global_grid_res - 1)
    dy = (top_bound - bottom_bound) / (global_grid_res - 1)
    avg_lat = (top_bound + bottom_bound) / 2.0
    meters_per_deg_lon = 111320 * np.cos(np.radians(avg_lat))
    meters_per_deg_lat = 111320
    dx_meters = dx * meters_per_deg_lon
    dy_meters = dy * meters_per_deg_lat

    dz_dx, dz_dy = np.gradient(grid_z, dx_meters, dy_meters)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360

    transform = from_origin(left_bound, top_bound, dx, dy)

    area_m2 = catchment_area * 10000.0
    total_rain_m = (rainfall_intensity / 1000.0) * event_duration
    V_runoff = total_rain_m * area_m2 * runoff_coeff
    Q_peak = V_runoff / event_duration

    t = np.linspace(0, simulation_hours, int(simulation_hours * 60))
    Q = np.zeros_like(t)
    for i, time in enumerate(t):
        if time <= event_duration:
            Q[i] = Q_peak * (time / event_duration)
        else:
            Q[i] = Q_peak * np.exp(-recession_rate * (time - event_duration))

    retention_time = storage_volume / (V_runoff / event_duration) if V_runoff > 0 else None
    nutrient_load = soil_nutrient * (1 - veg_retention) * erosion_factor * catchment_area

    # -----------------------------------------------------------------------------
    # 8. Επεξεργασία δεδομένων καμένων περιοχών (ανεβάζονται TIFF, KMZ ή εικόνα)
    # -----------------------------------------------------------------------------
    if uploaded_burned is not None:
        ext = os.path.splitext(uploaded_burned.name)[1].lower()
        if ext == ".kmz":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz") as tmp_kmz:
                tmp_kmz.write(uploaded_burned.read())
                kmz_filename = tmp_kmz.name
            try:
                with zipfile.ZipFile(kmz_filename, 'r') as zf:
                    kml_filename = [name for name in zf.namelist() if name.endswith('.kml')][0]
                    kml_data = zf.read(kml_filename)
            except Exception as e:
                st.warning(f"Σφάλμα ανάγνωσης KMZ: {e}")
                kml_data = None
            if kml_data is not None:
                try:
                    k_obj = kml.KML()
                    k_obj.from_string(kml_data)
                    for feature in k_obj.features:
                        for placemark in feature.features:
                            if hasattr(placemark, 'geometry') and placemark.geometry is not None:
                                geom = placemark.geometry
                                if geom.geom_type == "Polygon":
                                    burned_polygons.append(geom)
                                elif geom.geom_type == "MultiPolygon":
                                    burned_polygons.extend(list(geom.geoms))
                    if not burned_polygons:
                        st.warning("Δεν βρέθηκαν πολύγωνα στο αρχείο KMZ.")
                except Exception as e:
                    st.warning(f"Σφάλμα ανάλυσης KML: {e}")
                    burned_polygons = []
            if burned_polygons:
                shapes = [(poly, 1) for poly in burned_polygons]
                burned_mask = rasterize(
                    shapes,
                    out_shape=grid_z.shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
            else:
                burned_mask = None

        elif ext in [".tif", ".tiff"]:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_tif:
                    tmp_tif.write(uploaded_burned.read())
                    tif_filename = tmp_tif.name
                with rasterio.open(tif_filename) as src:
                    src_crs = src.crs if src.crs is not None else "EPSG:4326"
                    burned_img_raw = src.read(1)
                    st.write("Ακατέργαστες τιμές TIFF: min", burned_img_raw.min(), "max", burned_img_raw.max())
                    src_transform = src.transform
                    # Κανονικοποίηση των τιμών σε [0, 1]
                    burned_img_norm = (burned_img_raw - burned_img_raw.min()) / (burned_img_raw.max() - burned_img_raw.min() + 1e-9)
                    burned_mask_temp = burned_img_norm
                    burned_mask_resampled = np.empty(grid_z.shape, dtype=np.float32)
                    reproject(
                        source=burned_mask_temp,
                        destination=burned_mask_resampled,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.nearest
                    )
                    burned_mask = burned_mask_resampled
            except Exception as e:
                st.warning(f"Σφάλμα ανάγνωσης TIFF: {e}")
                burned_mask = None

        elif ext in [".jpg", ".jpeg", ".png"]:
            try:
                burned_img = imageio.imread(uploaded_burned)
                if burned_img.ndim == 3:
                    burned_mask = ((burned_img[..., 0] > 150) & 
                                   (burned_img[..., 1] < 100) & 
                                   (burned_img[..., 2] < 100)).astype(np.uint8)
                else:
                    burned_mask = (burned_img > 128).astype(np.uint8)
                burned_mask = np.array(Image.fromarray(burned_mask).resize((grid_z.shape[1], grid_z.shape[0]), resample=Image.NEAREST))
            except Exception as e:
                st.warning(f"Σφάλμα ανάγνωσης JPG/PNG: {e}")
                burned_mask = None
        else:
            st.warning("Μη υποστηριζόμενη μορφή αρχείου για δεδομένα καμένων περιοχών.")
            burned_mask = None

    # -----------------------------------------------------------------------------
    # 9. Υπολογισμός Χάρτη Κινδύνου Καμένων Περιοχών
    # -----------------------------------------------------------------------------
    if burned_mask is not None:
        # Κανονικοποίηση των τοπογραφικών παραμέτρων
        norm_slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-9)
        norm_dem   = (grid_z - grid_z.min()) / (grid_z.max() - grid_z.min() + 1e-9)
        norm_rain = (rainfall_intensity * event_duration) / 100.0
        # Ο σταθμισμένος συνδυασμός (μπορείτε να προσαρμόσετε τους συντελεστές)
        risk_map = (risk_slope_weight * norm_slope +
                    risk_dem_weight   * norm_dem +
                    risk_burned_weight * burned_mask +
                    risk_rain_weight  * norm_rain)
        risk_map = (risk_map - risk_map.min()) / (risk_map.max() - risk_map.min() + 1e-9)

    # -----------------------------------------------------------------------------
    # 10. Πρόσθετα Τοπογραφικά Παράγωγα: Συγκέντρωση Ροής, TWI και Καμπυλότητα
    # -----------------------------------------------------------------------------
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
            if best_neighbor is not None:
                rr, cc = best_neighbor
                acc[rr, cc] += acc[r, c]
        return acc

    flow_acc = compute_flow_accumulation(grid_z)
    slope_radians = np.radians(slope)
    cell_area = dx_meters * dy_meters
    twi = np.log((flow_acc * cell_area) / (np.tan(slope_radians) + 1e-9))
    d2z_dx2 = np.gradient(dz_dx, dx_meters, axis=1)
    d2z_dy2 = np.gradient(dz_dy, dy_meters, axis=0)
    curvature = d2z_dx2 + d2z_dy2

    # -----------------------------------------------------------------------------
    # 11. Βοηθητική συνάρτηση: Δημιουργία GIF
    # -----------------------------------------------------------------------------
    def create_placeholder_gif(data_array, frames=10, fps=2, scenario_name="flow"):
        images = []
        for i in range(frames):
            factor = 1 + 0.1 * i
            array_i = np.clip(data_array * factor, 0, 1e9)
            fig, ax = plt.subplots(figsize=(4,4))
            im = ax.imshow(array_i, origin='lower', cmap='hot')
            ax.set_title(f"{scenario_name.capitalize()} Frame {i+1}")
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

    flow_placeholder = np.clip(grid_z / (np.max(grid_z) + 1e-9), 0, 1)
    retention_placeholder = np.clip(slope / (np.max(slope) + 1e-9), 0, 1)
    nutrient_placeholder = np.clip(aspect / 360.0, 0, 1)
    risk_gif = None
    if risk_map is not None:
        risk_placeholder = np.clip(risk_map, 0, 1)
        risk_gif = create_placeholder_gif(risk_placeholder, scenario_name="risk")

    # -----------------------------------------------------------------------------
    # 12. Εμφάνιση αποτελεσμάτων σε καρτέλες (tabs)
    # -----------------------------------------------------------------------------
    tabs = st.tabs([
        "DEM Heatmap", "Slope Map", "Aspect Map",
        "Flow Simulation", "Retention Time", "GeoTIFF Export",
        "Nutrient Leaching", "Burned Area Analysis", "Burned Risk",
        "Flow Accumulation", "Topographic Wetness Index", "Curvature Analysis",
        "Scenario GIFs", "Raw Burned TIFF"
    ])

    with tabs[0]:
        with st.expander("DEM Heatmap Parameters", expanded=False):
            dem_vmin = st.number_input("Ελάχιστη κλίμακα DEM (m)", value=global_dem_min, step=1.0, key="dem_min_tab")
            dem_vmax = st.number_input("Μέγιστη κλίμακα DEM (m)", value=global_dem_max, step=1.0, key="dem_max_tab")
        st.subheader("Χάρτης DEM")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(grid_z, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='hot', vmin=dem_vmin, vmax=dem_vmax, aspect='auto')
        ax.set_title("DEM (Ρυθμισμένο Ύψος)")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="Ύψος (m)")
        st.pyplot(fig)

    with tabs[1]:
        with st.expander("Ρυθμίσεις Χάρτη Κλίσης", expanded=False):
            slope_cmap = st.selectbox("Επιλογή χρωματικής παλέτας κλίσης", ["viridis", "plasma", "inferno", "magma"], key="slope_cmap")
        st.subheader("Χάρτης Κλίσης")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(slope, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=slope_cmap,
                       vmin=slope_display_min, vmax=slope_display_max, aspect='auto')
        ax.set_title("Κλίση (μοίρες)")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="Κλίση (°)")
        st.pyplot(fig)

    with tabs[2]:
        with st.expander("Ρυθμίσεις Χάρτη Προσανατολισμού", expanded=False):
            aspect_cmap = st.selectbox("Επιλογή χρωματικής παλέτας προσανατολισμού", ["twilight", "hsv", "cool"], key="aspect_cmap")
        st.subheader("Χάρτης Προσανατολισμού")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(aspect, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap=aspect_cmap, aspect='auto')
        ax.set_title("Προσανατολισμός (μοίρες)")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="Προσανατολισμός (°)")
        st.pyplot(fig)

    with tabs[3]:
        with st.expander("Ρυθμίσεις Προσομοίωσης Ροής", expanded=False):
            flow_fps = st.number_input("FPS για GIF Ροής", value=2, step=1, key="flow_fps")
            flow_frames = st.number_input("Αριθμός καρέ GIF Ροής", value=10, step=1, key="flow_frames")
        st.subheader("Υδρογράφημα Ροής")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(t, Q, 'b-')
        ax.set_title("Υδρογράφημα Ροής")
        ax.set_xlabel("Χρόνος (hr)")
        ax.set_ylabel("Ροή (m³/hr)")
        st.pyplot(fig)
        st.write(f"Μέγιστη Ροή: {Q_peak:.2f} m³/hr")
        st.write(f"Συνολικός Όγκος Ροής: {V_runoff:.2f} m³")
        st.markdown("**GIF Σενάριο Ροής:**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(flow_frames), fps=int(flow_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")

    with tabs[4]:
        with st.expander("Ρυθμίσεις Χρόνου Διατήρησης", expanded=False):
            st.write("Ο χρόνος διατήρησης υπολογίζεται από την αποτελεσματική ροή και τον όγκο αποθήκευσης.")
        st.subheader("Χρόνος Διατήρησης")
        if retention_time is not None:
            st.write(f"Εκτιμώμενος Χρόνος Διατήρησης: {retention_time:.2f} hr")
        else:
            st.warning("Δεν υπάρχουν επαρκή δεδομένα ροής για υπολογισμό χρόνου διατήρησης.")

    with tabs[5]:
        st.subheader("Εξαγωγή GeoTIFF")
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
        st.download_button("Λήψη DEM (GeoTIFF)", dem_tiff, "DEM.tif", "image/tiff")
        st.download_button("Λήψη Χάρτη Κλίσης (GeoTIFF)", slope_tiff, "Slope.tif", "image/tiff")
        st.download_button("Λήψη Χάρτη Προσανατολισμού (GeoTIFF)", aspect_tiff, "Aspect.tif", "image/tiff")
        if burned_mask is not None:
            burned_tiff = export_geotiff(burned_mask.astype("float32"), transform)
            st.download_button("Λήψη Μάσκας Καμένων (GeoTIFF)", burned_tiff, "BurnedArea.tif", "image/tiff")
        if risk_map is not None:
            risk_tiff = export_geotiff(risk_map, transform)
            st.download_button("Λήψη Χάρτη Κινδύνου (GeoTIFF)", risk_tiff, "RiskMap.tif", "image/tiff")

    with tabs[6]:
        with st.expander("Ρυθμίσεις Θρεπτικής Διάχυσης", expanded=False):
            nutrient_scale = st.number_input("Συντελεστής Θρεπτικής Διάχυσης", value=1.0, step=0.1, key="nutrient_scale")
        st.subheader("Θρεπτική Διάχυση")
        st.write(f"Θρεπτικό Περιεχόμενο Εδάφους: {soil_nutrient} kg/ha")
        st.write(f"Διατήρηση Βλάστησης: {veg_retention}")
        st.write(f"Συντελεστής Διάβρωσης: {erosion_factor}")
        st.write(f"Έκταση Λεκάνης: {catchment_area} ha")
        st.write(f"Εκτιμώμενο Θρεπτικό Φορτίο: {nutrient_load * nutrient_scale:.2f} kg")

    with tabs[7]:
        st.subheader("Ανάλυση Καμένων Περιοχών")
        if burned_mask is not None:
            percent_burned = 100.0 * np.sum(burned_mask) / burned_mask.size
            st.write(f"Ποσοστό Καμένων Περιοχών: {percent_burned:.2f}%")
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(burned_mask, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='gray', aspect='auto')
            ax.set_title("Μάσκα Καμένων Περιοχών")
            ax.set_xlabel("Γεωγραφικό Μήκος")
            ax.set_ylabel("Γεωγραφικό Πλάτος")
            fig.colorbar(im, ax=ax, label="Καμένα (1) / Μη Καμένα (0)")
            if burned_polygons:
                for poly in burned_polygons:
                    x, y = poly.exterior.coords.xy
                    ax.plot(x, y, color='red', linewidth=2)
            st.pyplot(fig)
            st.info("Οι κόκκινες γραμμές δείχνουν τα όρια των καμένων περιοχών (εάν υπάρχουν).")
        else:
            st.info("Δεν υπάρχουν διαθέσιμα δεδομένα καμένων περιοχών (burned_mask είναι None).")

    with tabs[8]:
        st.subheader("Ενισχυμένος Χάρτης Κινδύνου Καμένων Περιοχών")
        if risk_map is not None:
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(risk_map, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='lower', cmap='inferno', aspect='auto')
            ax.set_title("Χάρτης Κινδύνου (Weighted Factors)")
            ax.set_xlabel("Γεωγραφικό Μήκος")
            ax.set_ylabel("Γεωγραφικό Πλάτος")
            fig.colorbar(im, ax=ax, label="Κίνδυνος (0-1)")
            st.pyplot(fig)
        else:
            st.info("Ο χάρτης κινδύνου καμένων δεν είναι διαθέσιμος.")

    with tabs[9]:
        st.subheader("Χάρτης Συγκέντρωσης Ροής")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(flow_acc, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='viridis', aspect='auto')
        ax.set_title("Flow Accumulation (D8)")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="Συγκεντρωμένη Ροή")
        st.pyplot(fig)

    with tabs[10]:
        st.subheader("Δείκτης Υγρασίας Τοπογραφίας (TWI)")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(twi, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='coolwarm', aspect='auto')
        ax.set_title("TWI")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="TWI")
        st.pyplot(fig)

    with tabs[11]:
        st.subheader("Ανάλυση Καμπυλότητας")
        fig, ax = plt.subplots(figsize=(6,4))
        im = ax.imshow(curvature, extent=(left_bound, right_bound, bottom_bound, top_bound),
                       origin='lower', cmap='Spectral', aspect='auto')
        ax.set_title("Καμπυλότητα (Laplacian)")
        ax.set_xlabel("Γεωγραφικό Μήκος")
        ax.set_ylabel("Γεωγραφικό Πλάτος")
        fig.colorbar(im, ax=ax, label="Καμπυλότητα")
        st.pyplot(fig)

    with tabs[12]:
        with st.expander("Ρυθμίσεις GIF Σεναρίου", expanded=False):
            gif_frames = st.number_input("Αριθμός καρέ GIF", value=10, step=1, key="gif_frames")
            gif_fps = st.number_input("FPS για GIF", value=2, step=1, key="gif_fps")
        st.subheader("Κινούμενα Σχέδια σε Σεναριακό Πλαίσιο")
        st.markdown("**Flow Scenario**")
        st.image(create_placeholder_gif(flow_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="flow"), caption="Flow Scenario (Demo)")
        st.markdown("**Retention Scenario**")
        st.image(create_placeholder_gif(retention_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="retention"), caption="Retention Scenario (Demo)")
        st.markdown("**Nutrient Scenario**")
        st.image(create_placeholder_gif(nutrient_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="nutrient"), caption="Nutrient Scenario (Demo)")
        if risk_gif is not None:
            st.markdown("**Risk Scenario**")
            st.image(create_placeholder_gif(risk_placeholder, frames=int(gif_frames), fps=int(gif_fps), scenario_name="risk"), caption="Risk Scenario (Demo)")

    with tabs[13]:
        st.subheader("Ακατέργαστα Δεδομένα Burned TIFF")
        if burned_img_raw is not None:
            # Κανονικοποίηση για απεικόνιση
            raw_vis = (burned_img_raw - burned_img_raw.min()) / (burned_img_raw.max() - burned_img_raw.min() + 1e-9)
            fig, ax = plt.subplots(figsize=(6,4))
            im = ax.imshow(raw_vis, extent=(left_bound, right_bound, bottom_bound, top_bound),
                           origin='upper', cmap='gray', aspect='auto')
            ax.set_title("Raw Burned TIFF")
            ax.set_xlabel("Γεωγραφικό Μήκος")
            ax.set_ylabel("Γεωγραφικό Πλάτος")
            fig.colorbar(im, ax=ax, label="Κανονικοποιημένη Τιμή")
            st.pyplot(fig)
        else:
            st.info("Δεν υπάρχουν ακατέργαστα δεδομένα Burned TIFF.")
else:
    st.info("Παρακαλώ ανεβάστε αρχείο STL για δημιουργία DEM και ανάλυση.")
