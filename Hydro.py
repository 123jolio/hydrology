import os
import zipfile
import tempfile
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from affine import Affine
from pysheds.grid import Grid
import streamlit as st

# Page configuration
st.set_page_config(page_title="Hydrological Analysis", layout="wide")
st.title("Hydrological Analysis")
st.markdown("""
This application creates a DEM from contour lines contained in a KMZ file, performs hydrological processing (sink filling, flow direction, accumulation), extracts the stream network, and delineates the watershed draining to a lake.
""")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the expected KMZ file paths
contour_kmz_path = os.path.join(script_dir, "Model_25m.kmz")
lake_kmz_path = os.path.join(script_dir, "Lake Polygon.kmz")

# Check if both KMZ files exist
if not os.path.exists(contour_kmz_path):
    st.error(f"Contour KMZ file not found: {contour_kmz_path}")
    st.stop()
if not os.path.exists(lake_kmz_path):
    st.error(f"Lake KMZ file not found: {lake_kmz_path}")
    st.stop()

def load_gdf_from_kmz(kmz_path):
    """
    Extracts the first KML file found within the KMZ archive,
    reads it into a GeoDataFrame, and returns the GeoDataFrame.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            with zipfile.ZipFile(kmz_path, 'r') as kmz:
                kmz.extractall(tmpdirname)
        except Exception as e:
            st.error(f"Error extracting KMZ file {kmz_path}: {e}")
            st.stop()
        # Look for a file ending with .kml (case-insensitive)
        kml_files = [f for f in os.listdir(tmpdirname) if f.lower().endswith('.kml')]
        if not kml_files:
            st.error(f"No KML file found inside the KMZ: {kmz_path}")
            st.stop()
        kml_path = os.path.join(tmpdirname, kml_files[0])
        try:
            gdf = gpd.read_file(kml_path, driver='KML')
        except Exception as e:
            st.error(f"Error reading file {kml_path}: {e}")
            st.stop()
        return gdf

# Load the GeoDataFrames from the KMZ files
gdf_contours = load_gdf_from_kmz(contour_kmz_path)
gdf_lake = load_gdf_from_kmz(lake_kmz_path)

# Filter for contour lines in the contours dataset (LineString or MultiLineString)
contours = gdf_contours[gdf_contours.geometry.type.isin(['LineString', 'MultiLineString'])]
# Filter for lake polygons in the lake dataset (Polygon or MultiPolygon)
lakes = gdf_lake[gdf_lake.geometry.type.isin(['Polygon', 'MultiPolygon'])]

if lakes.empty:
    st.error("No lake polygon found in the lake KMZ.")
    st.stop()

# Extract contour points and elevation values
points = []
values = []
for idx, row in contours.iterrows():
    geom = row.geometry
    # Handle both MultiLineString and LineString geometries
    if geom.type == 'MultiLineString':
        lines = list(geom)
    else:
        lines = [geom]
    for line in lines:
        for coord in line.coords:
            # Prefer 3D coordinates; if not available, try to get elevation from an attribute
            if len(coord) >= 3:
                x, y, z = coord[:3]
            else:
                x, y = coord
                z = row.get('elevation', None)
                if z is None:
                    continue
            points.append((x, y))
            values.append(z)
points = np.array(points)
values = np.array(values)

if len(points) == 0:
    st.error("No elevation data found in the contours.")
    st.stop()

st.success("Contour data extracted successfully!")

# Sidebar parameters for grid resolution and flow accumulation threshold
grid_res = st.sidebar.number_input("Grid resolution (points per dimension)", min_value=100, max_value=1000, value=500, step=50)
threshold = st.sidebar.number_input("Flow accumulation threshold", min_value=10, max_value=1000, value=100, step=10)

# Create a grid for DEM interpolation based on contour extents
minx, miny = points.min(axis=0)
maxx, maxy = points.max(axis=0)
grid_x, grid_y = np.mgrid[minx:maxx:complex(grid_res), miny:maxy:complex(grid_res)]
grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

# Create an affine transformation for the DEM grid
dx = (maxx - minx) / grid_res
dy = (maxy - miny) / grid_res
affine_transform = Affine(dx, 0, minx, 0, -dy, maxy)

# Initialize a PySheds Grid and add the interpolated DEM
grid_obj = Grid()
grid_obj.add_gridded_data(grid_z, data_name='dem', affine=affine_transform, nodata=-9999)

# Fill depressions in the DEM
filled_dem = grid_obj.fill_depressions(data='dem', out_name='filled_dem')

# Compute flow direction using the D8 algorithm
grid_obj.flowdir(data='filled_dem', out_name='flowdir')

# Compute flow accumulation
grid_obj.accumulation(data='flowdir', out_name='acc')

# Extract streams by applying the threshold
acc_array = grid_obj.view('acc')
streams = np.where(acc_array > threshold, 1, 0)

# Use the lake polygon's centroid as the pour point for watershed delineation
lake = lakes.iloc[0]
centroid = lake.geometry.centroid
pour_point_x = centroid.x
pour_point_y = centroid.y

# Convert the pour point coordinates to grid indices using the inverse affine transform
col, row = ~affine_transform * (pour_point_x, pour_point_y)
col = int(col)
row = int(row)

# Delineate the watershed (catchment) draining to the pour point
catchment = grid_obj.catchment(data='flowdir', x=pour_point_x, y=pour_point_y,
                               dirmap='D8', recursionlimit=25000, out_name='watershed')

st.success("Hydrological analysis completed!")

# Create tabs for visualizing each result
tab1, tab2, tab3, tab4 = st.tabs(["Interpolated DEM", "Filled DEM", "Stream Network", "Watershed"])

with tab1:
    st.subheader("Interpolated DEM")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid_z, cmap='terrain')
    ax.set_title("Interpolated DEM")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Filled DEM")
    filled = grid_obj.view('filled_dem')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(filled, cmap='terrain')
    ax.set_title("Filled DEM")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Stream Network")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(streams, cmap='Blues')
    ax.set_title("Extracted Streams (Threshold = {})".format(threshold))
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

with tab4:
    st.subheader("Watershed")
    ws = grid_obj.view('watershed')
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ws, cmap='viridis')
    ax.set_title("Watershed Delineation")
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
