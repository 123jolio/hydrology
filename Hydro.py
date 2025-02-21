import os
import zipfile
import tempfile
import glob
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from affine import Affine
from pysheds.grid import Grid
import streamlit as st
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Page configuration
st.set_page_config(page_title="Hydrological Analysis", layout="wide")
st.title("Hydrological Analysis")
st.markdown("""
This application creates a DEM from contour lines contained in a KMZ file,
performs hydrological processing (sink filling, flow direction, accumulation),
extracts the stream network, and delineates the watershed draining to a lake.
""")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the expected KMZ file names
contour_kmz_path = os.path.join(script_dir, "Model_25m.kmz")
lake_kmz_path = os.path.join(script_dir, "Lake Polygon.kmz")

# Check if both KMZ files exist
if not os.path.exists(contour_kmz_path):
    st.error(f"Contour KMZ file not found: {contour_kmz_path}")
    st.stop()
if not os.path.exists(lake_kmz_path):
    st.error(f"Lake KMZ file not found: {lake_kmz_path}")
    st.stop()

def extract_kml_from_kmz(kmz_path):
    """
    Extracts the first KML file found within the KMZ archive to a temporary directory
    (which is not automatically deleted) and returns its path.
    """
    tmpdirname = tempfile.mkdtemp()  # persistent temporary directory
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
    return os.path.join(tmpdirname, kml_files[0])

# Extract the KML file paths from each KMZ
contour_kml_path = extract_kml_from_kmz(contour_kmz_path)
lake_kml_path = extract_kml_from_kmz(lake_kmz_path)

# Read the extracted KML files using GeoPandas
try:
    gdf_contours = gpd.read_file(contour_kml_path, driver='KML')
except Exception as e:
    st.error(f"Error reading contours from KMZ: {e}")
    st.stop()
try:
    gdf_lake = gpd.read_file(lake_kml_path, driver='KML')
except Exception as e:
    st.error(f"Error reading lake polygon from KMZ: {e}")
    st.stop()

# Debug: Show available geometry types in the lake file
st.write("Lake file geometry types:", gdf_lake.geometry.geom_type.unique())

# Filter for contour lines in the contours dataset
contours = gdf_contours[gdf_contours.geometry.type.isin(['LineString', 'MultiLineString'])]

# First, try filtering for polygons in the lake dataset
lakes = gdf_lake[gdf_lake.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# If no polygons are found, attempt to convert closed lines into polygons
if lakes.empty:
    st.warning("No lake polygon found. Attempting to convert closed lines to polygons.")
    
    def is_closed_line(line):
        coords = list(line.coords)
        return coords[0] == coords[-1]

    def linestring_to_polygon(linestring):
        if is_closed_line(linestring):
            return Polygon(linestring.coords)
        return linestring

    def multilinestring_to_polygon(multilinestring):
        polygons = []
        for line in multilinestring.geoms:
            if is_closed_line(line):
                polygons.append(Polygon(line.coords))
        if polygons:
            if len(polygons) == 1:
                return polygons[0]
            else:
                return unary_union(polygons)
        return multilinestring

    def convert_geometry(geom):
        if geom.type == 'LineString':
            return linestring_to_polygon(geom)
        elif geom.type == 'MultiLineString':
            return multilinestring_to_polygon(geom)
        return geom

    gdf_lake['geometry'] = gdf_lake['geometry'].apply(convert_geometry)
    lakes = gdf_lake[gdf_lake.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# If still no polygon, try buffering any line geometries to force a polygon
if lakes.empty:
    st.warning("No lake polygon found after conversion. Attempting to buffer line geometries.")
    
    def buffer_to_polygon(geom, buffer_distance=0.0001):
        if geom.type in ['LineString', 'MultiLineString']:
            return geom.buffer(buffer_distance)
        return geom

    gdf_lake['geometry'] = gdf_lake['geometry'].apply(buffer_to_polygon)
    lakes = gdf_lake[gdf_lake.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
if lakes.empty:
    st.error("No lake polygon found in the lake KMZ even after conversion and buffering attempts.")
    st.stop()

# Extract contour points and elevations from the contours dataset
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
            # If the coordinate has a z value, use it
            if len(coord) >= 3:
                x, y, z = coord[:3]
            else:
                x, y = coord
                # Try to get elevation from an attribute 'elevation'
                z = row.get('elevation', None)
                # Fallback: try to parse the 'Name' field as a number
                if z is None and 'Name' in row and row['Name']:
                    try:
                        z = float(row['Name'])
                    except ValueError:
                        z = None
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
    ax.set_title(f"Extracted Streams (Threshold = {threshold})")
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
