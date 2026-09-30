import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import requests
import gdown
import zipfile
import os
import tempfile
import io
import re
from fpdf import FPDF
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- 1. DATA DOWNLOADER ---
@st.cache_resource
def download_and_extract_data():
    data_dir = "pmp_data"
    
    if not os.path.exists(data_dir):
        st.info("First-time setup: Downloading data from Google Drive (this takes a minute)...")
        file_id = '1r9mZGQcSGZ_iCdrV3tAbDdOY_Hueej5t'
        output_zip = 'pmp_data.zip'
        
        gdown.download(id=file_id, output=output_zip, quiet=False, fuzzy=True)
        
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        os.remove(output_zip)
    return True

download_and_extract_data()

# --- 2. DYNAMIC CONFIGURATION PATHS ---
DATA_ROOT = Path("pmp_data")

def find_file(filename):
    for p in DATA_ROOT.rglob('*'):
        if "__MACOSX" not in p.parts and p.is_file() and p.name.lower() == filename.lower():
            return str(p)
    return ""

def find_dir_starts_with(dirname):
    for p in DATA_ROOT.rglob('*'):
        if "__MACOSX" not in p.parts and p.is_dir() and p.name.lower().startswith(dirname.lower()):
            return str(p)
    return ""

def find_gdb():
    for p in DATA_ROOT.rglob('*'):
        if "__MACOSX" not in p.parts and p.is_file() and p.suffix.lower() in ['.gdbtable', '.gdbtablx']:
            return str(p.parent)
    for p in DATA_ROOT.rglob('*.gdb'):
        if "__MACOSX" not in p.parts and p.is_dir():
            if any(f.is_file() for f in p.iterdir()):
                return str(p)
    return ""

LOCAL_GEOFABRIC_DB = find_gdb()
MASTER_PMP_ZONES_SHP = find_file("zones_all.shp")
GSAM_CD_ROOT = find_dir_starts_with("gsam_cd")
GSAM_MAF_GRID = find_file("epw_annual.asc")
GSAM_TOPO_GRID = find_file("gsam_taf.asc")
GSAM_STANDARD_EPW = 56.7

GTSMR_CD_ROOT = find_dir_starts_with("gtsmr_cd")
GTSMR_MAF_GRID = find_file("yearly005pwm.asc")
GTSMR_TOPO_GRID = None  
GTSMR_STANDARD_EPW = 73.0

GSDM_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
GSDM_AREAS = [1.0, 2.6, 16.0, 65.0, 153.0, 280.0, 433.0, 635.0, 847.0]
GSDM_SMOOTH_DATA = [
    [250, 360, 460, 570, 640, 710, 760, 810, 900, 960, 1000],
    [232, 336, 425, 493, 563, 628, 669, 705, 771, 832, 879],
    [204, 301, 383, 449, 513, 575, 612, 642, 711, 765, 811],
    [177, 260, 330, 397, 453, 511, 546, 576, 643, 695, 737],
    [157, 230, 292, 355, 404, 459, 493, 527, 591, 639, 679],
    [141, 207, 264, 321, 367, 418, 452, 490, 551, 594, 634],
    [129, 190, 243, 294, 340, 387, 422, 460, 520, 562, 599],
    [118, 174, 223, 269, 314, 357, 394, 434, 491, 531, 568],
    [108, 161, 208, 250, 293, 335, 373, 414, 468, 506, 544]
]
GSDM_ROUGH_DATA = [
    [250, 360, 460, 570, 740, 880, 990, 1090, 1250, 1360, 1450],
    [232, 336, 425, 493, 636, 744, 821, 901, 1030, 1135, 1200],
    [204, 301, 383, 449, 575, 672, 742, 810, 926, 1018, 1084],
    [177, 260, 330, 397, 511, 590, 663, 717, 811, 890, 950],
    [157, 230, 292, 355, 459, 527, 598, 647, 728, 794, 845],
    [141, 207, 264, 321, 418, 480, 546, 590, 669, 720, 767],
    [129, 190, 243, 294, 387, 446, 506, 548, 621, 664, 709],
    [118, 174, 223, 269, 357, 417, 469, 509, 578, 613, 656],
    [108, 161, 208, 250, 335, 395, 441, 477, 541, 578, 614]
]
gsdm_smooth_df = pd.DataFrame(GSDM_SMOOTH_DATA, index=GSDM_AREAS, columns=GSDM_DURATIONS)
gsdm_rough_df = pd.DataFrame(GSDM_ROUGH_DATA, index=GSDM_AREAS, columns=GSDM_DURATIONS)

# --- 3. HELPER & EXPORT FUNCTIONS ---
def sanitize_catchment(catchment):
    # UNIVERSAL FIX: Sanitize all columns to remove Timestamps for Folium maps and JSON exports
    for col in catchment.columns:
        if col != catchment.geometry.name:
            if pd.api.types.is_datetime64_any_dtype(catchment[col]):
                catchment[col] = catchment[col].astype(str)
            elif catchment[col].dtype == 'object':
                catchment[col] = catchment[col].apply(lambda x: str(x) if isinstance(x, pd.Timestamp) else x)
    return catchment

def fetch_catchment_from_geofabric(catchment_input, layer_type="AHGFCatchment"):
    try:
        catchment_id = int(str(catchment_input).strip(' "\''))
    except ValueError:
        raise ValueError("Input must be a valid numeric ID.")
        
    api_layer = "34" if layer_type == "NCBLevel2DrainageBasinGroup" else "7"
    base_url = f"https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/ahgf/Geofabric_V3x_All_Products/FeatureServer/{api_layer}/query"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Try HydroID first (standard for both layers)
    params = {'where': f"HydroID={catchment_id}", 'outFields': '*', 'returnGeometry': 'true', 'f': 'geojson'}
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If no result and looking for layer 7, fallback to SegmentNo
        if ("error" in data or "features" not in data or len(data["features"]) == 0) and layer_type == "AHGFCatchment":
            params['where'] = f"SegmentNo={catchment_id}"
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            data = response.json()
            
        if "error" in data or "features" not in data or len(data["features"]) == 0:
            raise ValueError(f"Could not find ID {catchment_id} in Geofabric API.")
            
        catchment = gpd.GeoDataFrame.from_features(data["features"])
        catchment.set_crs(epsg=4326, inplace=True) 
        
    except Exception:
        if not LOCAL_GEOFABRIC_DB or not Path(LOCAL_GEOFABRIC_DB).exists():
            raise FileNotFoundError("BoM API is blocked, and the offline database could not be found.")
            
        try:
            catchment = gpd.read_file(LOCAL_GEOFABRIC_DB, layer=layer_type, where=f"HydroID={catchment_id}", engine="pyogrio")
            if catchment.empty and layer_type == "AHGFCatchment":
                catchment = gpd.read_file(LOCAL_GEOFABRIC_DB, layer=layer_type, where=f"SegmentNo={catchment_id}", engine="pyogrio")
        except Exception:
            catchment = gpd.GeoDataFrame()
            
        if catchment.empty:
            raise ValueError(f"Could not find ID {catchment_id} in the local offline database ({layer_type}).")
             
        if catchment.crs is None:
            catchment.set_crs(epsg=4283, inplace=True)
        catchment = catchment.to_crs(epsg=4326)

    return sanitize_catchment(catchment)

def load_custom_catchment(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext == "geojson":
        catchment = gpd.read_file(uploaded_file)
    elif file_ext == "zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            extract_dir = os.path.join(tmpdir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Gather all valid shapefiles in the zip
            valid_shps = []
            for root, dirs, files in os.walk(extract_dir):
                if "__MACOSX" in root:
                    continue
                for file in files:
                    if file.lower().endswith('.shp') and not file.startswith('._') and not file.startswith('.'):
                        valid_shps.append(os.path.join(root, file))
                        
            if not valid_shps:
                raise ValueError("No valid .shp file found inside the uploaded zip archive.")
                
            # Iterate through shapefiles to find the one containing Polygons (ignoring Points/Lines)
            catchment = None
            for shp_path in valid_shps:
                temp_gdf = gpd.read_file(shp_path)
                if not temp_gdf.empty and temp_gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
                    catchment = temp_gdf
                    break
                    
            if catchment is None:
                raise ValueError("The uploaded shapefiles only contain Points or Lines. PMP calculations require Polygon areas.")
    else:
        raise ValueError("Unsupported file format. Please upload a .geojson or .zip (Shapefile).")
        
    # --- SAFETY CHECKS ---
    if catchment.empty:
        raise ValueError("The uploaded shapefile was found, but it contains no features (it is empty).")
        
    catchment = catchment[~catchment.geometry.is_empty & catchment.geometry.notnull()]
    if catchment.empty:
        raise ValueError("The uploaded shapefile contains no valid geographic polygons.")
        
    if catchment.crs is None:
        raise ValueError("Uploaded file has no Coordinate Reference System (CRS) defined. Ensure your zip includes the .prj file.")
        
    # FIX: Repair invalid geometries (self-intersections) to prevent TopologyExceptions
    catchment.geometry = catchment.geometry.make_valid()
        
    # If the user uploads a file with multiple smaller catchments, dissolve them into one total area
    if len(catchment) > 1:
        catchment['dissolve_field'] = 1
        catchment = catchment.dissolve(by='dissolve_field').reset_index(drop=True)
        
    catchment = catchment.to_crs(epsg=4326)
    return sanitize_catchment(catchment)

def find_dad_file_in_cd(cd_root_path, zone_name):
    cd_path = Path(cd_root_path)
    if not cd_path.exists():
        raise FileNotFoundError(f"CD directory not found: {cd_root_path}")
    zone_keyword = "coast" if "Coast" in zone_name else "inland"
    for file_path in cd_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.txt', '.xls', '.xlsx']:
            if zone_keyword in file_path.name.lower():
                return file_path
    raise FileNotFoundError(f"Could not find a '{zone_keyword}' table in {cd_root_path}")

def get_catchment_average(raster_path, geom):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geom], crop=True, filled=False)
        valid_data = out_image[~out_image.mask]
        if valid_data.size == 0:
            raise ValueError(f"Catchment does not overlap with valid data in {raster_path}.")
        return np.mean(valid_data)

def create_csv(pmp_dict):
    df = pd.DataFrame(list(pmp_dict.items()), columns=['Duration', 'Depth (mm)'])
    return df.to_csv(index=False).encode('utf-8')

def create_geojson(catchment_gdf, metadata_dict, pmp_dict):
    gdf = catchment_gdf.copy()
    for key, val in metadata_dict.items():
        gdf[key] = val
    for dur, depth in pmp_dict.items():
        col_name = f"PMP_{dur.replace(' ', '')}"
        gdf[col_name] = depth
    return gdf.to_json()

def create_pdf(cid, tool, metadata, pmp_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=16, style="B")
    pdf.cell(0, 10, text="BoM PMP Calculation Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.cell(0, 10, text="", new_x="LMARGIN", new_y="NEXT") 
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, text=f"Catchment ID / File: {cid}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, text=f"Calculation Method: {tool}", new_x="LMARGIN", new_y="NEXT")
    for key, val in metadata.items():
        pdf.cell(0, 10, text=f"{key}: {val}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, text="", new_x="LMARGIN", new_y="NEXT") 
    pdf.set_font("helvetica", size=12, style="B")
    pdf.cell(0, 10, text="Final PMP Depths:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    for dur, depth in pmp_dict.items():
        pdf.cell(0, 8, text=f"  - {dur}: {depth} mm", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())

def create_swmm_timeseries(cid, tool, pmp_dict):
    lines = [
        f"; BoM PMP Depth-Duration Curve",
        f"; Catchment ID: {cid}",
        f"; Method: {tool}",
        f"; Format: Duration(Hours)    Depth(mm)"
    ]
    for dur_str, depth in pmp_dict.items():
        # Clean the string to just output the numeric hour and depth
        dur_val = dur_str.replace(" Hours", "").strip()
        lines.append(f"{dur_val}\t{depth}")
        
    return "\n".join(lines).encode('utf-8')

def fetch_bom_ifd(lat, lon):
    url = f"https://www.bom.gov.au/water/designRainfalls/revised-ifd/?year=2016&coordinate_type=dd&latitude={lat}&longitude={lon}&sdmin=true&sdhr=true&sdday=true"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise ValueError("The Bureau of Meteorology server timed out. Please click download again.")
    except Exception as e:
        raise ValueError(f"Connection failed: {e}")
    
    # FIX: Sanitize the BoM HTML to remove percentage signs in colspans/rowspans
    # This prevents the Pandas "invalid literal for int() with base 10: '100%'" crash
    safe_html = re.sub(r'(colspan|rowspan)\s*=\s*["\']?[0-9]+%["\']?', r'\1="1"', response.text, flags=re.IGNORECASE)
    
    try:
        tables = pd.read_html(io.StringIO(safe_html))
    except Exception as e:
        raise ValueError(f"Could not parse HTML tables. Error: {e}")
        
    ifd_df = None
    for df in tables:
        cols_str = str(df.columns.values)
        if "Duration" in cols_str or "1EY" in cols_str or "1%" in cols_str:
            ifd_df = df
            break
            
    if ifd_df is None:
        raise ValueError("Could not find the IFD data table inside the BoM webpage.")
        
    # Flatten the multi-level header BoM uses
    if isinstance(ifd_df.columns, pd.MultiIndex):
        ifd_df.columns = ifd_df.columns.droplevel(0)
        
    # Clean up any empty columns or rows
    ifd_df = ifd_df.dropna(how='all').dropna(axis=1, how='all')
    clean_csv_text = ifd_df.to_csv(index=False)
    
    return ifd_df, clean_csv_text

def create_interactive_map(catchment_geo):
    # Reproject to WGS84 for web mapping
    catchment_wgs84 = catchment_geo.to_crs(epsg=4326)
    centroid = catchment_wgs84.geometry.iloc[0].centroid
    
    # Initialize map without a default basemap
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles=None)
    
    # Add High-Res Satellite and Street basemaps
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False
    ).add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street View', overlay=False).add_to(m)
    
    # Add catchment with semi-transparent red styling
    folium.GeoJson(
        catchment_wgs84,
        name='Catchment Boundary',
        style_function=lambda x: {
            'fillColor': '#ef4444', 
            'color': '#ef4444', 
            'weight': 2, 
            'fillOpacity': 0.4
        }
    ).add_to(m)
    
    # Add the layer toggle menu to the top right
    folium.LayerControl().add_to(m)
    
    return m

# --- 4. CALCULATION FUNCTIONS ---
def calculate_automated_pmp(catchment, progress_bar=None, status_text=None, custom_gsam_epw=56.7, custom_gtsmr_epw=73.0):
    def update_status(val, text):
        if progress_bar is not None and status_text is not None:
            progress_bar.progress(val)
            status_text.write(f"⏳ {text}")

    update_status(10, "Extracting catchment geometry and calculating area...")
    if not MASTER_PMP_ZONES_SHP:
        raise FileNotFoundError("Master PMP Zones shapefile is missing. Check System Diagnostics.")
        
    catchment_albers = catchment.to_crs(epsg=3577) 
    area_km2 = catchment_albers.geometry.area.sum() / 1e6
    catchment_centroid = catchment_albers.geometry.centroid.iloc[0]
    
    update_status(30, "Intersecting catchment with Master PMP Zones...")
    master_zones = gpd.read_file(MASTER_PMP_ZONES_SHP)
    if master_zones.crs is None:
        master_zones.set_crs(epsg=4283, inplace=True)
    master_zones = master_zones.to_crs(epsg=3577)
    intersecting_zone = master_zones[master_zones.contains(catchment_centroid)]
    
    if intersecting_zone.empty:
        raise ValueError("Catchment is outside all Australian BoM boundaries.")
        
    full_zone_name = intersecting_zone['PMP_ZONE'].iloc[0]
    
    if "GSAM" in full_zone_name and "Transition" not in full_zone_name:
        if not GSAM_CD_ROOT or not GSAM_MAF_GRID:
            raise FileNotFoundError("GSAM data files are missing. Check System Diagnostics.")
        method, cd_root = "GSAM", GSAM_CD_ROOT
        maf_grid_path, topo_grid_path, standard_epw = GSAM_MAF_GRID, GSAM_TOPO_GRID, custom_gsam_epw
    elif "GTSMR" in full_zone_name and "Transition" not in full_zone_name:
        if not GTSMR_CD_ROOT or not GTSMR_MAF_GRID:
            raise FileNotFoundError("GTSMR data files are missing. Check System Diagnostics.")
        method, cd_root = "GTSMR", GTSMR_CD_ROOT
        maf_grid_path, topo_grid_path, standard_epw = GTSMR_MAF_GRID, GTSMR_TOPO_GRID, custom_gtsmr_epw
    else:
        raise NotImplementedError("Transition zones require manual dual-method weighting.")

    update_status(50, f"Zone identified as {full_zone_name}. Extracting Moisture Adjustment Factor (MAF)...")
    with rasterio.open(maf_grid_path) as src:
        raster_crs = src.crs if src.crs else "EPSG:4283"
        geom = catchment.to_crs(raster_crs).geometry.iloc[0].__geo_interface__
    maf_value = get_catchment_average(maf_grid_path, geom) / standard_epw

    update_status(75, "Extracting Topographic Adjustment Factor (TAF)...")
    if topo_grid_path and Path(topo_grid_path).exists():
        with rasterio.open(topo_grid_path) as src:
            topo_crs = src.crs if src.crs else "EPSG:4283"
            topo_geom = catchment.to_crs(topo_crs).geometry.iloc[0].__geo_interface__
        topo_value = get_catchment_average(topo_grid_path, topo_geom)
    else:
        topo_value = 1.0
    
    update_status(85, "Reading Depth-Area-Duration (DAD) tables...")
    dad_table_path = find_dad_file_in_cd(cd_root, full_zone_name)
    raw_df = pd.read_csv(dad_table_path, header=None) if dad_table_path.suffix.lower() == '.csv' else pd.read_excel(dad_table_path, header=None)
        
    header_idx = raw_df[raw_df[0].astype(str).str.contains('AREA', case=False, na=False)].index[0]
    dad_df = raw_df.iloc[header_idx + 1:].copy()
    dad_df.columns = raw_df.iloc[header_idx]
    dad_df.rename(columns={dad_df.columns[0]: 'Area_km2'}, inplace=True)
    
    valid_cols = ['Area_km2']
    valid_cols.extend([str(c) for c in dad_df.columns if str(c).endswith('h') or str(c).isdigit()])
    dad_df = dad_df[valid_cols]
    dad_df.columns = [c.replace('h', '') if isinstance(c, str) else c for c in dad_df.columns]
    dad_df = dad_df.apply(pd.to_numeric, errors='coerce').dropna(subset=['Area_km2'])
    
    update_status(95, "Interpolating final PMP depths...")
    final_pmp = {}
    for duration in [col for col in dad_df.columns if col != 'Area_km2']:
        f_interp = interp1d(dad_df['Area_km2'], dad_df[duration], kind='linear', fill_value='extrapolate')
        final_pmp[f"{duration} Hours"] = round(float(f_interp(area_km2)) * maf_value * topo_value, 1)
        
    update_status(100, "Calculation complete!")
    return {
        "Area (km2)": round(area_km2, 2), 
        "Zone": full_zone_name, 
        "Method": method, 
        "MAF": round(maf_value, 3), 
        "TAF": round(topo_value, 3), 
        "PMP (mm)": final_pmp,
        "Catchment_Geo": catchment
    }

# --- 5. STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="PMP Calculator", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden !important;}
            header {visibility: hidden !important;}
            a[href^="https://streamlit.io/cloud"] {display: none !important;}
            [data-testid="stSidebarFooter"] {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

if "long_pmp_results" not in st.session_state:
    st.session_state.long_pmp_results = None
if "gsdm_results" not in st.session_state:
    st.session_state.gsdm_results = None
    
# Initialize EPW defaults in memory
if "gsam_epw" not in st.session_state:
    st.session_state.gsam_epw = 56.7
if "gtsmr_epw" not in st.session_state:
    st.session_state.gtsmr_epw = 73.0

# Callback function to reset EPW values
def reset_epw_defaults():
    st.session_state.gsam_epw = 56.7
    st.session_state.gtsmr_epw = 73.0

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    app_mode = st.radio("Select Module:", ["Probable Maximum Precipitation (PMP)", "Intensity-Frequency-Duration (IFD)"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Engineering Parameters (PMP)")
    st.write("Adjust standard EPW values for sensitivity testing:")
    
    # Tie the inputs directly to the session state keys
    gsam_epw_input = st.number_input("GSAM Standard EPW", key="gsam_epw", step=0.1)
    gtsmr_epw_input = st.number_input("GTSMR Standard EPW", key="gtsmr_epw", step=0.1)
    
    st.button("↺ Reset EPW Defaults", on_click=reset_epw_defaults)
    
    st.markdown("---")
    st.markdown("### 📊 System Diagnostics")
    st.write("Streamlit servers are hosted outside Australia, so the BoM API is geofenced. The app will rely exclusively on these offline files:")
    st.write("✅ Database Found" if LOCAL_GEOFABRIC_DB else "❌ Database Missing")
    st.write("✅ PMP Zones Found" if MASTER_PMP_ZONES_SHP else "❌ PMP Zones Missing")
    st.write("✅ GSAM CD Found" if GSAM_CD_ROOT else "❌ GSAM CD Missing")
    st.write("✅ GTSMR CD Found" if GTSMR_CD_ROOT else "❌ GTSMR CD Missing")
    
    if st.button("Reload Data (Clear Cache)"):
        st.cache_resource.clear()
        st.session_state.long_pmp_results = None
        st.session_state.gsdm_results = None
        reset_epw_defaults()
        st.rerun()

st.title("Design Rainfall & PMP Dashboard")

# ==========================================
# MODULE 1: PROBABLE MAXIMUM PRECIPITATION
# ==========================================
if "PMP" in app_mode:
    st.markdown("Calculate GSAM, GTSMR, and GSDM instantly.")

    # --- INPUT METHOD UI ---
    st.markdown("### 1. Define Catchment Boundary")
    input_method = st.radio("Select input method:", 
        ["Sub-Catchment (Geofabric SH_Network)", 
         "Drainage Basin (Geofabric NCBLevel2)", 
         "Upload Custom GIS File (.geojson or .zip)"],
        horizontal=True, label_visibility="collapsed")

    catchment_id_input = None
    uploaded_file = None
    cid_display = "Custom_Boundary"

    if "Sub-Catchment" in input_method:
        st.info("🔍 Find your target HydroID or SegmentNo using the [BoM Geofabric Portal (Layer 7)](https://portal.wsapi.cloud.bom.gov.au/arcgis/apps/sites/#/australian-water-data-service/datasets/35719064c4ea4ad79faa82f5c9c22068/explore?layer=7).")
        catchment_id_input = st.text_input("Enter Sub-Catchment ID:")
        if catchment_id_input: cid_display = str(catchment_id_input)
    elif "Drainage Basin" in input_method:
        st.info("🔍 Find your target HydroID using the [BoM Geofabric Portal (Layer 34)](https://portal.wsapi.cloud.bom.gov.au/arcgis/apps/sites/#/australian-water-data-service/datasets/35719064c4ea4ad79faa82f5c9c22068/explore?layer=34).")
        catchment_id_input = st.text_input("Enter Drainage Basin HydroID:")
        if catchment_id_input: cid_display = str(catchment_id_input)
    else:
        st.info("Upload your own catchment boundary. If your file contains multiple polygons, they will be dissolved into a single unified area.")
        uploaded_file = st.file_uploader("Upload Geometry", type=['geojson', 'zip'])
        if uploaded_file: cid_display = uploaded_file.name.split('.')[0]

    st.markdown("### 2. Run Calculation")
    tab_long, tab_short = st.tabs(["Long-Duration PMP (GSAM / GTSMR)", "Short-Duration PMP (GSDM)"])

    with tab_long:
        if st.button("Calculate Long-Duration PMP", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.write("⏳ Initializing spatial engine...")
            
            try:
                if "Sub-Catchment" in input_method:
                    if not catchment_id_input: raise ValueError("Please enter a Catchment ID.")
                    catchment_gdf = fetch_catchment_from_geofabric(catchment_id_input, "AHGFCatchment")
                elif "Drainage Basin" in input_method:
                    if not catchment_id_input: raise ValueError("Please enter a Basin ID.")
                    catchment_gdf = fetch_catchment_from_geofabric(catchment_id_input, "NCBLevel2DrainageBasinGroup")
                else:
                    if not uploaded_file: raise ValueError("Please upload a file.")
                    catchment_gdf = load_custom_catchment(uploaded_file)

                res = calculate_automated_pmp(catchment_gdf, progress_bar, status_text, gsam_epw_input, gtsmr_epw_input)
                st.session_state.long_pmp_results = {"results": res, "catchment_id": cid_display}
                
                progress_bar.empty()
                status_text.empty()
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.long_pmp_results = None

        if st.session_state.long_pmp_results is not None:
            results = st.session_state.long_pmp_results["results"]
            cid = st.session_state.long_pmp_results["catchment_id"]
            
            st.success("Calculation Complete!")
            
            col1, col2 = st.columns(2)
            col1.metric("Catchment Area", f"{results['Area (km2)']} km²")
            col1.metric("PMP Zone", results['Zone'])
            col2.metric("MAF", results['MAF'])
            col2.metric("TAF", results['TAF'])
            
            st.subheader("Final PMP Depths")
            df_pmp = pd.DataFrame(list(results['PMP (mm)'].items()), columns=['Duration', 'Depth (mm)'])
            
            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.table(df_pmp)
                
            with col_chart:
                df_pmp['Duration (hrs)'] = df_pmp['Duration'].str.replace(' Hours', '').astype(float)
                df_pmp['MAF Applied'] = results['MAF']
                df_pmp['TAF Applied'] = results['TAF']
                
                log_toggle_long = st.checkbox("Logarithmic X-Axis (Duration)", value=False, key="log_long")
                
                fig = px.line(
                    df_pmp, 
                    x='Duration (hrs)', 
                    y='Depth (mm)', 
                    markers=True, 
                    title="Long-Duration PMP Curve",
                    hover_data={'Duration (hrs)': True, 'Depth (mm)': True, 'Duration': False, 'MAF Applied': True, 'TAF Applied': True}
                )
                fig.update_traces(line_color='#ef4444', marker=dict(size=8))
                
                if log_toggle_long:
                    fig.update_layout(xaxis_type='log')
                    
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Catchment Location")
            catchment_geo = results['Catchment_Geo']
            m = create_interactive_map(catchment_geo)
            st_folium(m, width=720, height=400, returned_objects=[], key="map_long")
                    
            st.markdown("### 📥 Export Results")
            col_csv, col_gis, col_pdf, col_swmm = st.columns(4)
            
            csv_data = create_csv(results['PMP (mm)'])
            col_csv.download_button("Download CSV", data=csv_data, file_name=f"PMP_{cid}.csv", mime="text/csv", key="long_csv")
            
            meta_dict = {"Area_km2": results["Area (km2)"], "Zone": results["Zone"], "Method": results["Method"], "MAF": results["MAF"], "TAF": results["TAF"]}
            geojson_data = create_geojson(results['Catchment_Geo'], meta_dict, results['PMP (mm)'])
            col_gis.download_button("Download GIS", data=geojson_data, file_name=f"Catchment_{cid}.geojson", mime="application/geo+json", key="long_gis")
            
            pdf_data = create_pdf(cid, "Long-Duration PMP (GSAM / GTSMR)", meta_dict, results['PMP (mm)'])
            col_pdf.download_button("Download PDF", data=pdf_data, file_name=f"PMP_Report_{cid}.pdf", mime="application/pdf", key="long_pdf")
            
            swmm_data = create_swmm_timeseries(cid, "Long-Duration PMP (GSAM / GTSMR)", results['PMP (mm)'])
            col_swmm.download_button("Download SWMM/12d", data=swmm_data, file_name=f"PMP_Curve_{cid}.dat", mime="text/plain", key="long_swmm")

    with tab_short:
        with st.expander("📖 View GSDM Terrain & Moisture Rules"):
            st.markdown("Reference the official [BoM GSDM Guidebook (PDF)](http://www.bom.gov.au/water/designRainfalls/document/GSDM.pdf) for the required inputs. See Section 3 for terrain classification rules.")
        
        col1, col2 = st.columns(2)
        with col1:
            maf_input = st.number_input("Moisture Adjustment Factor (MAF) from BoM Figure 3:", min_value=0.0, max_value=2.0, value=1.0)
            elev_input = st.number_input("Mean Elevation (m)", min_value=0, value=500)
        with col2:
            r_percent = st.slider("Percentage of ROUGH terrain (%)", 0, 100, 0) / 100
            
        if st.button("Calculate Short-Duration PMP", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.write("⏳ Fetching catchment geometry...")
            
            try:
                if "Sub-Catchment" in input_method:
                    if not catchment_id_input: raise ValueError("Please enter a Catchment ID.")
                    catchment_gdf = fetch_catchment_from_geofabric(catchment_id_input, "AHGFCatchment")
                elif "Drainage Basin" in input_method:
                    if not catchment_id_input: raise ValueError("Please enter a Basin ID.")
                    catchment_gdf = fetch_catchment_from_geofabric(catchment_id_input, "NCBLevel2DrainageBasinGroup")
                else:
                    if not uploaded_file: raise ValueError("Please upload a file.")
                    catchment_gdf = load_custom_catchment(uploaded_file)

                progress_bar.progress(30)
                status_text.write("⏳ Calculating area and applying elevation factors...")
                area_km2 = catchment_gdf.to_crs(epsg=3577).geometry.area.sum() / 1e6
                eaf_value = 1.0 if elev_input <= 1500 else 1.0 - (((elev_input - 1500) / 300) * 0.05)
                s_percent = 1.0 - r_percent
                target_log_area = np.log10(max(1.0, min(area_km2, 1000.0)))
                log_areas = np.log10(gsdm_smooth_df.index)
                
                progress_bar.progress(70)
                status_text.write("⏳ Interpolating GSDM depth-duration curves...")
                final_gsdm = {}
                for dur in GSDM_DURATIONS:
                    ds_val = float(interp1d(log_areas, gsdm_smooth_df[dur], kind='linear', fill_value='extrapolate')(target_log_area))
                    dr_val = float(interp1d(log_areas, gsdm_rough_df[dur], kind='linear', fill_value='extrapolate')(target_log_area))
                    final_gsdm[f"{dur} Hours"] = round((s_percent * ds_val + r_percent * dr_val) * maf_input * eaf_value, 1)
                
                progress_bar.progress(100)
                status_text.write("⏳ Finalizing results...")
                
                st.session_state.gsdm_results = {
                    "final_gsdm": final_gsdm, "catchment": catchment_gdf, "area_km2": area_km2, 
                    "maf_input": maf_input, "eaf_value": eaf_value, "r_percent": r_percent, "catchment_id": cid_display
                }
                
                progress_bar.empty()
                status_text.empty()
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.gsdm_results = None

        if st.session_state.gsdm_results is not None:
            res = st.session_state.gsdm_results
            cid = res["catchment_id"]
            area_km2 = res["area_km2"]
            final_gsdm = res["final_gsdm"]
            
            if area_km2 > 1000:
                st.warning(f"Catchment area ({area_km2:.2f} km²) exceeds the 1,000 km² limit for GSDM. Results may not be valid.")
                
            st.success("Calculation Complete!")
            st.metric("Catchment Area", f"{area_km2:.2f} km²")
            df_gsdm = pd.DataFrame(list(final_gsdm.items()), columns=['Duration', 'Depth (mm)'])
            
            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.table(df_gsdm)
                
            with col_chart:
                df_gsdm['Duration (hrs)'] = df_gsdm['Duration'].str.replace(' Hours', '').astype(float)
                df_gsdm['MAF Applied'] = res['maf_input']
                df_gsdm['EAF Applied'] = round(res['eaf_value'], 3)
                df_gsdm['Rough Terrain (%)'] = res['r_percent'] * 100
                
                log_toggle_short = st.checkbox("Logarithmic X-Axis (Duration)", value=False, key="log_short")
                
                fig = px.line(
                    df_gsdm, 
                    x='Duration (hrs)', 
                    y='Depth (mm)', 
                    markers=True, 
                    title="Short-Duration PMP Curve",
                    hover_data={'Duration (hrs)': True, 'Depth (mm)': True, 'Duration': False, 'MAF Applied': True, 'EAF Applied': True, 'Rough Terrain (%)': True}
                )
                fig.update_traces(line_color='#ef4444', marker=dict(size=8))
                
                if log_toggle_short:
                    fig.update_layout(xaxis_type='log')
                    
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Catchment Location")
            catchment_geo = res["catchment"]
            m = create_interactive_map(catchment_geo)
            st_folium(m, width=720, height=400, returned_objects=[], key="map_short")
            
            st.markdown("### 📥 Export Results")
            col_csv, col_gis, col_pdf, col_swmm = st.columns(4)
            
            csv_data = create_csv(final_gsdm)
            col_csv.download_button("Download CSV", data=csv_data, file_name=f"GSDM_{cid}.csv", mime="text/csv", key="gsdm_csv")
            
            meta_dict = {"Area_km2": round(area_km2, 2), "MAF_Input": res["maf_input"], "EAF": round(res["eaf_value"], 3), "Rough_Pct": res["r_percent"] * 100}
            geojson_data = create_geojson(res["catchment"], meta_dict, final_gsdm)
            col_gis.download_button("Download GIS", data=geojson_data, file_name=f"Catchment_{cid}.geojson", mime="application/geo+json", key="gsdm_gis")
            
            pdf_data = create_pdf(cid, "Short-Duration PMP (GSDM)", meta_dict, final_gsdm)
            col_pdf.download_button("Download PDF", data=pdf_data, file_name=f"GSDM_Report_{cid}.pdf", mime="application/pdf", key="gsdm_pdf")
            
            swmm_data = create_swmm_timeseries(cid, "Short-Duration PMP (GSDM)", final_gsdm)
            col_swmm.download_button("Download SWMM/12d", data=swmm_data, file_name=f"GSDM_Curve_{cid}.dat", mime="text/plain", key="gsdm_swmm")

# ==========================================
# MODULE 2: INTENSITY-FREQUENCY-DURATION (IFD)
# ==========================================
elif "IFD" in app_mode:
    st.markdown("Retrieve 2016 ARR IFD design rainfalls directly from the Bureau of Meteorology.")
    
    st.markdown("### 1. Define Location")
    ifd_input_method = st.radio("Select location method:", 
        ["Manual Coordinate Entry (Lat/Lon)", "Use Catchment Centroid from GIS"], 
        horizontal=True, key="ifd_radio")
        
    if "Manual" in ifd_input_method:
        col1, col2 = st.columns(2)
        ifd_lat = col1.number_input("Latitude (e.g., -33.8688)", value=-33.8688, format="%.5f")
        ifd_lon = col2.number_input("Longitude (e.g., 151.2093)", value=151.2093, format="%.5f")
    else:
        st.info("Extract the centroid coordinates automatically from a BoM Geofabric ID or uploaded shapefile.")
        ifd_gis_method = st.radio("GIS Input Source:", 
            ["Sub-Catchment ID", "Drainage Basin ID", "Upload File"], 
            horizontal=True, key="ifd_gis_radio")
        
        ifd_catchment_id = None
        ifd_uploaded_file = None
        
        if "Sub-Catchment" in ifd_gis_method:
            ifd_catchment_id = st.text_input("Enter Sub-Catchment ID:", key="ifd_sub_id")
        elif "Drainage Basin" in ifd_gis_method:
            ifd_catchment_id = st.text_input("Enter Drainage Basin HydroID:", key="ifd_basin_id")
        else:
            ifd_uploaded_file = st.file_uploader("Upload Geometry", type=['geojson', 'zip'], key="ifd_upload")
        
    st.markdown("### 2. Fetch IFD Data")
    if st.button("Download IFD Data from BoM", type="primary"):
        with st.spinner("Fetching data from the Bureau of Meteorology..."):
            try:
                # 1. Determine Coordinates
                if "Manual" in ifd_input_method:
                    target_lat = ifd_lat
                    target_lon = ifd_lon
                else:
                    if "Sub-Catchment" in ifd_gis_method:
                        if not ifd_catchment_id: raise ValueError("Please enter a Catchment ID.")
                        catchment_gdf = fetch_catchment_from_geofabric(ifd_catchment_id, "AHGFCatchment")
                    elif "Drainage Basin" in ifd_gis_method:
                        if not ifd_catchment_id: raise ValueError("Please enter a Basin ID.")
                        catchment_gdf = fetch_catchment_from_geofabric(ifd_catchment_id, "NCBLevel2DrainageBasinGroup")
                    else:
                        if not ifd_uploaded_file: raise ValueError("Please upload a spatial file to extract the centroid.")
                        catchment_gdf = load_custom_catchment(ifd_uploaded_file)
                        
                    # Convert to WGS84 (Lat/Lon) to ensure correct API coordinates
                    catchment_wgs84 = catchment_gdf.to_crs(epsg=4326)
                    centroid = catchment_wgs84.geometry.iloc[0].centroid
                    target_lat = round(centroid.y, 5)
                    target_lon = round(centroid.x, 5)
                    st.info(f"📍 Extracted Centroid: Latitude {target_lat}, Longitude {target_lon}")

                # 2. Fetch the Data
                ifd_df, raw_csv_text = fetch_bom_ifd(target_lat, target_lon)
                
                # Save to session state so it persists on the screen
                st.session_state.ifd_results = {
                    "df": ifd_df, 
                    "raw_csv": raw_csv_text, 
                    "lat": target_lat, 
                    "lon": target_lon
                }
                
            except Exception as e:
                st.error(f"Error fetching IFD data: {e}")
                st.session_state.ifd_results = None

    # 3. Display the Results
    if getattr(st.session_state, 'ifd_results', None) is not None:
        res = st.session_state.ifd_results
        
        st.success("IFD Data successfully retrieved!")
        st.write(f"**Coordinates:** {res['lat']}, {res['lon']}")
        
        # Display the raw dataframe
        st.dataframe(res['df'], use_container_width=True)
        
        # Add a quick download button for the original BoM CSV
        st.download_button(
            label="Download Original BoM CSV", 
            data=res['raw_csv'].encode('utf-8'), 
            file_name=f"IFD_{res['lat']}_{res['lon']}.csv", 
            mime="text/csv"
        )
