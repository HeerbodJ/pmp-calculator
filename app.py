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

# --- 3. HELPER FUNCTIONS ---
def fetch_catchment_from_geofabric(catchment_input):
    try:
        catchment_id = int(str(catchment_input).strip(' "\''))
    except ValueError:
        raise ValueError("Input must be a valid Catchment ID number.")
        
    base_url = "https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/ahgf/Geofabric_V3x_All_Products/FeatureServer/7/query"
    headers = {'User-Agent': 'Mozilla/5.0'}
    params = {'where': f"SegmentNo={catchment_id}", 'outFields': '*', 'returnGeometry': 'true', 'f': 'geojson'}
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data or "features" not in data or len(data["features"]) == 0:
            params['where'] = f"HydroID={catchment_id}"
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            data = response.json()
            if "error" in data or "features" not in data or len(data["features"]) == 0:
                raise ValueError(f"Could not find catchment ID {catchment_id} in Geofabric API.")
                
        catchment = gpd.GeoDataFrame.from_features(data["features"])
        catchment.set_crs(epsg=4326, inplace=True) 
        return catchment
        
    except Exception:
        if not LOCAL_GEOFABRIC_DB or not Path(LOCAL_GEOFABRIC_DB).exists():
            raise FileNotFoundError("BoM API is blocked, and the offline database could not be found. Check System Diagnostics.")
            
        catchment = gpd.read_file(LOCAL_GEOFABRIC_DB, layer='AHGFCatchment', where=f"SegmentNo={catchment_id}", engine="pyogrio")
        if catchment.empty:
            catchment = gpd.read_file(LOCAL_GEOFABRIC_DB, layer='AHGFCatchment', where=f"HydroID={catchment_id}", engine="pyogrio")
            
        if catchment.empty:
            raise ValueError(f"Could not find Catchment ID {catchment_id} in the local offline database.")
             
        if catchment.crs is None:
            catchment.set_crs(epsg=4283, inplace=True)
        catchment = catchment.to_crs(epsg=4326)
        return catchment

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

# --- 4. CALCULATION FUNCTIONS ---
def calculate_automated_pmp(catchment_id):
    if not MASTER_PMP_ZONES_SHP:
        raise FileNotFoundError("Master PMP Zones shapefile is missing. Check System Diagnostics.")
        
    catchment = fetch_catchment_from_geofabric(catchment_id)
    catchment_albers = catchment.to_crs(epsg=3577) 
    area_km2 = catchment_albers.geometry.area.sum() / 1e6
    catchment_centroid = catchment_albers.geometry.centroid.iloc[0]
    
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
        maf_grid_path, topo_grid_path, standard_epw = GSAM_MAF_GRID, GSAM_TOPO_GRID, GSAM_STANDARD_EPW
    elif "GTSMR" in full_zone_name and "Transition" not in full_zone_name:
        if not GTSMR_CD_ROOT or not GTSMR_MAF_GRID:
            raise FileNotFoundError("GTSMR data files are missing. Check System Diagnostics.")
        method, cd_root = "GTSMR", GTSMR_CD_ROOT
        maf_grid_path, topo_grid_path, standard_epw = GTSMR_MAF_GRID, GTSMR_TOPO_GRID, GTSMR_STANDARD_EPW
    else:
        raise NotImplementedError("Transition zones require manual dual-method weighting.")

    with rasterio.open(maf_grid_path) as src:
        raster_crs = src.crs if src.crs else "EPSG:4283"
        geom = catchment.to_crs(raster_crs).geometry.iloc[0].__geo_interface__
    maf_value = get_catchment_average(maf_grid_path, geom) / standard_epw

    if topo_grid_path and Path(topo_grid_path).exists():
        with rasterio.open(topo_grid_path) as src:
            topo_crs = src.crs if src.crs else "EPSG:4283"
            topo_geom = catchment.to_crs(topo_crs).geometry.iloc[0].__geo_interface__
        topo_value = get_catchment_average(topo_grid_path, topo_geom)
    else:
        topo_value = 1.0
    
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
    
    final_pmp = {}
    for duration in [col for col in dad_df.columns if col != 'Area_km2']:
        f_interp = interp1d(dad_df['Area_km2'], dad_df[duration], kind='linear', fill_value='extrapolate')
        final_pmp[f"{duration} Hours"] = round(float(f_interp(area_km2)) * maf_value * topo_value, 1)
        
    return {"Area (km2)": round(area_km2, 2), "Zone": full_zone_name, "Method": method, "MAF": round(maf_value, 3), "TAF": round(topo_value, 3), "PMP (mm)": final_pmp}

# --- 5. STREAMLIT WEB INTERFACE ---
st.set_page_config(page_title="PMP Calculator", layout="wide")

# Sidebar Diagnostics
with st.sidebar:
    st.markdown("### 📊 System Diagnostics")
    st.write("Streamlit servers are hosted outside Australia, so the BoM API is geofenced. The app will rely exclusively on these offline files:")
    st.write("✅ Database Found" if LOCAL_GEOFABRIC_DB else "❌ Database Missing")
    st.write("✅ PMP Zones Found" if MASTER_PMP_ZONES_SHP else "❌ PMP Zones Missing")
    st.write("✅ GSAM CD Found" if GSAM_CD_ROOT else "❌ GSAM CD Missing")
    st.write("✅ GTSMR CD Found" if GTSMR_CD_ROOT else "❌ GTSMR CD Missing")
    
    if st.button("Reload Data (Clear Cache)"):
        st.cache_resource.clear()
        st.rerun()

st.title("PMP Calculator")
st.markdown("Calculate GSAM, GTSMR, and GSDM instantly.")

# NEW: Info box with direct links to the BoM Geofabric and National Map
st.info("🔍 **Need a Catchment ID?** Find your target HydroID or SegmentNo using the official [BoM Geofabric Portal](http://www.bom.gov.au/water/geofabric/) or by exploring the Catchment layers on [NationalMap](https://nationalmap.gov.au/).")

tool = st.selectbox("Select a tool:", ["Long-Duration PMP (GSAM / GTSMR)", "Short-Duration PMP (GSDM)"])
catchment_id = st.text_input("Enter Geofabric Catchment ID:")

if tool == "Long-Duration PMP (GSAM / GTSMR)":
    if st.button("Calculate Long-Duration PMP", type="primary"):
        if catchment_id:
            with st.spinner("Processing geospatial data..."):
                try:
                    results = calculate_automated_pmp(catchment_id)
                    st.success("Calculation Complete!")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Catchment Area", f"{results['Area (km2)']} km²")
                    col1.metric("PMP Zone", results['Zone'])
                    col2.metric("MAF", results['MAF'])
                    col2.metric("TAF", results['TAF'])
                    
                    st.subheader("Final PMP Depths")
                    df_pmp = pd.DataFrame(list(results['PMP (mm)'].items()), columns=['Duration', 'Depth (mm)'])
                    st.table(df_pmp)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a Catchment ID.")

elif tool == "Short-Duration PMP (GSDM)":
    st.markdown("### Catchment Details")
    
    # NEW: Direct reference link to the official GSDM guidebook for finding MAF and Roughness percentages
    st.markdown("📖 *Reference the official [BoM GSDM Guidebook (PDF)](http://www.bom.gov.au/water/designRainfalls/document/GSDM.pdf) for the required inputs below.*")
    
    col1, col2 = st.columns(2)
    with col1:
        maf_input = st.number_input("Moisture Adjustment Factor (MAF) from BoM Figure 3:", min_value=0.0, max_value=2.0, value=1.0)
        elev_input = st.number_input("Mean Elevation (m)", min_value=0, value=500)
    with col2:
        r_percent = st.slider("Percentage of ROUGH terrain (%)", 0, 100, 0) / 100
        st.caption("See Guidebook Section 3 for terrain classification rules.")
        
    if st.button("Calculate Short-Duration PMP", type="primary"):
        if catchment_id:
            with st.spinner("Processing GSDM..."):
                try:
                    catchment = fetch_catchment_from_geofabric(catchment_id)
                    area_km2 = catchment.to_crs(epsg=3577).geometry.area.sum() / 1e6
                    
                    if area_km2 > 1000:
                        st.warning(f"Catchment area ({area_km2:.2f} km²) exceeds the 1,000 km² limit for GSDM. Results may not be valid.")
                        
                    eaf_value = 1.0 if elev_input <= 1500 else 1.0 - (((elev_input - 1500) / 300) * 0.05)
                    s_percent = 1.0 - r_percent
                    target_log_area = np.log10(max(1.0, min(area_km2, 1000.0)))
                    log_areas = np.log10(gsdm_smooth_df.index)
                    
                    final_gsdm = {}
                    for dur in GSDM_DURATIONS:
                        ds_val = float(interp1d(log_areas, gsdm_smooth_df[dur], kind='linear', fill_value='extrapolate')(target_log_area))
                        dr_val = float(interp1d(log_areas, gsdm_rough_df[dur], kind='linear', fill_value='extrapolate')(target_log_area))
                        final_gsdm[f"{dur} Hours"] = round((s_percent * ds_val + r_percent * dr_val) * maf_input * eaf_value, 1)
                    
                    st.success("Calculation Complete!")
                    st.metric("Catchment Area", f"{area_km2:.2f} km²")
                    df_gsdm = pd.DataFrame(list(final_gsdm.items()), columns=['Duration', 'Depth (mm)'])
                    st.table(df_gsdm)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a Catchment ID.")