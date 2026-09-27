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

# --- 1. DATA DOWNLOADER (Runs only once when app starts) ---
@st.cache_resource
def download_and_extract_data():
    data_dir = "pmp_data"
    
    # Check if we already downloaded the data to prevent re-downloading
    if not os.path.exists(data_dir):
        st.info("First-time setup: Downloading data from Google Drive (this takes a minute)...")
        
        # Your exact Google Drive File ID
        file_id = '1r9mZGQcSGZ_iCdrV3tAbDdOY_Hueej5t'
        output_zip = 'pmp_data.zip'
        
        # gdown downloads the heavy file using just the ID
        gdown.download(id=file_id, output=output_zip, quiet=False)
        
        # Extract the ZIP folder onto the server
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        # Clean up the zip file to save server space
        os.remove(output_zip)
        
    return True

# Trigger the download
download_and_extract_data()

# --- 2. CONFIGURATION PATHS (Point to the extracted folders) ---
LOCAL_GEOFABRIC_DB = "pmp_data/SH_Catchments_GDB_V3_3/SH_Catchments_GDB"
MASTER_PMP_ZONES_SHP = "pmp_data/GTSMR_CD_2005/pmp_zones/zones_all.shp"
GSAM_MAF_GRID = "pmp_data/GSAM_CD_Oct_06/gridded_data/epw_annual.asc/epw_annual.asc"
GSAM_TOPO_GRID = "pmp_data/GSAM_CD_Oct_06/gridded_data/gsam_taf.asc/gsam_taf.asc"
# Add your other paths here...

# --- 3. HELPER FUNCTIONS ---
# (Paste your entire fetch_catchment_from_geofabric and calculate_automated_pmp functions here exactly as they are)


# --- 4. STREAMLIT WEB INTERFACE ---
st.title("BoM PMP Automated Calculator")
st.markdown("Calculate GSAM, GTSMR, and GSDM directly from the Geofabric API.")

# Sidebar / Main menu
tool = st.selectbox("Select a tool:", ["Long-Duration PMP (GSAM / GTSMR)", "Short-Duration PMP (GSDM)"])

if tool == "Long-Duration PMP (GSAM / GTSMR)":
    catchment_id = st.text_input("Enter Geofabric Catchment ID:")
    
    if st.button("Calculate"):
        if catchment_id:
            with st.spinner("Fetching Geofabric API and processing grids..."):
                try:
                    # Run your calculation function
                    # NOTE: Modify your calculate_automated_pmp function to RETURN the results instead of just printing them!
                    results = calculate_automated_pmp(catchment_id)
                    st.success("Calculation Complete!")
                    st.write(results)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a Catchment ID.")