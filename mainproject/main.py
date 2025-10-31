from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import xarray as xr
import pandas as pd
import os
import numpy as np
import joblib 
from datetime import datetime
import sys
import requests
from io import BytesIO

# --- Constants: Remote URLs and Local Paths ---
# IMPORTANT: You MUST replace the FILE_ID_... placeholders below with the actual file IDs 
# from your Google Drive shared links (after setting permission to "Anyone with the link").

# Remote URLs (Google Drive Direct Download Format)
# model :  https://drive.google.com/file/d/1PiLLVaJovlzmPCNIR1XBYP3F4uogqDb_/view?usp=sharing
# aod : https://drive.google.com/file/d/1e8qErZVliCRNgXntJ6-8ZS3uqjyMOLCY/view?usp=sharing
MODEL_URL = "https://drive.google.com/uc?export=download&id=1PiLLVaJovlzmPCNIR1XBYP3F4uogqDb_"
ERA5_URL = "https://drive.google.com/uc?export=download&id=1yD6CZ5Qbu3HNUbWRVbJGAzpxOewf6aIG"
AOD_URL = "https://drive.google.com/uc?export=download&id=1e8qErZVliCRNgXntJ6-8ZS3uqjyMOLCY"

# Local Paths (Serverless functions typically use /tmp for writable storage)
LOCAL_TMP_DIR = "/tmp"
ERA5_NETCDF_PATH = os.path.join(LOCAL_TMP_DIR, "93916cbc2757f8631fb61281c6978d1f.nc")
AOD_NETCDF_PATH = os.path.join(LOCAL_TMP_DIR, "E06OCML3AD_20251030_01km_LAC_v1.0.0.nc") 
MODEL_PATH = os.path.join(LOCAL_TMP_DIR, "pm25_rf_package.joblib")


KELVIN_TO_CELSIUS = 273.15
ERA5_VARS_MAP = {
    "t2m": "t2m",       # 2m Temperature (Kelvin)
    "d2m": "d2m",       # 2m Dew Point Temperature (Kelvin)
    "u10": "u10",       # 10m U-component of wind (m/s)
    "v10": "v10",       # 10m V-component of wind (m/s)
    "sp": "sp",         # Surface Pressure (Pascals)
    "tp": "tp",         # Total Precipitation (Meters)
}

# AOD Variable Name (Adjust if your file uses a different name)
AOD_VARIABLE_NAME = "AOD"


# --- Pydantic Models for Data Structure (Unchanged) ---

class Coordinates(BaseModel):
    """Defines the expected structure for the input request body."""
    latitude: float
    longitude: float

class AodEra5Data(BaseModel):
    """
    Defines the structure for the comprehensive output response data, 
    including calculated and converted ERA5 variables and the PM2.5 prediction.
    """
    latitude: float
    longitude: float
    aod_value: float 
    aod_status: str 
    
    # New Field
    pm25_prediction: float 
    
    # Converted ERA5 Variables (for display)
    era5_temp_2m_c: float
    era5_dew_point_2m_c: float
    era5_wind_speed_10m_ms: float
    era5_surface_pressure_hpa: float
    era5_total_precip_m: float 
    
    source: str 


# --- Utility Function: Download Files ---

def download_file_if_not_exists(url: str, local_path: str):
    """
    Downloads a file from a URL to the local_path if it doesn't already exist.
    Handles Google Drive large file warning by passing the 'confirm' cookie.
    Raises an exception if the download fails.
    """
    if os.path.exists(local_path):
        print(f"File already exists: {os.path.basename(local_path)}. Skipping download.")
        return

    print(f"Downloading {os.path.basename(local_path)} from {url}...")
    try:
        # Check for Google Drive 'confirm' message for large files
        response = requests.get(url, stream=True, timeout=300) 
        token = None
        if 'confirm' in response.cookies:
            token = response.cookies['confirm']
            url = url + "&confirm=" + token
            response = requests.get(url, stream=True, timeout=300)
            
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete: {os.path.basename(local_path)}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {os.path.basename(local_path)}: {e}")
        # Re-raise to prevent the function from continuing with missing data
        raise RuntimeError(f"Failed to download required file: {os.path.basename(local_path)}")


# --- PM2.5 Model Setup (Updated to use download logic) ---

# Global variables to store the model and features
BEST_MODEL = None
MODEL_FEATURES = []
PM25_MODEL_LOADED = False

# Function to initialize the model and data files
def initialize_resources():
    global BEST_MODEL, MODEL_FEATURES, PM25_MODEL_LOADED
    
    # 1. Ensure /tmp directory exists
    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
    
    # 2. Download NetCDF files first (if missing, they trigger fallbacks later)
    try:
        download_file_if_not_exists(ERA5_URL, ERA5_NETCDF_PATH)
        download_file_if_not_exists(AOD_URL, AOD_NETCDF_PATH)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to download NetCDF data files: {e}. API will use fallback data for ERA5/AOD.")
        
    # 3. Download and load Model
    try:
        download_file_if_not_exists(MODEL_URL, MODEL_PATH)
        model_package = joblib.load(MODEL_PATH)
        BEST_MODEL = model_package["tuned_model"]
        MODEL_FEATURES = model_package["features"]
        PM25_MODEL_LOADED = True
        print("✅ PM2.5 Prediction Model loaded successfully.")
    except Exception as e:
        PM25_MODEL_LOADED = False
        print(f"❌ CRITICAL ERROR: Failed to load PM2.5 model package: {e}. Prediction disabled.")
        

# Run initialization upon FastAPI startup
initialize_resources()


# --- Helper Functions for PM2.5 Prediction (Unchanged) ---

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as training."""
    # Note: df['date'] must be a datetime object
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['wind_dir_deg'] = (np.degrees(np.arctan2(-df['u10'], -df['v10'])) + 360) % 360
    return df

def predict_pm25(raw_data: Dict[str, Any]) -> float:
    """
    Predicts PM2.5 based on raw input values (Kelvin, Pascals, etc.).
    raw_data must include: lat, lon, aod, d2m, t2m, u10, v10, sp, tp, date (datetime object).
    """
    if not PM25_MODEL_LOADED:
        return -1.0 # Sentinel value for model not loaded

    try:
        # The raw_data dictionary already contains all necessary keys including 
        # lat, lon, aod, d2m, t2m, u10, v10, sp, tp, and date.
        df = pd.DataFrame([raw_data])
        df = feature_engineering(df)
        
        # Ensure all expected features are present before prediction
        X = df[MODEL_FEATURES]
        
        pred = BEST_MODEL.predict(X)[0]
        # PM2.5 cannot be negative; cap it at 0 and round
        return round(max(0.0, pred), 2)
    except Exception as e:
        print(f"Error during PM2.5 prediction: {e}")
        return -2.0 # Sentinel value for prediction error


# --- Helper Function for AOD Processing (Uses Updated Path) ---

def get_aod_data(lat: float, lon: float, nc_file_path: str) -> Tuple[float, str]:
    """
    Extracts the nearest AOD value from the NetCDF file.
    Returns the AOD value and a status message.
    """
    FALLBACK_AOD = -999.0 
    
    if not os.path.exists(nc_file_path):
        status_msg = f"AOD Error: File {os.path.basename(nc_file_path)} not found at {nc_file_path}. Using fallback value."
        return FALLBACK_AOD, status_msg
        
    try:
        ds = xr.open_dataset(nc_file_path)
        
        if AOD_VARIABLE_NAME not in ds.data_vars:
            status_msg = f"AOD Error: Variable '{AOD_VARIABLE_NAME}' not found in file."
            return FALLBACK_AOD, status_msg

        aod_xarray_point = ds[AOD_VARIABLE_NAME].sel(latitude=lat, longitude=lon, method="nearest")
        aod_value = aod_xarray_point.values.item() 

        if np.isnan(aod_value):
            status_msg = "AOD Status: Missing (e.g., Cloud Cover)"
            return FALLBACK_AOD, status_msg
        
        if aod_value < 0:
            aod_value = 0.0
            
        status_msg = f"AOD Status: OK from {os.path.basename(nc_file_path)}"
        return round(aod_value, 3), status_msg

    except Exception as e:
        status_msg = f"AOD Critical Error: {e}"
        print(status_msg)
        return FALLBACK_AOD, status_msg


# --- Helper Function for ERA5 Processing (Uses Updated Path) ---

def get_era5_data(lat: float, lon: float, nc_file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Extracts raw and processed ERA5 data. Returns: 
    (processed_data_for_api, raw_data_for_prediction, source_message)
    """
    
    # --- Fallback Data Definitions ---
    error_msg = f"Error: File {os.path.basename(nc_file_path)} not found. Using fallback data."
    
    # Fallback values for API display (Converted units)
    fallback_processed_data = {
        'era5_temp_2m_c': 20.0,
        'era5_dew_point_2m_c': 10.0,
        'era5_wind_speed_10m_ms': 5.0,
        'era5_surface_pressure_hpa': 1013.25,
        'era5_total_precip_m': 0.0,
    }
    # Fallback values for Model Input (Raw units)
    fallback_raw_data = {
        't2m': KELVIN_TO_CELSIUS + 20.0, 
        'd2m': KELVIN_TO_CELSIUS + 10.0,
        'u10': 3.0, # Default components (speed = 5 m/s)
        'v10': 4.0, 
        'sp': 101325.0, # Default hPa back to Pascals
        'tp': 0.0,
        'date': datetime.now(), # Use current datetime as fallback
        'lat': lat,
        'lon': lon,
        'aod': -999.0 # Placeholder, will be overwritten by AOD call
    }
    
    if not os.path.exists(nc_file_path):
        print(f"Error: ERA5 file not found at {nc_file_path}")
        return fallback_processed_data, fallback_raw_data, error_msg
        
    try:
        ds = xr.open_dataset(nc_file_path)
        lat_name, lon_name, time_name = "latitude", "longitude", "time" if "time" in ds.coords else "valid_time" 
        
        for var_name in ERA5_VARS_MAP.values():
            if var_name not in ds.variables:
                 raise ValueError(f"Required variable '{var_name}' not found in NetCDF file.")

        if ds[lon_name].max() > 180 and lon < 0:
            lon += 360

        ds_point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
        latest_data = ds_point.sortby(time_name, ascending=False).isel({time_name: 0})
        
        # --- 1. Extract RAW values (for PM2.5 Model Input) ---
        raw_values = {
            key: latest_data[val].item() 
            for key, val in ERA5_VARS_MAP.items()
        }
        
        # FIXED: Robust datetime extraction
        time_scalar = latest_data[time_name].item()
        if isinstance(time_scalar, datetime):
            raw_values['date'] = time_scalar
        else:
            # Assumes it's a numpy datetime64 or a format that can be converted by pandas
            raw_values['date'] = pd.to_datetime(time_scalar).to_pydatetime()

        raw_values['lat'] = lat
        raw_values['lon'] = lon
        raw_values['aod'] = -999.0 # Placeholder, will be replaced by actual AOD later
        
        # --- 2. Processed values (for API Display) ---
        processed_data = {}
        t2m_kelvin = raw_values['t2m']
        d2m_kelvin = raw_values['d2m']
        u10 = raw_values['u10']
        v10 = raw_values['v10']
        sp_pa = raw_values['sp']
        
        processed_data['era5_temp_2m_c'] = round(t2m_kelvin - KELVIN_TO_CELSIUS, 2)
        processed_data['era5_dew_point_2m_c'] = round(d2m_kelvin - KELVIN_TO_CELSIUS, 2)
        wind_speed = (u10**2 + v10**2)**0.5
        processed_data['era5_wind_speed_10m_ms'] = round(wind_speed, 2)
        pressure_hpa = sp_pa / 100 
        processed_data['era5_surface_pressure_hpa'] = round(pressure_hpa, 2)
        processed_data['era5_total_precip_m'] = raw_values['tp']
        
        source_msg = f"ERA5 from {os.path.basename(nc_file_path)} (latest time step extracted)"
        return processed_data, raw_values, source_msg

    except Exception as e:
        error_msg = f"Critical error during ERA5 processing: {e}. Using fallback data."
        print(error_msg)
        return fallback_processed_data, fallback_raw_data, error_msg


# --- FastAPI Initialization ---
app = FastAPI()

# --- Endpoints (Unchanged) ---

@app.get("/")
def read_home():
    """Returns a simple greeting at the root path."""
    return "Hello from fastApi"

@app.get("/items/{item_id}")
def read_item(item_id: int):
    """Returns the item ID provided in the path, now under a clear /items prefix."""
    return f"The ID provided is {item_id}"

@app.get("/api/status")
def api_status():
    """Endpoint often used to check if the API is alive under the Vercel-assumed /api prefix."""
    return {"status": "ok", "message": "API is running"}

# --- Data Query Endpoint (Uses Updated Paths) ---

@app.post("/data/query", response_model=AodEra5Data)
def query_aod_era5_data(coords: Coordinates):
    """
    Receives latitude and longitude, fetches AOD and ERA5 data, 
    and predicts PM2.5 using the loaded model.
    """
    
    lat = coords.latitude
    lon = coords.longitude

    # --- 1. ERA5 Data Retrieval ---
    # Returns processed data for display, and raw data for the model
    era5_data, raw_era5_data, era5_source = get_era5_data(lat, lon, ERA5_NETCDF_PATH)


    # --- 2. AOD Data Retrieval ---
    aod_value, aod_status = get_aod_data(lat, lon, AOD_NETCDF_PATH)
    
    
    # --- 3. PM2.5 Prediction ---
    # Prepare the final raw data dictionary by inserting the AOD value
    raw_prediction_data = raw_era5_data
    raw_prediction_data['aod'] = aod_value
    
    pm25_pred = predict_pm25(raw_prediction_data)
    
    
    # --- COMBINED RESPONSE ---
    
    response_data = {
        "latitude": lat,
        "longitude": lon,
        "aod_value": aod_value,
        "source": era5_source,
        "aod_status": aod_status,
        "pm25_prediction": pm25_pred, # Include the new prediction
    }
    # Merge the processed ERA5 data dictionary into the final response
    response_data.update(era5_data)
    
    return response_data
