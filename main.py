from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple
import xarray as xr
import pandas as pd
import os
import numpy as np
import joblib 
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import sys
from scipy.spatial import cKDTree 

# --- Global Data Storage for AOD (Loaded from CSV) ---
AOD_DF = None
AOD_KD_TREE = None
AOD_DATA_LOADED = False

# --- Constants ---
AOD_CSV_PATH = r"AODdataset\AOD_India_Valid_Nearest_Imputed.csv" 
AOD_VARIABLE_NAME = "AOD" # Column name in your CSV
IMPUTED_AOD = 0.1 # Fallback value for critical errors

# --- Helper Function to Load AOD Data (Runs at startup) ---
def load_aod_data():
    """Loads the AOD CSV into a DataFrame and builds the KDTree once."""
    global AOD_DF, AOD_KD_TREE, AOD_DATA_LOADED
    
    if not os.path.exists(AOD_CSV_PATH):
        print(f"❌ AOD Data Error: Imputed CSV file not found at {AOD_CSV_PATH}")
        AOD_DATA_LOADED = False
        return

    try:
        # Load the CSV, assuming the columns are latitude, longitude, and AOD
        AOD_DF = pd.read_csv(AOD_CSV_PATH)
        
        required_cols = ['latitude', 'longitude', AOD_VARIABLE_NAME]
        if not all(col in AOD_DF.columns for col in required_cols):
             print(f"❌ AOD Data Error: CSV missing required columns (expected: {required_cols})")
             AOD_DATA_LOADED = False
             return

        # Prepare the coordinates for the KD-Tree (Latitude, Longitude)
        coords = AOD_DF[['latitude', 'longitude']].values
        AOD_KD_TREE = cKDTree(coords)
        
        AOD_DATA_LOADED = True
        print(f"✅ AOD Imputed Data loaded successfully from {AOD_CSV_PATH}")
        
    except Exception as e:
        print(f"❌ AOD Data Error during loading or KDTree building: {e}")
        AOD_DATA_LOADED = False

# Execute the loading function immediately upon import
load_aod_data()

# -----------------------------------------------------------------------------
# --- The 'get_aod_data' method that retrieves values from the CSV ---
def get_aod_data(lat: float, lon: float, aod_file_path: str) -> Tuple[float, str]:
    """
    Extracts the nearest AOD value from the pre-loaded Imputed CSV (DataFrame) 
    using KD-Tree for efficient geographical search.
    
    The 'aod_file_path' argument is kept for consistency with the overall API structure 
    but is not used for file loading here.
    """
    if not AOD_DATA_LOADED:
        status_msg = f"AOD Error: Imputed CSV data not loaded. Using default AOD={IMPUTED_AOD}."
        return IMPUTED_AOD, status_msg

    try:
        # Use the KD-Tree to find the index of the nearest neighbor
        # d is distance, i is the index of the nearest point in the DataFrame
        d, i = AOD_KD_TREE.query([lat, lon], k=1) 
        
        # Retrieve the AOD value using the index
        nearest_aod_value = AOD_DF.loc[i, AOD_VARIABLE_NAME]
        
        status_msg = f"AOD Status: OK from pre-loaded Imputed Data ({os.path.basename(AOD_CSV_PATH)})"
        
        return round(nearest_aod_value, 3), status_msg

    except Exception as e:
        status_msg = f"AOD Critical Error during KD-Tree query: {e}. Using default AOD."
        print(status_msg)
        return IMPUTED_AOD, status_msg


# --- PM2.5 Model Setup ---
# Load model package and features once when the API starts
try:
    # NOTE: Ensure pm25_rf_package.joblib is in the same directory as main.py
    model_package = joblib.load("pm25_rf_package.joblib")
    BEST_MODEL = model_package["tuned_model"]
    MODEL_FEATURES = model_package["features"]
    PM25_MODEL_LOADED = True
    print("✅ PM2.5 Prediction Model loaded successfully.")
except Exception as e:
    PM25_MODEL_LOADED = False
    print(f"❌ WARNING: Failed to load PM2.5 model package: {e}. PM2.5 prediction will return -1.0.")


# --- Constants ---
# NOTE: Replace these paths with the actual paths to your NetCDF files
ERA5_NETCDF_PATH = r"era5septdataset\93916cbc2757f8631fb61281c6978d1f.nc"
AOD_NETCDF_PATH = r"AODdataset\AOD_India_Subset.nc" 

# ERA5 Variable Names in the NetCDF File
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

# NEW CONSTANT: Default AOD to use when satellite data is missing (NaN, e.g., cloud cover)
IMPUTED_AOD = 0.1 


# --- Pydantic Models for Data Structure ---

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


# --- Helper Functions for PM2.5 Prediction ---

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


# --- Helper Function for AOD Processing (MODIFIED) ---

# --- Helper Function for ERA5 Processing ---

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
        
        # Use Xarray/Pandas conversion to ensure a proper Python datetime object.
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

# --- Existing Endpoints (omitted for brevity) ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all HTTP methods: GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # allows all headers
)

@app.get("/")
def read_home():
    """Returns a simple greeting at the root path."""
    return "Hello from fastApi"

# --- NEW Endpoint: Data Query ---

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
