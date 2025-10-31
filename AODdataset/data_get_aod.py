# import xarray as xr
# import pandas as pd

# # ======== USER INPUT =========
# nc_file = "E06OCML3AD_20251030_01km_LAC_v1.0.0.nc"  # AOD NetCDF file
# lat_lon_points = [
#     (23.68679430535101, 68.93233351187288)
# ]
# # =============================

# # Load dataset
# ds = xr.open_dataset(nc_file)

# # Inspect available variables
# print("📘 Variables:", list(ds.data_vars))

# # Pick the main AOD variable (adjust if needed)
# aod_var = "AOD" if "AOD" in ds.data_vars else list(ds.data_vars)[0]

# print("\n🌍 Extracting AOD values for given points:")

# for lat, lon in lat_lon_points:
#     try:
#         # Select nearest grid point
#         aod_value = ds[aod_var].sel(latitude=lat, longitude=lon, method="nearest").values
        
#         # Check if the value is valid
#         if pd.isna(aod_value):
#             print(f"Latitude: {lat}, Longitude: {lon} --> AOD: MISSING")
#         else:
#             print(f"Latitude: {lat}, Longitude: {lon} --> AOD: {aod_value:.3f}")
            
#     except Exception as e:
#         print(f"Latitude: {lat}, Longitude: {lon} --> ERROR: {e}")
import xarray as xr
import pandas as pd

# ======== USER INPUT =========
nc_file = "E06OCML3AD_20251030_01km_LAC_v1.0.0.nc"  # Global dataset
output_nc = "AOD_India_Subset.nc"       # Optional: save cropped dataset
output_csv = "AOD_India_Valid.csv"      # Optional: export valid AOD grid points
# India's bounding box
lat_min, lat_max = 6.5, 37.1
lon_min, lon_max = 68.0, 97.25
# =============================

# Load dataset
ds = xr.open_dataset(nc_file)

# Auto-detect coordinate names
lat_name = "latitude" if "latitude" in ds.coords else "lat"
lon_name = "longitude" if "longitude" in ds.coords else "lon"

# Auto-detect AOD variable
aod_var = "AOD" if "AOD" in ds.data_vars else list(ds.data_vars)[0]
print(f"\n🌍 Using variable: {aod_var}")

# Subset to India's bounding box
ds_india = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
print(f"🗺️ Extracted subset shape: {ds_india[aod_var].shape}")

# Drop missing values (NaNs)
# Flatten into a DataFrame
df = ds_india[aod_var].to_dataframe().reset_index()

# Keep only valid values
df_valid = df.dropna(subset=[aod_var])

print(f"✅ Valid data points: {len(df_valid)}")

# Save cropped dataset (optional)
if output_nc:
    ds_india.to_netcdf(output_nc)
    print(f"💾 Saved subset NetCDF: {output_nc}")

# Save CSV with valid grid points (optional)
if output_csv and not df_valid.empty:
    df_valid.to_csv(output_csv, index=False)
    print(f"💾 Saved valid AOD points CSV: {output_csv}")
else:
    print("🚫 No valid AOD values found in region.")

