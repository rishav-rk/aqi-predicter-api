import xarray as xr
import pandas as pd
import numpy as np # Import numpy for conditional operations

# ======== USER INPUT =========
nc_file = "E06OCML3AD_20251030_01km_LAC_v1.0.0.nc" 
output_nc = "AOD_India_Subset.nc"
output_csv = "AOD_India_Valid_Nearest_Imputed.csv" 
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

# --- 🚀 NEW IMPUTATION LOGIC: NEAREST NEIGHBOR ---

# 1. Select the AOD DataArray
aod_da = ds_india[aod_var]

# 2. **CRITICAL FIX:** Ensure dimensions match coordinate names for interpolation.
# This prevents KeyErrors if xarray's default dimension names (e.g., 'y', 'x') 
# don't match the coordinate names (lat_name, lon_name).
rename_map = {}
# Assuming latitude is the first dimension and longitude is the second after subsetting
if len(aod_da.dims) >= 1 and aod_da.dims[0] not in aod_da.coords:
    rename_map[aod_da.dims[0]] = lat_name
if len(aod_da.dims) >= 2 and aod_da.dims[1] not in aod_da.coords:
    rename_map[aod_da.dims[1]] = lon_name
    
# Rename if necessary
if rename_map:
    aod_da = aod_da.rename(rename_map)
    print(f"🔄 Renamed DataArray dimensions to: {aod_da.dims}")


# 3. Identify 0.0 entries and mark them as NaN
# Interpolation tools only fill NaNs. We treat 0.0s (and any negatives) as NaNs to be filled.
initial_zeros = (aod_da == 0.0).sum().item()
aod_da_imputed = aod_da.where((aod_da > 0.0) | np.isnan(aod_da), other=np.nan)


# 4. Perform Nearest Neighbor Imputation
# This fills the NaNs (originally 0.0s) with the closest non-NaN neighbor.
try:
    aod_da_imputed = aod_da_imputed.interpolate_na(
        dim=[lat_name, lon_name], 
        method="nearest"
    )
    print(f"🔄 Total 0.0 entries replaced using Nearest Neighbor: {initial_zeros}")
except Exception as e:
    print(f"❌ Nearest Neighbor Imputation failed: {e}")
    print("Falling back to dropping 0.0 entries...")
    # If imputation fails, we fall back to the safest method: dropping the 0.0s.
    aod_da_imputed = aod_da_imputed.dropna(dim=lat_name, how='all').dropna(dim=lon_name, how='all')

# --- Convert to DataFrame and Finalize ---

# Flatten the imputed DataArray into a DataFrame
df = aod_da_imputed.to_dataframe().reset_index()

# Drop any remaining missing values (NaNs), which were either originally NaNs or
# NaNs that couldn't be filled even by the nearest neighbor (e.g., area entirely 0.0/NaN).
df_valid = df.dropna(subset=[aod_var])

print(f"✅ Final valid data points (after imputation): {len(df_valid)}")


# Save cropped dataset (optional)
if output_nc:
    # Create a new dataset with the imputed AOD DataArray for saving
    ds_imputed = ds_india.copy()
    ds_imputed[aod_var] = aod_da_imputed
    ds_imputed.to_netcdf(output_nc)
    print(f"💾 Saved subset NetCDF (with 0.0 imputed): {output_nc}")

# Save CSV with valid grid points (optional)
if output_csv and not df_valid.empty:
    df_valid.to_csv(output_csv, index=False)
    print(f"💾 Saved valid AOD points CSV: {output_csv}")
else:
    print("🚫 No valid AOD values found in region after final cleaning.")