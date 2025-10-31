import xarray as xr
import pandas as pd

# ======== USER INPUT =========
nc_file = "E06OCML3AD_20251030_01km_LAC_v1.0.0.nc"  # AOD NetCDF file
output_csv = "aod_full_grid_clean.csv"
# =============================

# Load dataset
ds = xr.open_dataset(nc_file)

# Inspect available variables
print("📘 Variables:", list(ds.data_vars))

# Select the main AOD variable (adjust if needed)
aod_var = "AOD" if "AOD" in ds.data_vars else list(ds.data_vars)[0]

# Convert to DataFrame
df = ds[aod_var].to_dataframe().reset_index()

# Drop missing (NaN) AOD values
df = df.dropna(subset=[aod_var])

# Optional: sort by latitude/longitude
df = df.sort_values(["latitude", "longitude"]).reset_index(drop=True)

# Save to CSV
df.to_csv(output_csv, index=False)

print(f"✅ Clean AOD grid (no missing values) saved to: {output_csv}")
print(f"🧾 Rows after cleaning: {len(df)}")
print(df.head())
