import xarray as xr
import pandas as pd

# ======== USER INPUT =========
nc_file = "93916cbc2757f8631fb61281c6978d1f.nc"  # ERA5 file
lat_point, lon_point =  23.68679430535101, 68.93233351187288
output_csv = "era5_daily_avg.csv"
# =============================

# Load dataset
ds = xr.open_dataset(nc_file)

# Coordinate names
lat_name = "latitude"
lon_name = "longitude"
time_name = "valid_time"

# Handle longitude range
if ds[lon_name].max() > 180 and lon_point < 0:
    lon_point = lon_point + 360

# Select nearest grid point
ds_point = ds.sel({lat_name: lat_point, lon_name: lon_point}, method="nearest")

# Convert to DataFrame
df = ds_point.to_dataframe().reset_index()

# Ensure time is datetime
df[time_name] = pd.to_datetime(df[time_name])

# ✅ Keep only numeric columns for averaging
numeric_cols = df.select_dtypes(include="number").columns
df_numeric = df[[time_name] + list(numeric_cols)]

# Compute daily mean
df_daily = (
    df_numeric
    .resample(on=time_name, rule="1D")
    .mean()
    .sort_values(time_name)
)

# Save to CSV
df_daily.to_csv(output_csv, index=False)

print(f"✅ Daily ERA5 data extracted for lat={lat_point}, lon={lon_point}")
print(f"📁 Saved to: {output_csv}")
print(df_daily.head())
