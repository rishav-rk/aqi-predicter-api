import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from datetime import datetime

# Load model package
model_package = joblib.load("pm25_rf_package.joblib")
best_model = model_package["tuned_model"]
features = model_package["features"]

def feature_engineering(df):
    """Apply the same feature engineering as training."""
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['wind_dir_deg'] = (np.degrees(np.arctan2(-df['u10'], -df['v10'])) + 360) % 360
    return df

def predict_from_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = feature_engineering(df)
    X = df[features]
    preds = best_model.predict(X)
    df['PM25_pred'] = preds
    print(df[['date'] + features + ['PM25_pred']].head())
    return df

def predict_from_values(values_dict):
    """
    values_dict must include:
    lat, lon, aod, d2m, t2m, u10, v10, sp, tp, date (YYYY-MM-DD)
    """
    df = pd.DataFrame([values_dict])
    df['date'] = pd.to_datetime(df['date'])
    df = feature_engineering(df)
    X = df[features]
    pred = best_model.predict(X)[0]
    print("Predicted PM2.5:", pred)
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PM2.5 Prediction Script")
    parser.add_argument("--csv", type=str, help="Path to test.csv file")
    parser.add_argument("--values", nargs="+", help="Direct values: lat lon aod d2m t2m u10 v10 sp tp date(YYYY-MM-DD)")
    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv)
    elif args.values:
        if len(args.values) != 10:
            print("Error: Must provide 10 values: lat lon aod d2m t2m u10 v10 sp tp date")
            sys.exit(1)
        keys = ['lat','lon','aod','d2m','t2m','u10','v10','sp','tp','date']
        vals = args.values
        vals[:9] = map(float, vals[:9])  # numeric features
        values_dict = dict(zip(keys, vals))
        predict_from_values(values_dict)
    else:
        print("Please provide either --csv test.csv or --values ...")
