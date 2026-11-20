"""Feature engineering module incorporating meteorological and Guinean climate domain knowledge."""

import numpy as np
import pandas as pd

from src.stations_metadata import get_stations_df


def compute_dew_point(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Approximate dew point using Magnus formula."""
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(np.clip(rh, 1.0, 100.0) / 100.0)
    td = (b * alpha) / (a - alpha)
    return td


def compute_vpd(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Vapor Pressure Deficit (kPa)."""
    # Saturation vapor pressure es (kPa)
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual vapor pressure ea (kPa)
    ea = es * (np.clip(rh, 0.0, 100.0) / 100.0)
    vpd = es - ea
    return vpd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add meteorological and thermodynamic indicators."""
    d = df.copy()

    # 1. Thermodynamics
    if "temperature" in d.columns and "humidity" in d.columns:
        d["dew_point"] = compute_dew_point(d["temperature"], d["humidity"])
        d["dew_point_depression"] = d["temperature"] - d["dew_point"]
        d["vpd"] = compute_vpd(d["temperature"], d["humidity"])

    if "heat_index" in d.columns and "temperature" in d.columns:
        d["heat_temp_diff"] = d["heat_index"] - d["temperature"]

    # 2. Wind vectors and West African wind regimes (Harmattan vs Mousson)
    if "wind_speed" in d.columns and "wind_deg" in d.columns:
        rad = np.radians(d["wind_deg"])
        # Wind components: u (eastward) and v (northward)
        d["wind_u"] = -d["wind_speed"] * np.sin(rad)
        d["wind_v"] = -d["wind_speed"] * np.cos(rad)

        # Harmattan alignment (from NE ~ 45 deg)
        d["harmattan_proj"] = d["wind_speed"] * np.cos(np.radians(d["wind_deg"] - 45.0))
        # Monsoon alignment (from SW ~ 225 deg, brings heavy moisture from Gulf of Guinea)
        d["monsoon_proj"] = d["wind_speed"] * np.cos(np.radians(d["wind_deg"] - 225.0))

    # 3. Cyclical temporal features
    if "hour" in d.columns:
        d["hour_sin"] = np.sin(2.0 * np.pi * d["hour"] / 24.0)
        d["hour_cos"] = np.cos(2.0 * np.pi * d["hour"] / 24.0)

    if "dayofyear" in d.columns:
        d["doy_sin"] = np.sin(2.0 * np.pi * d["dayofyear"] / 365.25)
        d["doy_cos"] = np.cos(2.0 * np.pi * d["dayofyear"] / 365.25)

    if "month" in d.columns:
        d["month_sin"] = np.sin(2.0 * np.pi * d["month"] / 12.0)
        d["month_cos"] = np.cos(2.0 * np.pi * d["month"] / 12.0)

    # 4. Proxy for Front Inter-Tropical (FIT / ITCZ) seasonal migration
    # In West Africa, FIT oscillates between ~8°N (Jan) and ~20°N (Aug)
    if "dayofyear" in d.columns:
        d["fit_latitude_est"] = 14.0 + 6.0 * np.sin(2.0 * np.pi * (d["dayofyear"] - 110) / 365.25)

    # 5. Merge stations metadata
    stations_df = get_stations_df()
    d = d.merge(stations_df, on="location", how="left")

    if "lat" in d.columns and "fit_latitude_est" in d.columns:
        # Distance to FIT: positive means station is south of FIT (in the moist monsoon zone)
        d["fit_lat_diff"] = d["fit_latitude_est"] - d["lat"]
        d["is_south_of_fit"] = (d["fit_lat_diff"] > 0).astype(np.float32)

    if "altitude_m" in d.columns and "distance_cote_km" in d.columns:
        # Orographic relief index (important for Fouta Djallon)
        d["orographic_index"] = d["altitude_m"] / (d["distance_cote_km"] + 15.0)

    # 6. Interaction features
    if "humidity" in d.columns and "cloud_cover" in d.columns:
        d["moisture_cloud_index"] = (d["humidity"] / 100.0) * (d["cloud_cover"] / 100.0)

    return d


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal lags and rolling aggregations per station."""
    d = df.copy()

    # Ensure sorted by location and timestamp
    d = d.sort_values(["location", "timestamp"]).reset_index(drop=True)

    # Group by location
    grouped = d.groupby("location")

    # Time delta to previous record (to verify regularity)
    d["time_delta_hours"] = grouped["timestamp"].diff().dt.total_seconds() / 3600.0

    # Lags for rain_3h, pressure, temperature, humidity
    for col in ["rain_3h", "pressure", "temperature", "humidity"]:
        if col in d.columns:
            d[f"{col}_lag1"] = grouped[col].shift(1)
            d[f"{col}_lag2"] = grouped[col].shift(2)

            # Differences (e.g. pressure drop indicates storm front)
            d[f"{col}_diff1"] = d[col] - d[f"{col}_lag1"]
            d[f"{col}_diff2"] = d[col] - d[f"{col}_lag2"]

    # Rolling statistics over past records (3h intervals: 2 steps ~ 6h, 4 steps ~ 12h, 8 steps ~ 24h)
    if "rain_3h" in d.columns:
        # Closed='left' to prevent leak of current row in rolling window
        d["rain_roll_sum_6h"] = grouped["rain_3h"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).sum())
        d["rain_roll_sum_12h"] = grouped["rain_3h"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
        d["rain_roll_sum_24h"] = grouped["rain_3h"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).sum())
        d["rain_roll_max_24h"] = grouped["rain_3h"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).max())

    if "pressure" in d.columns:
        d["pressure_roll_mean_12h"] = grouped["pressure"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        d["pressure_anomaly"] = d["pressure"] - d["pressure_roll_mean_12h"]

    if "temperature" in d.columns:
        d["temp_roll_mean_24h"] = grouped["temperature"].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
        d["temp_anomaly"] = d["temperature"] - d["temp_roll_mean_24h"]

    return d


def extract_features(df: pd.DataFrame, categorical_as_category: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Complete feature pipeline from cleaned dataframe."""
    d = add_domain_features(df)
    d = add_lag_and_rolling_features(d)

    # Categorical columns
    cat_cols = ["location", "main_condition", "guinea_season", "temp_category", "region"]
    for col in cat_cols:
        if col in d.columns:
            if categorical_as_category:
                d[col] = d[col].astype("category")
            else:
                d[col] = d[col].astype(str)

    # Exclude non-feature columns
    exclude_cols = [
        "ID", "ingestion_timestamp", "timestamp", "target", "cultures_principales",
        "location_raw", "year", "is_test"
    ]
    feature_cols = [c for c in d.columns if c not in exclude_cols]

    return d, feature_cols
