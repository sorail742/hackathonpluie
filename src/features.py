"""Strictly causal feature engineering module with meteorological, thermodynamic, and domain knowledge.
Neutralized against adversarial calendar drift and optimized for monsoon/transition dynamics.
"""

import logging
import numpy as np
import pandas as pd

from src.stations_metadata import get_stations_df

logger = logging.getLogger(__name__)


def compute_dew_point_magnus(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Compute dew point temperature (°C) using the Magnus-Tetens approximation."""
    a = 17.27
    b = 237.7
    rh_safe = np.clip(rh, 1.0, 100.0)
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_safe / 100.0)
    td = (b * alpha) / (a - alpha)
    return td


def compute_vapor_pressure_deficit(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Compute Vapor Pressure Deficit (kPa)."""
    # Saturation vapor pressure es (kPa)
    es = 0.61078 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual vapor pressure ea (kPa)
    ea = es * (np.clip(rh, 0.0, 100.0) / 100.0)
    vpd = np.clip(es - ea, 0.0, None)
    return vpd


def add_instantaneous_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable at instant t without history, strictly calendar-drift-free."""
    d = df.copy()

    # 1. Cyclical Time Representations (Lisse, sans empreinte calendariste d'année)
    hour = d["timestamp"].dt.hour
    month = d["timestamp"].dt.month
    dayofyear = d["timestamp"].dt.dayofyear

    # Smooth periodic harmonics
    d["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    d["doy_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25)
    d["doy_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25)
    d["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    # Inferred season from calendar (May to October is rainy season in Guinea)
    d["is_monsoon_season"] = ((month >= 5) & (month <= 10)).astype(np.float32)
    # Transition months indicator (May = onset, November = retreat)
    d["is_transition_season"] = ((month == 5) | (month == 11)).astype(np.float32)

    # 2. Thermodynamics & Moisture
    if "temperature" in d.columns and "humidity" in d.columns:
        d["dew_point"] = compute_dew_point_magnus(d["temperature"], d["humidity"])
        d["dew_point_depression"] = d["temperature"] - d["dew_point"]
        d["vpd"] = compute_vapor_pressure_deficit(d["temperature"], d["humidity"])

        # Near saturation flag (RH close to 100%, high condensation probability)
        d["is_near_saturation"] = (d["dew_point_depression"] < 2.5).astype(np.float32)
        d["rh_gt_85"] = (d["humidity"] > 85.0).astype(np.float32)

        # Specific humidity proxy q ~ 0.622 * ea / P
        if "pressure" in d.columns:
            # Saturation vapor pressure in hPa
            es_hpa = 6.1078 * np.exp((17.27 * d["temperature"]) / (d["temperature"] + 237.3))
            ea_hpa = es_hpa * (np.clip(d["humidity"], 1.0, 100.0) / 100.0)
            p_safe = np.clip(d["pressure"], 800.0, 1050.0)
            d["specific_humidity_proxy"] = 0.622 * ea_hpa / (p_safe - 0.378 * ea_hpa)

    if "heat_index" in d.columns and "temperature" in d.columns:
        d["heat_temp_diff"] = d["heat_index"] - d["temperature"]

    # 3. Wind Components & Guinean Wind Regimes
    if "wind_speed" in d.columns and "wind_deg" in d.columns:
        rad = np.radians(d["wind_deg"])
        d["wind_u"] = -d["wind_speed"] * np.sin(rad)
        d["wind_v"] = -d["wind_speed"] * np.cos(rad)

        # Harmattan projection (NE ~ 45 deg, dry continental Saharan airflow)
        d["harmattan_proj"] = d["wind_speed"] * np.cos(np.radians(d["wind_deg"] - 45.0))
        # Monsoon projection (SW ~ 225 deg, warm humid oceanic airflow from Gulf of Guinea)
        d["monsoon_proj"] = d["wind_speed"] * np.cos(np.radians(d["wind_deg"] - 225.0))

        # Moisture flux proxies
        if "humidity" in d.columns:
            d["moisture_flux"] = d["wind_speed"] * (d["humidity"] / 100.0)
            d["monsoon_moisture_flux"] = np.clip(d["monsoon_proj"], 0.0, None) * (d["humidity"] / 100.0)

    # 4. Station Geography & Climatology
    stations_df = get_stations_df()
    d = d.merge(stations_df, on="location", how="left")

    # Orographic Relief Index (Fouta Djallon uplift)
    if "altitude_m" in d.columns and "distance_cote_km" in d.columns:
        d["orographic_index"] = d["altitude_m"] / (d["distance_cote_km"] + 15.0)

    # Front Inter-Tropical (FIT / ITCZ) latitudinal migration proxy
    d["fit_latitude_est"] = 14.0 + 6.0 * np.sin(2.0 * np.pi * (dayofyear - 110) / 365.25)
    if "lat" in d.columns:
        d["fit_lat_diff"] = d["fit_latitude_est"] - d["lat"]
        d["is_south_of_fit"] = (d["fit_lat_diff"] > 0).astype(np.float32)

    # Moisture & cloud interaction
    if "humidity" in d.columns and "cloud_cover" in d.columns:
        d["moisture_cloud_index"] = (d["humidity"] / 100.0) * (d["cloud_cover"] / 100.0)
        d["heavy_overcast"] = (d["cloud_cover"] > 80.0).astype(np.float32)

    return d


def add_strictly_causal_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add strictly causal history features using closed='left' on timestamp windows."""
    d = df.copy()

    # Sort causally by location and timestamp
    d = d.sort_values(["location", "timestamp"]).reset_index(drop=True)

    grouped = d.groupby("location")

    # Time delta to previous record in hours
    d["time_delta_hours"] = grouped["timestamp"].diff().dt.total_seconds() / 3600.0
    d["is_gap_gt_6h"] = (d["time_delta_hours"] > 6.0).astype(np.float32)

    # Immediate discrete shifts (t-1, t-2)
    for col in ["rain_3h", "pressure", "temperature", "humidity", "cloud_cover"]:
        if col in d.columns:
            d[f"{col}_lag1"] = grouped[col].shift(1)
            d[f"{col}_diff1"] = d[col] - d[f"{col}_lag1"]

    # Wind direction change (wind shear / rotation)
    if "wind_deg" in d.columns:
        wind_deg_lag1 = grouped["wind_deg"].shift(1)
        # Circular angular difference [-180, 180]
        ang_diff = (d["wind_deg"] - wind_deg_lag1 + 180.0) % 360.0 - 180.0
        d["wind_dir_change_3h"] = np.abs(ang_diff)

    # Strictly causal time-window rollings (closed='left' guarantees only past timestamps are included)
    time_windows = ["6h", "12h", "24h", "48h", "72h"]

    for win in time_windows:
        d[f"rain_roll_sum_{win}"] = grouped.rolling(win, on="timestamp", closed="left")["rain_3h"].sum().values

    # Rolling rainfall max over 24h
    d["rain_roll_max_24h"] = grouped.rolling("24h", on="timestamp", closed="left")["rain_3h"].max().values

    # Rainfall acceleration ratio: rate in past 6h relative to 24h baseline
    # High acceleration (> 1.0) indicates onset of heavy precipitation
    d["rain_acceleration"] = (d["rain_roll_sum_6h"] * 4.0) / (d["rain_roll_sum_24h"] + 1.0)

    # Count of rain events (rain_3h >= 0.5mm) in the past 24h
    is_rainy_col = (d["rain_3h"] >= 0.5).astype(np.float32)
    d["_temp_rain_event"] = is_rainy_col
    d["rain_event_count_24h"] = d.groupby("location").rolling("24h", on="timestamp", closed="left")["_temp_rain_event"].sum().values
    d = d.drop(columns=["_temp_rain_event"])

    # Pressure rollings & trends
    d["pressure_roll_mean_12h"] = grouped.rolling("12h", on="timestamp", closed="left")["pressure"].mean().values
    d["pressure_roll_mean_24h"] = grouped.rolling("24h", on="timestamp", closed="left")["pressure"].mean().values
    d["pressure_diff_12h"] = d["pressure"] - d["pressure_roll_mean_12h"]
    d["pressure_diff_24h"] = d["pressure"] - d["pressure_roll_mean_24h"]
    d["rapid_pressure_drop"] = (d["pressure_diff_12h"] < -1.5).astype(np.float32)

    # Temperature diurnal amplitude over 24h (max - min in past 24h)
    temp_max_24h = grouped.rolling("24h", on="timestamp", closed="left")["temperature"].max().values
    temp_min_24h = grouped.rolling("24h", on="timestamp", closed="left")["temperature"].min().values
    diurnal_raw = temp_max_24h - temp_min_24h
    # Fill gap NaNs with median/0 to prevent missingness drift
    d["temp_diurnal_range_24h"] = np.nan_to_num(diurnal_raw, nan=0.0)

    # Cloud cover rolling
    d["cloud_roll_mean_24h"] = grouped.rolling("24h", on="timestamp", closed="left")["cloud_cover"].mean().values
    d["cloud_diff_24h"] = d["cloud_cover"] - d["cloud_roll_mean_24h"]

    # Squall line signature (convective storm front)
    # Characterized by sudden pressure drop, temperature plunge, and high humidity
    d["squall_line_indicator"] = (
        (d["pressure_diff_12h"] < -1.5) &
        (d["humidity"] > 80.0) &
        (d["cloud_cover"] > 70.0)
    ).astype(np.float32)

    return d


def add_regional_causal_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal regional weather context so unseen cities inherit neighbors' weather."""
    d = df.copy()

    # Group by region and rolling over time across all stations in the region
    d = d.sort_values(["region", "timestamp"]).reset_index(drop=True)
    grouped_reg = d.groupby("region")

    d["region_rain_sum_24h"] = grouped_reg.rolling("24h", on="timestamp", closed="left")["rain_3h"].mean().values
    d["region_pressure_mean_24h"] = grouped_reg.rolling("24h", on="timestamp", closed="left")["pressure"].mean().values

    # Resort back by location and timestamp
    d = d.sort_values(["location", "timestamp"]).reset_index(drop=True)
    return d


def build_feature_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_raw_location: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build complete, strictly causal feature matrix for train and test."""
    logger.info("Building feature pipeline (Train: %d, Test: %d)...", len(train_df), len(test_df))

    # Mark dataset origin
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_is_test"] = 0
    test_df["_is_test"] = 1

    # Concatenate chronologically per city for continuous causal history
    combined = pd.concat([train_df, test_df], ignore_index=True)

    # 1. Instantaneous features
    combined = add_instantaneous_features(combined)

    # 2. Strictly causal rolling features per station
    combined = add_strictly_causal_rolling_features(combined)

    # 3. Regional causal context (neighbors)
    combined = add_regional_causal_context(combined)

    # 3b. Synoptic National Simultaneous Context (Mesoscale convective systems)
    combined["_ts_round"] = combined["timestamp"].dt.round("3h")
    nat = combined.groupby("_ts_round").agg(
        national_rain_mean=("rain_3h", "mean"),
        national_rain_frac=("rain_3h", lambda x: (x > 0).mean()),
        national_cloud_mean=("cloud_cover", "mean"),
        national_humidity_mean=("humidity", "mean")
    ).reset_index()
    combined = combined.merge(nat, on="_ts_round", how="left")
    combined["cloud_diff_vs_nat"] = combined["cloud_cover"] - combined["national_cloud_mean"]
    combined["humidity_diff_vs_nat"] = combined["humidity"] - combined["national_humidity_mean"]
    combined["rain_log1p"] = np.log1p(np.clip(combined["rain_3h"], 0.0, None))
    combined = combined.drop(columns=["_ts_round"])

    # 4. Categorical variables formatting
    cat_cols = ["main_condition", "guinea_season", "temp_category", "region"]
    if include_raw_location:
        cat_cols.append("location")

    for col in cat_cols:
        if col in combined.columns:
            combined[col] = combined[col].astype("category")

    # Exclude non-feature columns, including raw calendar fingerprints and drifting columns
    non_feature_cols = [
        "ID", "location_raw", "ingestion_timestamp", "timestamp", "year",
        "dayofweek", "dayofyear", "month", "hour",  # Exclude raw integer calendar indices to prevent year memorization
        "region_pressure_mean_24h", "temp_diurnal_range_24h",  # Exclude drifting features identified in adversarial validation
        "target", "cultures_principales", "_is_test"
    ]
    if not include_raw_location:
        non_feature_cols.append("location")

    feature_cols = [c for c in combined.columns if c not in non_feature_cols]

    # Split back cleanly
    train_feat = combined[combined["_is_test"] == 0].drop(columns=["_is_test"]).reset_index(drop=True)
    test_feat = combined[combined["_is_test"] == 1].drop(columns=["_is_test"]).reset_index(drop=True)

    logger.info("Engineered %d features cleanly (drift-neutralized).", len(feature_cols))
    return train_feat, test_feat, feature_cols
