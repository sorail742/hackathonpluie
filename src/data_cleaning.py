"""Data cleaning and preprocessing for IndabaX Guinea 2026."""

import re
import unicodedata
import pandas as pd
import numpy as np

from src.config import LOCATION_MAPPING, CANONICAL_CITIES


def normalize_string(s: str) -> str:
    """Strip, lower, and normalize accents for fuzzy matching."""
    s = s.strip().lower()
    return s


def normalize_location(raw_loc: str) -> str:
    """Normalize raw station name to one of the 20 canonical Guinean cities."""
    if not isinstance(raw_loc, str):
        return "Conakry"
    cleaned = raw_loc.strip()
    key = cleaned.lower()
    if key in LOCATION_MAPPING:
        return LOCATION_MAPPING[key]
    
    # Fallback: remove non-ascii and match
    norm_key = unicodedata.normalize('NFKD', key).encode('ASCII', 'ignore').decode('utf-8')
    if norm_key in LOCATION_MAPPING:
        return LOCATION_MAPPING[norm_key]
    
    for c in CANONICAL_CITIES:
        c_norm = unicodedata.normalize('NFKD', c.lower()).encode('ASCII', 'ignore').decode('utf-8')
        if norm_key in c_norm or c_norm in norm_key:
            return c

    return cleaned


def clean_dataframe(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Perform primary cleaning and typing on the dataframe."""
    d = df.copy()

    # 1. Normalize location
    d["location_raw"] = d["location"]
    d["location"] = d["location"].apply(normalize_location)

    # 2. Parse timestamps (UTC = Guinea GMT)
    d["timestamp"] = pd.to_datetime(d["ingestion_timestamp"])
    d["year"] = d["timestamp"].dt.year
    d["month"] = d["timestamp"].dt.month
    d["day"] = d["timestamp"].dt.day
    d["hour"] = d["timestamp"].dt.hour
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    d["dayofyear"] = d["timestamp"].dt.dayofyear

    # 3. Handle impossible / physical sensor outliers
    if "humidity" in d.columns:
        d["humidity"] = d["humidity"].clip(lower=0.0, upper=100.0)

    if "cloud_cover" in d.columns:
        d["cloud_cover"] = d["cloud_cover"].clip(lower=0.0, upper=100.0)

    if "wind_speed" in d.columns:
        d["wind_speed"] = d["wind_speed"].clip(lower=0.0, upper=60.0)

    if "wind_deg" in d.columns:
        d["wind_deg"] = d["wind_deg"] % 360.0

    if "rain_3h" in d.columns:
        d["rain_3h"] = d["rain_3h"].clip(lower=0.0, upper=500.0)

    # Missing indicators for critical sensors
    for col in ["pressure", "humidity", "temperature", "wind_speed", "rain_3h", "cloud_cover"]:
        if col in d.columns:
            d[f"{col}_isna"] = d[col].isna().astype(np.float32)

    # Sort temporally per station to allow proper lag/rolling computation
    d = d.sort_values(["location", "timestamp"]).reset_index(drop=True)

    return d
