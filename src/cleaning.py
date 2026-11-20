"""Cleaning module: sentinel detection, °F to °C conversion, physical sanity bounds, and logging."""

import logging
from typing import NamedTuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CleaningReport(NamedTuple):
    fahrenheit_temp_corrected: int
    fahrenheit_heat_corrected: int
    pressure_sentinels_removed: int
    humidity_zero_sentinels_removed: int
    bounds_clipped: dict[str, int]
    duplicates_removed: int


def clean_weather_data(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean meteorological variables according to physical and sensor rules."""
    d = df.copy()
    init_len = len(d)

    # 1. Deduplication (exact match on location and timestamp)
    dups_mask = d.duplicated(subset=["location", "timestamp"], keep="first")
    n_dups = int(dups_mask.sum())
    if n_dups > 0:
        d = d[~dups_mask].copy().reset_index(drop=True)

    # 2. Fahrenheit to Celsius correction
    # Normal air temperature in Guinea is in [10°C, 45°C].
    # Relevés above 50°C (e.g., 75°F to 105°F) correspond to Fahrenheit sensor stream errors.
    f_temp_mask = d["temperature"] > 50.0
    n_f_temp = int(f_temp_mask.sum())
    if n_f_temp > 0:
        d.loc[f_temp_mask, "temperature"] = (d.loc[f_temp_mask, "temperature"] - 32.0) * (5.0 / 9.0)

    f_heat_mask = d["heat_index"] > 50.0
    n_f_heat = int(f_heat_mask.sum())
    if n_f_heat > 0:
        d.loc[f_heat_mask, "heat_index"] = (d.loc[f_heat_mask, "heat_index"] - 32.0) * (5.0 / 9.0)

    # 3. Pressure sentinels -> NaN
    # Sentinels like -999.0 or values < 500 hPa (impossible in Guinea, even at 1500m elevation)
    press_sentinel_mask = (d["pressure"] == -999.0) | (d["pressure"] < 500.0) | (d["pressure"] > 1100.0)
    n_press_sentinels = int(press_sentinel_mask.sum())
    if n_press_sentinels > 0:
        d.loc[press_sentinel_mask, "pressure"] = np.nan

    # 4. Humidity = 0 suspect -> NaN
    # In tropical Guinea, RH=0% is a sensor failure / disconnection indicator.
    hum_zero_mask = d["humidity"] == 0.0
    n_hum_zeros = int(hum_zero_mask.sum())
    if n_hum_zeros > 0:
        d.loc[hum_zero_mask, "humidity"] = np.nan

    # 5. Physical bounds clipping
    bounds_clipped = {}

    if "humidity" in d.columns:
        hum_clip_mask = (d["humidity"] < 0.0) | (d["humidity"] > 100.0)
        bounds_clipped["humidity"] = int(hum_clip_mask.sum())
        d["humidity"] = d["humidity"].clip(lower=0.0, upper=100.0)

    if "cloud_cover" in d.columns:
        cloud_clip_mask = (d["cloud_cover"] < 0.0) | (d["cloud_cover"] > 100.0)
        bounds_clipped["cloud_cover"] = int(cloud_clip_mask.sum())
        d["cloud_cover"] = d["cloud_cover"].clip(lower=0.0, upper=100.0)

    if "wind_speed" in d.columns:
        wind_clip_mask = (d["wind_speed"] < 0.0) | (d["wind_speed"] > 60.0)
        bounds_clipped["wind_speed"] = int(wind_clip_mask.sum())
        d["wind_speed"] = d["wind_speed"].clip(lower=0.0, upper=60.0)

    if "wind_deg" in d.columns:
        d["wind_deg"] = d["wind_deg"] % 360.0

    if "rain_3h" in d.columns:
        rain_neg_mask = d["rain_3h"] < 0.0
        bounds_clipped["rain_3h"] = int(rain_neg_mask.sum())
        d["rain_3h"] = d["rain_3h"].clip(lower=0.0, upper=500.0)

    # 6. Sensor missingness indicators
    for col in ["pressure", "humidity", "temperature", "wind_speed", "rain_3h", "cloud_cover"]:
        if col in d.columns:
            d[f"{col}_isna"] = d[col].isna().astype(np.float32)

    report = CleaningReport(
        fahrenheit_temp_corrected=n_f_temp,
        fahrenheit_heat_corrected=n_f_heat,
        pressure_sentinels_removed=n_press_sentinels,
        humidity_zero_sentinels_removed=n_hum_zeros,
        bounds_clipped=bounds_clipped,
        duplicates_removed=n_dups,
    )

    if verbose:
        logger.info(
            "Cleaning Report (Init rows: %d -> Out rows: %d): "
            "Temp °F corrected: %d | Heat °F corrected: %d | Pressure sentinels: %d | "
            "Humidity zeros -> NaN: %d | Duplicates dropped: %d",
            init_len, len(d), n_f_temp, n_f_heat, n_press_sentinels, n_hum_zeros, n_dups
        )

    return d, report
