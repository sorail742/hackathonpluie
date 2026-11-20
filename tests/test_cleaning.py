"""Unit tests for data cleaning, city normalization, and sensor validation."""

import numpy as np
import pandas as pd
import pytest

from src.config import CANONICAL_TRAIN_CITIES, CANONICAL_TEST_UNSEEN_CITIES
from src.data import normalize_city_name, prepare_datasets
from src.cleaning import clean_weather_data


def test_city_normalization():
    assert normalize_city_name("conakry") == "Conakry"
    assert normalize_city_name("Conakry ") == "Conakry"
    assert normalize_city_name("Boke") == "Boké"
    assert normalize_city_name("Boké") == "Boké"
    assert normalize_city_name("NZEREKORE") == "Nzérékoré"
    assert normalize_city_name("N'Zérékoré") == "Nzérékoré"
    assert normalize_city_name("Kissi") == "Kissidougou"
    assert normalize_city_name("Gueckedou") == "Guéckédou"
    assert normalize_city_name("LABE") == "Labé"
    assert normalize_city_name("Siguiri") == "Siguiri"


def test_train_test_city_counts():
    train_df, test_df, _ = prepare_datasets()
    assert set(train_df["location"].unique()) == set(CANONICAL_TRAIN_CITIES)
    assert len(train_df["location"].unique()) == 18

    test_unseen = set(test_df["location"].unique()) - set(train_df["location"].unique())
    assert test_unseen == set(CANONICAL_TEST_UNSEEN_CITIES)


def test_fahrenheit_and_sentinel_cleaning():
    sample = pd.DataFrame({
        "location": ["Conakry", "Kindia", "Mamou", "Labé"],
        "timestamp": pd.to_datetime(["2022-01-01 00:00", "2022-01-01 03:00", "2022-01-01 06:00", "2022-01-01 09:00"]),
        "temperature": [77.0, 25.0, 30.0, 86.0],  # 77°F = 25°C, 86°F = 30°C
        "heat_index": [77.0, 25.0, 30.0, 86.0],
        "pressure": [-999.0, 1012.0, 300.0, 950.0],
        "humidity": [0.0, 75.0, 80.0, 60.0],
        "cloud_cover": [50.0, 120.0, 0.0, 40.0],
        "wind_speed": [5.0, 2.0, 10.0, -5.0],
        "wind_deg": [90.0, 180.0, 270.0, 400.0],
        "rain_3h": [0.0, 10.0, -1.0, 5.0],
    })

    cleaned, report = clean_weather_data(sample, verbose=False)

    assert report.fahrenheit_temp_corrected == 2
    assert np.isclose(cleaned.loc[0, "temperature"], 25.0)
    assert np.isclose(cleaned.loc[3, "temperature"], 30.0)

    # Sentinels
    assert np.isnan(cleaned.loc[0, "pressure"])
    assert np.isnan(cleaned.loc[2, "pressure"])
    assert np.isnan(cleaned.loc[0, "humidity"])

    # Bounds
    assert cleaned.loc[1, "cloud_cover"] == 100.0
    assert cleaned.loc[3, "wind_speed"] == 0.0
    assert cleaned.loc[3, "wind_deg"] == 40.0
    assert cleaned.loc[2, "rain_3h"] == 0.0
