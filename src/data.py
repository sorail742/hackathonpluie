"""Data loading, normalization, schema verification, and merging."""

import logging
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    SAMPLE_SUBMISSION_PATH,
    CANONICAL_TRAIN_CITIES,
    CANONICAL_TEST_UNSEEN_CITIES,
    CANONICAL_CITIES,
    CITY_ALIAS_MAP,
    STATIONS_FILE,
    REGIONAL_DAILY_FILE,
    CLIMAT_NATIONAL_FILE,
    USE_EXTRA_FILES,
)

logger = logging.getLogger(__name__)


def normalize_city_name(raw_name: str) -> str:
    """Normalize raw city name to canonical Guinean city name."""
    if not isinstance(raw_name, str):
        return "Conakry"

    cleaned = raw_name.strip()
    key = cleaned.lower()

    if key in CITY_ALIAS_MAP:
        return CITY_ALIAS_MAP[key]

    # Normalize unicode accents and check again
    norm_key = unicodedata.normalize("NFKD", key).encode("ASCII", "ignore").decode("utf-8")
    if norm_key in CITY_ALIAS_MAP:
        return CITY_ALIAS_MAP[norm_key]

    for canonical in CANONICAL_CITIES:
        c_ascii = unicodedata.normalize("NFKD", canonical.lower()).encode("ASCII", "ignore").decode("utf-8")
        if norm_key == c_ascii:
            return canonical

    return cleaned


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission files."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"train.csv not found at {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"test.csv not found at {TEST_PATH}")
    if not SAMPLE_SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"sample_submission.csv not found at {SAMPLE_SUBMISSION_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sub_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    logger.info("Loaded raw train: %s, test: %s, sample_submission: %s", train_df.shape, test_df.shape, sub_df.shape)
    return train_df, test_df, sub_df


def prepare_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, normalize city names, verify schema and sort chronologically per city."""
    train_df, test_df, sub_df = load_raw_data()

    # 1. Normalize city names
    train_df["location_raw"] = train_df["location"]
    train_df["location"] = train_df["location"].apply(normalize_city_name)

    test_df["location_raw"] = test_df["location"]
    test_df["location"] = test_df["location"].apply(normalize_city_name)

    # 2. Check canonical city sets
    train_cities = set(train_df["location"].unique())
    test_cities = set(test_df["location"].unique())

    assert train_cities == set(CANONICAL_TRAIN_CITIES), (
        f"Train cities count mismatch: expected 18, got {len(train_cities)}: {train_cities}"
    )
    assert test_cities == set(CANONICAL_CITIES), (
        f"Test cities count mismatch: expected 20, got {len(test_cities)}: {test_cities}"
    )
    assert test_cities - train_cities == set(CANONICAL_TEST_UNSEEN_CITIES), (
        f"Expected unseen test cities {CANONICAL_TEST_UNSEEN_CITIES}, got {test_cities - train_cities}"
    )

    # 3. Parse timestamps (UTC = Guinea GMT)
    train_df["timestamp"] = pd.to_datetime(train_df["ingestion_timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["ingestion_timestamp"])

    train_df["year"] = train_df["timestamp"].dt.year
    test_df["year"] = test_df["timestamp"].dt.year

    # 4. Sort causally per city
    train_df = train_df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["location", "timestamp"]).reset_index(drop=True)

    # 5. External files status
    if USE_EXTRA_FILES:
        logger.info("Extra files found! Merging external climatological indicators.")
    else:
        logger.info(
            "Extra files (stations.csv, regional_daily.csv, climat_national.csv) are not present. "
            "Pipeline is running with built-in domain geographical metadata and proxies."
        )

    return train_df, test_df, sub_df
