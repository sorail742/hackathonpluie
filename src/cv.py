"""Validation module implementing temporal, spatial, and spatio-temporal validation schemes."""

from typing import Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold


def temporal_purged_split(
    df: pd.DataFrame,
    train_end_date: str = "2023-12-30 23:59:59",
    val_start_date: str = "2024-01-01 00:00:00"
) -> tuple[np.ndarray, np.ndarray]:
    """Train on 2022-2023, validate on 2024, with a 24-hour purge gap before 2024.
    
    The purge gap ensures that no training 24-hour forward target window overlaps with validation observations.
    """
    ts = pd.to_datetime(df["timestamp"])
    tr_mask = ts <= pd.to_datetime(train_end_date)
    va_mask = ts >= pd.to_datetime(val_start_date)

    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(va_mask)[0]

    return tr_idx, va_idx


def leave_city_group_out_folds(
    df: pd.DataFrame,
    n_splits: int = 4
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Spatial holdout: GroupKFold by canonical city."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["location"]
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        yield tr_idx, va_idx


def spatio_temporal_folds(
    df: pd.DataFrame,
    n_splits: int = 4,
    random_state: int = 42
) -> Generator[tuple[np.ndarray, np.ndarray, list[str]], None, None]:
    """Main benchmark scheme: train on seen cities for years <= 2023,
    validate on held-out unseen cities in year 2024 with a 24h purge.
    """
    ts = pd.to_datetime(df["timestamp"])
    train_pool_mask = ts <= pd.to_datetime("2023-12-30 23:59:59")
    val_pool_mask = ts >= pd.to_datetime("2024-01-01 00:00:00")

    cities = np.array(sorted(df["location"].unique()))
    rng = np.random.default_rng(random_state)
    shuffled_cities = rng.permutation(cities)

    city_chunks = np.array_split(shuffled_cities, n_splits)

    for fold, heldout_cities in enumerate(city_chunks):
        heldout_set = set(heldout_cities)
        # Validation: only heldout cities in 2024
        va_mask = val_pool_mask & df["location"].isin(heldout_set)
        # Train: seen cities in <= 2023
        tr_mask = train_pool_mask & (~df["location"].isin(heldout_set))

        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        yield tr_idx, va_idx, list(heldout_cities)
