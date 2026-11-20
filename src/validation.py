"""Validation strategies designed specifically for temporal and spatial generalization."""

from typing import Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold


def temporal_split(df: pd.DataFrame, train_years: list[int] = [2022, 2023], val_year: int = 2024) -> tuple[np.ndarray, np.ndarray]:
    """Train on past years, validate on the holdout year (mimicking test 2025)."""
    tr_idx = df.index[df["year"].isin(train_years)].to_numpy()
    va_idx = df.index[df["year"] == val_year].to_numpy()
    return tr_idx, va_idx


def unseen_city_split(df: pd.DataFrame, holdout_cities: list[str] = ["Pita", "Kouroussa"]) -> tuple[np.ndarray, np.ndarray]:
    """Hold out representative cities to simulate the test set condition (unseen Labé & Siguiri).
    
    Pita is in Moyenne Guinée (near Labé).
    Kouroussa is in Haute Guinée (near Siguiri).
    """
    va_idx = df.index[df["location"].isin(holdout_cities)].to_numpy()
    tr_idx = df.index[~df["location"].isin(holdout_cities)].to_numpy()
    return tr_idx, va_idx


def get_stratified_time_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Stratified K-Fold on target with shuffling, ensuring balanced classes in each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target = df["target"]
    for tr_idx, va_idx in skf.split(df, target):
        yield tr_idx, va_idx


def get_group_location_folds(
    df: pd.DataFrame,
    n_splits: int = 5
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Group K-Fold by location to evaluate spatial transfer."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = df["location"]
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        yield tr_idx, va_idx
