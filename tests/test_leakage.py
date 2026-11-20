"""Unit test guaranteeing strict temporal non-leakage in feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import add_strictly_causal_rolling_features


def test_no_future_leakage():
    """Verify that altering future records does not change features at current and past records."""
    base_data = pd.DataFrame({
        "location": ["Conakry"] * 5,
        "timestamp": pd.to_datetime([
            "2022-01-01 00:00:00",
            "2022-01-01 03:00:00",
            "2022-01-01 06:00:00",
            "2022-01-01 09:00:00",
            "2022-01-01 12:00:00",
        ]),
        "rain_3h": [0.0, 5.0, 10.0, 0.0, 20.0],
        "pressure": [1013.0, 1012.0, 1010.0, 1008.0, 1005.0],
        "temperature": [28.0, 27.0, 26.0, 25.0, 24.0],
        "humidity": [70.0, 75.0, 80.0, 85.0, 90.0],
        "cloud_cover": [40.0, 50.0, 60.0, 70.0, 80.0],
    })

    # Scenario A: Base data
    feat_a = add_strictly_causal_rolling_features(base_data)

    # Scenario B: Future at t=4 (12:00) is drastically modified (e.g. huge rainfall spike 100mm, pressure crash)
    perturbed_data = base_data.copy()
    perturbed_data.loc[4, "rain_3h"] = 150.0
    perturbed_data.loc[4, "pressure"] = 950.0
    perturbed_data.loc[4, "temperature"] = 15.0

    feat_b = add_strictly_causal_rolling_features(perturbed_data)

    # Verify that features at rows 0, 1, 2, 3 are 100% IDENTICAL between A and B!
    check_cols = [c for c in feat_a.columns if c not in ["location", "timestamp"]]
    for col in check_cols:
        val_a = feat_a.loc[:3, col].to_numpy()
        val_b = feat_b.loc[:3, col].to_numpy()
        # Handle NaNs
        nan_mask = np.isnan(val_a)
        assert (nan_mask == np.isnan(val_b)).all(), f"NaN mask mismatch in column {col}"
        assert np.allclose(val_a[~nan_mask], val_b[~nan_mask]), f"Future leakage detected in feature: {col}!"
