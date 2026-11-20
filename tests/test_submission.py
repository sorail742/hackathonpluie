"""Unit tests ensuring submission files meet strict competition guidelines."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import SAMPLE_SUBMISSION_PATH, CLASSES, SUBMISSIONS_DIR


def test_sample_submission_integrity():
    sub_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    assert len(sub_df) == 1820
    assert list(sub_df.columns) == ["ID"] + CLASSES


def test_generated_submission_format():
    # If a submission file exists in submissions/, verify its compliance
    sub_files = list(SUBMISSIONS_DIR.glob("*.csv"))
    if not sub_files:
        pytest.skip("No submission file generated yet.")

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    for sub_path in sub_files:
        sub = pd.read_csv(sub_path)
        assert len(sub) == 1820, f"Expected 1820 rows, got {len(sub)} in {sub_path.name}"
        assert list(sub.columns) == ["ID"] + CLASSES, f"Incorrect columns in {sub_path.name}"
        assert (sub["ID"].values == sample_sub["ID"].values).all(), f"ID mismatch or reordering in {sub_path.name}"
        assert not sub[CLASSES].isna().any().any(), f"NaNs detected in {sub_path.name}"
        assert not np.isinf(sub[CLASSES].to_numpy()).any(), f"Infs detected in {sub_path.name}"

        # Sum per row
        row_sums = sub[CLASSES].sum(axis=1)
        max_diff = float(np.abs(row_sums - 1.0).max())
        assert max_diff < 1e-4, f"Probabilities do not sum to 1 (max diff={max_diff:.2e}) in {sub_path.name}"
