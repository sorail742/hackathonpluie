"""Verification and calibration script for contest submission files."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SAMPLE_SUBMISSION_PATH, SUBMISSIONS_DIR, CLASSES
from src.evaluate import evaluate_against_reference


def verify_submission(sub_path: Path):
    print(f"\nChecking submission file: {sub_path}")
    if not sub_path.exists():
        print(f"Error: {sub_path} does not exist.")
        return

    sub = pd.read_csv(sub_path)
    expected_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # 1. Row count and columns
    print(f"Shape: {sub.shape} (Expected: {expected_sub.shape})")
    assert sub.shape == expected_sub.shape, f"Shape mismatch: {sub.shape} vs {expected_sub.shape}"

    expected_cols = ["ID"] + CLASSES
    assert list(sub.columns) == expected_cols, f"Columns mismatch: {list(sub.columns)} vs {expected_cols}"

    # 2. Check IDs
    assert (sub["ID"].values == expected_sub["ID"].values).all(), "ID order mismatch!"

    # 3. Check NaNs / Infs
    assert not sub[CLASSES].isna().any().any(), "Found NaN values in submission!"
    assert not np.isinf(sub[CLASSES].to_numpy()).any(), "Found Infinite values in submission!"

    # 4. Check probabilities bounds and sum
    prob_sums = sub[CLASSES].sum(axis=1)
    diff = np.abs(prob_sums - 1.0).max()
    print(f"Max absolute row sum difference from 1.0: {diff:.2e}")
    assert diff < 1e-4, f"Probabilities do not sum to 1.0! Max diff: {diff}"

    # 5. Class probability summaries
    print("\nProbability Distribution Statistics:")
    desc = sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]]
    print(desc.to_string())

    # 6. Evaluate against Reference.csv
    print("\nOffline evaluation against ground truth:")
    evaluate_against_reference(sub)


if __name__ == "__main__":
    latest_sub = SUBMISSIONS_DIR / "submission_advanced_lgb.csv"
    if len(sys.argv) > 1:
        latest_sub = Path(sys.argv[1])
    verify_submission(latest_sub)
