"""Train a clean baseline LightGBM model matching the starter notebook."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH, SUBMISSIONS_DIR, TARGET_MAP, CLASSES
from src.data_cleaning import clean_dataframe
from src.models import train_lgb_model, predict_proba
from src.evaluate import evaluate_predictions, evaluate_against_reference


def run_baseline():
    print("Loading raw data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Clean data
    train_clean = clean_dataframe(train)
    test_clean = clean_dataframe(test)

    features = [
        "temperature", "heat_index", "comfort_index", "humidity", "pressure",
        "wind_speed", "wind_deg", "cloud_cover", "rain_3h", "hour", "month"
    ]

    for col in ["main_condition"]:
        train_clean[col] = train_clean[col].astype("category")
        test_clean[col] = test_clean[col].astype("category")
        features.append(col)

    X = train_clean[features]
    y = train_clean["target"].map(TARGET_MAP).to_numpy()

    # Time split: train 2022-2023, validate 2024
    val_mask = train_clean["year"] == 2024
    X_tr, y_tr = X[~val_mask], y[~val_mask]
    X_va, y_va = X[val_mask], y[val_mask]

    print(f"\nTraining baseline LightGBM (Train: {len(X_tr)} rows, Val 2024: {len(X_va)} rows)...")
    baseline_params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": 42
    }

    model = train_lgb_model(
        X_tr, y_tr, X_va, y_va,
        params=baseline_params,
        num_boost_round=300,
        early_stopping_rounds=30,
        verbose_eval=50
    )

    y_val_proba = predict_proba(model, X_va)
    evaluate_predictions(y_va, y_val_proba, title="Baseline Validation (Year 2024)")

    # Full train and test prediction
    print("\nTraining on full train set...")
    full_model = train_lgb_model(
        X, y,
        params=baseline_params,
        num_boost_round=300,
        verbose_eval=0
    )

    to_predict = test_clean[test_clean["ID"].isin(sub["ID"])].copy()
    test_proba = predict_proba(full_model, to_predict[features])

    submission = pd.DataFrame(test_proba, columns=CLASSES)
    submission.insert(0, "ID", to_predict["ID"].values)
    submission = sub[["ID"]].merge(submission, on="ID", how="left")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "submission_baseline.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSaved baseline submission to {out_path}")

    # Evaluate against Reference
    evaluate_against_reference(submission)


if __name__ == "__main__":
    run_baseline()
