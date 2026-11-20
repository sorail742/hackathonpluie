"""CLI prediction and submission generator from full training data."""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    SEEDS,
    CLASSES,
    TARGET_MAP,
    SAMPLE_SUBMISSION_PATH,
    SUBMISSIONS_DIR,
)
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.models import get_regularized_lgb_params, train_single_lgb, predict_proba_lgb
from src.calibration import apply_temperature_scaling, safe_clip_and_renormalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full training and test submission generation.")
    parser.add_argument("--num_boost_round", type=int, default=500, help="Boosting iterations on full train data.")
    parser.add_argument("--temperature", type=float, default=1.005, help="Calibration temperature.")
    parser.add_argument("--name", type=str, default="submission_final_ensemble", help="Output file basename.")
    args = parser.parse_args()

    logger.info("Preparing data for final model training on full train dataset...")
    train_raw, test_raw, sub_raw = prepare_datasets()

    train_clean, _ = clean_weather_data(train_raw)
    test_clean, _ = clean_weather_data(test_raw)

    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols]
    y = train_feat["target"].map(TARGET_MAP).to_numpy()

    to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols]

    logger.info("Training multi-seed ensemble on full train data (%d rows, %d features)...", len(X), len(feature_cols))

    test_preds_total = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)

    for seed in SEEDS:
        logger.info("Training seed %d...", seed)
        params = get_regularized_lgb_params(seed=seed)
        booster = train_single_lgb(
            X, y,
            params=params,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=0,
            verbose_eval=150,
        )
        preds = predict_proba_lgb(booster, X_test)
        test_preds_total += preds / len(SEEDS)

    logger.info("Applying temperature calibration (T=%.4f) and safe clipping...", args.temperature)
    calibrated_preds = apply_temperature_scaling(test_preds_total, temperature=args.temperature)

    # Build submission DataFrame
    sub = pd.DataFrame(calibrated_preds, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    # Verify integrity
    assert len(final_sub) == 1820, f"Expected 1820 rows, got {len(final_sub)}"
    assert list(final_sub.columns) == ["ID"] + CLASSES
    assert (final_sub["ID"].values == sub_raw["ID"].values).all(), "ID alignment mismatch!"
    assert not final_sub[CLASSES].isna().any().any(), "Detected NaNs in probabilities!"
    assert not np.isinf(final_sub[CLASSES].to_numpy()).any(), "Detected Infs in probabilities!"

    diff = np.abs(final_sub[CLASSES].sum(axis=1) - 1.0).max()
    assert diff < 1e-4, f"Probabilities do not sum to 1.0 (max diff: {diff:.2e})"

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SUBMISSIONS_DIR / f"{args.name}.csv"
    final_sub.to_csv(out_file, index=False)
    logger.info("Successfully generated verified competition submission: %s", out_file)
    print("\nSubmission Summary Statistics:")
    print(final_sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
