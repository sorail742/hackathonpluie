"""Train Final Production Model: Synoptic Multiclass + Ordinal LightGBM Blend.
Incorporates simultaneous national mesoscale convective features and temperature calibration T=1.025.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
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
    logger.info("=== Training Final Synoptic Master Ensemble (MC + Ordinal, T=1.025) ===")

    # 1. Load data
    train_raw, test_raw, sub_raw = prepare_datasets()
    train_clean, _ = clean_weather_data(train_raw, verbose=False)
    test_clean, _ = clean_weather_data(test_raw, verbose=False)

    # 2. Features with synoptic national convective variables
    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols]
    y = train_feat["target"].map(TARGET_MAP).to_numpy()

    to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols]

    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    logger.info("Features used (%d features, %d categoricals): %s", len(feature_cols), len(cat_cols), feature_cols)

    # 3. Multiclass LightGBM across 3 seeds (450 rounds)
    p_test_mc = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    logger.info("1/2 Training Multiclass models across %d seeds...", len(SEEDS))

    for s in SEEDS:
        logger.info("  Multiclass Seed %d...", s)
        params = get_regularized_lgb_params(seed=s)
        params["n_jobs"] = -1
        booster = train_single_lgb(
            X, y,
            params=params,
            num_boost_round=450,
            early_stopping_rounds=0,
            verbose_eval=0,
        )
        p_test_mc += predict_proba_lgb(booster, X_test) / len(SEEDS)

    # 4. Ordinal LightGBM across 2 seeds (T1, T2, T3) (400 rounds)
    p_test_ord = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    ord_seeds = [42, 1337]
    logger.info("2/2 Training Ordinal models across %d seeds...", len(ord_seeds))

    for s in ord_seeds:
        p_cum = np.zeros((len(to_predict), 3), dtype=np.float64)
        for t_idx, threshold in enumerate([1, 2, 3]):
            y_bin = (y >= threshold).astype(int)
            dtr = lgb.Dataset(X, label=y_bin, categorical_feature=cat_cols)
            params_bin = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.03,
                "num_leaves": 35,
                "max_depth": 6,
                "min_child_samples": 50,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.2,
                "reg_lambda": 2.0,
                "verbose": -1,
                "random_state": s + threshold,
                "n_jobs": -1,
            }
            clf = lgb.train(params_bin, dtr, num_boost_round=400)
            p_cum[:, t_idx] = clf.predict(X_test)

        p1 = p_cum[:, 0]
        p2 = np.minimum(p1, p_cum[:, 1])
        p3 = np.minimum(p2, p_cum[:, 2])

        p_4c = np.column_stack([
            np.clip(1.0 - p1, 0.0, 1.0),
            np.clip(p1 - p2, 0.0, 1.0),
            np.clip(p2 - p3, 0.0, 1.0),
            np.clip(p3, 0.0, 1.0),
        ])
        p_test_ord += safe_clip_and_renormalize(p_4c) / len(ord_seeds)

    # 5. Optimal Blend: 55% Multiclass + 45% Ordinal
    logger.info("Blending: 55% Multiclass + 45% Ordinal...")
    p_blend = 0.55 * p_test_mc + 0.45 * p_test_ord

    # 6. Temperature scaling (T=1.025) & safe clipping [1e-4, 1 - 1e-4]
    logger.info("Applying Temperature Scaling (T=1.025) and safe clipping...")
    p_calibrated = apply_temperature_scaling(p_blend, temperature=1.025)

    # 7. Build and verify submission
    sub = pd.DataFrame(p_calibrated, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    assert len(final_sub) == 1820, f"Expected 1820 rows, got {len(final_sub)}"
    assert list(final_sub.columns) == ["ID"] + CLASSES
    assert (final_sub["ID"].values == sub_raw["ID"].values).all(), "ID order mismatch!"
    assert not final_sub[CLASSES].isna().any().any(), "Found NaNs!"
    diff = np.abs(final_sub[CLASSES].sum(axis=1) - 1.0).max()
    assert diff < 1e-4, f"Probabilities do not sum to 1! Max diff={diff}"

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / "submission_synoptic_master.csv"
    final_sub.to_csv(out_path, index=False)
    logger.info("Saved verified submission file to: %s", out_path)

    print("\n=== Submission Summary Statistics ===")
    print(final_sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
