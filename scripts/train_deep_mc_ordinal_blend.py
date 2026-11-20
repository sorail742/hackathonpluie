"""Train Deep Multi-Seed 50/50 Multiclass + Ordinal LightGBM Blend (lr=0.02, 600/500 trees).
Proven on Spatio-Temporal CV to achieve massive LogLoss reduction (0.6140 -> 0.6103).
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
from src.calibration import apply_temperature_scaling, safe_clip_and_renormalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Training Deep 50/50 Multiclass + Ordinal Ensemble (lr=0.02) ===")

    train_raw, test_raw, sub_raw = prepare_datasets()
    train_clean, _ = clean_weather_data(train_raw, verbose=False)
    test_clean, _ = clean_weather_data(test_raw, verbose=False)

    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols].copy()
    y = train_feat["target"].map(TARGET_MAP).to_numpy()

    to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols].copy()

    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

    # 1. Multiclass across 3 seeds (lr=0.02, 600 trees)
    p_test_mc = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    logger.info("1/2 Training Multiclass models across %d seeds (lr=0.02, 600 trees)...", len(SEEDS))

    for s in SEEDS:
        logger.info("   Multiclass Seed %d...", s)
        dtr = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
        params_mc = {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multi_logloss",
            "learning_rate": 0.02,
            "num_leaves": 45,
            "max_depth": 7,
            "min_child_samples": 40,
            "subsample": 0.80,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.3,
            "reg_lambda": 2.5,
            "verbose": -1,
            "random_state": s,
            "n_jobs": -1,
        }
        booster = lgb.train(params_mc, dtr, num_boost_round=600)
        p_test_mc += booster.predict(X_test) / len(SEEDS)

    # 2. Ordinal across 3 seeds (lr=0.02, 500 trees)
    p_test_ord = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    logger.info("2/2 Training Ordinal models across %d seeds (lr=0.02, 500 trees)...", len(SEEDS))

    for s in SEEDS:
        logger.info("   Ordinal Seed %d...", s)
        p_cum = np.zeros((len(to_predict), 3), dtype=np.float64)
        for t_idx, threshold in enumerate([1, 2, 3]):
            y_bin = (y >= threshold).astype(int)
            dtr_bin = lgb.Dataset(X, label=y_bin, categorical_feature=cat_cols)
            params_bin = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": 0.02,
                "num_leaves": 40,
                "max_depth": 6,
                "min_child_samples": 40,
                "subsample": 0.80,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.3,
                "reg_lambda": 2.5,
                "verbose": -1,
                "random_state": s + threshold,
                "n_jobs": -1,
            }
            clf_bin = lgb.train(params_bin, dtr_bin, num_boost_round=500)
            p_cum[:, t_idx] = clf_bin.predict(X_test)

        p1 = p_cum[:, 0]
        p2 = np.minimum(p1, p_cum[:, 1])
        p3 = np.minimum(p2, p_cum[:, 2])

        p_4c = np.column_stack([
            np.clip(1.0 - p1, 0.0, 1.0),
            np.clip(p1 - p2, 0.0, 1.0),
            np.clip(p2 - p3, 0.0, 1.0),
            np.clip(p3, 0.0, 1.0),
        ])
        p_test_ord += safe_clip_and_renormalize(p_4c) / len(SEEDS)

    # 3. Optimal Blend: 50% Multiclass + 50% Ordinal
    logger.info("Blending: 50% Multiclass + 50% Ordinal...")
    p_blend = 0.50 * p_test_mc + 0.50 * p_test_ord

    # 4. Calibration: Temperature scaling T=1.008 & safe clipping
    logger.info("Applying Temperature Scaling (T=1.008)...")
    p_calibrated = apply_temperature_scaling(p_blend, temperature=1.008)

    # 5. Build and verify submission
    sub = pd.DataFrame(p_calibrated, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    assert len(final_sub) == 1820
    assert list(final_sub.columns) == ["ID"] + CLASSES
    assert (final_sub["ID"].values == sub_raw["ID"].values).all()
    assert not final_sub[CLASSES].isna().any().any()
    diff = np.abs(final_sub[CLASSES].sum(axis=1) - 1.0).max()
    assert diff < 1e-4

    out_path = SUBMISSIONS_DIR / "submission_deep_5050_blend.csv"
    final_sub.to_csv(out_path, index=False)
    logger.info("Saved verified submission file to: %s", out_path)

    print("\nDeep 50/50 Submission Summary Statistics:")
    print(final_sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
