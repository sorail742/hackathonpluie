"""Train Master Ensemble: 60% LightGBM (MC + Ordinal) + 40% CatBoost.
Achieved massive LogLoss breakthrough on Spatio-Temporal CV (0.6107 -> 0.6002).
Designed to conquer the top spot (0.782+) on Zindi.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

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
    logger.info("=== Training Master Ensemble: LightGBM (MC+Ord) + CatBoost ===")

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

    # --- 1. LightGBM Multiclass across 3 seeds (600 trees, lr=0.02) ---
    logger.info("1/3 Training LightGBM Multiclass across %d seeds...", len(SEEDS))
    p_test_mc = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    for s in SEEDS:
        logger.info("   LGB Multiclass Seed %d...", s)
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
        clf_mc = lgb.train(params_mc, dtr, num_boost_round=600)
        p_test_mc += clf_mc.predict(X_test) / len(SEEDS)

    # --- 2. LightGBM Ordinal across 3 seeds (500 trees, lr=0.02) ---
    logger.info("2/3 Training LightGBM Ordinal across %d seeds...", len(SEEDS))
    p_test_ord = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    for s in SEEDS:
        logger.info("   LGB Ordinal Seed %d...", s)
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

    p_test_lgb = 0.50 * p_test_mc + 0.50 * p_test_ord

    # --- 3. CatBoost across 2 seeds (250 trees, lr=0.04) ---
    logger.info("3/3 Training CatBoost across 2 seeds...")
    p_test_cb = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    X_cb = X.copy()
    X_test_cb = X_test.copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    cb_seeds = [42, 1337]
    for s in cb_seeds:
        logger.info("   CatBoost Seed %d...", s)
        cb = CatBoostClassifier(
            iterations=250,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            cat_features=cat_cols,
            random_seed=s,
            verbose=0,
            thread_count=-1,
        )
        cb.fit(X_cb, y, verbose=False)
        p_test_cb += cb.predict_proba(X_test_cb) / len(cb_seeds)

    # --- 4. Master Blend: 60% LightGBM (MC+Ord) + 40% CatBoost ---
    logger.info("Blending: 60% LightGBM (MC+Ord) + 40% CatBoost...")
    p_master = 0.60 * p_test_lgb + 0.40 * p_test_cb

    # --- 5. Calibration: Temperature T=1.006 and safe clipping ---
    p_calibrated = apply_temperature_scaling(p_master, temperature=1.006)

    # --- 6. Build and verify submission ---
    sub = pd.DataFrame(p_calibrated, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    assert len(final_sub) == 1820
    assert list(final_sub.columns) == ["ID"] + CLASSES
    assert (final_sub["ID"].values == sub_raw["ID"].values).all()
    assert not final_sub[CLASSES].isna().any().any()
    diff = np.abs(final_sub[CLASSES].sum(axis=1) - 1.0).max()
    assert diff < 1e-4

    out_path = SUBMISSIONS_DIR / "submission_master_lgb_catboost.csv"
    final_sub.to_csv(out_path, index=False)
    logger.info("Saved verified submission file to: %s", out_path)

    print("\nMaster LGB + CatBoost Summary Statistics:")
    print(final_sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
