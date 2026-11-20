"""Train Ultimate 3-Way Ensemble Blend:
1. LightGBM Multiclass (Seeds 42, 1337, 2026) - Optimal 220 rounds
2. LightGBM Ordinal (Seeds 42, 1337) - 3 Thresholds, Optimal 180 rounds
3. XGBoost Multiclass (Seeds 42, 1337) - Hist gradient boosting, Optimal 150 rounds
Calibrated with T=1.02, safe clipping [1e-4, 1-1e-4], and verified strictly.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

# Ensure project root is in sys.path
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
    logger.info("=== Training Ultimate 3-Way Ensemble (LGB MC + LGB Ordinal + XGBoost) ===")

    # 1. Load and clean
    train_raw, test_raw, sub_raw = prepare_datasets()
    train_clean, _ = clean_weather_data(train_raw, verbose=False)
    test_clean, _ = clean_weather_data(test_raw, verbose=False)

    # 2. Features (Drift-neutralized, strictly causal, no location leakage)
    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols].copy()
    y = train_feat["target"].map(TARGET_MAP).to_numpy()

    to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols].copy()

    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

    # --- MODEL 1: LightGBM Multiclass (Optimal ~220 trees) across 3 seeds ---
    p_test_lgb_mc = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    logger.info("1/3 Training LightGBM Multiclass across %d seeds (num_boost_round=220)...", len(SEEDS))

    for s in SEEDS:
        logger.info("   LGB Multiclass Seed %d...", s)
        params = get_regularized_lgb_params(seed=s)
        params["learning_rate"] = 0.03
        dtr = lgb.Dataset(X, label=y, categorical_feature=cat_cols)
        clf = lgb.train(params, dtr, num_boost_round=220)
        p_test_lgb_mc += clf.predict(X_test) / len(SEEDS)

    # --- MODEL 2: LightGBM Ordinal (3 binary thresholds) across 2 seeds ---
    p_test_lgb_ord = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    ord_seeds = [42, 1337]
    logger.info("2/3 Training LightGBM Ordinal across %d seeds (num_boost_round=180)...", len(ord_seeds))

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
            }
            clf = lgb.train(params_bin, dtr, num_boost_round=180)
            p_cum[:, t_idx] = clf.predict(X_test)

        # Enforce monotonicity
        p1 = p_cum[:, 0]
        p2 = np.minimum(p1, p_cum[:, 1])
        p3 = np.minimum(p2, p_cum[:, 2])

        p_4c = np.column_stack([
            np.clip(1.0 - p1, 0.0, 1.0),
            np.clip(p1 - p2, 0.0, 1.0),
            np.clip(p2 - p3, 0.0, 1.0),
            np.clip(p3, 0.0, 1.0),
        ])
        p_test_lgb_ord += safe_clip_and_renormalize(p_4c) / len(ord_seeds)

    # --- MODEL 3: XGBoost Multiclass (tree_method='hist') across 2 seeds ---
    p_test_xgb = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)
    xgb_seeds = [42, 1337]
    logger.info("3/3 Training XGBoost Multiclass across %d seeds (n_estimators=150)...", len(xgb_seeds))

    # Prepare data for XGBoost (convert category codes)
    X_xgb = X.copy()
    X_test_xgb = X_test.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes.astype("category")
        X_test_xgb[c] = X_test_xgb[c].cat.codes.astype("category")

    for s in xgb_seeds:
        logger.info("   XGBoost Seed %d...", s)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.80,
            colsample_bytree=0.75,
            reg_alpha=0.3,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            enable_categorical=True,
            random_state=s,
            n_jobs=-1,
        )
        xgb_clf.fit(X_xgb, y, verbose=False)
        p_test_xgb += xgb_clf.predict_proba(X_test_xgb) / len(xgb_seeds)

    # --- BLEND: 55% LGB MC + 30% LGB Ordinal + 15% XGBoost ---
    logger.info("Blending: 55% LGB MC + 30% LGB Ordinal + 15% XGBoost...")
    p_blend = 0.55 * p_test_lgb_mc + 0.30 * p_test_lgb_ord + 0.15 * p_test_xgb

    # --- CALIBRATION: Temperature Scaling T=1.02 & Safe Clipping ---
    logger.info("Applying Temperature Scaling (T=1.02) and safe clipping...")
    p_calibrated = apply_temperature_scaling(p_blend, temperature=1.02)

    # --- BUILD & VALIDATE SUBMISSION ---
    sub = pd.DataFrame(p_calibrated, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    assert len(final_sub) == 1820
    assert list(final_sub.columns) == ["ID"] + CLASSES
    assert (final_sub["ID"].values == sub_raw["ID"].values).all()
    assert not final_sub[CLASSES].isna().any().any()
    diff = np.abs(final_sub[CLASSES].sum(axis=1) - 1.0).max()
    assert diff < 1e-4

    out_path = SUBMISSIONS_DIR / "submission_ultimate_3way_blend.csv"
    final_sub.to_csv(out_path, index=False)
    logger.info("Saved verified submission file to: %s", out_path)

    print("\nUltimate Submission Summary Statistics:")
    print(final_sub[CLASSES].describe().T[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
