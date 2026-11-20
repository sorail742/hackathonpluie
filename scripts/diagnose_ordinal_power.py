"""Diagnose and optimize Multiclass + Ordinal blend with deeper trees and lower learning rate."""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_MAP, CLASSES
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds
from src.calibration import safe_clip_and_renormalize, apply_temperature_scaling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    train_raw, test_raw, _ = prepare_datasets()
    train_clean, _ = clean_weather_data(train_raw, verbose=False)
    test_clean, _ = clean_weather_data(test_raw, verbose=False)

    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols].copy()
    y = train_feat["target"].map(TARGET_MAP).to_numpy()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

    # Evaluate on Fold 0:
    for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        logger.info("Fold 0 with heldout: %s", heldout_cities)

        # 1. Multiclass with lr=0.02, num_boost_round=600
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols)

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
            "random_state": 42,
        }
        booster_mc = lgb.train(params_mc, dtr, num_boost_round=600)
        p_mc = booster_mc.predict(X_va)
        loss_mc = log_loss(y_va, p_mc)
        logger.info("Multiclass (lr=0.02, 600 trees): LogLoss = %.5f", loss_mc)

        # 2. Ordinal with lr=0.02, num_boost_round=500
        p_cum = np.zeros((len(X_va), 3), dtype=np.float64)
        for t_idx, threshold in enumerate([1, 2, 3]):
            y_bin = (y_tr >= threshold).astype(int)
            dtr_bin = lgb.Dataset(X_tr, label=y_bin, categorical_feature=cat_cols)
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
                "random_state": 42 + threshold,
            }
            clf_bin = lgb.train(params_bin, dtr_bin, num_boost_round=500)
            p_cum[:, t_idx] = clf_bin.predict(X_va)

        p1 = p_cum[:, 0]
        p2 = np.minimum(p1, p_cum[:, 1])
        p3 = np.minimum(p2, p_cum[:, 2])

        p_ord = np.column_stack([
            np.clip(1.0 - p1, 0.0, 1.0),
            np.clip(p1 - p2, 0.0, 1.0),
            np.clip(p2 - p3, 0.0, 1.0),
            np.clip(p3, 0.0, 1.0),
        ])
        p_ord = safe_clip_and_renormalize(p_ord)
        loss_ord = log_loss(y_va, p_ord)
        logger.info("Ordinal (lr=0.02, 500 trees): LogLoss = %.5f", loss_ord)

        # Test various blend weights
        for w_ord in [0.2, 0.3, 0.4, 0.5, 0.6]:
            p_blend = (1 - w_ord) * p_mc + w_ord * p_ord
            for T in [1.0, 1.005, 1.01]:
                p_cal = apply_temperature_scaling(p_blend, temperature=T)
                loss_blend = log_loss(y_va, p_cal)
                logger.info("  Blend (MC=%.1f, Ord=%.1f, T=%.3f): LogLoss = %.5f", 1 - w_ord, w_ord, T, loss_blend)

        break


if __name__ == "__main__":
    main()
