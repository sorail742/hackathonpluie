"""Evaluate CatBoost on Spatio-Temporal CV to benchmark performance and class loss breakdown."""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_MAP, CLASSES
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds

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

    # CatBoost handles string/object/category columns natively
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    logger.info("Evaluating CatBoost on Fold 0...")
    for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            cat_features=cat_cols,
            random_seed=42,
            verbose=100,
            task_type="CPU",
            thread_count=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=40,
            verbose=100,
        )

        p_va = model.predict_proba(X_va)
        p_va = np.clip(p_va, 1e-4, 1 - 1e-4)
        p_va = p_va / p_va.sum(axis=1, keepdims=True)

        loss_total = log_loss(y_va, p_va, labels=[0, 1, 2, 3])
        logger.info("CatBoost Fold %d: Total LogLoss = %.5f (Score = %.5f)", fold, loss_total, 1.0 - loss_total / 4.0)

        # Print class breakdown
        for c_idx, c_name in enumerate(CLASSES):
            mask_c = (y_va == c_idx)
            class_contrib = -np.sum(np.log(p_va[mask_c, c_idx])) / len(y_va)
            logger.info("  Class %-8s: %.5f", c_name, class_contrib)

        break


if __name__ == "__main__":
    main()
