"""Test regularized XGBoost on Spatio-Temporal CV to verify diversity."""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
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

    # Convert category dtypes to int or enable categorical in XGBoost
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]
    for c in cat_cols:
        X[c] = X[c].cat.codes.astype("category")

    logger.info("Evaluating XGBoost on Fold 0...")
    for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.3,
            reg_lambda=2.0,
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=30,
            n_jobs=-1,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        p_va = model.predict_proba(X_va)
        loss = log_loss(y_va, p_va, labels=[0, 1, 2, 3])
        logger.info("XGBoost Fold %d (%s): LogLoss = %.5f (best_iter=%d)", fold, ",".join(heldout_cities), loss, model.best_iteration)
        break  # Test Fold 0 first


if __name__ == "__main__":
    main()
