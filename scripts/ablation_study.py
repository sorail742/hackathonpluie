"""Ablation study script evaluating new feature families on Spatio-Temporal CV."""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_MAP, CLASSES
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds
from src.models import get_regularized_lgb_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_features_on_cv(train_clean, test_clean, custom_fn=None, name="Baseline"):
    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )
    
    if custom_fn is not None:
        train_feat, feature_cols = custom_fn(train_feat, feature_cols)

    X = train_feat[feature_cols]
    y = train_feat["target"].map(TARGET_MAP).to_numpy()
    cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

    fold_losses = []
    logger.info("Evaluating '%s' (%d features) across 4 Spatio-Temporal Folds...", name, len(feature_cols))

    for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols)

        params = get_regularized_lgb_params(seed=42)
        params["learning_rate"] = 0.05  # slightly higher LR for fast ablation speed
        booster = lgb.train(
            params,
            dtr,
            num_boost_round=300,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        p_va = booster.predict(X_va)
        loss = log_loss(y_va, p_va, labels=[0, 1, 2, 3])
        fold_losses.append(loss)
        logger.info("  Fold %d (%s): LogLoss = %.5f (best_iter=%d)", fold, ",".join(heldout_cities), loss, booster.best_iteration)

    mean_loss = float(np.mean(fold_losses))
    std_loss = float(np.std(fold_losses))
    logger.info(">>> Result for '%s': Mean LogLoss = %.5f (+/- %.5f)\n", name, mean_loss, std_loss)
    return mean_loss, std_loss


def main():
    train_raw, test_raw, _ = prepare_datasets()
    train_clean, _ = clean_weather_data(train_raw, verbose=False)
    test_clean, _ = clean_weather_data(test_raw, verbose=False)

    # 1. Baseline features (83 features)
    base_loss, _ = evaluate_features_on_cv(train_clean, test_clean, name="Baseline (83 feats)")

    # 2. Add Candidate features
    def add_candidates(df, cols):
        d = df.copy()
        new_cols = []
        
        # Candidate A: Condensation Potential
        if "dew_point_depression" in d.columns and "humidity" in d.columns:
            d["condensation_potential"] = (d["humidity"] / 100.0) / (1.0 + np.maximum(0.0, d["dew_point_depression"]))
            new_cols.append("condensation_potential")

        # Candidate B: Mean rain intensity when raining
        if "rain_roll_sum_24h" in d.columns and "rain_event_count_24h" in d.columns:
            d["rain_mean_intensity_24h"] = d["rain_roll_sum_24h"] / (d["rain_event_count_24h"] + 1.0)
            new_cols.append("rain_mean_intensity_24h")

        # Candidate C: Cloud bursting proxy
        if "cloud_cover" in d.columns and "cloud_cover_diff1" in d.columns:
            d["cloud_burst"] = ((d["cloud_cover"] > 75.0) & (d["cloud_cover_diff1"] > 20.0)).astype(np.float32)
            new_cols.append("cloud_burst")

        # Candidate D: Vector wind shear
        if "wind_u" in d.columns and "wind_u_diff1" in d.columns:
            # Let's check if wind_u_diff1 exists
            pass

        return d, cols + [c for c in new_cols if c not in cols]

    cand_loss, _ = evaluate_features_on_cv(train_clean, test_clean, custom_fn=add_candidates, name="Candidates (Condensation + Intensity + CloudBurst)")

    print("=" * 60)
    print(f"Baseline LogLoss:   {base_loss:.5f}")
    print(f"Candidates LogLoss: {cand_loss:.5f}")
    print(f"Delta LogLoss:      {cand_loss - base_loss:+.5f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
