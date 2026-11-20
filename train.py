"""CLI end-to-end training script with spatio-temporal validation and model persistence."""

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
    MODELS_DIR,
)
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds, temporal_purged_split, leave_city_group_out_folds
from src.models import get_regularized_lgb_params, train_single_lgb, predict_proba_lgb
from src.calibration import (
    evaluate_log_loss_breakdown,
    fit_temperature_scaling,
    apply_temperature_scaling,
    safe_clip_and_renormalize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train rainfall prediction model.")
    parser.add_argument(
        "--cv",
        choices=["spatio_temporal", "temporal", "spatial"],
        default="spatio_temporal",
        help="Validation scheme to evaluate model.",
    )
    parser.add_argument("--num_boost_round", type=int, default=1200, help="Max boosting rounds.")
    parser.add_argument("--multi_seed", action="store_true", help="Average multiple seeds for stability.")
    args = parser.parse_args()

    logger.info("Starting training pipeline with CV scheme: %s", args.cv)

    # 1. Load data
    train_raw, test_raw, sub_raw = prepare_datasets()

    # 2. Clean data
    train_clean, rep_tr = clean_weather_data(train_raw)
    test_clean, rep_te = clean_weather_data(test_raw)

    # 3. Features
    train_feat, test_feat, feature_cols = build_feature_pipeline(
        train_clean, test_clean, include_raw_location=False
    )

    X = train_feat[feature_cols]
    y = train_feat["target"].map(TARGET_MAP).to_numpy()

    # Filter test rows for sample submission
    to_predict = test_feat[test_feat["ID"].isin(sub_raw["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols]

    seeds = SEEDS if args.multi_seed else [RANDOM_SEED]
    oof_proba = np.zeros((len(train_feat), len(CLASSES)), dtype=np.float64)
    test_proba = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)

    # 4. Cross-Validation
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    val_indices = []

    if args.cv == "spatio_temporal":
        folds = list(spatio_temporal_folds(train_feat, n_splits=4, random_state=RANDOM_SEED))
        logger.info("Executing 4-Fold Spatio-Temporal CV (Train <=2023 seen cities, Val 2024 unseen cities)...")

        fold_losses = []
        for fold_idx, (tr_idx, va_idx, heldout_cities) in enumerate(folds):
            logger.info("--- Fold %d/4 | Held-out 2024 cities: %s ---", fold_idx + 1, heldout_cities)
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            fold_preds_val = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
            fold_preds_test = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)

            for s in seeds:
                params = get_regularized_lgb_params(seed=s)
                booster = train_single_lgb(
                    X_tr, y_tr, X_va, y_va,
                    params=params,
                    num_boost_round=args.num_boost_round,
                    early_stopping_rounds=40,
                    verbose_eval=150,
                )
                models.append(booster)
                fold_preds_val += predict_proba_lgb(booster, X_va) / len(seeds)
                fold_preds_test += predict_proba_lgb(booster, X_test) / (len(seeds) * len(folds))

            oof_proba[va_idx] = fold_preds_val
            val_indices.extend(va_idx)
            test_proba += fold_preds_test

            fold_eval = evaluate_log_loss_breakdown(y_va, fold_preds_val, title=f"Fold {fold_idx + 1} (Unseen Cities 2024)")
            fold_losses.append(fold_eval["log_loss"])

        logger.info(
            "Spatio-Temporal CV Result: Mean Log Loss = %.5f ± %.5f",
            np.mean(fold_losses), np.std(fold_losses)
        )

    elif args.cv == "temporal":
        tr_idx, va_idx = temporal_purged_split(train_feat)
        logger.info("Executing Temporal Purged Split (Train 2022-2023, Val 2024)...")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        val_preds = np.zeros((len(va_idx), len(CLASSES)), dtype=np.float64)
        for s in seeds:
            params = get_regularized_lgb_params(seed=s)
            booster = train_single_lgb(
                X_tr, y_tr, X_va, y_va,
                params=params,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=40,
                verbose_eval=150,
            )
            val_preds += predict_proba_lgb(booster, X_va) / len(seeds)
            test_proba += predict_proba_lgb(booster, X_test) / len(seeds)

        val_indices = va_idx
        oof_proba[va_idx] = val_preds
        evaluate_log_loss_breakdown(y_va, val_preds, title="Temporal Purged Holdout (Year 2024)")

    # 5. Out-Of-Fold Evaluation & Calibration
    val_indices = np.array(val_indices)
    y_val_all = y[val_indices]
    oof_val_preds = oof_proba[val_indices]

    logger.info("Fitting temperature calibration on out-of-fold validation...")
    best_temp = fit_temperature_scaling(y_val_all, oof_val_preds)
    logger.info("Optimal Temperature T: %.4f", best_temp)

    calibrated_oof = apply_temperature_scaling(oof_val_preds, best_temp)
    logger.info("Evaluation BEFORE calibration:")
    evaluate_log_loss_breakdown(y_val_all, oof_val_preds, title="OOF Uncalibrated")

    logger.info("Evaluation AFTER calibration:")
    evaluate_log_loss_breakdown(y_val_all, calibrated_oof, title="OOF Calibrated")

    # 6. Apply calibration to test set and save submission
    calibrated_test_proba = apply_temperature_scaling(test_proba, best_temp)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame(calibrated_test_proba, columns=CLASSES)
    sub.insert(0, "ID", to_predict["ID"].values)
    final_sub = sub_raw[["ID"]].merge(sub, on="ID", how="left")

    out_file = SUBMISSIONS_DIR / f"submission_{args.cv}.csv"
    final_sub.to_csv(out_file, index=False)
    logger.info("Generated final verified submission at: %s", out_file)


if __name__ == "__main__":
    main()
