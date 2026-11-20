"""Advanced training pipeline: meteorological feature engineering, K-fold CV, and ensembled LightGBM."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH, SUBMISSIONS_DIR, TARGET_MAP, CLASSES, RANDOM_SEED
from src.data_cleaning import clean_dataframe
from src.feature_engineering import extract_features
from src.validation import get_stratified_time_folds
from src.models import get_default_lgb_params, train_lgb_model, predict_proba
from src.evaluate import evaluate_predictions, evaluate_against_reference


def run_advanced_pipeline():
    print("==================================================")
    print(" IndabaX Guinea 2026 - Advanced Training Pipeline ")
    print("==================================================")

    print("\n1. Loading datasets...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print(f"Raw Train: {train_raw.shape}, Raw Test: {test_raw.shape}")

    # Combine for consistent temporal sorting and feature extraction
    print("\n2. Cleaning and standardizing locations...")
    train_clean = clean_dataframe(train_raw)
    test_clean = clean_dataframe(test_raw)

    print("\n3. Engineering meteorological and climatological features...")
    # Concatenate train and test temporarily to allow seamless station rolling features
    train_clean["is_test"] = 0
    test_clean["is_test"] = 1

    combined = pd.concat([train_clean, test_clean], ignore_index=True)
    combined_featured, feature_cols = extract_features(combined, categorical_as_category=True)

    print(f"Total features extracted ({len(feature_cols)}):")
    print(", ".join(feature_cols))

    # Split back to train and test
    train_df = combined_featured[combined_featured["is_test"] == 0].copy().reset_index(drop=True)
    test_df = combined_featured[combined_featured["is_test"] == 1].copy().reset_index(drop=True)

    X = train_df[feature_cols]
    y = train_df["target"].map(TARGET_MAP).to_numpy()

    # Filter test set rows matching sample submission
    to_predict = test_df[test_df["ID"].isin(sub["ID"])].copy().reset_index(drop=True)
    X_test = to_predict[feature_cols]

    # 4. Stratified 5-Fold Cross Validation
    n_splits = 5
    print(f"\n4. Starting {n_splits}-Fold Cross-Validation...")

    oof_proba = np.zeros((len(train_df), len(CLASSES)), dtype=np.float64)
    test_proba_folds = np.zeros((len(to_predict), len(CLASSES)), dtype=np.float64)

    feature_importances = []
    lgb_params = get_default_lgb_params()

    for fold, (tr_idx, va_idx) in enumerate(get_stratified_time_folds(train_df, n_splits=n_splits, random_state=RANDOM_SEED)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = train_lgb_model(
            X_tr, y_tr, X_va, y_va,
            params=lgb_params,
            num_boost_round=1200,
            early_stopping_rounds=50,
            verbose_eval=150
        )

        va_pred = predict_proba(model, X_va)
        oof_proba[va_idx] = va_pred

        fold_loss = evaluate_predictions(y_va, va_pred, title=f"Fold {fold + 1} Validation")["log_loss"]
        print(f"Fold {fold + 1} Best Iteration: {model.best_iteration}, Log Loss: {fold_loss:.5f}")

        # Predict on test
        test_pred = predict_proba(model, X_test)
        test_proba_folds += test_pred / n_splits

        # Track feature importance
        importances = model.feature_importance(importance_type="gain")
        feature_importances.append(importances)

    # 5. Out-of-fold Overall Evaluation
    print("\n==================================================")
    print(" OVERALL OUT-OF-FOLD EVALUATION ")
    print("==================================================")
    oof_eval = evaluate_predictions(y, oof_proba, title="Out-of-Fold (All 5 Folds)")

    # Print Top 20 Most Important Features
    avg_imp = np.mean(feature_importances, axis=0)
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": avg_imp
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print("\nTop 20 Most Important Features (Gain):")
    print(feat_imp_df.head(20).to_string(index=False))

    # 6. Generate Submission
    print("\n6. Generating test submission file...")
    submission = pd.DataFrame(test_proba_folds, columns=CLASSES)
    submission.insert(0, "ID", to_predict["ID"].values)
    submission = sub[["ID"]].merge(submission, on="ID", how="left")

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SUBMISSIONS_DIR / "submission_advanced_lgb.csv"
    submission.to_csv(out_file, index=False)
    print(f"Saved advanced submission to: {out_file}")

    # 7. Evaluate against Reference.csv
    print("\n7. Offline Leaderboard Evaluation:")
    evaluate_against_reference(submission)


if __name__ == "__main__":
    run_advanced_pipeline()
