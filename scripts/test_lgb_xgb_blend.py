"""Test blending LightGBM and XGBoost on Spatio-Temporal Fold 0."""

import sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TARGET_MAP
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds
from src.models import get_regularized_lgb_params

train_raw, test_raw, _ = prepare_datasets()
train_clean, _ = clean_weather_data(train_raw, verbose=False)
test_clean, _ = clean_weather_data(test_raw, verbose=False)
train_feat, test_feat, feature_cols = build_feature_pipeline(train_clean, test_clean, include_raw_location=False)

X = train_feat[feature_cols].copy()
y = train_feat["target"].map(TARGET_MAP).to_numpy()
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

# For XGBoost
X_xgb = X.copy()
for c in cat_cols:
    X_xgb[c] = X_xgb[c].cat.codes.astype("category")

for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    # LightGBM
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=cat_cols)
    p_lgb_params = get_regularized_lgb_params(seed=42)
    p_lgb_params["learning_rate"] = 0.04
    lgb_model = lgb.train(p_lgb_params, dtr, num_boost_round=300, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    p_lgb = lgb_model.predict(X_va)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.04, max_depth=5, subsample=0.8, colsample_bytree=0.75,
        reg_alpha=0.3, reg_lambda=2.0, objective="multi:softprob", num_class=4,
        tree_method="hist", enable_categorical=True, random_state=42, early_stopping_rounds=30, n_jobs=-1
    )
    xgb_model.fit(X_xgb.iloc[tr_idx], y_tr, eval_set=[(X_xgb.iloc[va_idx], y_va)], verbose=False)
    p_xgb = xgb_model.predict_proba(X_xgb.iloc[va_idx])

    loss_lgb = log_loss(y_va, p_lgb, labels=[0, 1, 2, 3])
    loss_xgb = log_loss(y_va, p_xgb, labels=[0, 1, 2, 3])

    print(f"LGB LogLoss: {loss_lgb:.5f}")
    print(f"XGB LogLoss: {loss_xgb:.5f}")

    for w_xgb in [0.1, 0.2, 0.3, 0.4, 0.5]:
        p_blend = (1 - w_xgb) * p_lgb + w_xgb * p_xgb
        loss_blend = log_loss(y_va, p_blend, labels=[0, 1, 2, 3])
        print(f"Blend (LGB={1-w_xgb:.1f}, XGB={w_xgb:.1f}): LogLoss = {loss_blend:.5f}")

    break
