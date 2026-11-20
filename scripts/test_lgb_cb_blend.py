"""Test blending LightGBM (MC + Ordinal) with CatBoost on Fold 0."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from src.config import TARGET_MAP
from src.data import prepare_datasets
from src.cleaning import clean_weather_data
from src.features import build_feature_pipeline
from src.cv import spatio_temporal_folds
from src.calibration import safe_clip_and_renormalize

train_raw, test_raw, _ = prepare_datasets()
train_clean, _ = clean_weather_data(train_raw, verbose=False)
test_clean, _ = clean_weather_data(test_raw, verbose=False)
train_feat, test_feat, feature_cols = build_feature_pipeline(train_clean, test_clean, include_raw_location=False)

X = train_feat[feature_cols].copy()
y = train_feat["target"].map(TARGET_MAP).to_numpy()
cat_cols = [c for c in X.columns if X[c].dtype.name == "category"]

# For CatBoost
X_cb = X.copy()
for c in cat_cols:
    X_cb[c] = X_cb[c].astype(str)

for fold, (tr_idx, va_idx, heldout_cities) in enumerate(spatio_temporal_folds(train_feat, n_splits=4, random_state=42)):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    # 1. LGB Multiclass (600 trees, lr=0.02)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    params_mc = {"objective": "multiclass", "num_class": 4, "learning_rate": 0.02, "num_leaves": 45, "max_depth": 7, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.3, "reg_lambda": 2.5, "verbose": -1, "random_state": 42}
    booster_mc = lgb.train(params_mc, dtr, num_boost_round=600)
    p_mc = booster_mc.predict(X_va)

    # 2. LGB Ordinal (500 trees, lr=0.02)
    p_cum = np.zeros((len(X_va), 3), dtype=np.float64)
    for t_idx, threshold in enumerate([1, 2, 3]):
        y_bin = (y_tr >= threshold).astype(int)
        dtr_bin = lgb.Dataset(X_tr, label=y_bin, categorical_feature=cat_cols)
        params_bin = {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.02, "num_leaves": 40, "max_depth": 6, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.75, "reg_alpha": 0.3, "reg_lambda": 2.5, "verbose": -1, "random_state": 42 + threshold}
        clf_bin = lgb.train(params_bin, dtr_bin, num_boost_round=500)
        p_cum[:, t_idx] = clf_bin.predict(X_va)
    p1 = p_cum[:, 0]
    p2 = np.minimum(p1, p_cum[:, 1])
    p3 = np.minimum(p2, p_cum[:, 2])
    p_ord = np.column_stack([np.clip(1.0 - p1, 0.0, 1.0), np.clip(p1 - p2, 0.0, 1.0), np.clip(p2 - p3, 0.0, 1.0), np.clip(p3, 0.0, 1.0)])
    p_ord = safe_clip_and_renormalize(p_ord)

    p_lgb = 0.5 * p_mc + 0.5 * p_ord

    # 3. CatBoost (200 trees, lr=0.04)
    cb = CatBoostClassifier(iterations=220, learning_rate=0.04, depth=6, l2_leaf_reg=5.0, loss_function="MultiClass", cat_features=cat_cols, random_seed=42, verbose=0, thread_count=-1)
    cb.fit(X_cb.iloc[tr_idx], y_tr, verbose=False)
    p_cb = cb.predict_proba(X_cb.iloc[va_idx])

    loss_lgb = log_loss(y_va, p_lgb)
    loss_cb = log_loss(y_va, p_cb)
    print(f"LGB (MC+Ord) LogLoss: {loss_lgb:.5f}")
    print(f"CatBoost LogLoss:     {loss_cb:.5f}")

    for w_cb in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        p_blend = (1 - w_cb) * p_lgb + w_cb * p_cb
        loss_blend = log_loss(y_va, p_blend)
        print(f"Blend (LGB={1-w_cb:.2f}, CB={w_cb:.2f}): LogLoss = {loss_blend:.5f}")

    break
