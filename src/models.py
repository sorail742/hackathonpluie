"""Model training and prediction wrappers for LightGBM with regularization and multi-seeding."""

import logging
from typing import Any
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.calibration import safe_clip_and_renormalize

logger = logging.getLogger(__name__)


def get_regularized_lgb_params(seed: int = 42) -> dict[str, Any]:
    """Production regularized hyperparameters designed to generalize spatio-temporally."""
    return {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 35,
        "max_depth": 6,
        "min_child_samples": 50,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "n_jobs": -1,
        "random_state": seed,
        "verbose": -1,
    }


def train_single_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 0,
) -> lgb.Booster:
    """Train a single LightGBM model with early stopping."""
    if params is None:
        params = get_regularized_lgb_params()

    cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = []

    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        if verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks if callbacks else None,
    )
    return booster


def predict_proba_lgb(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict probabilities and renormalize cleanly."""
    raw_proba = booster.predict(X)
    return safe_clip_and_renormalize(raw_proba)
