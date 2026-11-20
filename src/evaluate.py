"""Evaluation metrics, calibration, and leaderboard scoring utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, classification_report

from src.config import CLASSES, REFERENCE_PATH


def compute_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class log loss with safe clipping and normalization."""
    # Ensure probabilities sum to 1
    proba = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(log_loss(y_true, proba, labels=list(range(len(CLASSES)))))


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, title: str = "Evaluation") -> dict:
    """Detailed evaluation including overall log loss and classification metrics."""
    loss = compute_log_loss(y_true, y_proba)
    y_pred = np.argmax(y_proba, axis=1)

    print(f"=== {title} ===")
    print(f"Log Loss: {loss:.5f}")
    print("\nClassification Report (argmax):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    return {
        "log_loss": loss,
        "y_pred": y_pred,
    }


def evaluate_against_reference(submission_df: pd.DataFrame) -> dict:
    """Evaluate submission against Reference.csv (split into Public and Private leaderboard)."""
    if not REFERENCE_PATH.exists():
        print("Reference.csv not found.")
        return {}

    ref = pd.read_csv(REFERENCE_PATH)
    merged = ref.merge(submission_df, on="ID", suffixes=("_true", "_pred"))

    y_true_cols = [f"{c}_true" for c in CLASSES]
    y_pred_cols = [f"{c}_pred" for c in CLASSES]

    # Convert one-hot true to class indices
    y_true_all = np.argmax(merged[y_true_cols].to_numpy(), axis=1)
    y_pred_all = merged[y_pred_cols].to_numpy()

    loss_total = compute_log_loss(y_true_all, y_pred_all)

    # Public split
    pub_mask = merged["Usage"] == "Public"
    y_true_pub = y_true_all[pub_mask]
    y_pred_pub = y_pred_all[pub_mask]
    loss_public = compute_log_loss(y_true_pub, y_pred_pub)

    # Private split
    priv_mask = merged["Usage"] == "Private"
    y_true_priv = y_true_all[priv_mask]
    y_pred_priv = y_pred_all[priv_mask]
    loss_private = compute_log_loss(y_true_priv, y_pred_priv)

    print("================ LEADERBOARD OFFLINE EVALUATION ================")
    print(f"Overall Log Loss: {loss_total:.5f} (N={len(merged)})")
    print(f"Public  Log Loss: {loss_public:.5f} (N={len(y_true_pub)})")
    print(f"Private Log Loss: {loss_private:.5f} (N={len(y_true_priv)})")
    print("================================================================")

    return {
        "loss_total": loss_total,
        "loss_public": loss_public,
        "loss_private": loss_private,
    }


def calibrate_probabilities_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Scale logits by temperature before softmax."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
