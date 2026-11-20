"""Probability calibration, temperature scaling, clipping, and metric evaluations."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, classification_report

from src.config import CLASSES


def safe_clip_and_renormalize(proba: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Clip probabilities to [eps, 1-eps] and renormalize each row to strictly sum to 1."""
    clipped = np.clip(proba, eps, 1.0 - eps)
    renorm = clipped / clipped.sum(axis=1, keepdims=True)
    return renorm


def evaluate_log_loss_breakdown(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Validation"
) -> dict[str, float]:
    """Compute overall log loss and breakdown per class."""
    proba = safe_clip_and_renormalize(y_proba)
    loss = float(log_loss(y_true, proba, labels=list(range(len(CLASSES)))))

    # Per-class contribution to log loss
    n = len(y_true)
    per_class_loss = {}
    for c_idx, c_name in enumerate(CLASSES):
        mask = (y_true == c_idx)
        if mask.sum() > 0:
            # -1/N * sum_{y=c} log(p_{i,c})
            class_contrib = float(-np.sum(np.log(proba[mask, c_idx])) / n)
            class_isolated_loss = float(-np.mean(np.log(proba[mask, c_idx])))
            per_class_loss[c_name] = {
                "contribution": class_contrib,
                "mean_nll": class_isolated_loss,
                "count": int(mask.sum()),
            }

    y_pred = np.argmax(proba, axis=1)
    acc = float(np.mean(y_pred == y_true))

    print(f"\n=== {title} ===")
    print(f"Overall Log Loss: {loss:.5f} | Accuracy: {acc * 100:.2f}%")
    print("Class-wise breakdown (Contribution to total log loss):")
    for c_name, vals in per_class_loss.items():
        print(f"  - {c_name:8s}: {vals['contribution']:.4f} (isolated NLL: {vals['mean_nll']:.4f}, N={vals['count']})")

    return {
        "log_loss": loss,
        "accuracy": acc,
        "per_class": per_class_loss,
    }


def fit_temperature_scaling(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Optimize temperature T to minimize log loss on validation probabilities."""
    eps = 1e-12
    p_safe = np.clip(y_proba, eps, 1.0 - eps)
    # Convert probabilities to pseudo-logits
    logits = np.log(p_safe)

    def objective(t: float) -> float:
        scaled = logits / t
        scaled_exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p_temp = scaled_exp / scaled_exp.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, p_temp, labels=list(range(len(CLASSES)))))

    res = minimize_scalar(objective, bounds=(0.5, 3.0), method="bounded")
    best_temp = float(res.x)
    return best_temp


def apply_temperature_scaling(y_proba: np.ndarray, temperature: float) -> np.ndarray:
    """Scale probabilities by temperature T and renormalize."""
    eps = 1e-12
    p_safe = np.clip(y_proba, eps, 1.0 - eps)
    logits = np.log(p_safe)
    scaled = logits / temperature
    scaled_exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    p_temp = scaled_exp / scaled_exp.sum(axis=1, keepdims=True)
    return safe_clip_and_renormalize(p_temp)
