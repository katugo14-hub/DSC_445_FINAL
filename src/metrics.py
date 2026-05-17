"""
Regression metrics for heart-rate estimation (Week 2+).
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def summarize_fold(
    y_true, y_pred, fold_id: int, subject: str, model: str, input_config: str,
    n_train: int, n_test: int,
) -> dict:
    """
    One row per (model, input_config, fold). Ready to append to a DataFrame.
    """
    return {
        "model": model,
        "input": input_config,
        "fold": int(fold_id),
        "subject": str(subject),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
