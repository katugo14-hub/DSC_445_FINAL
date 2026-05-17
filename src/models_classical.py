"""
Week 2 classical-ML baselines: LinearRegression / RandomForest / XGBoost,
evaluated under LOSO with PPG-only and PPG+ACC feature configurations.

Inputs are the handcrafted features from features.extract_features().
Scaler is fit on TRAIN ONLY each fold.

Outputs (under outputs/week2/):
    results_classical.csv      — one row per (model, input, fold)
    summary_classical.csv      — mean±std across folds per (model, input)
    predictions/<m>_<c>_<S>.npz — y_true, y_pred per fold
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from metrics import rmse, mae, summarize_fold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK2_DIR = PROJECT_ROOT / "outputs" / "week2"
PREDS_DIR = WEEK2_DIR / "predictions"


def _make_models(rf_n_estimators: int = 300, xgb_n_estimators: int = 500) -> dict:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_n_estimators,
            n_jobs=-1,
            random_state=42,
        ),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=xgb_n_estimators,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
    return models


def _feature_slices(
    feat_names: list[str],
) -> dict[str, np.ndarray]:
    """Column indices for each input configuration."""
    ppg_cols = np.array([i for i, n in enumerate(feat_names) if n.startswith("ppg_")])
    acc_cols = np.array([i for i, n in enumerate(feat_names) if n.startswith("acc")])
    all_cols = np.arange(len(feat_names))
    return {
        "ppg_only": ppg_cols,
        "ppg+acc": np.concatenate([ppg_cols, acc_cols]),
        "_acc_cols": acc_cols,
    }


def loso_folds(subjects: np.ndarray, max_folds: int | None = None) -> Iterable[tuple[int, str, np.ndarray, np.ndarray]]:
    """Yield (fold_idx, held_subject, train_idx, test_idx) for each subject."""
    unique = sorted(np.unique(subjects).tolist())
    if max_folds is not None:
        unique = unique[:max_folds]
    for k, s in enumerate(unique):
        test_idx = np.where(subjects == s)[0]
        train_idx = np.where(subjects != s)[0]
        yield k, s, train_idx, test_idx


def run_classical_baselines(
    X_feat: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feat_names: list[str],
    out_dir: Path | None = None,
    max_folds: int | None = None,
    models: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train and evaluate classical models under LOSO with two input configs.

    Returns a DataFrame of per-fold results. Also writes:
        out_dir/results_classical.csv
        out_dir/predictions/<model>_<config>_<subject>.npz
    """
    out_dir = Path(out_dir or WEEK2_DIR)
    preds_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    slices = _feature_slices(feat_names)
    configs = {
        "ppg_only": slices["ppg_only"],
        "ppg+acc": slices["ppg+acc"],
    }

    models = models or _make_models()

    if not _HAS_XGB and "xgboost" in (models or {}):
        # caller forced xgboost but it's missing
        models.pop("xgboost", None)
    if not _HAS_XGB:
        print("[warn] xgboost not installed — skipping XGBRegressor.")

    rows = []
    folds = list(loso_folds(subjects, max_folds=max_folds))
    total = len(folds) * len(configs) * len(models)
    done = 0

    for fold_idx, subject, train_idx, test_idx in folds:
        y_train = y[train_idx]
        y_test = y[test_idx]

        for config_name, cols in configs.items():
            X_train_raw = X_feat[train_idx][:, cols]
            X_test_raw = X_feat[test_idx][:, cols]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            for model_name, model in models.items():
                t0 = time.time()
                # important: clone-ish — fit a fresh copy per fold
                from sklearn.base import clone as sk_clone
                try:
                    fit_model = sk_clone(model)
                except Exception:
                    # xgboost works with sklearn clone too but guard anyway
                    fit_model = model.__class__(**model.get_params())

                fit_model.fit(X_train, y_train)
                y_pred = fit_model.predict(X_test)

                row = summarize_fold(
                    y_test, y_pred,
                    fold_id=fold_idx, subject=subject,
                    model=model_name, input_config=config_name,
                    n_train=len(train_idx), n_test=len(test_idx),
                )
                row["train_seconds"] = round(time.time() - t0, 2)
                rows.append(row)

                # save per-fold predictions for later Bland-Altman / scatter
                np.savez_compressed(
                    preds_dir / f"{model_name}__{config_name}__{subject}.npz",
                    y_true=y_test.astype(np.float32),
                    y_pred=np.asarray(y_pred, dtype=np.float32),
                )

                done += 1
                if verbose:
                    print(
                        f"  [{done:>3}/{total}] {subject} | {model_name:<17} | "
                        f"{config_name:<8} | RMSE={row['rmse']:6.2f}  MAE={row['mae']:6.2f}  "
                        f"({row['train_seconds']:.1f}s)"
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_classical.csv", index=False)
    if verbose:
        print(f"[ok] wrote {out_dir/'results_classical.csv'} ({len(df)} rows)")
    return df


def summarize(results_df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = Path(out_dir or WEEK2_DIR)
    grouped = (
        results_df.groupby(["model", "input"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["model", "input"])
    )
    grouped.to_csv(out_dir / "summary_classical.csv", index=False)
    return grouped


if __name__ == "__main__":
    # quick CLI: run with already-saved X_features.npy
    week2 = WEEK2_DIR
    X = np.load(week2 / "X_features.npy")
    y = np.load(week2 / "clean" / "y.npy")
    subjects = np.load(week2 / "clean" / "subjects.npy")
    names = json.loads((week2 / "feature_names.json").read_text())
    df = run_classical_baselines(X, y, subjects, names)
    print(summarize(df).to_string(index=False))
