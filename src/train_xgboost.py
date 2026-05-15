"""
XGBoost training + LOSO evaluation for PPG-DaLiA HR estimation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import extract_features_batch, feature_names
from features_temporal import add_lag_features, DEFAULT_LAG_COLS
from loso import loso_split
from smoothing import smooth_per_subject


# Config

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


ROBUST_LOSS_OVERRIDE = {
    "objective": "reg:absoluteerror",
}


@dataclass
class Config:
    # feature engineering
    use_temporal_features: bool = True
    lag_cols: tuple = field(default_factory=lambda: tuple(DEFAULT_LAG_COLS))
    lags: tuple = (1, 2, 3)
    add_deltas: bool = True

    # data hygiene
    filter_label_outliers: bool = True

    max_label_jump_bpm: float = 20.0

    use_robust_loss: bool = True

    # post-prediction smoothing per subject (see smoothing.py)
    smooth_predictions: bool = True
    smoothing_method: str = "median"
    smoothing_params: dict = field(default_factory=lambda: {"window_size": 11})

    # base xgb params (overlaid on DEFAULT_XGB_PARAMS)
    xgb_params: dict = field(default_factory=dict)

    def merged_xgb_params(self):
        out = dict(DEFAULT_XGB_PARAMS)
        out.update(self.xgb_params)
        if self.use_robust_loss:
            out.update(ROBUST_LOSS_OVERRIDE)
        return out


# Feature cache

def build_or_load_features(ppg_path, acc_path, cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    print("Loading raw arrays...")
    X_ppg = np.load(ppg_path)
    X_acc = np.load(acc_path)
    print(f"  X_ppg: {X_ppg.shape}, X_acc: {X_acc.shape}")

    print("Extracting features...")
    t0 = time.time()
    X_feat = extract_features_batch(X_ppg, X_acc)
    print(f"Done in {time.time() - t0:.1f}s. Feature matrix: {X_feat.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X_feat)
    print(f"Saved features to {cache_path}")
    return X_feat


def assert_label_cadence_matches(X_feat, y, subjects):
    n_feat = len(X_feat)
    n_y = len(y)
    n_subj = len(subjects)
    if not (n_feat == n_y == n_subj):
        raise AssertionError(
            f"Label / feature / subject count mismatch: "
            f"{n_feat} feature rows, {n_y} HR labels, {n_subj} subject IDs. "
            f"The Week-1 `align_labels` silently truncates — check that HR "
            f"labels are at 0.5 Hz and the window step is 2 s in "
            f"preprocessing.py."
        )


# Data hygiene

def label_outlier_mask(y, subjects, max_jump_bpm=25.0):
    """
    Per subject, flag windows where |y[t] - y[t-1]| > max_jump_bpm — those are
    almost certainly label-side artifacts (ECG dropouts / mis-detections).
    First window per subject is always kept.
    """
    y = np.asarray(y)
    subj = np.asarray(subjects)
    keep = np.ones(len(y), dtype=bool)
    for sid in np.unique(subj):
        idx = np.where(subj == sid)[0]
        if len(idx) < 2:
            continue
        deltas = np.abs(np.diff(y[idx]))
        bad_local = deltas > max_jump_bpm     # length len(idx) - 1
        # mark the *later* of each bad pair as bad
        keep[idx[1:][bad_local]] = False
    return keep


#  Training

def train_one_fold(X_train, y_train, subj_train,
                   X_test, y_test, cfg: Config):
    """
    Train one LOSO fold. Returns (model, y_pred).
    """
    params = cfg.merged_xgb_params()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    return model, y_pred


# Metrics

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"mae": float(mae), "rmse": float(rmse), "bias": bias,
            "n": int(len(y_true))}


# LOSO driver

def run_loso(X_feat, y, subjects, output_dir, cfg: Optional[Config] = None,
             tag: str = "xgb",
             feature_name_list: Optional[list] = None):
    """
    Train one model per held-out subject and aggregate metrics.
    """
    assert_label_cadence_matches(X_feat, y, subjects)

    if cfg is None:
        cfg = Config()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if feature_name_list is None:
        feature_name_list = feature_names()
        if len(feature_name_list) != X_feat.shape[1]:
            feature_name_list = [f"f{i}" for i in range(X_feat.shape[1])]

    X_used = X_feat

    fold_metrics = []
    all_preds, all_truth, all_subj = [], [], []
    feat_imp_sum = None

    for fold in loso_split(X_used, y, subjects):
        sid = fold["test_subject"]
        print(f"\n--- Fold: held-out subject {sid} ---")

        X_tr, y_tr = fold["X_train"], fold["y_train"]
        subj_tr = subjects[subjects != sid]
        X_te, y_te = fold["X_test"], fold["y_test"]

        if cfg.filter_label_outliers:
            keep = label_outlier_mask(y_tr, subj_tr,
                                      max_jump_bpm=cfg.max_label_jump_bpm)
            dropped = int((~keep).sum())
            X_tr = X_tr[keep]
            y_tr = y_tr[keep]
            subj_tr = subj_tr[keep]
            print(f"  filtered out {dropped} label-jump rows from train")

        print(f"  train: {X_tr.shape}, test: {X_te.shape}")

        t0 = time.time()
        model, y_pred = train_one_fold(
            X_tr, y_tr, subj_tr, X_te, y_te, cfg
        )
        train_time = time.time() - t0

        # raw test metrics
        m_raw = evaluate(y_te, y_pred)

        # optional smoothing
        if cfg.smooth_predictions:
            subj_te = np.array([sid] * len(y_pred))
            y_pred_smooth = smooth_per_subject(
                y_pred, subj_te, cfg.smoothing_method, **cfg.smoothing_params,
            )
            m_smooth = evaluate(y_te, y_pred_smooth)
            y_pred_final = y_pred_smooth
        else:
            m_smooth = m_raw
            y_pred_final = y_pred

        metrics = {
            "test_subject": str(sid),
            "train_time_s": float(train_time),
            "mae_raw": m_raw["mae"], "rmse_raw": m_raw["rmse"], "bias_raw": m_raw["bias"],
            "mae": m_smooth["mae"], "rmse": m_smooth["rmse"], "bias": m_smooth["bias"],
            "n": m_smooth["n"],
        }
        print(f"  MAE: {metrics['mae']:.2f} (raw {metrics['mae_raw']:.2f}) | "
              f"RMSE: {metrics['rmse']:.2f} | bias: {metrics['bias']:+.2f} | "
              f"time: {train_time:.1f}s")

        fold_metrics.append(metrics)
        all_preds.append(y_pred_final)
        all_truth.append(y_te)
        all_subj.append(np.array([str(sid)] * len(y_pred_final)))

        imp = model.feature_importances_
        feat_imp_sum = imp if feat_imp_sum is None else feat_imp_sum + imp

    y_pred_all = np.concatenate(all_preds)
    y_true_all = np.concatenate(all_truth)
    subj_all = np.concatenate(all_subj)

    overall = evaluate(y_true_all, y_pred_all)
    per_subject_mae = {m["test_subject"]: m["mae"] for m in fold_metrics}

    summary = {
        "tag": tag,
        "overall": overall,
        "per_subject_mae": per_subject_mae,
        "mean_subject_mae": float(np.mean(list(per_subject_mae.values()))),
        "std_subject_mae": float(np.std(list(per_subject_mae.values()))),
        "fold_metrics": fold_metrics,
        "config": asdict(cfg),
        "merged_xgb_params": cfg.merged_xgb_params(),
        "n_features": int(X_used.shape[1]),
        "n_windows": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
    }

    summary_path = output_dir / f"{tag}_loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    np.savez(
        output_dir / f"{tag}_loso_predictions.npz",
        y_true=y_true_all,
        y_pred=y_pred_all,
        subjects=subj_all,
    )

    feat_imp_avg = feat_imp_sum / len(fold_metrics)
    order = np.argsort(feat_imp_avg)[::-1]
    importance_lines = [
        f"{rank+1:>3}. {feature_name_list[i]:<40} {feat_imp_avg[i]:.4f}"
        for rank, i in enumerate(order)
    ]
    (output_dir / f"{tag}_feature_importance.txt").write_text(
        "\n".join(importance_lines)
    )

    print(f"\n=== LOSO summary [{tag}] ===")
    print(f"Overall MAE:           {overall['mae']:.2f} bpm")
    print(f"Overall RMSE:          {overall['rmse']:.2f} bpm")
    print(f"Mean per-subject MAE:  {summary['mean_subject_mae']:.2f} "
          f"(±{summary['std_subject_mae']:.2f}) bpm")
    print(f"Saved {summary_path.name}")
    return summary
