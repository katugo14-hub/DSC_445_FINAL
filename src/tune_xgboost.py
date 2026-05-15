"""
Hyperparameter search + post-hoc smoothing sweep for the XGBoost pipeline.

Usage:
    python src/tune_xgboost.py --trials 30 \\
        --val-subjects S2 S4 S11 \\
        --sweep-smoothing --final-loso
"""
from __future__ import annotations
import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
from features import feature_names
from features_temporal import add_lag_features
from train_xgboost import (
    Config,
    build_or_load_features,
    label_outlier_mask,
    train_one_fold,
    evaluate,
    run_loso,
)
from smoothing import smooth_per_subject, smoothing_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# Wide enough to be useful, narrow enough to converge fast.
SEARCH_SPACE = {
    "n_estimators":      [400, 600, 800, 1200, 1600],
    "max_depth":         [4, 5, 6, 7, 8],
    "learning_rate":     [0.03, 0.05, 0.08, 0.12],
    "min_child_weight":  [1, 4, 10, 20],
    "subsample":         [0.7, 0.85, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "reg_lambda":        [0.0, 1.0, 5.0],
    "reg_alpha":         [0.0, 0.1, 1.0],
}


def sample_params(rng):
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def load_full_feature_matrix(cfg: Config):
    X_feat = build_or_load_features(
        OUTPUT_DIR / "X_ppg.npy",
        OUTPUT_DIR / "X_acc.npy",
        OUTPUT_DIR / "X_features_xgb.npy",
    )
    y = np.load(OUTPUT_DIR / "y.npy").astype(np.float32)
    subjects = np.load(OUTPUT_DIR / "subjects.npy")

    fnames = feature_names()
    if len(fnames) != X_feat.shape[1]:
        fnames = [f"f{i}" for i in range(X_feat.shape[1])]

    if cfg.use_temporal_features:
        X_feat, fnames = add_lag_features(
            X_feat, subjects, fnames,
            lag_cols=list(cfg.lag_cols),
            lags=cfg.lags,
            also_add_deltas=cfg.add_deltas,
        )
    return X_feat, y, subjects, fnames


# multi-subject scoring

def _predict_for_subject(X, y, subjects, cfg, val_subject):
    """
    Train on subjects != val_subject; return predicted y_pred on val_subject
    (BEFORE any smoothing — caller decides how to smooth).
    """
    val_mask = (subjects == val_subject)
    tr_mask = ~val_mask
    X_tr, y_tr, subj_tr = X[tr_mask], y[tr_mask], subjects[tr_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if cfg.filter_label_outliers:
        keep = label_outlier_mask(y_tr, subj_tr, max_jump_bpm=cfg.max_label_jump_bpm)
        X_tr, y_tr, subj_tr = X_tr[keep], y_tr[keep], subj_tr[keep]

    _, y_pred = train_one_fold(X_tr, y_tr, subj_tr, X_val, y_val, cfg)
    return y_pred, y_val


def score_multi_validation(X, y, subjects, cfg: Config, val_subjects):
    """
    Trains one model per validation subject (each time excluding that subject
    from training), applies the current smoothing config, and returns the
    MEAN MAE across the validation subjects.
    """
    per_subj = []
    for vs in val_subjects:
        y_pred, y_val = _predict_for_subject(X, y, subjects, cfg, vs)
        if cfg.smooth_predictions:
            subj_val = np.array([vs] * len(y_pred))
            y_pred = smooth_per_subject(
                y_pred, subj_val, cfg.smoothing_method, **cfg.smoothing_params,
            )
        per_subj.append({"subject": vs, "mae": evaluate(y_val, y_pred)["mae"]})
    mean_mae = float(np.mean([r["mae"] for r in per_subj]))
    return mean_mae, per_subj


def search(trials, val_subjects, base_cfg: Config, seed=0):
    X, y, subjects, _ = load_full_feature_matrix(base_cfg)
    missing = [vs for vs in val_subjects if vs not in subjects]
    if missing:
        raise ValueError(f"val subjects not in dataset: {missing}")

    print(f"Tuning over val subjects = {list(val_subjects)}")
    print(f"X shape = {X.shape}")

    rng = random.Random(seed)
    candidates = [sample_params(rng) for _ in range(trials)]

    results = []
    best = None
    for i, p in enumerate(candidates):
        cfg = deepcopy(base_cfg)
        cfg.xgb_params = {**cfg.xgb_params, **p}
        t0 = time.time()
        mean_mae, per_subj = score_multi_validation(X, y, subjects, cfg, val_subjects)
        dt = time.time() - t0
        results.append({"params": p, "mean_mae": mean_mae,
                        "per_subject": per_subj, "time_s": dt})
        marker = ""
        if best is None or mean_mae < best["mean_mae"]:
            best = {"params": p, "mean_mae": mean_mae, "per_subject": per_subj}
            marker = "  <-- new best"
        subj_str = "  ".join(f"{r['subject']}={r['mae']:.2f}" for r in per_subj)
        print(f"[{i+1:>2}/{len(candidates)}] meanMAE={mean_mae:.3f}  "
              f"({subj_str})  depth={p['max_depth']} lr={p['learning_rate']} "
              f"n={p['n_estimators']}  ({dt:.1f}s){marker}")

    print(f"\nBest mean-validation MAE across {list(val_subjects)}: "
          f"{best['mean_mae']:.3f}")
    print(f"Best params: {best['params']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "xgb_tuning.json"
    with open(out_path, "w") as f:
        json.dump({
            "val_subjects": list(val_subjects),
            "best": best,
            "results": results,
        }, f, indent=2)
    print(f"Saved tuning log to {out_path}")
    return best


# smoothing sweep

def sweep_smoothing(X, y, subjects, cfg: Config, val_subjects, grid=None):
    """
    Generate predictions ONCE per validation subject (no smoothing applied),
    then evaluate every smoothing method/param combination on the cached
    predictions.
    """
    if grid is None:
        grid = smoothing_grid()

    cached = {}
    for vs in val_subjects:
        y_pred, y_val = _predict_for_subject(X, y, subjects, cfg, vs)
        cached[vs] = (y_pred, y_val)

    results = []
    for params in grid:
        method = params["method"]
        method_params = {k: v for k, v in params.items() if k != "method"}
        per_subj = []
        for vs in val_subjects:
            y_pred, y_val = cached[vs]
            subj_val = np.array([vs] * len(y_pred))
            y_smooth = smooth_per_subject(y_pred, subj_val, method, **method_params)
            per_subj.append({"subject": vs,
                             "mae": evaluate(y_val, y_smooth)["mae"]})
        mean_mae = float(np.mean([r["mae"] for r in per_subj]))
        row = {"method": method, "params": method_params,
               "per_subject": per_subj, "mean_mae": mean_mae}
        results.append(row)

    best = min(results, key=lambda r: r["mean_mae"])
    print("\nTop 10 smoothing variants:")
    for r in sorted(results, key=lambda x: x["mean_mae"])[:10]:
        ps = "  ".join(f"{x['subject']}={x['mae']:.2f}" for x in r["per_subject"])
        print(f"   meanMAE={r['mean_mae']:.3f}  {r['method']:<22}"
              f"{r['params']}  ({ps})")
    return best, results


# main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--val-subjects", nargs="+", default=["S2", "S4", "S11"],
                   help="Three subjects covering easy/medium/hard difficulty.")
    p.add_argument("--sweep-smoothing", action="store_true")
    p.add_argument("--final-loso", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    base_cfg = Config(
        use_temporal_features=True,
        filter_label_outliers=True,
        max_label_jump_bpm=20.0,
        use_robust_loss=True,
        smooth_predictions=True,
        smoothing_method="median",
        smoothing_params={"window_size": 15},
        xgb_params={"n_estimators": 800, "learning_rate": 0.05},
    )

    best = search(args.trials, args.val_subjects, base_cfg, seed=args.seed)

    final_cfg = deepcopy(base_cfg)
    final_cfg.xgb_params = {**final_cfg.xgb_params, **best["params"]}

    if args.sweep_smoothing:
        print("\n>>> Sweeping smoothing methods with best hyperparams")
        X, y, subjects, _ = load_full_feature_matrix(final_cfg)
        best_smooth, all_smooth = sweep_smoothing(
            X, y, subjects, final_cfg, args.val_subjects,
        )
        print(f"\nBest smoothing: {best_smooth['method']} {best_smooth['params']}"
              f"  meanMAE={best_smooth['mean_mae']:.3f}")
        final_cfg.smoothing_method = best_smooth["method"]
        final_cfg.smoothing_params = best_smooth["params"]
        if best_smooth["method"] == "none":
            final_cfg.smooth_predictions = False
        with open(OUTPUT_DIR / "xgb_smoothing_sweep.json", "w") as f:
            json.dump({"val_subjects": args.val_subjects,
                       "best": best_smooth,
                       "results": all_smooth}, f, indent=2, default=str)

    if args.final_loso:
        print("\n>>> Running full LOSO with tuned params")
        X, y, subjects, fnames = load_full_feature_matrix(final_cfg)
        run_cfg = deepcopy(final_cfg)
        run_cfg.use_temporal_features = False  # already applied to X
        run_loso(X, y, subjects, OUTPUT_DIR, cfg=run_cfg,
                 tag="xgb_robust_final", feature_name_list=fnames)


if __name__ == "__main__":
    main()
