"""
Entry point for XGBoost.

Expects:
    outputs/X_ppg.npy, outputs/X_acc.npy, outputs/y.npy, outputs/subjects.npy
already produced by `run_week1_pipeline.py`.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from features import feature_names
from features_temporal import add_lag_features
from train_xgboost import (
    Config,
    build_or_load_features,
    run_loso,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_arrays():
    ppg_path = OUTPUT_DIR / "X_ppg.npy"
    acc_path = OUTPUT_DIR / "X_acc.npy"
    y_path = OUTPUT_DIR / "y.npy"
    subj_path = OUTPUT_DIR / "subjects.npy"
    for p in (ppg_path, acc_path, y_path, subj_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run `python src/run_week1_pipeline.py` first."
            )
    return ppg_path, acc_path, y_path, subj_path


def get_feature_matrix(rebuild=False):
    ppg_path, acc_path, y_path, subj_path = load_arrays()
    feat_cache = OUTPUT_DIR / "X_features_xgb.npy"
    if rebuild and feat_cache.exists():
        feat_cache.unlink()
    X_feat = build_or_load_features(ppg_path, acc_path, feat_cache)
    y = np.load(y_path).astype(np.float32)
    subjects = np.load(subj_path)
    return X_feat, y, subjects


def maybe_add_temporal(X_feat, subjects, cfg, fnames):
    if not cfg.use_temporal_features:
        return X_feat, fnames
    X_out, names_out = add_lag_features(
        X_feat, subjects, fnames,
        lag_cols=list(cfg.lag_cols),
        lags=cfg.lags,
        also_add_deltas=cfg.add_deltas,
    )
    print(f"Added temporal features: {X_feat.shape[1]} -> {X_out.shape[1]} cols")
    return X_out, names_out


def single_run(cfg, tag, rebuild=False):
    X_feat, y, subjects = get_feature_matrix(rebuild=rebuild)
    fnames = feature_names()
    if len(fnames) != X_feat.shape[1]:
        fnames = [f"f{i}" for i in range(X_feat.shape[1])]

    X_used, names_used = maybe_add_temporal(X_feat, subjects, cfg, fnames)

    print(f"\n>>> Run [{tag}]   features: {X_used.shape[1]}   "
          f"windows: {len(y)}")
    summary = run_loso(
        X_used, y, subjects, OUTPUT_DIR,
        cfg=cfg, tag=tag, feature_name_list=names_used,
    )
    return summary


# Ablation 
def ablation_configs():
    """
    Cumulative ablations. Each step ADDS one improvement to the previous one
    so we can see the incremental contribution.
    """
    base = Config(
        use_temporal_features=False,
        filter_label_outliers=False,
        use_robust_loss=False,
        smooth_predictions=False,
    )
    plus_temporal = Config(**{**base.__dict__, "use_temporal_features": True})
    plus_outlier  = Config(**{**plus_temporal.__dict__,
                              "filter_label_outliers": True,
                              "max_label_jump_bpm": 20.0})
    plus_robust   = Config(**{**plus_outlier.__dict__, "use_robust_loss": True})
    plus_smooth   = Config(**{**plus_robust.__dict__, "smooth_predictions": True})

    return [
        ("01_baseline",       base),
        ("02_+temporal",      plus_temporal),
        ("03_+label_outlier", plus_outlier),
        ("04_+robust_loss",   plus_robust),
        ("05_+smoothing",     plus_smooth),
    ]


def run_ablation():
    rows = []
    for tag, cfg in ablation_configs():
        s = single_run(cfg, tag, rebuild=False)
        rows.append({
            "tag": tag,
            "overall_mae": s["overall"]["mae"],
            "overall_rmse": s["overall"]["rmse"],
            "mean_subject_mae": s["mean_subject_mae"],
            "std_subject_mae": s["std_subject_mae"],
            "n_features": s["n_features"],
        })

    table_path = OUTPUT_DIR / "ablation_summary.json"
    with open(table_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n================ ABLATION SUMMARY ================")
    print(f"{'tag':<28}{'features':>10}{'overall MAE':>14}{'mean MAE':>12}{'± std':>10}")
    for r in rows:
        print(f"{r['tag']:<28}{r['n_features']:>10}"
              f"{r['overall_mae']:>14.2f}{r['mean_subject_mae']:>12.2f}"
              f"{r['std_subject_mae']:>10.2f}")
    print(f"Saved {table_path}")


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ablate", action="store_true",
                   help="Run the cumulative ablation sweep.")
    p.add_argument("--tag", default="xgb_final",
                   help="Tag for single-run output files.")
    p.add_argument("--rebuild-features", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ablate:
        run_ablation()
    else:
        cfg = Config(
            max_label_jump_bpm=20.0,
            xgb_params={"n_estimators": 2000, "learning_rate": 0.03},
        )
        single_run(cfg, tag=args.tag, rebuild=args.rebuild_features)


if __name__ == "__main__":
    main()
