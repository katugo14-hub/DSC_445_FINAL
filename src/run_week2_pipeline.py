"""
End-to-end Week 2 pipeline.

Run from project root:
    python src/run_week2_pipeline.py                  # full LOSO
    python src/run_week2_pipeline.py --smoke          # 3 folds only

Inputs:
    outputs/X_ppg.npy, outputs/X_acc.npy, outputs/y.npy, outputs/subjects.npy
    (produced by run_week1_pipeline.py)

Outputs (all under outputs/week2/):
    clean/X_ppg.npy, clean/X_acc.npy, clean/y.npy, clean/subjects.npy
    X_features.npy + feature_names.json
    results_classical.csv + summary_classical.csv + acc_uplift.csv
    predictions/<model>__<input>__<subject>.npz
    figures/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing import clean_arrays
from features import extract_features
from models_classical import run_classical_baselines, summarize
from evaluate_classical import main as evaluate_main
from plots_week2 import make_all_figures
from sanity_checks import check_no_subject_leakage


WEEK1_OUT = PROJECT_ROOT / "outputs"
WEEK2_OUT = PROJECT_ROOT / "outputs" / "week2"
CLEAN_OUT = WEEK2_OUT / "clean"


def _load_week1():
    X_ppg = np.load(WEEK1_OUT / "X_ppg.npy")
    X_acc = np.load(WEEK1_OUT / "X_acc.npy")
    y = np.load(WEEK1_OUT / "y.npy")
    subjects = np.load(WEEK1_OUT / "subjects.npy")
    print(f"[load] Week1 arrays: X_ppg{tuple(X_ppg.shape)}  X_acc{tuple(X_acc.shape)}  y{y.shape}  subjects{subjects.shape}")
    return X_ppg, X_acc, y, subjects


def _save_clean(X_ppg, X_acc, y, subjects):
    CLEAN_OUT.mkdir(parents=True, exist_ok=True)
    np.save(CLEAN_OUT / "X_ppg.npy", X_ppg)
    np.save(CLEAN_OUT / "X_acc.npy", X_acc)
    np.save(CLEAN_OUT / "y.npy", y)
    np.save(CLEAN_OUT / "subjects.npy", subjects)
    print(f"[save] cleaned arrays -> {CLEAN_OUT}/")


def _quick_loso_leakage_check(subjects: np.ndarray) -> None:
    unique = sorted(np.unique(subjects).tolist())
    s0 = unique[0]
    train = subjects[subjects != s0]
    test = subjects[subjects == s0]
    check_no_subject_leakage(train, test)


def main(smoke: bool = False, max_folds: int | None = None,
         skip_clean: bool = False, skip_features: bool = False) -> None:
    t_pipeline = time.time()
    WEEK2_OUT.mkdir(parents=True, exist_ok=True)

    if not skip_clean:
        print("\n== Step 1/5: Preprocess (filter + zscore + label cleanup) ==")
        X_ppg, X_acc, y, subjects = _load_week1()
        X_ppg, X_acc, y, subjects, mask = clean_arrays(
            X_ppg, X_acc, y, subjects,
            hr_min=30.0, hr_max=220.0,
            bandpass_ppg=True, bandpass_acc=False, zscore=True,
        )
        print(f"[clean] kept {mask.sum()}/{len(mask)} windows  "
              f"(HR range {float(y.min()):.1f}-{float(y.max()):.1f} bpm)")
        _save_clean(X_ppg, X_acc, y, subjects)
        _quick_loso_leakage_check(subjects)
    else:
        X_ppg = np.load(CLEAN_OUT / "X_ppg.npy")
        X_acc = np.load(CLEAN_OUT / "X_acc.npy")
        y = np.load(CLEAN_OUT / "y.npy")
        subjects = np.load(CLEAN_OUT / "subjects.npy")
        print(f"[skip] using cached cleaned arrays from {CLEAN_OUT}")

    if not skip_features:
        print("\n== Step 2/5: Extract handcrafted features ==")
        X_feat, feat_names = extract_features(X_ppg, X_acc)
        print(f"[feat] {X_feat.shape}  (finite share {float(np.isfinite(X_feat).mean()):.4f})")
        np.save(WEEK2_OUT / "X_features.npy", X_feat)
        (WEEK2_OUT / "feature_names.json").write_text(json.dumps(feat_names))
    else:
        X_feat = np.load(WEEK2_OUT / "X_features.npy")
        feat_names = json.loads((WEEK2_OUT / "feature_names.json").read_text())
        print(f"[skip] using cached features from {WEEK2_OUT}")

    print("\n== Step 3/5: Train classical models under LOSO ==")
    folds_arg = 3 if smoke else max_folds
    df = run_classical_baselines(
        X_feat, y, subjects, feat_names,
        out_dir=WEEK2_OUT, max_folds=folds_arg, verbose=True,
    )
    g = summarize(df, out_dir=WEEK2_OUT)
    print("\n== Step 4/5: Summary ==")
    print(g.to_string(index=False))
    evaluate_main(WEEK2_OUT)

    print("\n== Step 5/5: Figures ==")
    make_all_figures(WEEK2_OUT)

    elapsed = time.time() - t_pipeline
    print(f"\n[done] Week 2 pipeline finished in {elapsed:.1f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Smoke run on 3 LOSO folds (correctness only).")
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap the number of LOSO folds (debugging).")
    p.add_argument("--skip-clean", action="store_true",
                   help="Reuse outputs/week2/clean/ from a previous run.")
    p.add_argument("--skip-features", action="store_true",
                   help="Reuse outputs/week2/X_features.npy from a previous run.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        smoke=args.smoke,
        max_folds=args.max_folds,
        skip_clean=args.skip_clean,
        skip_features=args.skip_features,
    )
