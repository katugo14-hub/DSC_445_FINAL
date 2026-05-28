"""
run_week3_pipeline.py
Week 3 orchestrator: preprocess → sanity check → tune → LOSO for CNN and CNN-LSTM.

Usage:
    python src/run_week3_pipeline.py                   # full run
    python src/run_week3_pipeline.py --skip-preprocess # skip if already preprocessed
"""
import argparse
import json
import time
from pathlib import Path

import torch

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"
RESULTS_DIR   = PROJECT_ROOT / "outputs" / "week3"

# Deep learning modules (imported after path setup)
from dl_preprocessing import preprocess_all_subjects
from dl_dataset import get_all_subjects
from train_dl import tune_hyperparameters, train_loso
from models_dl import CNNModel, CNNLSTMModel
from sanity_checks import check_data_shapes, check_for_nans, check_model_forward_pass


def _processed_exists():
    """Return list of preprocessed subject IDs, or None if none found."""
    subjects = sorted({p.name.split("_")[0] for p in PROCESSED_DIR.glob("*_ppg_acc.npy")})
    return subjects if subjects else None


def main(skip_preprocess: bool = False):
    t0     = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("\n=== Step 1/5: Preprocessing (PPG + ACC → 4-channel windows) ===")
    if skip_preprocess:
        subjects = _processed_exists()
        if subjects is None:
            raise RuntimeError("--skip-preprocess set but no processed files found. Run without the flag first.")
        print(f"[skip] Found {len(subjects)} preprocessed subjects.")
    else:
        preprocess_all_subjects()
        subjects = _processed_exists()
        if not subjects:
            raise RuntimeError("Preprocessing completed but no output files found. Check dl_preprocessing.py paths.")
        print(f"[done] Preprocessed {len(subjects)} subjects.")

    # ── Step 2: Sanity checks ─────────────────────────────────────────────────
    print("\n=== Step 2/5: Sanity Checks ===")
    check_data_shapes()
    check_for_nans()
    check_model_forward_pass()

    # ── Step 3: Tune CNN ──────────────────────────────────────────────────────
    print("\n=== Step 3/5: Hyperparameter Tuning — CNN (val subject: S2) ===")
    cnn_hparams = tune_hyperparameters(
        model_class=CNNModel,
        val_subject="S2",
        device=device,
    )
    print("[CNN] Best hparams:", cnn_hparams)
    (RESULTS_DIR / "cnn_hparams.json").write_text(json.dumps(cnn_hparams, indent=2))

    # ── Step 4: Tune CNN-LSTM ────────────────────────────────────────────────
    print("\n=== Step 4/5: Hyperparameter Tuning — CNN-LSTM (val subject: S2) ===")
    cnnlstm_hparams = tune_hyperparameters(
        model_class=CNNLSTMModel,
        val_subject="S2",
        device=device,
    )
    print("[CNN-LSTM] Best hparams:", cnnlstm_hparams)
    (RESULTS_DIR / "cnnlstm_hparams.json").write_text(json.dumps(cnnlstm_hparams, indent=2))

    # ── Step 5: Full LOSO for both models ────────────────────────────────────
    print("\n=== Step 5/5: Full LOSO Training ===")

    print("\n--- CNN LOSO ---")
    cnn_results, cnn_summary = train_loso(
        model_class=CNNModel,
        hparams=cnn_hparams,
        device=device,
        max_epochs=40,
        patience=8,
        model_name="CNN",
    )

    print("\n--- CNN-LSTM LOSO ---")
    cnnlstm_results, cnnlstm_summary = train_loso(
        model_class=CNNLSTMModel,
        hparams=cnnlstm_hparams,
        device=device,
        max_epochs=40,
        patience=8,
        model_name="CNN_LSTM",
    )

    # ── Final comparison ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("WEEK 3 FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Model':<12} {'MAE mean':>10} {'MAE std':>9} {'RMSE mean':>11} {'RMSE std':>10}")
    print("-" * 60)
    for s in [cnn_summary, cnnlstm_summary]:
        print(f"{s['model']:<12} {s['mae_mean']:>10.3f} {s['mae_std']:>9.3f} "
              f"{s['rmse_mean']:>11.3f} {s['rmse_std']:>10.3f}")
    print("=" * 60)
    print(f"\n[done] Week 3 pipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if outputs/processed_data/ already exists")
    args = parser.parse_args()
    main(skip_preprocess=args.skip_preprocess)
