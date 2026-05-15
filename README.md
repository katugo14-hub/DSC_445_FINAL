# PPG Heart Rate Estimation (DSC 445 Final Project)

## Overview

This project focuses on improving heart rate estimation from PPG signals using machine learning under motion conditions.

We use the **PPG-DaLiA dataset** and evaluate models using **subject-independent validation (LOSO)** to ensure real-world generalization.

---

## Status

| Phase                                              | Status        |
| -------------------------------------------------- | ------------- |
| Week 1 — Data pipeline (load, window, LOSO splits) | Complete      |
| Week 2 — Classical ML                              | XGBoost comp. |
| Week 3 — Deep learning (CNN / LSTM)                | ⏳ Next       |
| Week 4 — Final evaluation + report                 | ⏳ Pending    |

---

## Week 1: Data Pipeline (recap)

- Loaded all 15 subject `.pkl` files from PPG-DaLiA
- Segmented signals into 8-second windows with a 2-second stride
- Built leave-one-subject-out (LOSO) split generator
- Sanity-checked: no subject leakage, valid HR range, correct shapes
- Output: cached numpy arrays for modeling

### Dataset Summary

- Total windows: **64,697**
- PPG shape: `(N, 512, 1)` (8 s × 64 Hz)
- ACC shape: `(N, 256, 3)` (8 s × 32 Hz, 3 axes)
- HR range: ~41–187 bpm
- Subjects: S1 – S15

---

## Week 2: XGBoost Pipeline

A full classical-ML pipeline built on top of the Week-1 outputs, optimized for **deployment-realistic** constraints (no per-user calibration, no activity-specific models).

### Headline result

| Metric                           | Value           |
| -------------------------------- | --------------- |
| **Overall LOSO MAE**             | **6.67 bpm**    |
| Overall RMSE                     | 10.98 bpm       |
| Mean per-subject MAE             | 6.76 ± 3.59 bpm |
| Best subject (S7)                | 3.77 bpm        |
| Worst subject (S5, heavy motion) | 17.47 bpm       |

### Pipeline components

1. **Feature extraction** (`src/features.py`) — 101 features per window:
   - PPG time-domain statistics, peak/IBI/HRV features, frequency-band powers, harmonic ratios
   - Per-axis and magnitude ACC statistics
   - PPG ↔ ACC spectral cancellation (TROIKA-style motion artifact suppression)
2. **Temporal context** (`src/features_temporal.py`) — adds lag-1/2/3 and Δ-1 of 10 HR-rate-like columns per subject, giving 40 extra features for a total of 141.
3. **Training** (`src/train_xgboost.py`) — XGBoost with `reg:absoluteerror` (L1 loss) and a label-outlier filter (drops training rows where HR jumps > 20 bpm in 2 s).
4. **Post-prediction smoothing** (`src/smoothing.py`) — three production methods (median / zero-phase EMA / Kalman + RTS smoother) retained from a broader 8-method evaluation (5 additional methods tested and rejected as either inferior or redundant). Final pipeline uses **zero-phase EMA with α = 0.20**, the winner of an 80-variant grid sweep against three validation subjects.
5. **Hyperparameter tuning** (`src/tune_xgboost.py`) — random search over 30 trials scored against multiple validation subjects (S2, S4, S11) covering easy/medium/hard difficulty.

## Project Structure

```
DSC_445_FINAL/
├── data/
│   └── raw/              # PPG-DaLiA dataset (NOT included in repo)
├── src/
│   ├── data_loader.py            # Week 1 — load .pkl
│   ├── preprocessing.py          # Week 1 — window segmentation
│   ├── loso.py                   # Week 1 — LOSO split generator
│   ├── sanity_checks.py          # Week 1 — leakage/shape validation
│   ├── run_week1_pipeline.py     # Week 1 — orchestrates the above
│   ├── features.py               # Week 2 — 101-dim feature extractor
│   ├── features_temporal.py      # Week 2 — lag/delta features
│   ├── smoothing.py              # Week 2 — 3 smoothing methods (production)
│   ├── train_xgboost.py          # Week 2 — Config dataclass, LOSO driver
│   ├── run_xgboost.py            # Week 2 — CLI: single run or ablation
│   ├── tune_xgboost.py           # Week 2 — random search + smoothing sweep
│   └── compare_runs.py           # Week 2 — comparison utility
├── outputs/              # Generated files (ignored by git)
├── README.md
```

---

## Dataset Setup (Required)

⚠️ The dataset is **NOT included** in this repository.

Each team member must:

1. Download the PPG-DaLiA dataset
2. Extract it
3. Place it in the following directory:

```
data/raw/
```

Expected structure:

```
data/
└── raw/
    └── ppg+dalia/
        └── PPG_FieldStudy/
            ├── S1/
            ├── S2/
            ├── ...
```

---

## How to Run

From the project root directory:

```bash
# 1. Week-1 pipeline (one-time setup — produces cached .npy arrays)
python src/run_week1_pipeline.py

# 2. Single XGBoost LOSO run with default config
python src/run_xgboost.py

# 3. Cumulative 5-step ablation
python src/run_xgboost.py --ablate

# 4. Random hyperparameter search + smoothing-method sweep + final LOSO
python src/tune_xgboost.py --trials 30 \
    --val-subjects S2 S4 S11 --sweep-smoothing --final-loso

# 5. Compare all saved LOSO runs side-by-side
python src/compare_runs.py
```

The first run also triggers feature extraction (~14 s in parallel on a modern Mac, then cached for subsequent runs).

---

## Output Files

Generated in `outputs/`:

### From Week-1 pipeline

- `X_ppg.npy` → segmented PPG signals
- `X_acc.npy` → segmented accelerometer data
- `y.npy` → heart rate labels
- `subjects.npy` → subject IDs for LOSO validation

### From Week-2 XGBoost pipeline

- `X_features_xgb.npy` → cached 101-dim feature matrix
- `xgb_*_loso_summary.json` → per-fold + aggregate metrics per run
- `xgb_*_loso_predictions.npz` → predictions per window for plotting
- `xgb_*_feature_importance.txt` → averaged feature importance across folds
- `ablation_summary.json` → ablation comparison table
- `xgb_tuning.json` → random search trial log
- `xgb_smoothing_sweep.json` → smoothing-method comparison (28 variants in current code; 80 in the historical sweep that originally selected EMA-centered)
- `xgb_ablation.log`, `xgb_tuning.log` → full console logs

---

## Design Notes

- **LOSO validation** is used throughout to prevent data leakage and ensure subject generalization.
- Random train/test splits are never used.
- **Smoothing is post-hoc** on predictions only - never sees ground truth, so no leakage.

---

## Contributors

- Ekaterina Golovkina
- Alex Turczynski
