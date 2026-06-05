# PPG-Based Heart Rate Estimation — DSC 445 Final Project

> **Status: Complete.** All three weeks of the pipeline are implemented, evaluated, and documented.

## Results Summary

| Model | MAE ± SD (bpm) | 95% CI | RMSE (bpm) |
|-------|----------------|--------|------------|
| Linear Regression | 11.77 ± 4.45 | [9.51, 14.02] | 15.37 |
| **Random Forest** | **8.90 ± 4.52** | **[6.61, 11.19]** | **12.96** |
| XGBoost | 9.00 ± 4.36 | [6.79, 11.20] | 12.82 |
| 1D CNN | 12.69 ± 7.94 | [8.67, 16.71] | 16.06 |
| CNN-BiLSTM† | 9.50 ± 6.75 | [6.09, 12.92] | 13.55 |

† SD uses sample estimator (ddof=1). CNN-BiLSTM predictions use causal median smoothing (window=5); RF does not.

**Key finding:** No statistically significant difference between Random Forest and CNN-BiLSTM (Wilcoxon p = 0.82). Both significantly outperform the 1D CNN (p < 0.001).

---

## Table of Contents

- [Overview](#overview)
- [Dataset Setup](#dataset-setup-required)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Week 1 — Data Pipeline](#week-1--data-pipeline)
- [Week 2 — Classical ML](#week-2--classical-ml)
- [Week 3 — Deep Learning](#week-3--deep-learning)
- [Bug Fixes & Reproducibility](#bug-fixes--reproducibility)
- [Contributors](#contributors)

---

## Overview

This project builds a complete, reproducible wrist-PPG heart rate estimation pipeline evaluated on the **PPG-DaLiA** dataset (15 subjects, 9 naturalistic activities). All models are evaluated under **Leave-One-Subject-Out (LOSO)** cross-validation to measure true subject-independent generalization.

**Methods compared:**
- Classical ML: Linear Regression, Random Forest, XGBoost — 46 hand-engineered PPG+ACC features
- Deep Learning: 1D CNN (~20K params), CNN-BiLSTM with causal smoothing (~77K params)

> Param counts reflect the HPO-selected configuration: `num_filters=32` for both DL models.

**Evaluation protocol:** 15-fold LOSO, MAE + RMSE, 95% CIs (ddof=1), Wilcoxon signed-rank tests.

---

## Dataset Setup (Required)

The raw dataset files are included in `data/raw/ppg+dalia/`. If they are missing, download from:
[PPG-DaLiA](https://ubicomp.eti.uni-siegen.de/home/datasets/sensors19/) and place under `data/raw/`:

```
data/
└── raw/
    └── ppg+dalia/
        └── PPG_FieldStudy/
            ├── S1/S1.pkl
            ├── S2/S2.pkl
            └── ... (S1–S15)
```

---

## Project Structure

```
DSC_445_FINAL/
├── data/raw/                        # PPG-DaLiA raw subject .pkl files
│
├── src/
│   ├── ── Week 1 ──────────────────────────────────────────────────────────
│   ├── data_loader.py               # Load subject .pkl files
│   ├── preprocessing.py             # Butterworth bandpass + per-window z-score
│   ├── loso.py                      # LOSO splitter (leakage-free)
│   ├── sanity_checks.py             # Shape/leakage/HR-range assertions
│   ├── run_week1_pipeline.py        # Week 1 orchestrator
│   │
│   ├── ── Week 2 ──────────────────────────────────────────────────────────
│   ├── features.py                  # 46 hand-crafted PPG+ACC features (pure NumPy)
│   ├── metrics.py                   # RMSE, MAE, fold summarizer
│   ├── models_classical.py          # LR / RF / XGBoost LOSO training
│   ├── evaluate_classical.py        # Summary table + ACC-uplift analysis
│   ├── plots_week2.py               # Bar/box charts, Bland-Altman figures
│   ├── run_week2_pipeline.py        # Week 2 orchestrator
│   │
│   ├── ── Week 3 ──────────────────────────────────────────────────────────
│   ├── dl_dataset.py                # WindowDataset + get_loso_split()
│   ├── dl_preprocessing.py          # PPG+ACC windowing → .npy (with ACC resampling)
│   ├── models_dl.py                 # CNNModel + CNNLSTMModel architectures
│   ├── train_dl.py                  # Training loop, LOSO, early stopping, smoothing
│   ├── plots_week3.py               # DL result figures
│   └── run_week3_pipeline.py        # Week 3 orchestrator
│
├── outputs/
│   ├── processed_data/              # DL per-subject .npy files (S*_ppg_acc.npy, S*_y.npy)
│   ├── week2/                       # Classical ML results
│   │   ├── results_classical.csv    # Per-fold: model × input × subject
│   │   ├── summary_classical.csv    # Mean ± SD per model
│   │   ├── predictions/             # Per-fold .npz files (y_true, y_pred)
│   │   └── figures/
│   └── week3/                       # DL results
│       ├── loso_CNN.csv / loso_CNN_LSTM.csv
│       ├── summary_CNN.json / summary_CNN_LSTM.json
│       ├── cnn_hparams.json / cnnlstm_hparams.json
│       └── analysis/                # Final paper figures + statistical outputs
│           ├── results_with_ci.csv      # 95% CIs (ddof=1) for all 5 models
│           ├── wilcoxon_results.csv     # Pairwise Wilcoxon tests
│           ├── fair_comparison.csv      # Smoothing-matched comparison
│           ├── feature_importance.csv   # |Pearson r| per feature vs HR
│           ├── activity_performance.csv # Per-activity MAE breakdown
│           ├── bland_altman_summary.txt
│           └── *.png                    # All paper figures
│
├── DSC_445_Final/
│   └── ppg_hr_final_report_v2.docx  # Final corrected report
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
# PyTorch (already in requirements.txt, but CPU-only install):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Week 1 — Data loading & windowing

```bash
python src/run_week1_pipeline.py
```

### Week 2 — Classical ML (~20 min on laptop)

```bash
python src/run_week2_pipeline.py                        # Full LOSO
python src/run_week2_pipeline.py --smoke                # 3 folds only (quick check)
python src/run_week2_pipeline.py --skip-clean --skip-features  # Re-run models only
```

### Week 3 — Deep Learning (~2–4 hrs CPU / ~30 min GPU)

```bash
python src/run_week3_pipeline.py                        # Full: preprocess + HPO + LOSO
python src/run_week3_pipeline.py --skip-preprocess      # Skip if processed_data/ exists
```

Results are saved to `outputs/week3/`.

---

## Week 1 — Data Pipeline

- Loaded all 15 subject `.pkl` files; extracted PPG (64 Hz) and ACC (32 Hz)
- Segmented into 8-second windows, 2-second stride (75% overlap between adjacent windows)
- LOSO splitter with leakage assertion in `sanity_checks.py`
- **Total windows:** 64,697 | **PPG shape:** `(N, 512, 1)` | **ACC shape:** `(N, 256, 3)`

---

## Week 2 — Classical ML

**46 hand-crafted features per window** (pure NumPy, confirmed from `feature_names.json`):

| Group | Count | Features |
|-------|-------|---------|
| PPG time-domain | 13 | mean, std, min, max, range, skew, kurt, median, IQR, RMS, ZCR, MAD, energy |
| PPG frequency-domain | 10 | dom_freq, dom_pow, centroid, entropy, bandwidth, 4×sub-band powers, est_hr_bpm |
| ACC per-axis (x/y/z) | 18 | mean, std, RMS, energy, dom_freq, dom_pow × 3 axes |
| ACC magnitude | 5 | mean, std, energy, dom_freq, dom_pow on ‖acc‖ |

**Models and hyperparameters:**

| Model | Key parameters |
|-------|---------------|
| Linear Regression | OLS, no regularization |
| Random Forest | **300 trees**, n_jobs=−1, random_state=42, **not hyperparameter tuned** |
| XGBoost | **500 estimators**, max_depth=6, lr=0.05, tree_method=hist |

> Note: Random Forest was intentionally not hyperparameter optimized — the finding that it matches a tuned DL model is therefore conservative.

**StandardScaler is fit on the training fold only** (no leakage).

---

## Week 3 — Deep Learning

### Preprocessing (`dl_preprocessing.py`)

The DL pipeline differs from Week 2 by design — it produces a single 4-channel tensor:
- Bandpass filter PPG: Butterworth 4th-order, 0.5–4 Hz
- **ACC resampled 32 Hz → 64 Hz** (Fourier `scipy.signal.resample`) to align time axes
- **Global per-subject normalization** (mean/std of full subject signal)
- Windows: 512 samples (8 s), stride 128 (2 s) → `(N, 512, 4)` tensor
- **Label alignment:** `hr_idx = window_end // 128` (64 Hz / 0.5 Hz = 128 samples/label) — see Bug Fixes

### Architectures (HPO-selected configuration)

**1D CNN** (~20K params, `num_filters=32` selected by HPO):
- 2× Conv1D blocks: 4→32→64 filters, kernel=5, BatchNorm+ReLU+MaxPool
- Global Average Pooling → FC(64→128) → Dropout → FC(128→1)

**CNN-BiLSTM** (~77K params, `num_filters=32`, `kernel=7` selected by HPO):
- Same CNN backbone (4→32→64)
- 1-layer bidirectional LSTM: hidden_size=64 per direction
- Concat forward `h[-1]` + backward `h[0]` → 128-dim → FC(128→1)
- Causal moving-median smoothing post-inference (window=5 = 10 s)

> ⚠️ HPO note: Hyperparameters were selected using S2 as the validation subject. S2 then appears in the training set for 14 of 15 LOSO folds, introducing a minor architectural bias. This is disclosed as a limitation in the final report.

### Training

- Optimizer: Adam, L1 (MAE) loss
- Gradient clipping: max_norm=5.0
- Early stopping: patience=8 on 10% held-out validation slice
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=4, min_lr=1e-5)
- HPO: **32 trials** (2×2×2×2×2 grid — code comment incorrectly says 18)
- Reproducibility: global SEED=42 + per-fold seed (42+fold_i), deterministic cuDNN

### DL Results (per-subject)

| Model | MAE mean | MAE SD (ddof=1) | RMSE mean |
|-------|----------|-----------------|-----------|
| 1D CNN | 12.69 | 7.94 | 16.06 |
| CNN-BiLSTM | 9.50 | **6.75** | 13.55 |

**Outlier subjects:**
- **S5:** Error ranges 22.2–34.5 bpm across all models (mean HR=125.8 bpm vs dataset avg ~87 bpm — population prior mismatch)
- **S8:** RF=12.3 bpm vs BiLSTM=23.2 bpm — unusual PPG morphology causes DL distributional shift; classical normalized features are more robust

---

## Bug Fixes & Reproducibility

### Critical bug fix — label alignment

```python
hr_idx = end // 32   # ❌ WRONG (factor-of-4 misalignment)
hr_idx = end // 128  # ✅ CORRECT: 64 Hz PPG / 0.5 Hz label rate = 128 samples per label
```

This was corrected in both `dl_preprocessing.py` and the classical pipeline before all reported results were computed.

### HPO grid comment

```python
# Code comment says "18-combination grid" — INCORRECT
# Actual grid: lr(2) × dropout(2) × batch_size(2) × num_filters(2) × kernel_size(2) = 32 trials
```

### Reproducibility

```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
# Per-fold: set_seed(42 + fold_index)  — implemented in train_dl.py
```

Minor numeric differences (~0.1–0.5 bpm) may occur across CPU vs GPU hardware.

---

## Contributors

| Name | Role |
|------|------|
| Ekaterina Golovkina | Pipeline architecture, feature engineering, preprocessing, classical ML, DL training, report lead |
| Aayesha Kaleem Syeda | Data exploration, Week 1 pipeline, evaluation scripts |
| Alex Turczynski | XGBoost tuning, hyperparameter optimization |
| Afshaan Fathima Syeda | Visualization, results analysis, figures |
| Smitha Nithyananda | Literature review, presentation, documentation |

---
