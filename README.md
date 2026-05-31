# PPG-Based Heart Rate Estimation — DSC 445 Final Project

> **Status: Complete.** All three weeks of the pipeline are implemented, evaluated, and documented.

## Results Summary

| Model | MAE (bpm) | 95% CI | RMSE (bpm) |
|-------|-----------|--------|------------|
| Linear Regression | 11.77 ± 4.45 | [9.51, 14.02] | 15.37 |
| **Random Forest** | **8.90 ± 4.52** | **[6.61, 11.19]** | **12.96** |
| XGBoost | 9.00 ± 4.36 | [6.79, 11.20] | 12.82 |
| 1D CNN | 12.69 ± 7.94 | [8.67, 16.71] | 16.06 |
| **CNN-BiLSTM** | **9.50 ± 6.52** | **[6.09, 12.92]** | **13.55** |

**Key finding:** Random Forest vs. CNN-BiLSTM is statistically equivalent (Wilcoxon p = 0.82). Both significantly outperform the 1D CNN (p < 0.001).

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
- Classical ML: Linear Regression, Random Forest, XGBoost (46 hand-engineered PPG+ACC features)
- Deep Learning: 1D CNN (~59K params), CNN-BiLSTM with causal smoothing (~385K params)

**Evaluation protocol:** 15-fold LOSO, MAE + RMSE, 95% CIs, Wilcoxon signed-rank tests for statistical significance.

---

## Dataset Setup (Required)

The dataset is **NOT** included in this repository. Each team member must download it manually.

1. Download [PPG-DaLiA](https://ubicomp.eti.uni-siegen.de/home/datasets/sensors19/) from the original source
2. Extract and place under `data/raw/`:

```
data/
└── raw/
    └── ppg+dalia/
        └── PPG_FieldStudy/
            ├── S1/
            │   └── S1.pkl
            ├── S2/
            │   └── S2.pkl
            └── ... (S1–S15)
```

---

## Project Structure

```
DSC_445_FINAL/
├── data/
│   └── raw/                         # PPG-DaLiA dataset (NOT in repo — see above)
│
├── src/
│   │── Week 1 ────────────────────────────────────────────────────────────
│   ├── data_loader.py               # Load subject .pkl files
│   ├── preprocessing.py             # Butterworth bandpass + z-score normalization
│   ├── loso.py                      # LOSO splitter (no leakage guaranteed)
│   ├── sanity_checks.py             # Shape/leakage/HR-range assertions
│   ├── run_week1_pipeline.py        # Week 1 orchestrator
│   │
│   │── Week 2 ────────────────────────────────────────────────────────────
│   ├── features.py                  # 46 hand-crafted PPG+ACC features (pure numpy)
│   ├── metrics.py                   # RMSE, MAE, fold summarizer
│   ├── models_classical.py          # LR / RF / XGBoost LOSO training
│   ├── evaluate_classical.py        # Summary table + ACC-uplift analysis
│   ├── plots_week2.py               # Bar/box charts, pred-vs-true, Bland-Altman
│   ├── run_week2_pipeline.py        # Week 2 orchestrator
│   ├── linear_rf_windows.py         # Window-level classical pipeline (bugfixed)
│   │
│   │── Week 3 ────────────────────────────────────────────────────────────
│   ├── dl_dataset.py                # WindowDataset + get_loso_split()
│   ├── dl_preprocessing.py          # PPG+ACC windowing → .npy files (bugfixed)
│   ├── models_dl.py                 # CNNModel + CNNLSTMModel definitions
│   ├── train_dl.py                  # Training loop, LOSO, early stopping, smoothing
│   ├── plots_week3.py               # DL result figures
│   └── run_week3_pipeline.py        # Week 3 orchestrator
│
├── outputs/
│   ├── processed_data/              # Windowed .npy files (S1_ppg_acc.npy, S1_y.npy, …)
│   ├── week2/                       # Classical ML results
│   │   ├── results_classical.csv    # Per-fold results (model × input × subject)
│   │   ├── summary_classical.csv    # Mean ± std per model
│   │   └── figures/
│   └── week3/                       # DL results
│       ├── loso_CNN.csv
│       ├── loso_CNN_LSTM.csv
│       ├── summary_CNN.json
│       ├── summary_CNN_LSTM.json
│       ├── figures/
│       └── analysis/                # Statistical analyses + all paper figures
│           ├── results_with_ci.csv      # 95% CIs for all models
│           ├── wilcoxon_results.csv     # Pairwise significance tests
│           ├── fair_comparison.csv      # Equal-smoothing comparison
│           ├── feature_importance.csv   # |Pearson r| per feature
│           ├── activity_performance.csv # Per-activity MAE
│           ├── fig_mae_with_ci.png
│           ├── bland_altman_*.png
│           ├── prediction_*.png
│           └── s5_s8_analysis.png
│
├── DSC_445_Final/
│   └── ppg_hr_estimation_report_v8.docx   # Final report
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
# Also needed for DL:
pip install torch  # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Week 1 — Data loading & windowing

```bash
python src/run_week1_pipeline.py
```

### Week 2 — Classical ML (LOSO, ~20 min on laptop)

```bash
# Full run
python src/run_week2_pipeline.py

# Quick smoke test (3 folds)
python src/run_week2_pipeline.py --smoke

# Skip preprocess/features if re-running models only
python src/run_week2_pipeline.py --skip-clean --skip-features
```

### Week 3 — Deep Learning

```bash
# Step 1: Preprocess windows to .npy (skip if already done)
python src/dl_preprocessing.py

# Step 2: Train CNN and CNN-BiLSTM (LOSO, ~2–4 hrs CPU / ~30 min GPU)
python src/run_week3_pipeline.py

# To skip preprocessing and jump straight to training:
#   Comment out preprocess_all_subjects() in run_week3_pipeline.py
#   or check that outputs/processed_data/ already has S*.npy files
```

Results are saved to `outputs/week3/`.

---

## Week 1 — Data Pipeline

- Loaded all 15 subject `.pkl` files from `data/raw/ppg+dalia/PPG_FieldStudy/`
- Segmented PPG (64 Hz) and ACC (32 Hz) into 8-second windows with 2-second stride
- LOSO splitter with leakage assertion
- Sanity checks: no subject overlap, valid HR ranges (40–200 bpm), aligned shapes
- **Total windows:** ~64,697 | **PPG shape:** `(N, 512, 1)` | **ACC shape:** `(N, 256, 3)`

---

## Week 2 — Classical ML

**46 hand-crafted features per window** (pure NumPy, no leakage):
- **PPG time-domain (9):** mean, std, skewness, kurtosis, peak-to-peak, IQR, zero-crossing rate, slope, RMS
- **PPG frequency-domain (9):** band powers (0.5–1, 1–2, 2–4 Hz), peak frequency, spectral entropy, band ratios, peak amplitude/width
- **ACC features (13):** equivalent stats on 3D magnitude + per-axis std + inter-axis correlations
- **Cross-modal (6):** PPG–ACC correlation, coherence, phase alignment, movement intensity, SNR estimate, breathing rate proxy

**Models:** Linear Regression (baseline), Random Forest (100 trees), XGBoost (100 estimators, max_depth=6)

**Outputs** → `outputs/week2/`:

```
results_classical.csv    # 90 rows: model × input_config × subject
summary_classical.csv    # Mean ± std across 15 folds
acc_uplift.csv           # ΔRMSE(ppg+acc vs ppg_only) per subject
```

---

## Week 3 — Deep Learning

### Preprocessing (`dl_preprocessing.py`)

- Bandpass filter: Butterworth 4th-order, 0.5–4 Hz on PPG
- ACC resampled 32 Hz → 64 Hz (Fourier method)
- Global normalization per subject
- Windows: 512 samples (8 s), stride 128 (2 s) → 4-channel tensor `(N, 512, 4)`
- **Label alignment:** `hr_idx = window_end // 128` (= 64 Hz / 0.5 Hz label rate) — see Bug Fixes

### Architectures

**1D CNN** (~59K params):
- 3× Conv1D blocks (32→64→128 filters) with BatchNorm, ReLU, MaxPool
- Global average pooling → FC(256) → FC(128) → scalar output

**CNN-BiLSTM** (~385K params):
- Same CNN backbone + 2-layer bidirectional LSTM (128 hidden units)
- Readout: forward `h[-1]` + backward `h[0]` concatenated (256-dim)
- Causal moving-median smoothing post-inference (window=5, 10 s)

### Training

- Optimizer: Adam, L1 loss
- Early stopping: patience=8 on 10%-held-out validation MAE
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=4, min_lr=1e-5)
- Gradient clipping: max_norm=5.0
- Reproducibility: global SEED=42 + per-fold seed (SEED+fold_i), deterministic cuDNN

### Results

| Model | MAE (bpm) | MAE SD | RMSE (bpm) | RMSE SD |
|-------|-----------|--------|------------|---------|
| 1D CNN | 12.694 | 7.671 | 16.055 | 8.620 |
| CNN-BiLSTM | 9.504 | 6.522 | 13.550 | 9.163 |

**Outlier subjects:**
- **S5** (MAE ~27 bpm): mean HR 125.8 bpm vs. dataset average ~87 bpm — physiological outlier, not sensor failure
- **S8** (MAE ~23 bpm, DL only): unusual PPG waveform morphology causes DL distribution shift; classical RF degrades much less (12.3 bpm)

---

## Bug Fixes & Reproducibility

### Critical bug fix — label alignment index

Both `dl_preprocessing.py` and `linear_rf_windows.py` originally computed:
```python
hr_idx = end // 32   # ❌ WRONG — paired each window with a label 4× too far ahead
```
Corrected to:
```python
hr_idx = end // 128  # ✅ correct: 64 Hz PPG / 0.5 Hz label rate = 128 samples per label
```
This fix materially changed all results. All numbers in the final report use the corrected pipeline.

### Reproducibility

Full deterministic reproduction requires setting:
```python
import random, numpy as np, torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Plus per-fold: set_seed(42 + fold_index)
```
This is implemented in `train_dl.py`. Minor numeric differences (~0.1–0.5 bpm) may occur across different hardware (CPU vs GPU) due to floating-point non-associativity.

---

## Contributors

| Name                | Role                                                                                                 |
|---------------------|------------------------------------------------------------------------------------------------------|
| Ekaterina Golovkina | Pipeline architecture,feature engineering, data preprocessing classical ML, DL training, report lead |
|                     |                                                        |
|                     |                                                                    |
|                     |                                                                 |
|                     |                                                      |

---

