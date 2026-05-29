# PPG Heart Rate Estimation (DSC 445 Final Project)

## Overview
This project improves heart rate estimation from PPG signals using machine
learning under motion conditions. We use the **PPG-DaLiA** dataset and
evaluate models under **subject-independent (LOSO)** cross-validation.

---

## Week 1 Progress (Completed)

A complete and reproducible data pipeline:

- Loaded all subject data from `data/raw/ppg+dalia/PPG_FieldStudy/`
- Segmented PPG and accelerometer signals with 8-second windows / 2-second shift
- Built Leave-One-Subject-Out (LOSO) splitter
- Sanity checks: no subject leakage, valid HR ranges, aligned shapes
- Saved processed arrays for downstream modeling

### Dataset summary
- Total segments: ~64,697
- PPG shape: `(N, 512, 1)` at 64 Hz
- ACC shape: `(N, 256, 3)` at 32 Hz
- HR range: ~41–187 bpm

Run with:

```bash
python src/run_week1_pipeline.py
```

---

## Week 2 Progress (Completed)

Classical-ML baselines and the supporting infrastructure that Week 1 had
left as TODOs (feature extraction + evaluation metrics).

What was implemented:

- **`src/features.py`** — handcrafted PPG and ACC features per window
  (time-domain stats, FFT band powers, spectral centroid/entropy/bandwidth,
  dominant frequency, plus a naive HR-from-peak-FFT-bin estimate). 46
  features in total (23 PPG + 23 ACC). Pure numpy.
- **`src/metrics.py`** — RMSE, MAE, and `summarize_fold` helper.
- **`src/preprocessing.py`** — added Butterworth bandpass (0.5–4 Hz, 4th
  order) + per-window z-score + `clean_arrays()` to drop windows with
  non-finite or out-of-range labels. Backward compatible (Week 1 pipeline
  unchanged).
- **`src/dl_dataset.py`** — PyTorch `Dataset` + `get_loso_loaders()` for
  Week 3 deep models (shape-check only, no training yet).
- **`src/models_classical.py`** — LinearRegression / RandomForest / XGBoost
  trained under LOSO with two input configurations (`ppg_only`, `ppg+acc`),
  scaler fit on TRAIN ONLY each fold, per-fold predictions persisted.
- **`src/evaluate_classical.py`** — summary table + paired ACC-uplift
  analysis (RMSE(`ppg+acc`) − RMSE(`ppg_only`) per fold, answers RQ1).
- **`src/plots_week2.py`** — bar/box charts, pred-vs-true scatter coloured
  by subject, Bland-Altman.
- **`src/run_week2_pipeline.py`** — end-to-end orchestrator.

### How to run

```bash
# install deps (one time)
pip install -r requirements.txt

# full 15-fold LOSO (≈ tens of minutes on a laptop)
python src/run_week2_pipeline.py

# quick correctness check on 3 folds first
python src/run_week2_pipeline.py --smoke

# skip preprocess/feature extraction if rerunning models
python src/run_week2_pipeline.py --skip-clean --skip-features
```

### Outputs (all under `outputs/week2/`)

```
outputs/week2/
├── clean/                         # filtered + zscored arrays (kept separate
│   ├── X_ppg.npy                  #   from Week 1's outputs/)
│   ├── X_acc.npy
│   ├── y.npy
│   └── subjects.npy
├── X_features.npy                 # (N, 46) handcrafted feature matrix
├── feature_names.json
├── results_classical.csv          # one row per (model, input, fold)
├── summary_classical.csv          # mean ± std across folds per (model, input)
├── acc_uplift.csv                 # paired ΔRMSE per model per subject
├── predictions/
│   └── <model>__<input>__<S>.npz  # y_true, y_pred per fold (for plots)
└── figures/
    ├── fig_rmse_bar.png
    ├── fig_mae_bar.png
    ├── fig_rmse_box.png
    ├── fig_scatter_best.png
    └── fig_bland_altman_best.png
```

### Week 2 check-in
- Classical model results completed (3 models × 2 configs × 15 folds = 90 rows).
- Per-fold RMSE / MAE computed; aggregated mean ± std in `summary_classical.csv`.
- ACC-vs-no-ACC paired comparison ready for the writeup.
- All scaling fit on TRAIN ONLY each fold.
- No subject leakage (asserted by `sanity_checks.check_no_subject_leakage`).

---

## Project Structure

```
DSC_445_FINAL/
├── data/
│   └── raw/                       # PPG-DaLiA (NOT in repo)
├── src/
│   ├── data_loader.py             # Week 1
│   ├── preprocessing.py           # Week 1 + Week 2 additions
│   ├── loso.py                    # Week 1
│   ├── sanity_checks.py           # Week 1
│   ├── run_week1_pipeline.py
│   ├── features.py                # Week 2
│   ├── metrics.py                 # Week 2
│   ├── dl_dataset.py              # Week 2 (used in Week 3)
│   ├── models_classical.py        # Week 2
│   ├── evaluate_classical.py      # Week 2
│   ├── plots_week2.py             # Week 2
│   └── run_week2_pipeline.py      # Week 2
├── outputs/                       # Week 1 arrays + Week 2 results
├── requirements.txt
└── README.md
```

---

## Dataset Setup (Required)

The dataset is **NOT** included in this repository.

Each team member must:

1. Download the PPG-DaLiA dataset
2. Extract it
3. Place it under `data/raw/`:

```
data/
└── raw/
    └── ppg+dalia/
        └── PPG_FieldStudy/
            ├── S1/
            ├── S2/
            └── ...
```

---

## Next Steps

- **Week 3:** Deep learning — train CNN and LSTM models on the cleaned
  raw windows using `src/dl_dataset.py`; hyperparameter-tune the classical
  baselines.
- **Week 4:** Final evaluation (LOSO RMSE/MAE + motion-based comparison,
  Bland-Altman), writeup, slides.

---

## Notes

- LOSO validation is enforced everywhere. Random train/test splits are not
  used at any point in the pipeline.
- Dataset and large output files are excluded from git via `.gitignore`.

---

## Contributors

- Ekaterina Golovkina
- Aayesha Kaleem Syeda
- Alex Turczynski
- (Afshaan Fathima Syeda)
- (Member 5)
