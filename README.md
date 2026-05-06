# PPG Heart Rate Estimation (DSC 445 Final Project)

## Overview
This project focuses on improving heart rate estimation from PPG signals using machine learning under motion conditions.

We use the **PPG-DaLiA dataset** and evaluate models using **subject-independent validation (LOSO)** to ensure real-world generalization.

---

## Week 1 Progress (Completed ✅)

A complete and reproducible data pipeline has been implemented:

- Loaded all subject data
- Segmented signals using:
  - 8-second windows
  - 2-second shift
- Created Leave-One-Subject-Out (LOSO) split
- Performed sanity checks:
  - No data leakage
  - Valid heart rate ranges
  - Correct data shapes
- Saved processed arrays for modeling

### Dataset Summary
- Total segments: ~64,697
- PPG shape: `(N, 512, 1)`
- ACC shape: `(N, 256, 3)`
- HR range: ~41–187 bpm

---

## Project Structure

```
DSC_445_FINAL/
├── data/
│   └── raw/              # PPG-DaLiA dataset (NOT included in repo)
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── loso.py
│   ├── sanity_checks.py
│   ├── run_week1_pipeline.py
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
python src/run_week1_pipeline.py
```

---

## Output Files

Generated in `outputs/`:

- `X_ppg.npy` → segmented PPG signals
- `X_acc.npy` → segmented accelerometer data
- `y.npy` → heart rate labels
- `subjects.npy` → subject IDs for LOSO validation

---

## Next Steps

- **Week 2:** Classical ML models (Linear Regression, Random Forest, XGBoost)
- **Week 3:** Deep Learning models (CNN, LSTM)
- **Week 4:** Evaluation + final report

---

## Notes

- LOSO validation is used to prevent data leakage and ensure subject generalization
- Random train/test splits are NOT used
- Dataset and outputs are excluded from GitHub via `.gitignore`

---

## Contributors

- Ekaterina Golovkina
- Team Members (to be updated)