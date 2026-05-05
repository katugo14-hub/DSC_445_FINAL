# PPG Heart Rate Estimation (DSC 445 Final Project)

## Overview
This project focuses on improving heart rate estimation from PPG signals using machine learning under motion conditions.

We use the PPG-DaLiA dataset and evaluate models using subject-independent validation.

---

## Week 1 Progress (Completed ✅)

Implemented a full data pipeline:
- Loaded all subject data
- Segmented signals using:
  - 8-second windows
  - 2-second shift
- Created Leave-One-Subject-Out (LOSO) split
- Performed sanity checks:
  - No data leakage
  - Valid label ranges
  - Correct shapes
- Saved processed data

---

## Project Structure
DSC_445_FINAL/
├── data/
│   └── raw/              # PPG-DaLiA dataset
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── loso.py
│   ├── sanity_checks.py
│   ├── run_week1_pipeline.py
├── outputs/
│   ├── X_ppg.npy
│   ├── X_acc.npy
│   ├── y.npy
│   ├── subjects.npy
├── README.md

---

## How to Run

```bash
python src/run_week1_pipeline.py