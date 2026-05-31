import numpy as np
import pickle
import glob
import os
import gc
from pathlib import Path
from scipy.signal import butter, filtfilt, resample

#Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
# Raw subject pkl files live at: data/raw/ppg+dalia/PPG_FieldStudy/S<N>/S<N>.pkl
RAW_DIR      = DATA_DIR / "raw" / "ppg+dalia" / "PPG_FieldStudy"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "processed_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_subject(path):
    """load one subject safely"""
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def preprocess_subject(subject_path):
    """process ONE subject and save X/y"""
    gc.collect()     #free RAM before loading

    subject_id = Path(subject_path).stem
    print(f"\n---Processing {subject_id}---")

    data = load_subject(subject_path)

    #extract signals
    ppg = data["signal"]["wrist"]["BVP"]
    acc = data["signal"]["wrist"]["ACC"]
    hr = data["label"]          # PPG-DaLiA: data["label"] is the HR array directly

    #filter PPG
    fs = 64
    b, a = butter(4, [0.5/(fs/2), 4/(fs/2)], btype="band")
    ppg_f = filtfilt(b, a, ppg.squeeze())

    #resample ACC from 32 Hz -> 64 Hz
    acc_resampled = resample(acc, len(ppg_f))

    #Normalize
    ppg_norm = (ppg_f - np.mean(ppg_f)) / (np.std(ppg_f) + 1e-8)
    acc_norm = (acc_resampled - np.mean(acc_resampled, axis=0))/(np.std(acc_resampled, axis=0) + 1e-8)

    #combine into 4-channel signal
    X_full = np.column_stack([ppg_norm, acc_norm]).astype(np.float32)

    #Windowing
    X, y = [], []
    win = 512
    step = 128

    for start in range(0, len(X_full) - win, step):
        end = start + win
        X.append(X_full[start:end])
        
        hr_idx = end//128  # PPG@64Hz / HR_labels@0.5Hz = 128 samples per label
        if hr_idx < len(hr):
            y.append(hr[hr_idx])
        else:
            y.append(hr[-1])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    np.save(OUTPUT_DIR / f"{subject_id}_ppg_acc.npy", X)   # shape (N, 512, 4)
    np.save(OUTPUT_DIR / f"{subject_id}_y.npy", y)         # shape (N,)

    print(f"Saved {len(X)} windows for {subject_id}.")


    #free memory
    del data, ppg, acc, hr, X_full, X, y
    gc.collect()


def preprocess_all_subjects():
    """process all subjects safely"""
    # Files are nested: data/raw/ppg+dalia/PPG_FieldStudy/S<N>/S<N>.pkl
    subject_files = sorted(glob.glob(str(RAW_DIR / "S*" / "*.pkl")))

    if not subject_files:
        print("No subject files found. Check DATA_DIR path.")
        return
    
    for subject_path in subject_files:
        preprocess_subject(subject_path)

    print("[done] All subjects processed")

    
