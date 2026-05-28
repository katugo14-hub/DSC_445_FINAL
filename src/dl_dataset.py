import numpy as np
import os
from pathlib import Path

# ── Absolute paths — work regardless of where the script is launched from ──────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Week-2 cleaned arrays (used by classical pipeline)
_W2_CLEAN_DIR = PROJECT_ROOT / "outputs" / "week2" / "clean"

# Week-3 DL preprocessed per-subject arrays (written by dl_preprocessing.py)
_DL_PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed_data"


def load_all_data():
    """Load cleaned Week-2 arrays (PPG, ACC, labels, subject IDs)."""
    X_ppg    = np.load(_W2_CLEAN_DIR / "X_ppg.npy")
    X_acc    = np.load(_W2_CLEAN_DIR / "X_acc.npy")
    y        = np.load(_W2_CLEAN_DIR / "y.npy")
    subjects = np.load(_W2_CLEAN_DIR / "subjects.npy")
    return X_ppg, X_acc, y, subjects


def load_subject(subject_name, data_dir=None):
    """Load one subject's preprocessed DL arrays.
    Returns:
        X: (num_windows, 512, 4)  — PPG + ACC, 4-channel
        y: (num_windows,)         — HR labels in bpm
    """
    d = Path(data_dir) if data_dir else _DL_PROCESSED_DIR
    X = np.load(d / f"{subject_name}_ppg_acc.npy")
    y = np.load(d / f"{subject_name}_y.npy")
    return X, y


def get_all_subjects(data_dir=None):
    """Return a sorted list of all subjects available in the DL processed folder."""
    d = Path(data_dir) if data_dir else _DL_PROCESSED_DIR
    subjects = sorted({f.split("_")[0] for f in os.listdir(d) if f.endswith("_ppg_acc.npy")})
    return subjects


def get_loso_split(test_subject, all_subjects, data_dir=None):
    """LOSO split: test subject loaded fully, training subjects memory-mapped.
    Returns: X_train, y_train, X_test, y_test
    """
    d = Path(data_dir) if data_dir else _DL_PROCESSED_DIR

    # Test subject — full load
    X_test = np.load(d / f"{test_subject}_ppg_acc.npy")
    y_test = np.load(d / f"{test_subject}_y.npy")

    # Training subjects — memory-mapped to avoid RAM overload
    X_train_list, y_train_list = [], []
    for subject in all_subjects:
        if subject == test_subject:
            continue
        X_train_list.append(np.load(d / f"{subject}_ppg_acc.npy", mmap_mode="r"))
        y_train_list.append(np.load(d / f"{subject}_y.npy",       mmap_mode="r"))

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, y_train, X_test, y_test

#optional: PyTorch dataset wrapper
try:
    import torch
    from torch.utils.data import Dataset
    
    class WindowDataset(Dataset):
        """converts numpy windows into a PyTorch-friendly dataset"""

        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
except ImportError:
    #if PyTorch isn't installed yet, skip the Dataset class
    pass
    
    

                             
