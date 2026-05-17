"""
PyTorch Dataset + LOSO-aware DataLoader factory for Week 3 deep models.

Week 2 deliverable: only verify loaders iterate and shapes are correct.
No model is trained here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore[misc, assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "outputs" / "week2" / "clean"


class PPGDaLiAWindows(Dataset):  # type: ignore[misc]
    """
    Returns (ppg_window, acc_window, hr) or (concat, hr) depending on use_acc.

    ppg_window: torch.float32 of shape (1, 512)   -> channel-first for Conv1d
    acc_window: torch.float32 of shape (3, 256)   -> channel-first
    hr:         torch.float32 scalar
    """

    def __init__(
        self,
        X_ppg: np.ndarray,
        X_acc: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray | None = None,
        use_acc: bool = True,
    ):
        if not _HAS_TORCH:
            raise RuntimeError(
                "PyTorch not available. `pip install torch` to use the DL pipeline."
            )
        self.X_ppg = X_ppg
        self.X_acc = X_acc
        self.y = y
        self.indices = indices if indices is not None else np.arange(len(y))
        self.use_acc = use_acc

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # ppg: (512, 1) -> (1, 512)
        ppg = np.ascontiguousarray(self.X_ppg[idx]).reshape(-1)[None, :]
        # acc: (256, 3) -> (3, 256)
        acc = np.ascontiguousarray(self.X_acc[idx]).T  # (3, 256)
        hr = float(self.y[idx])

        ppg_t = torch.from_numpy(ppg.astype(np.float32))
        hr_t = torch.tensor(hr, dtype=torch.float32)
        if self.use_acc:
            acc_t = torch.from_numpy(acc.astype(np.float32))
            return ppg_t, acc_t, hr_t
        return ppg_t, hr_t


def load_clean_arrays(clean_dir: Path | None = None):
    """Load Week-2 cleaned arrays produced by run_week2_pipeline.py."""
    clean_dir = Path(clean_dir or CLEAN_DIR)
    X_ppg = np.load(clean_dir / "X_ppg.npy")
    X_acc = np.load(clean_dir / "X_acc.npy")
    y = np.load(clean_dir / "y.npy")
    subjects = np.load(clean_dir / "subjects.npy")
    return X_ppg, X_acc, y, subjects


def get_loso_loaders(
    fold_idx: int,
    batch_size: int = 64,
    use_acc: bool = True,
    num_workers: int = 0,
    clean_dir: Path | None = None,
):
    """
    Return (train_loader, test_loader, held_out_subject) for one LOSO fold.

    Fold ordering matches sorted(unique(subjects)).
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available.")

    X_ppg, X_acc, y, subjects = load_clean_arrays(clean_dir)
    unique = sorted(np.unique(subjects).tolist())
    if not 0 <= fold_idx < len(unique):
        raise IndexError(f"fold_idx {fold_idx} out of range; have {len(unique)} subjects")
    held = unique[fold_idx]

    test_mask = subjects == held
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    train_ds = PPGDaLiAWindows(X_ppg, X_acc, y, indices=train_idx, use_acc=use_acc)
    test_ds = PPGDaLiAWindows(X_ppg, X_acc, y, indices=test_idx, use_acc=use_acc)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, test_loader, held


def smoke_check(fold_idx: int = 0, batch_size: int = 16) -> None:
    """Sanity check: iterate one batch and print shapes."""
    train_loader, test_loader, held = get_loso_loaders(
        fold_idx, batch_size=batch_size, use_acc=True
    )
    print(f"[DL] held-out subject: {held}")
    print(f"[DL] train batches: {len(train_loader)}, test batches: {len(test_loader)}")
    for batch in train_loader:
        ppg, acc, hr = batch
        print(f"[DL] ppg {tuple(ppg.shape)} acc {tuple(acc.shape)} hr {tuple(hr.shape)}")
        break


if __name__ == "__main__":
    smoke_check()
