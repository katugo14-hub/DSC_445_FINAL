import copy
import json
import csv
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from dl_dataset import get_loso_split, get_all_subjects, WindowDataset
from models_dl import CNNModel, CNNLSTMModel

# ── Global seed for full reproducibility ─────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "week3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Low-level training helpers ────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Returns MAE and RMSE over the full loader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).squeeze()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    mae  = float(np.mean(np.abs(preds_np - targets_np)))
    rmse = float(np.sqrt(np.mean((preds_np - targets_np) ** 2)))
    return mae, rmse


def causal_smooth(preds, window=5):
    """Causal moving-median over a 1-D prediction sequence.
    Each output value uses only the current and (window-1) past predictions,
    so no future information leaks into the evaluation.
    Window=5 covers 10 s of signal at 2-second stride — physiologically motivated:
    HR cannot change faster than ~1-2 bpm/beat, so sharp jumps are artifacts.
    """
    smoothed = np.empty_like(preds)
    for i in range(len(preds)):
        start = max(0, i - window + 1)
        smoothed[i] = np.median(preds[start:i + 1])
    return smoothed


# ── Hyperparameter tuning (fast 18-combination grid on one val subject) ───────

def tune_hyperparameters(model_class, val_subject="S2", device="cpu"):
    """
    Runs a compact 18-trial grid (not 162!) against one held-out validation
    subject and returns the best hyperparameter dict.
    Each trial trains for 5 quick epochs — just enough to rank configs.
    """
    all_subjects = get_all_subjects()
    X_train, y_train, X_val, y_val = get_loso_split(val_subject, all_subjects)

    train_ds = WindowDataset(X_train, y_train)
    val_ds   = WindowDataset(X_val,   y_val)

    # Compact search space — 3×2×3 = 18 combos (was 162, took hours)
    search_space = [
        {"lr": lr, "dropout": dropout, "batch_size": bs, "num_filters": nf, "kernel_size": ks}
        for lr      in [3e-4, 1e-3]
        for dropout in [0.2, 0.4]
        for bs      in [32, 64]
        for nf      in [32, 64]
        for ks      in [5, 7]
    ]

    criterion     = nn.L1Loss()
    best_val_mae  = float("inf")
    best_hparams  = None
    TUNE_EPOCHS   = 5   # cheap ranking pass

    print(f"[tune] Running {len(search_space)} hyperparameter combinations on {val_subject}...")
    for i, hp in enumerate(search_space, 1):
        model     = model_class(num_filters=hp["num_filters"],
                                dropout=hp["dropout"],
                                kernel_size=hp["kernel_size"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, num_workers=0)

        for _ in range(TUNE_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_mae, _ = evaluate(model, val_loader, device)
        print(f"  [{i:2d}/{len(search_space)}] lr={hp['lr']:.0e} do={hp['dropout']} "
              f"bs={hp['batch_size']} nf={hp['num_filters']} ks={hp['kernel_size']} "
              f"→ val MAE={val_mae:.3f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_hparams = hp

    print(f"\n[tune] Best hparams (val MAE={best_val_mae:.3f}): {best_hparams}")
    return best_hparams


# ── Full LOSO with early stopping ─────────────────────────────────────────────

def train_loso(model_class, hparams, device="cpu", max_epochs=40, patience=8,
               model_name="model"):
    """
    LOSO cross-validation with early stopping (patience on val loss from a
    10% held-out slice of each training fold).
    Saves per-fold results + aggregate summary to outputs/week3/.
    Returns dict: {subject: {"mae": float, "rmse": float}}
    """
    all_subjects = get_all_subjects()
    results      = {}
    rows         = []   # for CSV

    for fold_i, test_sub in enumerate(all_subjects):
        set_seed(SEED + fold_i)   # deterministic per fold, independent across folds
        t0 = time.time()
        print(f"\n[LOSO] {model_name} | held-out: {test_sub}")

        X_train_full, y_train_full, X_test, y_test = get_loso_split(test_sub, all_subjects)

        # Hold out 10% of training data as a validation set for early stopping
        n_val   = max(1, int(len(X_train_full) * 0.10))
        idx     = np.random.permutation(len(X_train_full))
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]

        X_tr,  y_tr  = X_train_full[tr_idx],  y_train_full[tr_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        train_ds = WindowDataset(X_tr,   y_tr)
        val_ds   = WindowDataset(X_val,  y_val)
        test_ds  = WindowDataset(X_test, y_test)

        bs = hparams["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

        model = model_class(num_filters=hparams["num_filters"],
                            dropout=hparams["dropout"],
                            kernel_size=hparams["kernel_size"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
        criterion = nn.L1Loss()

        # LR scheduler — halves LR after 4 epochs of no val improvement
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )

        # Early stopping
        best_val_mae   = float("inf")
        best_state     = copy.deepcopy(model.state_dict())
        patience_count = 0

        for epoch in range(1, max_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, _ = evaluate(model, val_loader, device)
            scheduler.step(val_mae)          # adjust LR based on val MAE

            if val_mae < best_val_mae - 1e-4:
                best_val_mae   = val_mae
                best_state     = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                print(f"  Early stop at epoch {epoch} (val MAE={best_val_mae:.3f})")
                break

        # Restore best weights, collect raw test predictions, then smooth
        model.load_state_dict(best_state)
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                all_preds.append(model(X).squeeze().cpu().numpy())
                all_targets.append(y.numpy())
        raw_preds  = np.concatenate(all_preds)
        targets_np = np.concatenate(all_targets)

        # Causal median smoothing (window=5 → 10 s at 2 s stride)
        smoothed_preds = causal_smooth(raw_preds, window=5)

        test_mae  = float(np.mean(np.abs(smoothed_preds - targets_np)))
        test_rmse = float(np.sqrt(np.mean((smoothed_preds - targets_np) ** 2)))
        elapsed   = time.time() - t0

        results[test_sub] = {"mae": test_mae, "rmse": test_rmse}
        rows.append({"model": model_name, "subject": test_sub,
                     "mae": round(test_mae, 4), "rmse": round(test_rmse, 4)})

        print(f"  {test_sub}: MAE={test_mae:.3f}  RMSE={test_rmse:.3f}  ({elapsed:.0f}s)")

    # ── Aggregate summary ──────────────────────────────────────────────────
    maes  = [v["mae"]  for v in results.values()]
    rmses = [v["rmse"] for v in results.values()]
    summary = {
        "model":     model_name,
        "mae_mean":  round(float(np.mean(maes)),  3),
        "mae_std":   round(float(np.std(maes)),   3),
        "rmse_mean": round(float(np.mean(rmses)), 3),
        "rmse_std":  round(float(np.std(rmses)),  3),
        "n_folds":   len(results),
    }

    print(f"\n{'='*50}")
    print(f"[{model_name}] LOSO summary ({summary['n_folds']} folds)")
    print(f"  MAE  = {summary['mae_mean']:.3f} ± {summary['mae_std']:.3f} bpm")
    print(f"  RMSE = {summary['rmse_mean']:.3f} ± {summary['rmse_std']:.3f} bpm")
    print(f"{'='*50}")

    # ── Save results ───────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"loso_{model_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "subject", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] per-fold results → {csv_path}")

    json_path = RESULTS_DIR / f"summary_{model_name}.json"
    with open(json_path, "w") as f:
        json.dump({**summary, "per_subject": results}, f, indent=2)
    print(f"[save] summary          → {json_path}")

    return results, summary











        






















                                




























    






















            
