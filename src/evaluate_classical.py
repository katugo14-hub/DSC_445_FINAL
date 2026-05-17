"""
Week 2 evaluation: load per-fold classical results, print a clean comparison
table, compute the paired ACC-uplift per model, and write summary CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK2_DIR = PROJECT_ROOT / "outputs" / "week2"


def summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """mean ± std across folds per (model, input)."""
    g = (
        results_df.groupby(["model", "input"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["model", "input"])
    )
    return g


def acc_uplift(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fold paired comparison:
        delta_rmse = RMSE(ppg+acc) - RMSE(ppg_only)
        delta_mae  = MAE(ppg+acc)  - MAE(ppg_only)
    Negative deltas mean ACC features help. One row per (model, subject).
    """
    pivot_rmse = results_df.pivot_table(
        index=["model", "subject"], columns="input", values="rmse"
    ).reset_index()
    pivot_mae = results_df.pivot_table(
        index=["model", "subject"], columns="input", values="mae"
    ).reset_index()

    if "ppg_only" not in pivot_rmse or "ppg+acc" not in pivot_rmse:
        raise ValueError("Need both 'ppg_only' and 'ppg+acc' configs in results")

    out = pivot_rmse[["model", "subject"]].copy()
    out["rmse_ppg_only"] = pivot_rmse["ppg_only"]
    out["rmse_ppg_acc"] = pivot_rmse["ppg+acc"]
    out["delta_rmse"] = out["rmse_ppg_acc"] - out["rmse_ppg_only"]

    out["mae_ppg_only"] = pivot_mae["ppg_only"].values
    out["mae_ppg_acc"] = pivot_mae["ppg+acc"].values
    out["delta_mae"] = out["mae_ppg_acc"] - out["mae_ppg_only"]
    return out.sort_values(["model", "subject"])


def print_table(g: pd.DataFrame) -> None:
    lines = [
        f"{'model':<17} {'input':<10} {'RMSE_mean ± std':>20} {'MAE_mean ± std':>20} {'n':>4}"
    ]
    for _, r in g.iterrows():
        lines.append(
            f"{r['model']:<17} {r['input']:<10} "
            f"{r['rmse_mean']:>10.2f} ± {r['rmse_std']:>5.2f}   "
            f"{r['mae_mean']:>10.2f} ± {r['mae_std']:>5.2f} "
            f"{int(r['n_folds']):>4d}"
        )
    print("\n".join(lines))


def main(out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or WEEK2_DIR)
    results_path = out_dir / "results_classical.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run models_classical.run_classical_baselines first."
        )
    df = pd.read_csv(results_path)

    g = summary_table(df)
    g.to_csv(out_dir / "summary_classical.csv", index=False)
    print("=== Summary (mean ± std across LOSO folds) ===")
    print_table(g)

    uplift = acc_uplift(df)
    uplift.to_csv(out_dir / "acc_uplift.csv", index=False)

    print("\n=== ACC uplift (RMSE delta: ppg+acc minus ppg_only, per model) ===")
    agg = (
        uplift.groupby("model")
        .agg(
            mean_delta_rmse=("delta_rmse", "mean"),
            n_negative_folds=("delta_rmse", lambda s: int((s < 0).sum())),
            n_folds=("delta_rmse", "count"),
        )
        .reset_index()
    )
    for _, r in agg.iterrows():
        sign = "improves" if r["mean_delta_rmse"] < 0 else "worsens"
        print(
            f"  {r['model']:<17} mean ΔRMSE = {r['mean_delta_rmse']:+.2f} bpm "
            f"({sign}); ACC helps in {r['n_negative_folds']}/{int(r['n_folds'])} folds"
        )


if __name__ == "__main__":
    main()
