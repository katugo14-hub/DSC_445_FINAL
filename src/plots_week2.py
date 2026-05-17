"""
Week 2 figures (saved to outputs/week2/figures/):
    fig_rmse_bar.png         — mean RMSE per (model, input) ± std
    fig_mae_bar.png          — mean MAE  per (model, input) ± std
    fig_rmse_box.png         — per-fold RMSE distribution per (model, input)
    fig_scatter_best.png     — pred vs true HR (best model, PPG+ACC), per subject
    fig_bland_altman_best.png — Bland-Altman for best model PPG+ACC
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK2_DIR = PROJECT_ROOT / "outputs" / "week2"
FIG_DIR = WEEK2_DIR / "figures"


def _load(out_dir: Path):
    results = pd.read_csv(out_dir / "results_classical.csv")
    summary = pd.read_csv(out_dir / "summary_classical.csv")
    return results, summary


def _bar_plot(summary: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="model", y=f"{metric}_mean",
        hue="input", ax=ax,
        errorbar=None,
    )
    # add manual error bars from *_std
    width = 0.4
    models = list(summary["model"].unique())
    inputs = list(summary["input"].unique())
    for ix, m in enumerate(models):
        for iy, c in enumerate(inputs):
            row = summary[(summary["model"] == m) & (summary["input"] == c)]
            if row.empty:
                continue
            x_pos = ix + (iy - (len(inputs) - 1) / 2) * (width)
            y_val = row[f"{metric}_mean"].iloc[0]
            y_err = row[f"{metric}_std"].iloc[0]
            ax.errorbar(x_pos, y_val, yerr=y_err, fmt="none",
                        ecolor="black", capsize=4, lw=1)
    ax.set_ylabel(f"{metric.upper()} (bpm)")
    ax.set_xlabel("model")
    ax.set_title(f"LOSO {metric.upper()} by model and input config")
    ax.legend(title="input")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _box_plot(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=results, x="model", y="rmse", hue="input", ax=ax)
    sns.stripplot(
        data=results, x="model", y="rmse", hue="input",
        ax=ax, dodge=True, color="black", alpha=0.5, size=3, legend=False,
    )
    ax.set_ylabel("RMSE (bpm) — per held-out subject")
    ax.set_xlabel("model")
    ax.set_title("Per-fold LOSO RMSE distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _best_model(summary: pd.DataFrame) -> tuple[str, str]:
    """Lowest RMSE_mean wins."""
    row = summary.sort_values("rmse_mean").iloc[0]
    return str(row["model"]), str(row["input"])


def _load_predictions(out_dir: Path, model: str, input_config: str):
    """Concatenate all per-fold predictions for one (model, input)."""
    pred_dir = out_dir / "predictions"
    y_true_all, y_pred_all, subj_all = [], [], []
    pattern = f"{model}__{input_config}__"
    for f in sorted(pred_dir.glob(f"{pattern}*.npz")):
        subj = f.stem.split("__")[-1]
        d = np.load(f)
        y_true_all.append(d["y_true"])
        y_pred_all.append(d["y_pred"])
        subj_all.append(np.array([subj] * len(d["y_true"])))
    if not y_true_all:
        raise FileNotFoundError(f"No predictions found for {model}/{input_config}")
    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(subj_all),
    )


def _scatter_pred_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray,
    title: str, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = sns.color_palette("tab20", n_colors=len(set(subjects.tolist())))
    for i, s in enumerate(sorted(set(subjects.tolist()))):
        m = subjects == s
        ax.scatter(y_true[m], y_pred[m], s=4, alpha=0.4, color=palette[i], label=s)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True HR (bpm)")
    ax.set_ylabel("Predicted HR (bpm)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _bland_altman(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std())
    loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean, diff, s=4, alpha=0.3)
    ax.axhline(bias, color="red", lw=1.5, label=f"bias = {bias:+.2f} bpm")
    ax.axhline(loa_low, color="grey", lw=1, ls="--", label=f"95% LoA = [{loa_low:+.1f}, {loa_high:+.1f}]")
    ax.axhline(loa_high, color="grey", lw=1, ls="--")
    ax.set_xlabel("Mean of pred & true HR (bpm)")
    ax.set_ylabel("Pred − True (bpm)")
    ax.set_title(f"Bland-Altman: {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_all_figures(out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or WEEK2_DIR)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results, summary = _load(out_dir)

    _bar_plot(summary, "rmse", fig_dir / "fig_rmse_bar.png")
    _bar_plot(summary, "mae", fig_dir / "fig_mae_bar.png")
    _box_plot(results, fig_dir / "fig_rmse_box.png")

    best_model, best_config = _best_model(summary)
    if best_config == "ppg_only":
        # prefer ppg+acc for the diagnostic plots if available
        cand = summary[(summary["model"] == best_model) & (summary["input"] == "ppg+acc")]
        if not cand.empty:
            best_config = "ppg+acc"
    print(f"[plots] best model: {best_model} ({best_config})")

    try:
        y_true, y_pred, subj = _load_predictions(out_dir, best_model, best_config)
        title = f"{best_model} ({best_config})"
        _scatter_pred_vs_true(y_true, y_pred, subj, title, fig_dir / "fig_scatter_best.png")
        _bland_altman(y_true, y_pred, title, fig_dir / "fig_bland_altman_best.png")
    except FileNotFoundError as e:
        print(f"[plots] skipping per-prediction plots: {e}")

    print(f"[plots] wrote figures to {fig_dir}")


if __name__ == "__main__":
    make_all_figures()
