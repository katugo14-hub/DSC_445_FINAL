"""
Week 3 figures (saved to outputs/week3/figures/):

    fig1_model_comparison.png   — mean MAE ± SD for all 5 models (bar chart, paper-ready)
    fig2_per_subject.png        — per-subject MAE: XGBoost vs CNN vs CNN-BiLSTM (grouped bars)
    fig3_acc_uplift.png         — ACC uplift: PPG-only vs PPG+ACC for RF and XGBoost

Run from the project root OR from inside src/:
    python src/plots_week3.py
    python plots_week3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless — no display needed; remove if you want a popup
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEEK2_DIR    = PROJECT_ROOT / "outputs" / "week2"
WEEK3_DIR    = PROJECT_ROOT / "outputs" / "week3"
FIG_DIR      = WEEK3_DIR / "figures"

# ── color palette (colorblind-friendly) ───────────────────────────────────────
C_LR    = "#AAAAAA"   # light grey  — Linear Regression
C_RF    = "#5DAA68"   # green       — Random Forest
C_XGB   = "#2E75B6"   # blue        — XGBoost
C_CNN   = "#E07B39"   # orange      — CNN
C_LSTM  = "#C0392B"   # red         — CNN-BiLSTM

SUBJECT_ORDER = [f"S{i}" for i in range(1, 16)]


# ── data loading ──────────────────────────────────────────────────────────────

def load_classical_summary() -> pd.DataFrame:
    """Returns summary_classical.csv from Week 2 outputs."""
    return pd.read_csv(WEEK2_DIR / "summary_classical.csv")


def load_classical_per_subject() -> pd.DataFrame:
    """
    Returns per-subject MAE for the best input config (PPG+ACC) for each classical model.
    Columns: subject, model, mae, rmse
    """
    df = pd.read_csv(WEEK2_DIR / "results_classical.csv")
    df = df[df["input"] == "ppg+acc"].copy()
    df["subject"] = df["subject"].str.upper()
    return df[["subject", "model", "mae", "rmse"]]


def load_dl_summary() -> pd.DataFrame:
    """Returns a summary DataFrame for CNN and CNN-LSTM."""
    rows = []
    for fname, model_name in [("summary_CNN.json", "CNN"),
                               ("summary_CNN_LSTM.json", "CNN-BiLSTM")]:
        with open(WEEK3_DIR / fname) as f:
            d = json.load(f)
        rows.append({
            "model":    model_name,
            "mae_mean": d["mae_mean"],
            "mae_std":  d["mae_std"],
            "rmse_mean":d["rmse_mean"],
            "rmse_std": d["rmse_std"],
        })
    return pd.DataFrame(rows)


def load_dl_per_subject() -> pd.DataFrame:
    """Returns per-subject MAE for CNN and CNN-BiLSTM from the LOSO CSVs."""
    frames = []
    for fname, model_name in [("loso_CNN.csv", "CNN"),
                               ("loso_CNN_LSTM.csv", "CNN-BiLSTM")]:
        df = pd.read_csv(WEEK3_DIR / fname)
        df["model"] = model_name
        df["subject"] = df["subject"].str.upper()
        frames.append(df[["subject", "model", "mae", "rmse"]])
    return pd.concat(frames, ignore_index=True)


# ── figure 1: all-model MAE comparison ───────────────────────────────────────

def fig1_model_comparison(out_path: Path) -> None:
    """
    Horizontal grouped bar chart comparing mean MAE ± SD for all five models.
    Best config used: PPG+ACC for classical, raw signal for DL.
    Suitable for a single paper column (3.5 in wide).
    """
    cls_sum = load_classical_summary()

    # best config per model = PPG+ACC
    rows = []
    for model, label, color in [
        ("linear_regression", "Linear Regression", C_LR),
        ("random_forest",     "Random Forest",     C_RF),
        ("xgboost",           "XGBoost",           C_XGB),
    ]:
        r = cls_sum[(cls_sum["model"] == model) & (cls_sum["input"] == "ppg+acc")].iloc[0]
        rows.append({"label": label, "mae_mean": r["mae_mean"], "mae_std": r["mae_std"],
                     "color": color, "group": "Classical ML"})

    dl_sum = load_dl_summary()
    for _, r in dl_sum.iterrows():
        rows.append({"label": r["model"], "mae_mean": r["mae_mean"], "mae_std": r["mae_std"],
                     "color": C_CNN if r["model"] == "CNN" else C_LSTM, "group": "Deep Learning"})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")

    x    = np.arange(len(df))
    bars = ax.bar(x, df["mae_mean"], yerr=df["mae_std"],
                  color=df["color"], width=0.6, capsize=5, zorder=3,
                  error_kw=dict(elinewidth=1.5, ecolor="#444444", capthick=1.5))

    # value labels
    for bar, mae, std in zip(bars, df["mae_mean"], df["mae_std"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.4,
                f"{mae:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#222222")

    # divider between classical and DL
    ax.axvline(2.5, color="#BBBBBB", lw=1.4, linestyle="--", zorder=2)
    ax.text(1.0, ax.get_ylim()[1] * 0.96, "Classical ML",
            ha="center", fontsize=10, color="#333333", fontstyle="italic")
    ax.text(3.5, ax.get_ylim()[1] * 0.96, "Deep Learning",
            ha="center", fontsize=10, color="#333333", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=10)
    ax.set_ylabel("Mean Absolute Error (bpm)", fontsize=11)
    ax.set_title("LOSO Heart Rate Estimation — MAE ± SD (PPG+ACC / raw signal)",
                 fontsize=11, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, df["mae_mean"].max() + df["mae_std"].max() + 4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig1] saved → {out_path}")


# ── figure 2: per-subject MAE (XGB vs CNN vs CNN-BiLSTM) ─────────────────────

def fig2_per_subject(out_path: Path) -> None:
    """
    Grouped bar chart: per-subject MAE for XGBoost (PPG+ACC), CNN, CNN-BiLSTM.
    S5 and S8 are annotated as outliers.
    Suitable for full paper width (7 in).
    """
    cls_ps = load_classical_per_subject()
    dl_ps  = load_dl_per_subject()

    xgb = cls_ps[cls_ps["model"] == "xgboost"].set_index("subject")["mae"]
    cnn = dl_ps[dl_ps["model"] == "CNN"].set_index("subject")["mae"]
    lst = dl_ps[dl_ps["model"] == "CNN-BiLSTM"].set_index("subject")["mae"]

    subjects = SUBJECT_ORDER
    xgb_vals = [xgb.get(s, np.nan) for s in subjects]
    cnn_vals = [cnn.get(s, np.nan) for s in subjects]
    lst_vals = [lst.get(s, np.nan) for s in subjects]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    xs = np.arange(len(subjects))
    w  = 0.26

    ax.bar(xs - w, xgb_vals, width=w, color=C_XGB,  label="XGBoost (PPG+ACC)", zorder=3, alpha=0.92)
    ax.bar(xs,     cnn_vals, width=w, color=C_CNN,   label="CNN",               zorder=3, alpha=0.92)
    ax.bar(xs + w, lst_vals, width=w, color=C_LSTM,  label="CNN-BiLSTM",        zorder=3, alpha=0.92)

    # shaded regions for outlier subjects
    for subj_i, shade, label_txt, y_ann, txt_color in [
        (4, "#FFE5E5", "S5: signal artifact\n(all models)", 56, "#C0392B"),
        (7, "#FFF3CD", "S8: DL-specific\nfailure",         44, "#9B7000"),
    ]:
        ax.axvspan(subj_i - 0.5, subj_i + 0.5, color=shade, alpha=0.55, zorder=1)
        ax.annotate(label_txt,
                    xy=(subj_i, max(xgb_vals[subj_i], cnn_vals[subj_i], lst_vals[subj_i]) + 1),
                    xytext=(subj_i, y_ann),
                    ha="center", fontsize=8.5, color=txt_color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=txt_color, lw=1.2))

    ax.set_xticks(xs)
    ax.set_xticklabels(subjects, fontsize=10)
    ax.set_ylabel("Mean Absolute Error (bpm)", fontsize=11)
    ax.set_title("Per-Subject LOSO MAE — XGBoost vs CNN vs CNN-BiLSTM",
                 fontsize=11, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 68)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig2] saved → {out_path}")


# ── figure 3: ACC uplift for classical models ─────────────────────────────────

def fig3_acc_uplift(out_path: Path) -> None:
    """
    Paired bar chart showing PPG-only vs PPG+ACC MAE for each classical model.
    Demonstrates that ACC benefits RF and XGBoost but not Linear Regression.
    Suitable for a single paper column (3.5 in).
    """
    cls_sum = load_classical_summary()

    models = [
        ("linear_regression", "Linear\nRegression"),
        ("random_forest",     "Random\nForest"),
        ("xgboost",           "XGBoost"),
    ]

    ppg_mae, acc_mae, ppg_std, acc_std, labels = [], [], [], [], []
    for model_key, model_label in models:
        r_ppg = cls_sum[(cls_sum["model"] == model_key) & (cls_sum["input"] == "ppg_only")].iloc[0]
        r_acc = cls_sum[(cls_sum["model"] == model_key) & (cls_sum["input"] == "ppg+acc")].iloc[0]
        ppg_mae.append(r_ppg["mae_mean"])
        acc_mae.append(r_acc["mae_mean"])
        ppg_std.append(r_ppg["mae_std"])
        acc_std.append(r_acc["mae_std"])
        labels.append(model_label)

    xs = np.arange(len(labels))
    w  = 0.32

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.patch.set_facecolor("white")

    b1 = ax.bar(xs - w / 2, ppg_mae, width=w, yerr=ppg_std,
                color="#90CAF9", label="PPG only", capsize=5, zorder=3,
                error_kw=dict(elinewidth=1.4, ecolor="#444444", capthick=1.4))
    b2 = ax.bar(xs + w / 2, acc_mae, width=w, yerr=acc_std,
                color=C_XGB,    label="PPG + ACC", capsize=5, zorder=3,
                error_kw=dict(elinewidth=1.4, ecolor="#444444", capthick=1.4))

    # delta annotations above each pair
    for i, (pm, am) in enumerate(zip(ppg_mae, acc_mae)):
        delta = am - pm
        sign  = "+" if delta >= 0 else ""
        color = "#C0392B" if delta > 0.1 else "#27AE60"   # red = worse, green = better
        ax.text(xs[i], max(pm, am) + max(ppg_std[i], acc_std[i]) + 0.5,
                f"Δ{sign}{delta:.2f}", ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean Absolute Error (bpm)", fontsize=11)
    ax.set_title("ACC Uplift: PPG-only vs PPG+ACC\n(LOSO MAE ± SD)",
                 fontsize=11, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig3] saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def make_all_figures(fig_dir: Path | None = None) -> None:
    out = Path(fig_dir or FIG_DIR)
    out.mkdir(parents=True, exist_ok=True)

    fig1_model_comparison(out / "fig1_model_comparison.png")
    fig2_per_subject(out / "fig2_per_subject.png")
    fig3_acc_uplift(out / "fig3_acc_uplift.png")

    print(f"\n✓ All figures saved to: {out}")


if __name__ == "__main__":
    make_all_figures()
