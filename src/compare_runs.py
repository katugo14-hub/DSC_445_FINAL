"""
Compare LOSO summaries side-by-side. Reads all xgb_*_loso_summary.json files
from outputs/ and prints a per-subject and overall MAE table.

Usage:
    python src/compare_runs.py
    python src/compare_runs.py --tags 04_+robust_loss 05a_+smoothing_only
"""
import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_summaries(tag_filter=None):
    files = sorted(OUTPUT_DIR.glob("*_loso_summary.json"))
    summaries = []
    for fp in files:
        with open(fp) as f:
            s = json.load(f)
        tag = s.get("tag") or fp.stem.replace("_loso_summary", "")
        if tag_filter and tag not in tag_filter:
            continue
        summaries.append((tag, s))
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="*", default=None,
                   help="Only include these tags. Default: all.")
    args = p.parse_args()

    summaries = load_summaries(args.tags)
    if not summaries:
        print("No summaries found.")
        return

    # collect all subjects across runs
    all_subjects = sorted({s for _, summ in summaries
                           for s in summ.get("per_subject_mae", {}).keys()},
                          key=lambda x: int(x.lstrip("S")) if x.lstrip("S").isdigit() else 999)

    # print header
    tags = [t for t, _ in summaries]
    col_width = max(8, max(len(t) for t in tags) + 1)
    print()
    print(f"{'subject':<10}" + "".join(f"{t:>{col_width}}" for t in tags))
    print("-" * (10 + col_width * len(tags)))

    for sid in all_subjects:
        row = f"{sid:<10}"
        for tag, summ in summaries:
            mae = summ.get("per_subject_mae", {}).get(sid)
            row += f"{mae:>{col_width}.2f}" if mae is not None else f"{'-':>{col_width}}"
        print(row)

    print("-" * (10 + col_width * len(tags)))
    print(f"{'overall':<10}" + "".join(
        f"{summ['overall']['mae']:>{col_width}.2f}" for _, summ in summaries))
    print(f"{'mean':<10}" + "".join(
        f"{summ['mean_subject_mae']:>{col_width}.2f}" for _, summ in summaries))
    print(f"{'± std':<10}" + "".join(
        f"{summ['std_subject_mae']:>{col_width}.2f}" for _, summ in summaries))
    print(f"{'n_feats':<10}" + "".join(
        f"{summ['n_features']:>{col_width}}" for _, summ in summaries))


if __name__ == "__main__":
    main()
