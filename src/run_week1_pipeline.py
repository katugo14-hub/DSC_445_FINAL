import numpy as np
from pathlib import Path

from data_loader import load_subject_pkl
from preprocessing import preprocess_subject
from loso import loso_split
from sanity_checks import (
    check_shapes,
    check_labels,
    check_no_subject_leakage
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    all_ppg = []
    all_acc = []
    all_y = []
    all_subjects = []

    subject_files = sorted(RAW_DATA_DIR.rglob("*.pkl"))

    if not subject_files:
        raise FileNotFoundError(
            f"No .pkl files found anywhere inside {RAW_DATA_DIR}. "
            "Make sure the PPG-DaLiA subject folders contain .pkl files."
        )

    print(f"Found {len(subject_files)} subject files.")

    for file_path in subject_files:
        subject_id = file_path.stem

        print(f"\nLoading {subject_id}...")

        ppg, acc, hr = load_subject_pkl(file_path)
        processed = preprocess_subject(ppg, acc, hr, subject_id)

        all_ppg.append(processed["ppg"])
        all_acc.append(processed["acc"])
        all_y.append(processed["y"])
        all_subjects.append(processed["subject"])

        print(f"{subject_id} windows:", len(processed["y"]))

    X_ppg = np.concatenate(all_ppg, axis=0)
    X_acc = np.concatenate(all_acc, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)

    print("\nFinal dataset:")
    check_shapes(X_ppg, y, subjects)
    print("X_acc shape:", X_acc.shape)
    check_labels(y)

    np.save(OUTPUT_DIR / "X_ppg.npy", X_ppg)
    np.save(OUTPUT_DIR / "X_acc.npy", X_acc)
    np.save(OUTPUT_DIR / "y.npy", y)
    np.save(OUTPUT_DIR / "subjects.npy", subjects)

    print("\nSaved processed arrays to outputs/")

    print("\nTesting LOSO split...")
    first_fold = next(loso_split(X_ppg, y, subjects))

    check_no_subject_leakage(
        first_fold["train_subjects"],
        first_fold["test_subjects"]
    )

    print("Held-out subject:", first_fold["test_subject"])
    print("X_train:", first_fold["X_train"].shape)
    print("X_test:", first_fold["X_test"].shape)

    print("\nWeek 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()