import numpy as np


def check_no_subject_leakage(train_subjects, test_subjects):
    overlap = set(train_subjects).intersection(set(test_subjects))

    if overlap:
        raise ValueError(f"Data leakage detected. Overlap: {overlap}")

    print("PASS: No subject leakage.")


def check_shapes(X, y, subjects):
    assert len(X) == len(y) == len(subjects), "X, y, and subjects length mismatch"

    print("PASS: Shapes aligned.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subjects shape:", subjects.shape)


def check_labels(y):
    y = np.asarray(y)

    print("HR label min:", np.min(y))
    print("HR label max:", np.max(y))
    print("HR label mean:", np.mean(y))

    if np.any(np.isnan(y)):
        raise ValueError("NaN values found in labels.")

    print("PASS: Labels look valid.")