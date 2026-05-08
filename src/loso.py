import numpy as np


def loso_split(X, y, subjects):
    """
    Leave-One-Subject-Out split.
    """
    unique_subjects = np.unique(subjects)

    for test_subject in unique_subjects:
        train_idx = subjects != test_subject
        test_idx = subjects == test_subject

        yield {
            "test_subject": test_subject,
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
            "train_subjects": np.unique(subjects[train_idx]),
            "test_subjects": np.unique(subjects[test_idx])
        }