import numpy as np

# ── Week 1 / Week 2 checks ────────────────────────────────────────────────────

def check_shapes(X_ppg, y, subjects):
    """Assert that arrays are consistent in length and non-empty."""
    assert len(X_ppg) == len(y) == len(subjects), (
        f"Length mismatch: X_ppg={len(X_ppg)}, y={len(y)}, subjects={len(subjects)}"
    )
    assert len(X_ppg) > 0, "Dataset is empty."
    print(f"[check_shapes] OK — {len(X_ppg)} windows, X_ppg shape: {X_ppg.shape}")


def check_labels(y, hr_min=30.0, hr_max=220.0):
    """Assert HR labels are finite and within a physiological range."""
    assert np.all(np.isfinite(y)), "y contains NaN or Inf values."
    assert float(y.min()) >= hr_min, f"HR below {hr_min} bpm detected: {float(y.min()):.1f}"
    assert float(y.max()) <= hr_max, f"HR above {hr_max} bpm detected: {float(y.max()):.1f}"
    print(f"[check_labels] OK — HR range {float(y.min()):.1f}–{float(y.max()):.1f} bpm")


def check_no_subject_leakage(train_subjects, test_subjects):
    """Assert no subject appears in both train and test."""
    train_set = set(np.unique(train_subjects).tolist())
    test_set  = set(np.unique(test_subjects).tolist())
    overlap   = train_set & test_set
    assert not overlap, f"Subject leakage detected! Overlap: {overlap}"
    print(f"[check_no_subject_leakage] OK — no overlap between train and test subjects.")


# ── Week 3 DL checks (only imported when torch is available) ──────────────────

def check_data_shapes():
    from dl_dataset import get_all_subjects, load_subject
    subjects = get_all_subjects()
    print("Subjects found:", subjects)

    X, y = load_subject(subjects[0])
    print("X shape:", X.shape)     #(num_windows, 512, 4)
    print("y shape:", y.shape)     #(num_windows,)


def check_for_nans():
    from dl_dataset import get_all_subjects, load_subject
    subjects = get_all_subjects()
    X, y = load_subject(subjects[0])

    print("NaNs in X:", np.isnan(X).sum())
    print("NaNs in y:", np.isnan(y).sum())


def check_model_forward_pass():
    import torch
    from models_dl import CNNModel
    model = CNNModel()
    dummy = torch.randn(2, 512, 4)    #batch=2
    out = model(dummy)
    print("Model output shape:", out.shape)


if __name__ == "__main__":
    check_data_shapes()
    check_for_nans()
    check_model_forward_pass()
