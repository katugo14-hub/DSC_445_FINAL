"""
Temporal / lag features for the XGBoost pipeline.

"""
from __future__ import annotations
import numpy as np


DEFAULT_LAG_COLS = [
    "ppg_hr_from_peaks",
    "ppg_hr_from_peaks_median",
    "ppg_peak_rate_bpm",
    "ppg_dom_f",
    "ppg_hr_from_fft",
    "ppg_hr_from_fft_cleaned",
    "ppg_dom_f_cleaned",
    "acc_mag_motion_energy",
    "acc_mag_dom_f",
    "ppg_bp_total",
]


def add_lag_features(X_feat, subjects, feature_names_list,
                     lag_cols=None, lags=(1, 2, 3),
                     also_add_deltas=True):
    """
    Append lagged copies of selected features.

    X_feat              : (N, F) feature matrix, rows ordered by window time
                          within each subject.
    subjects            : (N,) subject id per row.
    feature_names_list  : list of length F with column names.
    lag_cols            : names of columns to lag. Falls back to DEFAULT_LAG_COLS.
    lags                : iterable of integer lag steps. lag=1 -> 2s ago.
    also_add_deltas     : if True, also append (current - lag1) for each lag_col,
                          which is a cheap "rate of change" signal.

    Returns
    -------
    X_out         : (N, F + len(lag_cols) * len(lags) [+ len(lag_cols) if deltas]) array
    new_names     : list of column names for X_out
    """
    if lag_cols is None:
        lag_cols = DEFAULT_LAG_COLS

    name_to_idx = {n: i for i, n in enumerate(feature_names_list)}
    missing = [c for c in lag_cols if c not in name_to_idx]
    if missing:
        raise KeyError(f"Lag columns not in feature matrix: {missing}")

    col_idx = np.array([name_to_idx[c] for c in lag_cols])
    n_rows, n_feat = X_feat.shape

    # Group by subject, preserving row order within each subject.
    subj_arr = np.asarray(subjects)
    unique_subjects = np.unique(subj_arr)

    lag_blocks = {lag: np.empty((n_rows, len(col_idx)), dtype=X_feat.dtype)
                  for lag in lags}
    delta_block = np.empty((n_rows, len(col_idx)), dtype=X_feat.dtype) \
        if also_add_deltas else None

    for sid in unique_subjects:
        idx = np.where(subj_arr == sid)[0]
        block = X_feat[idx][:, col_idx]            # (n_s, k)
        for lag in lags:
            shifted = np.empty_like(block)
            if lag >= len(block):
                shifted[:] = block[0]
            else:
                shifted[:lag] = block[0]           # forward-fill head
                shifted[lag:] = block[:-lag]
            lag_blocks[lag][idx] = shifted

        if also_add_deltas:
            lag1 = lag_blocks[1][idx] if 1 in lags else None
            if lag1 is None:
                lag1 = np.empty_like(block)
                lag1[0] = block[0]
                lag1[1:] = block[:-1]
            delta_block[idx] = block - lag1

    pieces = [X_feat]
    new_names = list(feature_names_list)
    for lag in lags:
        pieces.append(lag_blocks[lag])
        new_names += [f"{c}_lag{lag}" for c in lag_cols]
    if also_add_deltas:
        pieces.append(delta_block)
        new_names += [f"{c}_delta1" for c in lag_cols]

    X_out = np.concatenate(pieces, axis=1).astype(np.float32)
    return X_out, new_names
