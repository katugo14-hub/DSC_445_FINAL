import numpy as np

# Optional scipy bandpass filter. We import lazily so the Week 1 pipeline
# (which never used scipy) keeps running even if scipy isn't installed.
try:
    from scipy.signal import butter, filtfilt  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Week 2 additions: per-window post-processing
# ---------------------------------------------------------------------------

def bandpass_filter_windows(
    x: np.ndarray, fs: int, low_hz: float = 0.5, high_hz: float = 4.0, order: int = 4
) -> np.ndarray:
    """
    Apply a 4th-order Butterworth bandpass filter along axis=-1.
    Requires scipy. If scipy is unavailable, returns x unchanged.

    Accepts shape (N, T) or (N, T, C); filters along T.
    """
    if not _HAS_SCIPY:
        return x

    nyq = fs * 0.5
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    x_in = np.asarray(x, dtype=np.float64)

    if x_in.ndim == 2:
        y = filtfilt(b, a, x_in, axis=-1)
    elif x_in.ndim == 3:
        # filter each channel independently
        y = np.empty_like(x_in)
        for c in range(x_in.shape[-1]):
            y[..., c] = filtfilt(b, a, x_in[..., c], axis=-1)
    else:
        raise ValueError(f"Unsupported shape for bandpass: {x_in.shape}")
    return y.astype(x.dtype if hasattr(x, "dtype") else np.float32)


def zscore_per_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-window z-score along axis=-1 (for shape (N, T)) or for each channel
    independently (for shape (N, T, C)).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        mu = arr.mean(axis=-1, keepdims=True)
        sd = arr.std(axis=-1, keepdims=True)
        return (arr - mu) / (sd + eps)
    if arr.ndim == 3:
        mu = arr.mean(axis=1, keepdims=True)
        sd = arr.std(axis=1, keepdims=True)
        return (arr - mu) / (sd + eps)
    raise ValueError(f"Unsupported shape for zscore: {arr.shape}")


def clean_arrays(
    X_ppg: np.ndarray,
    X_acc: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    hr_min: float = 30.0,
    hr_max: float = 220.0,
    bandpass_ppg: bool = True,
    bandpass_acc: bool = False,
    zscore: bool = True,
):
    """
    Drop windows whose label is non-finite or outside [hr_min, hr_max],
    then optionally bandpass-filter and z-score each window.

    Returns cleaned (X_ppg, X_acc, y, subjects, mask).
    """
    y = np.asarray(y, dtype=np.float32)
    mask = np.isfinite(y) & (y >= hr_min) & (y <= hr_max)

    X_ppg = X_ppg[mask]
    X_acc = X_acc[mask]
    y = y[mask]
    subjects = subjects[mask]

    # PPG: (N, 512, 1) -> filter on the single channel, then zscore
    if bandpass_ppg:
        X_ppg = bandpass_filter_windows(X_ppg.squeeze(-1), fs=64,
                                        low_hz=0.5, high_hz=4.0, order=4)[..., None]
    if zscore:
        X_ppg = zscore_per_window(X_ppg)

    # ACC: (N, 256, 3)
    if bandpass_acc:
        X_acc = bandpass_filter_windows(X_acc, fs=32, low_hz=0.5, high_hz=10.0, order=4)
    if zscore:
        X_acc = zscore_per_window(X_acc)

    return (
        X_ppg.astype(np.float32),
        X_acc.astype(np.float32),
        y.astype(np.float32),
        subjects,
        mask,
    )


# ---------------------------------------------------------------------------
# Original Week 1 segmentation pipeline (unchanged)
# ---------------------------------------------------------------------------

def segment_signal(signal, window_size, step_size):
    """
    Segment signal into sliding windows.
    """
    segments = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        end = start + window_size
        segments.append(signal[start:end])

    return np.array(segments)


def align_labels(hr_labels, num_windows):
    """
    Align HR labels with segmented windows.
    """
    hr_labels = np.asarray(hr_labels).reshape(-1)

    if len(hr_labels) >= num_windows:
        return hr_labels[:num_windows]

    raise ValueError(
        f"Not enough HR labels. Labels: {len(hr_labels)}, windows: {num_windows}"
    )


def preprocess_subject(ppg, acc, hr, subject_id):
    """
    Preprocess one subject using:
    - 8-second window
    - 2-second shift
    """
    ppg_fs = 64
    acc_fs = 32

    window_seconds = 8
    step_seconds = 2

    ppg_window = window_seconds * ppg_fs
    ppg_step = step_seconds * ppg_fs

    acc_window = window_seconds * acc_fs
    acc_step = step_seconds * acc_fs

    ppg_segments = segment_signal(ppg, ppg_window, ppg_step)
    acc_segments = segment_signal(acc, acc_window, acc_step)

    n = min(len(ppg_segments), len(acc_segments), len(hr))

    ppg_segments = ppg_segments[:n]
    acc_segments = acc_segments[:n]
    labels = align_labels(hr, n)

    subject_ids = np.array([subject_id] * n)

    return {
        "ppg": ppg_segments,
        "acc": acc_segments,
        "y": labels,
        "subject": subject_ids
    }