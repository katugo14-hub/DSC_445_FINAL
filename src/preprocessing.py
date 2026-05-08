import numpy as np


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