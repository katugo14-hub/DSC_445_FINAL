import numpy as np

from data_loader import load_subject_pkl


PPG_WINDOW_SIZE = 512
PPG_STEP_SIZE = 128
ACC_WINDOW_SIZE = 256


def create_subject_windows(file_path):
    """
    Create aligned PPG, ACC, and HR windows for one subject.
    PPG is sampled at 64 Hz and ACC is sampled at 32 Hz.
    """

    ppg, acc, hr = load_subject_pkl(file_path)

    ppg = np.squeeze(ppg)

    X_ppg_windows = []
    X_acc_windows = []
    y_values = []

    for ppg_start in range(0, len(ppg) - PPG_WINDOW_SIZE, PPG_STEP_SIZE):
        ppg_end = ppg_start + PPG_WINDOW_SIZE

        acc_start = ppg_start // 2
        acc_end = acc_start + ACC_WINDOW_SIZE

        if acc_end > len(acc):
            break

        ppg_window = ppg[ppg_start:ppg_end]
        acc_window = acc[acc_start:acc_end]

        hr_idx = ppg_end // 32

        if hr_idx < len(hr):
            y_val = hr[hr_idx]
        else:
            y_val = hr[-1]

        X_ppg_windows.append(ppg_window)
        X_acc_windows.append(acc_window)
        y_values.append(y_val)

    return (
        np.array(X_ppg_windows, dtype=np.float32),
        np.array(X_acc_windows, dtype=np.float32),
        np.array(y_values, dtype=np.float32),
    )
