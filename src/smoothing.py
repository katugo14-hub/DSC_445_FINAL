"""
Post-prediction smoothing methods.

All smoothers operate on a 1-D sequence of HR predictions for ONE subject,
in window-order (2 s stride). The caller (`smooth_per_subject`) is
responsible for splitting by subject and re-stitching.

Production methods

"none"          pass-through (baseline)
"median"        sliding-window median               param: window_size
"ema_centered"  forward + backward EMA (zero-phase) param: alpha
"kalman"        constant-velocity Kalman + RTS      params: process_var, meas_var

"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, medfilt


#  core smoothers

def smooth_median(y, window):
    if window <= 1 or len(y) < 3:
        return y.copy()
    w = window if window % 2 else window + 1
    w = min(w, len(y) if len(y) % 2 else len(y) - 1)
    return medfilt(y, kernel_size=w)


def _smooth_ema_causal(y, alpha):
    """
    Internal helper. Causal exponential moving average:
        y_hat[t] = a*y[t] + (1-a)*y_hat[t-1],  y_hat[0] = y[0]
    """
    y = np.asarray(y, dtype=np.float32)
    if alpha >= 1.0 or len(y) < 2:
        return y.copy()
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    # zi chosen so the first output equals y[0]:
    #   y_out[0] = b[0]*y[0] + zi[0]  =>  zi[0] = (1 - alpha) * y[0]
    zi = np.array([(1.0 - alpha) * y[0]], dtype=np.float64)
    out, _ = lfilter(b, a, y, zi=zi)
    return out.astype(np.float32)


def smooth_ema_centered(y, alpha):
    """Forward EMA + backward EMA, averaged."""
    fwd = _smooth_ema_causal(y, alpha)
    bwd = _smooth_ema_causal(y[::-1], alpha)[::-1]
    return 0.5 * (fwd + bwd)


def smooth_kalman(y, process_var=1.0, meas_var=25.0):
    """
    1-D constant-velocity Kalman filter, smoothed (forward-backward).
    State = [hr, hr_rate].

    Defaults:
      process_var = 1.0
      meas_var    = 25.0  (model's per-window MAE on the order of 5 bpm)
    """
    if len(y) < 2:
        return y.copy()

    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = process_var * np.array([[dt**4 / 4, dt**3 / 2],
                                [dt**3 / 2, dt**2]])
    R = np.array([[meas_var]])

    n = len(y)
    xs = np.zeros((n, 2))
    Ps = np.zeros((n, 2, 2))

    # init
    x = np.array([y[0], 0.0])
    P = np.eye(2) * meas_var
    xs[0] = x
    Ps[0] = P

    # forward pass
    for t in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        z = y[t]
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ (np.array([z]) - H @ x))
        P = (np.eye(2) - K @ H) @ P
        xs[t] = x
        Ps[t] = P

    # RTS smoother (backward pass) - gives optimal smoothed estimates
    xs_s = xs.copy()
    Ps_s = Ps.copy()
    for t in range(n - 2, -1, -1):
        x_pred = F @ xs[t]
        P_pred = F @ Ps[t] @ F.T + Q
        try:
            G = Ps[t] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            continue
        xs_s[t] = xs[t] + G @ (xs_s[t + 1] - x_pred)
        Ps_s[t] = Ps[t] + G @ (Ps_s[t + 1] - P_pred) @ G.T

    return xs_s[:, 0].astype(np.float32)


# dispatcher

_AVAILABLE = ("none", "median", "ema_centered", "kalman")


def smooth_sequence(y, method, **params):
    """
    Apply a single smoother to one 1-D sequence.

    method: one of {"none", "median", "ema_centered", "kalman"}
    params: method-specific (window_size, alpha, process_var, meas_var)
    """
    if method == "none":
        return np.asarray(y, dtype=np.float32).copy()
    if method == "median":
        return smooth_median(y, params.get("window_size", 11))
    if method == "ema_centered":
        return smooth_ema_centered(y, params.get("alpha", 0.20))
    if method == "kalman":
        return smooth_kalman(
            y,
            params.get("process_var", 1.0),
            params.get("meas_var", 25.0),
        )
    raise ValueError(
        f"Unknown smoothing method: {method!r}. "
        f"Available: {_AVAILABLE}."
    )


# subject-aware wrapper

def smooth_per_subject(y_pred, subjects, method="ema_centered", **params):
    """Apply `method` per subject, preserving subject ordering."""
    y_pred = np.asarray(y_pred, dtype=np.float32)
    subj = np.asarray(subjects)
    out = y_pred.copy()
    for sid in np.unique(subj):
        idx = np.where(subj == sid)[0]
        out[idx] = smooth_sequence(y_pred[idx], method, **params)
    return out


# sweep helper

def smoothing_grid():
    grid = [{"method": "none"}]
    for w in (5, 9, 11, 15, 19, 25):
        grid.append({"method": "median", "window_size": w})
    for a in (0.10, 0.15, 0.20, 0.30, 0.40):
        grid.append({"method": "ema_centered", "alpha": a})
    for pv in (0.25, 0.5, 1.0, 2.0):
        for mv in (16.0, 25.0, 36.0, 49.0):
            grid.append({"method": "kalman", "process_var": pv, "meas_var": mv})
    return grid
