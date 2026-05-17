"""
Handcrafted time- and frequency-domain features for PPG-DaLiA.

Designed for inputs produced by the Week 1 pipeline:
    - X_ppg shape (N, 512, 1)  -> 64 Hz, 8 s windows
    - X_acc shape (N, 256, 3)  -> 32 Hz, 8 s windows, 3 axes

This module is numpy-only (uses numpy.fft) so it is portable.
"""
from __future__ import annotations

import numpy as np

PPG_FS = 64
ACC_FS = 32
HR_BAND_HZ = (0.5, 4.0)  # 30-240 bpm


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _safe_skew(x: np.ndarray) -> np.ndarray:
    """Population skew, axis=-1, NaN-safe for zero-variance windows."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    z = (x - mu) / sd
    return (z ** 3).mean(axis=-1)


def _safe_kurt(x: np.ndarray) -> np.ndarray:
    """Excess kurtosis, axis=-1, NaN-safe."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    z = (x - mu) / sd
    return (z ** 4).mean(axis=-1) - 3.0


def _zero_crossing_rate(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    signs = np.sign(centered)
    # avoid double-counting exact zeros
    signs = np.where(signs == 0, 1, signs)
    return np.mean(np.abs(np.diff(signs, axis=-1)) / 2.0, axis=-1)


def _iqr(x: np.ndarray) -> np.ndarray:
    q75 = np.quantile(x, 0.75, axis=-1)
    q25 = np.quantile(x, 0.25, axis=-1)
    return q75 - q25


def _mad(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    return np.mean(np.abs(x - mu), axis=-1)


# ---------------------------------------------------------------------------
# time-domain block (works for any 1D-ish signal along axis=-1)
# ---------------------------------------------------------------------------

def _time_domain_block(x: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Compute 13 time-domain stats along axis=-1. Returns (N, 13) and names."""
    feats = np.stack([
        x.mean(axis=-1),
        x.std(axis=-1),
        x.min(axis=-1),
        x.max(axis=-1),
        x.max(axis=-1) - x.min(axis=-1),         # range
        _safe_skew(x),
        _safe_kurt(x),
        np.median(x, axis=-1),
        _iqr(x),
        np.sqrt(np.mean(x ** 2, axis=-1)),       # RMS
        _zero_crossing_rate(x),
        _mad(x),
        np.sum(x ** 2, axis=-1),                 # energy
    ], axis=-1)

    names = [
        f"{prefix}_mean", f"{prefix}_std", f"{prefix}_min", f"{prefix}_max",
        f"{prefix}_range", f"{prefix}_skew", f"{prefix}_kurt", f"{prefix}_median",
        f"{prefix}_iqr", f"{prefix}_rms", f"{prefix}_zcr", f"{prefix}_mad",
        f"{prefix}_energy",
    ]
    return feats.astype(np.float32), names


# ---------------------------------------------------------------------------
# frequency-domain block
# ---------------------------------------------------------------------------

def _freq_domain_block(
    x: np.ndarray, fs: int, prefix: str, hr_band: tuple[float, float] = HR_BAND_HZ
) -> tuple[np.ndarray, list[str]]:
    """
    Single-sided FFT magnitude features along axis=-1 over the HR band.
    Includes dominant freq, dominant power, spectral centroid/entropy/bandwidth,
    band powers in 4 sub-bands, and an estimated-HR-from-peak-bin (bpm).
    """
    n = x.shape[-1]
    # detrend per window to avoid DC dominating the spectrum
    x = x - x.mean(axis=-1, keepdims=True)

    # rfft -> magnitude
    spec = np.fft.rfft(x, axis=-1)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)  # shape (n//2+1,)

    # band mask
    lo, hi = hr_band
    band_mask = (freqs >= lo) & (freqs <= hi)
    band_idx = np.where(band_mask)[0]
    band_freqs = freqs[band_idx]                         # (B,)
    band_mag = mag[..., band_idx]                        # (..., B)
    band_pow = band_mag ** 2                             # (..., B)

    # dominant frequency / power within HR band
    peak_idx = band_pow.argmax(axis=-1)                  # (...,)
    dom_freq = band_freqs[peak_idx]                      # (...,)
    dom_pow = np.take_along_axis(band_pow, peak_idx[..., None], axis=-1).squeeze(-1)
    estimated_hr_bpm = dom_freq * 60.0

    # spectral centroid: sum(f * p) / sum(p)
    total_pow = band_pow.sum(axis=-1)
    safe_total = np.where(total_pow == 0, 1.0, total_pow)
    centroid = (band_freqs * band_pow).sum(axis=-1) / safe_total

    # spectral bandwidth: sqrt( sum( (f - centroid)^2 * p ) / sum(p) )
    diff_sq = (band_freqs[None, :] - centroid[..., None]) ** 2
    bandwidth = np.sqrt((diff_sq * band_pow).sum(axis=-1) / safe_total)

    # spectral entropy on normalized band power
    p_norm = band_pow / safe_total[..., None]
    p_safe = np.where(p_norm > 0, p_norm, 1.0)
    entropy = -(p_norm * np.log(p_safe)).sum(axis=-1)

    # 4 sub-band powers within the HR band
    sub_edges = [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    sub_powers = []
    for a, b in sub_edges:
        m = (band_freqs >= a) & (band_freqs < b)
        if m.sum() == 0:
            sub_powers.append(np.zeros_like(dom_pow))
        else:
            sub_powers.append(band_pow[..., m].sum(axis=-1))

    feats = np.stack([
        dom_freq, dom_pow, centroid, entropy, bandwidth,
        sub_powers[0], sub_powers[1], sub_powers[2], sub_powers[3],
        estimated_hr_bpm,
    ], axis=-1)
    names = [
        f"{prefix}_dom_freq", f"{prefix}_dom_pow", f"{prefix}_centroid",
        f"{prefix}_entropy", f"{prefix}_bandwidth",
        f"{prefix}_pwr_0p5_1", f"{prefix}_pwr_1_2", f"{prefix}_pwr_2_3", f"{prefix}_pwr_3_4",
        f"{prefix}_est_hr_bpm",
    ]
    return feats.astype(np.float32), names


# ---------------------------------------------------------------------------
# PPG / ACC feature blocks
# ---------------------------------------------------------------------------

def ppg_features(ppg_windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    ppg_windows: (N, 512, 1) -> (N, 23), with 23 features.
    """
    x = np.asarray(ppg_windows, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    elif x.ndim != 2:
        raise ValueError(f"ppg_windows must be (N, 512) or (N, 512, 1), got {x.shape}")

    t_feats, t_names = _time_domain_block(x, prefix="ppg")
    f_feats, f_names = _freq_domain_block(x, fs=PPG_FS, prefix="ppg")

    feats = np.concatenate([t_feats, f_feats], axis=-1)
    names = t_names + f_names
    return feats, names


def acc_features(acc_windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    acc_windows: (N, 256, 3) -> (N, P), per-axis stats + magnitude block.

    Per axis (x/y/z): mean, std, rms, energy, dom_freq, dom_pow = 6 feats.
    Magnitude ||a||: mean, std, energy + dom_freq, dom_pow = 5 feats.
    Total = 6*3 + 5 = 23 features.
    """
    x = np.asarray(acc_windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError(f"acc_windows must be (N, T, 3), got {x.shape}")

    feats_list, names_list = [], []

    for ax_idx, ax in enumerate(["x", "y", "z"]):
        a = x[..., ax_idx]  # (N, T)
        t_subset = np.stack([
            a.mean(axis=-1),
            a.std(axis=-1),
            np.sqrt(np.mean(a ** 2, axis=-1)),
            np.sum(a ** 2, axis=-1),
        ], axis=-1).astype(np.float32)
        feats_list.append(t_subset)
        names_list.extend([
            f"acc{ax}_mean", f"acc{ax}_std", f"acc{ax}_rms", f"acc{ax}_energy",
        ])
        # freq features over a wider band: ACC at 32 Hz, motion can be up to ~10 Hz
        f_feats, f_names = _freq_domain_block(
            a, fs=ACC_FS, prefix=f"acc{ax}", hr_band=(0.5, 10.0)
        )
        # keep only dom_freq + dom_pow from that block to limit dimensionality
        keep = [0, 1]
        f_feats = f_feats[..., keep]
        f_names = [f_names[i] for i in keep]
        feats_list.append(f_feats)
        names_list.extend(f_names)

    # magnitude
    mag = np.linalg.norm(x, axis=-1)  # (N, T)
    m_basic = np.stack([
        mag.mean(axis=-1),
        mag.std(axis=-1),
        np.sum(mag ** 2, axis=-1),
    ], axis=-1).astype(np.float32)
    feats_list.append(m_basic)
    names_list.extend(["accmag_mean", "accmag_std", "accmag_energy"])

    f_feats, f_names = _freq_domain_block(
        mag, fs=ACC_FS, prefix="accmag", hr_band=(0.5, 10.0)
    )
    keep = [0, 1]
    f_feats = f_feats[..., keep]
    f_names = [f_names[i] for i in keep]
    feats_list.append(f_feats)
    names_list.extend(f_names)

    feats = np.concatenate(feats_list, axis=-1)
    return feats.astype(np.float32), names_list


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def extract_features(
    ppg_windows: np.ndarray, acc_windows: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """
    Concatenate PPG and ACC handcrafted features.

    Returns
    -------
    X_feat : (N, P) float32
    feat_names : list[str] of length P
    """
    p_feat, p_names = ppg_features(ppg_windows)
    a_feat, a_names = acc_features(acc_windows)

    X = np.concatenate([p_feat, a_feat], axis=-1)
    names = p_names + a_names
    return X.astype(np.float32), names


def ppg_only_indices(feat_names: list[str]) -> np.ndarray:
    """Column indices corresponding to PPG features (for PPG-only configs)."""
    return np.array([i for i, n in enumerate(feat_names) if n.startswith("ppg_")])


def acc_only_indices(feat_names: list[str]) -> np.ndarray:
    return np.array([i for i, n in enumerate(feat_names) if n.startswith("acc")])
