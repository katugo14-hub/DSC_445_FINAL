"""
Feature extraction for PPG + ACC windows.

Each PPG window, ACC window pair is reduced
to a fixed-length feature vector covering:
  - time-domain statistics on the bandpassed PPG and on each ACC axis
  - peak-based HRV-style features
  - frequency-domain content in the cardiac band (Welch PSD)
  - harmonic-structure features (power at 2f, 3f, f/2) to detect when the
    dominant peak is actually a motion artifact or a harmonic
  - PPG↔ACC spectral cancellation: dominant PPG peak after suppressing the
    bands where ACC has strong peaks (TROIKA-style motion-artifact removal)
"""
import numpy as np
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.stats import skew, kurtosis


PPG_FS = 64
ACC_FS = 32

HR_BAND_LOW = 0.5    # ~30 bpm
HR_BAND_HIGH = 3.5   # ~210 bpm

# Welch nperseg in seconds. Choosing the same duration for PPG (PPG_FS=64) and
# ACC (ACC_FS=32) gives matching df = fs/nperseg = 0.25 Hz on both PSDs, so
# can use PPG and ACC PSDs on a common frequency grid for cancellation.
WELCH_SECONDS = 4
PPG_NPERSEG = WELCH_SECONDS * PPG_FS    # 256
ACC_NPERSEG = WELCH_SECONDS * ACC_FS    # 128


_PPG_BUTTER_B, _PPG_BUTTER_A = butter(
    4,
    [HR_BAND_LOW / (PPG_FS / 2), HR_BAND_HIGH / (PPG_FS / 2)],
    btype="band",
)


def _bandpass_ppg(signal):
    """Apply the cached PPG bandpass (0.5–3.5 Hz, 4th-order Butterworth)."""
    return filtfilt(_PPG_BUTTER_B, _PPG_BUTTER_A, signal)


def _zscore(x, axis=0, eps=1e-8):
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def _safe_skew(x):
    if np.std(x) <= 1e-10:
        return 0.0
    v = skew(x)
    return 0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v)


def _safe_kurt(x):
    if np.std(x) <= 1e-10:
        return 0.0
    v = kurtosis(x)
    return 0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v)


def _time_stats(x):
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.ptp(x)),
        _safe_skew(x),
        _safe_kurt(x),
        float(np.sqrt(np.mean(x ** 2))),       # RMS
        float(np.mean(np.abs(np.diff(x)))),    # mean abs diff
    ]


def _zero_crossing_rate(x):
    s = np.signbit(x - np.mean(x))
    return float(np.mean(s[1:] != s[:-1]))


def _welch_psd(x, fs, nperseg):
    nperseg = min(len(x), nperseg)
    f, p = welch(x, fs=fs, nperseg=nperseg)
    return f, p


_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _band_power(f, p, low, high):
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return float(_TRAPZ(p[mask], f[mask]))


def _dominant_freq(f, p, low=HR_BAND_LOW, high=HR_BAND_HIGH):
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0, 0.0
    f_band = f[mask]
    p_band = p[mask]
    idx = int(np.argmax(p_band))
    return float(f_band[idx]), float(p_band[idx])


def _spectral_entropy(p, eps=1e-12):
    p_norm = p / (np.sum(p) + eps)
    p_norm = p_norm[p_norm > 0]
    return float(-np.sum(p_norm * np.log(p_norm)))


def _power_at(f, p, target_f, bandwidth=0.15):
    """Power within ±bandwidth Hz of `target_f`."""
    if target_f <= 0:
        return 0.0
    return _band_power(f, p, target_f - bandwidth, target_f + bandwidth)


# PPG

def _ppg_features(window):
    """
    Returns (feats list, ppg_dom_f, ppg_dom_p, f, p_ppg, x_norm).
    Also returns the PSD and normalized signal so cancellation + harmonic
    routines can reuse them without recomputing.
    """
    x = np.asarray(window).reshape(-1).astype(float)
    x_filt = _bandpass_ppg(x)
    x_norm = _zscore(x_filt)

    feats = []
    feats += _time_stats(x_norm)
    feats.append(_zero_crossing_rate(x_norm))

    # peak / HRV-style features
    peaks, props = find_peaks(
        x_norm, distance=int(PPG_FS * 0.4), prominence=0.1
    )
    n_peaks = len(peaks)
    window_seconds = len(x_norm) / PPG_FS
    if n_peaks >= 2:
        ibi = np.diff(peaks) / PPG_FS
        ibi_mean = float(np.mean(ibi))
        ibi_std = float(np.std(ibi))
        ibi_median = float(np.median(ibi))
        ibi_iqr = float(np.percentile(ibi, 75) - np.percentile(ibi, 25))
        hr_from_peaks = 60.0 / ibi_mean if ibi_mean > 0 else 0.0
        hr_from_peaks_median = 60.0 / ibi_median if ibi_median > 0 else 0.0
        if len(ibi) >= 2:
            rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        else:
            rmssd = 0.0
    else:
        ibi_mean = ibi_std = ibi_median = ibi_iqr = 0.0
        hr_from_peaks = hr_from_peaks_median = 0.0
        rmssd = 0.0

    prom = props.get("prominences", np.array([]))
    prom_mean = float(np.mean(prom)) if len(prom) else 0.0
    prom_std = float(np.std(prom)) if len(prom) else 0.0
    prom_median = float(np.median(prom)) if len(prom) else 0.0

    peak_rate_bpm = (n_peaks / window_seconds) * 60.0

    feats += [
        float(n_peaks),
        ibi_mean, ibi_std, ibi_median, ibi_iqr,
        hr_from_peaks, hr_from_peaks_median,
        rmssd,
        prom_mean, prom_std, prom_median,
        peak_rate_bpm,
    ]

    # frequency-domain
    f, p = _welch_psd(x_norm, PPG_FS, PPG_NPERSEG)
    dom_f, dom_p = _dominant_freq(f, p, HR_BAND_LOW, HR_BAND_HIGH)
    bp_total = _band_power(f, p, HR_BAND_LOW, HR_BAND_HIGH)
    bp_low = _band_power(f, p, 0.5, 1.5)
    bp_mid = _band_power(f, p, 1.5, 2.5)
    bp_high = _band_power(f, p, 2.5, 3.5)
    sp_entropy = _spectral_entropy(p)
    hr_from_fft = dom_f * 60.0

    feats += [
        dom_f, dom_p, bp_total, bp_low, bp_mid, bp_high,
        sp_entropy, hr_from_fft,
    ]

    # top-3 peaks in cardiac band
    mask = (f >= HR_BAND_LOW) & (f <= HR_BAND_HIGH)
    f_band = f[mask]
    p_band = p[mask]
    if len(p_band) >= 3:
        order = np.argsort(p_band)[::-1]
        f2, p2 = float(f_band[order[1]]), float(p_band[order[1]])
        f3, p3 = float(f_band[order[2]]), float(p_band[order[2]])
    else:
        f2 = p2 = f3 = p3 = 0.0
    feats += [f2, p2, f3, p3]

    # harmonic structure: power at 2f, 3f, f/2, plus ratios
    p_at_2f = _power_at(f, p, 2 * dom_f)
    p_at_3f = _power_at(f, p, 3 * dom_f)
    p_at_half = _power_at(f, p, 0.5 * dom_f) if (0.5 * dom_f) >= HR_BAND_LOW else 0.0
    p_at_f = max(_power_at(f, p, dom_f), 1e-12)
    harmonic_ratio_2 = p_at_2f / p_at_f
    harmonic_ratio_3 = p_at_3f / p_at_f
    subharmonic_ratio = p_at_half / p_at_f
    feats += [
        p_at_2f, p_at_3f, p_at_half,
        harmonic_ratio_2, harmonic_ratio_3, subharmonic_ratio,
    ]

    return feats, dom_f, dom_p, f, p, x_norm


def _acc_features(window, ppg_dom_f):
    """
    Returns (feats, f_acc_mag, p_acc_mag).
    f_acc_mag / p_acc_mag are the ACC-magnitude PSD on the SAME df=0.25 Hz grid
    as PPG (within the overlap 0..16 Hz)
    """
    a = np.asarray(window).astype(float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    feats = []

    for axis in range(a.shape[1]):
        ax = a[:, axis]
        ax_norm = _zscore(ax)
        feats += _time_stats(ax_norm)

        f, p = _welch_psd(ax_norm, ACC_FS, ACC_NPERSEG)
        dom_f, dom_p = _dominant_freq(f, p, HR_BAND_LOW, HR_BAND_HIGH)
        bp_total = _band_power(f, p, HR_BAND_LOW, HR_BAND_HIGH)
        sp_entropy = _spectral_entropy(p)
        feats += [
            dom_f, dom_p, bp_total, sp_entropy,
            abs(dom_f - ppg_dom_f),
        ]

    mag = np.sqrt(np.sum(a ** 2, axis=1))
    mag_norm = _zscore(mag)
    feats += _time_stats(mag_norm)

    f_acc, p_acc = _welch_psd(mag_norm, ACC_FS, ACC_NPERSEG)
    dom_f_mag, dom_p_mag = _dominant_freq(f_acc, p_acc, HR_BAND_LOW, HR_BAND_HIGH)
    bp_total_mag = _band_power(f_acc, p_acc, HR_BAND_LOW, HR_BAND_HIGH)
    feats += [
        dom_f_mag, dom_p_mag, bp_total_mag,
        abs(dom_f_mag - ppg_dom_f),
    ]

    motion_energy = float(np.sum(mag_norm ** 2))
    feats.append(motion_energy)

    return feats, f_acc, p_acc


# PPG-ACC cancellation

def _ppg_acc_cancellation_features(f_ppg, p_ppg, f_acc, p_acc,
                                   ppg_dom_f, ppg_dom_p,
                                   n_acc_peaks=3, suppress_hz=0.25):
    """
    Suppress PPG PSD bins near the top ACC PSD peaks, then
    re-pick the dominant frequency. Returns a feature vector.
    """
    # restrict both to HR band
    mask_ppg = (f_ppg >= HR_BAND_LOW) & (f_ppg <= HR_BAND_HIGH)
    mask_acc = (f_acc >= HR_BAND_LOW) & (f_acc <= HR_BAND_HIGH)

    f_ppg_band = f_ppg[mask_ppg]
    p_ppg_band = p_ppg[mask_ppg].copy()
    f_acc_band = f_acc[mask_acc]
    p_acc_band = p_acc[mask_acc]

    if len(p_ppg_band) == 0 or len(p_acc_band) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # top ACC peaks in the HR band
    n_top = min(n_acc_peaks, len(p_acc_band))
    acc_top_idx = np.argsort(p_acc_band)[::-1][:n_top]
    acc_peak_freqs = f_acc_band[acc_top_idx]

    cleaned = p_ppg_band.copy()
    for f_peak in acc_peak_freqs:
        suppress_mask = (f_ppg_band >= f_peak - suppress_hz) & \
                        (f_ppg_band <= f_peak + suppress_hz)
        cleaned[suppress_mask] = 0.0

    if np.max(cleaned) <= 0:
        # everything suppressed → fall back to raw dom_f
        cleaned_dom_f = ppg_dom_f
        cleaned_dom_p = ppg_dom_p
    else:
        idx = int(np.argmax(cleaned))
        cleaned_dom_f = float(f_ppg_band[idx])
        cleaned_dom_p = float(cleaned[idx])

    hr_from_fft_cleaned = cleaned_dom_f * 60.0
    diff_vs_raw = abs(cleaned_dom_f - ppg_dom_f)
    # how much PPG band power survived the suppression
    survival_ratio = (
        float(np.sum(cleaned)) / float(np.sum(p_ppg_band) + 1e-12)
    )

    return [
        cleaned_dom_f, cleaned_dom_p,
        hr_from_fft_cleaned, diff_vs_raw,
        survival_ratio,
    ]

def extract_window_features(ppg_window, acc_window):
    """
    Returns a 1-D feature vector for one (PPG, ACC) window pair.
    """
    ppg_feats, ppg_dom_f, ppg_dom_p, f_ppg, p_ppg, _ = _ppg_features(ppg_window)
    acc_feats, f_acc, p_acc = _acc_features(acc_window, ppg_dom_f)
    canc_feats = _ppg_acc_cancellation_features(
        f_ppg, p_ppg, f_acc, p_acc, ppg_dom_f, ppg_dom_p
    )

    vec = np.array(ppg_feats + acc_feats + canc_feats, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def feature_names():
    """Stable list of feature names matching `extract_window_features` output."""
    base_time = ["mean", "std", "min", "max", "ptp", "skew", "kurt", "rms", "mad"]

    def acc_axis(axis):
        return [
            *(f"acc_{axis}_{n}" for n in base_time),
            f"acc_{axis}_dom_f", f"acc_{axis}_dom_p", f"acc_{axis}_bp_total",
            f"acc_{axis}_spectral_entropy", f"acc_{axis}_dom_f_diff_ppg",
        ]

    return [
        # PPG: time-domain stats + zero-crossing
        *(f"ppg_{n}" for n in base_time), "ppg_zcr",
        # PPG: peak / HRV
        "ppg_n_peaks",
        "ppg_ibi_mean", "ppg_ibi_std", "ppg_ibi_median", "ppg_ibi_iqr",
        "ppg_hr_from_peaks", "ppg_hr_from_peaks_median", "ppg_rmssd",
        "ppg_peak_prom_mean", "ppg_peak_prom_std", "ppg_peak_prom_median",
        "ppg_peak_rate_bpm",
        # PPG: frequency-domain (Welch PSD)
        "ppg_dom_f", "ppg_dom_p",
        "ppg_bp_total", "ppg_bp_low", "ppg_bp_mid", "ppg_bp_high",
        "ppg_spectral_entropy", "ppg_hr_from_fft",
        "ppg_dom_f2", "ppg_dom_p2", "ppg_dom_f3", "ppg_dom_p3",
        # PPG: harmonic structure
        "ppg_p_at_2f", "ppg_p_at_3f", "ppg_p_at_halff",
        "ppg_harmonic_ratio_2", "ppg_harmonic_ratio_3", "ppg_subharmonic_ratio",
        # ACC per axis (x, y, z)
        *acc_axis("x"), *acc_axis("y"), *acc_axis("z"),
        # ACC magnitude
        *(f"acc_mag_{n}" for n in base_time),
        "acc_mag_dom_f", "acc_mag_dom_p", "acc_mag_bp_total",
        "acc_mag_dom_f_diff_ppg", "acc_mag_motion_energy",
        # PPG <-> ACC spectral cancellation
        "ppg_dom_f_cleaned", "ppg_dom_p_cleaned",
        "ppg_hr_from_fft_cleaned", "ppg_dom_f_diff_cleaned_vs_raw",
        "ppg_psd_survival_ratio",
    ]


def _extract_chunk(args):
    """
    Worker function for `extract_features_batch`. 
    """
    ppg_chunk, acc_chunk = args
    out = [extract_window_features(ppg_chunk[i], acc_chunk[i])
           for i in range(len(ppg_chunk))]
    return np.stack(out, axis=0)


def _extract_sequential(X_ppg, X_acc, verbose_every):
    n = len(X_ppg)
    out = []
    for i in range(n):
        out.append(extract_window_features(X_ppg[i], X_acc[i]))
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  features: {i + 1}/{n}")
    return np.stack(out, axis=0)


def extract_features_batch(X_ppg, X_acc, n_workers=None,
                           chunk_size=2048, verbose_every=2000):
    """
    Compute the feature matrix for every (PPG, ACC) window pair.

    X_ppg: (N, T_ppg, 1) or (N, T_ppg)
    X_acc: (N, T_acc, 3)

    Returns: (N, F) feature matrix.
    """
    n = len(X_ppg)
    if n_workers == 1 or n < 1000:
        return _extract_sequential(X_ppg, X_acc, verbose_every)

    # Defer the import so single-threaded paths don't pay for it.
    from multiprocessing import Pool

    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # np.asarray() materializes mmap-backed slices into the IPC payload.
        chunks.append((
            np.asarray(X_ppg[start:end]),
            np.asarray(X_acc[start:end]),
        ))
    n_chunks = len(chunks)
    print(f"  parallel feature extraction: {n} windows -> "
          f"{n_chunks} chunks of {chunk_size} "
          f"across {n_workers or 'all'} workers")

    results = []
    with Pool(processes=n_workers) as pool:
        for i, block in enumerate(pool.imap(_extract_chunk, chunks)):
            results.append(block)
            done = min((i + 1) * chunk_size, n)
            print(f"  features: {done}/{n}")
    return np.concatenate(results, axis=0)
