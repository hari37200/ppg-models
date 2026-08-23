"""Handcrafted PPG morphology features.

On 657 segments a well-chosen feature set plus gradient boosting is a genuinely
competitive baseline -- often stronger than a deep net -- and it is interpretable,
which matters for a clinical target.

The features follow the standard PPG pulse-contour literature: fiducial points
(systolic peak, dicrotic notch, diastolic peak), the derived timing/amplitude
ratios, and the first and second derivative waves (VPG and APG, whose a-e peaks
give the ageing index). Features are computed per beat and aggregated by mean
and standard deviation across the beats in a segment.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis, skew

from hyperppg.config import FS_RAW

__all__ = ["extract_features", "extract_features_batch", "FEATURE_NAMES"]


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _ratio(a: float, b: float) -> float:
    return _safe(a / b) if abs(b) > 1e-12 else 0.0


def _find_systolic_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Locate systolic peaks with a physiologically constrained peak finder."""
    # Heart rate 30-220 bpm -> 0.27-2.0 s between beats.
    min_distance = max(int(0.27 * fs), 1)
    prominence = 0.15 * float(np.std(x)) if np.std(x) > 0 else None
    peaks, _ = sps.find_peaks(x, distance=min_distance, prominence=prominence)
    return peaks


def _beat_features(
    beat: np.ndarray, fs: float, sys_idx: int
) -> dict[str, float]:
    """Contour features for one pulse, given the systolic peak position."""
    n = beat.size
    out: dict[str, float] = {}
    if n < 8:
        return out

    amp = float(beat[sys_idx] - beat.min())
    baseline = float(beat.min())
    norm = beat - baseline
    peak_amp = max(amp, 1e-12)

    out["crest_time"] = sys_idx / fs
    out["beat_duration"] = n / fs
    out["sys_amp"] = amp
    out["diastolic_time"] = (n - sys_idx) / fs
    out["crest_ratio"] = _ratio(sys_idx, n)

    # Pulse widths at fractions of the systolic amplitude.
    for frac in (0.25, 0.5, 0.75):
        level = frac * peak_amp
        above = norm >= level
        out[f"width_{int(frac*100)}"] = float(above.sum()) / fs

    # Dicrotic notch: the most prominent local minimum after the systolic peak;
    # the diastolic peak is the largest local maximum after that notch.
    tail = norm[sys_idx:]
    notch_rel = -1
    dia_rel = -1
    if tail.size > 4:
        minima, _ = sps.find_peaks(-tail)
        if minima.size:
            notch_rel = int(minima[np.argmin(tail[minima])])
            after = tail[notch_rel:]
            if after.size > 3:
                maxima, _ = sps.find_peaks(after)
                if maxima.size:
                    dia_rel = notch_rel + int(maxima[np.argmax(after[maxima])])

    if notch_rel > 0:
        out["notch_time"] = (sys_idx + notch_rel) / fs
        out["notch_amp"] = _ratio(float(tail[notch_rel]), peak_amp)
        out["notch_delay"] = notch_rel / fs
    if dia_rel > 0:
        dia_amp = float(tail[dia_rel])
        out["dia_time"] = (sys_idx + dia_rel) / fs
        out["dia_amp_ratio"] = _ratio(dia_amp, peak_amp)
        # Augmentation index and stiffness-related timing.
        out["augmentation_index"] = _ratio(peak_amp - dia_amp, peak_amp)
        out["delta_t"] = dia_rel / fs
        out["stiffness_index"] = _ratio(1.0, dia_rel / fs) if dia_rel > 0 else 0.0

    # Areas: systolic vs diastolic portion of the pulse.
    total_area = float(np.trapezoid(norm)) if hasattr(np, "trapezoid") else float(np.trapz(norm))
    sys_area = (
        float(np.trapezoid(norm[:sys_idx])) if hasattr(np, "trapezoid") else float(np.trapz(norm[:sys_idx]))
    )
    out["area_total"] = total_area / fs
    out["area_ratio"] = _ratio(sys_area, total_area)

    return out


def _apg_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Second-derivative (APG) a-e wave ratios and the ageing index.

    Computed on the whole segment rather than per beat: the a-wave is the
    dominant APG peak, and the b/c/d/e waves follow it within one pulse.
    """
    out: dict[str, float] = {}
    vpg = np.gradient(x) * fs
    apg = np.gradient(vpg) * fs

    out["vpg_max"] = _safe(vpg.max())
    out["vpg_min"] = _safe(vpg.min())
    out["vpg_std"] = _safe(vpg.std())
    out["apg_std"] = _safe(apg.std())

    peaks, _ = sps.find_peaks(apg, distance=max(int(0.02 * fs), 1))
    if peaks.size == 0:
        return out
    a_idx = int(peaks[np.argmax(apg[peaks])])
    a = float(apg[a_idx])
    if abs(a) < 1e-12:
        return out

    # Search the following ~0.5 s for the alternating b, c, d, e waves.
    window_end = min(a_idx + int(0.5 * fs), apg.size)
    seg = apg[a_idx:window_end]
    if seg.size < 8:
        return out

    troughs, _ = sps.find_peaks(-seg)
    crests, _ = sps.find_peaks(seg)

    b = float(seg[troughs[0]]) if troughs.size else 0.0
    c = float(seg[crests[0]]) if crests.size else 0.0
    d = float(seg[troughs[1]]) if troughs.size > 1 else 0.0
    e = float(seg[crests[1]]) if crests.size > 1 else 0.0

    out["apg_b_a"] = _ratio(b, a)
    out["apg_c_a"] = _ratio(c, a)
    out["apg_d_a"] = _ratio(d, a)
    out["apg_e_a"] = _ratio(e, a)
    # Takazawa ageing index.
    out["apg_aging_index"] = _ratio(b - c - d - e, a)
    return out


def _spectral_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Band powers and spectral shape descriptors.

    The analysis bands here are all below 10 Hz, so frequency *resolution* is
    the binding constraint, not variance reduction: at fs=1000 a 256-sample
    Welch segment gives 3.9 Hz bins, which is wider than every band and leaves
    the 0.5-2 Hz band empty. Use the whole segment as one Welch block instead
    (2100 samples -> 0.48 Hz bins) and accept the noisier estimate, which the
    band integration averages out anyway.
    """
    out: dict[str, float] = {}
    nperseg = int(min(x.size, 4096))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
    total = float(psd.sum()) + 1e-12

    bands = {"vlf": (0.0, 0.5), "lf": (0.5, 2.0), "mf": (2.0, 5.0), "hf": (5.0, 10.0)}
    for name, (lo, hi) in bands.items():
        sel = (freqs >= lo) & (freqs < hi)
        out[f"pow_{name}"] = _safe(float(psd[sel].sum()) / total)

    out["spec_centroid"] = _safe(float((freqs * psd).sum() / total))
    p = psd / total
    out["spec_entropy"] = _safe(float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))))
    dom = int(np.argmax(psd))
    out["dominant_freq"] = _safe(float(freqs[dom]))
    return out


def extract_features(x: np.ndarray, fs: float = FS_RAW) -> dict[str, float]:
    """Full feature dictionary for one PPG segment.

    The input should already be filtered (use
    :func:`hyperppg.data.preprocess.bandpass`); this function does not filter,
    so the caller controls the conditioning.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    feats: dict[str, float] = {}

    # Global statistics.
    feats["mean"] = _safe(x.mean())
    feats["std"] = _safe(x.std())
    feats["skew"] = _safe(skew(x))
    feats["kurtosis"] = _safe(kurtosis(x))
    feats["ptp"] = _safe(np.ptp(x))
    feats["rms"] = _safe(np.sqrt(np.mean(x**2)))

    peaks = _find_systolic_peaks(x, fs)
    feats["n_beats"] = float(peaks.size)

    if peaks.size >= 2:
        ibi = np.diff(peaks) / fs
        feats["hr_mean"] = _ratio(60.0, float(np.mean(ibi)))
        feats["ibi_mean"] = _safe(np.mean(ibi))
        feats["ibi_std"] = _safe(np.std(ibi))
        # Short-window HRV proxies; 2.1 s gives only a couple of intervals, so
        # these are weak but not meaningless.
        feats["ibi_rmssd"] = _safe(np.sqrt(np.mean(np.diff(ibi) ** 2))) if ibi.size > 1 else 0.0
    else:
        feats.update({"hr_mean": 0.0, "ibi_mean": 0.0, "ibi_std": 0.0, "ibi_rmssd": 0.0})

    # Per-beat contour features, aggregated.
    per_beat: list[dict[str, float]] = []
    if peaks.size >= 2:
        for i in range(peaks.size - 1):
            start, end = int(peaks[i]), int(peaks[i + 1])
            # Extend the window back to the preceding trough so the beat starts
            # at pulse onset rather than at the previous peak.
            onset = start
            if i == 0:
                onset = int(np.argmin(x[: start + 1])) if start > 0 else 0
            else:
                prev = int(peaks[i - 1])
                onset = prev + int(np.argmin(x[prev : start + 1]))
            beat = x[onset:end]
            sys_idx = start - onset
            if 0 < sys_idx < beat.size:
                per_beat.append(_beat_features(beat, fs, sys_idx))

    if per_beat:
        keys = sorted({k for d in per_beat for k in d})
        for k in keys:
            vals = [d[k] for d in per_beat if k in d]
            feats[f"{k}_mean"] = _safe(np.mean(vals))
            feats[f"{k}_std"] = _safe(np.std(vals)) if len(vals) > 1 else 0.0

    feats.update(_apg_features(x, fs))
    feats.update(_spectral_features(x, fs))

    return {k: _safe(v) for k, v in feats.items()}


#: Canonical, stable feature ordering. Built once from a synthetic pulse train
#: so that every segment yields the same columns even when a fiducial point is
#: not detectable in a particular beat.
def _reference_feature_names(fs: float = FS_RAW) -> list[str]:
    t = np.arange(0, 2.1, 1.0 / fs)
    # A crude two-component pulse: systolic hump + dicrotic hump.
    hr = 1.2
    phase = (t * hr) % 1.0
    wave = np.exp(-((phase - 0.15) ** 2) / 0.004) + 0.4 * np.exp(
        -((phase - 0.42) ** 2) / 0.006
    )
    return sorted(extract_features(wave, fs).keys())


FEATURE_NAMES: list[str] = _reference_feature_names()


def extract_features_batch(
    X: np.ndarray,
    fs: float = FS_RAW,
    progress: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Extract features for ``(N, T)`` -> ``(N, F)`` plus the column names.

    Missing features (an undetected dicrotic notch, say) become 0.0 rather than
    NaN, so the matrix drops straight into scikit-learn.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected (N, T), got {X.shape}")

    it = range(X.shape[0])
    if progress:
        try:
            from tqdm.auto import tqdm

            it = tqdm(it, desc="features")
        except ImportError:
            pass

    rows = [extract_features(X[i], fs=fs) for i in it]
    names = sorted({k for r in rows for k in r} | set(FEATURE_NAMES))
    out = np.zeros((len(rows), len(names)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, name in enumerate(names):
            out[i, j] = r.get(name, 0.0)
    return out, names
