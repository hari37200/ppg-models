"""PPG preprocessing -- Section 3.3 of the paper.

Band-pass 0.5-8 Hz to remove the DC offset and the drift that finger movement
causes, then Equation (1) to put every recording on the same amplitude scale.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import BAND_HIGH, BAND_LOW, BAND_ORDER

__all__ = ["bandpass", "normalize_paper", "preprocess", "clean_nans"]


def clean_nans(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Linearly interpolate over NaN/inf samples. Returns (signal, n_filled).

    Real recordings drop samples. Leaving them in makes filtfilt return an
    all-NaN array, which is a confusing way to find out.
    """
    x = np.asarray(x, dtype=np.float64).ravel().copy()
    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return x, 0
    if bad.all():
        raise ValueError("signal contains no finite samples")
    idx = np.arange(x.size)
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x, n_bad


def bandpass(x, fs, low=BAND_LOW, high=BAND_HIGH, order=BAND_ORDER):
    """Zero-phase Butterworth band-pass.

    0.5 Hz strips baseline wander and respiration; 8 Hz keeps roughly the first
    ten harmonics of the pulse, which is where the dicrotic notch lives. Being
    zero-phase matters here more than usual: every fiducial point downstream is
    a *position*, and a filter with group delay would shift them all.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    nyq = fs / 2.0
    hi = min(high, nyq * 0.99)
    if not 0.0 < low < hi:
        raise ValueError(f"invalid band: low={low}, high={hi}, fs={fs}")
    if x.size < 16:
        raise ValueError(f"signal too short to filter: {x.size} samples")
    sos = sps.butter(order, [low / nyq, hi / nyq], btype="bandpass", output="sos")
    padlen = int(min(3 * (2 * order), x.size - 1))
    return sps.sosfiltfilt(sos, x, padlen=max(padlen, 0))


def normalize_paper(x, literal=True):
    """Equation (1): ``X_normalize = 2 * x / (x_max - x_min)``.

    Written literally this scales the *range* to 2 without centring. It lands
    roughly on [-1, 1], as the paper says, only because the zero-phase
    band-pass has already removed the mean; for a strongly asymmetric pulse the
    positive side still overshoots 1.

    ``literal=False`` centres explicitly. The difference is a constant offset
    per recording and it *does* move the fatigue index, so the default stays
    faithful to Equation (1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    rng = float(np.ptp(x))
    if rng < 1e-12:
        return np.zeros_like(x)
    if literal:
        return 2.0 * x / rng
    return 2.0 * (x - 0.5 * (float(x.max()) + float(x.min()))) / rng


def preprocess(raw, fs, low=BAND_LOW, high=BAND_HIGH, order=BAND_ORDER, literal=True):
    """Section 3.3 end to end: NaN repair -> band-pass -> Equation (1)."""
    x, _ = clean_nans(raw)
    return normalize_paper(bandpass(x, fs, low, high, order), literal=literal)
