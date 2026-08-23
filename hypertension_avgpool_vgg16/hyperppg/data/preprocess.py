"""Signal conditioning for PPG-BP.

Two pipelines are provided:

``paper``  -- what the paper describes: a length-50 moving average, then
              min-max normalisation to [0, 1]. Kept faithful so the
              replication numbers mean something.

``clean``  -- a standard PPG chain: linear detrend, 0.5-8 Hz Butterworth
              band-pass (zero-phase), decimation to 125 Hz, per-segment
              z-scoring. This is what the improved 1-D models consume.

All functions accept either a single ``(T,)`` signal or a batch ``(N, T)`` and
operate along the last axis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from hyperppg.config import FS_RAW

__all__ = [
    "moving_average",
    "detrend",
    "bandpass",
    "resample_to",
    "resampled_length",
    "minmax",
    "zscore",
    "paper_pipeline",
    "clean_pipeline",
    "CLEAN_FS",
    "CLEAN_SEQ_LEN",
]


def _ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """Reduced (up, down) integer resampling ratio."""
    from math import gcd

    fi, fo = int(round(fs_in)), int(round(fs_out))
    g = gcd(fi, fo)
    return fo // g, fi // g


def resampled_length(n: int, fs_in: float, fs_out: float) -> int:
    """Output length of :func:`resample_to`.

    ``scipy.signal.resample_poly`` yields ``ceil(n * up / down)``, so 2100
    samples at 1000 -> 125 Hz gives 263 (not 262). Use this instead of
    computing lengths by hand -- an off-by-one here silently reshapes every
    downstream tensor.
    """
    if fs_in == fs_out:
        return int(n)
    up, down = _ratio(fs_in, fs_out)
    return -(-int(n) * up // down)  # ceil division


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D input, got shape {x.shape}")
    return x, False


def _restore(x: np.ndarray, squeezed: bool) -> np.ndarray:
    return x[0] if squeezed else x


def moving_average(x: np.ndarray, window: int = 50) -> np.ndarray:
    """Length-preserving moving average -- the paper's smoothing step.

    ``window=50`` at 1000 Hz is a 50 ms box filter (first null at 20 Hz), which
    removes high-frequency sensor noise while leaving the systolic upstroke and
    dicrotic notch intact.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x, sq = _as_2d(x)
    out = uniform_filter1d(x, size=window, axis=-1, mode="nearest")
    return _restore(out.astype(np.float32), sq)


def detrend(x: np.ndarray) -> np.ndarray:
    """Remove a per-segment linear trend (slow baseline drift)."""
    x, sq = _as_2d(x)
    out = sps.detrend(x, axis=-1, type="linear")
    return _restore(out.astype(np.float32), sq)


def bandpass(
    x: np.ndarray,
    fs: float = FS_RAW,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    0.5 Hz strips baseline wander / respiration; 8 Hz keeps roughly the first
    ten harmonics of the pulse, which is where the dicrotic notch lives.
    """
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    if not 0 < low < high:
        raise ValueError(f"invalid band: low={low}, high={high}, fs={fs}")

    x, sq = _as_2d(x)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    # padlen must stay below the segment length for short records.
    padlen = min(3 * (2 * order), x.shape[-1] - 1)
    out = sps.sosfiltfilt(sos, x, axis=-1, padlen=max(padlen, 0))
    return _restore(out.astype(np.float32), sq)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Rational-factor resampling along the last axis."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float32)
    x, sq = _as_2d(x)
    up, down = _ratio(fs_in, fs_out)
    out = sps.resample_poly(x, up=up, down=down, axis=-1)
    return _restore(out.astype(np.float32), sq)


def minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each segment independently into [0, 1]."""
    x, sq = _as_2d(x)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    out = (x - lo) / np.maximum(hi - lo, eps)
    return _restore(out.astype(np.float32), sq)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean unit-variance per segment."""
    x, sq = _as_2d(x)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    out = (x - mu) / np.maximum(sd, eps)
    return _restore(out.astype(np.float32), sq)


# --------------------------------------------------------------------------
# Named pipelines
# --------------------------------------------------------------------------

def paper_pipeline(x: np.ndarray, window: int = 50) -> np.ndarray:
    """Faithful reproduction of the paper's preprocessing.

    "Moving average method was used with window size of 50 [...] the signals
    are normalized from 0 to 1."
    """
    return minmax(moving_average(x, window=window))


def clean_pipeline(
    x: np.ndarray,
    fs_in: float = FS_RAW,
    fs_out: float = 125.0,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Detrend -> band-pass -> decimate -> z-score.

    At the default 125 Hz a 2.1 s segment becomes 262 samples, which is a
    comfortable sequence length for the 1-D encoder while preserving the
    morphology the classifier needs.
    """
    out = detrend(x)
    out = bandpass(out, fs=fs_in, low=low, high=high, order=order)
    out = resample_to(out, fs_in=fs_in, fs_out=fs_out)
    return zscore(out)


#: Default working rate for the 1-D models, and the sequence length a full
#: 2100-sample PPG-BP segment becomes at that rate.
CLEAN_FS = 125.0
CLEAN_SEQ_LEN = resampled_length(2100, FS_RAW, CLEAN_FS)  # -> 263

PIPELINES = {
    "paper": paper_pipeline,
    "clean": clean_pipeline,
    "raw": lambda x: np.asarray(x, dtype=np.float32),
}
