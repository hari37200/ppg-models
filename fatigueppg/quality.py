"""Signal quality -- not in the paper, but required outside its recording setup.

The paper measures seated fingertip PPG from a clinical device for two minutes.
Point the same pipeline at wrist BVP during exercise and it will return
confident, meaningless fiducials. The template-matching SQI below is the
standard guard: it scores above 0.97 on a clean pulse train and below 0.5 on
white noise, which is a wide enough gap to filter windows on.
"""
from __future__ import annotations

import numpy as np

__all__ = ["beat_template_sqi", "choose_orientation"]


def beat_template_sqi(x, peaks, cycle):
    """Mean correlation of each beat with the median beat (0-1); NaN if < 3 beats.

    Beats are aligned on the systolic peak, not the pulse onset: the onset is a
    broad minimum whose sample-level position wanders, and aligning on it costs
    about 0.5 of correlation on 64 Hz wrist BVP.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 3:
        return np.nan
    pre, post = int(0.3 * cycle), int(0.7 * cycle)
    if pre + post < 8:
        return np.nan
    mat = [x[p - pre: p + post] for p in peaks
           if p - pre >= 0 and p + post <= x.size]
    mat = [m for m in mat if m.size == pre + post]
    if len(mat) < 3:
        return np.nan
    mat = np.asarray(mat)
    mat = mat - mat.mean(axis=1, keepdims=True)
    mat = mat / np.maximum(mat.std(axis=1, keepdims=True), 1e-9)
    tmpl = np.median(mat, axis=0)
    tmpl = (tmpl - tmpl.mean()) / max(tmpl.std(), 1e-9)
    return float(np.clip(np.mean((mat @ tmpl) / (pre + post)), 0.0, 1.0))


def choose_orientation(raw, fs, analyse):
    """Decide whether a recording is inverted, by scoring both orientations.

    Transmissive PPG sensors output absorbance, so their pulses point *down*;
    reflective ones point up. Every fiducial rule in this package assumes the
    systolic peak is a maximum, so an inverted input silently produces
    garbage. Analysing both and keeping the higher-SQI orientation is cheap and
    removes a whole class of "it ran but the numbers are wrong".

    Returns ``(signal, inverted, sqi_normal, sqi_inverted)``.
    """
    raw = np.asarray(raw, dtype=np.float64).ravel()

    def score(sig):
        try:
            return analyse(sig, fs).sqi
        except Exception:
            return np.nan

    up, down = score(raw), score(-raw)
    if np.isfinite(down) and (not np.isfinite(up) or down > up + 0.02):
        return -raw, True, up, down
    return raw, False, up, down
