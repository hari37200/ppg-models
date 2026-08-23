"""HRV indices -- Section 3.5 of the paper.

R-R intervals come from adjacent systolic peaks, each stamped at the *previous*
peak's time. The series is resampled to 250 Hz, Fourier transformed with a
Hamming window, and integrated over VLF, LF and HF:

    NLF = LF / (total power - VLF) * 100      (5)
    NHF = HF / (total power - VLF) * 100      (6)

with total power = LF + HF + VLF, so the denominator is LF + HF and the two
normalised units always sum to 100. NHF is also divided by 10 to sit on the
same 0-10 scale as the fatigue index, per Section 3.5 step 4.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import (HF_BAND, HRV_FS_INTERP, HRV_MIN_SECONDS, LF_BAND, RR_MAX,
                     RR_MIN, VLF_BAND)

__all__ = ["rr_series", "hrv_indices", "HRV_KEYS"]

HRV_KEYS = ("nlf", "nhf", "nhf_0_10", "lf_hf", "sdnn", "rmssd", "n_rr")


def rr_series(peaks, fs, rr_min=RR_MIN, rr_max=RR_MAX):
    """-> ``(t, rr)``: the time of the previous systolic peak, and the interval.

    Intervals outside the physiological band are dropped rather than repaired;
    a missed beat produces a double-length interval that would otherwise inject
    a large spurious low-frequency component.
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size < 3:
        return np.empty(0), np.empty(0)
    rr, t = np.diff(peaks) / fs, peaks[:-1] / fs
    ok = (rr >= rr_min) & (rr <= rr_max)
    return t[ok], rr[ok]


def hrv_indices(t, rr, fs_interp=HRV_FS_INTERP, min_seconds=HRV_MIN_SECONDS):
    """Equations (5) and (6), plus the usual time-domain companions.

    Two things the paper leaves unsaid are done here: the tachogram is
    mean-removed (otherwise the DC bin swamps VLF) and expressed in
    milliseconds, so powers come out in the conventional ms^2. 250 Hz is a
    wildly oversampled grid for a signal whose content stops at 0.4 Hz -- it
    costs nothing and the band ratios are unaffected, but it is worth knowing
    that the paper's choice is not the usual 4 Hz.
    """
    out = dict(nlf=np.nan, nhf=np.nan, nhf_0_10=np.nan, lf_hf=np.nan,
               vlf=np.nan, lf=np.nan, hf=np.nan, mean_rr=np.nan,
               sdnn=np.nan, rmssd=np.nan, n_rr=int(np.size(rr)))
    if np.size(rr) < 8 or (t[-1] - t[0]) < min_seconds:
        return out

    out.update(mean_rr=float(np.mean(rr)), sdnn=float(np.std(rr, ddof=1)),
               rmssd=float(np.sqrt(np.mean(np.diff(rr) ** 2))))

    grid = np.arange(t[0], t[-1], 1.0 / fs_interp)
    tach = np.interp(grid, t, rr) * 1000.0            # ms -> powers in ms^2
    freqs, psd = sps.periodogram(tach - tach.mean(), fs=fs_interp,
                                 window="hamming", scaling="density")

    trapz = getattr(np, "trapezoid", None) or np.trapz

    def band(lo, hi):
        sel = (freqs >= lo) & (freqs < hi)
        return float(trapz(psd[sel], freqs[sel])) if sel.sum() > 1 else 0.0

    vlf, lf, hf = band(*VLF_BAND), band(*LF_BAND), band(*HF_BAND)
    if lf + hf <= 0:
        return out
    out.update(vlf=vlf, lf=lf, hf=hf,
               nlf=100.0 * lf / (lf + hf), nhf=100.0 * hf / (lf + hf),
               lf_hf=(lf / hf) if hf > 0 else np.nan)
    out["nhf_0_10"] = out["nhf"] / 10.0
    return out
