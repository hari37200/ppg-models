"""One recording, end to end: Sections 3.3 - 3.6 chained.

``analyse_ppg`` is the function everything else in the package calls. It keeps
every fiducial point per cycle, so a plot and a number always come from the
same object.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (BAND_HIGH, BAND_LOW, BAND_ORDER, DURATION_PAPER, MIN_SQI,
                     SEARCH_FRAC, ZERO_FRAC)
from .fatigue import cycle_fatigue_index, dicrotic_ratio
from .hrv import HRV_KEYS, hrv_indices, rr_series
from .peaks import estimate_cycle, find_dicrotic, find_onsets, find_systolic_peaks
from .preprocess import preprocess
from .quality import beat_template_sqi

__all__ = ["PPGAnalysis", "analyse_ppg", "window_analysis", "BEAT_COLUMNS"]

BEAT_COLUMNS = ["cycle_i", "onset", "systolic", "notch", "diastolic",
                "strategy", "rr", "ratio", "fi", "fi_onset"]


@dataclass
class PPGAnalysis:
    """Everything Sections 3.3-3.6 produce for one recording."""

    fs: float
    signal: np.ndarray            # preprocessed, Equation (1) scale
    cycle: int                    # samples per beat
    hr: float                     # bpm
    how: str                      # which branch of the cycle search fired
    peaks: np.ndarray
    onsets: np.ndarray
    beats: pd.DataFrame
    fatigue_index: float          # Section 3.6, zero at mid-height
    fatigue_index_onset: float    # same, zero at the pulse onset
    n_cycles: int
    n_valid: int
    sqi: float
    duration: float

    @property
    def detection_rate(self) -> float:
        """Fraction of cycles with a locatable diastolic peak."""
        return self.n_valid / self.n_cycles if self.n_cycles else 0.0

    def summary(self) -> dict:
        """Flat dict of the headline numbers, ready for JSON or a DataFrame."""
        return dict(
            fatigue_index=self.fatigue_index,
            fatigue_index_onset=self.fatigue_index_onset,
            hr=self.hr, sqi=self.sqi, cycle_samples=self.cycle,
            n_cycles=self.n_cycles, n_valid=self.n_valid,
            detection_rate=self.detection_rate, duration_s=self.duration,
            fs=self.fs, cycle_search=self.how,
        )


def analyse_ppg(raw, fs, low=BAND_LOW, high=BAND_HIGH, order=BAND_ORDER,
                search_frac=SEARCH_FRAC, method="paper", literal=True,
                zero_frac=ZERO_FRAC, preprocessed=False) -> PPGAnalysis:
    """Sections 3.3 - 3.6 for one recording.

    Parameters
    ----------
    raw
        Any 1-D PPG signal. Not preprocessed unless ``preprocessed=True``.
    fs
        Sampling rate in Hz. There is no default on purpose -- getting this
        wrong silently rescales every timing feature.
    method
        ``"paper"`` for Section 3.4.1's procedure, ``"scipy"`` for a
        conventional peak finder (used in the detector comparison).
    zero_frac
        Where the index's zero sits; 0.5 is the paper. See
        :func:`fatigueppg.fatigue.cycle_fatigue_index`.
    """
    x = (np.asarray(raw, dtype=np.float64).ravel() if preprocessed
         else preprocess(raw, fs, low=low, high=high, order=order, literal=literal))
    if x.size < 16:
        raise ValueError(f"signal too short: {x.size} samples at {fs} Hz")

    cycle, hr, how = estimate_cycle(x, fs)
    peaks = find_systolic_peaks(x, fs, cycle, method=method)
    onsets = find_onsets(x, peaks, cycle)

    rows = []
    for i in range(peaks.size - 1):
        o, p, o_next = int(onsets[i]), int(peaks[i]), int(onsets[i + 1])
        if not o < p < o_next:
            continue
        notch, dia, strat = find_dicrotic(x, p, o_next, fs, search_frac=search_frac)
        has_dia = dia > p
        rows.append(dict(
            cycle_i=i, onset=o, systolic=p, notch=notch, diastolic=dia,
            strategy=strat, rr=(int(peaks[i + 1]) - p) / fs,
            ratio=dicrotic_ratio(x, o, p, dia) if has_dia else np.nan,
            fi=cycle_fatigue_index(x, o, p, dia, zero_frac=zero_frac) if has_dia else np.nan,
            fi_onset=cycle_fatigue_index(x, o, p, dia, zero_frac=0.0) if has_dia else np.nan,
        ))

    beats = pd.DataFrame(rows, columns=BEAT_COLUMNS)
    fi = beats["fi"].to_numpy(dtype=float) if len(beats) else np.array([])
    fi0 = beats["fi_onset"].to_numpy(dtype=float) if len(beats) else np.array([])
    ok = np.isfinite(fi)

    return PPGAnalysis(
        fs=float(fs), signal=x, cycle=int(cycle), hr=float(hr), how=how,
        peaks=peaks, onsets=onsets, beats=beats,
        fatigue_index=float(np.mean(fi[ok])) if ok.any() else np.nan,
        fatigue_index_onset=float(np.mean(fi0[ok])) if ok.any() else np.nan,
        n_cycles=int(len(beats)), n_valid=int(ok.sum()),
        sqi=beat_template_sqi(x, peaks, cycle),
        duration=float(x.size / fs),
    )


def window_analysis(sig, fs, window_s=DURATION_PAPER, stride_s=None,
                    min_sqi=0.0, max_windows=None, with_hrv=True,
                    zero_frac=ZERO_FRAC, **tags) -> pd.DataFrame:
    """Slide the paper's 2-minute measurement over a long recording.

    Extra keyword arguments are copied onto every row, which is how subject and
    session identifiers get attached during batch extraction.
    """
    stride_s = window_s if stride_s is None else stride_s
    n = int(round(window_s * fs))
    step = max(int(round(stride_s * fs)), 1)
    starts = np.arange(0, max(len(sig) - n + 1, 0), step)
    if max_windows:
        starts = starts[:max_windows]

    rows = []
    for k, s in enumerate(starts):
        seg = np.asarray(sig[s: s + n], dtype=np.float64)
        if not np.isfinite(seg).all() or np.ptp(seg) < 1e-9:
            continue
        try:
            res = analyse_ppg(seg, fs, zero_frac=zero_frac)
        except Exception:
            continue
        row = dict(tags, window=k, t_start=float(s / fs), **res.summary())
        if with_hrv:
            h = hrv_indices(*rr_series(res.peaks, fs))
            row.update({key: h[key] for key in HRV_KEYS})
        rows.append(row)

    df = pd.DataFrame(rows)
    if min_sqi > 0 and len(df):
        df = df[df["sqi"] >= min_sqi]
    return df.reset_index(drop=True)
