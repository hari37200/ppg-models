"""Synthetic PPG with known ground truth.

Used by the self-check (which needs to know where the peaks really are) and by
the demo cohort (which needs a fatigue level that provably drives the dicrotic
peak). Not a physiological model -- a two-Gaussian pulse with a diastolic decay
tail, which is enough to exercise every branch of the pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import DURATION_PAPER, FS_PAPER, RR_MIN

__all__ = ["synth_ppg", "shoulder_beat", "PULSE"]

#: Pulse shape in beat-phase units. ``b``/``c`` are the centre and width of the
#: systolic (1) and dicrotic (2) Gaussians.
PULSE = dict(b1=0.20, c1=0.085, b2=0.45, c2=0.115, tail=0.18, decay=2.5)


def _pulse(ph, dicrotic, p=PULSE):
    return (np.exp(-((ph - p["b1"]) / p["c1"]) ** 2)
            + dicrotic * np.exp(-((ph - p["b2"]) / p["c2"]) ** 2)
            + p["tail"] * np.exp(-p["decay"] * np.clip(ph - p["b1"], 0, None))
            * (ph > p["b1"]))


def _template_phases(dicrotic):
    """Where the systolic and diastolic peaks actually sit, in beat phase.

    Measured off the template rather than assumed from ``b1``/``b2``: the
    systolic decay pulls the dicrotic maximum earlier than its Gaussian centre,
    and the self-check compares detections against these positions.
    """
    grid = np.linspace(0.0, 1.0, 2001)
    loc, _ = sps.find_peaks(_pulse(grid, dicrotic))
    sys_ph = float(grid[loc[0]]) if loc.size else PULSE["b1"]
    dia_ph = float(grid[loc[1]]) if loc.size > 1 else PULSE["b2"]
    return sys_ph, dia_ph


def synth_ppg(duration=DURATION_PAPER, fs=FS_PAPER, hr=72.0, dicrotic=0.65,
              hrv_sd=0.03, resp_hz=0.25, resp_depth=0.0, noise=0.01,
              drift=0.0, seed=0):
    """-> ``(signal, truth)``.

    Parameters
    ----------
    dicrotic
        Height of the dicrotic component relative to the systolic one. This is
        the knob the fatigue index is supposed to track.
    hrv_sd
        Beat-to-beat jitter as a fraction of the mean R-R interval.
    resp_depth, resp_hz
        Sinusoidal R-R modulation, i.e. respiratory sinus arrhythmia. Drives
        HF power, so it moves NHF without touching the fatigue index.
    drift
        Amplitude of a 0.05 Hz baseline wander, for testing the band-pass.
    """
    rng = np.random.default_rng(seed)
    rr0 = 60.0 / hr
    times, t = [], 0.0
    while t < duration:
        times.append(t)
        jitter = hrv_sd * rr0 * rng.standard_normal()
        if resp_depth > 0:
            jitter += resp_depth * rr0 * np.sin(2 * np.pi * resp_hz * t)
        t += max(rr0 + jitter, RR_MIN)
    times = np.asarray(times)

    n = int(round(duration * fs))
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for t0 in times:
        ph = (tt - t0) / rr0
        m = (ph >= -0.12) & (ph < 1.12)
        x[m] += _pulse(ph[m], dicrotic)
    x += noise * rng.standard_normal(n)
    if drift > 0:
        x += drift * np.sin(2 * np.pi * 0.05 * tt)

    sys_ph, dia_ph = _template_phases(dicrotic)
    truth = dict(onsets=np.round(times * fs).astype(int),
                 peaks=np.round((times + sys_ph * rr0) * fs).astype(int),
                 dpeaks=np.round((times + dia_ph * rr0) * fs).astype(int),
                 hr=hr, dicrotic=dicrotic, sys_phase=sys_ph, dia_phase=dia_ph,
                 fs=fs)
    return x, truth


def shoulder_beat(fs=FS_PAPER, amp=0.10):
    """One pulse whose dicrotic wave never becomes a local maximum (Fig. 3b).

    Returns ``(signal, systolic_index)``. This is the case that forces the
    second-derivative strategy of Section 3.4.3.
    """
    t = np.arange(0, 0.9, 1.0 / fs)
    beat = (np.exp(-3.0 * t) + amp * np.exp(-((t - 0.30) ** 2) / 0.02)
            - amp * np.exp(-(0.30 ** 2) / 0.02))
    return np.concatenate([np.linspace(-0.6, beat[0], 30), beat]), 30
