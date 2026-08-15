"""Peak detection -- Section 3.4 of the paper.

Four fiducial points per cycle: pulse onset, systolic peak, dicrotic notch and
diastolic peak. The systolic search is the paper's derivative/zero-crossing
procedure, implemented from the text; the deviations it needs to survive real
signals are documented on the functions that make them.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import CYCLE0, CYCLE_STEP, RR_MAX, RR_MIN, SEARCH_FRAC

__all__ = [
    "estimate_cycle",
    "find_systolic_peaks",
    "find_onsets",
    "find_dicrotic",
]


# --------------------------------------------------------------------------
# Section 3.4.1 -- systolic peak
# --------------------------------------------------------------------------

def _merge_close(peaks, x, min_dist):
    """Collapse peaks closer than ``min_dist`` samples, keeping the taller."""
    if peaks.size == 0:
        return peaks
    keep = [int(peaks[0])]
    for p in peaks[1:].astype(int):
        if p - keep[-1] < min_dist:
            if x[p] > x[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    return np.asarray(keep, dtype=int)


def _temple_peaks(seg, cycle, min_dist):
    """Steps 2A-2C: block maxima -> slope code -> refined maxima.

    ``point(n)`` is the maximum of block n. Equations (2) and (3) code the
    slope between adjacent blocks; where ``slope(n) = 1`` and
    ``slope(n+1) = 0`` the sequence turns over, so the maximum from
    ``point(n)`` to ``point(n) + cycle`` is a temple systolic peak.
    """
    nb = seg.size // cycle
    if nb < 3:
        return np.empty(0, dtype=int)
    blocks = seg[: nb * cycle].reshape(nb, cycle)
    point = blocks.max(axis=1)
    loc = blocks.argmax(axis=1) + np.arange(nb) * cycle
    slope = (point[:-1] < point[1:]).astype(int)              # Eq. (2) / (3)
    if slope.size < 2:
        return np.empty(0, dtype=int)
    turns = np.flatnonzero((slope[:-1] == 1) & (slope[1:] == 0))
    cand = []
    for n in turns:
        s = int(loc[n])
        e = min(s + cycle, seg.size)
        if e - s >= 2:
            cand.append(s + int(np.argmax(seg[s:e])))
    if not cand:
        return np.empty(0, dtype=int)
    # DEVIATION: de-duplicate. Without this, a small block size locks onto
    # systolic *and* dicrotic peaks together and the search accepts a period
    # half the true one on its first iteration.
    return _merge_close(np.unique(np.asarray(cand, dtype=int)), seg, min_dist)


def _autocorr_period(seg, fs, rr_min=RR_MIN, rr_max=RR_MAX):
    """Fallback beat period: strongest autocorrelation lag in the valid band."""
    s = seg - seg.mean()
    ac = np.correlate(s, s, mode="full")[s.size - 1:]
    lo, hi = max(int(rr_min * fs), 1), min(int(rr_max * fs), ac.size - 1)
    if hi <= lo:
        return 0.8
    return float((lo + int(np.argmax(ac[lo:hi]))) / fs)


def estimate_cycle(x, fs, probe_seconds=10.0, cycle0=CYCLE0, step=CYCLE_STEP,
                   rr_min=RR_MIN, rr_max=RR_MAX, cv_max=0.30, fallback=True):
    """Section 3.4.1, steps 1-2 -> ``(cycle_samples, hr_bpm, how)``.

    Take 10 s of normalised signal, block it at ``cycle0`` points, find the
    temple systolic peaks, average their spacing. If that lands outside
    0.3-1.5 s, add 5 points to the block and try again (step 2E).

    DEVIATION: the paper's step 3 says to reuse "the calculation cycle" over
    the whole 2-minute record, but never says whether that is the search block
    or one heartbeat. As the block it would slice 2 minutes into thousands of
    pieces; Section 3.4 defines a cycle as one heart contraction to the next.
    So the *accepted median R-R interval* is returned as the cycle and the
    block size stays an internal search parameter.

    A coefficient-of-variation bound is also applied, because a period can be
    numerically inside 0.3-1.5 s while the peaks it came from are nonsense.
    """
    x = np.asarray(x, dtype=np.float64)
    seg = x[: int(min(x.size, round(probe_seconds * fs)))]
    if seg.size < 16:
        raise ValueError(
            f"need at least 16 samples to estimate a cycle, got {seg.size}")

    min_dist = max(int(round(rr_min * fs)), 2)
    limit = max(seg.size // 3, cycle0 + 1)
    cycle = int(cycle0)
    while cycle <= limit:
        peaks = _temple_peaks(seg, cycle, min_dist)
        if peaks.size >= 2:
            rr = np.diff(peaks) / fs                          # step 2D
            med = float(np.median(rr))
            cv = float(np.std(rr) / max(med, 1e-9)) if rr.size >= 2 else 0.0
            if rr_min <= med <= rr_max and cv <= cv_max:
                return int(round(med * fs)), 60.0 / med, f"paper(C={cycle})"
        cycle += step                                         # step 2E

    if not fallback:
        raise RuntimeError("cycle search did not converge")
    rr = _autocorr_period(seg, fs, rr_min, rr_max)
    return int(round(rr * fs)), 60.0 / rr, "autocorr"


def _fill_gaps(x, peaks, cycle):
    """Recover beats lost when a block boundary splits a pulse.

    Fixed-length blocks do not align with beats, so a block whose maximum sits
    on its edge contributes no local maximum and that beat disappears. Any
    interval longer than 1.5 cycles is re-searched for the tallest local peak.
    On synthetic data this takes recall from 141/145 beats to 145/145.
    """
    if peaks.size < 2:
        return peaks
    out = list(map(int, peaks))
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b - a <= 1.5 * cycle:
            continue
        lo, hi = int(a) + int(0.5 * cycle), int(b) - int(0.5 * cycle)
        if hi - lo < 3:
            continue
        loc, _ = sps.find_peaks(x[lo:hi])
        if loc.size:
            out.append(lo + int(loc[np.argmax(x[lo:hi][loc])]))
    return _merge_close(np.unique(np.asarray(out, dtype=int)), x,
                        max(int(0.5 * cycle), 2))


def find_systolic_peaks(x, fs, cycle, method="paper"):
    """Section 3.4.1, step 3: the highest peak in each calculation cycle.

    ``method="scipy"`` swaps in a conventional constrained peak finder. On the
    657 real fingertip segments of PPG-BP the two agree on the exact sample for
    94.8% of peaks, which is the evidence that the paper's procedure is sound.
    """
    x = np.asarray(x, dtype=np.float64)
    cycle = max(int(cycle), 2)
    if method == "scipy":
        prom = 0.15 * float(np.std(x))
        peaks, _ = sps.find_peaks(x, distance=max(int(0.6 * cycle), 1),
                                  prominence=prom if prom > 0 else None)
        return peaks.astype(int)
    if method != "paper":
        raise ValueError(f"unknown method {method!r}; use 'paper' or 'scipy'")

    cand = []
    for b in range(int(np.ceil(x.size / cycle))):
        s, e = b * cycle, min((b + 1) * cycle, x.size)
        if e - s >= 2:
            cand.append(s + int(np.argmax(x[s:e])))
    cand = np.unique(np.asarray(cand, dtype=int))
    interior = np.array([p for p in cand if 0 < p < x.size - 1
                         and x[p] >= x[p - 1] and x[p] >= x[p + 1]], dtype=int)
    if interior.size >= 2:
        cand = interior
    return _fill_gaps(x, _merge_close(cand, x, max(int(0.5 * cycle), 2)), cycle)


# --------------------------------------------------------------------------
# Section 3.4.2 -- pulse onset
# --------------------------------------------------------------------------

def find_onsets(x, peaks, cycle):
    """The minimum of the cycle preceding each systolic peak."""
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    onsets = np.empty(peaks.size, dtype=int)
    for i, p in enumerate(peaks):
        start = int(peaks[i - 1]) if i > 0 else max(0, int(p) - int(cycle))
        start = min(start, max(int(p) - 1, 0))
        onsets[i] = start + int(np.argmin(x[start: int(p) + 1]))
    return onsets


# --------------------------------------------------------------------------
# Section 3.4.3 -- dicrotic notch and diastolic peak
# --------------------------------------------------------------------------

def find_dicrotic(x, sys_idx, next_onset, fs, search_frac=SEARCH_FRAC,
                  smooth_ms=20.0, min_rise=0.02):
    """-> ``(notch, diastolic_peak, strategy)``; ``-1, -1, "none"`` if absent.

    The window runs from the systolic peak to half way to the next pulse onset,
    as specified. Strategy 1 (conspicuous dicrotic wave, Fig. 3a) reads the
    zeros of the first-order difference ``y[n] = x[n] - x[n-1]``, Equation (4).
    Strategy 2 (inconspicuous, Fig. 3b/4) reads the second-order difference.

    Two deviations, both forced by noise:

    * The paper switches to strategy 2 when *no* first-order difference is
      positive. With any real sensor noise that never happens, so strategy 1
      must additionally clear a prominence bar -- the notch-to-peak rise must
      exceed ``min_rise`` of the pulse height -- before it counts as a genuine
      dicrotic wave.
    * Strategy 2 uses the most prominent *local* extrema of the second-order
      difference rather than its global argmax/argmin: on a decaying limb the
      global curvature maximum always sits at the systolic peak itself.

    Derivatives are taken on a lightly smoothed copy (``smooth_ms``, an order
    of magnitude shorter than the dicrotic feature) because differentiating raw
    samples at 1 kHz is dominated by quantisation noise.
    """
    x = np.asarray(x, dtype=np.float64)
    sys_idx, next_onset = int(sys_idx), int(next_onset)
    end = sys_idx + int(round(search_frac * (next_onset - sys_idx)))
    end = min(max(end, sys_idx + 4), x.size - 1)
    w = x[sys_idx: end + 1]
    if w.size < 5:
        return -1, -1, "none"

    k = int(round(smooth_ms * 1e-3 * fs))
    if k >= 3 and w.size > 3 * k:
        w = np.convolve(w, np.ones(k) / k, mode="same")
        w[:k] = x[sys_idx: sys_idx + k]           # keep the peak edge honest
    bar = max(min_rise * float(w[0] - w.min()), 1e-9)

    y = np.diff(w)                                            # Eq. (4)
    if np.any(y > 0):
        sign = np.sign(y)
        up = np.flatnonzero((sign[:-1] <= 0) & (sign[1:] > 0))   # local minima
        dn = np.flatnonzero((sign[:-1] >= 0) & (sign[1:] < 0))   # local maxima
        for u in up:
            after = dn[dn > u]
            if not after.size:
                continue
            n_rel, d_rel = int(u) + 1, int(after[0]) + 1
            if w[d_rel] - w[n_rel] >= bar:
                return sys_idx + n_rel, sys_idx + d_rel, "first-derivative"

    z = np.diff(w, n=2)
    if z.size < 5:
        return -1, -1, "none"
    hi, _ = sps.find_peaks(z, prominence=0.0)
    if hi.size == 0:
        return -1, -1, "none"
    a = int(hi[int(np.argmax(sps.peak_prominences(z, hi)[0]))])
    lo, _ = sps.find_peaks(-z[a:], prominence=0.0)
    if lo.size == 0:
        return -1, -1, "none"
    b = a + int(lo[int(np.argmax(sps.peak_prominences(-z[a:], lo)[0]))])
    return sys_idx + a + 1, sys_idx + b + 1, "second-derivative"
