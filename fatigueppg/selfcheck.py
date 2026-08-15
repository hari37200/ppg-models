"""Validate every stage against ground truth before trusting any number.

Each check names the section of the paper it covers. Run it after installing,
after changing anything, and before reporting a result:

    python -m fatigueppg.selfcheck
"""
from __future__ import annotations

import numpy as np

from .analysis import analyse_ppg
from .config import ALERT_THRESHOLD, FI_MAX, FS_PAPER, PAPER_EQ9
from .hrv import hrv_indices, rr_series
from .model import paper_model
from .peaks import find_dicrotic
from .preprocess import bandpass, normalize_paper
from .stats import linreg
from .synth import shoulder_beat, synth_ppg

__all__ = ["selfcheck", "main"]


def selfcheck(verbose=True):
    """-> ``[(name, passed, detail)]``."""
    checks, fs = [], FS_PAPER

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # -- Section 3.3 -------------------------------------------------------
    t = np.arange(0, 20, 1 / fs)
    edge = int(2 * fs)
    drift = np.sin(2 * np.pi * 0.05 * t)
    att = np.ptp(bandpass(drift, fs)[edge:-edge]) / np.ptp(drift)
    add("3.3 band-pass rejects 0.05 Hz drift", att < 0.1, f"residual {att:.4f}")

    pulse = np.sin(2 * np.pi * 1.2 * t)
    gain = np.ptp(bandpass(pulse, fs)[edge:-edge]) / np.ptp(pulse)
    add("3.3 band-pass preserves the 1.2 Hz pulse", 0.9 < gain < 1.1,
        f"gain {gain:.3f}")

    z = normalize_paper(bandpass(synth_ppg(30, fs, seed=0)[0], fs))
    add("3.3 Eq. (1) gives a peak-to-peak range of 2",
        abs(np.ptp(z) - 2.0) < 1e-6, f"ptp {np.ptp(z):.6f}")

    # -- Section 3.4 -------------------------------------------------------
    x, truth = synth_ppg(120, fs, hr=72, dicrotic=0.8, seed=1)
    res = analyse_ppg(x, fs)
    add("3.4.1 cycle search recovers the heart rate", abs(res.hr - 72) < 3,
        f"{res.hr:.1f} bpm vs 72, via {res.how}")

    err = abs(res.peaks.size - truth["peaks"].size) / truth["peaks"].size
    add("3.4.1 systolic peak count within 3% of truth", err <= 0.03,
        f"{res.peaks.size} vs {truth['peaks'].size}")

    off = np.median([np.min(np.abs(truth["peaks"] - p)) for p in res.peaks]) / fs
    add("3.4.1 systolic peaks land within 20 ms", off <= 0.020,
        f"median {off*1e3:.1f} ms")

    add("3.4.2 onsets precede and undercut their peaks",
        bool(np.all(res.onsets < res.peaks))
        and bool(np.all(res.signal[res.onsets] < res.signal[res.peaks])))

    add("3.4.3 dicrotic peak found in >=95% of cycles",
        res.detection_rate >= 0.95,
        f"{res.n_valid}/{res.n_cycles} = {res.detection_rate:.3f}")

    d = res.beats.loc[res.beats["diastolic"] > 0, "diastolic"].to_numpy()
    off = np.median([np.min(np.abs(truth["dpeaks"] - p)) for p in d]) / fs
    add("3.4.3 diastolic peaks land within 40 ms", off <= 0.040,
        f"median {off*1e3:.1f} ms")

    beat, sysi = shoulder_beat(fs)
    n_, d_, strat = find_dicrotic(beat, sysi, beat.size - 1, fs)
    add("3.4.3 strategy 2 handles an inconspicuous dicrotic wave",
        strat == "second-derivative" and 0 < n_ < d_,
        f"{strat}, notch < peak = {n_ < d_}")

    # -- Section 3.6 -------------------------------------------------------
    fi = res.beats["fi"].to_numpy(dtype=float)
    fi = fi[np.isfinite(fi)]
    add("3.6 every cycle index lies in [0, 10]",
        bool(np.all((fi >= 0) & (fi <= FI_MAX))),
        f"[{fi.min():.2f}, {fi.max():.2f}]")

    heights = np.arange(0.55, 0.95, 0.05)
    idx = [analyse_ppg(synth_ppg(60, fs, dicrotic=h, seed=2)[0], fs).fatigue_index
           for h in heights]
    r = float(np.corrcoef(heights, idx)[0, 1])
    add("3.6 index tracks the dicrotic-peak height",
        r >= 0.99 and bool(np.all(np.diff(idx) > 0)), f"Pearson r = {r:.4f}")

    # -- Section 3.5 -------------------------------------------------------
    tt, rr = rr_series(res.peaks, fs)
    add("3.5 R-R series matches the synthetic heart rate",
        abs(np.mean(rr) - 60 / 72) / (60 / 72) < 0.02, f"mean RR {np.mean(rr):.4f} s")

    hrv = hrv_indices(tt, rr)
    add("3.5 Eq. (5) + Eq. (6): NLF + NHF = 100",
        abs(hrv["nlf"] + hrv["nhf"] - 100) < 1e-6,
        f"{hrv['nlf']:.2f} + {hrv['nhf']:.2f}")

    nhf_r = hrv_indices(*rr_series(analyse_ppg(
        synth_ppg(180, fs, hrv_sd=0.02, resp_depth=0.06, seed=3)[0], fs).peaks, fs))["nhf"]
    nhf_0 = hrv_indices(*rr_series(analyse_ppg(
        synth_ppg(180, fs, hrv_sd=0.02, resp_depth=0.0, seed=3)[0], fs).peaks, fs))["nhf"]
    add("3.5 NHF responds to 0.25 Hz respiratory modulation", nhf_r > nhf_0 + 5,
        f"{nhf_0:.1f} -> {nhf_r:.1f}")

    # -- quality guard (not in the paper) ----------------------------------
    noise = np.random.default_rng(0).standard_normal(int(120 * 64))
    sqi_noise = analyse_ppg(noise, 64.0).sqi
    add("SQI separates a pulse train from white noise",
        res.sqi > 0.90 and sqi_noise < 0.70,
        f"pulse {res.sqi:.2f} vs noise {sqi_noise:.2f}")

    inv = analyse_ppg(-x, fs).sqi
    add("inverted input scores worse than upright", inv < res.sqi,
        f"{inv:.2f} vs {res.sqi:.2f}")

    # -- Sections 3.7 - 3.9 ------------------------------------------------
    xs = np.linspace(2, 9, 25)
    fit = linreg(xs, PAPER_EQ9[0] + PAPER_EQ9[1] * xs)
    add("3.7 Eq. (7) recovers the published Eq. (9) coefficients",
        abs(fit["a"] - 3.1) < 1e-6 and abs(fit["b"] - 0.6) < 1e-6,
        f"a = {fit['a']:.4f}, b = {fit['b']:.4f}")

    m = paper_model()
    add("3.9 Eq. (9) and the reminder threshold of 6",
        abs(float(m.predict(6.5)) - 7.0) < 1e-9 and m.alert(6.5) and not m.alert(5.5),
        f"FI 6.5 -> {float(m.predict(6.5)):.2f}")

    if verbose:
        for name, ok, detail in checks:
            print(f"  [{'ok  ' if ok else 'FAIL'}] {name}"
                  + (f"   ({detail})" if detail else ""))
        n_ok = sum(ok for _, ok, _ in checks)
        print(f"\n{n_ok}/{len(checks)} checks passed")
    return checks


def main(argv=None) -> int:
    checks = selfcheck()
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
