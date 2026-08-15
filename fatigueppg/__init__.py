"""Fatigue estimation from PPG dicrotic-peak position.

A faithful, tested implementation of

    Chen, Y.-X.; Tseng, C.-K.; Kuo, J.-T.; Wang, C.-J.; Chao, S.-H.; Kau, L.-J.;
    Hwang, Y.-S.; Lin, C.-L. "Fatigue Estimation Using Peak Features from PPG
    Signals." Mathematics 2023, 11, 3580. doi:10.3390/math11163580

Quick start
-----------
>>> from fatigueppg import analyse_ppg, assess, synth_ppg
>>> signal, _ = synth_ppg(duration=120, fs=200, dicrotic=0.8)
>>> result, analysis = assess(signal, fs=200, name="demo")
>>> round(result["fatigue_index"], 2)                      # doctest: +SKIP
7.04

Command line
------------
    python -m fatigueppg.selfcheck
    python -m fatigueppg.infer   --input recording.csv --fs 200
    python -m fatigueppg.extract --manifest cohort.csv -o features.csv
    python -m fatigueppg.train   --features features.csv -o models/mine.json
"""
from __future__ import annotations

__version__ = "1.0.0"

from .analysis import PPGAnalysis, analyse_ppg, window_analysis
from .fatigue import cycle_fatigue_index, dicrotic_ratio
from .hrv import hrv_indices, rr_series
from .model import FatigueModel, load_model, paper_model
from .peaks import estimate_cycle, find_dicrotic, find_onsets, find_systolic_peaks
from .preprocess import bandpass, normalize_paper, preprocess
from .quality import beat_template_sqi
from .signals import Recording, load_signal
from .stats import bfi_score, linreg, pearson
from .synth import synth_ppg


def __getattr__(name):
    """Import the CLI modules only when asked for.

    Importing ``fatigueppg.infer`` eagerly here would make
    ``python -m fatigueppg.infer`` load the module twice and warn about it.
    """
    if name in ("assess", "assess_file"):
        from . import infer
        return getattr(infer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    # pipeline
    "preprocess", "bandpass", "normalize_paper",
    "estimate_cycle", "find_systolic_peaks", "find_onsets", "find_dicrotic",
    "cycle_fatigue_index", "dicrotic_ratio",
    "rr_series", "hrv_indices",
    "analyse_ppg", "window_analysis", "PPGAnalysis",
    "beat_template_sqi",
    # use
    "assess", "assess_file", "load_signal", "Recording",
    "FatigueModel", "load_model", "paper_model",
    "pearson", "linreg", "bfi_score",
    "synth_ppg",
]
