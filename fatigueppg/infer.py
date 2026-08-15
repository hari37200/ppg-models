"""Inference: a raw PPG recording in, a fatigue index out.

This is the paper's Section 3.9 / 4.4 "fatigue evaluation system" -- the C#
window that loaded a COMGO file, printed an index and popped up a reminder --
as a library function and a command line.

    python -m fatigueppg.infer --input recording.csv --fs 200
    python -m fatigueppg.infer --demo
    python -m fatigueppg.infer --input data/ --glob "*.csv" --csv out.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import analyse_ppg
from .config import ALERT_THRESHOLD, FS_PAPER, MIN_SQI, ZERO_FRAC
from .hrv import hrv_indices, rr_series
from .model import FatigueModel, load_model
from .quality import choose_orientation
from .signals import Recording, expand_inputs, load_signal
from .synth import synth_ppg

__all__ = ["assess", "assess_file", "main"]

#: Below this the index is a guess, not a measurement.
MIN_CYCLES = 3
MIN_DURATION = 5.0


def assess(signal, fs, name="recording", model=None, invert="auto",
           zero_frac=None, threshold=None, verbose=False) -> tuple[dict, object]:
    """Index one recording and apply the calibration model.

    Returns ``(result_dict, PPGAnalysis)``. The analysis object carries every
    fiducial point, so a caller that wants a plot does not have to re-run
    anything.

    ``invert="auto"`` scores both orientations and keeps the better one:
    transmissive sensors output absorbance, so their pulses point down, and
    every rule in this package assumes the systolic peak is a maximum.
    """
    model = model or load_model()
    zero_frac = model.zero_frac if zero_frac is None else zero_frac
    threshold = model.threshold if threshold is None else threshold

    x = np.asarray(signal, dtype=np.float64).ravel()
    duration = x.size / fs
    if duration < MIN_DURATION:
        raise ValueError(
            f"{name}: {duration:.2f} s is too short. The paper measures for "
            f"120 s; below about {MIN_DURATION:g} s there are too few cycles "
            f"for the mean index to mean anything.")

    notes = []
    if invert == "auto":
        x, inverted, sqi_up, sqi_down = choose_orientation(
            x, fs, lambda s, f: analyse_ppg(s, f, zero_frac=zero_frac))
        if inverted:
            notes.append(f"signal looked inverted (SQI {sqi_up:.2f} upright vs "
                         f"{sqi_down:.2f} flipped); analysed flipped")
    elif invert in (True, "yes", "true"):
        x, notes = -x, notes + ["signal inverted on request"]

    res = analyse_ppg(x, fs, zero_frac=zero_frac)
    hrv = hrv_indices(*rr_series(res.peaks, fs))

    if res.n_valid < MIN_CYCLES:
        notes.append(f"only {res.n_valid} cycle(s) had a detectable diastolic "
                     f"peak -- treat the index as unreliable")
    if np.isfinite(res.sqi) and res.sqi < MIN_SQI:
        notes.append(f"signal quality {res.sqi:.2f} is below {MIN_SQI:g}; the "
                     f"waveform may be too noisy for the fiducials to be real")

    fi = res.fatigue_index
    result = dict(
        name=name,
        fatigue_index=fi,
        fatigue_index_onset=res.fatigue_index_onset,
        subjective_pred=float(model.predict(fi)) if np.isfinite(fi) else float("nan"),
        alert=model.alert(fi),
        threshold=float(threshold),
        model=model.name,
        equation=model.equation(),
        hr=res.hr,
        nhf=hrv["nhf"],
        nlf=hrv["nlf"],
        sdnn_ms=hrv["sdnn"] * 1000 if np.isfinite(hrv["sdnn"]) else float("nan"),
        rmssd_ms=hrv["rmssd"] * 1000 if np.isfinite(hrv["rmssd"]) else float("nan"),
        sqi=res.sqi,
        n_cycles=res.n_cycles,
        n_valid=res.n_valid,
        detection_rate=res.detection_rate,
        duration_s=res.duration,
        fs=float(fs),
        cycle_search=res.how,
        notes=notes,
    )
    if verbose:
        print(format_result(result))
    return result, res


def assess_file(path, fs=None, column=None, model=None, invert="auto",
                zero_frac=None, name=None, verbose=False):
    """:func:`assess` for a file on disk. Returns ``(result, analysis, rec)``."""
    rec = load_signal(path, fs=fs, column=column, name=name)
    result, res = assess(rec.signal, rec.fs, name=rec.name, model=model,
                         invert=invert, zero_frac=zero_frac, verbose=False)
    result["source"] = rec.source
    result["notes"] = rec.notes + result["notes"]
    if verbose:
        print(format_result(result))
    return result, res, rec


def format_result(r) -> str:
    """The human-readable block printed by the CLI."""
    nan = float("nan")
    lines = [
        f"  recording            {r['name']}  "
        f"({r['duration_s']:.1f} s at {r['fs']:g} Hz)",
        f"  fatigue index        {r['fatigue_index']:.4f}     "
        f"(0-10, Section 3.6)",
        f"  onset-referenced     {r['fatigue_index_onset']:.4f}     "
        f"(same index, zero at the pulse onset)",
        f"  predicted subjective {r['subjective_pred']:.2f}       "
        f"via {r['equation']}",
        f"  heart rate           {r['hr']:.1f} bpm",
        f"  NHF / NLF            {r.get('nhf', nan):.1f} / {r.get('nlf', nan):.1f}"
        f"     (Eq. 5, 6; needs > 60 s)",
        f"  signal quality       {r['sqi']:.2f}",
        f"  cycles               {r['n_cycles']} analysed, {r['n_valid']} with a "
        f"diastolic peak ({r['detection_rate']:.0%})",
    ]
    if r["alert"]:
        lines.append(f"\n  ** index above {r['threshold']:g} -- "
                     f"Take more rest today!!! **")
    else:
        lines.append(f"\n  index at or below {r['threshold']:g} -- no reminder")
    for n in r.get("notes", []):
        lines.append(f"  note: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="fatigue-infer",
        description="Fatigue index from a raw PPG recording (Chen et al. 2023).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python -m fatigueppg.infer --demo
  python -m fatigueppg.infer --input examples/demo_ppg_200hz.csv
  python -m fatigueppg.infer --input rec.txt --fs 200 --plot report.png
  python -m fatigueppg.infer --input recordings/ --glob "*.csv" --csv out.csv
""")
    p.add_argument("--input", "-i", nargs="+",
                   help="file, directory or glob. Any of "
                        ".csv .tsv .txt .dat .npy .json")
    p.add_argument("--glob", default="*", help="pattern used inside a directory")
    p.add_argument("--fs", type=float,
                   help="sampling rate in Hz. Required unless the file carries "
                        "one (time column, Empatica header, JSON 'fs')")
    p.add_argument("--column", help="signal column name or index, for tables")
    p.add_argument("--model", help="calibration model JSON "
                                   "(default: the paper's Eq. 9)")
    p.add_argument("--invert", choices=["auto", "yes", "no"], default="auto",
                   help="handle sensors whose pulses point down (default: auto)")
    p.add_argument("--zero-frac", type=float, default=None,
                   help=f"index zero point; {ZERO_FRAC} is the paper's, 0.0 "
                        f"references it to the pulse onset")
    p.add_argument("--json", help="write the full result(s) here")
    p.add_argument("--csv", help="write one row per recording here")
    p.add_argument("--plot", help="write a report figure here (single input), "
                                  "or a directory (multiple)")
    p.add_argument("--demo", action="store_true",
                   help="run on a synthetic recording; needs no data at all")
    p.add_argument("--quiet", "-q", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input and not args.demo:
        build_parser().print_help()
        print("\nnothing to do: pass --input or --demo")
        return 2

    model = load_model(args.model)
    if not args.quiet:
        print(f"model: {model.name}   {model.equation()}")
        if model.name == "paper-eq9":
            print("       (the published coefficients, fitted on the paper's 16 "
                  "participants --\n        refit them on your own cohort with "
                  "fatigueppg.train before clinical use)")
        print()

    results, analyses, failures = [], [], []

    if args.demo:
        sig, _ = synth_ppg(120.0, FS_PAPER, hr=72, dicrotic=0.82, noise=0.02, seed=3)
        r, res = assess(sig, FS_PAPER, name="synthetic-demo", model=model,
                        invert="no", zero_frac=args.zero_frac)
        r["source"] = "synthetic"
        results.append(r)
        analyses.append(res)
        if not args.quiet:
            print(format_result(r), "\n")

    for path in (expand_inputs(args.input, args.glob) if args.input else []):
        try:
            r, res, _ = assess_file(path, fs=args.fs, column=args.column,
                                    model=model, invert=args.invert,
                                    zero_frac=args.zero_frac)
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            print(f"[skip] {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(r)
        analyses.append(res)
        if not args.quiet:
            print(format_result(r), "\n")

    if not results:
        print("no recording could be processed", file=sys.stderr)
        return 1

    table = pd.DataFrame([{k: v for k, v in r.items() if k != "notes"}
                          for r in results])
    if len(results) > 1 and not args.quiet:
        cols = ["name", "fatigue_index", "subjective_pred", "alert", "hr",
                "sqi", "n_valid"]
        print(table[cols].round(4).to_string(index=False))
        print()

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2, default=str),
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    if args.plot:
        from .plotting import plot_report
        out = Path(args.plot)
        many = len(results) > 1
        if many:
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        for r, res in zip(results, analyses):
            dest = out / f"{r['name']}.png" if many else out
            plot_report(res, r, path=dest)
            print(f"wrote {dest}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
