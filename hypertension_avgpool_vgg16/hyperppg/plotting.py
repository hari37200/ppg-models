"""A one-page report figure for :mod:`hyperppg.predict`.

Colours are a fixed order, never cycled, and the class scale runs cool-to-warm
in clinical order so the reader can rank the four stages without consulting a
legend. Nothing here is decorative: the panels show the waveform the model was
given, the verdict with its uncertainty, and -- for a long recording -- whether
that verdict held across the recording or only in places.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_style", "CLASS_COLOURS", "plot_report"]

INK, MUTED, GRIDC, SIGNAL_C = "#1c1c1c", "#5a5a5a", "#dcdcdc", "#4a4a4a"
PEAK_C = "#eb6834"

#: Clinical order, cool -> warm. Index matches config.CLASS_NAMES.
CLASS_COLOURS = ["#1baf7a", "#eda100", "#e8743b", "#d1344b"]


def apply_style():
    """Recessive grid, no top/right spines, fixed categorical cycle."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.edgecolor": GRIDC, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRIDC, "grid.linewidth": 0.6, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "font.size": 9, "lines.linewidth": 1.4,
    })
    return plt


def _representative(proba, verdict_idx):
    """Index of a window that speaks for the verdict, not the best one.

    The most confident window would flatter the model. This takes the one whose
    confidence in the winning class is the median among the windows that agree
    with the verdict, so the panel shows a typical beat rather than a trophy.
    """
    agree = np.flatnonzero(proba.argmax(axis=1) == verdict_idx)
    if agree.size == 0:
        return int(proba[:, verdict_idx].argmax())
    conf = proba[agree, verdict_idx]
    return int(agree[np.argsort(conf)[conf.size // 2]])


def _plot_window(ax, x, fs, title):
    """One conditioned 2.1 s window with its systolic peaks marked."""
    from hyperppg.data.features import _find_systolic_peaks

    t = np.arange(x.size) / fs
    ax.plot(t, x, color=SIGNAL_C)
    peaks = _find_systolic_peaks(x, fs)
    if peaks.size:
        ax.plot(t[peaks], x[peaks], linestyle="none", marker="*", markersize=9,
                color=PEAK_C, label=f"systolic peak ({peaks.size})")
        ax.legend(loc="upper right", ncol=1)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("conditioned amplitude")


def _plot_probabilities(ax, classes, proba, verdict_idx):
    order = np.argsort(proba)[::-1]
    labels = [classes[i] for i in order]
    values = [proba[i] for i in order]
    colours = [CLASS_COLOURS[i] for i in order]
    alphas = [1.0 if i == verdict_idx else 0.45 for i in order]

    y = np.arange(len(labels))[::-1]
    for yi, v, c, a in zip(y, values, colours, alphas):
        ax.barh(yi, v, color=c, alpha=a, height=0.62)
        ax.text(v + 0.015, yi, f"{v:.1%}", va="center", fontsize=9,
                color=INK if a == 1.0 else MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    ax.set_title("class probabilities")


def _plot_window_timeline(ax, classes, proba, win_s, verdict_idx):
    """Per-window verdicts over the recording: does the call actually hold?"""
    pred = proba.argmax(axis=1)
    t = np.arange(pred.size) * win_s
    for k in range(len(classes)):
        hit = pred == k
        if hit.any():
            ax.plot(t[hit], np.full(hit.sum(), k), linestyle="none", marker="o",
                    markersize=3.2, color=CLASS_COLOURS[k], alpha=0.75)
    agree = int((pred == verdict_idx).sum())
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([c.replace(" hypertension", "") for c in classes])
    ax.set_ylim(-0.6, len(classes) - 0.4)
    ax.set_xlabel("time (s)")
    ax.set_title(f"per-window verdict -- {agree}/{pred.size} agree")


def _plot_beat_overlay(ax, x, fs):
    """Every detected beat aligned on its systolic peak, plus the mean pulse."""
    from hyperppg.data.features import _find_systolic_peaks

    peaks = _find_systolic_peaks(x, fs)
    pre, post = int(0.25 * fs), int(0.65 * fs)
    beats = [x[p - pre:p + post] for p in peaks
             if p - pre >= 0 and p + post <= x.size]

    if not beats:
        ax.text(0.5, 0.5, "no complete beat to overlay", ha="center",
                va="center", transform=ax.transAxes, color=MUTED)
        ax.set_title("beat morphology")
        return

    stack = np.vstack(beats)
    tt = (np.arange(-pre, post)) / fs * 1000.0
    for b in stack:
        ax.plot(tt, b, color=SIGNAL_C, alpha=0.28, linewidth=0.8)
    ax.plot(tt, stack.mean(axis=0), color=PEAK_C, linewidth=2.0,
            label=f"mean of {len(beats)} beats")
    ax.axvline(0, color=GRIDC, linewidth=1.0)
    ax.legend(loc="upper right")
    ax.set_xlabel("time from systolic peak (ms)")
    ax.set_title("beat morphology -- what the features measure")


def plot_report(result, detail, path=None):
    """Write the one-page report for a :func:`hyperppg.predict.predict` call.

    ``detail`` is the second element returned by ``predict(..., detail=True)``:
    the conditioned windows, the per-window probabilities and the model's
    sampling rate.
    """
    plt = apply_style()

    classes = detail["class_names"]
    proba = np.asarray(detail["proba"], dtype=float)
    windows = np.asarray(detail["windows"], dtype=float)
    fs = float(detail["fs"])
    win_s = windows.shape[1] / fs

    mean_proba = np.asarray([result["probabilities"][c] for c in classes])
    verdict_idx = int(np.argmax(mean_proba))
    verdict = classes[verdict_idx]

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42,
                          wspace=0.28, left=0.09, right=0.97,
                          top=0.86, bottom=0.09)

    # A window that agrees with the verdict, and is typical among those.
    rep = _representative(proba, verdict_idx)
    label = result.get("name", "recording")
    where = f"window {rep + 1} of {proba.shape[0]}" if proba.shape[0] > 1 \
        else f"the {win_s:.1f} s segment"
    _plot_window(fig.add_subplot(gs[0, :]), windows[rep], fs,
                 f"{label} -- {where}")

    _plot_probabilities(fig.add_subplot(gs[1, 0]), classes, mean_proba,
                        verdict_idx)

    ax = fig.add_subplot(gs[1, 1])
    if proba.shape[0] > 1:
        _plot_window_timeline(ax, classes, proba, win_s, verdict_idx)
    else:
        _plot_beat_overlay(ax, windows[rep], fs)

    head = f"predicted stage: {verdict}   ({result['confidence']:.1%} confidence)"
    fig.suptitle(head, x=0.09, ha="left", fontsize=15, fontweight="bold",
                 color=CLASS_COLOURS[verdict_idx], y=0.975)

    cv = result.get("model_cv") or {}
    sub = []
    if result.get("ground_truth"):
        hit = "correct" if result["ground_truth"] == verdict else "WRONG"
        sub.append(f"ground truth: {result['ground_truth']} ({hit})")
    if cv.get("accuracy"):
        sub.append(f"model is {cv['accuracy']:.1%} accurate subject-wise "
                   f"(baseline 38.8%) -- a screening prior, not a diagnosis")
    if result.get("out_of_fold"):
        sub.append("scored out-of-fold: this segment was in the training set")
    if sub:
        fig.text(0.09, 0.925, "   |   ".join(sub), fontsize=9, color=MUTED)

    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig
