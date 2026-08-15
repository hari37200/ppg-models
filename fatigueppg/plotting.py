"""Figures: the paper's, and a one-page inference report.

Colours are a fixed categorical order, never cycled, and every fiducial marker
also differs in shape -- identity is never carried by colour alone.
"""
from __future__ import annotations

import numpy as np

from .config import ALERT_THRESHOLD, FI_MAX

__all__ = ["PALETTE", "FID", "apply_style", "plot_fiducials",
           "plot_index_definition", "plot_report", "plot_preprocessing"]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRIDC, SIGNAL_C = "#1c1c1c", "#5a5a5a", "#dcdcdc", "#4a4a4a"

#: fiducial -> (colour, marker, label)
FID = {
    "onset": (PALETTE[0], "v", "pulse onset"),
    "systolic": (PALETTE[1], "*", "systolic peak"),
    "notch": (PALETTE[2], "o", "dicrotic notch"),
    "diastolic": (PALETTE[7], "D", "diastolic peak"),
}


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
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    return plt


def plot_preprocessing(raw, fs, ax=None):
    """The paper's Figure 2: raw -> band-passed -> Equation (1)."""
    from .preprocess import bandpass, normalize_paper

    plt = apply_style()
    filt = bandpass(raw, fs)
    edge = int(min(1.0 * fs, max(filt.size // 10, 1)))
    raw_v, filt_v = np.asarray(raw)[edge:-edge], filt[edge:-edge]
    norm = normalize_paper(filt_v)
    tt = np.arange(raw_v.size) / fs

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for a, y, ttl, ylab in [
            (axes[0], raw_v, "(a) raw PPG", "amplitude"),
            (axes[1], filt_v, "(b) after the 0.5-8 Hz band-pass", "amplitude"),
            (axes[2], norm, "(c) after Eq. (1)", "normalised amplitude")]:
        a.plot(tt, y, color=SIGNAL_C, linewidth=0.9)
        a.set_title(ttl)
        a.set_ylabel(ylab)
    axes[2].set_xlabel("time (s)")
    axes[2].axhline(0, color=GRIDC, linewidth=0.8)
    fig.suptitle("PPG preprocessing (Section 3.3)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_fiducials(res, t0=0.0, t1=None, ax=None, title=None, legend=True):
    """The four fiducial points over a slice of one recording (Figures 5, 8)."""
    plt = apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    fs, x = res.fs, res.signal
    t1 = t1 if t1 is not None else min(t0 + 5.0, x.size / fs)
    i0, i1 = int(t0 * fs), min(int(t1 * fs), x.size)
    ax.plot(np.arange(i0, i1) / fs, x[i0:i1], color=SIGNAL_C, linewidth=1.2, zorder=1)

    sel = res.beats[(res.beats["systolic"] >= i0) & (res.beats["systolic"] < i1)]
    for col, (c, m, lab) in FID.items():
        idx = sel[col].to_numpy() if len(sel) else np.array([], dtype=int)
        idx = idx[(idx > 0) & (idx < x.size)]
        if idx.size:
            ax.scatter(idx / fs, x[idx], s=70 if m == "*" else 26, marker=m,
                       color=c, edgecolors="white", linewidths=0.6, label=lab,
                       zorder=3)
    ax.margins(y=0.22)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised amplitude")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
                  fontsize=8.5)
    return ax


def plot_index_definition(res, ax=None):
    """The paper's Figure 6: one cycle, read off the 0-10 scale."""
    plt = apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.0))
    beats = res.beats[res.beats["diastolic"] > 0].dropna(subset=["fi"])
    if not len(beats):
        ax.text(0.5, 0.5, "no cycle with a detectable diastolic peak",
                ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return ax

    b = beats.iloc[int(np.argsort(beats["fi"].to_numpy())[len(beats) // 2])]
    fs, x = res.fs, res.signal
    o, sy, no, d = (int(b["onset"]), int(b["systolic"]), int(b["notch"]),
                    int(b["diastolic"]))
    i0 = max(o - int(0.1 * res.cycle), 0)
    i1 = min(o + int(1.15 * res.cycle), x.size)
    tt = np.arange(i0, i1) / fs
    ax.plot(tt, x[i0:i1], color=SIGNAL_C, linewidth=1.5, zorder=1)

    base, top = float(x[o]), float(x[sy])
    zero = base + 0.5 * (top - base)
    ax.annotate("", xy=(sy / fs, top), xytext=(sy / fs, base),
                arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.1))
    ax.text(sy / fs - 0.03, 0.5 * (base + top), "x", color=MUTED, ha="right",
            va="center", style="italic")
    ax.axhline(zero, color=GRIDC, linewidth=1.0, linestyle="--", zorder=0)

    right = tt[-1]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = zero + frac * (top - zero)
        ax.plot([right - 0.04, right], [yv, yv], color=MUTED, linewidth=1.0,
                clip_on=False)
        ax.text(right + 0.01, yv, f"{frac*FI_MAX:.1f}", color=MUTED,
                va="center", fontsize=8)

    for key, idx in (("onset", o), ("systolic", sy), ("notch", no), ("diastolic", d)):
        c, m, lab = FID[key]
        if idx > 0:
            ax.scatter([idx / fs], [x[idx]], s=80 if m == "*" else 34, marker=m,
                       color=c, edgecolors="white", linewidths=0.7, label=lab,
                       zorder=3)
    ax.annotate(f"fatigue index {b['fi']:.2f}", xy=(d / fs, x[d]),
                xytext=(d / fs - 0.24, x[d] + 0.42), color=PALETTE[7],
                fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE[7], linewidth=1.2))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalised amplitude")
    ax.margins(y=0.22, x=0.06)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4, fontsize=8.5)
    return ax


def plot_report(res, result, path=None):
    """One-page inference report: waveform, definition, per-cycle index."""
    plt = apply_style()
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.22)

    span = min(5.0, res.duration)
    start = max((res.duration - span) / 2, 0.0)
    ax0 = fig.add_subplot(gs[0, :])
    plot_fiducials(res, start, start + span, ax=ax0,
                   title=f"{result['name']} -- 5 s of the recording")

    ax1 = fig.add_subplot(gs[1, 0])
    plot_index_definition(res, ax=ax1)
    ax1.set_title("how one cycle's index is read")

    ax2 = fig.add_subplot(gs[1, 1])
    beats = res.beats.dropna(subset=["fi"])
    if len(beats):
        t = beats["systolic"].to_numpy() / res.fs
        ax2.plot(t, beats["fi"], marker="o", markersize=3, color=PALETTE[0],
                 label="paper index")
        ax2.plot(t, beats["fi_onset"], marker="o", markersize=3, color=PALETTE[1],
                 label="onset-referenced")
        ax2.axhline(result["threshold"], color=PALETTE[7], linewidth=1.2,
                    linestyle="--", label=f"reminder threshold ({result['threshold']:g})")
        ax2.set_ylim(-0.4, FI_MAX + 0.4)
        ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("index (0-10)")
    ax2.set_title(f"index per cycle -- mean {result['fatigue_index']:.2f}")

    fig.suptitle(
        f"fatigue index {result['fatigue_index']:.2f}   ->   predicted subjective "
        f"state {result['subjective_pred']:.2f}"
        + ("   [take more rest today]" if result["alert"] else ""),
        x=0.02, ha="left", fontsize=13, fontweight="bold",
        color=PALETTE[7] if result["alert"] else INK)
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
