"""End-to-end tests. Run with: pytest

The self-check is the substantive one -- it validates every stage of the paper's
method against synthetic ground truth. The rest guard the plumbing that the
self-check does not touch: file formats, the model round-trip, and the CLIs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fatigueppg import analyse_ppg, load_signal, paper_model, synth_ppg
from fatigueppg.infer import assess
from fatigueppg.model import FatigueModel
from fatigueppg.selfcheck import selfcheck
from fatigueppg.stats import bfi_score, linreg

REPO = Path(__file__).resolve().parent.parent
FS = 200.0


# --------------------------------------------------------------------------
# the paper's method
# --------------------------------------------------------------------------

def test_selfcheck_all_pass():
    checks = selfcheck(verbose=False)
    failed = [name for name, ok, _ in checks if not ok]
    assert not failed, f"failed: {failed}"


def test_index_tracks_dicrotic_height():
    idx = [analyse_ppg(synth_ppg(60, FS, dicrotic=d, seed=2)[0], FS).fatigue_index
           for d in (0.60, 0.70, 0.80, 0.90)]
    assert all(np.diff(idx) > 0), idx
    assert all(0 <= v <= 10 for v in idx)


def test_index_floors_below_mid_height():
    """The paper's zero point is half the pulse height, so low peaks read 0."""
    low = analyse_ppg(synth_ppg(60, FS, dicrotic=0.35, seed=2)[0], FS)
    assert low.fatigue_index == pytest.approx(0.0, abs=1e-6)
    assert low.fatigue_index_onset > 2.0        # the onset-referenced variant does not


def test_equation_9():
    m = paper_model()
    assert m.predict(6.5) == pytest.approx(7.0)
    assert m.alert(6.5) and not m.alert(5.5)


def test_bfi_score_uses_items_2_and_3():
    assert bfi_score([0, 8, 6, 0, 0, 0, 0, 0, 0]) == pytest.approx(7.0)
    with pytest.raises(ValueError):
        bfi_score([1, 2, 3])


# --------------------------------------------------------------------------
# inference on arbitrary input
# --------------------------------------------------------------------------

def test_assess_returns_expected_keys():
    sig, _ = synth_ppg(120, FS, dicrotic=0.82, seed=3)
    result, res = assess(sig, FS, name="t", invert="no")
    for key in ("fatigue_index", "subjective_pred", "alert", "hr", "sqi",
                "n_cycles", "detection_rate"):
        assert key in result
    assert 0 <= result["fatigue_index"] <= 10
    assert res.n_valid > 100


def test_assess_rejects_a_too_short_recording():
    sig, _ = synth_ppg(2.0, FS, seed=0)
    with pytest.raises(ValueError, match="too short"):
        assess(sig, FS, invert="no")


def test_inverted_signal_gives_the_same_index():
    sig, _ = synth_ppg(60, FS, dicrotic=0.82, seed=5)
    upright, _ = assess(sig, FS, invert="auto")
    flipped, _ = assess(-sig, FS, invert="auto")
    assert flipped["fatigue_index"] == pytest.approx(upright["fatigue_index"], abs=1e-9)


@pytest.mark.parametrize("suffix", [".txt", ".npy", ".json", ".csv"])
def test_load_signal_round_trip(tmp_path, suffix):
    sig, _ = synth_ppg(30, FS, dicrotic=0.8, seed=7)
    path = tmp_path / f"rec{suffix}"
    if suffix == ".txt":
        np.savetxt(path, sig)
        rec = load_signal(path, fs=FS)
    elif suffix == ".npy":
        np.save(path, sig)
        rec = load_signal(path, fs=FS)
    elif suffix == ".json":
        path.write_text(json.dumps({"fs": FS, "signal": sig.tolist()}))
        rec = load_signal(path)                    # fs comes from the file
    else:
        pd.DataFrame({"time": np.arange(sig.size) / FS,
                      "ppg": sig}).to_csv(path, index=False)
        rec = load_signal(path)                    # fs inferred from the clock
    assert rec.fs == pytest.approx(FS, rel=1e-6)
    assert rec.signal.size == sig.size
    assert np.allclose(rec.signal, sig, atol=1e-4)


def test_load_signal_without_a_rate_is_an_error(tmp_path):
    path = tmp_path / "bare.txt"
    np.savetxt(path, np.arange(100.0))
    with pytest.raises(ValueError, match="sampling rate unknown"):
        load_signal(path)


def test_empatica_header_is_read(tmp_path):
    sig, _ = synth_ppg(30, 64.0, dicrotic=0.8, seed=8)
    path = tmp_path / "BVP.csv"
    path.write_text("1500000000.00\n64.000000\n"
                    + "\n".join(f"{v:.4f}" for v in sig))
    rec = load_signal(path)
    assert rec.fs == pytest.approx(64.0)
    assert rec.signal.size == sig.size


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def test_model_round_trip(tmp_path):
    m = FatigueModel(a=1.5, b=0.8, feature="fatigue_index", name="t")
    path = m.save(tmp_path / "m.json")
    back = FatigueModel.load(path)
    assert (back.a, back.b, back.name) == (1.5, 0.8, "t")
    assert back.predict(2.0) == pytest.approx(3.1)


def test_fit_recovers_known_coefficients():
    x = np.linspace(2, 9, 30)
    m = FatigueModel.fit(x, 3.1 + 0.6 * x, name="exact")
    assert m.a == pytest.approx(3.1)
    assert m.b == pytest.approx(0.6)
    assert m.metrics["r"] == pytest.approx(1.0)


def test_linreg_reports_out_of_fold_worse_than_in_sample():
    rng = np.random.default_rng(0)
    x = rng.uniform(2, 9, 40)
    y = 3.1 + 0.6 * x + rng.normal(0, 1.5, 40)
    m = FatigueModel.fit(x, y, name="noisy")
    assert m.metrics["r_oof"] <= m.metrics["r"] + 1e-9


# --------------------------------------------------------------------------
# the command line, as a user meets it
# --------------------------------------------------------------------------

def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=REPO,
                          capture_output=True, text=True)


def test_cli_selfcheck():
    out = _run("fatigueppg.selfcheck")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "checks passed" in out.stdout


def test_cli_infer_demo():
    out = _run("fatigueppg.infer", "--demo")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "fatigue index" in out.stdout


def test_cli_infer_bundled_example():
    example = REPO / "examples" / "demo_ppg_200hz.csv"
    assert example.is_file(), "the bundled example recording is missing"
    out = _run("fatigueppg.infer", "--input", str(example))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "fatigue index" in out.stdout
    # the rate and the column are both inferred from the file itself
    assert "200 Hz" in out.stdout


def test_rounded_time_column_does_not_fool_the_rate(tmp_path):
    """PhysioNet's BIDMC CSVs round the clock to two decimals, so a 125 Hz
    record stores 0.01 where the step is really 0.008 and every sixth
    timestamp repeats. The median step then reads 100 Hz -- a 25% error on
    every timing feature in the paper. The total span survives the rounding."""
    fs, n = 125.0, 2000
    t = np.round(np.arange(n) / fs, 2)          # what BIDMC actually writes
    assert np.median(np.diff(t)) == pytest.approx(0.01)   # the trap
    csv = tmp_path / "rounded.csv"
    pd.DataFrame({"time": t, "ppg": synth_ppg(n / fs, fs, hr=70)[0]}).to_csv(csv, index=False)

    rec = load_signal(csv)
    # not exact: the last timestamp is rounded too, so the span is off by ~0.01%
    assert rec.fs == pytest.approx(fs, rel=1e-3)
    assert any("repeated timestamps" in note for note in rec.notes)


def test_a_genuine_gap_still_uses_the_median_step(tmp_path):
    """The span shortcut must not fire on a recording with a real dropout:
    there are no repeated timestamps there, and the median step is honest."""
    fs = 100.0
    t = np.concatenate([np.arange(500) / fs, 30.0 + np.arange(500) / fs])
    csv = tmp_path / "gap.csv"
    pd.DataFrame({"time": t, "ppg": np.random.default_rng(0).normal(size=t.size)}
                 ).to_csv(csv, index=False)
    assert load_signal(csv).fs == pytest.approx(fs, rel=1e-6)
