# fatigue-ppg

[![tests](https://github.com/sriramd23/ppg-models/actions/workflows/ci.yml/badge.svg)](https://github.com/sriramd23/ppg-models/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue)](pyproject.toml)
[![paper](https://img.shields.io/badge/paper-10.3390%2Fmath11163580-b31b1b)](https://doi.org/10.3390/math11163580)

Fatigue estimation from the position of the **dicrotic peak** in a
photoplethysmogram — a faithful, tested implementation of

> Chen, Y.-X.; Tseng, C.-K.; Kuo, J.-T.; Wang, C.-J.; Chao, S.-H.; Kau, L.-J.;
> Hwang, Y.-S.; Lin, C.-L. **Fatigue Estimation Using Peak Features from PPG
> Signals.** *Mathematics* **2023**, 11, 3580.
> [doi:10.3390/math11163580](https://doi.org/10.3390/math11163580)

The paper's claim is that where the diastolic peak sits inside a pulse tracks how
tired someone says they are — better than the HRV index NHF does. There is no
neural network anywhere in it: band-pass, peak detection, one ratio per heartbeat,
and a least-squares line. That makes it reproducible exactly, and it means
**everything here runs on a CPU in seconds**.

```
raw PPG ──► band-pass 0.5–8 Hz ──► normalise ──► fiducial points ──► fatigue index ──► Eq. (9) ──► 0–10 state
            §3.3                    Eq. (1)      §3.4               §3.6              §3.7        + rest reminder
```

---

## Install and run, in one go

```bash
git clone https://github.com/sriramd23/ppg-models
cd ppg-models/fatigue-ppg
pip install -r requirements.txt

python -m fatigueppg.selfcheck                              # 19/19 checks
python -m fatigueppg.infer --input examples/demo_ppg_200hz.csv
```

That second command needs no dataset, no download and no configuration — the
example recording is in the repo, and its sampling rate and signal column are
read out of the file.

```
model: paper-eq9   subjective_fatigue_state = 3.1000 + 0.6000 * fatigue_index

  recording            demo_ppg_200hz  (130.0 s at 200 Hz)
  fatigue index        6.3771     (0-10, Section 3.6)
  onset-referenced     8.1886     (same index, zero at the pulse onset)
  predicted subjective 6.93       via subjective_fatigue_state = 3.1000 + 0.6000 * fatigue_index
  heart rate           73.6 bpm
  NHF / NLF            81.0 / 19.0     (Eq. 5, 6; needs > 60 s)
  signal quality       1.00
  cycles               153 analysed, 153 with a diastolic peak (100%)

  ** index above 6 — Take more rest today!!! **
  note: sampling rate 200 Hz inferred from column 'time'
  note: using column 'ppg'
```

`pip install -e .` additionally puts `fatigue-infer`, `fatigue-train`,
`fatigue-extract`, `fatigue-selfcheck` and `fatigue-data` on your PATH. Every
command below also works as `python -m fatigueppg.<name>` without installing.

Full end-to-end demo including training, on synthetic data (no downloads):

```bash
bash scripts/quickstart.sh          # or:  powershell -File scripts/quickstart.ps1
```

---

## Inference — give it any raw PPG signal

```bash
python -m fatigueppg.infer --input recording.csv                 # rate read from the file
python -m fatigueppg.infer --input recording.txt --fs 200        # rate given
python -m fatigueppg.infer --input rec.npy --fs 100 --plot report.png
python -m fatigueppg.infer --input recordings/ --glob "*.csv" --csv results.csv
python -m fatigueppg.infer --demo                                # synthetic, needs nothing
```

**Formats.** `.csv` `.tsv` (any column layout), `.txt` `.dat` (whitespace or one
value per line), `.npy`, `.json` (`[...]` or `{"signal": [...], "fs": 200}`), and
Empatica E4 `BVP.csv` exports.

**The sampling rate** comes from `--fs` if given, otherwise from a time column, an
Empatica header, or a JSON `fs` field. If none of those exist the run stops and
says so rather than guessing — a wrong rate silently rescales every timing
feature in the paper.

**The signal column** is picked by name (`ppg`, `bvp`, `pleth`, `value`, …), else
the last numeric column that is not the clock. Override with `--column`. Whatever
it decides is printed as a note.

**Inverted sensors.** Transmissive PPG points *down*; every rule in the paper
assumes the systolic peak is a maximum. `--invert auto` (the default) scores both
orientations by beat-template correlation and keeps the better one. An inverted
copy of a recording returns a bit-identical index.

**Guard rails.** Under ~5 s it refuses. Fewer than 3 usable cycles, or a signal
quality below 0.70, and it still returns a number but flags it as unreliable —
which is exactly what you want when a wrist recording is 40% motion artefact.

Options: `--model` (a calibration model to apply), `--zero-frac`, `--json`,
`--csv`, `--plot`, `--quiet`. `--plot` writes a one-page report: the waveform
with fiducials, how one cycle's index is read, and the index over time.

### Running it on your own signal

Nothing here is tied to this machine or this folder. Point `--input` at your file
with whatever path your shell uses.

```bash
# Linux / macOS
python -m fatigueppg.infer --input ~/recordings/session01.csv --plot ~/report.png

# Windows PowerShell
python -m fatigueppg.infer --input C:\Users\you\recordings\session01.csv --plot C:\Users\you\report.png

# relative to wherever you are
python -m fatigueppg.infer --input ../data/session01.csv
```

Paths may be absolute or relative. The one requirement is that `python -m
fatigueppg.*` runs from the `fatigue-ppg/` folder, because that is where the
`fatigueppg` package sits. Two ways to stop caring:

```bash
pip install -e .                                   # from fatigue-ppg/, once
fatigue-infer --input /any/path/recording.csv      # now works from any directory
```

```bash
export PYTHONPATH=/path/to/ppg-models/fatigue-ppg  # $env:PYTHONPATH in PowerShell
python -m fatigueppg.infer --input /any/path/recording.csv
```

**What your file needs.** One PPG channel and at least ~5 seconds; 60 s or more if
you want NLF/NHF to mean anything. A time column is convenient but optional — pass
`--fs` instead. Everything else (which column, which orientation, whether the clock
is seconds or milliseconds) is worked out and reported back as notes, so a wrong
guess shows up in the output rather than in the result.

**On a machine with none of your data yet:**

```bash
python -m fatigueppg.infer --demo         # synthetic, needs no files at all
```

### From Python

```python
from fatigueppg import assess, analyse_ppg, load_signal

rec = load_signal("recording.csv")              # or pass fs=200
result, analysis = assess(rec.signal, rec.fs, name=rec.name)

result["fatigue_index"]        # 0-10, Section 3.6
result["subjective_pred"]      # Equation (9)
result["alert"]                # index > 6
analysis.beats                 # one row per cycle: every fiducial point, index, R-R
```

---

## Training — fit Equation (9) on your own people

"Training" here means what it means in the paper: fitting the two coefficients of

```
subjective fatigue state = a + b × fatigue index          (7), (9)
```

by least squares on a labelled cohort. The published `a = 3.1, b = 0.6` came from
sixteen healthy 22–24-year-olds measured with one specific device. **Refit them
before using the predicted state for anything.**

```bash
# 1. build a synthetic cohort (or point step 2 at your own recordings)
python scripts/make_demo_cohort.py --out demo_data

# 2. features: the same pipeline as inference, over the whole cohort
python -m fatigueppg.extract --manifest demo_data/manifest.csv -o demo_data/features.csv

# 3. fit, and write a model file
python -m fatigueppg.train --features demo_data/features.csv \
    --out models/demo_cohort.json --plot demo_data/regression.png

# 4. use it
python -m fatigueppg.infer --model models/demo_cohort.json --input recording.csv
```

Step 3 prints:

```
label: mean of BFI items [2, 3] (q2, q3)
grouped 16 recordings into 16 subjects before fitting

model      demo_cohort
equation   subjective_fatigue_state = 0.3090 + 0.9129 * fatigue_index
in-sample  r = 0.8756  p = 8.82e-06  n = 16  MAE = 0.917
out-of-fold r = 0.8343  MAE = 1.048  (5-fold, n = 16)

predictor comparison (Pearson r against the subjective state):
  predictor                     r          p     n   paper
  fatigue_index           +0.8756   8.82e-06    16   0.907
  nhf_0_10                -0.2722     0.3078    16   0.14875

the paper's Eq. (9) applied unchanged to this cohort: MAE 1.26 points on the 0-10 scale
```

Two things this does that the paper does not, both because sixteen points is very
few: it **groups by subject** before fitting, so one person contributing three
recordings cannot count as three participants, and it reports an **out-of-fold**
correlation next to the in-sample one. The gap between them is the most
informative number in the file.

### The manifest

A CSV with a `path` column, plus whatever you have:

```csv
path,subject,session,fs,q1,q2,q3,q4,q5,q6,q7,q8,q9
recordings/P01.csv,P01,1,200,6,7,8,5,4,2,6,3,5
recordings/P02.csv,P02,1,200,2,3,3,2,1,0,2,1,2
```

Paths are relative to the manifest. Supply `q1..q9` (the BFI-Taiwan answers, 0–10,
in the order of the paper's Table 1) and training averages items 2 and 3 into the
"revised subjective fatigue state", as the paper did. Or supply a single `score`
column, or point `--label-col` at any column you like.

Extraction also works straight off the public corpora:

```bash
python -m fatigueppg.extract --dataset ppgbp      -o ppgbp_features.csv
python -m fatigueppg.extract --dataset fatigueset -o fset_features.csv --min-sqi 0.7
```

---

## Repository layout

```
fatigueppg/
  config.py        every constant, each traceable to a section of the paper
  preprocess.py    §3.3   band-pass 0.5-8 Hz, Equation (1)
  peaks.py         §3.4   cycle search (Eq. 2, 3), onsets, dicrotic wave (Eq. 4)
  fatigue.py       §3.6   the fatigue index
  hrv.py           §3.5   NLF and NHF (Eq. 5, 6)
  analysis.py             one recording end to end; sliding-window version
  quality.py              beat-template SQI, inversion detection  (not in the paper)
  stats.py         §3.7-8 Pearson r (Eq. 8), least squares (Eq. 7), k-fold
  model.py                the fitted line: save / load / apply
  signals.py              read "any raw PPG" off disk
  datasets.py             PPG-BP, FatigueSet, PPG-DaLiA
  synth.py                synthetic PPG with known ground truth
  plotting.py             the paper's figures, and the inference report
  selfcheck.py            19 checks against ground truth
  infer.py         §3.9   the evaluation system: signal in, index + reminder out
  extract.py              cohort -> feature table
  train.py         §3.7   feature table -> fitted Equation (9)
scripts/
  make_demo_cohort.py     16 synthetic participants with BFI-style answers
  quickstart.sh / .ps1    the whole path, start to finish
models/paper_eq9.json     the published coefficients, used by default
examples/                 one 130 s recording, so inference runs on a fresh clone
tests/                    pytest; the self-check is the substantive one
```

### Where each equation lives

| Paper | Code |
|---|---|
| §3.3 band-pass, Eq. (1) normalisation | `preprocess.bandpass`, `preprocess.normalize_paper` |
| §3.4.1 Eq. (2)/(3) cycle search → systolic peaks | `peaks.estimate_cycle`, `peaks.find_systolic_peaks` |
| §3.4.2 pulse onset | `peaks.find_onsets` |
| §3.4.3 Eq. (4) dicrotic notch + diastolic peak | `peaks.find_dicrotic` |
| §3.5 Eq. (5)/(6) NLF, NHF | `hrv.rr_series`, `hrv.hrv_indices` |
| §3.6 fatigue index | `fatigue.cycle_fatigue_index` |
| §3.7 Eq. (7), §3.8 Eq. (8) | `stats.linreg`, `stats.pearson` |
| §3.9 / §4.4 evaluation system, Eq. (9), threshold 6 | `infer.assess`, `model.FatigueModel` |

---

## Datasets

**The paper's own data is not public** — sixteen participants measured with a
COMGO device plus BFI-Taiwan questionnaires, available "from the corresponding
author on reasonable request". Nothing here can reproduce its numbers. These three
public corpora are what the implementation was validated against:

| Corpus | What it gives | Size | How |
|---|---|---|---|
| **PPG-BP** | 657 clean seated fingertip segments, 219 subjects, clinical records. No fatigue labels. | 1.5 MB | downloads itself |
| **FatigueSet** | 12 participants × 3 sessions of wrist BVP **with fatigue self-reports** | ~2 GB → 50 MB | manual |
| **PPG-DaLiA** | 15 subjects × ~2.5 h of wrist BVP, daily-life protocol. No fatigue labels. | ~40 MB | manual |

```bash
python -m fatigueppg.fetch --status              # what is visible right now
python -m fatigueppg.fetch --download ppgbp      # 1.5 MB from figshare
```

**FatigueSet** — download from <https://www.esense.io/datasets/fatigueset/>, then
keep only `<participant>/<session>/wrist_bvp.csv` plus the survey/metadata files
(that is 2 GB → ~50 MB) and set `FATIGUESET_ROOT=/path/to/fatigueset`. Its survey
schema is not something this package hard-codes; `python -m fatigueppg.fetch
--tables <dir>` lists every table and every fatigue-like column it can see so you
can map them yourself.

**PPG-DaLiA** — download from
<https://archive.ics.uci.edu/dataset/495/ppg+dalia>, keep only the `S*/S*_E4.zip`
archives (the wrist BVP lives there; the 1.4 GB `SX.pkl` holds ECG-synchronised
heart-rate labels that are not needed), and set `DALIA_ROOT`.

Auto-detection searches `/kaggle/input`, `/kaggle/working`, `/content`, `./data`
and `.` three levels deep for a marker path, so a dataset in a normal place needs
no configuration at all.

---

## What the reproduction actually found

Measured, not asserted — reproduce them with `python -m fatigueppg.extract
--dataset ppgbp -o f.csv`.

**The peak detector is sound.** Against synthetic ground truth it recovers 145/145
beats with a median error of 0 samples, and locates diastolic peaks to within
0 ms. On the 657 real fingertip segments of PPG-BP it agrees with a conventional
constrained peak finder on the exact same sample for **94.8%** of peaks.

**The index as published does not transfer.** Section 3.6 puts the index's zero at
half the pulse height, so any diastolic peak below mid-pulse clips to 0. On PPG-BP
the median diastolic peak sits at **0.33** of the pulse height — only 18% of
cycles clear the zero point — so the index reads 0 for most subjects and carries
almost no information there. That mid-height zero is a calibration to the paper's
cohort of sixteen healthy 22–24-year-olds, whose reflected wave is strong. Every
output therefore also reports `fatigue_index_onset`: the same physiological
quantity (relative dicrotic-peak height) with the zero at the pulse onset and no
floor. Use that one on any cohort that is not the paper's.

**Nothing public tests the fatigue claim properly.** FatigueSet has the
self-reports but wrist BVP under exercise; PPG-BP has the clean signal but no
reports. Equation (9)'s coefficients are taken on trust — which is why
`fatigueppg.train` exists.

### Deviations from the printed method

Each is documented at its point of use in the code.

| Where | Paper | Here | Why |
|---|---|---|---|
| §3.4.1 step 3 | "the calculation cycle" | the accepted median R-R interval | as the search block it would slice 2 min into thousands of pieces; §3.4 defines a cycle as one heartbeat |
| §3.4.1 step 2 | temple peaks as found | merged below 0.3 s apart, with a CV bound on the period | otherwise a small block locks onto systolic + dicrotic peaks and accepts half the true period immediately |
| §3.4.1 step 3 | highest peak per cycle | plus a re-search of gaps > 1.5 cycles | fixed blocks lose beats that straddle a boundary: 141/145 → 145/145 |
| §3.4.3 | strategy 2 when no first derivative is positive | strategy 1 must also clear a prominence bar | with real noise some derivative is always positive, so strategy 2 would never run |
| §3.4.3 | max/min of the second derivative | most prominent *local* extrema | on a decaying limb the global curvature maximum is the systolic peak itself |
| §3.5 | FFT of the resampled R-R series | mean removed first, ms units | otherwise DC dominates VLF; ms² is the conventional unit |
| — | — | beat-template SQI, inversion detection | needed the moment you leave clean seated fingertip PPG |

### Known behaviours

- The first and last cycle of a recording sit inside the zero-phase filter's edge
  transient, so their per-cycle index can be an outlier. On a 2-minute recording
  that is 1 cycle in 150 and does not move the mean; on a 15 s recording it does.
  `--plot` shows the per-cycle series, so you can see it.
- NHF and NLF need more than 60 s of beats and return `nan` below that. The
  fatigue index does not — it is a per-cycle quantity.
- `--zero-frac 0.0` changes the index's meaning. A model file records the
  `zero_frac` it was fitted with and inference follows it, so the two cannot drift
  apart by accident.

---

## Tests

```bash
pip install pytest
pytest
```

Twenty tests: the 19-point self-check (every stage against synthetic ground
truth), the file-format round trips, the model round trip, and the CLIs as a user
meets them. `python -m fatigueppg.selfcheck` on its own is the fast gate — run it
after any change.

---

## Limitations

This is a reproduction of a research method, not a medical device and not a
diagnosis. The index is derived from a single feature of pulse shape; it moves
with vascular tone, posture, temperature and sensor pressure as well as with
whatever fatigue is. The published Equation (9) was fitted on sixteen people of
one age range on one device. Treat any absolute number it produces as a research
output.

## Licence

To be added by the group. Until a LICENSE file lands, no usage rights are
granted by default — treat the code as all-rights-reserved.

## Attribution

The method reproduced here is published work, not ours:

```bibtex
@article{chen2023fatigue,
  title   = {Fatigue Estimation Using Peak Features from PPG Signals},
  author  = {Chen, Yi-Xiang and Tseng, Chin-Kun and Kuo, Jung-Tsung and
             Wang, Chien-Jen and Chao, Shu-Hung and Kau, Lih-Jen and
             Hwang, Yuh-Shyan and Lin, Chun-Ling},
  journal = {Mathematics},
  volume  = {11}, number = {16}, pages = {3580}, year = {2023},
  doi     = {10.3390/math11163580}
}
```

Dataset papers: PPG-BP — Liang et al., *Sci. Data* **2018**, 5, 180020.
FatigueSet — Kalanadhabhatta et al., *PervasiveHealth* **2021**.
PPG-DaLiA — Reiss et al., *Sensors* **2019**, 19, 3079.

The datasets are distributed by their own authors under their own terms; get
them from the original sources.
