# AvgPool_VGG-16 for PPG Hypertension Staging — replication and improvement

Replication of:

> G. Frederick, Yaswant T, Brintha Therese A.
> **"PPG Signals for Hypertension Diagnosis: A Novel Method Using Deep Learning Models."**
> arXiv:2304.06952 (2023).

…followed by a substantially stronger pipeline built on the same data.

**Task.** Given a 2.1-second fingertip PPG segment, classify the subject into one
of four hypertension stages: Normal, Prehypertension, Stage 1, Stage 2.

---

## Why this paper

Of the three PDFs in `E:\ppg`, this is the only one that is both *easy to
implement* and *reproducible from public data*:

| Paper | Model | Data | Verdict |
|---|---|---|---|
| **Hypertension (this one)** | VGG-16 with avg-pool | PPG-BP, public, **1.5 MB** | ✅ chosen |
| Fatigue Estimation | linear regression on a dicrotic-peak index | 16 in-house COMGO recordings | ✗ private data; a correlation study with no accuracy metric |
| Sleep Apnea (TCN-LSTM) | TCN + Bi-LSTM | 315 private subjects + Physionet | ✗ private data |

The proposed architecture is a one-line change to a standard backbone (swap
every `MaxPool2d` for `AvgPool2d`), which makes it genuinely easy to replicate
faithfully.

---

## The headline finding: the paper's 80% is inflated

The paper reports 80% accuracy for AvgPool_VGG-16. Two facts, both verified
directly against the released dataset:

1. **The subject-level class counts are exactly `80 / 85 / 34 / 20` = 219
   subjects.** The paper's Table 1 reports test-set support of exactly
   `nt 80 / pt 85 / ht1 34 / ht2 20` = 219.
2. The paper states: *"Using data augmentation, a test dataset was created by
   adding and removing noise to the PPG signals and is used for validation."*

So the evaluation set is **noise-perturbed copies of the same subjects used for
training**. PPG-BP gives 3 segments per subject; any split that is not grouped
by `subject_id` puts the same person on both sides.

This repo measures the cost of that directly. Same model, same features, same
data — only the split changes:

| Split scheme | Accuracy | Macro-F1 |
|---|---|---|
| `segment` (the paper's setup — leaks) | **0.560** | 0.510 |
| `subject` (grouped, honest) | **0.487** | 0.417 |

**A 7.3-point inflation from the split alone.** Everything in this repo defaults
to the subject-wise protocol, and every script prints an explicit leakage audit
before training:

```
fold   train    val  trn subj  val subj  shared
   0     525    132       218       111     110
-> SUBJECT LEAKAGE: 541 shared subject-folds
```

A useful reference point: the majority-class baseline is **0.388 accuracy**.

---

## Layout

```
hypertension_avgpool_vgg16/
├── GUIDE.md                    ← run instructions for Colab + Kaggle (start here)
├── RESULTS.md                  ← measured numbers, per-class breakdowns, negative results
├── README.md
├── requirements.txt
├── notebooks/quickstart.ipynb  ← ready-to-run Colab/Kaggle notebook
├── scripts/make_pretrain_bundle.ps1   3.5 GB of local data → a 50 MB upload
└── hyperppg/
    ├── config.py               dataset constants + Colab/Kaggle path resolution
    ├── selfcheck.py            14-point pipeline validation — run this first
    ├── metrics.py              accuracy / balanced acc / macro-F1 / kappa
    ├── datasets.py             torch Datasets (image, 1-D, SSL windows)
    ├── engine.py               train loop: warmup+cosine, mixup, early stopping
    ├── runner.py               cross-validation driver, pooled out-of-fold report
    ├── data/
    │   ├── download.py         fetch PPG-BP from figshare (1.5 MB)
    │   ├── ppgbp.py            index building, label join, signal loading
    │   ├── preprocess.py       paper pipeline + a proper PPG chain
    │   ├── render.py           fast numpy waveform→image rasteriser
    │   ├── augment.py          10 physiologically plausible augmentations
    │   ├── features.py         63 morphology features (fiducial, APG, spectral)
    │   ├── splits.py           subject-wise vs segment-wise CV + leakage audit
    │   └── corpora.py          PPG-DaLiA + FatigueSet unlabelled BVP loaders
    ├── models/
    │   ├── paper.py            AlexNet / ResNet-50 / VGG-16 / AvgPool_VGG-16
    │   └── hybrid.py           1-D CNN+transformer encoder, masked autoencoder
    ├── train_paper.py          ← replication
    ├── features_baseline.py    ← classical baseline (fast, CPU, run this early)
    ├── pretrain_ssl.py         ← self-supervised pretraining on the local data
    ├── train_hybrid.py         ← improved model
    ├── fit_model.py            ← fit once and persist, for deployment
    ├── predict.py              ← score one recording
    └── plotting.py             one-page inference report
```

---

## Using the datasets already in `E:\ppg`

PPG-BP has only 657 labelled segments — far too few to train a transformer from
scratch. The two datasets sitting next to this project are unlabelled for our
purposes but far larger, and both carry Empatica E4 wrist PPG:

| Dataset | Content | Used as |
|---|---|---|
| **PPG-DaLiA** | 15 subjects × ~2.5 h | unlabelled SSL corpus |
| **FatigueSet** | 12 participants × 3 sessions | unlabelled SSL corpus |

Together: **49.8 hours** of wrist PPG, ~51 recordings.

One practical detail that makes this feasible: the wrist BVP is read from
`SX/SX_E4.zip → BVP.csv` (**~2.6 MB zipped per subject**) rather than from
`SX.pkl` (**~1.4 GB per subject**). We only need the unlabelled waveform, not
the ECG-synchronised heart-rate labels — a ~500× saving that keeps this inside a
Colab/Kaggle disk quota.

`pretrain_ssl.py` trains a masked-span autoencoder on these windows, then
`train_hybrid.py --ssl-checkpoint` fine-tunes that encoder on PPG-BP.

> **Caveat, stated plainly:** DaLiA and FatigueSet are *wrist* PPG under
> daily-life motion; PPG-BP is a clean seated *fingertip* recording. Pulse
> morphology transfers, but this is not a matched-site corpus, so expect a
> modest gain rather than a dramatic one.

---

## What was changed, and why

| # | Change | Rationale |
|---|---|---|
| 1 | **Subject-wise CV** | The single largest correction. See above. |
| 2 | **1-D signal instead of a picture of it** | Rendering 2100 samples into 224 columns is a ~9× decimation, then a 134 M-parameter ImageNet stem must rediscover that the image is a time series. The 1-D hybrid is 2.2 M parameters — **60× smaller**. |
| 3 | **Class-weighted loss + macro-F1 selection** | Classes are 39/39/15/9%. Accuracy alone rewards abandoning Stage 2 — which is what the paper's models do (its AvgPool_VGG-16 gets recall **0.05** on Stage 2). |
| 4 | **Proper PPG conditioning** | Detrend → 0.5–8 Hz zero-phase Butterworth → 125 Hz → z-score, instead of a bare 50-sample moving average. |
| 5 | **10 waveform augmentations** | Applied to the *signal*, before rendering — noise, gain, baseline wander, time warp, crop-resize, cutout. Image-space flips would produce waveforms no sensor can emit. |
| 6 | **SSL pretraining on 49.8 h** | Uses the local datasets to compensate for 657 labelled samples. |
| 7 | **Morphology-feature baseline** | 63 interpretable features + gradient boosting. Runs in <1 min on CPU and is the bar every deep model must clear. |
| 8 | **Optional clinical fusion** | Age/sex/BMI/HR, kept as a *separate* run — a fused score is not a pure-PPG result. Measured effect here was neutral-to-negative; see `RESULTS.md`. |

---

## Scoring one recording

Everything above *evaluates*. To actually score a PPG recording you have not
seen before, build a model once and then call it:

```bash
python -m hyperppg.fit_model                        # ~60 s, writes models/hypertension_hgb.joblib
python -m hyperppg.predict --from-dataset 2         # a real labelled subject
python -m hyperppg.predict --input recording.csv    # anything else
```

```
  recording            PPG-BP subject 2  (2.1 s at 1000 Hz)
  predicted stage      Stage 2 hypertension
  confidence           61.2%   (runner-up: Normal)

  class probabilities
    Stage 2 hypertension      61.2%  #################
    Normal                    26.1%  #######
    Prehypertension           10.1%  ###
    Stage 1 hypertension       2.6%  #

  ground truth         Stage 2 hypertension   -> correct

  model accuracy       48.7% subject-wise out-of-fold (macro-F1 0.417)
                       majority-class baseline is 38.8%. This is a screening
                       prior on 4 classes, not a diagnosis.
  note: this segment is in the training set; showing its held-out (out-of-fold)
        score, not the fitted model's memory of it
```

**Formats.** `.txt`/`.dat` (PPG-BP's own whitespace-separated numbers), `.csv`/`.tsv`,
`.npy`, `.json`. The rate comes from `--fs`, else a time column, else it falls back
to PPG-BP's 1000 Hz and says so.

**Longer recordings.** The model reads the morphology of a 2.1 s pulse train, so
anything longer is resampled to 1000 Hz, cut into 2.1 s windows, scored per window,
and averaged. The output reports how many windows agreed.

**Running it on your own signal.** Paths may be absolute or relative:

```bash
# Linux / macOS
python -m hyperppg.predict --input ~/recordings/finger.csv --fs 125 --plot ~/report.png

# Windows PowerShell
python -m hyperppg.predict --input C:\Users\you\recordings\finger.csv --fs 125
```

`python -m hyperppg.*` must run from the `hypertension_avgpool_vgg16/` folder,
since that is where the `hyperppg` package sits. To be free of that, set
`PYTHONPATH=/path/to/ppg-models/hypertension_avgpool_vgg16` (or `$env:PYTHONPATH`
in PowerShell) and run from anywhere.

`--from-dataset` and `fit_model` additionally need PPG-BP. They look for it in
`data/ppgbp`, then `$PPGBP_ROOT`, then the usual Colab/Kaggle locations. Point
them anywhere with `--root`:

```bash
python -m hyperppg.fit_model --root /mnt/datasets/ppgbp
export PPGBP_ROOT=/mnt/datasets/ppgbp        # or pass --root every time
```

Plain `--input` needs none of that — only the `.joblib` model, which travels as
a single file. Copy it to another machine, install the requirements, and predict.

**The report figure.** `--plot report.png` writes one page: the analysed window with
its systolic peaks, the class probabilities, and — for a recording long enough to
window — the per-window verdict over time, which is where you see whether a call
held throughout or only in patches. A single 2.1 s segment gets a beat overlay in
that third panel instead. The window shown is the *median*-confidence one among
those agreeing with the verdict, not the best, so the panel is representative
rather than flattering.

```bash
python -m hyperppg.predict --from-dataset 2 --plot report.png
```

**Training segments are caught.** The deployed model is fitted on all 657 segments,
so handing it one of them back would return a memorised ~99% answer. Every training
waveform is fingerprinted at fit time; when one is recognised — by file or by
subject id — `predict` serves the score it got from the fold that had never seen it,
and says that is what it did. The honest number for a training segment is the
out-of-fold one.

---

## Reproducing the paper's own numbers

To see the paper's setup rather than the honest one:

```bash
python -m hyperppg.train_paper --model all --split segment
```

Replication note: training AlexNet on this data reproduces the paper's reported
failure mode exactly — it collapses to predicting a single class at **0.3881**
accuracy, against the paper's reported 0.39 with all predictions in class `pt`.

---

## Measured results

Full detail, including per-class breakdowns and runtimes, is in
**[`RESULTS.md`](RESULTS.md)**. Headline, all subject-wise 5-fold and pooled
out-of-fold over all 657 segments:

| Configuration | Accuracy | Macro-F1 |
|---|---|---|
| majority-class baseline | 0.3881 | 0.1398 |
| morphology + GBM (PPG only) | 0.4871 | 0.4204 |
| **1-D hybrid, from scratch** | **0.5388** | **0.5012** |
| 1-D hybrid + SSL init | 0.5266 | 0.4500 |
| 1-D hybrid + SSL + clinical covariates | 0.5373 | 0.4738 |

**0.5388 accuracy / 0.5012 macro-F1**, PPG only, no leakage — inside the
50–60% band, +15.1 points over the majority baseline.

The comparison that matters most against the paper is **Stage 2 recall**. The
paper's AvgPool_VGG-16 reports recall **0.05** on Stage 2 (1 of 20 cases); this
model reaches **0.483**, on a harder protocol, while also scoring higher
overall.

Two things did **not** work, reported because negative results are informative:

- **SSL pretraining slightly hurt** (0.5266 vs 0.5388). The pretext task trained
  fine (loss 0.202 → 0.035); the likely culprit is the wrist-vs-fingertip
  sensor-site gap, plus a deliberately light 12-epoch pretrain. See `RESULTS.md`.
- **Clinical covariate fusion did not help** (GBM 0.4612 vs 0.4871; hybrid
  0.5373 vs 0.5388). Genuinely surprising given age and BMI are established risk
  factors — under a subject-wise split, 7 extra covariates on 219 subjects
  appear to add more variance than signal.

Four-class staging from 2.1 s of PPG is genuinely hard. The dominant residual
error is `Normal ↔ Prehypertension`, a boundary defined by a blood-pressure
threshold (120/80) rather than by pulse morphology. The paper itself concedes
its models "aren't able to distinguish between stage 1 and stage 2
hypertension."

Any published number on this dataset above ~0.65 without subject-wise splitting
should be assumed to be measuring leakage.

---

## Data sources

- **PPG-BP**: Liang, Y., Liu, G., Chen, Z., Elgendi, M. *PPG-BP Database*,
  figshare (2018). [doi:10.6084/m9.figshare.5459299](https://doi.org/10.6084/m9.figshare.5459299) —
  downloaded automatically by `hyperppg.data.download`.
- **PPG-DaLiA**: Reiss, A., Indlekofer, I., Schmidt, P., Van Laerhoven, K.
  *Deep PPG: Large-scale Heart Rate Estimation with Convolutional Neural
  Networks*, Sensors 19(14), 2019.
- **FatigueSet**: Kalanadhabhatta, M. et al. *FatigueSet: A Multi-modal Dataset
  for Modeling Mental Fatigue and Fatigability*, PervasiveHealth 2021.
