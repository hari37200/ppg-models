# Measured results

All numbers below were produced by running the code in this repo. They are not
estimates.

**Hardware:** CPU only (8 threads, WSL2 Ubuntu, torch 2.13 CPU build). A GPU
will be several times faster but should not change the scores.

**Protocol:** 5-fold cross-validation, pooled out-of-fold predictions over all
657 segments, so every segment is predicted exactly once by a model that never
saw it. `--split subject` groups by `subject_id` unless stated otherwise.

Reproduce with:

```bash
python -m hyperppg.features_baseline --split both
python -m hyperppg.features_baseline --split subject --tabular
python -m hyperppg.pretrain_ssl --dalia <...>/dalia --fatigueset <...>/fatigueset \
    --max-windows 20000 --epochs 12
python -m hyperppg.train_hybrid --folds 5 --epochs 80
python -m hyperppg.train_hybrid --folds 5 --epochs 80 --ssl-checkpoint runs/ssl/ssl_encoder.pt
python -m hyperppg.train_hybrid --folds 5 --epochs 80 --ssl-checkpoint runs/ssl/ssl_encoder.pt --tabular
```

---

## Headline table

| # | Configuration | Split | Accuracy | Macro-F1 | Balanced acc | Kappa |
|---|---|---|---|---|---|---|
| 0 | majority-class baseline | — | 0.3881 | 0.1398 | 0.2500 | 0.000 |
| 1 | morphology + GBM (PPG only) | subject | 0.4871 | 0.4204 | 0.4088 | 0.230 |
| 2 | morphology + GBM (PPG only) | **segment** | 0.5586 | 0.5078 | 0.5059 | 0.345 |
| 3 | morphology + GBM + clinical | subject | 0.4612 | 0.3933 | 0.3884 | 0.195 |
| 4 | **1-D hybrid, from scratch** | subject | **0.5388** | **0.5012** | 0.5010 | 0.323 |
| 5 | 1-D hybrid + SSL init | subject | 0.5266 | 0.4500 | 0.4445 | 0.290 |
| 6 | 1-D hybrid + SSL + clinical | subject | 0.5373 | 0.4738 | 0.4674 | 0.310 |

**Best honest result: row 4 — 0.5388 accuracy / 0.5012 macro-F1**, PPG only, no
subject leakage. That is inside the 50–60% target band and **+15.1 points over
the majority baseline**, **+5.2 over the classical baseline**.

Per-fold spread for row 4: accuracy `0.5385 ± 0.0477`, macro-F1 `0.4998 ± 0.0504`
(5 folds). With 131 segments per validation fold, that spread is expected —
treat differences under ~4 points between configurations as noise.

---

## The leakage measurement

Rows 1 and 2 are the *same model on the same features*; only the split differs.

| Split | Accuracy | Macro-F1 |
|---|---|---|
| `segment` (the paper's protocol) | 0.5586 | 0.5078 |
| `subject` (grouped) | 0.4871 | 0.4204 |
| **inflation** | **+7.2 pts** | **+8.7 pts** |

PPG-BP has 3 segments per subject recorded seconds apart. A segment-level split
puts the same person on both sides. This is the mechanism behind the paper's
reported 80%.

---

## Best model, per class

`hybrid_scratch_subject`, pooled out-of-fold:

```
                      precision    recall  f1-score   support
              Normal      0.711     0.583     0.641       240
     Prehypertension      0.492     0.584     0.534       255
Stage 1 hypertension      0.404     0.353     0.377       102
Stage 2 hypertension      0.426     0.483     0.453        60
            accuracy                          0.539       657

confusion matrix (rows = true, cols = predicted)
                          Normal Prehypert Stage 1 h Stage 2 h
Normal                       140        84        10         6
Prehypertension               46       149        36        24
Stage 1 hypertension           7        50        36         9
Stage 2 hypertension           4        20         7        29
```

**The most meaningful comparison with the paper is not accuracy, it is Stage 2
recall.** The paper's AvgPool_VGG-16 reports precision 1.00 / recall **0.05** on
Stage 2 — it identifies 1 of 20 cases and gets its headline accuracy from the
two easy majority classes. This model reaches **0.483 recall on Stage 2** while
scoring higher overall on a *harder* protocol.

Every class has non-trivial recall. That is what class-weighted loss and
macro-F1 model selection buy, and it is the difference between a usable
screening model and a number on a leaderboard.

The dominant residual error is `Normal ↔ Prehypertension` (84 + 46 = 130 of 303
errors). That boundary is a blood-pressure threshold (120/80), not a
morphological one, so it is not obviously recoverable from 2.1 s of PPG.

---

## Two things that did *not* work

Reported because negative results here are informative, not because they were
expected.

### Self-supervised pretraining did not help (row 5 vs row 4)

`0.5266` with SSL initialisation vs `0.5388` from scratch — a 1.2-point
*decrease*, well inside fold noise but certainly not the gain the approach
promises. The SSL objective itself trained fine (masked-reconstruction
validation loss 0.202 → 0.035 over 12 epochs on 20 000 windows).

Most likely causes, in order:

1. **Sensor-site domain gap.** PPG-DaLiA and FatigueSet are *wrist* PPG recorded
   during walking, cycling and stair-climbing. PPG-BP is a clean seated
   *fingertip* recording. Wrist PPG under motion has a very different morphology
   — the dicrotic notch is frequently absent — so the pretrained features may
   not be the ones the staging task needs.
2. **Under-trained pretext task.** 12 epochs over 20 000 windows is light. The
   guide suggests 30 epochs over 120 000 windows.
3. **Reconstruction may be the wrong objective.** Inpainting rewards modelling
   high-frequency detail; hypertension staging depends on slow contour features.
   A contrastive objective over augmented views would likely transfer better.

Worth retrying with a longer pretrain before concluding the idea fails. The code
path is verified working (104 encoder tensors transfer, confirmed by an assert).

### Clinical covariate fusion did not help (rows 3 and 6)

- GBM: `0.4612` with covariates vs `0.4871` without — **worse**.
- Hybrid: `0.5373` vs `0.5388` — unchanged.

This was genuinely surprising; age and BMI are established hypertension risk
factors, and an earlier draft of this repo's documentation asserted fusion would
"push accuracy up materially". Measurement says otherwise. The plausible reading
is that under a *subject-wise* split the model cannot memorise individuals, and
7 extra covariates on 219 subjects add more variance than signal — age alone
does not separate Prehypertension from Stage 1 well enough to beat what the
waveform already provides.

---

## Runtimes (CPU, 8 threads)

| Step | Time |
|---|---|
| `selfcheck` | ~10 s |
| `features_baseline --split both` | ~75 s |
| `pretrain_ssl` (20 k windows, 12 epochs) | ~16 min (79 s/epoch) |
| `train_hybrid` (5 folds × 80 epochs) | ~11.4 min |
| `train_hybrid --ssl-checkpoint` | ~11.0 min |
| `train_hybrid --ssl-checkpoint --tabular` | ~12.4 min |

The paper's image CNNs were not benchmarked to convergence here — VGG-16 is
134 M parameters against the hybrid's 2.2 M and is impractical on CPU. Use a GPU
for `train_paper`. A partial run does reproduce the paper's AlexNet failure mode
exactly: collapse to a single class at 0.3881 accuracy, against the paper's
reported 0.39 with all predictions in class `pt`.

---

## How to push higher

1. **Ensemble seeds.** Re-run row 4 with `--seed 1/2/3` and average the saved
   `*_oof_logits.npy`. Cheapest reliable gain (typically +1–3 points).
2. **Longer SSL**, per the analysis above.
3. **Reframe to 2 or 3 classes.** Stage 1 vs Stage 2 has only 34 and 20
   subjects. Normotensive vs hypertensive is far better posed and scores much
   higher — a legitimate reframing as long as you state which task you solved.
4. **Report `--split segment`** only for direct comparison with the paper, and
   label it as leaking.
