# Run guide — Google Colab and Kaggle

Everything here runs on a **free** Colab or Kaggle GPU session. The labelled
dataset is 1.5 MB and downloads itself, so you can be training within two
minutes of opening a notebook.

---

## 0. What you are running

| Script | What it does | Measured runtime (CPU, 8 threads) |
|---|---|---|
| `hyperppg.selfcheck` | 14-point pipeline validation | ~10 s |
| `hyperppg.features_baseline` | 63 morphology features + gradient boosting | ~75 s |
| `hyperppg.train_hybrid` | the improved 1-D CNN+transformer, 5 folds × 80 epochs | ~11 min |
| `hyperppg.pretrain_ssl` | self-supervised pretraining on PPG-DaLiA + FatigueSet | ~16 min (20 k windows, 12 epochs) |
| `hyperppg.train_paper` | replicate AlexNet / ResNet-50 / VGG-16 / AvgPool_VGG-16 | **needs a GPU** |

Those are wall-clock times measured on CPU; a T4 will be faster. The one
exception is `train_paper` — VGG-16 is 134 M parameters against the hybrid's
2.2 M and is impractical without a GPU.

Recommended order: **selfcheck → features_baseline → train_hybrid → train_paper
→ pretrain_ssl → train_hybrid --ssl-checkpoint**.

Do the cheap baseline first. It takes one minute and tells you what "good"
means on this dataset before you spend GPU time.

---

## 1. Google Colab

### 1.1 Upload the code

`Runtime → Change runtime type → T4 GPU` first.

Zip this folder on your machine:

```powershell
Compress-Archive -Path E:\ppg\hypertension_avgpool_vgg16 -DestinationPath E:\ppg\hyperppg_code.zip -Force
```

Then in a Colab cell, click the folder icon in the left sidebar and upload
`hyperppg_code.zip`, or run:

```python
from google.colab import files
up = files.upload()          # choose hyperppg_code.zip
```

Unpack and enter it:

```python
import zipfile, os
zipfile.ZipFile('hyperppg_code.zip').extractall('/content')
os.chdir('/content/hypertension_avgpool_vgg16')
print(os.listdir())
```

> **Alternative — Google Drive.** Copy the folder into Drive once, then:
> ```python
> from google.colab import drive; drive.mount('/content/drive')
> %cd /content/drive/MyDrive/hypertension_avgpool_vgg16
> ```
> This also makes checkpoints survive a disconnect.

### 1.2 Install dependencies

```python
!pip install -q openpyxl lightgbm
```

That is all you need. **Do not `pip install torch`** — Colab already has the
correct CUDA build and reinstalling breaks the GPU runtime.

### 1.3 Get the labelled dataset (1.5 MB)

```python
!python -m hyperppg.data.download --dest data/ppgbp
```

Expected output:

```
[download] fetching https://ndownloader.figshare.com/files/9441097
[download] got 1.51 MB
[verify] 657 segments + spreadsheet OK
```

### 1.4 Validate before training

```python
!python -m hyperppg.selfcheck
```

You want `14/14 checks passed`. If anything fails, stop and read the message —
it names the exact stage. Do not train on a broken pipeline.

### 1.5 Run

```python
# 1 min, CPU — the bar every deep model must clear
!python -m hyperppg.features_baseline --split both

# ~8 min on a T4 — the improved model
!python -m hyperppg.train_hybrid --folds 5 --epochs 80

# replicate the paper (both protocols, so you can see the leakage gap)
!python -m hyperppg.train_paper --model avgpool_vgg16 --split both --epochs 30
```

---

## 2. Kaggle

`Notebook → Settings → Accelerator → GPU T4 x2`, and **turn Internet ON**
(Settings → Internet). The figshare download needs it.

### 2.1 Upload the code

Easiest route: `+ Add Input → Upload → New Dataset`, upload
`hyperppg_code.zip`, name it `hyperppg-code`. Then:

```python
import shutil, zipfile, os
shutil.copytree('/kaggle/input/hyperppg-code', '/kaggle/working/src', dirs_exist_ok=True)
os.chdir('/kaggle/working/src')
# if you uploaded the .zip rather than the unpacked folder:
if os.path.exists('hyperppg_code.zip'):
    zipfile.ZipFile('hyperppg_code.zip').extractall('.')
    os.chdir('hypertension_avgpool_vgg16')
print(os.listdir())
```

Kaggle input directories are **read-only**, so always copy into
`/kaggle/working` before running.

### 2.2 Install and fetch data

```python
!pip install -q openpyxl
!python -m hyperppg.data.download --dest /kaggle/working/data/ppgbp
```

If Internet is off and you cannot enable it, add the PPG-BP dataset as a Kaggle
input instead — `config.resolve_ppgbp_root()` already scans `/kaggle/input`
two levels deep for a directory containing `Data File/0_subject`, so no code
change is needed.

You can also point at it explicitly:

```python
import os
os.environ['PPGBP_ROOT'] = '/kaggle/input/<your-dataset-slug>'
```

### 2.3 Run

Identical to Colab:

```python
!python -m hyperppg.selfcheck
!python -m hyperppg.features_baseline --split both
!python -m hyperppg.train_hybrid --folds 5 --epochs 80
```

Outputs land in `/kaggle/working/hyperppg_runs/` and are downloadable from the
notebook's Output tab.

---

## 3. Self-supervised pretraining on your local datasets

This is the part that uses **PPG-DaLiA** and **FatigueSet** from `E:\ppg`.

### 3.1 Build a small upload bundle (on Windows)

The two datasets total ~3.5 GB, which is painful to upload. SSL only needs the
unlabelled wrist BVP — the E4 archives and the `wrist_bvp.csv` files:

```powershell
E:\ppg\hypertension_avgpool_vgg16\scripts\make_pretrain_bundle.ps1
```

```
  PPG-DaLiA: copied 15 E4 archives
  FatigueSet: extracted 36 wrist_bvp.csv files
Bundle written to E:\ppg\ppg_pretrain.zip (49.7 MB)
```

**3.5 GB → 49.7 MB.** The trick is that PPG-DaLiA's wrist BVP lives in
`SX_E4.zip` (~2.6 MB) rather than `SX.pkl` (~1.4 GB) — we need the waveform,
not the ECG-synchronised heart-rate labels.

### 3.2 Upload and pretrain

Upload `ppg_pretrain.zip` the same way you uploaded the code, then:

```python
import zipfile
zipfile.ZipFile('/content/ppg_pretrain.zip').extractall('/content/pretrain')

!python -m hyperppg.pretrain_ssl \
    --dalia /content/pretrain/dalia \
    --fatigueset /content/pretrain/fatigueset \
    --max-windows 120000 --epochs 30 \
    --cache /content/corpus.npy \
    --out runs/ssl
```

Expect roughly:

```
[corpus] 51 recordings, 49.8 hours of wrist PPG
[corpus] windows: (120000, 263)
model: 2.35M parameters
epoch   0 | train 0.38100 | val 0.29719 | 14.7s  *
...
encoder saved to runs/ssl/ssl_encoder.pt
```

`--cache` saves the window corpus so a re-run skips the slow decode step.

### 3.3 Fine-tune

```python
# pure PPG, SSL-initialised
!python -m hyperppg.train_hybrid --ssl-checkpoint runs/ssl/ssl_encoder.pt --folds 5 --epochs 80

# plus clinical covariates (see the caveat in section 5)
!python -m hyperppg.train_hybrid --ssl-checkpoint runs/ssl/ssl_encoder.pt --tabular --folds 5 --epochs 80
```

Confirm the transfer actually happened — you should see this line per fold:

```
  loaded 104 encoder tensors from SSL checkpoint
```

If it says 0, the run aborts rather than silently training from scratch.

---

## 3.5 Scoring a single recording

The sections above cross-validate. To score one recording you need a persisted
model, which nothing else in this repo writes:

```bash
python -m hyperppg.fit_model                     # ~60 s CPU -> models/hypertension_hgb.joblib
python -m hyperppg.predict --from-dataset 2      # a real labelled subject
python -m hyperppg.predict --input recording.csv --fs 125
python -m hyperppg.predict --from-dataset 2 --plot report.png
```

`fit_model` cross-validates subject-wise for the metrics on the model card, then
refits on all 657 segments for deployment. `predict` resamples to 1000 Hz, cuts
2.1 s windows, and averages the per-window probabilities.

One behaviour to know about: the deployed model has seen every PPG-BP segment,
so scoring one of them with the fitted model returns a memorised ~99% answer.
Every training waveform is fingerprinted at fit time, and `predict` recognises
one however it arrives — by file or by subject id — and serves the score it got
from the fold that had never seen it instead, saying so in a note. If you want a
demo number to be real, that is the number.

---

## 4. Reading the output

Every run prints a leakage audit before training:

```
fold   train    val  trn subj  val subj  shared
   0     525    132       109       110       0
-> NO subject leakage
```

then a pooled out-of-fold report over all 657 segments:

```
                      precision    recall  f1-score   support
              Normal      0.622     0.650     0.635       240
     Prehypertension      0.560     0.600     0.580       255
Stage 1 hypertension      0.411     0.294     0.343       102
Stage 2 hypertension      0.483     0.483     0.483        60
            accuracy                          0.560       657

accuracy 0.5601 | balanced acc 0.5069 | macro-F1 0.5103 | kappa 0.3469
majority-class baseline: accuracy 0.3881 | macro-F1 0.1398
```

**Always read accuracy next to the majority baseline (0.388) and macro-F1.** A
model at 0.45 accuracy with macro-F1 0.19 has simply stopped predicting the two
minority classes.

Artifacts per run: `<title>_results.json`, `<title>_report.txt`,
`<title>_oof_pred.npy`, `<title>_oof_logits.npy`. The saved logits let you
ensemble runs afterwards without retraining.

---

## 5. Hitting the 50–60% target

Four-class staging from 2.1 s of PPG is hard, and the split you choose changes
the number more than the model does.

**Good news: the default configuration already gets there.** These are measured
numbers from an actual run on this machine (CPU, 5-fold, pooled out-of-fold over
all 657 segments). Full detail in [`RESULTS.md`](RESULTS.md).

| Configuration | Split | Accuracy | Macro-F1 |
|---|---|---|---|
| majority-class baseline | — | 0.3881 | 0.1398 |
| morphology + GBM | segment (leaks) | 0.5586 | 0.5078 |
| morphology + GBM | subject | 0.4871 | 0.4204 |
| morphology + GBM + `--tabular` | subject | 0.4612 | 0.3933 |
| **1-D hybrid, from scratch** | **subject** | **0.5388** | **0.5012** |
| 1-D hybrid + SSL init | subject | 0.5266 | 0.4500 |
| 1-D hybrid + SSL + `--tabular` | subject | 0.5373 | 0.4738 |

So this single command clears the target honestly, with no leakage and no
clinical covariates:

```bash
python -m hyperppg.train_hybrid --folds 5 --epochs 80
```

Two results are worth knowing before you spend time on them:

- **SSL pretraining did not help** in the configuration tested (0.5266 vs
  0.5388). The pretext task trains fine; the likely cause is the wrist-vs-
  fingertip sensor gap plus a light 12-epoch pretrain. Try 30 epochs over
  120 000 windows before writing it off.
- **`--tabular` did not help either** — it made the GBM *worse* (0.4612 vs
  0.4871) and left the hybrid unchanged. Age and BMI are real risk factors, but
  under a subject-wise split 7 extra covariates on 219 subjects add more
  variance than signal.

Further routes, in order of expected value:

1. **Ensemble seeds.** Cheapest reliable gain. Re-run with `--seed 1`, `--seed 2`
   and average the saved logits (section 11 of the notebook):

   ```python
   import numpy as np, glob
   probs = []
   for p in glob.glob('runs/**/*_oof_logits.npy', recursive=True):
       z = np.load(p); z = z - z.max(1, keepdims=True)
       e = np.exp(z); probs.append(e / e.sum(1, keepdims=True))
   pred = np.mean(probs, axis=0).argmax(1)
   ```
2. **Longer SSL pretraining**, per the note above.
3. **Collapse to 3 or 2 classes.** Stage 1 vs Stage 2 has only 34 and 20
   subjects. Normotensive vs hypertensive is far better posed and lands much
   higher. A legitimate reframing, not a workaround — just say which task you
   solved.
4. **`--split segment`** gives 0.55–0.60 instantly, but that is the paper's
   protocol and it is leakage. Use it only for direct comparison, and label it.

Also worth reporting alongside accuracy: the hybrid reaches **0.483 recall on
Stage 2**, where the paper's AvgPool_VGG-16 reports **0.05**. On a
clinically-motivated task that difference matters more than the headline number.

---

## 6. Useful flags

```bash
--split subject|segment|both     evaluation protocol (default: subject)
--folds 5                        CV folds
--epochs 80
--augment none|light|medium|strong
--mixup 0.2                      0 disables
--tabular                        fuse age/sex/height/weight/BMI/HR/diabetes
--no-class-weights               disable inverse-frequency weighting
--seed 0                         change for ensembling
--device cpu|cuda|auto
--out runs/myrun
```

Paper-track only:

```bash
--model alexnet|resnet50|vgg16|avgpool_vgg16|all
--no-pretrained                  train from scratch instead of ImageNet init
--freeze-features                train only the classifier head (fast, heavily regularised)
--img-size 224
```

---

## 7. Troubleshooting

**`Could not locate the PPG-BP dataset`**
Run `python -m hyperppg.data.download --dest data/ppgbp`, or set
`PPGBP_ROOT=/path/to/dataset`. The error lists every path it probed.

**`index has N segments / M subjects, expected 657 / 219`**
Partial download. Re-run with `--force`, or pass `strict=False` to
`build_index` to continue with what you have.

**CUDA out of memory on VGG-16**
`--batch-size 16`, or `--img-size 160`, or `--freeze-features`. VGG-16 is 134 M
parameters; the hybrid is 2.2 M and will not run out.

**Kaggle: `OSError: Read-only file system`**
You are running from `/kaggle/input`. Copy to `/kaggle/working` first.

**Colab session disconnects mid-run**
Work from Drive (section 1.1) so checkpoints persist, and lower `--epochs`.

**DataLoader worker crashes / hangs**
`--num-workers 0`.

**`transferred 0 tensors` from the SSL checkpoint**
The checkpoint was written by a different encoder configuration. Re-run
`pretrain_ssl` with the current code — the encoder defaults must match.

---

## 8. Sanity checks worth keeping

- `selfcheck` must report **14/14** before you trust any number.
- The leakage audit must say **NO subject leakage** for any result you report
  as honest.
- Accuracy must be read against the **0.388** majority baseline.
- Macro-F1 well below accuracy means minority classes are being dropped.
- If SSL fine-tuning does not print **`loaded 104 encoder tensors`**, the
  pretraining is not being used.
