# ppg-models

Two PPG papers, reproduced from the text and measured honestly on public data.
Both run on CPU. Neither needs a GPU.

| | [`fatigue-ppg/`](fatigue-ppg/) | [`hypertension_avgpool_vgg16/`](hypertension_avgpool_vgg16/) |
|---|---|---|
| Paper | Chen et al., *Mathematics* 2023, 11, 3580 | Frederick et al., arXiv:2304.06952 |
| Task | fatigue index from dicrotic-peak position | 4-stage hypertension from a 2.1 s pulse |
| Data | PPG-BP, FatigueSet, PPG-DaLiA | PPG-BP (657 segments, 219 subjects) |
| Headline | 94.8% exact-sample peak agreement on 657 real segments | 0.5388 accuracy / 0.5012 macro-F1, no subject leakage |
| Finding | the published index floors on any cohort but the authors' | the paper's 80% is inflated 7.3 points by subject leakage |
| Inference | `python -m fatigueppg.infer --input rec.csv` | `python -m hyperppg.predict --input rec.csv` |

Each directory is self-contained: its own README, `requirements.txt` and tests.
Start with the README in whichever one you need.

    cd fatigue-ppg
    pip install -r requirements.txt
    python -m fatigueppg.selfcheck

    cd hypertension_avgpool_vgg16
    pip install -r requirements.txt
    python -m hyperppg.selfcheck --skip-torch

## Licence

To be added by the group. Until then treat the code as all-rights-reserved.
Both papers belong to their original authors; see each project's README for
attribution.
