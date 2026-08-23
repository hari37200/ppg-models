"""Replication + improvement of AvgPool_VGG-16 hypertension staging from PPG.

Paper
-----
G. Frederick, Yaswant T, Brintha Therese A,
"PPG Signals for Hypertension Diagnosis: A Novel Method Using Deep Learning
Models", arXiv:2304.06952 (2023).

Dataset
-------
Liang, Y., Liu, G., Chen, Z., Elgendi, M. "PPG-BP Database", figshare (2018).
657 fingertip PPG segments (2.1 s @ 1000 Hz) from 219 subjects, each labelled
with one of four hypertension stages.
"""

__version__ = "0.1.0"

from hyperppg.config import CLASS_NAMES, NUM_CLASSES, FS_RAW, N_SAMPLES_RAW

__all__ = ["CLASS_NAMES", "NUM_CLASSES", "FS_RAW", "N_SAMPLES_RAW", "__version__"]
