"""Write a synthetic labelled cohort, so training can be run with no data.

Sixteen participants, matching the paper's sample size. Each gets a latent
fatigue level that sets the dicrotic-peak height of a 2-minute recording, a
respiratory R-R modulation that is deliberately *unrelated* to fatigue (so NHF
stays a weak predictor, as the paper found), and nine BFI-Taiwan answers whose
loadings follow what the paper reported: Q2 dominant, Q3 second, the rest weak.

This exercises the whole extract -> train -> infer path end to end. It is a
demonstration that the pipeline recovers a known structure, **not** evidence
about human fatigue.

    python scripts/make_demo_cohort.py --out demo_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fatigueppg.config import DURATION_PAPER, FS_PAPER          # noqa: E402
from fatigueppg.synth import synth_ppg                          # noqa: E402

#: How strongly each BFI item is driven by the latent state. The paper found
#: Q2 (r = 0.8743) and Q3 (r = 0.5328) carried the signal and the rest did not.
Q_LOADING = np.array([0.55, 0.90, 0.60, 0.45, 0.15, 0.15, 0.10, 0.05, 0.20])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="demo_data", help="output directory")
    p.add_argument("--n", type=int, default=16, help="participants")
    p.add_argument("--duration", type=float, default=DURATION_PAPER)
    p.add_argument("--fs", type=float, default=FS_PAPER)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out)
    (out / "recordings").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for i, latent in enumerate(rng.uniform(2.0, 9.0, size=args.n)):
        pid = f"P{i+1:02d}"
        sig, _ = synth_ppg(args.duration, args.fs, hr=rng.uniform(58, 88),
                           dicrotic=0.50 + 0.045 * latent, hrv_sd=0.025,
                           resp_depth=rng.uniform(0.0, 0.07),
                           resp_hz=rng.uniform(0.20, 0.30), noise=0.02,
                           seed=1000 + i)
        path = out / "recordings" / f"{pid}.csv"
        pd.DataFrame({"time": np.arange(sig.size) / args.fs,
                      "ppg": sig}).to_csv(path, index=False)

        answers = np.clip(np.round(
            Q_LOADING * latent + (1 - Q_LOADING) * rng.uniform(0, 10)
            + rng.normal(0, 0.6, size=9)), 0, 10)
        rows.append(dict(path=f"recordings/{pid}.csv", subject=pid,
                         session="1", fs=args.fs, latent_fatigue=round(latent, 3),
                         **{f"q{j+1}": answers[j] for j in range(9)}))

    manifest = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"{args.n} recordings of {args.duration:g} s at {args.fs:g} Hz "
          f"-> {out/'recordings'}")
    print(f"manifest -> {manifest}")
    print("\nnext:")
    print(f"  python -m fatigueppg.extract --manifest {manifest} "
          f"-o {out/'features.csv'}")
    print(f"  python -m fatigueppg.train --features {out/'features.csv'} "
          f"-o models/demo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
