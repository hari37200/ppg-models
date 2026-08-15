#!/usr/bin/env bash
# The whole path, start to finish, on synthetic data. No downloads, no GPU.
#
#   bash scripts/quickstart.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}

echo "=== 1/5  validating the pipeline against ground truth ==============="
$PY -m fatigueppg.selfcheck

echo
echo "=== 2/5  inference on the bundled example =========================="
$PY -m fatigueppg.infer --input examples/demo_ppg_200hz.csv --plot demo_data/report.png

echo
echo "=== 3/5  building a 16-participant synthetic cohort ================"
$PY scripts/make_demo_cohort.py --out demo_data

echo
echo "=== 4/5  extracting features ======================================="
$PY -m fatigueppg.extract --manifest demo_data/manifest.csv -o demo_data/features.csv

echo
echo "=== 5/5  fitting Equation (9) on that cohort ======================="
$PY -m fatigueppg.train --features demo_data/features.csv \
    --out models/demo_cohort.json --plot demo_data/regression.png

echo
echo "=== applying the fitted model ======================================"
$PY -m fatigueppg.infer --model models/demo_cohort.json \
    --input demo_data/recordings --glob "P0[1-3].csv" \
    --csv demo_data/batch.csv --quiet
$PY - <<'EOF'
import pandas as pd
print(pd.read_csv("demo_data/batch.csv")[
    ["name", "fatigue_index", "subjective_pred", "alert", "hr", "sqi"]
].round(3).to_string(index=False))
EOF

echo
echo "done. artifacts in demo_data/ and models/demo_cohort.json"
