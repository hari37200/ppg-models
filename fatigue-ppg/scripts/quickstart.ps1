<#
.SYNOPSIS
    The whole path, start to finish, on synthetic data. No downloads, no GPU.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Write-Host "=== 1/5  validating the pipeline against ground truth ===============" -ForegroundColor Cyan
& $py -m fatigueppg.selfcheck
if ($LASTEXITCODE -ne 0) { throw "self-check failed" }

Write-Host "`n=== 2/5  inference on the bundled example ==========================" -ForegroundColor Cyan
& $py -m fatigueppg.infer --input examples/demo_ppg_200hz.csv --plot demo_data/report.png

Write-Host "`n=== 3/5  building a 16-participant synthetic cohort ================" -ForegroundColor Cyan
& $py scripts/make_demo_cohort.py --out demo_data

Write-Host "`n=== 4/5  extracting features =======================================" -ForegroundColor Cyan
& $py -m fatigueppg.extract --manifest demo_data/manifest.csv -o demo_data/features.csv

Write-Host "`n=== 5/5  fitting Equation (9) on that cohort =======================" -ForegroundColor Cyan
& $py -m fatigueppg.train --features demo_data/features.csv --out models/demo_cohort.json --plot demo_data/regression.png

Write-Host "`n=== applying the fitted model ======================================" -ForegroundColor Cyan
& $py -m fatigueppg.infer --model models/demo_cohort.json --input demo_data/recordings --glob "P0[1-3].csv" --csv demo_data/batch.csv --quiet
& $py -c "import pandas as pd; print(pd.read_csv('demo_data/batch.csv')[['name','fatigue_index','subjective_pred','alert','hr','sqi']].round(3).to_string(index=False))"

Write-Host "`ndone. artifacts in demo_data\ and models\demo_cohort.json" -ForegroundColor Green
