<#
.SYNOPSIS
    Build a small upload bundle of unlabelled wrist PPG for SSL pretraining.

.DESCRIPTION
    PPG-DaLiA and FatigueSet are ~3.5 GB on disk, which is painful to move onto
    Colab or Kaggle. Self-supervised pretraining only needs the unlabelled wrist
    BVP, which is a tiny fraction of that:

      PPG-DaLiA   SX/SX_E4.zip      ~2.6 MB per subject  (vs 1.4 GB for SX.pkl)
      FatigueSet  <pid>/<session>/wrist_bvp.csv

    The result is roughly 60-100 MB and preserves the directory layout that
    hyperppg.data.corpora expects, so it works unchanged after unzipping.

.EXAMPLE
    .\make_pretrain_bundle.ps1
    .\make_pretrain_bundle.ps1 -PpgRoot E:\ppg -OutFile E:\ppg\ppg_pretrain.zip
#>

[CmdletBinding()]
param(
    [string]$PpgRoot = "E:\ppg",
    [string]$OutFile = "E:\ppg\ppg_pretrain.zip"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$staging = Join-Path $env:TEMP ("ppg_pretrain_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $staging | Out-Null
Write-Host "Staging in $staging"

try {
    # ---------------- PPG-DaLiA: copy the E4 archives ----------------
    $daliaSrc = Join-Path $PpgRoot "ppg+dalia\data\PPG_FieldStudy"
    if (Test-Path $daliaSrc) {
        $daliaDst = Join-Path $staging "dalia"
        New-Item -ItemType Directory -Force -Path $daliaDst | Out-Null

        $subjects = Get-ChildItem -Path $daliaSrc -Directory |
                    Where-Object { $_.Name -match '^S\d+$' }
        $n = 0
        foreach ($s in $subjects) {
            $e4 = Join-Path $s.FullName ("{0}_E4.zip" -f $s.Name)
            if (Test-Path $e4) {
                $target = Join-Path $daliaDst $s.Name
                New-Item -ItemType Directory -Force -Path $target | Out-Null
                Copy-Item $e4 -Destination $target
                $n++
            }
        }
        Write-Host "  PPG-DaLiA: copied $n E4 archives"
    }
    else {
        Write-Warning "PPG-DaLiA not found at $daliaSrc -- skipping"
    }

    # ---------------- FatigueSet: extract wrist_bvp.csv only ----------------
    $fsZip = Join-Path $PpgRoot "fatigueset.zip"
    $fsDir = Join-Path $PpgRoot "fatigueset"
    $fsDst = Join-Path $staging "fatigueset"

    if (Test-Path $fsZip) {
        New-Item -ItemType Directory -Force -Path $fsDst | Out-Null
        $zip = [System.IO.Compression.ZipFile]::OpenRead($fsZip)
        try {
            $n = 0
            foreach ($entry in $zip.Entries) {
                if ($entry.FullName -like "*wrist_bvp.csv") {
                    # fatigueset/<pid>/<session>/wrist_bvp.csv
                    $parts = $entry.FullName -split '/'
                    if ($parts.Count -lt 3) { continue }
                    $pid_ = $parts[$parts.Count - 3]
                    $sess = $parts[$parts.Count - 2]
                    $target = Join-Path (Join-Path $fsDst $pid_) $sess
                    New-Item -ItemType Directory -Force -Path $target | Out-Null
                    $dest = Join-Path $target "wrist_bvp.csv"
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
                        $entry, $dest, $true)
                    $n++
                }
            }
            Write-Host "  FatigueSet: extracted $n wrist_bvp.csv files"
        }
        finally { $zip.Dispose() }
    }
    elseif (Test-Path $fsDir) {
        New-Item -ItemType Directory -Force -Path $fsDst | Out-Null
        $files = Get-ChildItem -Path $fsDir -Recurse -Filter "wrist_bvp.csv"
        foreach ($f in $files) {
            $sess = $f.Directory.Name
            $pid_ = $f.Directory.Parent.Name
            $target = Join-Path (Join-Path $fsDst $pid_) $sess
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            Copy-Item $f.FullName -Destination (Join-Path $target "wrist_bvp.csv")
        }
        Write-Host "  FatigueSet: copied $($files.Count) wrist_bvp.csv files"
    }
    else {
        Write-Warning "FatigueSet not found -- skipping"
    }

    # ---------------- Zip it up ----------------
    # NOTE: do not use ZipFile::CreateFromDirectory here. On Windows PowerShell
    # 5.1 (.NET Framework) it writes entry names with BACKSLASH separators,
    # which violates the ZIP spec. Python's zipfile then treats
    # "dalia\S1\S1_E4.zip" as one flat filename and creates no directories, so
    # the bundle silently fails to unpack on Colab/Kaggle. Write the entries
    # manually with forward slashes instead.
    if (Test-Path $OutFile) { Remove-Item $OutFile -Force }

    $outDir = Split-Path -Parent $OutFile
    if ($outDir -and -not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    }

    $archive = [System.IO.Compression.ZipFile]::Open(
        $OutFile, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        $prefixLen = $staging.Length + 1
        foreach ($file in Get-ChildItem -Path $staging -Recurse -File) {
            $rel = $file.FullName.Substring($prefixLen) -replace '\\', '/'
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $file.FullName, $rel,
                [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    }
    finally { $archive.Dispose() }

    $sizeMb = [math]::Round((Get-Item $OutFile).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Bundle written to $OutFile ($sizeMb MB)" -ForegroundColor Green
    Write-Host ""
    Write-Host "After unzipping on Colab/Kaggle, pretrain with:"
    Write-Host "  python -m hyperppg.pretrain_ssl --dalia <dir>/dalia --fatigueset <dir>/fatigueset"
}
finally {
    Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
}
