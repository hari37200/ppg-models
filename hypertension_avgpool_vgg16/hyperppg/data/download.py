"""Fetch and extract the PPG-BP database (~1.5 MB) from figshare.

Usage
-----
    python -m hyperppg.data.download --dest data/ppgbp
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from hyperppg.config import (
    EXPECTED_N_SEGMENTS,
    FIGSHARE_URL,
    SUBJECT_DIR_REL,
    XLSX_REL,
    _is_ppgbp_root,
)


def download_ppgbp(dest: str | Path = "data/ppgbp", force: bool = False) -> Path:
    """Download + extract PPG-BP so that ``dest/Data File/0_subject/`` exists.

    Returns the dataset root (``dest``). Idempotent: a complete existing copy
    is left alone unless ``force`` is set.
    """
    dest = Path(dest).expanduser()

    if _is_ppgbp_root(dest) and not force:
        print(f"[download] already present at {dest.resolve()} -- skipping")
        return dest.resolve()

    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "ppgbp.zip"
        print(f"[download] fetching {FIGSHARE_URL}")
        with urllib.request.urlopen(FIGSHARE_URL, timeout=300) as resp:
            zip_path.write_bytes(resp.read())
        print(f"[download] got {zip_path.stat().st_size / 1e6:.2f} MB")

        extract_dir = tmp / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # The archive may or may not carry a single top-level wrapper folder;
        # locate the directory that actually holds "Data File".
        root = _find_data_root(extract_dir)
        if root is None:
            raise RuntimeError(
                f"Extracted archive has an unexpected layout under {extract_dir}. "
                f"Expected to find '{SUBJECT_DIR_REL}'."
            )

        for item in root.iterdir():
            target = dest / item.name
            if target.exists():
                if force:
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                else:
                    continue
            shutil.move(str(item), str(target))

    verify(dest)
    print(f"[download] ready at {dest.resolve()}")
    return dest.resolve()


def _find_data_root(base: Path) -> Path | None:
    """Find the directory containing ``Data File/0_subject`` at depth <= 3."""
    if _is_ppgbp_root(base):
        return base
    for cand in base.rglob("0_subject"):
        if cand.is_dir():
            # cand = <root>/Data File/0_subject
            root = cand.parent.parent
            if _is_ppgbp_root(root):
                return root
    return None


def verify(root: str | Path) -> None:
    """Raise unless ``root`` holds the expected 657 segments and spreadsheet."""
    root = Path(root)
    subj_dir = root / SUBJECT_DIR_REL
    xlsx = root / XLSX_REL

    if not subj_dir.is_dir():
        raise FileNotFoundError(f"missing signal directory: {subj_dir}")
    if not xlsx.is_file():
        raise FileNotFoundError(f"missing spreadsheet: {xlsx}")

    n_txt = len(list(subj_dir.glob("*.txt")))
    if n_txt != EXPECTED_N_SEGMENTS:
        print(
            f"[verify] WARNING: found {n_txt} .txt segments, "
            f"expected {EXPECTED_N_SEGMENTS}",
            file=sys.stderr,
        )
    else:
        print(f"[verify] {n_txt} segments + spreadsheet OK")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/ppgbp", help="target directory")
    ap.add_argument("--force", action="store_true", help="re-download and overwrite")
    args = ap.parse_args(argv)
    download_ppgbp(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
