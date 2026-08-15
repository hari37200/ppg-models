"""Download or locate the public corpora.

    python -m fatigueppg.fetch --download ppgbp
    python -m fatigueppg.fetch --status
"""
from __future__ import annotations

import argparse

from .datasets import discover_tables, download_ppgbp, resolve_all

HINTS = {
    "ppgbp": "python -m fatigueppg.fetch --download ppgbp   (1.5 MB, needs internet)",
    "fatigueset": "https://www.esense.io/datasets/fatigueset/  -> keep "
                  "<participant>/<session>/wrist_bvp.csv plus the survey files, "
                  "then set FATIGUESET_ROOT",
    "dalia": "https://archive.ics.uci.edu/dataset/495/ppg+dalia  -> keep only "
             "S*/S*_E4.zip, then set DALIA_ROOT",
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="fatigue-data",
        description="Download or locate the corpora used for validation.")
    p.add_argument("--download", choices=["ppgbp"],
                   help="fetch a corpus (only PPG-BP is freely downloadable)")
    p.add_argument("--dest", help="where to put it")
    p.add_argument("--force", action="store_true", help="re-download")
    p.add_argument("--status", action="store_true", help="report what was found")
    p.add_argument("--tables", help="list the survey tables under a directory "
                                    "(for FatigueSet, whose schema varies)")
    args = p.parse_args(argv)

    if args.download == "ppgbp":
        download_ppgbp(args.dest, force=args.force)

    if args.tables:
        tables = discover_tables(args.tables)
        if not len(tables):
            print(f"no non-sensor tables under {args.tables}")
        else:
            print(tables.drop(columns=["all_cols"], errors="ignore").to_string(index=False))
            print("\nfull column lists:")
            for _, row in tables.head(6).iterrows():
                print(f"\n  {row['path']}"
                      + (f" [{row['sheet']}]" if row.get("sheet") else ""))
                print(f"    {row.get('all_cols', '')}")
        return 0

    if args.status or not args.download:
        print("corpora:")
        roots = resolve_all(verbose=True)
        missing = [k for k, v in roots.items() if v is None]
        if missing:
            print("\nto get the missing ones:")
            for k in missing:
                print(f"  {k:<11} {HINTS[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
