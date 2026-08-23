"""``python -m fatigueppg <command>`` -- a front door for the four entry points."""
from __future__ import annotations

import sys

COMMANDS = {
    "selfcheck": "validate every stage against ground truth",
    "infer": "fatigue index from a raw PPG recording",
    "extract": "features for a whole cohort",
    "train": "fit Equation (9) on your own labelled cohort",
    "data": "download or locate the public corpora",
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: python -m fatigueppg <command> [options]\n")
        for name, what in COMMANDS.items():
            print(f"  {name:<11} {what}")
        print("\nrun 'python -m fatigueppg <command> --help' for the options")
        return 0 if argv else 2

    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}; try one of {', '.join(COMMANDS)}",
              file=sys.stderr)
        return 2

    if cmd == "selfcheck":
        from .selfcheck import main as run
    elif cmd == "infer":
        from .infer import main as run
    elif cmd == "extract":
        from .extract import main as run
    elif cmd == "train":
        from .train import main as run
    else:
        from .fetch import main as run
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
