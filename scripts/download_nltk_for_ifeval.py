#!/usr/bin/env python3
"""Download NLTK data required for lmms-eval IFEval postprocessing.

IFEval calls ``nltk.word_tokenize`` / sentence tokenization, which needs the
``punkt_tab`` package on NLTK 3.9+ (and ``punkt`` on older stacks). These are
not shipped inside the ``nltk`` wheel; run this once per environment.

Usage (from repo root, venv active):

    uv run python scripts/download_nltk_for_ifeval.py
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Pass quiet=True to nltk.download",
    )
    args = ap.parse_args()
    try:
        import nltk
    except ImportError:
        print(
            "error: nltk is not installed. Install eval extras: uv sync --extra evals",
            file=sys.stderr,
        )
        return 1

    names = ("punkt_tab", "punkt")
    for name in names:
        nltk.download(name, quiet=args.quiet)
    if not args.quiet:
        print("NLTK resources for IFEval:", ", ".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
