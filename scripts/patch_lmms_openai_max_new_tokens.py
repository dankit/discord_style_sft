#!/usr/bin/env python3
"""Remove lmms-eval's silent cap on max_new_tokens in the OpenAI chat backend.

The default shared vLLM path in discord-sft uses ``lmms_eval --model openai`` against
a local vLLM OpenAPI server. Some lmms-eval releases apply::

    min(request_gen_kwargs.get("max_new_tokens", ...), 4096)

so values above 4096 are never sent, regardless of ``--gen_kwargs``.

This script patches the *installed* package (site-packages) idempotently, same
idea as the task.py postprocess hotfix in ``setup_gh200_evals.sh``.

Upstream: if this bit-rots, report or fix in
https://github.com/EvolvingLMMs-Lab/lmms-eval (remove the silent 4096 cap, tie
limits to server ``max_model_len``, or log when clamping).

Optional: set ``LMMS_EVAL_OPENAI_MAX_NEW_TOKENS`` to a positive integer to replace
the hardcoded 4096 with ``min(requested, env_value)`` instead of removing the
outer cap entirely (safety for accidental huge generations).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _openai_path() -> Path:
    import lmms_eval  # type: ignore[import-untyped]

    p = Path(lmms_eval.__file__).resolve().parent / "models" / "chat" / "openai.py"
    if not p.is_file():
        raise SystemExit(f"openai.py not found at {p} (unexpected lmms-eval layout)")
    return p


# Match min(request_gen_kwargs.get("max_new_tokens", DEFAULT), 4096) or single quotes.
_RE_MIN_4096 = re.compile(
    r"min\(\s*"
    r"request_gen_kwargs\.get\(\s*"
    r"(['\"])max_new_tokens\1\s*,\s*(\d+)\s*\)\s*"
    r",\s*4096\s*\)",
    re.MULTILINE,
)


def patch_text(
    text: str,
    *,
    ceiling: int | None,
) -> tuple[str, int]:
    """Return (new_text, n_replacements)."""

    def _replace_min(m: re.Match[str]) -> str:
        q = m.group(1)
        default = m.group(2)
        inner = f"request_gen_kwargs.get({q}max_new_tokens{q}, {default})"
        if ceiling is not None and ceiling > 0:
            return f"min({inner}, {ceiling})"
        return inner

    new_text, n = _RE_MIN_4096.subn(_replace_min, text)
    return new_text, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not write the file",
    )
    ap.add_argument(
        "--openai-path",
        type=Path,
        default=None,
        help="Override path to openai.py (default: from import lmms_eval)",
    )
    args = ap.parse_args()

    ceiling_env = os.environ.get("LMMS_EVAL_OPENAI_MAX_NEW_TOKENS", "").strip()
    ceiling: int | None
    if not ceiling_env:
        ceiling = None
    else:
        try:
            ceiling = int(ceiling_env)
        except ValueError:
            print(
                f"error: LMMS_EVAL_OPENAI_MAX_NEW_TOKENS must be an int, got {ceiling_env!r}",
                file=sys.stderr,
            )
            return 2
        if ceiling <= 0:
            ceiling = None

    path = args.openai_path or _openai_path()
    text = path.read_text(encoding="utf-8")
    new_text, n = patch_text(text, ceiling=ceiling)

    if n == 0:
        # Idempotent: already patched or upstream changed
        if "4096" not in text and "max_new_tokens" in text:
            print(f"no min(..., 4096) pattern found; likely already patched: {path}")
        else:
            print(
                f"no min(..., 4096) pattern matched; upstream may have changed: {path}",
            )
        return 0

    print(f"patch_lmms_openai_max_new_tokens: {n} replacement(s) in {path}")
    if args.dry_run:
        return 0

    path.write_text(new_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
