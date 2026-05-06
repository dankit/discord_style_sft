#!/usr/bin/env python3
"""Optionally disable Qwen thinking in lmms-eval OpenAI chat requests.

The default shared vLLM path in discord-sft uses ``lmms_eval --model openai``.
Persona already sets ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``
for instruct presets, but lmms-eval task requests (e.g. IFEval) typically do not.

This script patches the *installed* lmms-eval ``openai.py`` idempotently by adding
an env-gated payload tweak in ``build_payload_for_index``:

``LMMS_EVAL_DISABLE_THINKING=1`` -> add
``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.

If that gate references ``os.environ`` without an ``import os`` line (manual edits,
old patch runs), the script inserts ``import os`` after ``from __future__`` or before
other imports.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _openai_path() -> Path:
    import lmms_eval  # type: ignore[import-untyped]

    p = Path(lmms_eval.__file__).resolve().parent / "models" / "chat" / "openai.py"
    if not p.is_file():
        raise SystemExit(f"openai.py not found at {p} (unexpected lmms-eval layout)")
    return p


_GATED_SNIPPET = 'if os.environ.get("LMMS_EVAL_DISABLE_THINKING", "").strip() == "1":'
_FORCED_SNIPPET = 'payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}'
_OS_IMPORT_PATTERN = re.compile(r"(?m)^import os\b")
_RE_PAYLOAD_BLOCK = re.compile(
    r"(\s*payload\s*=\s*\{\n"
    r"(?:.*\n)*?"
    r"\s*\"temperature\":\s*temperature,\n"
    r"\s*\}\n)",
    re.MULTILINE,
)


def ensure_import_os_for_gated_env(text: str) -> tuple[str, bool]:
    """Insert ``import os`` when gated ``os.environ`` logic is present but ``os`` is not imported."""
    if _GATED_SNIPPET not in text:
        return text, False
    if _OS_IMPORT_PATTERN.search(text):
        return text, False

    future = re.search(
        r"(?m)^(from __future__ import[^\n]+\n)(\s*\n)?",
        text,
    )
    if future:
        ins = future.end()
        updated = text[:ins] + "import os\n" + text[ins:]
        return updated, True

    top_import = re.search(r"(?m)^(import [^\n]+\n)", text)
    if top_import:
        idx = top_import.start()
        return text[:idx] + "import os\n" + text[idx:], True

    return "import os\n\n" + text, True


def patch_text(text: str) -> tuple[str, int]:
    """Inject env-gated ``extra_body`` when missing and ensure ``import os`` alongside it."""

    def _inject(m: re.Match[str]) -> str:
        block = m.group(1)
        indent = re.match(r"(\s*)payload", block).group(1)  # type: ignore[union-attr]
        if _FORCED_SNIPPET in text:
            forced = (
                f'{indent}payload["extra_body"] = '
                '{"chat_template_kwargs": {"enable_thinking": False}}\n'
            )
            gated = (
                f'{indent}if os.environ.get("LMMS_EVAL_DISABLE_THINKING", "").strip() == "1":\n'
                f'{indent}    payload["extra_body"] = '
                '{"chat_template_kwargs": {"enable_thinking": False}}\n'
            )
            return block.replace(forced, gated)
        return (
            block
            + f'{indent}if os.environ.get("LMMS_EVAL_DISABLE_THINKING", "").strip() == "1":\n'
            f'{indent}    payload["extra_body"] = '
            '{"chat_template_kwargs": {"enable_thinking": False}}\n'
        )

    work = text
    mutations = 0
    if _GATED_SNIPPET not in work:
        work, inject_hits = _RE_PAYLOAD_BLOCK.subn(_inject, work, count=1)
        mutations += inject_hits

    final, imported = ensure_import_os_for_gated_env(work)
    if imported:
        mutations += 1

    return final, mutations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Do not write file")
    ap.add_argument("--openai-path", type=Path, default=None, help="Override openai.py path")
    args = ap.parse_args()

    path = args.openai_path or _openai_path()
    text = path.read_text(encoding="utf-8")
    new_text, n = patch_text(text)

    if n == 0:
        if _GATED_SNIPPET in text:
            print(f"already patched for env-gated no-thinking extra_body: {path}")
        else:
            print(f"no payload pattern matched; upstream may have changed: {path}")
        return 0

    print(f"patch_lmms_openai_disable_thinking: {n} edit(s) in {path}")
    if args.dry_run:
        return 0
    path.write_text(new_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
