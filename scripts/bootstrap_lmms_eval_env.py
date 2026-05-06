#!/usr/bin/env python3
"""Idempotent post-install for lmms-eval: task overlay, hotfixes, NLTK, health check.

Run from the repo root with an ``[evals]`` venv (must have PyTorch unless you pass
``--skip-health``), e.g.::

    source .venv/bin/activate
    python scripts/bootstrap_lmms_eval_env.py

    # If using ``uv run`` you must include eval extras so ``torch`` is present::
    uv run --extra evals --extra lang python scripts/bootstrap_lmms_eval_env.py

Environment (optional), aligned with ``scripts/setup_gh200_evals.sh``:

* ``LMMS_SRC_DIR`` — git clone path (default: ``<repo>/../lmms-eval-main``).
* ``LMMS_REPO_URL`` — clone URL.
* ``LMMS_REF`` — commit to checkout (default matches :data:`discord_sft.evals.env_health.DEFAULT_LMMS_REF`).
* ``EVAL_GIT_TOKEN`` — optional token for HTTPS clone.

Re-run after ``uv sync`` or any reinstall that overwrites ``site-packages``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip clone/fetch/checkout and task overlay (only patches + NLTK + health)",
    )
    ap.add_argument(
        "--skip-nltk",
        action="store_true",
        help="Skip NLTK punkt download",
    )
    ap.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip GPU stack import smoke test",
    )
    ap.add_argument(
        "--no-vllm-health",
        action="store_true",
        help=(
            "Health check skips vLLM (CPU-only, or Linux aarch64 without a vLLM wheel). "
            "Default runs the full stack check including vllm._C when --skip-health is off."
        ),
    )
    ap.add_argument("--quiet", action="store_true", help="Less stdout from NLTK etc.")
    args = ap.parse_args()

    try:
        from discord_sft.evals import env_health
    except ImportError as e:
        print(
            "error: install the project in editable mode with eval extras, e.g.\n"
            "  uv sync --active --extra evals --extra lang\n"
            "(use ``--active`` when your venv is not the project default ``.venv``; same venv must be activated.)\n"
            f"({e})",
            file=sys.stderr,
        )
        return 1

    scripts_dir = Path(__file__).resolve().parent
    repo = _repo_root()

    if not args.skip_git:
        src = Path(
            os.environ.get("LMMS_SRC_DIR", str(repo.parent / "lmms-eval-main"))
        ).expanduser()
        repo_url = os.environ.get("LMMS_REPO_URL", env_health.DEFAULT_LMMS_REPO_URL)
        ref = os.environ.get("LMMS_REF", env_health.DEFAULT_LMMS_REF)
        token = os.environ.get("EVAL_GIT_TOKEN", "").strip() or None

        if not args.quiet:
            print(f"[bootstrap] lmms-eval clone: {src} @ {ref}")
        env_health.ensure_lmms_eval_clone(
            src_dir=src,
            repo_url=repo_url,
            ref=ref,
            git_token=token,
        )
        env_health.overlay_tasks_from_clone(src)
        if not args.quiet:
            print(f"[bootstrap] task overlay complete -> {env_health.lmms_eval_site_dir() / 'tasks'}")

    wrote, msg = env_health.patch_task_py_none_safe_strip()
    if not args.quiet or wrote:
        print(f"[bootstrap] task.py: {msg}")

    rc = env_health.patch_openai_max_new_tokens_via_script(scripts_dir)
    if rc != 0:
        return rc

    rc = env_health.patch_openai_disable_thinking_via_script(scripts_dir)
    if rc != 0:
        return rc

    if not args.skip_nltk:
        rc = env_health.download_nltk_ifeval(quiet=args.quiet)
        if rc != 0:
            return rc

    if args.skip_health:
        return 0

    require_vllm = not args.no_vllm_health

    if not args.quiet:
        print(f"[bootstrap] health check (python: {sys.executable}):")
    return env_health.run_gpu_stack_health(require_vllm=require_vllm, qwen_config_smoke=True)


if __name__ == "__main__":
    raise SystemExit(main())
