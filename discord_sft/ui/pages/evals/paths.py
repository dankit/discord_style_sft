"""Filesystem roots for eval UI."""

from __future__ import annotations

from pathlib import Path

from discord_sft.ui.common import session_work_dir


def evals_root() -> Path:
    return session_work_dir() / "evals"
