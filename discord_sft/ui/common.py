from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from discord_sft.data_prep.curate import CurateReport, Session, record_to_session

STAGES = [
    ("Home", "🏠"),
    ("Data", "📦"),
    ("Analyze", "🔬"),
    ("Train", "🎯"),
    ("Evaluate", "📊"),
]


def repo_root() -> Path:
    """Repository root (parent of the ``discord_sft`` package directory)."""
    # This file: <repo>/discord_sft/ui/common.py
    return Path(__file__).resolve().parents[2]


def default_discord_export_root() -> str:
    """Default sidebar **Raw Discord export root**, relative to the repository."""
    return "discord_messages"


def default_work_dir() -> str:
    """Default sidebar **Working directory**, relative to the repository."""
    return "out"


def resolve_work_dir(raw: str | None) -> Path:
    """Resolve the sidebar working directory for filesystem access.

    **Relative** paths (e.g. ``out``) are anchored at **repository root**, not the
    process current working directory, so artifacts match ``<repo>/out/...`` even
    when Streamlit was started from another folder.
    """
    s = (raw or "").strip()
    if not s:
        return (repo_root() / "out").resolve()
    p = Path(s).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (repo_root() / p).resolve()


def session_work_dir() -> Path:
    """Working directory from session state, resolved for disk checks and defaults."""
    return resolve_work_dir(str(st.session_state.get("work_dir", "")))


def session_work_path(*parts: str) -> str:
    """Display path under the configured work dir, preserving relative input."""
    raw = str(st.session_state.get("work_dir") or default_work_dir()).strip()
    base = Path(raw).expanduser() if raw else Path(default_work_dir())
    return str(base.joinpath(*parts))


def resolve_repo_path(raw: str | Path | None) -> Path:
    """Resolve a user-entered path, anchoring relative values at repo root."""
    s = str(raw or "").strip()
    if not s:
        return repo_root().resolve()
    path = Path(s).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root() / path).resolve()


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("source_dir", default_discord_export_root())
    ss.setdefault("work_dir", default_work_dir())


def render_sidebar() -> str:
    with st.sidebar:
        st.title("discord-sft")
        st.caption("Discrub export -> multi-persona SFT corpus")

        stage = st.radio(
            "Stage",
            [f"{emoji} {name}" for name, emoji in STAGES],
            key="stage_radio",
            label_visibility="collapsed",
        )
        active_name = stage.split(" ", 1)[1]

        st.divider()
        st.subheader("Workspace")
        st.session_state["source_dir"] = st.text_input(
            "Raw Discord export root",
            value=st.session_state["source_dir"],
            placeholder=default_discord_export_root(),
            help="Directory containing one subfolder per DM, each with Discrub JSON pages. "
            "Defaults to a ``discord_messages`` folder at the repository root.",
        )
        st.session_state["work_dir"] = st.text_input(
            "Working directory",
            value=st.session_state["work_dir"],
            help="Outputs land under this folder (``messages/``, ``curated/``, ``sft/``, …). "
            "Relative paths are resolved from the **repository root**, not the shell cwd.",
        )

        work = session_work_dir()
        st.caption(f"Resolved to: `{work}`")
    return active_name


def existing_dir(p: str) -> Path | None:
    if not p:
        return None
    path = resolve_repo_path(p)
    return path if path.exists() and path.is_dir() else None


def existing_file(p: str) -> Path | None:
    if not p:
        return None
    path = resolve_repo_path(p)
    return path if path.exists() and path.is_file() else None


def report_to_df(report: CurateReport) -> dict[str, int]:
    d = dataclasses.asdict(report)
    return {k: v for k, v in d.items() if isinstance(v, int)}


def ts(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


def load_sessions(path: Path) -> list[Session]:
    out: list[Session] = []
    skipped = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(record_to_session(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            skipped += 1
            if skipped <= 3:
                st.warning(f"Skipped malformed session row {line_no}: {e}")
    if skipped > 3:
        st.warning(f"Skipped {skipped - 3} additional malformed session rows.")
    return out
