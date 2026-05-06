"""Home / workspace status page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from discord_sft.data_prep.ingest import iter_folders
from discord_sft.evals.storage import list_runs
from discord_sft.ui import workspace_status as ws
from discord_sft.ui.common import existing_dir, session_work_dir


def _artifact_mtimetuple(work: Path) -> tuple[float, ...]:
    """Bust Streamlit cache when any default artifact changes on disk."""
    parts: list[float] = []
    for rel in (
        "curated/sessions.jsonl",
        "sft/train.jsonl",
        "sft/val.jsonl",
        "sft/profiles.json",
        "sft/balance_report.json",
        "curated/curate_report.json",
    ):
        p = work / rel
        parts.append(p.stat().st_mtime if p.is_file() else 0.0)
    md = work / "messages"
    parts.append(md.stat().st_mtime if md.is_dir() else 0.0)
    rd = work / "evals" / "runs"
    if rd.is_dir():
        jsons = list(rd.glob("*.json"))
        parts.append(max((f.stat().st_mtime for f in jsons), default=0.0))
    else:
        parts.append(0.0)
    return tuple(parts)


@st.cache_data(ttl=45, show_spinner="Scanning workspace…")
def _cached_workspace_snapshot(work_str: str, _sig: tuple[float, ...]) -> dict:
    # _sig is part of the cache key so edits on disk invalidate the snapshot.
    return ws.build_snapshot(Path(work_str))


def _render_recent_eval_runs(work: Path) -> None:
    runs = list_runs(work / "evals")
    if not runs:
        return
    st.markdown("### Recent eval runs")
    tail = list(reversed(runs))[:5]
    for r in tail:
        rid = r.get("run_id", "")
        created = r.get("created_utc", "")
        label = r.get("label") or "—"
        st.caption(f"`{rid}` · {created} · label: {label}")


def _render_home_snapshot(snap: dict, work: Path) -> None:
    pipe = snap.get("pipeline") or {}
    frac = float(pipe.get("progress_fraction") or 0.0)
    status_title = pipe.get("status_title") or "Workspace not started"
    status_body = pipe.get("status_body") or "Start with Ingest once your Discord export path is set."

    st.markdown("### Status")
    st.progress(frac, text=f"{pipe.get('completed_consecutive', 0)}/{pipe.get('total', 5)} pipeline steps ready")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"**{status_title}**")
        st.caption(status_body)
    with c2:
        st.metric("Workspace", f"{int(frac * 100)}% ready")

    st.markdown("### Key Artifacts")
    src = existing_dir(st.session_state.get("source_dir") or "")
    source_count = 0
    if src:
        source_count = len(list(iter_folders(src)))
    msg = snap.get("messages") or {}
    sess = snap.get("sessions") or {}
    train = snap.get("train") or {}
    profiles = snap.get("profiles") or {}
    evals = snap.get("evals") or {}

    cards = [
        ("Source", f"{source_count:,} DMs" if source_count else "Not set", bool(source_count)),
        ("Messages", f"{msg.get('dm_folders', 0):,} DMs" if msg else "Missing", bool(msg)),
        ("Sessions", f"{sess.get('sessions', 0):,}" if sess else "Missing", bool(sess)),
        ("SFT", f"{train.get('samples', 0):,} samples" if train else "Missing", bool(train)),
        ("Profiles", f"{profiles.get('personas', 0):,} personas" if profiles else "Missing", bool(profiles)),
        ("Evals", f"{evals.get('runs', 0):,} runs" if evals.get("runs") else "None yet", bool(evals.get("runs"))),
    ]
    cols = st.columns(3)
    for idx, (label, value, ready) in enumerate(cards):
        with cols[idx % 3]:
            st.metric(("OK " if ready else "TODO ") + label, value)

    _render_recent_eval_runs(work)

    st.markdown("### Next Step")
    action = _home_next_action(snap, bool(source_count))
    st.info(action)

    st.markdown("### Quick Links")
    st.caption(
        "Typical flow: **Data** (Ingest → Curate → Build SFT) → **Analyze** "
        "(Fingerprint, tokens/templates, Browse corpus) → **Train** → "
        "**Evaluate** (Run → Compare / Persona outputs / Timeline)."
    )


def _home_next_action(snap: dict, has_source: bool) -> str:
    if not has_source:
        return "Set Raw Discord export root in the sidebar. Relative paths like `discord_messages` are resolved from the repo root."
    if not snap.get("messages"):
        return "Open **Data → Ingest** to create `out/messages/` from your Discord export."
    if not snap.get("sessions"):
        return "Open **Data → Curate** to create `out/curated/sessions.jsonl`."
    if not snap.get("train"):
        return "Open **Data → Build SFT** to create `out/sft/train.jsonl` and `out/sft/val.jsonl`."
    if not snap.get("profiles"):
        return "Open **Analyze → Fingerprint** to create `out/sft/profiles.json` for persona-aware evals."
    if not (snap.get("evals") or {}).get("runs"):
        return "Use **Train** or **Evaluate → Run** when ready; commands are shown before each run."
    return (
        "Core artifacts are in place. Inspect data under **Analyze → Browse**; "
        "persona eval dumps and compare-runs voting live under **Evaluate → Persona**."
    )


def render_home() -> None:
    st.title("discord-sft")
    st.caption("Workspace status for turning Discord exports into SFT data, profiles, and eval runs.")

    work = session_work_dir()
    st.caption(f"Working directory: `{st.session_state.get('work_dir', 'out')}`")
    sig = _artifact_mtimetuple(work)
    snap = _cached_workspace_snapshot(str(work.resolve()), sig)
    _render_home_snapshot(snap, work)
