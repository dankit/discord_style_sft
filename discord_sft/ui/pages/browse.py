"""Browse curated sessions, SFT samples, and fingerprint profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from discord_sft.data_prep.sft import persona_perspective_counts, read_samples
from discord_sft.ui.common import load_sessions, session_work_dir, ts


def _safe_float(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _render_profile_persona(prof: dict[str, Any]) -> None:
    pid = prof.get("persona_id", "—")
    st.markdown(f"**Persona id** `{pid}`")

    length = prof.get("length") or {}
    n = length.get("n", 0)
    mean_w = _safe_float(length.get("mean_words"))
    p50 = length.get("p50")
    p95 = length.get("p95")

    lc = _safe_float(prof.get("lowercase_start_rate"))
    burst = _safe_float(prof.get("burst_size"))
    emj = prof.get("emoji") or {}
    uni_e = _safe_float(emj.get("unicode_per_turn"))
    cust_e = _safe_float(emj.get("custom_per_turn"))

    st.markdown("##### Assistant-turn statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Turns scored", f"{int(n):,}" if n else "0")
    m2.metric("Mean words / turn", f"{mean_w:.1f}" if mean_w is not None else "—")
    m3.metric("Median words", str(p50) if p50 is not None else "—")
    m4.metric("p95 words", str(p95) if p95 is not None else "—")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Lowercase-start rate", f"{100 * lc:.1f}%" if lc is not None else "—")
    m6.metric("Avg lines / turn (burst proxy)", f"{burst:.2f}" if burst is not None else "—")
    m7.metric("Unicode emoji / turn", f"{uni_e:.2f}" if uni_e is not None else "—")
    m8.metric("Custom :emoji: / turn", f"{cust_e:.2f}" if cust_e is not None else "—")

    st.caption(
        "N-grams below are ranked by log-odds *z* vs the global corpus (higher = more distinctive for this persona)."
    )
    tf = prof.get("top_fillers") or {}
    gram_tabs = st.tabs(["1-grams", "2-grams", "3-grams"])
    for tab, key, label in zip(
        gram_tabs,
        ("1gram", "2gram", "3gram"),
        ("single token", "two-word phrase", "three-word phrase"),
    ):
        with tab:
            rows = tf.get(key) or []
            if not rows:
                st.info(f"No {label} fillers recorded.")
            else:
                table = [
                    {
                        "token": r.get("token", ""),
                        "z (distinctive)": r.get("z"),
                        "count": r.get("count"),
                    }
                    for r in rows
                    if isinstance(r, dict)
                ]
                st.dataframe(
                    table,
                    width="stretch",
                    hide_index=True,
                    height=min(420, 28 * (len(table) + 2)),
                )

    with st.expander("Raw JSON (debug)", expanded=False):
        st.json(prof)


def _render_sample_preview(samp: Any) -> None:
    meta = dict(samp.meta or {})
    st.markdown("##### Meta")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**persona** `{meta.get('persona_id', '')}` · {meta.get('persona_name', '')}")
    c2.markdown(f"**session** `{meta.get('session_id', '')}`")
    c3.markdown(f"**folder** `{meta.get('folder', '')}`  \n**turns** {meta.get('num_turns', '—')}")
    ts1, ts2 = meta.get("first_ts"), meta.get("last_ts")
    if ts1 or ts2:
        st.caption(f"{ts(ts1) if ts1 else '—'} → {ts(ts2) if ts2 else '—'}")

    with st.expander("System prompt", expanded=False):
        st.markdown(samp.system or "_(empty)_")

    st.markdown("##### Conversation (ShareGPT roles)")
    for turn in samp.conversations or []:
        role = turn.get("from", "user")
        body = turn.get("value", "") or ""
        st_chat = "assistant" if role == "assistant" else "user"
        with st.chat_message(st_chat):
            st.markdown(body)

    with st.expander("Raw JSON (debug)", expanded=False):
        st.json(samp.to_json())


def render_browse() -> None:
    st.caption("Sessions, SFT JSONL, and profiles from your working directory.")

    work = session_work_dir()
    default_sessions = work / "curated" / "sessions.jsonl"
    train_path = work / "sft" / "train.jsonl"
    tabs = st.tabs(["Sessions", "SFT samples", "Profiles"])

    with tabs[0]:
        sess_str = st.text_input(
            "sessions.jsonl path",
            value=str(default_sessions),
            help="Defaults under working directory / curated.",
            key="browse_sessions_path",
        )
        sessions_path = Path(sess_str)
        if not sessions_path.exists():
            st.info("No file at that path — run **Curate** (Data tab) or point to an existing sessions.jsonl.")
        else:
            sessions = load_sessions(sessions_path)
            st.caption(f"{len(sessions):,} sessions.")
            folders = sorted({s.folder for s in sessions})
            f_pick = st.multiselect("Folders", folders, default=folders, key="browse_folders")
            filtered = [s for s in sessions if s.folder in f_pick]
            st.caption(f"Showing {len(filtered):,} after filter.")
            idx = st.slider("Session index", 0, max(0, len(filtered) - 1), 0, key="browse_sess_idx") if filtered else 0
            if filtered:
                s = filtered[idx]
                st.markdown(
                    f"**{s.id}**  ·  {s.folder}  ·  "
                    f"{ts(s.first_ts.isoformat())} → {ts(s.last_ts.isoformat())}  ·  "
                    f"{len(s.turns)} turns  ·  authors: {', '.join(sorted(s.authors()))}"
                )
                with st.expander("Full transcript (curated turns)", expanded=True):
                    for t in s.turns:
                        st.markdown(
                            f"**{t.author_name}** `{t.author_id}` · "
                            f"{ts(t.start_ts.isoformat())} → {ts(t.end_ts.isoformat())}"
                        )
                        st.markdown(t.text or "_(empty)_")
                        st.divider()

    with tabs[1]:
        val_path = work / "sft" / "val.jsonl"
        corpus = st.radio(
            "Corpus",
            ["train.jsonl", "val.jsonl"],
            horizontal=True,
            key="browse_sft_corpus",
        )
        path = train_path if corpus == "train.jsonl" else val_path
        if not path.exists():
            st.info(f"No {corpus} yet." + (" Build SFT (Data tab) to create it." if corpus == "train.jsonl" else ""))
        else:
            samples = read_samples(path)
            st.caption(f"{len(samples):,} samples in {path.name}.")
            st.subheader("Per-persona coverage (their POV)")
            pov_rows = persona_perspective_counts(samples)
            st.table(pov_rows)
            if samples:
                sidx = st.slider("Sample index", 0, len(samples) - 1, 0, key="browse_sample_idx")
                samp = samples[sidx]
                st.subheader("Selected sample")
                _render_sample_preview(samp)

    with tabs[2]:
        prof_path = work / "sft" / "profiles.json"
        if not prof_path.exists():
            st.info("No profiles yet. Run **Fingerprint** (Analyze tab).")
        else:
            doc = json.loads(prof_path.read_text(encoding="utf-8"))
            personas: dict[str, Any] = doc.get("personas") or {}
            if not personas:
                st.warning("profiles.json has no personas.")
            else:
                keys = list(personas.keys())
                pid = st.selectbox("Persona (Discord author id)", keys, key="browse_prof_pid")
                if pid:
                    _render_profile_persona(personas[pid])
