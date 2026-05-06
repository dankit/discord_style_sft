"""Ingest page."""

from __future__ import annotations

import streamlit as st

from discord_sft.data_prep.ingest import ingest_root, iter_folders
from discord_sft.ui.common import existing_dir, resolve_repo_path, session_work_path


def render_ingest() -> None:
    st.title("📥 Ingest")
    st.caption("Load Discrub JSON pages into sorted, deduplicated parquet or jsonl (one file per DM folder).")

    src = existing_dir(st.session_state["source_dir"])
    if src is None:
        st.warning("Set **Raw Discord export root** in the sidebar first.")
        return

    folders = list(iter_folders(src))
    st.caption(f"Discovered {len(folders)} DM folders.")
    with st.form("ingest_form"):
        with st.expander("Advanced: output path", expanded=False):
            out_dir = st.text_input(
                "Output directory",
                value=session_work_path("messages"),
                help="Defaults to `<work_dir>/messages`.",
            )
        fmt = st.radio("Format", ["parquet", "jsonl"], horizontal=True, index=1)
        submitted = st.form_submit_button("Run ingest", type="primary")
    if not submitted:
        return
    with st.spinner(f"Ingesting {len(folders)} folders..."):
        try:
            report = ingest_root(src, resolve_repo_path(out_dir), fmt=fmt)
        except ImportError as e:
            st.error(str(e))
            return
    st.success(f"Ingested {sum(report.values()):,} messages across {len(report)} folders.")
    st.bar_chart(report, horizontal=True)
    st.json(report, expanded=False)
