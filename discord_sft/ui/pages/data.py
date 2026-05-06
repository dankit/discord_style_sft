"""Data pipeline: Ingest, Curate, Build SFT (single sidebar destination)."""

from __future__ import annotations

import streamlit as st

from discord_sft.ui.pages.build_sft import render_build_sft
from discord_sft.ui.pages.curate import render_curate
from discord_sft.ui.pages.ingest import render_ingest


def render_data() -> None:
    st.title("Data")
    st.caption("Ingest exports, curate sessions, then build ShareGPT SFT JSONL.")
    t1, t2, t3 = st.tabs(["Ingest", "Curate", "Build SFT"])
    with t1:
        render_ingest()
    with t2:
        render_curate()
    with t3:
        render_build_sft()
