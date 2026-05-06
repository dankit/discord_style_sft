"""Evals multi-tab Streamlit page."""

from __future__ import annotations

import streamlit as st

from discord_sft.ui.pages.evals.catalog import render_catalog_tab
from discord_sft.ui.pages.evals.compare import render_compare_tab
from discord_sft.ui.pages.evals.persona_browser import render_persona_generations_tab
from discord_sft.ui.pages.evals.run import render_run_tab
from discord_sft.ui.pages.evals.style_rank import render_style_rank_tab
from discord_sft.ui.pages.evals.timeline import render_timeline_tab


def render_evals() -> None:
    st.title("Evaluate")
    st.caption(
        "Run benchmarks via lmms-eval, save each run as one JSON, diff metric runs, "
        "inspect persona generations, **style-rank** LoRA comparisons (pairwise + fingerprint), "
        "compare checkpoints, and plot timelines."
    )
    tabs = st.tabs(["Run", "Compare", "Persona", "Style rank", "Timeline", "Benchmarks"])
    with tabs[0]:
        render_run_tab()
    with tabs[1]:
        render_compare_tab()
    with tabs[2]:
        render_persona_generations_tab()
    with tabs[3]:
        render_style_rank_tab()
    with tabs[4]:
        render_timeline_tab()
    with tabs[5]:
        render_catalog_tab()


__all__ = ["render_evals"]
