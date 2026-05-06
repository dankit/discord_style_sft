"""Benchmark catalog tab."""

from __future__ import annotations

import streamlit as st

from discord_sft.evals.benchmarks import BENCHMARKS


def render_catalog_tab() -> None:
    st.subheader("Benchmark catalog")
    st.caption("What each benchmark in the registry evaluates.")
    for key, bspec in BENCHMARKS.items():
        with st.container(border=True):
            st.markdown(f"**{key}**  — `{bspec.modality}` / `{bspec.category}`")
            st.write(bspec.description)
            if bspec.task != "__persona__":
                st.caption(f"lmms-eval task: `{bspec.task}` — primary metric: `{bspec.metric}`")
    st.info(
        "Not yet evaluated: long-context agentic tool-calling (WebArena / OSWorld / "
        "GAIA) and multilingual capability (MMLU-Pro / MGSM / Belebele / C-Eval). "
        "Both are natural extensions to the registry."
    )
