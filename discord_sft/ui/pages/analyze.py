"""Analyze: fingerprints, tokenizer/template checks, browse artifacts."""

from __future__ import annotations

import streamlit as st

from discord_sft.ui.pages.browse import render_browse
from discord_sft.ui.pages.fingerprint import render_fingerprint
from discord_sft.ui.pages_stats import render_stats


def render_analyze() -> None:
    st.title("Analyze")
    st.caption("Style profiles, token/template checks, and browsing curated data on disk.")
    t1, t2, t3 = st.tabs(["Fingerprint", "Tokens and Templates", "Browse"])
    with t1:
        render_fingerprint()
    with t2:
        render_stats()
    with t3:
        render_browse()
