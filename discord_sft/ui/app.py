"""Streamlit dashboard for the discord-sft pipeline."""
from __future__ import annotations

import streamlit as st

from discord_sft.ui.common import init_state, render_sidebar
from discord_sft.ui.pages import render_analyze, render_data, render_evals, render_home
from discord_sft.ui.pages.train import render_train


def main() -> None:
    st.set_page_config(
        page_title="discord-sft",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    active_name = render_sidebar()
    pages = {
        "Home": render_home,
        "Data": render_data,
        "Analyze": render_analyze,
        "Train": render_train,
        "Evaluate": render_evals,
    }
    pages[active_name]()


if __name__ == "__main__":
    main()
