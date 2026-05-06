"""Streamlit page renderers (split from legacy monolithic modules)."""

from __future__ import annotations

from discord_sft.ui.pages.analyze import render_analyze
from discord_sft.ui.pages.data import render_data
from discord_sft.ui.pages.evals import render_evals
from discord_sft.ui.pages.home import render_home

__all__ = [
    "render_home",
    "render_data",
    "render_analyze",
    "render_evals",
]
