"""Timeline tab for eval scores across runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from discord_sft.evals.storage import list_runs, load_run

from discord_sft.ui.pages.evals.paths import evals_root


def render_timeline_tab() -> None:
    runs = list_runs(evals_root())
    if len(runs) < 2:
        st.info("Need at least 2 saved runs to draw a timeline.")
        return
    runs_docs = [load_run(Path(r["path"])) for r in runs]
    all_metrics: set[str] = set()
    for doc in runs_docs:
        for k in (doc.get("scores") or {}):
            all_metrics.add(k)
    picked = st.multiselect(
        "Metrics to plot",
        sorted(all_metrics),
        default=[k for k in sorted(all_metrics) if not k.startswith("persona.judge.")][:5],
    )
    if not picked:
        return
    try:
        import pandas as pd
    except ImportError:
        st.warning("Install `pandas` to render the timeline chart.")
        return
    rows = []
    for doc in runs_docs:
        row: dict[str, Any] = {"created_utc": doc.get("created_utc")}
        for k in picked:
            v = (doc.get("scores") or {}).get(k)
            if isinstance(v, (int, float)):
                row[k] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("created_utc")
    st.line_chart(df)
    with st.expander("Raw data"):
        st.dataframe(df)
