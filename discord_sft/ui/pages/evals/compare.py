"""Compare saved eval runs tab."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from discord_sft.evals.compare import compare_runs, render_comparison
from discord_sft.evals.storage import list_runs

from discord_sft.ui.pages.evals.paths import evals_root
from discord_sft.ui.pages.evals.run_annotations import (
    format_compare_column_header,
    format_eval_run_label,
    get_run_annotation,
    streamlit_annotations,
)

_SESSION_REPORT = "eval_compare_report"
_SESSION_SIG = "eval_compare_sig"


def _signature(
    picked: list[str], baseline: str, metrics_raw: str, include_persona: bool
) -> tuple[Any, ...]:
    return (tuple(picked), baseline, metrics_raw.strip(), include_persona)


def _run_preview_rows(
    runs: list[dict[str, Any]],
    *,
    annotations: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in runs:
        m = r.get("model")
        name = ""
        if isinstance(m, dict):
            name = str(m.get("name_or_path") or m.get("path") or "")
        rid = str(r.get("run_id", ""))
        alias = get_run_annotation(annotations, rid).get("alias") or ""
        out.append(
            {
                "run_id": rid,
                "alias": alias,
                "label": r.get("label") or "",
                "model": name,
                "baseline": "yes" if r.get("is_baseline") else "",
            }
        )
    return out


def _display_dataframe(
    df: Any,
    run_ids: list[str],
    *,
    annotations: dict[str, Any],
    yaml_labels: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Return a copy with forgetting columns scaled to 0–100 for display, plus column_config."""
    import pandas as pd

    def _pct(x: Any) -> float | None:
        if x is None or not pd.notna(x):
            return None
        return float(x) * 100.0

    disp = df.copy()
    col_cfg: dict[str, Any] = {
        "metric": st.column_config.TextColumn("metric", width="large"),
    }
    for c in disp.columns:
        if c == "metric":
            continue
        if c.startswith("forgetting__"):
            disp[c] = disp[c].apply(_pct)
            rid = c.removeprefix("forgetting__")
            title = f"forget % · {format_compare_column_header(rid, yaml_label=yaml_labels.get(rid), annotations=annotations)}"
            col_cfg[c] = st.column_config.NumberColumn(
                title,
                format="%.1f%%",
                help=f"(baseline − run) / baseline · {rid}",
            )
        elif c.startswith("delta__"):
            rid = c.removeprefix("delta__")
            title = f"Δ · {format_compare_column_header(rid, yaml_label=yaml_labels.get(rid), annotations=annotations)}"
            col_cfg[c] = st.column_config.NumberColumn(
                title,
                format="%.4f",
                help=f"run − baseline · {rid}",
            )
        elif c in run_ids:
            title = format_compare_column_header(
                c, yaml_label=yaml_labels.get(c), annotations=annotations
            )
            col_cfg[c] = st.column_config.NumberColumn(title, format="%.4f", help=c)
        else:
            col_cfg[c] = st.column_config.NumberColumn(c, format="%.4f")
    return disp, col_cfg


def _top_metrics_by_abs_delta(df: Any, _run_ids: list[str], _baseline_id: str | None, n: int) -> list[str]:
    import pandas as pd

    delta_cols = [c for c in df.columns if c.startswith("delta__")]
    if not delta_cols:
        return df["metric"].astype(str).tolist()[: max(1, n)]
    sub = df[["metric"] + delta_cols].copy()
    for c in delta_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["_score"] = sub[delta_cols].abs().max(axis=1)
    sub = sub.sort_values("_score", ascending=False)
    return sub["metric"].astype(str).tolist()[: max(1, n)]


def _top_metrics_by_run_spread(df: Any, run_ids: list[str], n: int) -> list[str]:
    """Rank metrics by how much they differ across runs (no baseline / no Δ columns)."""
    import pandas as pd

    cols = [c for c in run_ids if c in df.columns]
    if not cols:
        return df["metric"].astype(str).tolist()[: max(1, n)]
    sub = df[["metric"] + cols].copy()
    for c in cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["_score"] = sub[cols].std(axis=1, ddof=0)
    sub = sub.sort_values("_score", ascending=False, na_position="last")
    return sub["metric"].astype(str).tolist()[: max(1, n)]


def _top_metrics_for_summary(df: Any, run_ids: list[str], baseline_id: str | None, n: int) -> list[str]:
    """Pick a small set of metrics for compact text summaries."""
    has_delta = any(c.startswith("delta__") for c in df.columns)
    if baseline_id and has_delta:
        return _top_metrics_by_abs_delta(df, run_ids, baseline_id, n)
    return _top_metrics_by_run_spread(df, run_ids, n)


def _text_ascii_delta_bars(df: Any, names: list[str], *, bar_width: int = 36) -> str:
    """Monospace lines: metric, run, Δ, bar proportional to |Δ| within this slice."""
    import pandas as pd

    delta_cols = [c for c in df.columns if c.startswith("delta__")]
    if not delta_cols:
        return ""
    sub = df[df["metric"].isin(names)][["metric"] + delta_cols].copy()
    for c in delta_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    max_abs = 0.0
    for c in delta_cols:
        s = sub[c].abs()
        if s.notna().any():
            max_abs = max(max_abs, float(s.max()))
    if max_abs <= 0:
        max_abs = 1e-12

    lines: list[str] = [
        "ASCII bars (length ∝ |Δ| within this top-N slice; not comparable across slices).",
        "",
    ]
    for _, row in sub.iterrows():
        m = str(row["metric"])
        for dc in delta_cols:
            v = row[dc]
            if v is None or not pd.notna(v):
                continue
            fv = float(v)
            rid = dc.removeprefix("delta__")
            nfill = int(bar_width * min(1.0, abs(fv) / max_abs))
            bar = "█" * nfill + "·" * (bar_width - nfill)
            sign = "+" if fv > 0 else ""
            m_disp = m if len(m) <= 52 else "…" + m[-51:]
            lines.append(f"{m_disp:<52}  {rid:<18}  {sign}{fv:>8.4f}  {bar}")
    return "\n".join(lines)


def _text_values_table(df: Any, names: list[str], run_ids: list[str]) -> str:
    import pandas as pd

    cols = ["metric"] + [c for c in run_ids if c in df.columns]
    if len(cols) < 2:
        return "(No per-run score columns.)\n"
    sub = df[df["metric"].isin(names)][cols].copy()
    for c in cols[1:]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.to_string(index=False, float_format=lambda x: f"{x:8.4f}")


def _text_delta_table(df: Any, names: list[str]) -> str:
    import pandas as pd

    delta_cols = [c for c in df.columns if c.startswith("delta__")]
    if not delta_cols:
        return "(No Δ columns — pick a baseline.)\n"
    cols = ["metric"] + delta_cols
    sub = df[df["metric"].isin(names)][cols].copy()
    for c in delta_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    return sub.to_string(index=False, float_format=lambda x: f"{x:8.4f}")


def render_compare_tab() -> None:
    ev_root = evals_root()
    runs = list_runs(ev_root)
    if len(runs) < 1:
        st.info("No runs saved yet — run at least one from the **Run** tab.")
        return
    ann_doc = streamlit_annotations(ev_root, st.session_state)
    options = {str(r["run_id"]): r for r in runs}
    run_ids_order = list(options.keys())

    def _pick_label(rid: str) -> str:
        row = options[rid]
        return format_eval_run_label(
            rid,
            yaml_label=row.get("label"),
            annotations=ann_doc,
        )

    c_left, c_right = st.columns([3, 1], vertical_alignment="top")
    with c_left:
        picked = st.multiselect(
            "Runs to compare",
            run_ids_order,
            default=run_ids_order[-min(3, len(run_ids_order)) :],
            format_func=_pick_label,
            key="eval_compare_runs",
            help="Display names include optional **aliases** from ``run_annotations.json`` (also used on Style rank / Persona).",
        )
    with c_right:
        if len(picked) < 1:
            st.empty()
        else:
            baseline = st.selectbox(
                "Baseline",
                picked,
                index=0,
                format_func=_pick_label,
                key="eval_compare_baseline",
            )

    if len(picked) < 1:
        st.info("Select at least one run.")
        return

    st.caption(
        "Unless you change it below, metric keys starting with ``persona.`` are omitted "
        "so benchmark diffs stay readable (use **Style rank** / **Persona** for persona work)."
    )

    with st.expander("Advanced / export", expanded=False):
        st.checkbox(
            "Include metrics whose keys start with ``persona.``",
            value=False,
            key="eval_compare_include_persona",
            help="When unchecked and the metric filter is blank, those keys are dropped from the comparison.",
        )
        st.text_input(
            "Metric filter (glob, comma-separated; blank = all)",
            value="",
            placeholder="ifeval.*, mmmu_val.mmmu_acc",
            key="eval_compare_metrics_glob",
        )
        fmt_preview = st.radio(
            "Preview in expander",
            ["table", "markdown", "json"],
            horizontal=True,
            key="eval_compare_fmt_preview",
        )

    do_compare = st.button("Compare", type="primary", key="eval_compare_go")

    if do_compare:
        globs_raw = st.session_state.get("eval_compare_metrics_glob", "")
        metrics = [m.strip() for m in globs_raw.split(",") if m.strip()] or None
        paths = [options[str(p)]["path"] for p in picked]
        baseline_path = options[str(baseline)]["path"]
        include_persona = bool(st.session_state.get("eval_compare_include_persona", False))
        report = compare_runs(
            paths,
            baseline=baseline_path,
            metrics=metrics,
            omit_persona_metrics=not include_persona,
        )
        st.session_state[_SESSION_REPORT] = report
        st.session_state[_SESSION_SIG] = _signature(picked, baseline, globs_raw, include_persona)

    report = st.session_state.get(_SESSION_REPORT)
    sig_stored = st.session_state.get(_SESSION_SIG)
    sig_now = _signature(
        picked,
        baseline,
        st.session_state.get("eval_compare_metrics_glob", ""),
        bool(st.session_state.get("eval_compare_include_persona", False)),
    )

    if report is None:
        return

    if sig_stored != sig_now:
        st.caption("Selections or filter changed — click **Compare** to refresh results.")

    run_ids = [r["run_id"] for r in report["runs"]]
    baseline_id = report.get("baseline_run_id")
    yaml_labels = {
        str(r.get("run_id", "")): str(r.get("label") or "").strip()
        for r in report["runs"]
        if r.get("run_id")
    }

    try:
        import pandas as pd
    except ImportError:
        st.warning("Install **pandas** (`discord-sft[ui]`) for sortable tables.")
        text = render_comparison(report, fmt="table")
        st.code(text, language="text")
        return

    st.caption("Main grid is sortable; summaries below are plain **monospace** text (no chart widgets).")

    preview = pd.DataFrame(_run_preview_rows(report["runs"], annotations=ann_doc))
    st.dataframe(
        preview,
        hide_index=True,
        width="stretch",
        column_config={
            "run_id": st.column_config.TextColumn("run_id", width="large"),
            "alias": st.column_config.TextColumn("alias"),
            "label": st.column_config.TextColumn("label"),
            "model": st.column_config.TextColumn("model", width="large"),
            "baseline": st.column_config.TextColumn("baseline"),
        },
    )

    rows = report["rows"]
    if not rows:
        st.info("No metrics in this comparison.")
        return

    df = pd.DataFrame(rows)
    disp_df, col_cfg = _display_dataframe(
        df,
        run_ids,
        annotations=ann_doc,
        yaml_labels=yaml_labels,
    )
    st.dataframe(disp_df, width="stretch", hide_index=True, column_config=col_cfg)

    with st.expander("Export / text preview", expanded=False):
        fmt_preview = st.session_state.get("eval_compare_fmt_preview", "table")
        ascii_t = render_comparison(report, fmt="table")
        md_t = render_comparison(report, fmt="markdown")
        json_t = render_comparison(report, fmt="json")
        if fmt_preview == "json":
            st.code(json_t, language="json")
        elif fmt_preview == "markdown":
            st.markdown(md_t)
        else:
            st.code(ascii_t, language="text")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button(
                "Download ASCII (.txt)",
                data=ascii_t.encode("utf-8"),
                file_name="eval_compare.txt",
                mime="text/plain",
                key="eval_compare_dl_txt",
            )
        with b2:
            st.download_button(
                "Download Markdown (.md)",
                data=md_t.encode("utf-8"),
                file_name="eval_compare.md",
                mime="text/markdown",
                key="eval_compare_dl_md",
            )
        with b3:
            st.download_button(
                "Download JSON",
                data=json.dumps(json.loads(json_t), indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="eval_compare.json",
                mime="application/json",
                key="eval_compare_dl_json",
            )

    st.subheader("Text summaries")
    top_n = st.slider(
        "How many metrics to include below (full table above is unchanged)",
        min_value=5,
        max_value=22,
        value=14,
        key="eval_compare_top_n",
        help="Ranked by |Δ vs baseline| or by spread across runs.",
    )
    chart_n = int(top_n)
    names = _top_metrics_for_summary(df, run_ids, baseline_id, chart_n)

    tab_v, tab_d = st.tabs(["Top scores (monospace)", "Top Δ + ASCII bars"])
    with tab_v:
        st.code(_text_values_table(df, names, run_ids), language="text")
    with tab_d:
        st.code(_text_delta_table(df, names), language="text")
        bar_block = _text_ascii_delta_bars(df, names)
        if bar_block:
            st.code(bar_block, language="text")
