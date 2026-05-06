"""Render persona_generations.jsonl rows (transcript, reference, compare layout)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

from discord_sft.ui.persona_compare import MergedPersonaSample, badge_label, widget_key_digest

from discord_sft.ui.pages.evals.persona_verdicts import render_compare_verdict_buttons


def read_persona_generations_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Return (rows, error_message). Skips blank lines."""
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return [], str(e)
    for line_no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            return [], f"Invalid JSON at line {line_no}: {e}"
        if not isinstance(obj, dict):
            return [], f"Line {line_no}: expected JSON object, got {type(obj).__name__}"
        rows.append(obj)
    return rows, None


def truncate_display(s: Any, limit: int) -> str:
    t = "" if s is None else str(s)
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"


def persona_sample_title_md(row: dict[str, Any]) -> str:
    persona_name = str(row.get("persona_name") or "Persona")
    persona_id = str(row.get("persona_id") or "")
    title_bits = [persona_name]
    if persona_id:
        title_bits.append(f"`{persona_id}`")
    return "**Sample** — " + " · ".join(title_bits)


def render_persona_turns_chunk(row: dict[str, Any]) -> None:
    """Render transcript only (everything before generation)."""
    persona_name = str(row.get("persona_name") or "Persona")
    turns = row.get("context_turns")
    context_fallback = row.get("context")
    structured = isinstance(turns, list) and turns
    if structured:
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("from", "")).strip().lower()
            body = str(t.get("value", "")).strip("\n") if t.get("value") is not None else ""
            if role == "user":
                with st.chat_message("user", avatar="🗨️"):
                    st.markdown("**Partner**")
                    st.text(body if body else " ")
            elif role == "assistant":
                with st.chat_message("assistant", avatar="💬"):
                    st.markdown(f"**{persona_name}**")
                    st.text(body if body else " ")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**{role or 'turn'}**")
                    st.text(body if body else " ")
    elif context_fallback:
        with st.container(border=True):
            st.markdown("**Context** (flattened)")
            st.text(str(context_fallback))
    else:
        st.info("No `context_turns` or `context` for this row.")


def render_persona_turns_compact(row: dict[str, Any]) -> None:
    """Color-coded transcript for compare (Partner vs Persona, full labels)."""
    persona_name = str(row.get("persona_name") or "Persona")
    turns = row.get("context_turns")
    context_fallback = row.get("context")
    blocks: list[str] = []

    def _turn_block(label: str, body: str, accent: str, bg: str) -> None:
        text = html.escape(body if body else "(empty)")
        blocks.append(
            f'<div style="border-left: 4px solid {accent}; background: {bg}; '
            f"padding: 8px 12px; margin: 8px 0; border-radius: 6px;\">"
            f'<div style="font-weight: 600; color: {accent}; margin-bottom: 6px;">'
            f"{html.escape(label)}</div>"
            f'<div style="white-space: pre-wrap; line-height: 1.45; color: #1f2937;">{text}</div></div>'
        )

    structured = isinstance(turns, list) and turns
    if structured:
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("from", "")).strip().lower()
            body = str(t.get("value", "")).strip("\n") if t.get("value") is not None else ""
            if role == "user":
                _turn_block(
                    "Partner",
                    body,
                    accent="#1565c0",
                    bg="#e3f2fd",
                )
            elif role == "assistant":
                _turn_block(
                    f"Persona · {persona_name}",
                    body,
                    accent="#2e7d32",
                    bg="#e8f5e9",
                )
            else:
                raw = (role or "").strip()
                disp = raw.replace("_", " ").title() if raw else "Other"
                _turn_block(
                    disp,
                    body,
                    accent="#5f6368",
                    bg="#f1f3f4",
                )
    elif context_fallback:
        _turn_block(
            "Context (flattened)",
            str(context_fallback),
            accent="#5f6368",
            bg="#f1f3f4",
        )

    with st.expander("Context", expanded=True):
        if blocks:
            st.markdown("\n".join(blocks), unsafe_allow_html=True)
        else:
            st.caption("No `context_turns` or `context`.")


def continuation_row_splits(n_variants: int) -> list[int]:
    """2–3 continuations per row; balance rows (e.g. 4 → 2+2)."""
    if n_variants <= 0:
        return []
    if n_variants <= 3:
        return [n_variants]
    if n_variants == 4:
        return [2, 2]
    rows: list[int] = []
    rem = n_variants
    while rem > 0:
        if rem <= 3:
            rows.append(rem)
            break
        if rem == 4:
            rows.extend([2, 2])
            break
        rows.append(3)
        rem -= 3
    return rows


def render_persona_reference_column(row: dict[str, Any], *, show_raw_row: bool = True) -> None:
    ref_text = str(row.get("reference", ""))
    system = row.get("system")
    st.markdown("**Reference reply**")
    st.caption("Gold label from val (what actually happened next). Compare to **generated**.")
    with st.container(border=True):
        if ref_text:
            st.text(ref_text)
        else:
            st.caption("Empty")

    sys_s = "" if system is None else str(system).strip()
    if sys_s:
        with st.expander("System prompt", expanded=False):
            st.text(sys_s)

    if show_raw_row:
        with st.expander("Raw row (debug)", expanded=False):
            dbg = dict(row)
            st.json(dbg)


def render_persona_reference_compact_compare(
    row: dict[str, Any], *, sample_key_hex: str
) -> None:
    ref_text = str(row.get("reference", ""))
    system = row.get("system")
    with st.expander("Reference (gold)", expanded=False):
        st.text(ref_text if ref_text else "(empty)")
    sys_s = "" if system is None else str(system).strip()
    if sys_s:
        with st.expander("System prompt", expanded=False):
            st.text(sys_s)
    dbg = dict(row)
    dbg["comparison_sample_key"] = sample_key_hex
    with st.expander("Debug", expanded=False):
        st.json(dbg)


def render_persona_detail_view(row: dict[str, Any]) -> None:
    """Main column: transcript + generated continuation; side: reference + system."""
    persona_name = str(row.get("persona_name") or "Persona")
    gen_text = str(row.get("generated", ""))
    st.markdown(persona_sample_title_md(row))
    c_flow, c_ref = st.columns([2.35, 1.0], gap="large")
    with c_flow:
        st.caption(
            "Read top to bottom: partner messages (**user**) and prior **persona** lines, "
            "then the model’s **next** persona reply."
        )
        render_persona_turns_chunk(row)
        st.divider()
        with st.chat_message("assistant", avatar="✨"):
            st.markdown(f"**{persona_name}** · generated continuation")
            st.text(gen_text if gen_text else " ")
    with c_ref:
        render_persona_reference_column(row, show_raw_row=True)


def render_persona_compare_detail_sample(
    m: MergedPersonaSample,
    *,
    comparison_id: str,
    store_path: Path,
    sample_index: int,
    sample_total: int,
) -> None:
    row = m.base_row
    digest = widget_key_digest(m.sample_key)
    persona_name = str(row.get("persona_name") or "Persona")
    persona_id = str(row.get("persona_id") or "")
    head = f"**{sample_index + 1}/{sample_total}** · {persona_name}"
    if persona_id:
        head += f" · `{persona_id}`"
    st.markdown(head)

    c_flow, c_ref = st.columns([1.55, 0.95], gap="small")

    with c_flow:
        render_persona_turns_compact(row)
        st.markdown("**Continuations**")
        variants_list = list(m.variants)
        splits = continuation_row_splits(len(variants_list))
        i0 = 0
        for ncols in splits:
            batch = variants_list[i0 : i0 + ncols]
            i0 += ncols
            cols = st.columns(ncols, gap="small")
            for col, vrow in zip(cols, batch, strict=True):
                with col:
                    badge_txt = badge_label(vrow.target_modules)
                    cap = truncate_display(f"{vrow.display_label} · {badge_txt}", 140)
                    st.caption(f"{cap} · `{vrow.variant_id}`")
                    gen_s = vrow.generated if vrow.generated else " "
                    with st.container(border=True):
                        st.text(gen_s)
                    render_compare_verdict_buttons(
                        store_path,
                        comparison_id,
                        digest,
                        m.sample_key,
                        vrow.variant_id,
                    )

    with c_ref:
        render_persona_reference_compact_compare(row, sample_key_hex=m.sample_key)
