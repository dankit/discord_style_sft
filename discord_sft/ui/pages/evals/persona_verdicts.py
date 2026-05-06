"""Accept / neutral / reject verdict buttons for persona compare UI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from discord_sft.ui.persona_compare import (
    VerdictKind,
    comparison_verdict_map,
    effective_verdict,
    load_vote_store,
    save_vote_store,
)


def apply_persona_compare_verdict(
    store_path: Path,
    comparison_id: str,
    sample_key_hex: str,
    variant_id: str,
    kind: VerdictKind,
) -> None:
    doc = load_vote_store(store_path)
    comp_block = doc.setdefault("comparisons", {}).setdefault(
        comparison_id, {"verdicts": {}, "variants": {}}
    )
    vroot = comp_block.setdefault("verdicts", {})
    if kind == "neutral":
        inner = vroot.get(sample_key_hex)
        if isinstance(inner, dict):
            row_copy = {k: v for k, v in inner.items() if k != variant_id}
            if row_copy:
                vroot[sample_key_hex] = row_copy
            else:
                vroot.pop(sample_key_hex, None)
    else:
        inner = dict(vroot.get(sample_key_hex) or {})
        inner[variant_id] = kind
        vroot[sample_key_hex] = inner
    comp_block.pop("votes", None)
    save_vote_store(store_path, doc)


def render_compare_verdict_buttons(
    store_path: Path,
    comparison_id: str,
    digest: str,
    sample_key_hex: str,
    variant_id: str,
) -> None:
    """Accept (green accent), Neutral (gray), Reject (red accent) — not theme-inferred.

    Buttons render first so a click persists before we read accents from disk — otherwise
    Streamlit runs top-to-bottom with stale verdicts computed earlier in the page.
    """
    opts = (
        ("accepted", "Accept", "#16a34a"),
        ("neutral", "Neutral", "#6b7280"),
        ("rejected", "Reject", "#dc2626"),
    )
    inactive_bar = "#e5e7eb"

    cols_btn = st.columns(3)
    for col, (kind, lbl, _) in zip(cols_btn, opts, strict=True):
        with col:
            if st.button(
                lbl,
                key=f"pbtn__{comparison_id}__{digest}__{variant_id}__{kind}",
                width="stretch",
            ):
                apply_persona_compare_verdict(
                    store_path, comparison_id, sample_key_hex, variant_id, kind
                )

    doc = load_vote_store(store_path)
    raw_compare = (doc.get("comparisons") or {}).get(comparison_id) or {}
    verdicts_now = comparison_verdict_map(raw_compare)
    cur = effective_verdict(verdicts_now, sample_key_hex, variant_id)

    cols_bar = st.columns(3)
    for col, (kind, lbl, hue) in zip(cols_bar, opts, strict=True):
        sel = cur == kind
        with col:
            st.markdown(
                f"<div style='height:4px;border-radius:2px;background:"
                f"{hue if sel else inactive_bar};margin-bottom:6px'></div>",
                unsafe_allow_html=True,
            )
