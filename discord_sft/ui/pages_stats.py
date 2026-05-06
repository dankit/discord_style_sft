from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from discord_sft.analysis.tokstats import (
    compare_tokenizer_configs,
    compare_tokenizers,
    compare_vocabs,
    encoder_from_tokenizer,
    load_hf_tokenizer,
    probe_tokenization,
    sample_unique_tokens,
    tokenizer_config,
)
from discord_sft.data_prep.sft import read_samples
from discord_sft.training.data import (
    render_sharegpt_text,
    response_only_char_labels,
    validate_chat_template_dataset,
)
from discord_sft.ui.common import existing_file, session_work_dir, session_work_path


def render_stats() -> None:
    st.title("Tokens and Templates")
    st.caption("Tokenizer-aware corpus stats and chat-template validation.")
    tok_tab, validate_tab = st.tabs(["Tokenizer stats", "Chat template"])

    with tok_tab:
        _render_tokenizer_comparison()
    with validate_tab:
        _render_chat_template_validation()


def _render_tokenizer_comparison() -> None:
    default_in = session_work_path("sft", "train.jsonl")
    with st.form("tokenizer_stats_form"):
        with st.expander("Advanced: SFT JSONL path", expanded=False):
            in_path = st.text_input("SFT JSONL path", value=default_in)
        tokenizers_raw = st.text_area(
            "HuggingFace tokenizer IDs (one per line; leave blank to use whitespace fallback)",
            value="Qwen/Qwen3-8B\nQwen/Qwen3.5-35B-A3B",
            height=100,
        )
        probes_raw = st.text_area(
            "Probe strings for side-by-side tokenization (one per line)",
            value="\n".join(_default_probes()),
            height=140,
        )
        submitted = st.form_submit_button("Run stats", type="primary")
    if not submitted:
        return
    path = existing_file(in_path)
    if path is None:
        st.warning("Run **Build SFT** first, or point to a train.jsonl.")
        return
    samples = read_samples(path)

    tokenizer_ids = [ln.strip() for ln in tokenizers_raw.splitlines() if ln.strip()]
    tokenizers: dict = {}
    encoders: dict = {"whitespace": lambda t: t.split()} if not tokenizer_ids else {}
    if tokenizer_ids:
        for tid in tokenizer_ids:
            tok = load_hf_tokenizer(tid)
            tokenizers[tid] = tok
            encoders[tid] = encoder_from_tokenizer(tok)
    report = compare_tokenizers(samples, encoders)
    st.bar_chart({name: rep["total_tokens"] for name, rep in report["tokenizers"].items()}, horizontal=True)
    if tokenizers:
        _render_tokenizer_config_diff(tokenizers)
        _render_vocab_comparison(tokenizers, probes_raw)


def _render_chat_template_validation() -> None:
    default_val = session_work_path("sft", "val.jsonl")
    default_train = session_work_path("sft", "train.jsonl")
    with st.expander("Advanced: dataset path", expanded=False):
        path_in = st.text_input(
            "Dataset path (train.jsonl or val.jsonl)",
            value=default_val,
            key="chat_validate_path",
        )
    path = existing_file(path_in)
    if path is None:
        path = existing_file(default_train)
    if path is None:
        st.warning("Run **Build SFT** first, or point to train/val jsonl.")
        return

    tokenizer_id = st.text_input(
        "Tokenizer id",
        value="Qwen/Qwen3.5-35B-A3B",
        key="chat_validate_tokenizer",
    )
    max_samples = int(
        st.number_input(
            "Max samples",
            min_value=1,
            max_value=50000,
            value=500,
            step=50,
            key="chat_validate_max_samples",
        )
    )
    max_errors = int(
        st.number_input(
            "Max failing previews",
            min_value=1,
            max_value=200,
            value=20,
            step=5,
            key="chat_validate_max_errors",
        )
    )
    enable_thinking = st.checkbox(
        "Enable thinking in template rendering",
        value=False,
        key="chat_validate_enable_thinking",
    )
    run_validation = st.button(
        "Validate chat template", type="primary", key="chat_validate_run"
    )
    if run_validation:
        tok = load_hf_tokenizer(tokenizer_id)
        with st.spinner("Rendering samples and validating turn boundaries..."):
            report = validate_chat_template_dataset(
                path,
                tok,
                enable_thinking=enable_thinking,
                max_samples=max_samples,
                max_errors=max_errors,
            )
        st.session_state["chat_validate_report"] = report
        st.session_state["chat_validate_resolved_path"] = str(path)
        st.session_state["chat_validate_resolved_tokenizer"] = tokenizer_id
        st.session_state["chat_validate_resolved_enable_thinking"] = bool(enable_thinking)
    else:
        report = st.session_state.get("chat_validate_report")
        resolved_path = st.session_state.get("chat_validate_resolved_path")
        resolved_tok = st.session_state.get("chat_validate_resolved_tokenizer")
        resolved_thinking = st.session_state.get("chat_validate_resolved_enable_thinking")
        if (
            not isinstance(report, dict)
            or resolved_path != str(path)
            or resolved_tok != tokenizer_id
            or bool(resolved_thinking) != bool(enable_thinking)
        ):
            st.info("Click **Validate chat template** to generate results.")
            return

    st.metric("Samples checked", report["samples_checked"])
    st.metric("Samples failed", report["samples_failed"])
    st.metric("Pass rate", f'{100.0 * float(report["pass_rate"]):.2f}%')
    st.json(report["counts"])
    if report["errors"]:
        st.subheader("Failing samples")
        st.json(report["errors"])
    else:
        st.success("No chat-template boundary issues found in checked samples.")

    st.divider()
    st.subheader("Rendered sample inspector")
    st.caption("Change sample index freely; no rerun is required unless settings change.")
    tok = load_hf_tokenizer(tokenizer_id)
    _render_chat_template_samples(path, tok, enable_thinking=enable_thinking)


def _render_chat_template_samples(path, tokenizer, *, enable_thinking: bool) -> None:
    sample_cap = int(
        st.number_input(
            "Load first N samples for inspection",
            min_value=1,
            max_value=5000,
            value=200,
            step=50,
            key="chat_validate_preview_cap",
        )
    )
    rows = _load_jsonl_rows(path, sample_cap)
    if not rows:
        st.info("No samples found.")
        return
    idx = int(
        st.slider(
            "Sample index",
            min_value=0,
            max_value=len(rows) - 1,
            value=min(int(st.session_state.get("chat_validate_preview_index", 0)), len(rows) - 1),
            step=1,
            key="chat_validate_preview_index",
        )
    )
    row = rows[idx]
    rendered, kwargs_path = render_sharegpt_text(
        row,
        tokenizer,
        enable_thinking=enable_thinking,
    )
    labels = response_only_char_labels(rendered)
    supervised_chars = sum(1 for x in labels if x != -100)
    user_turns = rendered.count("<|im_start|>user\n")
    assistant_turns = rendered.count("<|im_start|>assistant\n")

    c1, c2, c3 = st.columns(3)
    c1.metric("User turns", user_turns)
    c2.metric("Assistant turns", assistant_turns)
    c3.metric("Supervised chars", supervised_chars)
    st.caption(f"Template path used: `{kwargs_path}`")
    st.caption("Green text is supervised (assistant spans). Gray text is masked (non-assistant spans).")

    st.markdown(_masked_html(rendered, labels), unsafe_allow_html=True)
    st.code(rendered, language=None)
    st.json(row.get("meta", {}))


def _masked_html(text: str, labels: list[int]) -> str:
    chunks: list[str] = []
    if not text:
        return ""
    i = 0
    while i < len(text):
        supervised = labels[i] != -100
        j = i + 1
        while j < len(text) and (labels[j] != -100) == supervised:
            j += 1
        seg = html.escape(text[i:j]).replace("\n", "<br/>")
        color = "#166534" if supervised else "#475569"
        bg = "#dcfce7" if supervised else "#f1f5f9"
        chunks.append(f"<span style='color:{color};background:{bg}'>{seg}</span>")
        i = j
    return (
        "<div style='white-space:pre-wrap;line-height:1.35;border:1px solid #e2e8f0;"
        "padding:10px;border-radius:8px;font-family:ui-monospace, SFMono-Regular, Menlo, monospace;"
        "font-size:12px;'>"
        + "".join(chunks)
        + "</div>"
    )


def _load_jsonl_rows(path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= limit:
                break
    return rows


def _default_probes() -> list[str]:
    """Seed probe strings from mined 1-grams in ``sft/profiles.json`` (same file Fingerprint writes)."""
    prof_path = session_work_dir() / "sft" / "profiles.json"
    fallback = ["imma", "lowkey", "fr", "tbh", "ngl", "sheesh", "bruh", "lmao", ":skull:"]
    if not prof_path.exists():
        return fallback
    try:
        doc = json.loads(prof_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    seen: list[str] = []
    for prof in (doc.get("personas") or {}).values():
        unigrams = (prof.get("top_fillers") or {}).get("1gram") or []
        for item in unigrams:
            tok = item.get("token") if isinstance(item, dict) else None
            if tok and str(tok) not in seen:
                seen.append(str(tok))
            if len(seen) >= 20:
                return seen
    return seen or fallback


def _render_tokenizer_config_diff(tokenizers: dict) -> None:
    configs = {name: tokenizer_config(tok) for name, tok in tokenizers.items()}
    report = compare_tokenizer_configs(configs)
    st.subheader("Tokenizer config diff")
    st.json(report)


def _render_vocab_comparison(tokenizers: dict, probes_raw: str) -> None:
    vocabs = {name: set(tok.get_vocab().keys()) for name, tok in tokenizers.items()}
    names = list(vocabs.keys())
    if len(names) >= 2:
        st.json(compare_vocabs(vocabs))
    probes = [ln.strip() for ln in probes_raw.splitlines() if ln.strip()]
    if not probes:
        return
    rows = probe_tokenization(tokenizers, probes)
    display_rows = []
    for r in rows:
        display: dict[str, Any] = {"probe": r["probe"]}
        for name in names:
            display[f"{name} count"] = r[f"{name}_count"]
            display[f"{name} pieces"] = " · ".join(r[f"{name}_pieces"])
        display_rows.append(display)
    st.table(display_rows)
    if len(names) >= 2:
        a_sel, b_sel = names[0], names[1]
        st.code(" ".join(sample_unique_tokens(vocabs, a=a_sel, b=b_sel, limit=30)), language=None)

