"""Fingerprint page."""

from __future__ import annotations

import json

import streamlit as st

from discord_sft.analysis.fingerprint import build_profiles
from discord_sft.data_prep.sft import read_samples
from discord_sft.ui.common import existing_file, resolve_repo_path, session_work_path


def render_fingerprint() -> None:
    st.title("🔎 Fingerprint")
    st.caption("Per-persona style profile: mined filler n-grams, length, lowercase, emoji, burst size.")

    default_in = session_work_path("sft", "train.jsonl")
    default_out = session_work_path("sft", "profiles.json")

    with st.form("fingerprint_form"):
        with st.expander("Advanced: input/output paths", expanded=False):
            in_str = st.text_input("SFT JSONL path", value=str(default_in))
            out_str = st.text_input("Profiles output", value=str(default_out))
        top_k = st.slider("Top-K n-grams", 5, 100, 25, 5)
        submitted = st.form_submit_button("Build fingerprints", type="primary")

    if not submitted:
        return

    path = existing_file(in_str)
    if path is None:
        st.warning("Run **Build SFT** first, or point to a train.jsonl.")
        return
    samples = read_samples(path)
    profiles = build_profiles(samples, top_k_ngrams=int(top_k))
    out_path = resolve_repo_path(out_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    st.success(f"Wrote profiles for {len(profiles['personas'])} personas to {out_path}.")
