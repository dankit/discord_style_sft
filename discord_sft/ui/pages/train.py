"""LoRA training launcher (same CLI as `discord-sft train`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from discord_sft.ui._subprocess import build_train_cmd, shlex_join_cmd, stream_subprocess
from discord_sft.ui.common import repo_root, session_work_path


def _config_dir() -> Path:
    return repo_root() / "discord_sft" / "training" / "configs"


def _default_run_name() -> str:
    key = "ui_train_default_run_name"
    if key not in st.session_state:
        st.session_state[key] = "qwen35-style-lora-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(st.session_state[key])


def render_train() -> None:
    st.title("Train")
    st.caption("Run `discord-sft train` from this machine (needs `[train]` extra / GPU stack).")

    cfg_dir = _config_dir()
    yaml_names = sorted(p.name for p in cfg_dir.glob("*.yaml")) if cfg_dir.is_dir() else []
    if not yaml_names:
        st.error(f"No YAML configs under `{cfg_dir}`.")
        return

    default_cfg = (
        "qwen35_a3b_style_late.yaml" if "qwen35_a3b_style_late.yaml" in yaml_names else yaml_names[0]
    )
    idx = yaml_names.index(default_cfg) if default_cfg in yaml_names else 0

    output_dir = ""
    run_name = ""
    with st.form("train_form"):
        pick = st.selectbox("Training config", yaml_names, index=idx)
        ready = st.checkbox(
            "I have reviewed the config and SFT files exist where the YAML expects them",
            value=False,
        )
        output_dir = ""
        run_name = ""
        if ready:
            c1, c2 = st.columns(2)
            with c1:
                if st.toggle("Override output directory", value=True):
                    output_dir = st.text_input(
                        "Output directory",
                        value=session_work_path("lora", _default_run_name()),
                        help="Adapter checkpoints root (epoch-N, final, …).",
                    )
            with c2:
                if st.toggle("Set run name", value=True):
                    run_name = st.text_input("Run name", value=_default_run_name())
        submitted = st.form_submit_button("Run training", type="primary")

    config_rel = f"discord_sft/training/configs/{pick}"
    cmd = build_train_cmd(
        config=config_rel,
        output_dir=output_dir.strip() if output_dir else None,
        run_name=run_name.strip() if run_name else None,
    )
    cmd_str = shlex_join_cmd(cmd)

    with st.expander("Resolved shell command", expanded=False):
        st.code(cmd_str, language="bash")

    if not submitted:
        return
    if not ready:
        st.warning("Check the confirmation box when you are ready to launch training.")
        return

    log_area = st.empty()
    stream_subprocess(cmd, log_area)
