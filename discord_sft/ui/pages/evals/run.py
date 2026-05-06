"""Run evals tab (subprocess + form)."""

from __future__ import annotations

import sys

import streamlit as st

from discord_sft.evals.benchmarks import BENCHMARKS, DEFAULT_TASKS
from discord_sft.evals.qwen35_sampling import DEFAULT_QWEN_SAMPLING, QWEN_SAMPLING_CHOICES
from discord_sft.ui._subprocess import shlex_join_cmd, stream_subprocess
from discord_sft.ui.common import resolve_repo_path, session_work_path
from discord_sft.ui.pages._baseline_prompt import BASELINE_PROMPT_HELP, baseline_prompt_example


def render_run_tab() -> None:
    default_val = session_work_path("sft", "val.jsonl")
    default_profiles = session_work_path("sft", "profiles.json")
    default_out = session_work_path("evals")

    with st.form("eval_run_form"):
        c1, c2 = st.columns(2)
        with c1:
            model = st.text_input(
                "Model (HF id or local path)",
                value="Qwen/Qwen3.5-35B-A3B",
                help="Passed straight to lmms-eval --model_args pretrained= / model=.",
            )
            backend = st.selectbox(
                "Backend",
                ["hf", "vllm", "sglang", "qwen2_5_vl", "qwen3_vl"],
                index=1,
                help="lmms-eval --model wrapper. vllm/sglang are recommended for 35B-A3B scale.",
            )
            adapter = st.text_input("Adapter path (PEFT / LoRA, optional)", value="")
            dtype = st.selectbox("dtype", ["bfloat16", "float16", "float32"], index=0)
            label = st.text_input("Label (optional)", value="baseline")
        with c2:
            tasks = st.multiselect(
                "Benchmarks",
                list(BENCHMARKS.keys()),
                default=list(DEFAULT_TASKS),
                help="\n".join(f"{k}: {v.description}" for k, v in BENCHMARKS.items()),
            )
            limit = st.number_input(
                "Sample limit per task (0 = full)", min_value=0, value=0, step=50
            )
            batch_size = st.text_input(
                "Batch size (int or 'auto')",
                value="auto" if backend == "vllm" else "1",
                help="Use `auto` with vLLM when possible; use an integer for backends that do not support it.",
            )
            num_fewshot = st.number_input(
                "Num few-shot (blank = task default)", min_value=-1, value=-1, step=1
            )
            judge = st.selectbox(
                "Persona judge",
                ["none", "openrouter"],
                index=1,
                help=(
                    "LLM-as-judge on persona evals via OpenRouter "
                    "(requires OPENROUTER_API_KEY)."
                ),
            )

        baseline_prompt = st.selectbox(
            "Baseline prompt (no-adapter persona eval)",
            ["minimal", "style", "profile"],
            help=(
                "Controls only the base-model/no-adapter persona baseline; "
                "LoRA variants use the training prompt. See expander below for details."
            ),
        )
        st.caption(BASELINE_PROMPT_HELP[baseline_prompt])
        with st.expander("Example baseline system prompt", expanded=False):
            st.code(baseline_prompt_example(baseline_prompt), language="text")

        with st.expander("Advanced: paths, output, sampling, vLLM", expanded=False):
            val_path = st.text_input("val.jsonl (persona eval)", value=str(default_val))
            profile_path = st.text_input(
                "profiles.json (persona eval)", value=str(default_profiles)
            )
            out_dir = st.text_input("Eval output dir", value=str(default_out))
            qwen_sampling = st.selectbox(
                "Qwen sampling preset (persona eval)",
                sorted(QWEN_SAMPLING_CHOICES),
                index=sorted(QWEN_SAMPLING_CHOICES).index(DEFAULT_QWEN_SAMPLING),
            )
            max_len = st.number_input(
                "Max model length (vLLM only)",
                min_value=1024,
                value=16384,
                step=1024,
            )
            gpu_mem = st.slider(
                "GPU memory utilization (vLLM only)",
                min_value=0.50,
                max_value=0.98,
                value=0.90,
                step=0.01,
            )
            quantization = st.selectbox(
                "Quantization (vLLM only)",
                ["none", "fp8"],
            )

        submitted = st.form_submit_button("Run evals", type="primary")

    if not submitted:
        return
    if "persona" in tasks and not resolve_repo_path(val_path).exists():
        st.error(f"val.jsonl not found: {val_path}")
        return

    cmd = [
        sys.executable,
        "-m",
        "discord_sft.cli",
        "eval",
        "run",
        "--model",
        model,
        "--backend",
        backend,
        "--dtype",
        dtype,
        "--tasks",
        ",".join(tasks) if tasks else ",".join(DEFAULT_TASKS),
        "--out",
        out_dir,
        "--batch-size",
        batch_size,
        "--qwen-sampling",
        qwen_sampling,
    ]
    if adapter.strip():
        cmd += ["--adapter", adapter.strip()]
    if label.strip():
        cmd += ["--label", label.strip()]
    if limit and limit > 0:
        cmd += ["--limit", str(int(limit))]
    if num_fewshot >= 0:
        cmd += ["--num-fewshot", str(int(num_fewshot))]
    if judge != "none":
        cmd += ["--judge", judge]
    if "persona" in tasks:
        cmd += [
            "--val",
            val_path,
            "--profiles",
            profile_path,
            "--baseline-prompt",
            baseline_prompt,
        ]
    if backend == "vllm":
        cmd += [
            "--max-model-len",
            str(int(max_len)),
            "--gpu-memory-utilization",
            f"{float(gpu_mem):.2f}",
        ]
        if quantization != "none":
            cmd += ["--quantization", quantization]

    cmd_str = shlex_join_cmd(cmd)
    with st.expander("Resolved shell command", expanded=False):
        st.code(cmd_str, language="bash")

    log_area = st.empty()
    stream_subprocess(cmd, log_area)
