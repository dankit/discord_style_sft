"""Evaluation harness for discord-sft.

Drives `lmms-eval` (EvolvingLMMs-Lab) for a small, opinionated set of
"forgetting-canary" benchmarks (IFEval, MMMU-val, MMStar, ScreenSpot v2)
and folds in the project's native persona evals (style heuristics +
optional LLM-as-judge) into a single unified run JSON artifact.

Typical use:

    from discord_sft.evals import ModelSpec, run_evals, compare_runs

    spec = ModelSpec(name_or_path="Qwen/Qwen3.5-35B-A3B", backend="vllm")
    run = run_evals(spec, tasks=["ifeval", "persona"], limit=200,
                    out_dir="out/evals")
    rows = compare_runs(["out/evals/runs/<baseline>.json",
                         "out/evals/runs/<lora>.json"],
                        baseline=0)
    # By default ``persona.*`` keys are omitted when ``metrics`` is unset;
    # pass ``omit_persona_metrics=False`` or explicit ``metrics=[...]`` globs to include them.
"""
from __future__ import annotations

from discord_sft.evals.benchmarks import BENCHMARKS, BenchmarkSpec
from discord_sft.evals.compare import compare_runs, render_comparison
from discord_sft.evals.model import ModelSpec
from discord_sft.evals.runner import load_training_config_provenance, run_evals
from discord_sft.evals.storage import list_runs, load_run, run_id_for, save_run

__all__ = [
    "BENCHMARKS",
    "BenchmarkSpec",
    "load_training_config_provenance",
    "ModelSpec",
    "compare_runs",
    "list_runs",
    "load_run",
    "render_comparison",
    "run_evals",
    "run_id_for",
    "save_run",
]
