from __future__ import annotations

import argparse

from discord_sft.cli.commands_data import (
    _cmd_build_sft,
    _cmd_curate,
    _cmd_curate_sweep,
    _cmd_eval_heuristics,
    _cmd_fingerprint,
    _cmd_ingest,
    _cmd_stats,
    _cmd_validate_chat_template,
)
from discord_sft.cli.commands_eval import (
    _cmd_eval_backfill_style_rank_elo,
    _cmd_eval_compare,
    _cmd_eval_doctor,
    _cmd_eval_judge_persona,
    _cmd_eval_list,
    _cmd_eval_rank_style,
    _cmd_eval_run,
)
from discord_sft.cli.commands_train import _cmd_train
from discord_sft.cli.commands_ui import _cmd_ui


def build_parser() -> argparse.ArgumentParser:
    from discord_sft.evals.qwen35_sampling import (
        DEFAULT_QWEN_SAMPLING,
        QWEN_SAMPLING_CHOICES,
    )

    p = argparse.ArgumentParser(
        prog="discord-sft",
        description="Ingest, curate, and SFT-build Discord DM exports; evaluate style.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval-heuristics", help="Compare generated vs reference lines")
    e.add_argument("--references", required=True)
    e.add_argument("--generated", required=True)
    e.add_argument(
        "--profile",
        help="Optional profiles.json (from `fingerprint`) to seed persona-specific fillers",
    )
    e.add_argument(
        "--persona",
        help="persona_id to look up inside --profile",
    )
    e.set_defaults(func=_cmd_eval_heuristics)

    ing = sub.add_parser("ingest", help="Load Discrub JSON pages into sorted parquet/jsonl")
    ing.add_argument("--source", required=True, help="Root with one subfolder per DM")
    ing.add_argument("--out", required=True)
    ing.add_argument("--format", choices=("parquet", "jsonl"), default="parquet")
    ing.set_defaults(func=_cmd_ingest)

    cur = sub.add_parser("curate", help="Clean, burst-merge, and session-split messages")
    cur.add_argument("--source", required=True, help="Root folder of DM subfolders (raw JSON)")
    cur.add_argument("--out", required=True)
    cur.add_argument("--session-gap-min", type=int, default=60)
    cur.add_argument("--merge-gap-sec", type=int, default=30)
    cur.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum raw merged turns per session before quality gates (default 2).",
    )
    cur.add_argument("--min-authors", type=int, default=2)
    cur.add_argument("--monologue-max-share", type=float, default=0.80)
    cur.add_argument("--strip-urls", action="store_true")
    cur.add_argument("--no-pii-scrub", action="store_true")
    cur.add_argument("--lang", default=None)
    cur.add_argument("--near-dedup-threshold", type=float, default=0.85)
    cur.add_argument("--no-near-dedup", action="store_true")
    cur.add_argument(
        "--no-dedupe-exact-turns",
        action="store_true",
        help="Keep repeated (author, text) turns inside a session; default drops later repeats.",
    )
    cur.add_argument(
        "--exact-turn-dup-cap",
        type=int,
        default=1,
        metavar="N",
        help=(
            "With dedupe on: keep the first N identical (author, text) turns per session "
            "(default 1 = strictest). Ignored with --no-dedupe-exact-turns."
        ),
    )
    cur.set_defaults(func=_cmd_curate)

    csw = sub.add_parser(
        "curate-sweep",
        help="Cartesian grid over curate (metrics only; no sessions.jsonl written)",
    )
    csw.add_argument("--source", required=True, help="Root folder of DM subfolders (raw JSON)")
    csw.add_argument(
        "--out",
        required=True,
        help="JSONL path; one object per grid point with params, sessions_kept, report",
    )
    csw.add_argument("--session-gap-min", type=int, default=60)
    csw.add_argument("--merge-gap-sec", type=int, default=30)
    csw.add_argument("--min-turns", type=int, default=2)
    csw.add_argument("--min-authors", type=int, default=2)
    csw.add_argument("--monologue-max-share", type=float, default=0.80)
    csw.add_argument("--strip-urls", action="store_true")
    csw.add_argument("--no-pii-scrub", action="store_true")
    csw.add_argument("--lang", default=None)
    csw.add_argument("--near-dedup-threshold", type=float, default=0.85)
    csw.add_argument("--no-near-dedup", action="store_true")
    csw.add_argument("--no-dedupe-exact-turns", action="store_true")
    csw.add_argument("--exact-turn-dup-cap", type=int, default=1)
    csw.add_argument(
        "--max-combos",
        type=int,
        default=500,
        help="Refuse to run if the Cartesian product exceeds this (default: 500).",
    )
    csw.add_argument(
        "--sweep-session-gap-min",
        metavar="CSV",
        default=None,
        help="Comma-separated session_gap_min values (default: single --session-gap-min).",
    )
    csw.add_argument(
        "--sweep-merge-gap-sec",
        metavar="CSV",
        default=None,
        help="Comma-separated merge_gap_sec values (default: single --merge-gap-sec).",
    )
    csw.add_argument(
        "--sweep-min-turns",
        metavar="CSV",
        default=None,
        help="Comma-separated min_turns values (default: single --min-turns).",
    )
    csw.add_argument(
        "--sweep-min-authors",
        metavar="CSV",
        default=None,
        help="Comma-separated min_authors values (default: single --min-authors).",
    )
    csw.add_argument(
        "--sweep-monologue-max-share",
        metavar="CSV",
        default=None,
        help="Comma-separated monologue_max_share values (default: single --monologue-max-share).",
    )
    csw.set_defaults(func=_cmd_curate_sweep)

    sft = sub.add_parser("build-sft", help="Produce ShareGPT-style SFT JSONL")
    sft.add_argument("--input", required=True, help="sessions.jsonl from `curate`")
    sft.add_argument("--out", required=True)
    sft.add_argument("--window-turns", type=int, default=16)
    sft.add_argument(
        "--min-sharegpt-turns",
        type=int,
        default=2,
        help="Minimum merged ShareGPT messages per row (even, default 2).",
    )
    sft.add_argument(
        "--max-sharegpt-turns",
        default="8",
        metavar="N|none",
        help='Maximum merged ShareGPT messages per row (even). Use "none" for no cap (only --window-turns limits span).',
    )
    sft.add_argument(
        "--turn-mix",
        default="none",
        metavar="POLICY",
        help='After persona balance: none | uniform | JSON weights, e.g. {"2":1,"4":1,"6":1,"8":1}',
    )
    sft.add_argument("--personas", default="all")
    sft.add_argument(
        "--balance",
        default="median",
        help="median | none | min | cap:N",
    )
    sft.add_argument("--balance-k", type=float, default=1.5)
    sft.add_argument("--val-frac", type=float, default=0.10)
    sft.add_argument("--seed", type=int, default=0)
    sft.set_defaults(func=_cmd_build_sft)

    st = sub.add_parser("stats", help="Tokenizer-aware corpus stats (Qwen3 vs Qwen3.5 etc.)")
    st.add_argument("--input", required=True, help="train.jsonl or similar")
    st.add_argument("--tokenizer", action="append", default=[], help="HF model id; repeatable")
    st.set_defaults(func=_cmd_stats)

    vct = sub.add_parser(
        "validate-chat-template",
        help="Validate chat-template turn boundaries for a ShareGPT JSONL dataset.",
    )
    vct.add_argument("--input", required=True, help="train.jsonl or val.jsonl")
    vct.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-35B-A3B",
        help="HF tokenizer id used to render chat template.",
    )
    vct.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum number of samples to validate (default: 500).",
    )
    vct.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum failing sample previews to include (default: 20).",
    )
    vct.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking blocks in template rendering during validation.",
    )
    vct.set_defaults(func=_cmd_validate_chat_template)

    fp = sub.add_parser("fingerprint", help="Per-persona style profile")
    fp.add_argument("--input", required=True, help="train.jsonl")
    fp.add_argument("--out", required=True, help="profiles.json output path")
    fp.add_argument("--top-k", type=int, default=25)
    fp.set_defaults(func=_cmd_fingerprint)

    ui = sub.add_parser("ui", help="Launch the Streamlit dashboard")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--headless", action="store_true")
    ui.set_defaults(func=_cmd_ui)

    tr = sub.add_parser(
        "train",
        help="Fine-tune Qwen3.5-35B-A3B with LoRA via Unsloth.",
        description=(
            "Default action is `run`: one LoRA training job from a YAML config (see "
            "discord_sft/training/configs/). Use `merge-peft` to bake a checkpoint "
            "into dense weights for vLLM without --lora-modules. "
            "Examples: `discord-sft train --config ...`, `discord-sft train merge-peft --adapter ... --output ...`."
        ),
    )
    tr.add_argument(
        "train_action",
        nargs="?",
        default="run",
        choices=("run", "merge-peft"),
        help=(
            "run: fine-tune (default). merge-peft: merge LoRA into base weights "
            "(training venv; high VRAM/disk)."
        ),
    )
    tr.add_argument(
        "--config",
        required=False,
        default=None,
        help=(
            "Training YAML. Required for `run`. For `merge-peft`, optional; "
            "defaults to <adapter_parent>/config.resolved.yaml when present."
        ),
    )
    tr.add_argument(
        "--output-dir",
        default=None,
        help="Override checkpoint.output_dir from the YAML (run only)",
    )
    tr.add_argument(
        "--run-name",
        default=None,
        help="Override run_name from the YAML (run only)",
    )
    tr.add_argument(
        "--resume-adapter",
        default=None,
        help=(
            "Path to an existing adapter checkpoint directory (epoch-N/step-S/final) "
            "to continue training from; overrides train.resume_adapter_path in YAML"
        ),
    )
    tr.add_argument(
        "--adapter",
        dest="merge_adapter",
        default=None,
        help=(
            "merge-peft: adapter directory (adapter_config.json). "
            "Ignored for `run`."
        ),
    )
    tr.add_argument(
        "--output",
        dest="merge_output",
        default=None,
        help="merge-peft: directory for merged model + tokenizer + merge_manifest.json",
    )
    tr.add_argument(
        "--base-model",
        dest="merge_base_model",
        default=None,
        help="merge-peft: override base model id (else run.json / adapter_config.json)",
    )
    tr.add_argument(
        "--max-seq-length",
        dest="merge_max_seq_length",
        type=int,
        default=None,
        help="merge-peft: override FastModel max_seq_length (else config YAML / defaults)",
    )
    tr.add_argument(
        "--max-shard-size",
        dest="merge_max_shard_size",
        default="5GB",
        help='merge-peft: HF save_pretrained max_shard_size (default "5GB")',
    )
    tr.add_argument(
        "--merge-load-in-16bit",
        dest="merge_load_in_16bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "merge-peft: pass --merge-load-in-16bit/--no-merge-load-in-16bit "
            "to override model.load_in_16bit from the training YAML."
        ),
    )
    tr.set_defaults(func=_cmd_train)

    ev = sub.add_parser(
        "eval",
        help="Run benchmark + persona evals and compare saved runs.",
    )
    ev_sub = ev.add_subparsers(dest="eval_cmd", required=True)

    ev_run = ev_sub.add_parser(
        "run",
        help="Run lmms-eval benchmarks + persona evals; save a unified run JSON.",
        description=(
            "Drives python -m lmms_eval as a subprocess for standard benchmarks "
            "(IFEval, MMMU-val, MMStar, ScreenSpot v2) and runs native persona "
            "evals (style heuristics + optional OpenRouter LLM-as-judge). "
            "Writes one JSON per run under <out>/runs/."
        ),
    )
    ev_run.add_argument("--model", help="HF model id or local path (e.g. Qwen/Qwen3.5-35B-A3B)")
    ev_run.add_argument(
        "--backend",
        default="hf",
        help="lmms-eval --model backend: hf | vllm | sglang | qwen2_5_vl | qwen3_vl | ...",
    )
    ev_run.add_argument("--revision", default=None, help="Optional HF revision / git sha pin")
    ev_run.add_argument(
        "--adapter",
        action="append",
        default=[],
        help=(
            "PEFT / LoRA adapter directory. Repeatable: each flag adds one "
            "variant to a sweep. Accepts either /abs/path (alias auto-derived "
            "from the basename) or alias=/abs/path to name it explicitly. With "
            "backend=vllm, all adapters are pre-registered on a single shared "
            "server via --lora-modules so the base model only loads once."
        ),
    )
    ev_run.add_argument(
        "--adapter-dir",
        action="append",
        default=[],
        help=(
            "Directory to recursively scan for LoRA adapters. If final/ "
            "adapters exist, only final/ dirs are included; otherwise every "
            "directory with adapter_config.json is included. Repeatable."
        ),
    )
    ev_run.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also run the base model with no adapter in the sweep",
    )
    ev_run.add_argument(
        "--baseline-prompt",
        choices=("minimal", "style", "profile"),
        default="minimal",
        help=(
            "System-prompt mode for the base-model (no-adapter) persona eval. "
            "'minimal' uses the val.jsonl prompt verbatim (what the LoRA was "
            "trained with — fair control for LoRA uplift). 'style' appends "
            "generic Discord-DM bullets. 'profile' derives per-persona bullets "
            "from --profiles (length stats, lowercase rate, top fillers, emoji, "
            "bursts) and directly targets each judge axis, for the strongest "
            "no-LoRA baseline. Only applies to variants with no adapter; LoRA "
            "variants always use the training prompt."
        ),
    )
    ev_run.add_argument("--dtype", default="bfloat16")
    ev_run.add_argument("--device", default="cuda")
    ev_run.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code (default: on; required for Qwen modelling code)",
    )
    ev_run.add_argument(
        "--model-arg",
        action="append",
        default=[],
        help=(
            "Extra --model_args key=value, repeatable. For vLLM single-GPU "
            "these are rarely needed; prefer the dedicated flags below."
        ),
    )
    ev_run.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help=(
            "vLLM context length. Default 16384 is tuned for Qwen3.5-35B-A3B "
            "in bf16 on a single H100 80GB (~8GB KV cache headroom). Lower "
            "for more concurrency, or combine with --quantization fp8 for more."
        ),
    )
    ev_run.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM --gpu-memory-utilization (fraction of GPU memory).",
    )
    ev_run.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help=(
            "Max rank across all --adapter paths. Default: auto-detect the "
            "highest ``r`` from each adapter_config.json, fallback 16. Setting "
            "this too high wastes GPU memory (per the vLLM LoRA docs)."
        ),
    )
    ev_run.add_argument(
        "--reasoning-parser",
        default="qwen3",
        help=(
            "vLLM reasoning parser. Default 'qwen3' matches the official "
            "Qwen3.5 example script; pass '' to disable for non-reasoning models."
        ),
    )
    ev_run.add_argument(
        "--quantization",
        default=None,
        help=(
            "Optional vLLM --quantization (e.g. fp8). Halves weight memory so "
            "KV cache / concurrency has more room on a single H100."
        ),
    )
    ev_run.add_argument(
        "--no-shared-server",
        action="store_true",
        help=(
            "Disable the shared vLLM server path and fall back to one "
            "lmms-eval subprocess per run using lmms-eval's native vllm "
            "wrapper. Primarily useful for debugging; the shared-server path "
            "is the default and generally more robust/faster."
        ),
    )
    ev_run.add_argument(
        "--persona-max-concurrency",
        type=int,
        default=16,
        help=(
            "Max in-flight persona-eval requests when hitting the shared "
            "vLLM server. Higher values feed continuous batching better but "
            "need proportionally more KV-cache room."
        ),
    )
    ev_run.add_argument(
        "--qwen-sampling",
        default=DEFAULT_QWEN_SAMPLING,
        choices=sorted(QWEN_SAMPLING_CHOICES),
        help=(
            "Persona-eval sampling preset from the Qwen3.5-35B-A3B Hugging Face "
            "model card (instruct_general matches non-thinking general tasks)."
        ),
    )
    ev_run.add_argument(
        "--apply-chat-template",
        dest="apply_chat_template",
        action="store_true",
        help=(
            "Pass --apply_chat_template to lmms-eval for benchmark tasks. "
            "Enabled by default; use --no-apply-chat-template to disable."
        ),
    )
    ev_run.add_argument(
        "--no-apply-chat-template",
        dest="apply_chat_template",
        action="store_false",
        help="Disable lmms-eval --apply_chat_template forwarding.",
    )
    ev_run.set_defaults(apply_chat_template=True)
    ev_run.add_argument(
        "--gen-kwargs",
        dest="gen_kwargs",
        default=None,
        metavar="KWARGS",
        help=(
            "Forwarded to lmms-eval as --gen_kwargs (comma-separated), e.g. "
            "temperature=1.0,top_p=0.95,top_k=20. Does not change persona sampling "
            "(use --qwen-sampling for that)."
        ),
    )
    ev_run.add_argument(
        "--lmms-disable-thinking",
        dest="lmms_disable_thinking",
        action="store_true",
        default=None,
        help=(
            "When using lmms-eval's OpenAI backend (default shared vLLM path), set "
            "LMMS_EVAL_DISABLE_THINKING=1 for benchmark task requests."
        ),
    )
    ev_run.add_argument(
        "--no-lmms-disable-thinking",
        dest="lmms_disable_thinking",
        action="store_false",
        help=(
            "Set LMMS_EVAL_DISABLE_THINKING=0 for lmms-eval benchmark task requests. "
            "If neither flag is passed, keep current environment behavior."
        ),
    )
    ev_run.add_argument(
        "--lmms-verbosity",
        default=None,
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        help=(
            "Forwarded to lmms-eval as top-level --verbosity. Useful when "
            "debugging lmms task/model failures."
        ),
    )
    ev_run.add_argument(
        "--vllm-extra",
        action="append",
        default=[],
        metavar="FRAG",
        help=(
            "Extra argv tokens for the shared vLLM server (repeatable). Each "
            "value is split with shlex, e.g. --vllm-extra '--language-model-only'. "
            "Do not use --language-model-only with VLM benchmarks. For Qwen3.5 "
            "models, --max-num-seqs is auto-appended (512) unless already set here."
        ),
    )
    ev_run.add_argument(
        "--vllm-python",
        default=None,
        metavar="PATH",
        help=(
            "Python for `python -m vllm.entrypoints.openai.api_server` (shared-server "
            "path only). Use when discord-sft's interpreter resolves to `.venv/bin/python` "
            "but LoRA+torch live in `.venv-evals`; stronger than DISCORD_SFT_VLLM_PYTHON if "
            "your eval package is pinned to an older copy of this flag."
        ),
    )
    ev_run.add_argument(
        "--tasks",
        default="",
        help="Comma-separated benchmark keys. Default: ifeval,mmmu_val,mmstar,screenspot_v2,persona",
    )
    ev_run.add_argument("--val", default=None, help="val.jsonl path for persona eval")
    ev_run.add_argument("--profiles", default=None, help="profiles.json path for persona eval")
    ev_run.add_argument("--limit", type=int, default=None, help="Max samples per task")
    ev_run.add_argument("--batch-size", default="1")
    ev_run.add_argument("--num-fewshot", type=int, default=None)
    ev_run.add_argument(
        "--judge",
        choices=("none", "openrouter"),
        default="none",
        help=(
            "Optional persona rubric LLM-as-judge via OpenRouter "
            "(OPENROUTER_API_KEY; default model anthropic/claude-sonnet-4.6; "
            "JSON axes vocabulary/tone/length/authentic_persona plus reasoning)"
        ),
    )
    ev_run.add_argument("--label", default=None, help="Run label (appended to run id)")
    ev_run.add_argument("--seed", type=int, default=0)
    ev_run.add_argument(
        "--out",
        default="out/evals",
        help=(
            "Eval output root: writes <out>/runs/<run_id>.json and <out>/raw/<run_id>/ "
            "(default: %(default)s). Do not pass a path that already ends in …/runs "
            "or you will get nested runs/runs and runs/raw."
        ),
    )
    ev_run.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="Print the registered benchmark catalog and exit",
    )
    ev_run.set_defaults(func=_cmd_eval_run)

    ev_doc = ev_sub.add_parser(
        "doctor",
        help="Print eval stack versions, lmms-eval patch status, and optional GPU health check.",
    )
    ev_doc.add_argument(
        "--require-vllm",
        action="store_true",
        help="Fail if vllm is not importable (default: skip vLLM when absent, e.g. Windows dev).",
    )
    ev_doc.set_defaults(func=_cmd_eval_doctor)

    ev_list = ev_sub.add_parser("list", help="List saved runs under <out>/runs/")
    ev_list.add_argument("--dir", default="out/evals", help="Either out_dir or a direct runs/ dir")
    ev_list.add_argument("--format", choices=("table", "json"), default="table")
    ev_list.set_defaults(func=_cmd_eval_list)

    ev_cmp = ev_sub.add_parser("compare", help="Diff saved runs side by side")
    ev_cmp.add_argument("runs", nargs="+", help="Run paths or run_ids (first is baseline by default)")
    ev_cmp.add_argument(
        "--baseline",
        default=None,
        help="Run path / id / index to use as baseline (default: first). Pass '' to disable.",
    )
    ev_cmp.add_argument(
        "--metrics",
        default="",
        help="Comma-separated glob filters (e.g. ifeval.*,mmmu_val.mmmu_acc)",
    )
    ev_cmp.add_argument(
        "--include-persona-metrics",
        action="store_true",
        help="When --metrics is omitted, include persona.* score keys (default: omit them)",
    )
    ev_cmp.add_argument("--format", choices=("table", "json", "markdown"), default="table")
    ev_cmp.add_argument("--out-dir", default="out/evals", help="Fallback dir for resolving bare run_ids")
    ev_cmp.set_defaults(func=_cmd_eval_compare)

    ev_judge = ev_sub.add_parser(
        "judge-persona",
        help="Judge an existing persona_generations.jsonl without re-running eval.",
    )
    ev_judge.add_argument(
        "--generations",
        required=True,
        help="Path to persona_generations.jsonl from a prior eval run.",
    )
    ev_judge.add_argument(
        "--judge",
        choices=("openrouter",),
        default="openrouter",
        help="Judge backend to use for scoring existing generations.",
    )
    ev_judge.add_argument(
        "--out",
        default=None,
        help="Optional path to save the JSON report.",
    )
    ev_judge.set_defaults(func=_cmd_eval_judge_persona)

    ev_rank = ev_sub.add_parser(
        "rank-style",
        help=(
            "Sparse pairwise LLM style judging + stylometric fingerprint across "
            "multiple persona_generations.jsonl dumps."
        ),
    )
    ev_rank.add_argument(
        "--generations",
        action="append",
        required=True,
        help="persona_generations.jsonl from each run (repeat flag per file).",
    )
    ev_rank.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        help="Label for each run (repeat once per --generations; default: file stem).",
    )
    ev_rank.add_argument(
        "--run-json",
        dest="run_json",
        action="append",
        default=None,
        help="Saved eval run.json per --generations (same order) for val.jsonl provenance.",
    )
    ev_rank.add_argument(
        "--skip-provenance-check",
        action="store_true",
        help="Do not require matching val.jsonl across runs.",
    )
    ev_rank.add_argument(
        "--provenance-from-rows",
        action="store_true",
        help="Use eval_val_jsonl / eval_val_sha256 embedded in jsonl rows.",
    )
    ev_rank.add_argument(
        "--val-hash",
        action="store_true",
        help="Compare SHA256 of val.jsonl files instead of resolved paths.",
    )
    ev_rank.add_argument(
        "--pairs-per-prompt",
        type=int,
        default=6,
        metavar="N",
        help="Random pairwise comparisons per prompt (capped by all pairs). Default: 6.",
    )
    ev_rank.add_argument("--seed", type=int, default=0, help="RNG seed for pair sampling")
    ev_rank.add_argument(
        "--judge-model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model id for pairwise judge (default: %(default)s)",
    )
    ev_rank.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Judge sampling temperature (default: 0.2)",
    )
    ev_rank.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Max parallel judge requests",
    )
    ev_rank.add_argument(
        "--profiles",
        default=None,
        help="Optional profiles.json for mined filler list in fingerprints",
    )
    ev_rank.add_argument(
        "--no-reference-style",
        action="store_true",
        help="Omit reference completion block from the judge prompt",
    )
    ev_rank.add_argument(
        "--emit-comparisons",
        action="store_true",
        help=(
            "Include per-comparison judge payload in the report (prompt, outputs A/B, "
            "reference, judge model transcript); much larger JSON"
        ),
    )
    ev_rank.add_argument(
        "--prior-elo-json",
        default=None,
        metavar="PATH",
        help=(
            "JSON object mapping run label (string) to Elo (number); seeds ratings before "
            "this pairwise session (matches UI cumulative Elo)."
        ),
    )
    ev_rank.add_argument(
        "--prior-elo",
        dest="prior_elos",
        action="append",
        default=[],
        metavar="LABEL=RATING",
        help="Seed one label's Elo (repeatable); overrides same key from --prior-elo-json.",
    )
    ev_rank.add_argument(
        "--comparisons-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Append each pairwise result (and failures) to this JSONL as they finish "
            "(survives crashes; default: none)."
        ),
    )
    ev_rank.add_argument(
        "--pairwise-weight",
        type=float,
        default=0.7,
        help="Weight for normalized pairwise (Elo) in combined score",
    )
    ev_rank.add_argument(
        "--fingerprint-weight",
        type=float,
        default=0.3,
        help="Weight for normalized fingerprint similarity in combined score",
    )
    ev_rank.add_argument(
        "--out",
        required=True,
        help="Path to write the JSON report",
    )
    ev_rank.set_defaults(func=_cmd_eval_rank_style)

    ev_bfill_elo = ev_sub.add_parser(
        "backfill-style-rank-elo",
        help=(
            "Replay ``style_rank_checkpoints/*.jsonl`` pairwise rows to recompute Elo and "
            "write ``style_rank_group_elo.json`` (survives UI reload)."
        ),
    )
    ev_bfill_elo.add_argument(
        "--dir",
        default="out/evals",
        help="Eval output root containing ``style_rank_checkpoints/``",
    )
    ev_bfill_elo.add_argument(
        "--checkpoints",
        default=None,
        help="Optional directory of JSONL checkpoints (default: <dir>/style_rank_checkpoints)",
    )
    ev_bfill_elo.add_argument(
        "--fresh-elo-store",
        action="store_true",
        help="Ignore existing style_rank_group_elo.json groups (overwrite file with replay only).",
    )
    ev_bfill_elo.add_argument(
        "--elo-k",
        type=float,
        default=32.0,
        help="Must match pairwise K used when checkpoints were collected (default: 32).",
    )
    ev_bfill_elo.add_argument(
        "--elo-start",
        type=float,
        default=1500.0,
        help="Starting rating (default: 1500).",
    )
    ev_bfill_elo.set_defaults(func=_cmd_eval_backfill_style_rank_elo)

    return p
