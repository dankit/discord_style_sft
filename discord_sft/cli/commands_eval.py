from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path


def _parse_extra_model_args(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--model-arg expects key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_vllm_extra_argv(items: list[str] | None) -> list[str]:
    """Split each ``--vllm-extra`` string with :func:`shlex.split` for argv tokens."""
    out: list[str] = []
    for raw in items or []:
        s = raw.strip()
        if not s:
            continue
        out.extend(shlex.split(s, posix=(os.name != "nt")))
    return out


def _vllm_argv_has_max_num_seqs(argv: list[str]) -> bool:
    """True if argv already sets vLLM's max concurrent sequences."""
    for i, tok in enumerate(argv):
        if tok in ("--max-num-seqs", "--max_num_seqs"):
            return True
        if tok.startswith("--max-num-seqs=") or tok.startswith("--max_num_seqs="):
            return True
    return False


def _warn_adapter_base_model_mismatch(
    *,
    eval_model: str,
    adapters: list[tuple[str, str]],
    on_line: Callable[[str], None],
) -> None:
    """Log when ``--model`` disagrees with ``run.json`` ``base_model`` for a LoRA."""
    from discord_sft.evals.adapter_training_meta import read_training_base_model_from_adapter

    for alias, abs_path in adapters:
        trained = read_training_base_model_from_adapter(abs_path)
        if trained and trained != eval_model:
            on_line(
                f"[discord-sft] WARNING: --model {eval_model!r} != training base_model "
                f"{trained!r} (run.json for adapter {alias!r}). "
                "Match config.model.name for tokenizer/generation parity with training; "
                f"suggested: --model {trained!r}"
            )


def _apply_qwen35_mamba_max_num_seqs_default(
    model: str | None, argv: list[str], *, on_line: Callable[[str], None]
) -> list[str]:
    """Append ``--max-num-seqs`` for Qwen3.5 MoE when vLLM's default (1024) breaks startup.

    vLLM v1 can raise ``max_num_seqs exceeds available Mamba cache blocks`` on
    Qwen3.5 family models; see evals README (GH200 startup tuning).
    """
    from discord_sft.evals.vllm_server import _is_qwen35_model

    if not _is_qwen35_model(model) or _vllm_argv_has_max_num_seqs(argv):
        return argv
    on_line(
        "[discord-sft] adding --max-num-seqs 512 for Qwen3.5 (Mamba/vLLM); "
        "override with --vllm-extra \"--max-num-seqs N\""
    )
    return [*argv, "--max-num-seqs", "512"]


def _cmd_eval_doctor(args: argparse.Namespace) -> int:
    from discord_sft.evals.env_health import print_doctor_report

    require_vllm = bool(getattr(args, "require_vllm", False))
    return print_doctor_report(require_vllm=require_vllm)


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from discord_sft.evals import ModelSpec, run_evals
    from discord_sft.evals.benchmarks import BENCHMARKS, DEFAULT_TASKS

    if args.list_benchmarks:
        sys.stdout.write("Available benchmarks:\n")
        for key, bspec in BENCHMARKS.items():
            sys.stdout.write(f"  {key} [{bspec.modality}, {bspec.category}]\n")
            sys.stdout.write(f"      {bspec.description}\n")
        return 0

    if not args.model:
        sys.stderr.write("--model is required (or pass --list-benchmarks)\n")
        return 2

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(DEFAULT_TASKS)
    )

    judge = None
    if args.judge == "openrouter":
        from discord_sft.evals.judge import make_judge

        judge = make_judge(args.judge)

    def _on_line(line: str) -> None:
        sys.stderr.write(line + "\n")

    adapters_raw: list[str] = list(args.adapter or [])
    if args.adapter_dir:
        from discord_sft.evals.vllm_server import discover_adapter_dirs

        for root in args.adapter_dir:
            adapters_raw.extend(discover_adapter_dirs(root))

    from discord_sft.evals.vllm_server import adapters_from_cli

    parsed_adapters = adapters_from_cli(adapters_raw)
    if parsed_adapters:
        _warn_adapter_base_model_mismatch(
            eval_model=args.model,
            adapters=parsed_adapters,
            on_line=_on_line,
        )
    # For backend=vllm, default to the shared OpenAI-compatible server path.
    # This avoids lmms-eval's native vllm backend import surface (which can
    # pull optional deps like decord on aarch64) and keeps one model load per run.
    # --no-shared-server explicitly opts back into lmms-eval's native wrapper.
    want_server = args.backend == "vllm" and not args.no_shared_server
    effective_apply_chat_template = args.apply_chat_template
    if want_server and effective_apply_chat_template:
        # lmms-eval's OpenAICompatible backend does not expose chat_template.
        # Forwarding --apply_chat_template crashes on recent lmms-eval builds.
        _on_line(
            "[discord-sft] disabling --apply_chat_template for shared OpenAI/vLLM backend"
        )
        effective_apply_chat_template = False

    extra_model_args = _parse_extra_model_args(args.model_arg)
    vllm_extra_argv = _parse_vllm_extra_argv(list(args.vllm_extra or []))
    if want_server:
        vllm_extra_argv = _apply_qwen35_mamba_max_num_seqs_default(
            args.model, vllm_extra_argv, on_line=_on_line
        )
    lmms_extra_cli: list[str] = []
    if effective_apply_chat_template:
        lmms_extra_cli.append("--apply_chat_template")
    if args.gen_kwargs:
        lmms_extra_cli.extend(["--gen_kwargs", args.gen_kwargs])
    if args.lmms_verbosity:
        lmms_extra_cli.extend(["--verbosity", args.lmms_verbosity])
    out_files: list[str] = []
    run_ids: list[str] = []
    score_counts: list[int] = []
    prev_disable_thinking = os.environ.get("LMMS_EVAL_DISABLE_THINKING")
    if args.lmms_disable_thinking is not None:
        os.environ["LMMS_EVAL_DISABLE_THINKING"] = "1" if args.lmms_disable_thinking else "0"
        _on_line(
            "[discord-sft] LMMS_EVAL_DISABLE_THINKING="
            + os.environ["LMMS_EVAL_DISABLE_THINKING"]
        )

    try:
        if want_server:
            from discord_sft.evals.vllm_server import (
                VLLMServer,
                VLLMServerConfig,
                ensure_vllm_available,
                max_lora_rank,
            )

            ensure_vllm_available(args.model, spawn_python=args.vllm_python)
            adapters = parsed_adapters
            rank = args.max_lora_rank or max_lora_rank(adapters, default=16)
            server_cfg = VLLMServerConfig(
                model=args.model,
                adapters=adapters,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_lora_rank=rank,
                reasoning_parser=args.reasoning_parser,
                quantization=args.quantization,
                trust_remote_code=not args.no_trust_remote_code,
                extra_args=vllm_extra_argv,
                python_executable=args.vllm_python,
            )

            # Each sweep variant = (label_suffix, adapter_path_or_None,
            # lora_alias_or_None). "baseline" means request against the base
            # model id (no adapter routing), exactly what --include-baseline
            # buys you.
            variants: list[tuple[str, str | None, str | None]] = []
            if args.include_baseline or not adapters:
                # Single model load (no LoRA sweep): use --label verbatim so run
                # filenames don't get a spurious "-baseline" suffix. Adapter
                # sweeps keep "<label>-baseline" when --include-baseline tags
                # the no-LoRA column next to each finetune.
                if not adapters:
                    base_label = args.label or "baseline"
                else:
                    base_label = (
                        f"{args.label}-baseline" if args.label else "baseline"
                    )
                variants.append((base_label, None, None))
            for alias, path in adapters:
                variant_label = f"{args.label}-{alias}" if args.label else alias
                variants.append((variant_label, path, alias))

            if not variants:
                sys.stderr.write("nothing to run: no adapters and --include-baseline not set\n")
                return 2

            with VLLMServer(server_cfg, on_line=_on_line) as server:
                base_url = server.base_url
                for variant_label, path, alias in variants:
                    _on_line(f"[discord-sft] === variant {variant_label} ===")
                    variant_spec = ModelSpec(
                        name_or_path=args.model,
                        backend="openai",
                        revision=args.revision,
                        adapter_path=path,
                        lora_alias=alias,
                        lora_rank=rank if alias else None,
                        dtype=args.dtype,
                        device=args.device,
                        trust_remote_code=not args.no_trust_remote_code,
                        extra_args=extra_model_args,
                    )
                    run = run_evals(
                        variant_spec,
                        tasks=tasks,
                        val_jsonl=args.val,
                        profile_json=args.profiles,
                        out_dir=args.out,
                        limit=args.limit,
                        batch_size=args.batch_size,
                        num_fewshot=args.num_fewshot,
                        label=variant_label,
                        seed=args.seed,
                        judge=judge,
                        on_line=_on_line,
                        server_base_url=base_url,
                        persona_max_concurrency=args.persona_max_concurrency,
                        baseline_prompt_mode=args.baseline_prompt,
                        extra_cli=lmms_extra_cli or None,
                        qwen_sampling=args.qwen_sampling,
                        gen_kwargs_cli=args.gen_kwargs,
                        apply_chat_template=effective_apply_chat_template,
                        vllm_extra=vllm_extra_argv or None,
                    )
                    run_ids.append(run["run_id"])
                    out_files.append(str(Path(args.out) / "runs" / f"{run['run_id']}.json"))
                    score_counts.append(len(run.get("scores", {})))
        else:
            # Legacy single-run path (hf backend, or vllm with --no-shared-server
            # and at most one adapter). We now correctly serialise the adapter
            # as ``lora_local_path=`` for vllm and ``peft=`` for hf.
            single_adapter = adapters_raw[0] if adapters_raw else None
            if single_adapter and "=" in single_adapter:
                # allow alias=path but drop the alias in the single-run case
                _, single_adapter = single_adapter.split("=", 1)
            spec = ModelSpec(
                name_or_path=args.model,
                backend=args.backend,
                revision=args.revision,
                adapter_path=single_adapter,
                dtype=args.dtype,
                device=args.device,
                trust_remote_code=not args.no_trust_remote_code,
                extra_args=extra_model_args,
                lora_rank=args.max_lora_rank,
            )
            run = run_evals(
                spec,
                tasks=tasks,
                val_jsonl=args.val,
                profile_json=args.profiles,
                out_dir=args.out,
                limit=args.limit,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
                label=args.label,
                seed=args.seed,
                judge=judge,
                on_line=_on_line,
                persona_max_concurrency=args.persona_max_concurrency,
                baseline_prompt_mode=args.baseline_prompt,
                extra_cli=lmms_extra_cli or None,
                qwen_sampling=args.qwen_sampling,
                gen_kwargs_cli=args.gen_kwargs,
                apply_chat_template=effective_apply_chat_template,
                vllm_extra=vllm_extra_argv or None,
            )
            run_ids.append(run["run_id"])
            out_files.append(str(Path(args.out) / "runs" / f"{run['run_id']}.json"))
            score_counts.append(len(run.get("scores", {})))
    finally:
        if args.lmms_disable_thinking is not None:
            if prev_disable_thinking is None:
                os.environ.pop("LMMS_EVAL_DISABLE_THINKING", None)
            else:
                os.environ["LMMS_EVAL_DISABLE_THINKING"] = prev_disable_thinking

    sys.stdout.write(
        json.dumps(
            {
                "run_ids": run_ids,
                "out_files": out_files,
                "n_scores": score_counts,
                "tasks": tasks,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _cmd_eval_list(args: argparse.Namespace) -> int:
    from discord_sft.evals.storage import list_runs

    rows = list_runs(args.dir)
    if args.format == "json":
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
        return 0
    if not rows:
        sys.stdout.write("(no runs found)\n")
        return 0
    headers = ["run_id", "created_utc", "label", "model", "scores"]
    lines = [headers]
    for r in rows:
        model = r.get("model") or {}
        model_str = model.get("name_or_path", "?")
        if model.get("adapter_path"):
            model_str += f" + {model['adapter_path']}"
        lines.append(
            [
                r["run_id"],
                r.get("created_utc") or "",
                r.get("label") or "",
                model_str,
                str(r.get("n_scores", 0)),
            ]
        )
    widths = [max(len(row[i]) for row in lines) for i in range(len(headers))]
    for i, row in enumerate(lines):
        sys.stdout.write("  ".join(c.ljust(widths[j]) for j, c in enumerate(row)) + "\n")
        if i == 0:
            sys.stdout.write("  ".join("-" * w for w in widths) + "\n")
    return 0


def _cmd_eval_compare(args: argparse.Namespace) -> int:
    from discord_sft.evals.compare import compare_runs, render_comparison

    refs: list = list(args.runs)
    baseline: int | str | None
    if args.baseline is None:
        baseline = 0
    elif args.baseline == "":
        baseline = None
    else:
        try:
            baseline = int(args.baseline)
        except ValueError:
            baseline = args.baseline
            if baseline not in refs:
                refs = [baseline] + refs

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
    report = compare_runs(
        refs,
        baseline=baseline,
        metrics=metrics,
        out_dir=args.out_dir,
        omit_persona_metrics=not args.include_persona_metrics,
    )
    sys.stdout.write(render_comparison(report, fmt=args.format) + "\n")
    return 0


def _cmd_eval_judge_persona(args: argparse.Namespace) -> int:
    from discord_sft.evals.judge import make_judge
    from discord_sft.evals.persona import judge_persona_generations_file

    judge = make_judge(args.judge)
    report = judge_persona_generations_file(args.generations, judge=judge)
    payload = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload + "\n")
    return 0


def _cmd_eval_rank_style(args: argparse.Namespace) -> int:
    from discord_sft.evals.pairwise_style import (
        OpenRouterPairwiseJudge,
        RankStyleConfig,
        _wrap_pairwise_judge,
        run_rank_style_eval,
    )

    gens = [Path(p) for p in args.generations]
    labs = list(args.labels) if args.labels else [p.stem for p in gens]
    if len(labs) != len(gens):
        sys.stderr.write("Repeat --label once per --generations (or omit for path stems).\n")
        return 2

    if args.skip_provenance_check:
        rj_paths: list[Path | None] = [None] * len(gens)
    elif args.provenance_from_rows:
        rj_paths = [None] * len(gens)
    else:
        rjs = list(args.run_json or [])
        if len(rjs) != len(gens):
            sys.stderr.write(
                "Pass one --run-json per --generations, or use "
                "--skip-provenance-check / --provenance-from-rows.\n"
            )
            return 2
        rj_paths = [Path(p) for p in rjs]

    pw = float(args.pairwise_weight)
    fw = float(args.fingerprint_weight)
    if abs(pw + fw - 1.0) > 1e-3:
        sys.stderr.write(
            f"Warning: pairwise_weight ({pw}) + fingerprint_weight ({fw}) != 1.0\n"
        )

    ckpt = Path(args.comparisons_checkpoint) if args.comparisons_checkpoint else None

    prior_map: dict[str, float] = {}
    pj = getattr(args, "prior_elo_json", None)
    if pj:
        pj_path = Path(pj)
        doc = json.loads(pj_path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            sys.stderr.write("--prior-elo-json must contain a JSON object.\n")
            return 2
        for k_raw, v_raw in doc.items():
            try:
                prior_map[str(k_raw)] = float(v_raw)
            except (TypeError, ValueError):
                sys.stderr.write(f"Ignoring non-numeric prior Elo for key {k_raw!r}\n")
    for spec in getattr(args, "prior_elos", None) or []:
        if "=" not in spec:
            sys.stderr.write(f"--prior-elo expects LABEL=RATING, got {spec!r}\n")
            return 2
        pk, pv = spec.split("=", 1)
        try:
            prior_map[pk.strip()] = float(pv.strip())
        except ValueError:
            sys.stderr.write(f"Invalid rating in --prior-elo: {spec!r}\n")
            return 2
    prior_elo_by_label = prior_map if prior_map else None

    cfg = RankStyleConfig(
        generations_paths=gens,
        labels=labs,
        run_json_paths=rj_paths,
        skip_provenance=bool(args.skip_provenance_check),
        provenance_from_rows=bool(args.provenance_from_rows),
        use_val_hash=bool(args.val_hash),
        pairs_per_prompt=int(args.pairs_per_prompt),
        seed=int(args.seed),
        judge_model=str(args.judge_model),
        judge_temperature=float(args.temperature),
        max_concurrency=int(args.max_concurrency),
        include_reference_style=not bool(args.no_reference_style),
        profiles_path=Path(args.profiles) if args.profiles else None,
        emit_comparisons=bool(args.emit_comparisons),
        comparisons_checkpoint_path=ckpt,
        pairwise_weight=pw,
        fingerprint_weight=fw,
        prior_elo_by_label=prior_elo_by_label,
    )

    judge = OpenRouterPairwiseJudge(
        cfg.judge_model,
        temperature=cfg.judge_temperature,
    )
    judge_fn = _wrap_pairwise_judge(
        judge,
        include_reference_style=cfg.include_reference_style,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload + "\n")
    return 0


def _cmd_eval_backfill_style_rank_elo(args: argparse.Namespace) -> int:
    from discord_sft.evals.style_rank_checkpoint_replay import (
        backfill_elo_store_from_checkpoints_dir,
    )

    summ, store_path = backfill_elo_store_from_checkpoints_dir(
        Path(args.dir),
        Path(args.checkpoints) if args.checkpoints else None,
        merge_existing=not bool(args.fresh_elo_store),
        elo_k=float(args.elo_k),
        elo_start=float(args.elo_start),
    )
    sys.stderr.write(f"Saved (or unchanged): `{store_path}`\n")
    sys.stdout.write(json.dumps(summ, indent=2, ensure_ascii=False) + "\n")
    return 0 if summ.get("ok") else 1
