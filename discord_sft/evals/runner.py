"""Top-level ``run_evals`` orchestrator.

Composes standard benchmarks (via lmms-eval subprocess) and native persona
evals into a single unified run JSON artifact.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from discord_sft import __version__ as _pkg_version
from discord_sft.evals.benchmarks import (
    BENCHMARKS,
    DEFAULT_TASKS,
    PERSONA_KEY,
    split_tasks,
)
from discord_sft.evals.harness import run_lmms_eval
from discord_sft.evals.model import ModelSpec
from discord_sft.evals.paths import (
    merge_training_run_id_from_model_path,
    resolve_checkpoint_dir,
    resolve_lora_training_run_dir,
)
from discord_sft.evals.qwen35_sampling import DEFAULT_QWEN_SAMPLING, qwen35_preset
from discord_sft.evals.persona import (
    GenerateFn,
    default_hf_generate_fn,
    make_openai_generate_fn,
    run_persona_evals,
)
from discord_sft.evals.storage import (
    raw_dir,
    run_id_for,
    save_run,
    utc_now_stamp,
)


def _env_snapshot() -> dict[str, str]:
    """Best-effort capture of eval-relevant library versions, for the run JSON."""
    env = {
        "discord_sft": _pkg_version,
        "python": platform.python_version(),
    }
    for mod in ("lmms_eval", "transformers", "torch", "peft", "accelerate", "vllm"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            continue
    return env


def _current_uv_lock_sha256() -> str | None:
    """Return the SHA256 of the active repo's ``uv.lock``, or ``None``.

    Mirrors the training-side fingerprint (``discord_sft.training.callbacks``)
    so eval can compare the current runtime lockfile against the one a
    checkpoint was trained under. Failure to find or hash the file is silent
    — provenance is a best-effort signal, not a correctness prerequisite.
    """
    import hashlib

    cur = Path.cwd().resolve()
    for parent in (cur, *cur.parents):
        cand = parent / "uv.lock"
        if cand.is_file():
            try:
                h = hashlib.sha256()
                with cand.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 16), b""):
                        h.update(chunk)
                return h.hexdigest()
            except OSError:
                return None
    return None


def _load_training_config(adapter_path: str | Path | None) -> dict[str, Any] | None:
    """Best-effort provenance bundle for a LoRA checkpoint at ``adapter_path``.

    Reads the training subpackage's ``config.resolved.yaml`` (the LoRA
    recipe) and, when available, the run-level ``run.json`` (the environment
    snapshot, including ``uv_lock.sha256``). Both files are walked for at
    the adapter path and one directory up so ``out/lora/<run>/epoch-2``
    works alongside ``out/lora/<run>/``.

    The returned dict always carries:

    * ``__path__`` — the resolved-config file that was loaded.
    * ``uv_lock`` — ``{"training": <sha>, "current": <sha>, "matches": bool}``
      when a ``run.json`` and current ``uv.lock`` are both present. Mismatches
      surface at eval time as a warning and as a field in the saved run JSON,
      so score deltas can be attributed to env drift rather than LoRA changes.

    Returns ``None`` if no ``config.resolved.yaml`` is found — baseline runs,
    checkpoints produced outside this repo, etc.
    """
    if not adapter_path:
        return None
    p = Path(adapter_path)
    cfg_candidates = [p / "config.resolved.yaml", p.parent / "config.resolved.yaml"]
    run_candidates = [p / "run.json", p.parent / "run.json"]

    cfg_file: Path | None = next((c for c in cfg_candidates if c.exists()), None)
    if cfg_file is None:
        return None

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return {"__path__": str(cfg_file.resolve()), "__error__": "pyyaml not installed"}

    try:
        cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return {"__path__": str(cfg_file.resolve()), "__error__": str(e)}

    data: dict[str, Any] = {"__path__": str(cfg_file.resolve())}
    if isinstance(cfg_data, dict):
        data.update(cfg_data)

    run_file = next((c for c in run_candidates if c.exists()), None)
    if run_file is not None:
        try:
            import json as _json

            run_manifest = _json.loads(run_file.read_text(encoding="utf-8"))
        except Exception:
            run_manifest = None
        if isinstance(run_manifest, dict):
            training_lock = ((run_manifest.get("uv_lock") or {}).get("sha256"))
            current_lock = _current_uv_lock_sha256()
            data["uv_lock"] = {
                "training": training_lock,
                "current": current_lock,
                "matches": (
                    training_lock is not None
                    and current_lock is not None
                    and training_lock == current_lock
                ),
            }
            # Shallow git-SHA echo too, so eval runs don't have to open
            # run.json separately to answer "what commit produced this?".
            if run_manifest.get("git_sha"):
                data.setdefault("git_sha", run_manifest["git_sha"])

    return data


def _lora_yaml_run_name_matches_hint(tc: dict[str, Any], hint: str) -> bool:
    rn = tc.get("run_name")
    if rn is None or rn == "":
        return True
    if isinstance(rn, str):
        return rn.strip() == hint.strip()
    return True


def _training_config_via_lora_merge_path_hint(model_path_str: str) -> dict[str, Any] | None:
    """If ``model_path_str`` is ``out/merged/<training_run>/…``, load ``out/lora/<training_run>/`` YAML."""
    hint = merge_training_run_id_from_model_path(model_path_str)
    if hint is None:
        return None
    ldir = resolve_lora_training_run_dir(hint)
    if ldir is None:
        return None
    tc = _load_training_config(ldir)
    if tc is None:
        return None
    if not _lora_yaml_run_name_matches_hint(tc, hint):
        return None
    return tc


def _training_config_provenance(
    name_or_path: str | Path | None,
    adapter_path: str | Path | None,
    *,
    on_line: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Prefer adapter checkpoint; then local ``name_or_path`` dir; then ``merge_manifest``."""
    ap_dir = resolve_checkpoint_dir(adapter_path)
    tc = _load_training_config(ap_dir if ap_dir is not None else None)
    if tc is not None:
        return tc, "adapter_path"

    if not name_or_path:
        return None, "none"
    sop = str(name_or_path).strip()
    if not sop:
        return None, "none"

    def _fallback_or(reason: str) -> tuple[dict[str, Any] | None, str]:
        tc_hint = _training_config_via_lora_merge_path_hint(sop)
        if tc_hint is not None:
            return tc_hint, "lora_run_dir.merge_path_hint"
        return None, reason

    p = resolve_checkpoint_dir(sop)

    if p is not None:
        tc = _load_training_config(p)
        if tc is not None:
            return tc, "model_dir"

        mm = p / "merge_manifest.json"
        if not mm.is_file():
            return _fallback_or("model_dir_no_yaml")
        try:
            raw_man = json.loads(mm.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _fallback_or("merge_manifest_unreadable")
        if not isinstance(raw_man, dict):
            return _fallback_or("merge_manifest_invalid")
        adir = raw_man.get("adapter_dir")
        if not isinstance(adir, str) or not adir.strip():
            return _fallback_or("merge_manifest_bad_adapter_dir")
        adir_dir = resolve_checkpoint_dir(adir.strip())
        if adir_dir is None:
            return _fallback_or("merge_manifest_adapter_missing")
        tc_m = _load_training_config(adir_dir)
        if tc_m is not None:
            return tc_m, "merge_manifest.adapter_dir"
        if on_line is not None:
            on_line(
                "[discord-sft] WARNING: merge_manifest adapter_dir exists but "
                f"training config YAML not readable at {adir.strip()!r}"
            )
        return _fallback_or("merge_manifest_adapter_no_yaml")

    return _fallback_or("model_path_missing")


def load_training_config_provenance(
    name_or_path: str | Path | None,
    adapter_path: str | Path | None,
    *,
    on_line: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Load LoRA provenance YAML for adapter runs **or** merged local model dirs."""
    tc, _ = _training_config_provenance(name_or_path, adapter_path, on_line=on_line)
    return tc


def training_config_provenance_report(
    name_or_path: str | Path | None,
    adapter_path: str | Path | None,
) -> tuple[dict[str, Any] | None, str]:
    """Return `(training_config_or_none, source_tag)` without stderr warnings."""
    return _training_config_provenance(name_or_path, adapter_path, on_line=None)


def _run_lmms_block(
    spec: ModelSpec,
    *,
    lmms_tasks: list[str],
    raw_run_dir: Path,
    limit: int | None,
    batch_size: int | str,
    num_fewshot: int | None,
    on_line: Callable[[str], None] | None,
    extra_cli: list[str] | None,
    server_base_url: str | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run lmms-eval tasks and return (scores, harness_meta)."""
    if not lmms_tasks:
        return {}, {}
    harness = run_lmms_eval(
        spec,
        lmms_tasks,
        output_path=raw_run_dir,
        limit=limit,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        on_line=on_line,
        extra_cli=extra_cli or [],
        server_base_url=server_base_url,
    )
    scores = dict(harness.get("scores", {}))
    meta = {
        "raw_results_path": harness.get("raw_results_path"),
        "results_json": harness.get("results_json"),
        "cmd": harness.get("cmd"),
    }
    return scores, meta


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_persona_block(
    spec: ModelSpec,
    *,
    val_jsonl: str | Path,
    profile_json: str | Path | None,
    raw_run_dir: Path,
    limit: int | None,
    judge: Any,
    generate_fn: GenerateFn | None,
    server_base_url: str | None,
    persona_max_concurrency: int,
    baseline_prompt_mode: str,
    qwen_sampling: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run native persona evals and return (scores, persona_meta)."""
    from discord_sft.evals.baseline_prompt import build_baseline_system_prompt

    gen_fn = generate_fn
    if gen_fn is None:
        if server_base_url is not None:
            gen_fn = make_openai_generate_fn(
                server_base_url,
                spec.lora_alias or spec.name_or_path,
                max_concurrency=persona_max_concurrency,
                qwen_sampling=qwen_sampling,
            )
        else:
            from discord_sft.evals.model import load_hf

            model, tokenizer = load_hf(spec)
            gen_fn = default_hf_generate_fn(
                model,
                tokenizer,
                qwen_sampling=qwen_sampling,
            )
    per_persona_limit = None
    if limit is not None:
        per_persona_limit = max(1, limit // 4)

    is_base_model = not spec.adapter_path and not spec.lora_alias
    effective_mode = baseline_prompt_mode if is_base_model else "minimal"
    system_override_fn = None
    if is_base_model and baseline_prompt_mode != "minimal":

        def system_override_fn(sample, persona_profile):  # noqa: E306
            return build_baseline_system_prompt(
                baseline_prompt_mode,  # type: ignore[arg-type]
                persona_name=sample.persona_name,
                minimal_system=sample.system,
                profile=persona_profile,
            )

    val_path = Path(val_jsonl).expanduser()
    try:
        val_resolved = str(val_path.resolve(strict=False))
    except OSError:
        val_resolved = str(val_path)
    val_sha = _file_sha256(val_path) if val_path.is_file() else None

    persona_result = run_persona_evals(
        val_jsonl,
        gen_fn,
        profile_json=profile_json,
        limit_per_persona=per_persona_limit,
        judge=judge,
        system_override_fn=system_override_fn,
        eval_val_jsonl=val_resolved,
        eval_val_sha256=val_sha,
    )
    scores = persona_result.to_flat_scores()
    gen_log = raw_run_dir / "persona_generations.jsonl"
    _dump_persona_generations(persona_result, gen_log)
    meta = {
        "n_personas": len(persona_result.per_persona),
        "n_generations": len(persona_result.generations),
        "generations_path": str(gen_log),
        "baseline_prompt_mode": effective_mode,
        "is_base_model": is_base_model,
        "qwen_sampling": qwen_sampling,
    }
    return scores, meta


def run_evals(
    spec: ModelSpec,
    *,
    tasks: list[str] | None = None,
    val_jsonl: str | Path | None = None,
    profile_json: str | Path | None = None,
    out_dir: str | Path = "out/evals",
    limit: int | None = None,
    batch_size: int | str = 1,
    label: str | None = None,
    seed: int = 0,
    judge: Any = None,
    generate_fn: GenerateFn | None = None,
    on_line: Callable[[str], None] | None = None,
    extra_cli: list[str] | None = None,
    num_fewshot: int | None = None,
    server_base_url: str | None = None,
    persona_max_concurrency: int = 16,
    baseline_prompt_mode: str = "minimal",
    qwen_sampling: str = DEFAULT_QWEN_SAMPLING,
    gen_kwargs_cli: str | None = None,
    apply_chat_template: bool | None = None,
    vllm_extra: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full benchmark suite and persist one unified JSON artifact.

    Parameters
    ----------
    spec:
        Model under test. For text-only benchmarks (IFEval + persona)
        ``backend="hf"`` is fine; for VLM tasks use ``vllm``, ``sglang``,
        or a dedicated wrapper like ``qwen3_vl``.
    tasks:
        Short keys from :data:`BENCHMARKS`. Defaults to all MVP tasks.
        Use ``"persona"`` to run the native persona evals.
    val_jsonl, profile_json:
        Paths produced by ``discord-sft build-sft`` / ``fingerprint``.
        Required only if ``"persona"`` is in ``tasks``.
    out_dir:
        Run JSON lands in ``<out_dir>/runs/<run_id>.json``; lmms-eval raw
        results go to ``<out_dir>/raw/<run_id>/``.
    limit, batch_size, num_fewshot:
        Forwarded to lmms-eval.
    judge:
        Optional ``StyleJudge`` for persona evals.
    generate_fn:
        Persona generation callable. Defaults to a fresh HF model loaded
        from ``spec``; pass a stub for testing or to share a pre-loaded
        model across runs.
    on_line:
        Callback for each subprocess stdout line — used by the UI to show
        live progress.
    server_base_url:
        If set, route both lmms-eval and persona generation through a
        local vLLM OpenAI-compatible server at this ``/v1`` base URL. The
        CLI uses this when sweeping multiple LoRAs against one shared
        base model load.
    persona_max_concurrency:
        Max in-flight persona requests when talking to the vLLM server;
        higher values let continuous batching do its job but risk OOM on
        small KV caches.
    baseline_prompt_mode:
        When ``spec.adapter_path`` and ``spec.lora_alias`` are both unset
        (i.e. a base-model run with no LoRA), rewrite the persona system
        prompt via :func:`discord_sft.evals.baseline_prompt.build_baseline_system_prompt`
        with this mode. One of ``"minimal"`` (default; use the val.jsonl
        prompt verbatim), ``"style"`` (generic Discord-DM style bullets),
        or ``"profile"`` (rich per-persona bullets from
        ``profiles.json``). Adapter runs always use the val.jsonl prompt
        because that's what they were trained under. The chosen mode is
        recorded in the run JSON under ``persona_meta.baseline_prompt_mode``.
    qwen_sampling:
        Hugging Face Qwen3.5-35B-A3B sampling preset for persona generation
        (see :mod:`discord_sft.evals.qwen35_sampling`). Ignored when a custom
        ``generate_fn`` is supplied.
    gen_kwargs_cli:
        Raw ``--gen_kwargs`` string forwarded to lmms-eval (also stored in
        run JSON ``config``). Does not affect persona sampling.
    apply_chat_template:
        Whether lmms-eval was invoked with ``--apply_chat_template``.
        Recorded in run JSON ``config`` for provenance.
    vllm_extra:
        Parsed ``--vllm-extra`` argv fragments for the shared vLLM server, and
        echoed in run JSON ``config`` for reproducibility.
    """
    from discord_sft.evals.baseline_prompt import BASELINE_PROMPT_MODES

    _ = qwen35_preset(qwen_sampling)

    if baseline_prompt_mode not in BASELINE_PROMPT_MODES:
        raise ValueError(
            f"baseline_prompt_mode must be one of {BASELINE_PROMPT_MODES}, "
            f"got {baseline_prompt_mode!r}"
        )
    task_keys = list(tasks) if tasks is not None else list(DEFAULT_TASKS)
    lmms_tasks, include_persona = split_tasks(task_keys)

    stamp = utc_now_stamp()
    rid = run_id_for(spec.slug(), label, stamp=stamp)
    raw_run_dir = raw_dir(out_dir) / rid
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {}
    lmms_scores, harness_meta = _run_lmms_block(
        spec,
        lmms_tasks=lmms_tasks,
        raw_run_dir=raw_run_dir,
        limit=limit,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        on_line=on_line,
        extra_cli=extra_cli,
        server_base_url=server_base_url,
    )
    scores.update(lmms_scores)

    persona_meta: dict[str, Any] = {}
    if include_persona:
        if val_jsonl is None:
            raise ValueError(
                "persona eval requested but val_jsonl is None; "
                "point --val at out/sft/val.jsonl"
            )
        persona_scores, persona_meta = _run_persona_block(
            spec,
            val_jsonl=val_jsonl,
            profile_json=profile_json,
            raw_run_dir=raw_run_dir,
            limit=limit,
            judge=judge,
            generate_fn=generate_fn,
            server_base_url=server_base_url,
            persona_max_concurrency=persona_max_concurrency,
            baseline_prompt_mode=baseline_prompt_mode,
            qwen_sampling=qwen_sampling,
        )
        scores.update(persona_scores)

    training_config = load_training_config_provenance(
        spec.name_or_path,
        spec.adapter_path,
        on_line=on_line,
    )

    run = {
        "run_id": rid,
        "created_utc": stamp.replace("-", ":").replace("T", "T").replace("Z", "Z"),
        "label": label,
        "model": spec.to_dict(),
        "training_config": training_config,
        "config": {
            "seed": seed,
            "limit": limit,
            "batch_size": batch_size,
            "num_fewshot": num_fewshot,
            "tasks": task_keys,
            "lmms_tasks": lmms_tasks,
            "include_persona": include_persona,
            "val_jsonl": str(val_jsonl) if val_jsonl else None,
            "profile_json": str(profile_json) if profile_json else None,
            "judge_backend": type(judge).__name__ if judge is not None else None,
            "server_base_url": server_base_url,
            "persona_max_concurrency": persona_max_concurrency if include_persona else None,
            "qwen_sampling": qwen_sampling,
            "gen_kwargs": gen_kwargs_cli,
            "apply_chat_template": apply_chat_template,
            "vllm_extra": list(vllm_extra) if vllm_extra else None,
        },
        "benchmark_descriptions": {
            k: BENCHMARKS[k].description for k in task_keys if k in BENCHMARKS
        },
        "scores": scores,
        "env": _env_snapshot(),
        "raw_results_path": str(raw_run_dir),
        "harness": harness_meta,
        "persona": persona_meta,
    }
    _iso = _to_iso(stamp)
    run["created_utc"] = _iso

    save_run(run, out_dir)
    return run


def _dump_persona_generations(result: Any, path: Path) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in result.generations:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _to_iso(stamp: str) -> str:
    """Turn our filesystem-safe ``2026-04-21T14-32-05Z`` back into strict ISO-8601."""
    try:
        date_part, time_part = stamp.split("T", 1)
        if time_part.endswith("Z"):
            time_part = time_part[:-1]
        hh, mm, ss = time_part.split("-")
        return f"{date_part}T{hh}:{mm}:{ss}Z"
    except Exception:
        return stamp


__all__ = ["run_evals"]


if __name__ == "__main__":  # pragma: no cover
    sys.stderr.write(
        "discord_sft.evals.runner is a library module; use `discord-sft eval run`.\n"
    )
    sys.exit(2)
