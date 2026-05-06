"""Native persona evals: generate replies to held-out val.jsonl and score style.

This is the non-lmms-eval half of the harness. We:

1. Read ``val.jsonl`` from ``build-sft``; each line is a ShareGPT-style
   sample whose final turn is authored by the persona.
2. Strip the final assistant turn to form the prompt; the removed text is
   the *reference* completion for style heuristics.
3. Call a user-supplied ``generate_fn(prompts, system, **kwargs)`` to
   produce candidate completions — defaults to an HF ``model.generate``
   loop via :func:`discord_sft.evals.model.load_hf`, but any callable works
   (tests pass in a deterministic echo function).
4. Group by ``persona_id`` and score each group with the existing
   :func:`style_heuristics` / :func:`profile_heuristics`, optionally
   calling :class:`~discord_sft.evals.judge.OpenRouterJudge` per-sample for a fuller
   rating.

Scores come back as dotted-key floats ready to merge into the unified run
schema, e.g. ``persona.heuristics.<pid>.avg_length_diff`` and
``persona.judge.<pid>.overall``.
"""
from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from discord_sft.evals.qwen35_sampling import (
    DEFAULT_QWEN_SAMPLING,
    hf_generate_sampling_kwargs,
    openai_chat_sampling_kwargs,
    qwen35_preset,
)
from discord_sft.analysis.heuristics import profile_heuristics, style_heuristics
from discord_sft.data_prep.sft import Sample, read_samples


@dataclass
class PersonaEvalSample:
    persona_id: str
    persona_name: str
    system: str
    prompt_conversation: list[dict[str, str]]
    reference: str


@dataclass
class PersonaEvalResult:
    per_persona: dict[str, dict[str, Any]] = field(default_factory=dict)
    generations: list[dict[str, Any]] = field(default_factory=list)

    def to_flat_scores(self) -> dict[str, float]:
        """Emit dotted-key numeric scores for the run JSON."""
        out: dict[str, float] = {}
        for pid, blob in self.per_persona.items():
            heur = blob.get("heuristics") or {}
            for k, v in heur.items():
                if isinstance(v, (int, float)):
                    out[f"persona.heuristics.{pid}.{k}"] = float(v)
            judge = blob.get("judge") or {}
            for k, v in judge.items():
                if isinstance(v, (int, float)):
                    out[f"persona.judge.{pid}.{k}"] = float(v)
        return out


GenerateFn = Callable[[list[str], list[str], dict[str, Any]], list[str]]
"""Callable signature: ``(prompts, systems, gen_kwargs) -> completions``."""


def extract_eval_samples(
    samples: Iterable[Sample],
    *,
    limit_per_persona: int | None = None,
) -> list[PersonaEvalSample]:
    """Strip the final assistant turn from each sample; that's the reference.

    Samples whose final turn is not an assistant turn are skipped defensively
    (``build-sft`` guarantees this, but val files produced by older versions
    or hand-edited files may drift).
    """
    out: list[PersonaEvalSample] = []
    per_persona_count: dict[str, int] = {}
    for s in samples:
        conv = list(s.conversations)
        if not conv or conv[-1].get("from") != "assistant":
            continue
        reference = str(conv[-1].get("value", ""))
        prompt_conv = conv[:-1]
        if not prompt_conv:
            continue
        pid = str(s.meta.get("persona_id") or "unknown")
        if limit_per_persona is not None:
            if per_persona_count.get(pid, 0) >= limit_per_persona:
                continue
            per_persona_count[pid] = per_persona_count.get(pid, 0) + 1
        out.append(
            PersonaEvalSample(
                persona_id=pid,
                persona_name=str(s.meta.get("persona_name") or pid),
                system=s.system,
                prompt_conversation=prompt_conv,
                reference=reference,
            )
        )
    return out


def _build_chat_messages(
    prompt_conv: list[dict[str, str]], system: str
) -> list[dict[str, str]]:
    """Flatten a ShareGPT-style conversation into OpenAI chat messages."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in prompt_conv:
        role = "assistant" if turn.get("from") == "assistant" else "user"
        messages.append({"role": role, "content": str(turn.get("value", ""))})
    return messages


def make_openai_generate_fn(
    base_url: str,
    model_id: str,
    *,
    api_key: str = "EMPTY",
    max_concurrency: int = 16,
    max_new_tokens: int = 128,
    qwen_sampling: str = DEFAULT_QWEN_SAMPLING,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    disable_thinking_in_chat_template: bool | None = None,
    timeout_sec: float = 120.0,
    max_retries: int = 3,
) -> GenerateFn:
    """Build a persona ``generate_fn`` that targets a vLLM OpenAI server.

    When the CLI stands up a shared :class:`VLLMServer` for a multi-LoRA
    sweep, persona eval generation should hit that same process rather
    than loading another HF copy of the 35B base. This function returns a
    callable compatible with :data:`GenerateFn`.

    ``model_id`` is the server-side alias: either the base model id
    (``Qwen/Qwen3.5-35B-A3B``) or a LoRA alias registered via
    ``--lora-modules`` (``r8``, ``style-late``, etc.). Requests are
    dispatched concurrently up to ``max_concurrency`` in flight so vLLM's
    continuous batcher actually sees enough work to batch together;
    without that, throughput stays at roughly 1 req/s on 35B.

    Output order matches input order — tests rely on this.

    Sampling defaults follow the Hugging Face Qwen3.5-35B-A3B model card
    (preset ``qwen_sampling``). Pass ``temperature`` / ``top_p`` /
    ``top_k`` / ``presence_penalty`` explicitly to override the preset.
    vLLM-specific request fields go through ``extra_body`` so the call remains
    compatible with the real OpenAI Python SDK.
    """
    preset = qwen35_preset(qwen_sampling)
    oa = openai_chat_sampling_kwargs(preset)
    eff_temperature = float(oa["temperature"] if temperature is None else temperature)
    eff_top_p = float(oa["top_p"] if top_p is None else top_p)
    eff_top_k = int(oa["top_k"] if top_k is None else top_k)
    eff_presence = float(
        oa["presence_penalty"] if presence_penalty is None else presence_penalty
    )
    if disable_thinking_in_chat_template is None:
        disable_thinking_in_chat_template = qwen_sampling.startswith("instruct_")
    extra_body: dict[str, Any] = {"top_k": eff_top_k}
    if disable_thinking_in_chat_template:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "Install the evals extra: pip install 'discord-sft[evals]' "
            "(pulls openai)."
        ) from e

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)

    def _one(messages: list[dict[str, str]]) -> str:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=eff_temperature,
                    top_p=eff_top_p,
                    presence_penalty=eff_presence,
                    extra_body=extra_body,
                )
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()
                return text
            except Exception as e:
                last_exc = e
                # Exponential-ish backoff so a transient server hiccup
                # doesn't tank the run, but we also don't stall for long.
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"openai chat.completions failed after {max_retries} retries: {last_exc}"
        )

    def _gen(prompts: list[str], systems: list[str], gen_kwargs: dict[str, Any]) -> list[str]:
        _ = prompts  # unused: real content is in _conversations
        conversations: list[list[dict[str, str]]] = gen_kwargs.get("_conversations") or []
        if len(conversations) != len(systems):
            raise RuntimeError(
                "make_openai_generate_fn: conversation / system count mismatch "
                f"({len(conversations)} vs {len(systems)})"
            )
        all_messages = [
            _build_chat_messages(conv, sys_prompt)
            for conv, sys_prompt in zip(conversations, systems)
        ]
        if not all_messages:
            return []

        results: list[str | None] = [None] * len(all_messages)
        workers = max(1, min(max_concurrency, len(all_messages)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_one, msgs): idx
                for idx, msgs in enumerate(all_messages)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        # All slots should now be populated; narrow the type.
        return [r or "" for r in results]

    return _gen


def default_hf_generate_fn(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 128,
    qwen_sampling: str = DEFAULT_QWEN_SAMPLING,
    disable_thinking_in_chat_template: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
) -> GenerateFn:
    """Build a ``generate_fn`` using HF ``apply_chat_template`` + ``model.generate``.

    We batch one sample at a time for simplicity; persona eval sets are
    small (a few hundred val samples max) so throughput is not the
    bottleneck — model load time dominates.

    When ``disable_thinking_in_chat_template`` is true, we pass
    ``chat_template_kwargs={"enable_thinking": False}`` to match the
    model card's *instruct / non-thinking* path together with preset
    ``instruct_general``. If the tokenizer rejects that argument, we
    fall back to the template without those kwargs.
    """
    import torch

    preset = qwen35_preset(qwen_sampling)
    hf_kw = hf_generate_sampling_kwargs(preset)
    eff_temperature = float(hf_kw["temperature"] if temperature is None else temperature)
    eff_top_p = float(hf_kw["top_p"] if top_p is None else top_p)
    eff_top_k = int(hf_kw["top_k"] if top_k is None else top_k)
    eff_rep = float(
        hf_kw["repetition_penalty"] if repetition_penalty is None else repetition_penalty
    )

    def _gen(prompts: list[str], systems: list[str], gen_kwargs: dict[str, Any]) -> list[str]:
        _ = prompts  # prompts unused: we build messages ourselves
        outs: list[str] = []
        for p_conv, sys_prompt in zip(gen_kwargs["_conversations"], systems):
            messages = [{"role": "system", "content": sys_prompt}]
            for turn in p_conv:
                role = "assistant" if turn["from"] == "assistant" else "user"
                messages.append({"role": role, "content": turn["value"]})
            tmpl_kwargs: dict[str, Any] = {
                "messages": messages,
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if disable_thinking_in_chat_template:
                tmpl_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                text = tokenizer.apply_chat_template(**tmpl_kwargs)
            except TypeError:
                tmpl_kwargs.pop("chat_template_kwargs", None)
                text = tokenizer.apply_chat_template(**tmpl_kwargs)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=eff_temperature > 0,
                    temperature=max(eff_temperature, 1e-5),
                    top_p=eff_top_p,
                    top_k=eff_top_k,
                    repetition_penalty=eff_rep,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
            outs.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
        return outs

    return _gen


SystemOverrideFn = Callable[
    ["PersonaEvalSample", dict[str, Any] | None], str
]
"""Callable signature: ``(sample, persona_profile) -> replacement_system_str``."""


def run_persona_evals(
    val_jsonl: str | Path,
    generate_fn: GenerateFn,
    *,
    profile_json: str | Path | None = None,
    limit_per_persona: int | None = None,
    judge: Any = None,
    gen_kwargs: dict[str, Any] | None = None,
    system_override_fn: SystemOverrideFn | None = None,
    eval_val_jsonl: str | None = None,
    eval_val_sha256: str | None = None,
) -> PersonaEvalResult:
    """Generate persona replies to held-out val samples and score them.

    Parameters
    ----------
    val_jsonl:
        Path to ``val.jsonl`` produced by ``discord-sft build-sft``.
    generate_fn:
        Callable that returns one completion per sample. In production this
        wraps ``model.generate``; in tests pass an echo / dummy function.
    profile_json:
        Optional ``profiles.json`` from ``discord-sft fingerprint``; when
        provided, per-persona mined filler n-grams seed the style heuristics.
    limit_per_persona:
        Cap samples per persona to keep smoke-tests fast.
    judge:
        Optional object implementing the :class:`StyleJudge` interface
        (i.e. has a ``.score(real_message=..., generated_message=...,
        context=..., persona_name=...)`` method). If provided, every sample
        gets a 1-5 rating dict averaged per persona.
    gen_kwargs:
        Extra kwargs forwarded to ``generate_fn``.
    system_override_fn:
        Optional hook to rewrite the system prompt *per sample* before it
        is passed to ``generate_fn``. Receives the sample and, when
        ``profile_json`` is loaded, that persona's entry (or ``None``). Used
        by the baseline-prompt feature so no-adapter runs can inject
        style-aware instructions without mutating the val.jsonl. The
        override does not affect the heuristics / judge references (those
        still come from the ground-truth assistant turn).
    eval_val_jsonl / eval_val_sha256:
        When set, each row in ``generations`` includes these fields for
        cross-run provenance (``discord-sft eval rank-style``).
    """
    import json as _json

    samples = read_samples(Path(val_jsonl))
    eval_samples = extract_eval_samples(samples, limit_per_persona=limit_per_persona)

    profile_doc: dict | None = None
    if profile_json is not None:
        p = Path(profile_json)
        if p.exists():
            profile_doc = _json.loads(p.read_text(encoding="utf-8"))

    if system_override_fn is not None:
        personas_blob = (profile_doc or {}).get("personas") or {}
        for s in eval_samples:
            persona_profile = personas_blob.get(s.persona_id)
            s.system = system_override_fn(s, persona_profile)

    kwargs = dict(gen_kwargs or {})
    kwargs["_conversations"] = [s.prompt_conversation for s in eval_samples]

    prompts = [_render_prompt(s.prompt_conversation) for s in eval_samples]
    systems = [s.system for s in eval_samples]
    generations = generate_fn(prompts, systems, kwargs) if eval_samples else []
    if len(generations) != len(eval_samples):
        raise RuntimeError(
            f"generate_fn returned {len(generations)} completions for "
            f"{len(eval_samples)} samples"
        )

    result = PersonaEvalResult()
    grouped: dict[str, list[tuple[PersonaEvalSample, str]]] = {}
    for s, g in zip(eval_samples, generations):
        grouped.setdefault(s.persona_id, []).append((s, g))
        row: dict[str, Any] = {
            "persona_id": s.persona_id,
            "persona_name": s.persona_name,
            "reference": s.reference,
            "generated": g,
            "system": s.system,
            "context_turns": s.prompt_conversation,
            "context": _render_prompt(s.prompt_conversation),
        }
        if eval_val_jsonl:
            row["eval_val_jsonl"] = eval_val_jsonl
        if eval_val_sha256:
            row["eval_val_sha256"] = eval_val_sha256
        result.generations.append(row)

    for pid, pairs in grouped.items():
        refs = [ref.reference for ref, _ in pairs]
        gens = [g for _, g in pairs]
        persona_profile = None
        if profile_doc is not None:
            persona_profile = (profile_doc.get("personas") or {}).get(pid)
        if persona_profile:
            heur = profile_heuristics(gens, refs, profile=persona_profile)
        else:
            heur = style_heuristics(gens, refs)
        blob: dict[str, Any] = {"heuristics": heur, "n_samples": len(pairs)}
        if judge is not None:
            blob["judge"] = _score_with_judge(judge, pairs)
        result.per_persona[pid] = blob

    return result


def _render_prompt(prompt_conv: list[dict[str, str]]) -> str:
    """Cheap fallback prompt rendering used only when the ``generate_fn`` is
    modality-unaware (e.g. the echo dummy in tests). The real HF path ignores
    this and rebuilds messages via ``apply_chat_template``."""
    lines = []
    for turn in prompt_conv:
        role = turn.get("from", "user")
        lines.append(f"[{role}] {turn.get('value', '')}")
    return "\n".join(lines)


def _score_with_judge(judge: Any, pairs: list[tuple[PersonaEvalSample, str]]) -> dict[str, float]:
    """Average the judge's per-sample dimension scores across the persona."""
    accum: dict[str, list[float]] = {}
    errors = 0
    for s, g in pairs:
        context_text = _render_prompt(s.prompt_conversation)
        try:
            rating = judge.score(
                real_message=s.reference,
                generated_message=g,
                context=context_text,
                persona_name=s.persona_name,
                system_instructions=(s.system or "").strip() or None,
            )
        except Exception:
            errors += 1
            continue
        for k, v in rating.items():
            if isinstance(v, (int, float)):
                accum.setdefault(k, []).append(float(v))
    out = {k: statistics.fmean(vals) for k, vals in accum.items() if vals}
    if errors:
        out["errors"] = float(errors)
    return out


def judge_persona_generations_file(
    generations_jsonl: str | Path,
    *,
    judge: Any,
) -> dict[str, Any]:
    """Run the style judge over an existing ``persona_generations.jsonl`` file.

    This enables a judge-only pass outside the full eval loop (no regeneration,
    no lmms-eval subprocess). Expected row schema is the file written by
    :func:`run_evals`:
    ``persona_id``, ``persona_name``, ``reference``, ``generated``, ``system``.
    If ``context`` is present, it is the thread for the judge and ``system`` is
    passed as instructions-only; if ``context`` is absent, ``system`` is used
    as the context fallback (no duplicate instructions block).
    """
    path = Path(generations_jsonl)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pid = str(row.get("persona_id") or "unknown")
        grouped.setdefault(pid, []).append(row)

    per_persona: dict[str, dict[str, Any]] = {}
    all_axis_values: dict[str, list[float]] = {}
    total_errors = 0
    total_scored = 0

    for pid, prows in grouped.items():
        persona_name = str(prows[0].get("persona_name") or pid)
        axis_values: dict[str, list[float]] = {}
        errors = 0
        scored = 0
        for row in prows:
            try:
                ctx_raw = row.get("context")
                if ctx_raw is not None and str(ctx_raw).strip():
                    context_str = str(ctx_raw)
                    system_inst = (str(row.get("system") or "").strip() or None)
                else:
                    context_str = str(row.get("system") or "")
                    system_inst = None
                rating = judge.score(
                    real_message=str(row.get("reference") or ""),
                    generated_message=str(row.get("generated") or ""),
                    context=context_str,
                    persona_name=str(row.get("persona_name") or persona_name),
                    system_instructions=system_inst,
                )
            except Exception:
                errors += 1
                continue
            scored += 1
            for k, v in rating.items():
                if isinstance(v, (int, float)):
                    fv = float(v)
                    axis_values.setdefault(k, []).append(fv)
                    all_axis_values.setdefault(k, []).append(fv)

        total_errors += errors
        total_scored += scored
        persona_blob: dict[str, Any] = {
            "persona_name": persona_name,
            "n_samples": len(prows),
            "n_scored": scored,
            "errors": errors,
            "judge": {k: statistics.fmean(vals) for k, vals in axis_values.items() if vals},
        }
        per_persona[pid] = persona_blob

    overall = {k: statistics.fmean(vals) for k, vals in all_axis_values.items() if vals}
    return {
        "generations_path": str(path),
        "n_personas": len(per_persona),
        "n_samples": len(rows),
        "n_scored": total_scored,
        "errors": total_errors,
        "overall": overall,
        "per_persona": per_persona,
    }
