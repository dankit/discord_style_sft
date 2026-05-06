from __future__ import annotations

import json
from pathlib import Path

import pytest

from discord_sft.evals.baseline_prompt import (
    BASELINE_PROMPT_MODES,
    build_baseline_system_prompt,
)
from discord_sft.evals.persona import run_persona_evals
from discord_sft.data_prep.sft import Sample, write_samples


def _minimal_sample(pid: str, user_text: str, assistant_text: str) -> Sample:
    return Sample(
        system=f"You are Name-{pid} chatting with Other.",
        conversations=[
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
        meta={
            "persona_id": pid,
            "persona_name": f"Name-{pid}",
            "session_id": f"s-{pid}",
        },
    )


# ---------------------------------------------------------------------------
# build_baseline_system_prompt
# ---------------------------------------------------------------------------


def test_minimal_mode_is_verbatim_passthrough():
    out = build_baseline_system_prompt(
        "minimal",
        persona_name="Alice",
        minimal_system="You are Alice chatting with Bob.",
    )
    assert out == "You are Alice chatting with Bob."


def test_style_mode_adds_judge_axis_guidance():
    out = build_baseline_system_prompt(
        "style",
        persona_name="Alice",
        minimal_system="You are Alice chatting with Bob.",
    )
    assert out.startswith("You are Alice chatting with Bob.")
    # Covers every judge axis + heuristic at least implicitly.
    assert "short" in out.lower()                      # LENGTH / avg_length_diff
    assert "lowercase" in out.lower()                  # AUTHENTICITY / caps_rate
    assert '"lol"' in out or "lol" in out              # VOCABULARY / filler_rate
    assert "emoji" in out.lower()
    assert "as an ai" in out.lower()
    assert "Alice" in out                              # PERSONA FIT


def test_style_mode_reconstructs_header_without_minimal_system():
    out = build_baseline_system_prompt(
        "style",
        persona_name="Alice",
        counterparty_names=["Bob", "Cara"],
    )
    assert "You are Alice chatting with Bob, Cara" in out


def test_profile_mode_bakes_in_persona_fields():
    profile = {
        "length": {"mean_words": 7.4, "p50": 6, "p95": 22},
        "lowercase_start_rate": 0.83,
        "emoji": {"unicode_per_turn": 0.42, "custom_per_turn": 0.12},
        "burst_rate": 2.3,
        "top_fillers": {
            "1gram": [
                {"token": "imma", "score": 5.2},
                {"token": "lowkey", "score": 3.1},
                {"token": "fr", "score": 2.9},
            ]
        },
    }
    out = build_baseline_system_prompt(
        "profile",
        persona_name="Alice",
        minimal_system="You are Alice chatting with Bob.",
        profile=profile,
    )
    # Per-persona mined fillers should show up ahead of generic defaults.
    assert "imma" in out
    assert "lowkey" in out
    # Length bucket numbers.
    assert "median 6" in out
    assert "95th percentile 22" in out
    # Lowercase rate rendered as percent.
    assert "83%" in out
    # Emoji density stats.
    assert "0.42" in out
    assert "0.12" in out
    # Burst rate triggers at 2.3 (>1.05 threshold).
    assert "2.3" in out
    # Semantic guidance is included alongside raw numbers.
    assert "very short" in out
    assert "usually" in out
    assert "moderate" in out
    assert "multi-line bursts" in out


def test_profile_mode_tolerates_missing_fields():
    profile = {"length": {"mean_words": 5.0}}  # Partial profile
    out = build_baseline_system_prompt(
        "profile",
        persona_name="Alice",
        minimal_system="You are Alice chatting with Bob.",
        profile=profile,
    )
    assert out.startswith("You are Alice chatting with Bob.")
    assert "~5 words on average" in out


def test_profile_mode_falls_back_to_style_when_profile_empty():
    minimal = "You are Alice chatting with Bob."
    with_profile = build_baseline_system_prompt(
        "profile",
        persona_name="Alice",
        minimal_system=minimal,
        profile=None,
    )
    style_only = build_baseline_system_prompt(
        "style", persona_name="Alice", minimal_system=minimal
    )
    assert with_profile == style_only


def test_profile_mode_reads_legacy_flat_fillers():
    profile = {"top_fillers_1gram": [["imma", 5.0], ["lowkey", 4.0]]}
    out = build_baseline_system_prompt(
        "profile",
        persona_name="Alice",
        minimal_system="You are Alice chatting with Bob.",
        profile=profile,
    )
    assert "imma" in out
    assert "lowkey" in out


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        build_baseline_system_prompt(
            "bogus",                # type: ignore[arg-type]
            persona_name="Alice",
        )


def test_every_declared_mode_produces_prompt_with_persona_name():
    for mode in BASELINE_PROMPT_MODES:
        out = build_baseline_system_prompt(
            mode,                   # type: ignore[arg-type]
            persona_name="Alice",
            minimal_system="You are Alice chatting with Bob.",
            profile={
                "length": {"mean_words": 5.0, "p50": 4, "p95": 15},
                "top_fillers": {"1gram": [{"token": "imma", "score": 3.0}]},
            },
        )
        assert "Alice" in out


# ---------------------------------------------------------------------------
# run_persona_evals wiring
# ---------------------------------------------------------------------------


def test_run_persona_evals_applies_system_override(tmp_path: Path):
    samples = [
        _minimal_sample("A", "hi", "ya"),
        _minimal_sample("B", "sup", "hey"),
    ]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    captured_systems: list[str] = []

    def _capture_fn(prompts, systems, gen_kwargs):
        captured_systems.extend(systems)
        return [f"reply-{i}" for i in range(len(prompts))]

    def _override(sample, persona_profile):
        _ = persona_profile
        return f"OVERRIDE for {sample.persona_name}"

    result = run_persona_evals(
        val_path, _capture_fn, system_override_fn=_override
    )

    # Generator received the rewritten system strings, not the val.jsonl ones.
    assert all(s.startswith("OVERRIDE for Name-") for s in captured_systems)
    # Per-generation log also records the rewritten system for traceability.
    logged_systems = [g["system"] for g in result.generations]
    assert all(s.startswith("OVERRIDE for Name-") for s in logged_systems)


def test_run_persona_evals_passes_persona_profile_into_override(tmp_path: Path):
    samples = [_minimal_sample("A", "hi", "ya")]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    profile_doc = {
        "personas": {
            "A": {"length": {"mean_words": 7}, "marker": "per-persona-A"},
        }
    }
    prof_path = tmp_path / "profiles.json"
    prof_path.write_text(json.dumps(profile_doc), encoding="utf-8")

    seen: list[dict | None] = []

    def _override(sample, persona_profile):
        seen.append(persona_profile)
        return sample.system

    def _gen(prompts, systems, gen_kwargs):
        return ["x"] * len(prompts)

    run_persona_evals(
        val_path,
        _gen,
        profile_json=prof_path,
        system_override_fn=_override,
    )
    assert len(seen) == 1
    assert seen[0] is not None
    assert seen[0].get("marker") == "per-persona-A"


def test_run_persona_evals_passes_none_profile_for_unknown_persona(tmp_path: Path):
    samples = [_minimal_sample("Unknown", "hi", "ya")]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    profile_doc = {"personas": {"OtherPersona": {"length": {"mean_words": 3}}}}
    prof_path = tmp_path / "profiles.json"
    prof_path.write_text(json.dumps(profile_doc), encoding="utf-8")

    seen: list[dict | None] = []

    def _override(sample, persona_profile):
        seen.append(persona_profile)
        return sample.system

    def _gen(prompts, systems, gen_kwargs):
        return ["x"] * len(prompts)

    run_persona_evals(
        val_path,
        _gen,
        profile_json=prof_path,
        system_override_fn=_override,
    )
    assert seen == [None]


# ---------------------------------------------------------------------------
# run_evals (end-to-end, persona-only — no lmms-eval subprocess)
# ---------------------------------------------------------------------------


def test_run_evals_applies_profile_baseline_for_no_adapter_spec(tmp_path: Path):
    """When spec has no adapter and mode=profile, the persona generator
    must see the profile-derived system prompt and the run JSON must
    record the mode.
    """
    from discord_sft.evals.model import ModelSpec
    from discord_sft.evals.runner import run_evals

    samples = [_minimal_sample("A", "hi", "ya")]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    profile_doc = {
        "personas": {
            "A": {
                "length": {"mean_words": 6.0, "p50": 5, "p95": 14},
                "top_fillers": {"1gram": [{"token": "imma", "score": 3.0}]},
                "lowercase_start_rate": 0.8,
            }
        }
    }
    prof_path = tmp_path / "profiles.json"
    prof_path.write_text(json.dumps(profile_doc), encoding="utf-8")

    captured: list[str] = []

    def _capture_gen(prompts, systems, gen_kwargs):
        captured.extend(systems)
        return ["hi"] * len(prompts)

    spec = ModelSpec(name_or_path="Qwen/Test-Base", backend="hf")
    run = run_evals(
        spec,
        tasks=["persona"],
        val_jsonl=val_path,
        profile_json=prof_path,
        out_dir=tmp_path / "evals",
        generate_fn=_capture_gen,
        baseline_prompt_mode="profile",
    )
    assert captured, "generate_fn was not called"
    system = captured[0]
    assert "imma" in system
    assert "~6 words on average" in system
    assert run["harness"] == {}
    assert run["persona"]["baseline_prompt_mode"] == "profile"
    assert run["persona"]["is_base_model"] is True
    gen_path = Path(run["persona"]["generations_path"])
    first_row = json.loads(gen_path.read_text(encoding="utf-8").splitlines()[0])
    assert "context" in first_row
    assert "context_turns" in first_row


def test_run_evals_keeps_minimal_prompt_for_adapter_spec(tmp_path: Path):
    """Adapter runs must not inherit the baseline prompt — they need to
    see the exact system string they were trained under.
    """
    from discord_sft.evals.model import ModelSpec
    from discord_sft.evals.runner import run_evals

    samples = [_minimal_sample("A", "hi", "ya")]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    captured: list[str] = []

    def _capture_gen(prompts, systems, gen_kwargs):
        captured.extend(systems)
        return ["hi"] * len(prompts)

    spec = ModelSpec(
        name_or_path="Qwen/Test-Base",
        backend="hf",
        adapter_path="/fake/adapter",
    )
    run = run_evals(
        spec,
        tasks=["persona"],
        val_jsonl=val_path,
        out_dir=tmp_path / "evals",
        generate_fn=_capture_gen,
        baseline_prompt_mode="profile",  # Should be ignored for LoRA runs.
    )
    assert captured, "generate_fn was not called"
    # The adapter run sees the val.jsonl system string verbatim.
    assert captured[0] == "You are Name-A chatting with Other."
    assert run["persona"]["baseline_prompt_mode"] == "minimal"
    assert run["persona"]["is_base_model"] is False


def test_run_evals_rejects_unknown_baseline_mode(tmp_path: Path):
    from discord_sft.evals.model import ModelSpec
    from discord_sft.evals.runner import run_evals

    spec = ModelSpec(name_or_path="Qwen/Test-Base", backend="hf")
    with pytest.raises(ValueError):
        run_evals(
            spec,
            tasks=["persona"],
            val_jsonl=tmp_path / "val.jsonl",
            baseline_prompt_mode="bogus",
        )
