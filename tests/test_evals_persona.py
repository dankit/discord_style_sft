from __future__ import annotations

import json
from pathlib import Path

from discord_sft.evals.persona import (
    PersonaEvalResult,
    extract_eval_samples,
    run_persona_evals,
)
from discord_sft.data_prep.sft import Sample, write_samples


def _make_sample(pid: str, user_text: str, assistant_text: str) -> Sample:
    return Sample(
        system=f"You are {pid}.",
        conversations=[
            {"from": "user", "value": user_text},
            {"from": "assistant", "value": assistant_text},
        ],
        meta={"persona_id": pid, "persona_name": f"Name-{pid}", "session_id": f"s-{pid}"},
    )


def test_extract_eval_samples_strips_final_assistant_turn():
    samples = [_make_sample("A", "hi", "ya")]
    out = extract_eval_samples(samples)
    assert len(out) == 1
    s = out[0]
    assert s.persona_id == "A"
    assert s.reference == "ya"
    assert len(s.prompt_conversation) == 1
    assert s.prompt_conversation[0]["from"] == "user"


def test_extract_eval_samples_skips_malformed():
    bad = Sample(
        system="x",
        conversations=[{"from": "user", "value": "hi"}],
        meta={"persona_id": "A"},
    )
    out = extract_eval_samples([bad])
    assert out == []


def test_run_persona_evals_with_dummy_generator_groups_by_persona(tmp_path: Path):
    samples = [
        _make_sample("A", "hi a1", "imma go soon imma chill"),
        _make_sample("A", "hi a2", "imma nap"),
        _make_sample("B", "hi b1", "heading out for lunch"),
    ]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    def _echo_fn(prompts, systems, gen_kwargs):
        return [f"echo: {p[:40]}" for p in prompts]

    result = run_persona_evals(val_path, _echo_fn)
    assert isinstance(result, PersonaEvalResult)
    assert set(result.per_persona.keys()) == {"A", "B"}
    assert result.per_persona["A"]["n_samples"] == 2
    assert result.per_persona["B"]["n_samples"] == 1
    assert result.generations, "Expected per-sample generation rows"
    first = result.generations[0]
    assert "context" in first
    assert "context_turns" in first
    assert isinstance(first["context_turns"], list)

    flat = result.to_flat_scores()
    assert any(k.startswith("persona.heuristics.A.") for k in flat)
    assert any(k.startswith("persona.heuristics.B.") for k in flat)


def test_run_persona_evals_with_profile_json(tmp_path: Path):
    samples = [
        _make_sample("A", "u", "imma be chillin imma relax"),
        _make_sample("A", "u", "imma sleep soon imma rest"),
    ]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    profile = {
        "personas": {
            "A": {
                "top_fillers": {"1gram": [{"token": "imma", "score": 5.0}]},
                "top_fillers_1gram": [["imma", 5.0]],
            }
        }
    }
    prof_path = tmp_path / "profiles.json"
    prof_path.write_text(json.dumps(profile), encoding="utf-8")

    def _filler_fn(prompts, systems, gen_kwargs):
        return ["imma totally do it"] * len(prompts)

    result = run_persona_evals(val_path, _filler_fn, profile_json=prof_path)
    heur = result.per_persona["A"]["heuristics"]
    assert heur.get("filler_rate", 0.0) > 0.0


class _DummyJudge:
    def score(
        self,
        *,
        real_message,
        generated_message,
        context,
        persona_name,
        system_instructions=None,
    ):
        _ = (real_message, generated_message, context, persona_name, system_instructions)
        return {"overall": 4.0, "tone": 3.5, "error_msg": "not a number"}


def test_run_persona_evals_judge_averages(tmp_path: Path):
    samples = [
        _make_sample("A", "u1", "resp1"),
        _make_sample("A", "u2", "resp2"),
    ]
    val_path = tmp_path / "val.jsonl"
    write_samples(samples, val_path)

    def _gen(prompts, systems, gen_kwargs):
        return ["x"] * len(prompts)

    result = run_persona_evals(val_path, _gen, judge=_DummyJudge())
    judge_scores = result.per_persona["A"]["judge"]
    assert judge_scores["overall"] == 4.0
    assert judge_scores["tone"] == 3.5
    flat = result.to_flat_scores()
    assert "persona.judge.A.overall" in flat


def test_benchmarks_split_tasks_rejects_typos():
    from discord_sft.evals.benchmarks import split_tasks

    try:
        split_tasks(["ifeval", "not_a_benchmark"])
    except KeyError as e:
        assert "not_a_benchmark" in str(e)
    else:  # pragma: no cover
        raise AssertionError("split_tasks should have raised KeyError")


def test_benchmarks_split_tasks_persona_flag():
    from discord_sft.evals.benchmarks import split_tasks

    lmms, include_persona = split_tasks(["ifeval", "persona"])
    assert lmms == ["ifeval"]
    assert include_persona is True
