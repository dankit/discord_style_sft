from __future__ import annotations

import json
from pathlib import Path

from discord_sft.evals.compare import compare_runs, render_comparison
from discord_sft.evals.storage import save_run


def _write(tmp_path: Path, run_id: str, scores: dict) -> None:
    save_run(
        {
            "run_id": run_id,
            "created_utc": "2026-04-21T00:00:00Z",
            "label": run_id,
            "model": {"name_or_path": "m"},
            "config": {},
            "scores": scores,
        },
        tmp_path,
    )


def test_compare_computes_delta_and_forgetting(tmp_path: Path):
    _write(tmp_path, "baseline", {"ifeval.acc": 0.80, "gsm8k.em": 0.60})
    _write(tmp_path, "lora", {"ifeval.acc": 0.72, "gsm8k.em": 0.30})

    report = compare_runs(
        [tmp_path / "runs" / "baseline.json", tmp_path / "runs" / "lora.json"],
        baseline=0,
    )
    rows = {r["metric"]: r for r in report["rows"]}
    assert report["baseline_run_id"] == "baseline"
    assert abs(rows["ifeval.acc"]["delta__lora"] - (-0.08)) < 1e-9
    assert abs(rows["ifeval.acc"]["forgetting__lora"] - (0.08 / 0.80)) < 1e-9
    assert abs(rows["gsm8k.em"]["forgetting__lora"] - 0.5) < 1e-9


def test_compare_zero_baseline_yields_none_forgetting(tmp_path: Path):
    _write(tmp_path, "b", {"m": 0.0})
    _write(tmp_path, "c", {"m": 0.5})
    report = compare_runs(
        [tmp_path / "runs" / "b.json", tmp_path / "runs" / "c.json"], baseline=0
    )
    row = report["rows"][0]
    assert row["forgetting__c"] is None
    assert row["delta__c"] == 0.5


def test_compare_missing_metric_in_candidate_is_none(tmp_path: Path):
    _write(tmp_path, "b", {"m": 0.5, "only_base": 0.2})
    _write(tmp_path, "c", {"m": 0.5})
    report = compare_runs(
        [tmp_path / "runs" / "b.json", tmp_path / "runs" / "c.json"], baseline=0
    )
    rows = {r["metric"]: r for r in report["rows"]}
    assert rows["only_base"]["c"] is None
    assert rows["only_base"]["delta__c"] is None


def test_compare_metric_glob_filter(tmp_path: Path):
    _write(tmp_path, "b", {"ifeval.a": 1.0, "ifeval.b": 0.5, "mmmu.x": 0.3})
    _write(tmp_path, "c", {"ifeval.a": 0.8, "ifeval.b": 0.4, "mmmu.x": 0.2})
    report = compare_runs(
        [tmp_path / "runs" / "b.json", tmp_path / "runs" / "c.json"],
        baseline=0,
        metrics=["ifeval.*"],
    )
    metrics = [r["metric"] for r in report["rows"]]
    assert set(metrics) == {"ifeval.a", "ifeval.b"}


def test_compare_baseline_by_run_id(tmp_path: Path):
    _write(tmp_path, "a", {"m": 1.0})
    _write(tmp_path, "b", {"m": 0.5})
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"], baseline="b"
    )
    assert report["baseline_run_id"] == "b"


def test_compare_no_baseline_drops_delta_columns(tmp_path: Path):
    _write(tmp_path, "a", {"m": 1.0})
    _write(tmp_path, "b", {"m": 0.5})
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"], baseline=None
    )
    for row in report["rows"]:
        assert not any(k.startswith("delta__") or k.startswith("forgetting__") for k in row)


def test_compare_omits_persona_metrics_by_default(tmp_path: Path):
    _write(
        tmp_path,
        "a",
        {"ifeval.acc": 0.9, "persona.judge.pid.overall": 4.0},
    )
    _write(
        tmp_path,
        "b",
        {"ifeval.acc": 0.8, "persona.judge.pid.overall": 3.0},
    )
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"],
        baseline=0,
    )
    keys = [r["metric"] for r in report["rows"]]
    assert keys == ["ifeval.acc"]


def test_compare_include_persona_when_omit_disabled(tmp_path: Path):
    _write(tmp_path, "a", {"ifeval.acc": 0.9, "persona.judge.pid.overall": 4.0})
    _write(tmp_path, "b", {"ifeval.acc": 0.8, "persona.judge.pid.overall": 3.0})
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"],
        baseline=0,
        omit_persona_metrics=False,
    )
    keys = [r["metric"] for r in report["rows"]]
    assert "ifeval.acc" in keys
    assert any(k.startswith("persona.") for k in keys)


def test_compare_explicit_metrics_glob_still_matches_persona(tmp_path: Path):
    _write(tmp_path, "a", {"ifeval.acc": 0.9, "persona.judge.pid.overall": 4.0})
    _write(tmp_path, "b", {"ifeval.acc": 0.8, "persona.judge.pid.overall": 3.0})
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"],
        baseline=0,
        metrics=["persona.*"],
    )
    keys = [r["metric"] for r in report["rows"]]
    assert all(k.startswith("persona.") for k in keys)


def test_render_table_and_markdown_and_json(tmp_path: Path):
    _write(tmp_path, "a", {"m": 1.0})
    _write(tmp_path, "b", {"m": 0.5})
    report = compare_runs(
        [tmp_path / "runs" / "a.json", tmp_path / "runs" / "b.json"], baseline=0
    )

    table = render_comparison(report, fmt="table")
    assert "metric" in table and "a" in table and "b" in table

    md = render_comparison(report, fmt="markdown")
    assert md.startswith("| metric")

    as_json = render_comparison(report, fmt="json")
    parsed = json.loads(as_json)
    assert parsed["baseline_run_id"] == "a"
