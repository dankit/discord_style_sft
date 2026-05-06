from __future__ import annotations

import json
from pathlib import Path

from discord_sft.evals.storage import (
    list_runs,
    load_run,
    resolve_run_paths,
    run_id_for,
    save_run,
)


def _make_run(run_id: str, created_utc: str, scores: dict) -> dict:
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "label": "t",
        "model": {"name_or_path": "Qwen/Qwen3.5-35B-A3B"},
        "config": {"tasks": ["ifeval"]},
        "scores": scores,
    }


def test_run_id_format_is_sortable_and_slugged():
    a = run_id_for("Qwen/Qwen3.5-35B-A3B", "baseline", stamp="2026-04-21T14-00-00Z")
    b = run_id_for("Qwen/Qwen3.5-35B-A3B", "lora-r8", stamp="2026-04-21T15-00-00Z")
    assert a < b
    assert a.startswith("2026-04-21T14-00-00Z__")
    assert a.endswith("__baseline")
    assert "qwen" in a.lower()


def test_run_id_no_label():
    rid = run_id_for("foo", stamp="2026-01-01T00-00-00Z")
    assert rid == "2026-01-01T00-00-00Z__foo"


def test_save_and_load_run_roundtrip(tmp_path: Path):
    run = _make_run("abc__x", "2026-04-21T14:00:00Z", {"ifeval.acc": 0.5})
    path = save_run(run, tmp_path)
    assert path.exists()
    assert path.parent.name == "runs"
    loaded = load_run(path)
    assert loaded == run
    # Also resolvable by bare id:
    loaded_by_id = load_run("abc__x", out_dir=tmp_path)
    assert loaded_by_id == run


def test_list_runs_sorted_and_ignores_bad_files(tmp_path: Path):
    r1 = _make_run("id_b", "2026-04-22T00:00:00Z", {"a": 1})
    r2 = _make_run("id_a", "2026-04-21T00:00:00Z", {"a": 2})
    save_run(r1, tmp_path)
    save_run(r2, tmp_path)
    (tmp_path / "runs" / "corrupt.json").write_text("not json", encoding="utf-8")

    rows = list_runs(tmp_path)
    assert [r["run_id"] for r in rows] == ["id_a", "id_b"]
    assert rows[0]["created_utc"] < rows[1]["created_utc"]
    assert rows[0]["n_scores"] == 1


def test_list_runs_accepts_runs_dir_directly(tmp_path: Path):
    save_run(_make_run("x", "2026-01-01T00:00:00Z", {}), tmp_path)
    rows = list_runs(tmp_path / "runs")
    assert len(rows) == 1


def test_resolve_run_paths_mix_of_paths_and_ids(tmp_path: Path):
    save_run(_make_run("one", "2026-01-01T00:00:00Z", {}), tmp_path)
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps(_make_run("two", "2026-01-02T00:00:00Z", {})), encoding="utf-8")

    resolved = resolve_run_paths(["one", str(other)], out_dir=tmp_path)
    assert [p.stem for p in resolved] == ["one", "elsewhere"]
