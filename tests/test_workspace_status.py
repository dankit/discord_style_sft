import json
from pathlib import Path

from discord_sft.ui import workspace_status as ws


def test_pipeline_completion_none() -> None:
    raw = {"ingest": False, "curate": False, "sft": False, "fingerprint": False, "evals": False}
    p = ws.pipeline_completion(raw, raw_flags=raw)
    assert p["completed_consecutive"] == 0
    assert p["progress_fraction"] == 0.0
    assert p["status_title"] == "Not started"


def test_pipeline_completion_full_chain() -> None:
    raw = {"ingest": True, "curate": True, "sft": True, "fingerprint": True, "evals": True}
    p = ws.pipeline_completion(raw, raw_flags=raw)
    assert p["completed_consecutive"] == 5
    assert p["progress_fraction"] == 1.0
    assert p["status_title"] == "Evals"


def test_pipeline_completion_ready_for_training() -> None:
    raw = {"ingest": True, "curate": True, "sft": True, "fingerprint": True, "evals": False}
    p = ws.pipeline_completion(raw, raw_flags=raw)
    assert p["completed_consecutive"] == 4
    assert p["status_title"] == "Ready for training"


def test_pipeline_completion_partial() -> None:
    raw = {"ingest": True, "curate": True, "sft": False, "fingerprint": False, "evals": False}
    p = ws.pipeline_completion(raw, raw_flags=raw)
    assert p["completed_consecutive"] == 2
    assert p["status_title"] == "Curate"


def test_effective_flags_always_infer_ingest_from_sft() -> None:
    raw = {"ingest": False, "curate": False, "sft": True, "fingerprint": False, "evals": False}
    eff = ws.effective_pipeline_flags(raw)
    assert eff["ingest"] and eff["curate"] and eff["sft"]
    p = ws.pipeline_completion(eff, raw_flags=raw)
    assert p["completed_consecutive"] == 3
    assert p["status_title"] == "Build SFT"
    assert p["chain_broken"] is False


def test_effective_flags_fill_prereqs_from_curate() -> None:
    raw = {"ingest": False, "curate": True, "sft": False, "fingerprint": False, "evals": False}
    eff = ws.effective_pipeline_flags(raw)
    assert eff["ingest"] is True
    assert eff["curate"] is True
    p = ws.pipeline_completion(eff, raw_flags=raw)
    assert p["completed_consecutive"] == 2
    assert p["status_title"] == "Curate"


def test_chain_broken_when_middle_step_missing() -> None:
    raw = {"ingest": True, "curate": False, "sft": True, "fingerprint": False, "evals": False}
    eff = ws.effective_pipeline_flags(raw)
    p = ws.pipeline_completion(eff, raw_flags=raw)
    assert p["chain_broken"] is True


def test_build_snapshot_infers_from_curated_only(tmp_path: Path) -> None:
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "curated" / "sessions.jsonl").write_text("{}\n", encoding="utf-8")
    snap = ws.build_snapshot(tmp_path)
    assert snap["pipeline"]["completed_consecutive"] >= 1


def test_aggregate_sessions_jsonl(tmp_path: Path) -> None:
    rec = {
        "session_id": "s1",
        "folder": "dm_a",
        "authors": ["u1", "u2"],
        "turns": [
            {"author_id": "u1", "author_name": "a", "text": "hello world", "start_ts": "2025-01-01T00:00:00+00:00", "end_ts": "2025-01-01T00:00:01+00:00"},
            {"author_id": "u2", "author_name": "b", "text": "hi", "start_ts": "2025-01-01T00:00:02+00:00", "end_ts": "2025-01-01T00:00:03+00:00"},
        ],
    }
    p = tmp_path / "sessions.jsonl"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = ws.aggregate_sessions_jsonl(p)
    assert out is not None
    assert out["sessions"] == 1
    assert out["turns"] == 2
    assert out["words"] == 3  # hello world + hi
    assert out["distinct_authors"] == 2
    assert out["distinct_folders"] == 1


def test_aggregate_sft_jsonl(tmp_path: Path) -> None:
    line = {
        "system": "sys",
        "conversations": [{"from": "user", "value": "x"}, {"from": "assistant", "value": "y"}],
        "meta": {"persona_id": "p1", "persona_name": "One", "session_id": "sess-1", "num_turns": 2},
    }
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(line) + "\n", encoding="utf-8")
    out = ws.aggregate_sft_jsonl(p)
    assert out is not None
    assert out["samples"] == 1
    assert out["distinct_sessions"] == 1
    assert out["distinct_personas"] == 1
    assert out["persona_top"][0]["persona_id"] == "p1"
    assert out["persona_top"][0]["samples"] == 1


def test_pipeline_flags_empty(tmp_path: Path) -> None:
    flags = ws.pipeline_flags(tmp_path)
    assert flags == {
        "ingest": False,
        "curate": False,
        "sft": False,
        "fingerprint": False,
        "evals": False,
    }


def test_pipeline_flags_with_files(tmp_path: Path) -> None:
    (tmp_path / "messages" / "dm1").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "curated" / "sessions.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "sft").mkdir(parents=True)
    (tmp_path / "sft" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "sft" / "profiles.json").write_text('{"personas": {}}', encoding="utf-8")
    (tmp_path / "evals" / "runs").mkdir(parents=True)
    (tmp_path / "evals" / "runs" / "a.json").write_text("{}", encoding="utf-8")
    flags = ws.pipeline_flags(tmp_path)
    assert flags["ingest"] is True
    assert flags["curate"] is True
    assert flags["sft"] is True
    assert flags["fingerprint"] is True
    assert flags["evals"] is True
