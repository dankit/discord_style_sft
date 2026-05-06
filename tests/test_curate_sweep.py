import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from discord_sft.data_prep.curate_sweep import (
    default_sweep_lists,
    iter_curate_sweep_rows,
    parse_csv_floats,
    parse_csv_ints,
)


def _msg_raw(mid: str, ts: str, author_id: str, username: str, content: str) -> dict:
    return {
        "id": mid,
        "timestamp": ts,
        "author": {"id": author_id, "username": username},
        "content": content,
        "type": 0,
    }


def _write_min_export(root: Path) -> None:
    """One DM folder with a short two-person thread (4 messages)."""
    dm = root / "dm_a"
    dm.mkdir(parents=True)
    t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        _msg_raw("1", t0.isoformat(), "A", "alice", "hi"),
        _msg_raw(
            "2",
            (t0.replace(second=5)).isoformat(),
            "B",
            "bob",
            "hello",
        ),
        _msg_raw(
            "3",
            (t0.replace(second=10)).isoformat(),
            "A",
            "alice",
            "how are you",
        ),
        _msg_raw(
            "4",
            (t0.replace(second=15)).isoformat(),
            "B",
            "bob",
            "good",
        ),
    ]
    (dm / "dm_a-page-1.json").write_text(json.dumps(rows), encoding="utf-8")


def test_parse_csv_ints_and_floats():
    assert parse_csv_ints("2, 4 ,6") == [2, 4, 6]
    assert parse_csv_floats("0.8,1.0") == [0.8, 1.0]
    with pytest.raises(ValueError):
        parse_csv_ints("")
    with pytest.raises(ValueError):
        parse_csv_floats("   ")


def test_default_sweep_lists_uses_baseline_when_no_csv():
    sgm, mgs, mt, ma, mono = default_sweep_lists(
        session_gap_min=45,
        merge_gap_sec=20,
        min_turns=3,
        min_authors=2,
        monologue_max_share=0.7,
        sweep_session_gap_min=None,
        sweep_merge_gap_sec=None,
        sweep_min_turns=None,
        sweep_min_authors=None,
        sweep_monologue_max_share=None,
    )
    assert sgm == [45] and mgs == [20] and mt == [3] and ma == [2] and mono == [0.7]


def test_iter_curate_sweep_rows_min_turns_changes_sessions_kept(tmp_path: Path):
    src = tmp_path / "export"
    _write_min_export(src)

    rows = list(
        iter_curate_sweep_rows(
            src,
            session_gap_mins=[60],
            merge_gap_secs=[30],
            min_turns_list=[2, 8],
            min_authors_list=[2],
            monologue_max_shares=[0.80],
            url_strip=False,
            pii_scrub=True,
            lang=None,
            near_dedup_threshold=None,
            dedupe_exact_turns=True,
            exact_turn_dup_cap=1,
        )
    )
    assert len(rows) == 2
    assert rows[0]["params"]["min_turns"] == 2
    assert rows[1]["params"]["min_turns"] == 8
    assert rows[0]["sessions_kept"] >= 1
    assert rows[1]["sessions_kept"] == 0
    assert "dropped_short_session" in rows[1]["report"]


def test_cli_curate_sweep_smoke(tmp_path: Path):
    from discord_sft.cli.commands_data import _cmd_curate_sweep

    src = tmp_path / "export"
    _write_min_export(src)
    out_path = tmp_path / "sweep.jsonl"

    class Args:
        source = str(src)
        out = str(out_path)
        session_gap_min = 60
        merge_gap_sec = 30
        min_turns = 2
        min_authors = 2
        monologue_max_share = 0.80
        strip_urls = False
        no_pii_scrub = False
        lang = None
        near_dedup_threshold = 0.85
        no_near_dedup = True
        no_dedupe_exact_turns = False
        exact_turn_dup_cap = 1
        max_combos = 500
        sweep_session_gap_min = None
        sweep_merge_gap_sec = None
        sweep_min_turns = "2,8"
        sweep_min_authors = None
        sweep_monologue_max_share = None

    assert _cmd_curate_sweep(Args()) == 0
    lines = [ln for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    doc = json.loads(lines[0])
    assert doc["params"]["min_turns"] == 2
    assert "sessions_kept" in doc and "report" in doc
