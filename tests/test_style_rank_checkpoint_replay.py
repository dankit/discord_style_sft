from __future__ import annotations

import json
from pathlib import Path

import pytest

from discord_sft.evals.style_rank_checkpoint_replay import (
    backfill_elo_store_from_checkpoints_dir,
    infer_style_rank_group_key,
    load_ok_comparison_rows,
    merge_ok_comparison_rows_newest_wins,
    replay_checkpoint_file,
    replay_elo_from_comparison_rows,
)


def _row(win: str, lose: str, sk: str = "k1") -> dict:
    return {
        "sample_key": sk,
        "i": 0,
        "j": 1,
        "swap": False,
        "choice": "A",
        "winner_label": win,
        "loser_label": lose,
        "parse_error": False,
    }


def test_replay_two_runs_symmetric_games() -> None:
    rows = [
        _row("run_a", "run_b", "s1"),
        _row("run_b", "run_a", "s2"),
    ]
    elo, wr = replay_elo_from_comparison_rows(rows)
    assert set(elo.keys()) == {"run_a", "run_b"}
    assert wr["run_a"] == 0.5 and wr["run_b"] == 0.5
    # Order of updates is deterministic but not symmetric in two steps; stays near start.
    assert abs(elo["run_a"] - elo["run_b"]) < 5.0


def test_replay_elo_prior_raises_absolute_elo() -> None:
    rows = [_row("run_a", "run_b", "s1")]
    base, _ = replay_elo_from_comparison_rows(rows)
    seeded, _ = replay_elo_from_comparison_rows(
        rows,
        prior_elo_by_label={"run_a": 1700.0, "run_b": 1300.0},
    )
    assert seeded["run_a"] > base["run_a"]
    assert seeded["run_b"] < base["run_b"]


def test_infer_fallback_without_runs(tmp_path: Path) -> None:
    er = tmp_path / "evals"
    er.mkdir()
    gkey, hint = infer_style_rank_group_key(er, frozenset({"x", "y"}))
    assert hint == "checkpoint_hash"
    assert "\x1f" in gkey


def test_backfill_picks_newer_checkpoint_same_group(tmp_path: Path) -> None:
    ck = tmp_path / "style_rank_checkpoints"
    ck.mkdir()
    old = ck / "old.jsonl"
    new = ck / "new.jsonl"

    def write(path: Path, winner: str) -> None:
        loser = "run_b" if winner == "run_a" else "run_a"
        lines = [
            _row(winner, loser, "a"),
            _row(winner, loser, "b"),
        ]
        path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    write(old, "run_a")
    write(new, "run_b")
    import os
    import time

    ts = time.time()
    os.utime(old, (ts - 2000.0, ts - 2000.0))
    os.utime(new, (ts - 500.0, ts - 500.0))

    summ, store_p = backfill_elo_store_from_checkpoints_dir(
        tmp_path,
        ck,
        merge_existing=False,
    )
    assert summ["ok"]
    assert store_p.exists()
    gwritten = summ["groups_written"]
    assert len(gwritten) == 1
    assert gwritten[0]["n_checkpoints_merged"] == 1
    assert str(new.resolve()) == gwritten[0]["checkpoint"]
    top = elo_from_store(store_p, gwritten[0]["gkey"])
    assert top["run_b"] > top["run_a"], "Later checkpoint should dominate"


def test_merge_same_sample_key_different_run_pairs_keeps_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: epoch-1 vs epoch-4 checkpoints share val sample_key but must not dedupe each other."""
    monkeypatch.setattr(
        "discord_sft.evals.style_rank_checkpoint_replay.infer_style_rank_group_key",
        lambda _root, _labs: ("g_shared", "test"),
    )
    ck = tmp_path / "style_rank_checkpoints"
    ck.mkdir()
    sk = "same_val_row_hash"
    older = ck / "older.jsonl"
    newer = ck / "newer.jsonl"
    older.write_text(
        json.dumps(_row("run_epoch1_a", "run_epoch1_b", sk))
        + "\n"
        + json.dumps(_row("run_epoch1_a", "run_epoch1_b", "other_sk"))
        + "\n",
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(_row("run_epoch4_a", "run_epoch4_b", sk))
        + "\n"
        + json.dumps(_row("run_epoch4_a", "run_epoch4_b", "other_sk"))
        + "\n",
        encoding="utf-8",
    )
    import os
    import time

    ts = time.time()
    os.utime(older, (ts - 2000.0, ts - 2000.0))
    os.utime(newer, (ts - 500.0, ts - 500.0))

    def _mt_ns(p: Path) -> int:
        st = p.stat()
        return int(getattr(st, "st_mtime_ns", round(st.st_mtime * 1e9)))

    paths = sorted([older, newer], key=_mt_ns, reverse=True)
    merged, contributors = merge_ok_comparison_rows_newest_wins(paths)
    assert len(merged) == 4
    assert len(contributors) == 2
    labels = {str(r["winner_label"]) for r in merged} | {str(r["loser_label"]) for r in merged}
    assert labels == {
        "run_epoch1_a",
        "run_epoch1_b",
        "run_epoch4_a",
        "run_epoch4_b",
    }


def test_backfill_merges_disjoint_checkpoints_same_gkey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "discord_sft.evals.style_rank_checkpoint_replay.infer_style_rank_group_key",
        lambda _root, _labs: ("g_shared", "test"),
    )
    ck = tmp_path / "style_rank_checkpoints"
    ck.mkdir()
    path_a = ck / "a.jsonl"
    path_b = ck / "b.jsonl"
    path_a.write_text(
        "\n".join(
            json.dumps(x)
            for x in (
                _row("run_a", "run_b", "sk_a1"),
                _row("run_a", "run_b", "sk_a2"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    path_b.write_text(
        "\n".join(
            json.dumps(x)
            for x in (
                _row("run_c", "run_d", "sk_b1"),
                _row("run_c", "run_d", "sk_b2"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summ, store_p = backfill_elo_store_from_checkpoints_dir(tmp_path, ck, merge_existing=False)
    assert summ["ok"]
    gw = summ["groups_written"]
    assert len(gw) == 1
    assert gw[0]["gkey"] == "g_shared"
    assert gw[0]["n_checkpoints_merged"] == 2
    assert len(gw[0]["checkpoints"]) == 2
    top = elo_from_store(store_p, "g_shared")
    assert set(top) == {"run_a", "run_b", "run_c", "run_d"}


def elo_from_store(store_json: Path, gkey: str) -> dict[str, float]:
    doc = json.loads(store_json.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in doc["groups"][gkey]["elo"].items()}


def test_load_skips_errors_and_sorts(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    lines = [
        {"checkpoint_kind": "error", "error": "x"},
        {"winner_label": "a", "loser_label": "b", "sample_key": "z", "i": 0, "j": 1},
        {"winner_label": "a", "loser_label": "b", "sample_key": "a", "i": 0, "j": 1},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    rows = load_ok_comparison_rows(p)
    assert rows[0]["sample_key"] == "a"


def test_replay_checkpoint_requires_eval_root(tmp_path: Path) -> None:
    ck = tmp_path / "style_rank_checkpoints"
    ck.mkdir()
    fp = ck / "one.jsonl"
    fp.write_text(json.dumps(_row("a", "b")) + "\n", encoding="utf-8")
    pick = replay_checkpoint_file(fp, tmp_path)
    assert pick is not None
    assert {"a", "b"} <= set(pick.ranked_run_ids)
