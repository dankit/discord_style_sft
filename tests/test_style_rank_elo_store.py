from __future__ import annotations

from pathlib import Path

from discord_sft.ui.pages.evals.style_rank_elo_store import (
    default_elo_store,
    elo_group_store_path,
    get_group_elo_and_winrate,
    load_elo_store,
    save_elo_store,
    upsert_group_elo,
)


def test_round_trip_and_get_group(tmp_path: Path) -> None:
    p = elo_group_store_path(tmp_path)
    assert p.name == "style_rank_group_elo.json"
    doc = default_elo_store()
    gk = "evalset\x1ffp123"
    doc = upsert_group_elo(
        doc,
        gk,
        elo={"run_a": 1520.0, "run_b": 1480.0},
        win_rate={"run_a": 0.52, "run_b": None},
        ranked_run_ids=["run_a", "run_b"],
    )
    save_elo_store(p, doc)
    loaded = load_elo_store(p)
    elo, wr = get_group_elo_and_winrate(loaded, gk)
    assert elo["run_a"] == 1520.0
    assert wr["run_b"] is None


def test_upsert_preserves_other_groups(tmp_path: Path) -> None:
    doc = upsert_group_elo(
        default_elo_store(),
        "g1",
        elo={"a": 1.0},
        win_rate={},
        ranked_run_ids=["a"],
    )
    doc = upsert_group_elo(
        doc,
        "g2",
        elo={"b": 2.0},
        win_rate={},
        ranked_run_ids=["b"],
    )
    e1, _ = get_group_elo_and_winrate(doc, "g1")
    e2, _ = get_group_elo_and_winrate(doc, "g2")
    assert e1["a"] == 1.0 and e2["b"] == 2.0


def test_invalid_json_fallback(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{", encoding="utf-8")
    assert load_elo_store(p) == default_elo_store()
