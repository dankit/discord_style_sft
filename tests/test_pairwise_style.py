from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from discord_sft.evals.pairwise_style import (
    RankStyleConfig,
    check_val_provenance,
    parse_pairwise_answer,
    run_rank_style_eval,
    sample_pairs,
)


def test_parse_pairwise_answer() -> None:
    assert parse_pairwise_answer("A") == "A"
    assert parse_pairwise_answer("b\n") == "B"
    assert parse_pairwise_answer("Answer: A") == "A"
    assert parse_pairwise_answer("I choose B.") == "B"
    assert parse_pairwise_answer("maybe C") is None
    assert (
        parse_pairwise_answer(
            "A has more slang.\nB is stiff.\nAnswer: A\n"
        )
        == "A"
    )
    assert (
        parse_pairwise_answer("Answer: A\nAnswer: B\n") == "B"
    )


def test_sample_pairs_caps() -> None:
    rng = __import__("random").Random(0)
    pairs = sample_pairs(4, 4, rng)
    assert len(pairs) == 4
    assert len(set(pairs)) == 4
    all_pairs = 4 * 3 // 2
    assert all_pairs == 6
    pairs_all = sample_pairs(4, 20, rng)
    assert len(pairs_all) == 6


def test_provenance_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"config": {"val_jsonl": str(tmp_path / "v1.jsonl")}}),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps({"config": {"val_jsonl": str(tmp_path / "v2.jsonl")}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        check_val_provenance([a, b])


def _row(pid: str, ref: str, gen: str, ctx_turns: list[dict[str, str]]) -> dict:
    return {
        "persona_id": pid,
        "persona_name": "n",
        "reference": ref,
        "generated": gen,
        "system": "s",
        "context_turns": ctx_turns,
        "context": "[user] hi",
    }


def test_rank_style_sparse_pairwise_and_elo(tmp_path: Path) -> None:
    turn = [{"from": "user", "value": "hi"}]
    r1 = _row("1", "ref one", "g1a", turn)
    r2 = _row("1", "ref one", "g1b", turn)
    r3 = _row("1", "ref two", "g2a", turn)
    r4 = _row("1", "ref two", "g2b", turn)

    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(r1) + "\n" + json.dumps(r3) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(r2) + "\n" + json.dumps(r4) + "\n", encoding="utf-8")

    calls: list[int] = []

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str:
        calls.append(1)
        _ = user_prompt, reference_style, output_a, output_b
        return "A"

    cfg = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    assert report["n_merged_samples"] == 2
    assert report["k_variants"] == 2
    # 2 samples × 1 pair each = 2 comparisons (K=2 → only one unordered pair)
    assert len(calls) == 2
    assert report["pairwise"]["n_comparisons"] == 2
    assert report["pairwise"]["n_comparison_errors"] == 0
    elo = report["pairwise"]["elo"]
    pw = report["pairwise"]
    # Constant "A" judge wins the on-screen first variant; swap RNG decides which label that is (deterministic seed).
    assert pw["wins"]["run_a"] + pw["wins"]["run_b"] == 2
    assert (elo["run_a"] > elo["run_b"]) == (pw["wins"]["run_a"] > pw["wins"]["run_b"])
    assert (elo["run_a"] == elo["run_b"]) == (pw["wins"]["run_a"] == pw["wins"]["run_b"])


def test_prior_elo_seeds_pairwise_ratings(tmp_path: Path) -> None:
    turn = [{"from": "user", "value": "hi"}]
    r1 = _row("1", "ref one", "g1a", turn)
    r2 = _row("1", "ref one", "g1b", turn)
    r3 = _row("1", "ref two", "g2a", turn)
    r4 = _row("1", "ref two", "g2b", turn)

    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(r1) + "\n" + json.dumps(r3) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(r2) + "\n" + json.dumps(r4) + "\n", encoding="utf-8")

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str:
        _ = user_prompt, reference_style, output_a, output_b
        return "A"

    base = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
    )
    seeded = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
        prior_elo_by_label={"run_a": 1600.0, "run_b": 1400.0},
    )
    elo_base = run_rank_style_eval(base, judge_fn)["pairwise"]["elo"]
    elo_seeded = run_rank_style_eval(seeded, judge_fn)["pairwise"]["elo"]
    assert elo_seeded["run_a"] > elo_base["run_a"]
    assert elo_seeded["run_b"] < elo_base["run_b"]
    assert elo_seeded["run_a"] - elo_seeded["run_b"] > elo_base["run_a"] - elo_base["run_b"]


def test_emit_comparisons_includes_judge_texts(tmp_path: Path) -> None:
    """--emit-comparisons should record generations as presented to the pairwise judge."""
    turn = [{"from": "user", "value": "hi"}]
    r1 = _row("1", "ref one", "g1a", turn)
    r2 = _row("1", "ref one", "g1b", turn)
    r3 = _row("1", "ref two", "g2a", turn)
    r4 = _row("1", "ref two", "g2b", turn)

    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(r1) + "\n" + json.dumps(r3) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(r2) + "\n" + json.dumps(r4) + "\n", encoding="utf-8")

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> tuple[str, str]:
        _ = user_prompt, reference_style, output_a, output_b
        return ("A", "Short rationale.\nAnswer: A")

    cfg = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
        emit_comparisons=True,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    comps = report.get("comparisons") or []
    assert len(comps) == 2
    per_keys = {(c["sample_key"], c["i"], c["j"]) for c in comps}
    assert len(per_keys) == 2
    for row in comps:
        assert row["reference_style"] in ("ref one", "ref two")
        if row["choice"] == "A":
            assert row["winner_label"] == row["label_a"]
            assert row["loser_label"] == row["label_b"]
        else:
            assert row["winner_label"] == row["label_b"]
            assert row["loser_label"] == row["label_a"]
        gens = frozenset({row["output_a"], row["output_b"]})
        if row["reference_style"] == "ref one":
            assert gens == frozenset({"g1a", "g1b"})
        else:
            assert gens == frozenset({"g2a", "g2b"})
        assert "user_prompt" in row and row["user_prompt"]
        assert row.get("judge_response") == "Short rationale.\nAnswer: A"


def test_emit_comparisons_omits_judge_response_when_fn_returns_plain_str(
    tmp_path: Path,
) -> None:
    turn = [{"from": "user", "value": "hi"}]
    r1 = _row("1", "ref one", "g1a", turn)
    r2 = _row("1", "ref one", "g1b", turn)
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(r1) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(r2) + "\n", encoding="utf-8")

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str:
        _ = user_prompt, reference_style, output_a, output_b
        return "B"

    cfg = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
        emit_comparisons=True,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    row = report["comparisons"][0]
    assert row["choice"] == "B"
    assert "judge_response" not in row


def test_k4_fewer_than_all_pairs(tmp_path: Path) -> None:
    """With 4 runs, cap comparisons below K*(K-1)/2."""
    turn = [{"from": "user", "value": "x"}]
    rows_a = [
        _row("1", "r1", "a1", turn),
        _row("1", "r2", "a2", turn),
    ]
    paths = []
    for i, suf in enumerate(["w", "x", "y", "z"]):
        p = tmp_path / f"{suf}.jsonl"
        lines = []
        for j, row in enumerate(rows_a):
            d = dict(row)
            d["generated"] = f"{suf}_{j}"
            lines.append(json.dumps(d))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(p)

    calls: list[int] = []

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str:
        calls.append(1)
        _ = user_prompt, reference_style, output_a, output_b
        return "B"

    cfg = RankStyleConfig(
        generations_paths=paths,
        labels=["w", "x", "y", "z"],
        run_json_paths=[None, None, None, None],
        skip_provenance=True,
        pairs_per_prompt=4,
        seed=123,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    max_pairs_per_row = 4
    all_pairs = 6
    assert max_pairs_per_row < all_pairs
    assert len(calls) == 2 * max_pairs_per_row
    assert report["pairwise"]["n_comparison_errors"] == 0


def test_rank_style_partial_judge_keeps_elo_and_checkpoint(tmp_path: Path) -> None:
    turn = [{"from": "user", "value": "hi"}]
    r1 = _row("1", "ref one", "g1a", turn)
    r2 = _row("1", "ref one", "g1b", turn)
    r3 = _row("1", "ref two", "g2a", turn)
    r4 = _row("1", "ref two", "g2b", turn)

    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    p1.write_text(json.dumps(r1) + "\n" + json.dumps(r3) + "\n", encoding="utf-8")
    p2.write_text(json.dumps(r2) + "\n" + json.dumps(r4) + "\n", encoding="utf-8")

    n = {"v": 0}
    judge_lock = threading.Lock()

    def judge_fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str:
        _ = user_prompt, reference_style, output_a, output_b
        with judge_lock:
            n["v"] += 1
            cur = n["v"]
            if cur % 2 == 1:
                raise RuntimeError("simulated judge failure")
            return "A"

    ckpt = tmp_path / "ckpt.jsonl"
    cfg = RankStyleConfig(
        generations_paths=[p1, p2],
        labels=["run_a", "run_b"],
        run_json_paths=[None, None],
        skip_provenance=True,
        pairs_per_prompt=1,
        seed=42,
        comparisons_checkpoint_path=ckpt,
    )
    report = run_rank_style_eval(cfg, judge_fn)
    assert report["pairwise"]["n_comparison_errors"] == 1
    assert report["pairwise"]["n_comparisons"] == 1
    assert report["pairwise"]["partial"] is True
    assert any("failed" in w.lower() for w in report["merge_warnings"])
    raw = ckpt.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 2
    kinds = [json.loads(line).get("checkpoint_kind") for line in raw]
    assert "error" in kinds
    assert any("winner_label" in json.loads(line) for line in raw)
