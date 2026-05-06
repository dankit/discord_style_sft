"""Unit tests for multi-run persona generations compare helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from discord_sft.ui.persona_compare import (
    VOTE_STORE_VERSION,
    badge_label,
    comparison_verdict_map,
    infer_module_badges,
    MergedPersonaSample,
    reconcile_comparison_block_inplace,
    VariantRow,
    load_vote_store,
    merge_generation_sources,
    ranked_variant_order,
    save_vote_store,
    sample_key,
    summarize_verdicts_for_filtered,
    target_modules_from_resolved_yaml_near_adapter,
    target_modules_from_run,
)


def _minimal_row(persona_id: str, turns: list, reference: str, generated: str) -> dict:
    return {
        "persona_id": persona_id,
        "persona_name": "pn",
        "reference": reference,
        "generated": generated,
        "system": "sys",
        "context_turns": turns,
        "context": "x",
    }


def test_sample_key_stable_same_payload() -> None:
    turns = [{"from": "user", "value": "hi"}]
    r1 = _minimal_row("1", turns, "ref", "a")
    r2 = _minimal_row("1", list(turns), "ref", "different_gen")
    k1 = sample_key(r1)
    k2 = sample_key(r2)
    assert k1 == k2


def test_sample_key_differs_when_reference_changes() -> None:
    turns = [{"from": "user", "value": "hi"}]
    a = sample_key(_minimal_row("1", turns, "ref-a", "g"))
    b = sample_key(_minimal_row("1", turns, "ref-b", "g"))
    assert a != b


def test_infer_module_badges() -> None:
    assert infer_module_badges(None) == []
    assert infer_module_badges(["q_proj"]) == []
    assert infer_module_badges(["v_proj", "up_proj"]) == ["v_proj"]
    assert infer_module_badges(["gate_proj"]) == ["gate_proj"]
    assert infer_module_badges(["v_proj", "gate_proj"]) == ["v_proj", "gate_proj"]


def test_badge_label() -> None:
    assert badge_label(["v_proj", "up_proj"]) == "v_proj"
    assert badge_label(["q_proj"]) == "q_proj"
    assert badge_label([]) == "Base · no adapter / YAML"
    assert badge_label(None) == "Base · no adapter / YAML"


def test_target_modules_from_run_nested() -> None:
    doc = {"training_config": {"lora": {"target_modules": ["v_proj", "down_proj"]}}}
    assert target_modules_from_run(doc) == ["v_proj", "down_proj"]


def test_target_modules_from_run_reads_yaml_via_adapter(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    ckpt = tmp_path / "step-400"
    ckpt.mkdir()
    yml = ckpt / "config.resolved.yaml"
    yml.write_text(
        "lora:\n  target_modules:\n    - up_proj\n    - down_proj\n    - v_proj\n",
        encoding="utf-8",
    )
    doc = {"training_config": None, "model": {"adapter_path": str(ckpt)}}
    assert target_modules_from_run(doc) == ["up_proj", "down_proj", "v_proj"]


def test_target_modules_yaml_parent_dir(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    parent = tmp_path / "run"
    sub = parent / "epoch-2"
    sub.mkdir(parents=True)
    (parent / "config.resolved.yaml").write_text(
        "lora:\n  target_modules: [v_proj]\n", encoding="utf-8"
    )
    assert target_modules_from_resolved_yaml_near_adapter(sub) == ["v_proj"]


def test_target_modules_from_run_reads_yaml_via_merged_model_rel_path(
    tmp_path: Path,
) -> None:
    pytest.importorskip("yaml")
    fake_root = tmp_path / "repo"
    mdir = fake_root / "out" / "merged" / "epoch-1-x"
    mdir.mkdir(parents=True)
    (mdir / "config.resolved.yaml").write_text(
        "lora:\n  target_modules: [v_proj]\n", encoding="utf-8"
    )
    doc = {
        "training_config": None,
        "model": {
            "name_or_path": "out/merged/epoch-1-x",
            "adapter_path": None,
        },
    }
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        assert target_modules_from_run(doc) == ["v_proj"]


def test_target_modules_from_run_lora_when_merged_checkpoint_missing(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    fake_root = tmp_path / "repo"
    ldir = fake_root / "out" / "lora" / "style-late-r16-a16-5epochs"
    ldir.mkdir(parents=True)
    (ldir / "config.resolved.yaml").write_text(
        "run_name: style-late-r16-a16-5epochs\nlora:\n  target_modules: [v_proj]\n",
        encoding="utf-8",
    )
    doc = {
        "training_config": None,
        "model": {
            "name_or_path": "out/merged/style-late-r16-a16-5epochs/epoch-4-vllm-keys",
            "adapter_path": None,
        },
    }
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        assert target_modules_from_run(doc) == ["v_proj"]


def test_target_modules_nested_wins_over_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.resolved.yaml").write_text(
        "lora:\n  target_modules: [gate_proj]\n", encoding="utf-8"
    )
    doc = {
        "training_config": {"lora": {"target_modules": ["from_json"]}},
        "model": {"adapter_path": str(ckpt)},
    }
    assert target_modules_from_run(doc) == ["from_json"]


def test_merge_generation_sources_inner_join(tmp_path: Path) -> None:
    shared_turns = [{"from": "user", "value": "hey"}]
    ref = "next line"
    a = [_minimal_row("p1", shared_turns, ref, "gen-a")]
    b = [_minimal_row("p1", shared_turns, ref, "gen-b")]
    p_a = tmp_path / "a.jsonl"
    p_b = tmp_path / "b.jsonl"
    p_a.write_text(json.dumps(a[0]) + "\n", encoding="utf-8")
    p_b.write_text(json.dumps(b[0]) + "\n", encoding="utf-8")

    indexed = [
        (p_a, a, {"run_id": "run-a", "label": "la", "training_config": {"lora": {"target_modules": ["v_proj"]}}}, "run-a", 0),
        (p_b, b, {"run_id": "run-b", "training_config": {"lora": {"target_modules": ["gate_proj"]}}}, "run-b", 1),
    ]
    merged, warns = merge_generation_sources(indexed)
    assert isinstance(warns, list)
    assert len(merged) == 1
    m0 = merged[0]
    assert len(m0.variants) == 2
    assert m0.variants[0].generated == "gen-a"
    assert m0.variants[1].generated == "gen-b"
    assert m0.variants[0].target_modules == ["v_proj"]
    assert m0.variants[1].target_modules == ["gate_proj"]


def test_vote_store_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "votes.json"
    doc_in = {
        "version": VOTE_STORE_VERSION,
        "comparisons": {
            "cid1": {
                "verdicts": {"sk": {"va": "accepted", "vb": "rejected"}},
                "variants": {"va": {"label": "A"}},
            },
        },
    }
    save_vote_store(p, doc_in)
    doc_out = load_vote_store(p)
    assert doc_out["version"] == VOTE_STORE_VERSION
    assert doc_out["comparisons"]["cid1"]["verdicts"]["sk"] == {
        "va": "accepted",
        "vb": "rejected",
    }


def test_comparison_verdict_map_merges_legacy_votes() -> None:
    m = comparison_verdict_map({"votes": {"sk": ["va", "vb"]}})
    assert m["sk"]["va"] == "accepted"
    assert m["sk"]["vb"] == "accepted"


def test_comparison_verdict_map_verdict_overrides_legacy_list() -> None:
    merged = comparison_verdict_map(
        {"verdicts": {"sk": {"va": "rejected"}}, "votes": {"sk": ["va"]}}
    )
    assert merged["sk"]["va"] == "rejected"


def test_reconcile_comparison_block_inplace() -> None:
    blk: dict = {"votes": {"sk": ["a"]}, "verdicts": {}}
    reconcile_comparison_block_inplace(blk)
    assert "votes" not in blk
    assert blk["verdicts"] == {"sk": {"a": "accepted"}}


def test_summarize_verdicts_and_ranking() -> None:
    v_a = VariantRow("a", "g", "A", [], "/x/a.jsonl", None)
    v_b = VariantRow("b", "h", "B", [], "/x/b.jsonl", None)
    m1 = MergedPersonaSample("sk1", {"persona_name": "n"}, [v_a, v_b])
    m2 = MergedPersonaSample("sk2", {"persona_name": "n"}, [v_a, v_b])
    verdicts = {
        "sk1": {"a": "accepted", "b": "rejected"},
        "sk2": {"a": "accepted"},
    }
    summary = summarize_verdicts_for_filtered(verdicts, [m1, m2])
    assert summary["a"] == {"accepted": 2, "rejected": 0, "neutral": 0}
    assert summary["b"] == {"accepted": 0, "rejected": 1, "neutral": 1}
    assert ranked_variant_order(summary)[0] == "a"


def test_neutral_implicit_counts_every_pair() -> None:
    v_a = VariantRow("a", "g", "A", [], "/x/a.jsonl", None)
    m = MergedPersonaSample("sk", {"persona_name": "n"}, [v_a])
    summary = summarize_verdicts_for_filtered({}, [m])
    assert summary["a"] == {"accepted": 0, "rejected": 0, "neutral": 1}
