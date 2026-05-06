"""Tests for :mod:`discord_sft.training.merge_peft` and train merge-peft CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from discord_sft.cli.parser import build_parser
from discord_sft.training.merge_peft import (
    resolve_base_model_id,
    resolve_train_config_for_merge,
)


def test_resolve_base_model_override(tmp_path: Path) -> None:
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert resolve_base_model_id(ckpt, override="hub/base") == "hub/base"


def test_resolve_base_model_run_json(tmp_path: Path) -> None:
    out = tmp_path / "run"
    ckpt = out / "final"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"base_model": "unsloth/Qwen3.5-35B-A3B"}),
        encoding="utf-8",
    )
    assert resolve_base_model_id(ckpt) == "unsloth/Qwen3.5-35B-A3B"


def test_resolve_base_model_peft_fallback(tmp_path: Path) -> None:
    ckpt = tmp_path / "final"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-35B-A3B"}),
        encoding="utf-8",
    )
    assert resolve_base_model_id(ckpt) == "Qwen/Qwen3.5-35B-A3B"


def test_resolve_train_config_explicit_wins_over_parent_yaml(
    tmp_path: Path, tiny_train_yaml: Path
) -> None:
    """Explicit --config YAML is preferred over parent's config.resolved.yaml."""
    out = tmp_path / "training_out"
    ckpt = out / "final"
    ckpt.mkdir(parents=True)

    stale = {"model": {"name": "stale/StaleBase", "max_seq_length": 99}}
    (out / "config.resolved.yaml").write_text(
        f"run_name: stale\nmodel:\n  name: {stale['model']['name']}\n"
        f"  max_seq_length: {stale['model']['max_seq_length']}\n"
        + "data: {}\nlora:\n  target_modules:\n"
        + "  - q_proj\ntrain: {}\ncheckpoint: {}\n",
        encoding="utf-8",
    )

    cfg = resolve_train_config_for_merge(ckpt, tiny_train_yaml.resolve())
    assert cfg is not None
    assert cfg.model.name != "stale/StaleBase"
    assert cfg.model.max_seq_length != 99


@pytest.fixture(scope="session")
def tiny_train_yaml(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("merge_yaml")
    p = root / "tiny.yaml"
    p.write_text(
        "run_name: merge-test\n"
        "model:\n"
        "  name: unsloth/Qwen3.5-35B-A3B\n"
        "  max_seq_length: 2048\n"
        "  load_in_16bit: true\n"
        "data: {}\n"
        "lora:\n"
        "  target_modules:\n"
        "    - q_proj\n"
        "train: {}\n"
        "checkpoint: {}\n",
        encoding="utf-8",
    )
    return p


def test_parser_train_run_default_action() -> None:
    args = build_parser().parse_args(["train", "--config", "recipe.yaml"])
    assert args.train_action == "run"
    assert args.config == "recipe.yaml"


def test_parser_train_merge_peft() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "merge-peft",
            "--adapter",
            "out/lora/run/final",
            "--output",
            "out/merged/run-final",
            "--max-shard-size",
            "4GB",
        ]
    )
    assert args.train_action == "merge-peft"
    assert args.merge_adapter == "out/lora/run/final"
    assert args.merge_output == "out/merged/run-final"
    assert args.merge_max_shard_size == "4GB"
