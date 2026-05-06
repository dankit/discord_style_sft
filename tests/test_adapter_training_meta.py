"""Tests for :mod:`discord_sft.evals.adapter_training_meta`."""

from __future__ import annotations

import json
from pathlib import Path

from discord_sft.evals.adapter_training_meta import read_training_base_model_from_adapter


def test_read_training_base_model_from_adapter(tmp_path: Path) -> None:
    out = tmp_path / "run"
    ckpt = out / "final"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"base_model": "unsloth/Qwen3.5-35B-A3B"}),
        encoding="utf-8",
    )
    assert read_training_base_model_from_adapter(ckpt) == "unsloth/Qwen3.5-35B-A3B"


def test_read_training_base_model_missing_run_json(tmp_path: Path) -> None:
    ckpt = tmp_path / "final"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert read_training_base_model_from_adapter(ckpt) is None


def test_read_training_base_model_not_adapter_dir(tmp_path: Path) -> None:
    assert read_training_base_model_from_adapter(tmp_path) is None
