"""Tests for :func:`discord_sft.training.config.load_config_for_merge_parity`."""

from __future__ import annotations

from pathlib import Path

from discord_sft.training.config import load_config_for_merge_parity


def test_load_config_for_merge_parity_clears_layers_last_pct_when_both_set(
    tmp_path: Path,
) -> None:
    """Legacy resolved YAML could keep pct after expanding layers_to_transform."""
    p = tmp_path / "config.resolved.yaml"
    p.write_text(
        "run_name: x\n"
        "model:\n"
        "  name: unsloth/Qwen3.5-35B-A3B\n"
        "  max_seq_length: 1024\n"
        "data: {}\n"
        "lora:\n"
        "  layers_last_pct: 0.25\n"
        "  layers_to_transform:\n"
        "    - 40\n"
        "    - 41\n"
        "  target_modules:\n"
        "    - q_proj\n"
        "train: {}\n"
        "checkpoint: {}\n",
        encoding="utf-8",
    )
    cfg = load_config_for_merge_parity(p)
    assert cfg.lora.layers_to_transform == [40, 41]
    assert cfg.lora.layers_last_pct is None
