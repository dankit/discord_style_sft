"""Unsloth + LoRA supervised fine-tuning for Qwen3.5-35B-A3B.

The whole subpackage is lazy-imported: nothing here pulls in torch / unsloth /
trl at module load time, so `import discord_sft` stays fast and the core
install is not forced to drag in the training deps.
"""
from __future__ import annotations

__all__ = ["TrainConfig", "load_config", "run_training"]


def __getattr__(name: str):
    if name in {"TrainConfig", "load_config"}:
        from discord_sft.training.config import TrainConfig, load_config

        return {"TrainConfig": TrainConfig, "load_config": load_config}[name]
    if name == "run_training":
        from discord_sft.training.trainer import run_training

        return run_training
    raise AttributeError(f"module 'discord_sft.training' has no attribute {name!r}")
