"""Utilities for low-cost LoRA architecture search before full SFT runs."""

from __future__ import annotations

__all__ = ["collect_gradient_probe"]


def __getattr__(name: str):
    if name == "collect_gradient_probe":
        from discord_sft.training.lora_search.gradient_probe import collect_gradient_probe

        return collect_gradient_probe
    raise AttributeError(f"module 'discord_sft.training.lora_search' has no attribute {name!r}")
