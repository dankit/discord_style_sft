"""Shared baseline-prompt preview text for Evals UI and Commands Builder."""

from __future__ import annotations

from typing import Literal

from discord_sft.evals.baseline_prompt import build_baseline_system_prompt

BASELINE_PROMPT_HELP = {
    "minimal": "Use the exact held-out val.jsonl system prompt. This is the fairest no-adapter control because it matches what the LoRA saw during training.",
    "style": "Start from the minimal prompt, then add generic Discord-DM style guidance such as casual phrasing, brevity, slang, and chat-like rhythm.",
    "profile": "Start from the minimal prompt, then add persona-specific guidance mined from profiles.json, including length, lowercase rate, fillers, emoji, and burst patterns.",
}

EXAMPLE_PROFILE = {
    "length": {"mean_words": 9.0, "p50": 7, "p95": 18},
    "lowercase_start_rate": 0.72,
    "emoji": {"unicode_per_turn": 0.18, "custom_per_turn": 0.06},
    "burst_rate": 1.6,
    "top_fillers": {
        "1gram": [
            {"token": "lol"},
            {"token": "ngl"},
            {"token": "lowkey"},
            {"token": "fr"},
        ]
    },
}


def baseline_prompt_example(mode: Literal["minimal", "style", "profile"]) -> str:
    return build_baseline_system_prompt(
        mode,
        persona_name="Maya",
        counterparty_names=["Jordan"],
        minimal_system="You are Maya chatting with Jordan.",
        profile=EXAMPLE_PROFILE,
    )
