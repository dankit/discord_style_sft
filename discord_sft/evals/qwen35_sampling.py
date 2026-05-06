"""Official Qwen3.5-35B-A3B sampling presets from the Hugging Face model card.

Values are copied verbatim from:
https://huggingface.co/Qwen/Qwen3.5-35B-A3B
(section: recommended sampling parameters).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

MODEL_CARD_URL: Final[str] = "https://huggingface.co/Qwen/Qwen3.5-35B-A3B"

QwenSamplingName = Literal[
    "instruct_general",
    "thinking_general",
    "thinking_coding",
    "instruct_reasoning",
]


@dataclass(frozen=True)
class Qwen35SamplingPreset:
    """One row from the model card's sampling table."""

    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float


# Thinking mode — general tasks
THINKING_GENERAL = Qwen35SamplingPreset(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

# Thinking mode — precise coding (e.g. WebDev)
THINKING_CODING = Qwen35SamplingPreset(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=0.0,
    repetition_penalty=1.0,
)

# Instruct / non-thinking — general tasks (default for short persona replies)
INSTRUCT_GENERAL = Qwen35SamplingPreset(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

# Instruct / non-thinking — reasoning tasks
INSTRUCT_REASONING = Qwen35SamplingPreset(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
)

_PRESETS: dict[str, Qwen35SamplingPreset] = {
    "instruct_general": INSTRUCT_GENERAL,
    "thinking_general": THINKING_GENERAL,
    "thinking_coding": THINKING_CODING,
    "instruct_reasoning": INSTRUCT_REASONING,
}

QWEN_SAMPLING_CHOICES: tuple[str, ...] = tuple(_PRESETS.keys())

DEFAULT_QWEN_SAMPLING: Final[str] = "instruct_general"


def qwen35_preset(name: str) -> Qwen35SamplingPreset:
    """Resolve a CLI / API preset name to :class:`Qwen35SamplingPreset`."""
    key = (name or "").strip()
    if key not in _PRESETS:
        choices = ", ".join(sorted(_PRESETS))
        raise ValueError(f"unknown qwen_sampling {name!r}; expected one of: {choices}")
    return _PRESETS[key]


def openai_chat_sampling_kwargs(preset: Qwen35SamplingPreset) -> dict[str, float | int]:
    """Keyword args for ``openai.chat.completions.create`` / vLLM OpenAI API.

    ``repetition_penalty`` is not part of the OpenAI chat schema; callers
    that need it should use provider-specific ``extra_body`` (not set here).
    """
    return {
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "presence_penalty": preset.presence_penalty,
    }


def hf_generate_sampling_kwargs(preset: Qwen35SamplingPreset) -> dict[str, float | int]:
    """Keyword args for ``transformers`` ``model.generate`` sampling."""
    return {
        "temperature": preset.temperature,
        "top_p": preset.top_p,
        "top_k": preset.top_k,
        "repetition_penalty": preset.repetition_penalty,
    }


__all__ = [
    "DEFAULT_QWEN_SAMPLING",
    "INSTRUCT_GENERAL",
    "INSTRUCT_REASONING",
    "MODEL_CARD_URL",
    "QWEN_SAMPLING_CHOICES",
    "Qwen35SamplingPreset",
    "QwenSamplingName",
    "THINKING_CODING",
    "THINKING_GENERAL",
    "hf_generate_sampling_kwargs",
    "openai_chat_sampling_kwargs",
    "qwen35_preset",
]
