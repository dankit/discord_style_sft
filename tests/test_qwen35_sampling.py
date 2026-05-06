from __future__ import annotations

import pytest

from discord_sft.evals.qwen35_sampling import (
    INSTRUCT_GENERAL,
    THINKING_GENERAL,
    hf_generate_sampling_kwargs,
    openai_chat_sampling_kwargs,
    qwen35_preset,
)


def test_qwen35_preset_instruct_general_values():
    p = qwen35_preset("instruct_general")
    assert p is INSTRUCT_GENERAL
    assert p.temperature == pytest.approx(0.7)
    assert p.top_p == pytest.approx(0.8)
    assert p.top_k == 20
    assert p.presence_penalty == pytest.approx(1.5)


def test_qwen35_preset_thinking_general():
    p = qwen35_preset("thinking_general")
    assert p is THINKING_GENERAL
    assert p.temperature == pytest.approx(1.0)
    oa = openai_chat_sampling_kwargs(p)
    assert oa["top_p"] == pytest.approx(0.95)


def test_qwen35_preset_invalid():
    with pytest.raises(ValueError, match="unknown qwen_sampling"):
        qwen35_preset("not_a_mode")


def test_hf_generate_sampling_includes_repetition_penalty():
    kw = hf_generate_sampling_kwargs(INSTRUCT_GENERAL)
    assert kw["repetition_penalty"] == pytest.approx(1.0)
    assert "presence_penalty" not in kw
