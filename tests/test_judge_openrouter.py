"""Hermetic tests for OpenRouter-backed style judge (OpenAI-compatible client)."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest


def test_parse_judge_json_strips_markdown_fence():
    from discord_sft.evals.judge import _parse_judge_json

    raw = """```json
{"rationale":{"vocabulary":"a","tone":"b","length":"c","authentic_persona":"d"},"vocabulary":5,"tone":4,"length":3,"authentic_persona":5}
```"""
    out = _parse_judge_json(raw)
    assert out["vocabulary"] == 5
    assert out["tone"] == 4
    assert out["rationale"]["vocabulary"] == "a"


def test_finalize_style_rating_weighted_overall():
    from discord_sft.evals.judge import finalize_style_rating

    raw = {
        "rationale": {
            k: f"why {k}"
            for k in ("vocabulary", "tone", "length", "authentic_persona")
        },
        "vocabulary": 5,
        "tone": 4,
        "length": 3,
        "authentic_persona": 5,
        "overall": 1,
    }
    out = finalize_style_rating(raw)
    # (5+4+3+5 + 2) // 4 == 4
    assert out["overall"] == 4
    assert out["vocabulary"] == 5
    assert isinstance(out["overall"], int)
    assert "why vocabulary" in out["rationale"]["vocabulary"]
    assert out["reasoning"] == ""


def test_finalize_style_rating_preserves_reasoning():
    from discord_sft.evals.judge import finalize_style_rating

    raw = {
        "reasoning": "Compared slang and tone before scoring.",
        "rationale": {
            k: f"why {k}"
            for k in ("vocabulary", "tone", "length", "authentic_persona")
        },
        "vocabulary": 5,
        "tone": 5,
        "length": 5,
        "authentic_persona": 5,
    }
    out = finalize_style_rating(raw)
    assert out["reasoning"] == "Compared slang and tone before scoring."
    assert out["overall"] == 5


def test_parse_judge_json_extracts_object_after_preamble():
    from discord_sft.evals.judge import _parse_judge_json

    raw = """Thoughts first (invalid JSON prose).
{"reasoning":"x","rationale":{"vocabulary":"a","tone":"b","length":"c","authentic_persona":"d"},"vocabulary":5,"tone":5,"length":5,"authentic_persona":5}
"""
    out = _parse_judge_json(raw)
    assert out["reasoning"] == "x"
    assert out["vocabulary"] == 5


def test_make_judge_openrouter_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from discord_sft.evals.judge import make_judge

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_judge("openrouter")


@pytest.fixture
def fake_openai_for_judge(monkeypatch):
    """Stub ``openai.OpenAI`` so OpenRouterJudge never hits the network."""

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = types.SimpleNamespace(content=content)

    class _FakeResponse:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]

    class _FakeCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> _FakeResponse:
            self.calls.append(kwargs)
            rationale = {
                "vocabulary": "Matches slang.",
                "tone": "Similar energy.",
                "length": "Comparable density.",
                "authentic_persona": "Human and on-brand.",
            }
            payload = json.dumps(
                {
                    "reasoning": "Step through axes then score.",
                    "rationale": rationale,
                    "vocabulary": 5,
                    "tone": 4,
                    "length": 3,
                    "authentic_persona": 5,
                }
            )
            return _FakeResponse(payload)

    captured: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            comp = _FakeCompletions()
            self.chat = types.SimpleNamespace(completions=comp)

    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return captured


def test_openrouter_judge_uses_openrouter_defaults(monkeypatch, fake_openai_for_judge):
    captured = fake_openai_for_judge
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    from discord_sft.evals.judge import OpenRouterJudge

    j = OpenRouterJudge()
    out = j.score(
        real_message="hi",
        generated_message="hey",
        context="ctx",
        persona_name="Pat",
    )
    assert out["overall"] == 4
    assert isinstance(out["rationale"], dict)
    assert out["reasoning"] == "Step through axes then score."
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "sk-test"
    assert captured["default_headers"]["X-Title"] == "discord-sft"
    call = j._client.chat.completions.calls[0]
    assert call["model"] == "anthropic/claude-sonnet-4.6"
    assert call["max_tokens"] == 2000
    assert "Pat" in call["messages"][0]["content"]


def test_openrouter_judge_system_instructions_in_prompt(monkeypatch, fake_openai_for_judge):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    from discord_sft.evals.judge import OpenRouterJudge

    j = OpenRouterJudge()
    j.score(
        real_message="hi",
        generated_message="hey",
        context="ctx",
        persona_name="Pat",
        system_instructions="You are Pat in Discord.",
    )
    call = j._client.chat.completions.calls[0]
    content = call["messages"][0]["content"]
    assert "System / instructions" in content
    assert "You are Pat in Discord." in content


def test_openrouter_judge_optional_referer_header(monkeypatch, fake_openai_for_judge):
    captured = fake_openai_for_judge
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.org/judge")

    from discord_sft.evals.judge import OpenRouterJudge

    OpenRouterJudge()
    assert captured["default_headers"]["HTTP-Referer"] == "https://example.org/judge"
