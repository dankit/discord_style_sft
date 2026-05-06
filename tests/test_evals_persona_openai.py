"""Tests for the vLLM-backed persona generate_fn.

We never talk to a real vLLM server in these tests — we stub the openai
client with an in-process fake so the test suite stays hermetic. The
assertions focus on the invariants the runner relies on:

- completions come back in the same order as inputs (persona eval pairs
  each reference with its generated text by index)
- system prompt is forwarded as the first chat message
- every conversation turn is translated to the right role
- the registered alias is used as the ``model`` field, not the base id
"""
from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from discord_sft.evals.persona import _build_chat_messages


def test_build_chat_messages_round_trips_system_and_turns():
    conv = [
        {"from": "user", "value": "hey"},
        {"from": "assistant", "value": "yo"},
        {"from": "user", "value": "wyd"},
    ]
    msgs = _build_chat_messages(conv, "You are Alice.")
    assert msgs[0] == {"role": "system", "content": "You are Alice."}
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "wyd"


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = types.SimpleNamespace(content=content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Records each chat completion call and echoes a deterministic reply."""

    _ALLOWED_KWARGS = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "extra_body",
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> _FakeResponse:
        unknown = set(kwargs) - self._ALLOWED_KWARGS
        if unknown:
            raise TypeError(f"unexpected OpenAI chat kwargs: {sorted(unknown)}")
        with self._lock:
            self.calls.append(kwargs)
        # Deterministic reply: repeat the last user turn so we can check
        # that ordering is preserved even with concurrent dispatch.
        last_user = next(
            (m["content"] for m in reversed(kwargs["messages"]) if m["role"] == "user"),
            "",
        )
        return _FakeResponse(f"echo:{last_user}")


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, *, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.chat = _FakeChat(_FakeCompletions())


@pytest.fixture
def fake_openai_module(monkeypatch):
    """Install a stub ``openai`` module with our fake OpenAI client.

    ``make_openai_generate_fn`` does ``from openai import OpenAI`` at
    call time, so replacing ``sys.modules["openai"]`` is enough — no
    need for the real SDK to be installed.
    """
    mod = types.ModuleType("openai")
    mod.OpenAI = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


def _prompt_conv(user_text: str) -> list[dict[str, str]]:
    return [{"from": "user", "value": user_text}]


def test_openai_generate_fn_preserves_order_under_concurrency(fake_openai_module):
    from discord_sft.evals.persona import make_openai_generate_fn

    fn = make_openai_generate_fn(
        "http://127.0.0.1:8000/v1",
        model_id="r8",
        max_concurrency=8,
    )
    systems = [f"sys {i}" for i in range(16)]
    conversations = [_prompt_conv(f"msg {i}") for i in range(16)]
    gen_kwargs = {"_conversations": conversations}
    outputs = fn([], systems, gen_kwargs)

    # Ordering must be input-order even though the pool dispatches in
    # parallel — the runner pairs generations with references by index.
    assert outputs == [f"echo:msg {i}" for i in range(16)]


def test_openai_generate_fn_uses_lora_alias_as_model(fake_openai_module):
    # Wrap OpenAI BEFORE building the generate_fn, because the fn captures
    # the class reference at construction time.
    captured: dict[str, _FakeClient] = {}
    orig = fake_openai_module.OpenAI

    def _wrap(**kwargs):
        c = orig(**kwargs)
        captured["client"] = c
        return c

    fake_openai_module.OpenAI = _wrap  # type: ignore[attr-defined]

    from discord_sft.evals.persona import make_openai_generate_fn

    fn = make_openai_generate_fn(
        "http://127.0.0.1:8000/v1",
        model_id="style-late",
        max_concurrency=2,
    )

    fn(
        [],
        ["You are Alice."],
        {"_conversations": [_prompt_conv("hi")]},
    )
    client = captured["client"]
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    # Must hit the lora alias, not the base id.
    assert call["model"] == "style-late"
    # System prompt must be the first message.
    assert call["messages"][0] == {"role": "system", "content": "You are Alice."}
    assert call["messages"][-1] == {"role": "user", "content": "hi"}
    # Defaults follow HF instruct / non-thinking — general (model card).
    assert call["temperature"] == pytest.approx(0.7)
    assert call["top_p"] == pytest.approx(0.8)
    assert call["presence_penalty"] == pytest.approx(1.5)
    assert call["extra_body"]["top_k"] == 20
    assert call["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_generate_fn_thinking_preset_keeps_thinking_enabled(fake_openai_module):
    captured: dict[str, _FakeClient] = {}
    orig = fake_openai_module.OpenAI

    def _wrap(**kwargs):
        c = orig(**kwargs)
        captured["client"] = c
        return c

    fake_openai_module.OpenAI = _wrap  # type: ignore[attr-defined]

    from discord_sft.evals.persona import make_openai_generate_fn

    fn = make_openai_generate_fn(
        "http://127.0.0.1:8000/v1",
        model_id="style-late",
        max_concurrency=1,
        qwen_sampling="thinking_general",
    )

    fn([], ["You are thoughtful."], {"_conversations": [_prompt_conv("hi")]})

    call = captured["client"].chat.completions.calls[0]
    assert call["temperature"] == pytest.approx(1.0)
    assert call["top_p"] == pytest.approx(0.95)
    assert call["extra_body"] == {"top_k": 20}
