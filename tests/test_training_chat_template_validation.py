from __future__ import annotations

import json

from discord_sft.training.data import validate_chat_template_dataset


class _GoodTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        if chat_template_kwargs is not None:
            enable_thinking = bool(chat_template_kwargs.get("enable_thinking", False))
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "assistant" and enable_thinking:
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)


class _BrokenTokenizer(_GoodTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        rendered = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )
        return rendered.replace("<|im_end|>\n", "", 1)


def _sample() -> dict:
    return {
        "system": "You are Alice chatting with Bob.",
        "conversations": [
            {"from": "user", "value": "hey"},
            {"from": "assistant", "value": "yo"},
            {"from": "user", "value": "wyd"},
            {"from": "assistant", "value": "nm"},
        ],
        "meta": {"persona_id": "a"},
    }


def test_validate_chat_template_dataset_passes_good_template(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")

    report = validate_chat_template_dataset(path, _GoodTokenizer(), max_samples=10)

    assert report["samples_checked"] == 1
    assert report["samples_failed"] == 0
    assert report["errors"] == []


def test_validate_chat_template_dataset_flags_missing_im_end(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")

    report = validate_chat_template_dataset(path, _BrokenTokenizer(), max_samples=10)

    assert report["samples_checked"] == 1
    assert report["samples_failed"] == 1
    assert report["errors"]
    assert any(
        "im_end count too low" in issue
        for issue in report["errors"][0]["issues"]
    )
