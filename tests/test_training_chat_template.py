from __future__ import annotations

from discord_sft.training.data import (
    CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED,
    CHAT_TEMPLATE_KWARGS_ACCEPTED,
    CHAT_TEMPLATE_KWARGS_FALLBACK,
    response_only_char_labels,
    render_sharegpt_text,
)


class FakeQwenTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        if chat_template_kwargs is not None:
            enable_thinking = bool(chat_template_kwargs.get("enable_thinking", True))
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant" and enable_thinking:
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)


class LegacyFakeQwenTokenizer(FakeQwenTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        if enable_thinking is not True:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        if chat_template_kwargs is not None:
            raise TypeError("unexpected keyword argument 'chat_template_kwargs'")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True,
        )


class NestedKwargsFakeQwenTokenizer(FakeQwenTokenizer):
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = True,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        if chat_template_kwargs is None and enable_thinking is not True:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )


def _sample() -> dict:
    return {
        "system": "You are Alice chatting with Bob.",
        "conversations": [
            {"from": "user", "value": "hey"},
            {"from": "assistant", "value": "yo"},
            {"from": "user", "value": "wyd"},
            {"from": "assistant", "value": "nothing rn"},
        ],
    }


def _response_only_char_labels(
    text: str,
    *,
    instruction_part: str = "<|im_start|>user\n",
    response_part: str = "<|im_start|>assistant\n",
) -> list[int]:
    labels = [-100] * len(text)
    cursor = 0
    while True:
        response_start = text.find(response_part, cursor)
        if response_start < 0:
            break
        supervised_start = response_start + len(response_part)
        next_instruction = text.find(instruction_part, supervised_start)
        supervised_end = len(text) if next_instruction < 0 else next_instruction
        for idx in range(supervised_start, supervised_end):
            labels[idx] = idx
        cursor = supervised_end
    return labels


def test_render_sharegpt_text_disables_thinking_by_default():
    text, kwargs_path = render_sharegpt_text(_sample(), FakeQwenTokenizer())

    assert kwargs_path == CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED
    assert "<think>" not in text
    assert "<|im_start|>system\nYou are Alice chatting with Bob.<|im_end|>\n" in text
    assert "<|im_start|>user\nhey<|im_end|>\n" in text
    assert "<|im_start|>assistant\nyo<|im_end|>\n" in text


def test_render_sharegpt_text_can_enable_thinking():
    text, kwargs_path = render_sharegpt_text(
        _sample(), FakeQwenTokenizer(), enable_thinking=True
    )

    assert kwargs_path == CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\nyo<|im_end|>\n" in text


def test_render_sharegpt_text_records_legacy_template_fallback():
    text, kwargs_path = render_sharegpt_text(_sample(), LegacyFakeQwenTokenizer())

    assert kwargs_path == CHAT_TEMPLATE_KWARGS_FALLBACK
    # Legacy fallback preserves old tokenizer behavior, which is why the
    # manifest records it for review.
    assert "<think>" in text


def test_render_sharegpt_text_supports_nested_template_kwargs_fallback():
    text, kwargs_path = render_sharegpt_text(_sample(), NestedKwargsFakeQwenTokenizer())

    assert kwargs_path == CHAT_TEMPLATE_KWARGS_ACCEPTED
    assert "<think>" not in text


def test_response_only_mask_boundary_keeps_user_tokens_masked():
    text, _ = render_sharegpt_text(_sample(), FakeQwenTokenizer())
    labels = response_only_char_labels(text)

    for user_text in ("hey", "wyd"):
        start = text.index(user_text)
        assert labels[start : start + len(user_text)] == [-100] * len(user_text)

    for assistant_text in ("yo", "nothing rn"):
        start = text.index(assistant_text)
        assert all(
            label != -100 for label in labels[start : start + len(assistant_text)]
        )


def test_response_only_char_labels_empty_for_no_assistant_turn():
    text = "<|im_start|>user\nhello<|im_end|>\n"
    labels = response_only_char_labels(text)
    assert labels == [-100] * len(text)
