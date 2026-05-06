"""ShareGPT JSONL → HF ``Dataset`` with the Qwen chat template applied.

Schema expected (one JSON object per line, produced by
``discord-sft build-sft``):

.. code-block:: json

    {
      "system": "You are Alice chatting with Bob.",
      "conversations": [
        {"from": "user", "value": "..."},
        {"from": "assistant", "value": "..."}
      ],
      "meta": {...}
    }

Conversion mirrors Unsloth's standard chat-SFT recipe: each sample becomes
a single ``"text"`` string with the full multi-turn prompt rendered via
``tokenizer.apply_chat_template(..., tokenize=False)``. Qwen3.5's thinking
scaffold is disabled by default for chat/style SFT. Loss masking (user spans
ignored) is applied at trainer construction time via
``unsloth.chat_templates.train_on_responses_only``; we do not pre-tokenize
here because SFTTrainer handles tokenisation uniformly from ``"text"``.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterator


_ROLE_MAP = {"user": "user", "assistant": "assistant", "system": "system"}

CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED = "enable_thinking_kwarg"
CHAT_TEMPLATE_KWARGS_ACCEPTED = "chat_template_kwargs"
CHAT_TEMPLATE_KWARGS_FALLBACK = "fallback_no_thinking_control"


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from e


def sharegpt_to_messages(
    sample: dict[str, Any],
    *,
    system_prompt_override: str | None = None,
) -> list[dict[str, str]]:
    """Render one ShareGPT sample as OpenAI-style ``messages``.

    The stored ``system`` field (per-persona, populated by build-sft) is
    prepended as a system message unless ``system_prompt_override`` is set.
    Unknown ``from`` values raise — we'd rather fail loudly than silently
    drop turns.
    """
    messages: list[dict[str, str]] = []
    system = (
        system_prompt_override
        if system_prompt_override is not None
        else sample.get("system")
    )
    if system:
        messages.append({"role": "system", "content": str(system)})
    for turn in sample.get("conversations", []):
        src = turn.get("from")
        role = _ROLE_MAP.get(src)
        if role is None:
            raise ValueError(f"Unknown 'from' role {src!r} in sample")
        messages.append({"role": role, "content": str(turn.get("value", ""))})
    return messages


def render_chat_template_for_sft(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> tuple[str, str]:
    """Render messages for SFT and report the chat-template kwargs path used."""

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        return text, CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED
    except TypeError as e:
        if "enable_thinking" not in str(e) and "unexpected keyword" not in str(e):
            raise
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        return text, CHAT_TEMPLATE_KWARGS_ACCEPTED
    except TypeError as e:
        if "chat_template_kwargs" not in str(e) and "unexpected keyword" not in str(e):
            raise
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text, CHAT_TEMPLATE_KWARGS_FALLBACK


def render_sharegpt_text(
    sample: dict[str, Any],
    tokenizer: Any,
    *,
    system_prompt_override: str | None = None,
    enable_thinking: bool = False,
) -> tuple[str, str]:
    """Convert one ShareGPT sample to templated training text."""

    messages = sharegpt_to_messages(
        sample, system_prompt_override=system_prompt_override
    )
    return render_chat_template_for_sft(
        tokenizer, messages, enable_thinking=enable_thinking
    )


def response_only_char_labels(
    text: str,
    *,
    instruction_part: str = "<|im_start|>user\n",
    response_part: str = "<|im_start|>assistant\n",
    ignore_index: int = -100,
) -> list[int]:
    """Char-level approximation of response-only masking boundaries.

    Mirrors ``train_on_responses_only`` semantics: spans after each assistant
    turn marker are supervised until the next user marker (or end of text).
    """

    labels = [ignore_index] * len(text)
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


def load_sharegpt_dataset(
    jsonl_path: str | Path,
    tokenizer: Any,
    *,
    system_prompt_override: str | None = None,
    enable_thinking: bool = False,
    template_metadata: dict[str, Any] | None = None,
    num_proc: int | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 3407,
):
    """Load a ShareGPT JSONL file as an HF ``Dataset`` keyed by ``text``.

    Parameters
    ----------
    jsonl_path:
        File produced by ``discord-sft build-sft`` (``train.jsonl`` /
        ``val.jsonl``).
    tokenizer:
        The tokenizer returned by Unsloth's ``FastModel.from_pretrained``.
        Must have ``apply_chat_template`` — Qwen tokenizers ship one.
    system_prompt_override:
        If set, replaces each sample's stored ``system`` field (useful for
        ablations).
    enable_thinking:
        Passed to Qwen3-style chat templates via ``chat_template_kwargs``.
        Defaults to False so SFT does not supervise empty ``<think>`` blocks.
    template_metadata:
        Optional mutable dict populated with thinking-control support details
        for run manifests.
    num_proc:
        Forwarded to ``Dataset.map``; defaults to None (serial) because the
        templating step is cheap and process pools add startup overhead.
    shuffle:
        If True, shuffle jsonl row order before templating so file layout (e.g.
        turn-length blocks from build-sft) does not dictate optimization steps.
    shuffle_seed:
        Seed for row shuffle; training passes ``train.seed`` for reproducibility.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install with `pip install 'discord-sft[train]'`."
        ) from e

    rows = list(_iter_jsonl(jsonl_path))
    if not rows:
        raise ValueError(f"No samples found in {jsonl_path}")
    if shuffle:
        random.Random(int(shuffle_seed)).shuffle(rows)

    ds = Dataset.from_list(rows)

    # Probe once so the manifest can record whether the tokenizer accepted
    # Qwen's chat_template_kwargs. The map below still renders each row.
    _, kwargs_path = render_sharegpt_text(
        rows[0],
        tokenizer,
        system_prompt_override=system_prompt_override,
        enable_thinking=enable_thinking,
    )
    if template_metadata is not None:
        template_metadata["enable_thinking"] = bool(enable_thinking)
        template_metadata["thinking_control"] = kwargs_path

    def _format(example: dict[str, Any]) -> dict[str, Any]:
        text, _ = render_sharegpt_text(
            example,
            tokenizer,
            system_prompt_override=system_prompt_override,
            enable_thinking=enable_thinking,
        )
        return {"text": text}

    keep_columns = ["text"]
    ds = ds.map(
        _format,
        remove_columns=[c for c in ds.column_names if c not in keep_columns],
        num_proc=num_proc,
        desc=f"Templating {Path(jsonl_path).name}",
    )
    return ds


_USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n.*?<\|im_end\|>\n?", re.DOTALL)
_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>assistant\n.*?<\|im_end\|>\n?", re.DOTALL
)


def validate_chat_template_dataset(
    jsonl_path: str | Path,
    tokenizer: Any,
    *,
    system_prompt_override: str | None = None,
    enable_thinking: bool = False,
    max_samples: int | None = None,
    max_errors: int = 20,
) -> dict[str, Any]:
    """Validate rendered chat-template boundaries for a ShareGPT dataset.

    This checks whether rendered samples appear to contain properly terminated
    user/assistant turns (``<|im_end|>``) and expected role counts.
    """

    samples_checked = 0
    samples_passed = 0
    samples_failed = 0
    errors: list[dict[str, Any]] = []
    counts = {
        "expected_user_turns": 0,
        "expected_assistant_turns": 0,
        "rendered_user_turns": 0,
        "rendered_assistant_turns": 0,
        "rendered_im_end_tokens": 0,
    }

    for idx, sample in enumerate(_iter_jsonl(jsonl_path), start=1):
        if max_samples is not None and samples_checked >= max_samples:
            break
        text, kwargs_path = render_sharegpt_text(
            sample,
            tokenizer,
            system_prompt_override=system_prompt_override,
            enable_thinking=enable_thinking,
        )

        conversations = sample.get("conversations", [])
        expected_user = sum(1 for t in conversations if t.get("from") == "user")
        expected_assistant = sum(
            1 for t in conversations if t.get("from") == "assistant"
        )
        expected_messages = len(conversations) + (1 if sample.get("system") else 0)

        rendered_user = text.count("<|im_start|>user\n")
        rendered_assistant = text.count("<|im_start|>assistant\n")
        rendered_im_end = text.count("<|im_end|>")
        rendered_user_blocks = len(_USER_BLOCK_RE.findall(text))
        rendered_assistant_blocks = len(_ASSISTANT_BLOCK_RE.findall(text))

        counts["expected_user_turns"] += expected_user
        counts["expected_assistant_turns"] += expected_assistant
        counts["rendered_user_turns"] += rendered_user
        counts["rendered_assistant_turns"] += rendered_assistant
        counts["rendered_im_end_tokens"] += rendered_im_end

        issues: list[str] = []
        if rendered_user != expected_user:
            issues.append(f"user turn count mismatch: expected {expected_user}, got {rendered_user}")
        if rendered_assistant != expected_assistant:
            issues.append(
                f"assistant turn count mismatch: expected {expected_assistant}, got {rendered_assistant}"
            )
        if rendered_user_blocks != expected_user:
            issues.append(
                f"user block termination mismatch: expected {expected_user}, got {rendered_user_blocks}"
            )
        if rendered_assistant_blocks != expected_assistant:
            issues.append(
                "assistant block termination mismatch: "
                f"expected {expected_assistant}, got {rendered_assistant_blocks}"
            )
        if rendered_im_end < expected_messages:
            issues.append(
                f"im_end count too low: expected >= {expected_messages}, got {rendered_im_end}"
            )
        if text.rstrip().endswith("<|im_start|>assistant"):
            issues.append("sample appears to end with an unfinished assistant start token")

        samples_checked += 1
        if issues:
            samples_failed += 1
            if len(errors) < max_errors:
                errors.append(
                    {
                        "sample_index": idx,
                        "issues": issues,
                        "thinking_control": kwargs_path,
                        "preview": text[:400],
                    }
                )
        else:
            samples_passed += 1

    return {
        "dataset_path": str(Path(jsonl_path)),
        "samples_checked": samples_checked,
        "samples_passed": samples_passed,
        "samples_failed": samples_failed,
        "pass_rate": (samples_passed / samples_checked) if samples_checked else 0.0,
        "counts": counts,
        "errors": errors,
    }


__all__ = [
    "CHAT_TEMPLATE_ENABLE_THINKING_ACCEPTED",
    "CHAT_TEMPLATE_KWARGS_ACCEPTED",
    "CHAT_TEMPLATE_KWARGS_FALLBACK",
    "load_sharegpt_dataset",
    "response_only_char_labels",
    "render_chat_template_for_sft",
    "render_sharegpt_text",
    "sharegpt_to_messages",
    "validate_chat_template_dataset",
]
