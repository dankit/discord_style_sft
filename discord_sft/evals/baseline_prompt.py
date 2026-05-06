"""Style-aware baseline system prompts for no-adapter eval runs.

The default system prompt that ``build-sft`` bakes into ``val.jsonl`` is
deliberately weak:

    You are {persona_name} chatting with {others}.

That's what the LoRA is trained under, and keeping it weak is what makes
LoRA have something to learn (style is conveyed by the training data, not
the prompt). But it's a poor **baseline** for answering the question
"what can the base model do with explicit style instruction?" — which is
the natural control for any LoRA recipe.

This module builds three system-prompt variants, keyed by a
``BaselinePromptMode``:

* ``minimal`` — the val.jsonl prompt verbatim. This is the fair control
  for "does the LoRA lift style scores above no-prompt-help?" Default so
  existing recipes stay apples-to-apples with what shipped before.
* ``style`` — generic Discord-DM style guidance appended to the minimal
  prompt. Use when a ``profiles.json`` is unavailable or for personas
  that weren't fingerprinted.
* ``profile`` — per-persona style guidance derived from the mined
  fingerprint (length stats, lowercase-start rate, top n-gram fillers,
  emoji rates, burst rate). This is the strongest base-model baseline and
  directly targets each judge axis and heuristic the eval scores on.

The bullets in each mode map 1:1 to the things the eval is measuring so a
base-model reply has a fair shot at scoring on all of them:

* VOCABULARY (judge) / filler_rate (heuristic) → top-N mined fillers.
* TONE (judge)                                 → "casual, no-AI" clause.
* LENGTH (judge) / avg_length_diff (heuristic) → per-persona mean / p95.
* AUTHENTIC PERSONA (judge) / caps_rate + name-context heuristics → lowercase-start rate and persona-specific cues.

The output prompt is always a plain string, so it slots in wherever
``PersonaEvalSample.system`` is consumed (HF / OpenAI-server / custom
generate_fns alike).
"""
from __future__ import annotations

from typing import Any, Literal, Mapping

BaselinePromptMode = Literal["minimal", "style", "profile"]

BASELINE_PROMPT_MODES: tuple[str, ...] = ("minimal", "style", "profile")

# Mined filler list from :mod:`discord_sft.analysis.heuristics` shares the same
# defaults; keep them in sync so ``style`` mode's bullet matches what the
# heuristic is scoring against when no profile is provided.
_STYLE_FALLBACK_FILLERS = ("lol", "ngl", "tbh", "fr", "lowkey", "bro")


def build_baseline_system_prompt(
    mode: BaselinePromptMode,
    *,
    persona_name: str,
    counterparty_names: list[str] | None = None,
    minimal_system: str | None = None,
    profile: Mapping[str, Any] | None = None,
) -> str:
    """Return the system prompt to use for a base-model baseline run.

    Parameters
    ----------
    mode:
        Which prompt flavour to build. See module docstring.
    persona_name:
        The persona being imitated (``meta.persona_name`` from the sample).
    counterparty_names:
        The other chat participants, if known. Used only by ``style`` /
        ``profile`` modes when ``minimal_system`` isn't provided, to
        reconstruct the minimal header. ``build-sft`` already bakes these
        into the system string, so normally ``minimal_system`` is enough.
    minimal_system:
        The exact system string from the val.jsonl sample (i.e., what the
        LoRA was trained with). Returned verbatim in ``minimal`` mode;
        used as the first line of the prompt in ``style`` / ``profile``.
        If ``None``, a header is reconstructed from ``persona_name`` and
        ``counterparty_names``.
    profile:
        A single-persona entry from ``profiles.json``. Required for
        ``profile`` mode; ignored otherwise. Expected keys mirror
        :func:`discord_sft.analysis.fingerprint.build_profiles`:

        * ``length``: ``{mean_words, p50, p95}``
        * ``lowercase_start_rate``: float in [0, 1]
        * ``emoji``: ``{unicode_per_turn, custom_per_turn}``
        * ``burst_rate``: float (avg newline-separated segments per turn)
        * ``top_fillers``: ``{"1gram": [{"token": ..., "score": ...}, ...]}``
    """
    if mode not in BASELINE_PROMPT_MODES:
        raise ValueError(
            f"unknown baseline prompt mode {mode!r}; "
            f"expected one of {BASELINE_PROMPT_MODES}"
        )

    header = _header(persona_name, counterparty_names, minimal_system)

    if mode == "minimal":
        return header

    if mode == "style":
        return header + _style_bullets(persona_name, fillers=_STYLE_FALLBACK_FILLERS)

    # mode == "profile"
    if not profile:
        # Graceful downgrade: if a profile isn't available for this persona
        # (e.g. profiles.json has other personas but not this one), fall
        # back to generic style. Silent rather than erroring — missing
        # fingerprints shouldn't tank a whole eval run.
        return header + _style_bullets(persona_name, fillers=_STYLE_FALLBACK_FILLERS)

    return header + _profile_bullets(persona_name, profile)


def _header(
    persona_name: str,
    counterparty_names: list[str] | None,
    minimal_system: str | None,
) -> str:
    if minimal_system:
        return minimal_system.rstrip()
    others = ", ".join(counterparty_names) if counterparty_names else "someone"
    return f"You are {persona_name} chatting with {others}, continue the conversation."


def _style_bullets(persona_name: str, *, fillers: tuple[str, ...]) -> str:
    """Hardcoded Discord-DM style guidance used by ``style`` mode."""
    filler_str = ", ".join(f'"{f}"' for f in fillers)
    return (
        "\n\n"
        "Match a casual Discord DM style:\n"
        "- Keep replies short: usually one sentence or a few words. "
        "When you do reply longer, burst several short lines separated "
        'by "\\n" rather than one paragraph.\n'
        "- Start messages lowercase. Minimal punctuation; typos and "
        "missing apostrophes are fine.\n"
        f"- Use informal fillers when they fit naturally ({filler_str}); "
        "do not force them.\n"
        "- Natural emoji usage when the tone calls for it, including "
        "custom :shortcode: emotes.\n"
        '- No disclaimers, no "As an AI", no bullet lists or headings in '
        "the reply itself. Just the message body.\n"
        f"- Reply only with {persona_name}'s next message. Do not "
        "narrate actions, repeat earlier turns, or include multiple "
        "speakers."
    )


def _profile_bullets(persona_name: str, profile: Mapping[str, Any]) -> str:
    """Per-persona bullets built from a fingerprint profile entry.

    Every field is accessed defensively: an older or partial
    ``profiles.json`` (or a hand-rolled one in a test) should still
    produce a usable prompt.
    """
    length = profile.get("length") or {}
    mean_words = _as_float(length.get("mean_words"))
    p50 = _as_int(length.get("p50"))
    p95 = _as_int(length.get("p95"))

    lowercase_rate = _as_float(profile.get("lowercase_start_rate"))

    emoji = profile.get("emoji") or {}
    unicode_per_turn = _as_float(emoji.get("unicode_per_turn"))
    custom_per_turn = _as_float(emoji.get("custom_per_turn"))

    burst_rate = _as_float(profile.get("burst_rate"))

    fillers = _extract_top_fillers(profile, top_n=8)
    filler_str = (
        ", ".join(f'"{t}"' for t in fillers)
        if fillers
        else ", ".join(f'"{f}"' for f in _STYLE_FALLBACK_FILLERS)
    )

    lines: list[str] = ["\n", f"Match {persona_name}'s documented writing style:"]

    if mean_words is not None or p50 is not None or p95 is not None:
        length_anchor = p50 if p50 is not None else mean_words
        length_tag = _length_semantic(length_anchor)
        parts: list[str] = []
        if mean_words is not None:
            parts.append(f"~{mean_words:.0f} words on average")
        if p50 is not None:
            parts.append(f"median {p50}")
        if p95 is not None:
            parts.append(f"95th percentile {p95}")
        lines.append(
            "- Turn length: "
            + length_tag
            + ", "
            + ", ".join(parts)
            + ". Keep replies at that scale; err short."
        )

    if lowercase_rate is not None:
        lowercase_tag = _lowercase_semantic(lowercase_rate)
        lines.append(
            f"- Starts messages lowercase {lowercase_tag} (~{lowercase_rate:.0%} of the time). "
            "Capitalise only when the real person would."
        )

    lines.append(
        f"- Typical short fillers: {filler_str}. Use when natural; do not force them."
    )

    if unicode_per_turn is not None or custom_per_turn is not None:
        emoji_tag = _emoji_semantic(unicode_per_turn, custom_per_turn)
        emoji_parts: list[str] = []
        if unicode_per_turn is not None:
            emoji_parts.append(f"{unicode_per_turn:.2f} unicode emoji/turn")
        if custom_per_turn is not None:
            emoji_parts.append(f"{custom_per_turn:.2f} :custom: emotes/turn")
        if emoji_parts:
            lines.append("- Emoji density: " + emoji_tag + " (" + "; ".join(emoji_parts) + ").")

    if burst_rate is not None and burst_rate > 1.05:
        burst_tag = _burst_semantic(burst_rate)
        lines.append(
            f"- Message cadence: {burst_tag} (~{burst_rate:.1f} separate lines per turn); "
            'split bursts with "\\n".'
        )

    lines.extend(
        [
            '- No "As an AI", no disclaimers, no markdown headings, '
            "no bullet lists in the reply itself. Just the message body.",
            f"- Reply only with {persona_name}'s next message. "
            "Do not narrate actions or repeat earlier turns.",
        ]
    )

    return "\n".join(lines)


def _extract_top_fillers(profile: Mapping[str, Any], *, top_n: int) -> list[str]:
    """Pull the top-N 1-gram filler tokens from a profile entry.

    Tolerant of both shapes seen in the codebase:

    * :mod:`discord_sft.analysis.fingerprint` emits
      ``{"top_fillers": {"1gram": [{"token": "...", "score": ..., "count": ...}, ...]}}``
    * Older / test fixtures may use a flat ``"top_fillers_1gram": [["tok", score], ...]``.
    """
    out: list[str] = []
    top = profile.get("top_fillers")
    if isinstance(top, Mapping):
        grams = top.get("1gram") or []
        for item in grams:
            tok = item.get("token") if isinstance(item, Mapping) else None
            if isinstance(tok, str) and tok.strip():
                out.append(tok.strip().lower())
            if len(out) >= top_n:
                return out

    flat = profile.get("top_fillers_1gram")
    if isinstance(flat, list):
        for entry in flat:
            tok: str | None = None
            if isinstance(entry, (list, tuple)) and entry:
                tok = str(entry[0]) if entry[0] is not None else None
            elif isinstance(entry, Mapping):
                raw = entry.get("token")
                tok = str(raw) if raw is not None else None
            if tok:
                out.append(tok.strip().lower())
            if len(out) >= top_n:
                break

    # Dedupe while preserving order.
    seen: set[str] = set()
    dedup: list[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup[:top_n]


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _as_int(v: Any) -> int | None:
    f = _as_float(v)
    if f is None:
        return None
    return int(round(f))


def _length_semantic(anchor_words: float | int | None) -> str:
    """Map representative turn length to a plain-language label."""
    if anchor_words is None:
        return "short"
    words = float(anchor_words)
    if words <= 4:
        return "ultra-brief"
    if words <= 8:
        return "very short"
    if words <= 14:
        return "short"
    if words <= 24:
        return "medium"
    return "long-form"


def _lowercase_semantic(rate: float) -> str:
    """Map lowercase-start rate to a natural-language frequency."""
    if rate >= 0.85:
        return "almost always"
    if rate >= 0.65:
        return "usually"
    if rate >= 0.40:
        return "about half the time"
    if rate >= 0.20:
        return "occasionally"
    return "rarely"


def _emoji_semantic(unicode_per_turn: float | None, custom_per_turn: float | None) -> str:
    """Map emoji/emote totals to a qualitative density bucket."""
    total = (unicode_per_turn or 0.0) + (custom_per_turn or 0.0)
    if total >= 0.80:
        return "high"
    if total >= 0.35:
        return "moderate"
    if total >= 0.12:
        return "light"
    if total > 0:
        return "rare"
    return "minimal"


def _burst_semantic(rate: float) -> str:
    """Map line-burst rate to a cadence descriptor."""
    if rate >= 2.40:
        return "frequent multi-line bursts"
    if rate >= 1.70:
        return "regular multi-line bursts"
    if rate >= 1.20:
        return "occasional multi-line bursts"
    return "mostly single-line replies"
