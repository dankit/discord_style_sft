from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

# Integer weights per axis; ``overall`` is a rounded weighted mean in 1..5.
JUDGE_AXIS_WEIGHTS: dict[str, int] = {
    "vocabulary": 1,
    "tone": 1,
    "length": 1,
    "authentic_persona": 1,
}
JUDGE_AXIS_KEYS: tuple[str, ...] = tuple(JUDGE_AXIS_WEIGHTS.keys())


def _style_judge_prompt(
    *,
    persona_name: str,
    context: str,
    real_message: str,
    generated_message: str,
    system_instructions: str | None = None,
) -> str:
    w_desc = ", ".join(
        f"{k}={JUDGE_AXIS_WEIGHTS[k]:g}" for k in JUDGE_AXIS_KEYS
    )
    sys_block = ""
    si = (system_instructions or "").strip()
    if si:
        sys_block = f"""
## System / instructions the model received
The assistant was given these instructions before generating the reply. Consider whether the generation is compatible with them when judging authenticity and persona fit (style alignment matters more than literal obedience):
{si}
"""
    return f"""# Style and persona judge (rubric-based evaluation)

## Objective (PRIMARY)
You are an expert evaluator. Your PRIMARY objective is to assess how well the **Generated reply** matches the **writing style and voice** of **{persona_name}** as evidenced by the **Reference reply** in the same **Conversation context**. Judge **style, tone, and persona fit**, not factual correctness or whether the answer matches the reference's opinions.

## Evaluation approach
For each sample you receive:
- **Conversation context**: prior turns (who said what).
- **Reference reply**: the real message **{persona_name}** wrote in this situation — your anchor for vocabulary, tone, length, and authenticity.
- **Generated reply**: the model's candidate continuation — compare its **style** to the reference; topical content may differ.
{sys_block}
**CRITICAL**: Penalize robotic or assistant-like habits (over-formality, markdown emphasis theater, lecture tone, disclaimers) when the reference is casual Discord-style. Penalize **mid-sentence truncation** (cut-off thought), empty outputs, or obvious template filler.

## Evaluation order
Evaluate dimensions in this fixed sequence, then assign scores:
1. **vocabulary** → 2. **tone** → 3. **length** → 4. **authentic_persona**

## Scoring dimensions (1–5 integers each)

Use **independent** scores per dimension. Align each numeric score with observable evidence from the Generated reply vs Reference.

### 1. vocabulary
**Evaluates**: Similarity of word choice, slang, contractions, abbreviations, emoji usage, and informal phrasing to the Reference.

- **5**: Strong match — shares fillers/slang patterns and informal lexicon like the Reference (e.g. matching energy of informal shortenings where appropriate).
- **4**: Good match — mostly aligned phrasing; minor mismatches only.
- **3**: Partial — some overlap but noticeable generic or mismatched word choices vs Reference.
- **2**: Weak — formal/generic wording where Reference is casual, or inconsistent slang.
- **1**: Severe mismatch — wrong register entirely, **OR** empty/missing generation.

### 2. tone
**Evaluates**: Energy, attitude, sarcasm vs sincerity, hype vs chill — alignment with Reference **in this context**.

- **5**: Tone fits the scene and matches Reference attitude closely.
- **4**: Very close; small attitude drift.
- **3**: Moderately off (e.g. too stiff or too chaotic vs Reference).
- **2**: Clearly wrong tone for how Reference speaks here.
- **1**: Unreadable tone for the chat **OR** empty generation.

### 3. length
**Evaluates**: Similar **length and density** (short bursts vs paragraphs) vs Reference — not token counts to the word, but same ballpark.

- **5**: Density and verbosity match (e.g. both multi-line bursts or both one-liners as appropriate).
- **4**: Close; slightly longer or shorter but natural.
- **3**: Noticeably mismatched length (e.g. essay vs punchy reply) but acceptable.
- **2**: Strong mismatch (wall of text vs minimal reply or vice versa) without justification from context.
- **1**: Empty **OR** unusably truncated.

### 4. authentic_persona
**Evaluates**: Reads as a believable human in this Discord-style chat **and** plausibly as **{persona_name}**, not a generic chatbot.

- **5**: Human-like and on-persona; no obvious LLM tells unless Reference also has them.
- **4**: Mostly believable; tiny generic gloss.
- **3**: Some AI-ish or off-persona moments (e.g. unexplained formality, heavy markdown emphasis **like this** when Reference doesn't).
- **2**: Clearly assistant-like or wrong persona voice.
- **1**: Bot-like, empty, or nonsensical.

## Edge cases
- **Empty or whitespace-only Generated reply**: scores **1** on all dimensions unless Reference is also empty (then use judgment).
- **Truncated mid-word or mid-clause with no continuation**: cap **length** at **2** or lower and penalize **authentic_persona** appropriately.
- **Language mismatch** (Generated in a different language than Reference/context without cause): cap **vocabulary** and **tone** at **2** or lower.

## Scoring calculation (after your JSON scores)
**overall** is **not** output by you. It is computed as the **integer-rounded weighted mean** of the four dimension scores using weights: {w_desc}.

## Output format
Respond with **JSON only** (no markdown fences, no prose outside the JSON). Structure:

1. **`reasoning`** — FIRST: multi-sentence chain-of-thought. Walk through the evaluation order; cite **observable** contrasts between Generated and Reference **before** settling scores.
2. **`rationale`** — one concise sentence per dimension (no digits inside rationale strings).
3. **Numeric keys** — `vocabulary`, `tone`, `length`, `authentic_persona` as integers 1–5.

Exact shape:
{{
  "reasoning": "<step-by-step evaluation; cite concrete style cues>",
  "rationale": {{
    "vocabulary": "<one sentence, no digits>",
    "tone": "<one sentence, no digits>",
    "length": "<one sentence, no digits>",
    "authentic_persona": "<one sentence, no digits>"
  }},
  "vocabulary": <1-5>,
  "tone": <1-5>,
  "length": <1-5>,
  "authentic_persona": <1-5>
}}

Inputs for this evaluation:

PERSONA: {persona_name}

CONVERSATION CONTEXT:
{context}

REFERENCE reply ({persona_name}):
"{real_message}"

GENERATED reply:
"{generated_message}"
"""


def _first_balanced_json_object(s: str) -> str | None:
    """Return substring from first '{{' through matching '}}', respecting strings."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Parse model output; tolerate optional JSON fences and leading prose."""
    t = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?", t, re.IGNORECASE)
    if fence:
        t = t[fence.end() :]
        t = re.sub(r"\n?```\s*$", "", t, flags=re.IGNORECASE).strip()
    try:
        out = json.loads(t)
        if isinstance(out, dict):
            return out
        raise RuntimeError("Judge JSON root must be an object")
    except json.JSONDecodeError:
        blob = _first_balanced_json_object(t)
        if blob is None:
            raise
        out = json.loads(blob)
        if not isinstance(out, dict):
            raise RuntimeError("Judge JSON root must be an object") from None
        return out


def _axis_score_to_int(v: Any) -> int:
    if not isinstance(v, (int, float)):
        raise RuntimeError(f"Judge score must be a number, got {type(v).__name__}")
    r = int(round(float(v)))
    return max(1, min(5, r))


def _integer_weighted_overall(
    scores: dict[str, int], weights: dict[str, int]
) -> int:
    den = sum(weights[k] for k in JUDGE_AXIS_KEYS)
    if not den:
        return 1
    num = sum(scores[k] * weights[k] for k in JUDGE_AXIS_KEYS)
    rounded = (num + den // 2) // den
    return max(1, min(5, rounded))


def finalize_style_rating(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate axis scores, normalize rationale, drop model ``overall``, add weighted ``overall``.

    Returns a flat dict: four axis ints (1-5), ``overall`` int (1-5), ``rationale`` dict,
    and ``reasoning`` str (possibly empty).
    """
    reasoning = ""
    if "reasoning" in raw and raw["reasoning"] is not None:
        reasoning = str(raw["reasoning"]).strip()

    rationale_in = raw.get("rationale") or {}
    if not isinstance(rationale_in, dict):
        rationale_in = {}
    rationale: dict[str, str] = {
        k: str(rationale_in.get(k, "")).strip() for k in JUDGE_AXIS_KEYS
    }

    scores: dict[str, int] = {}
    for k in JUDGE_AXIS_KEYS:
        if k not in raw:
            raise RuntimeError(f"Judge JSON missing numeric key: {k!r}")
        scores[k] = _axis_score_to_int(raw[k])

    overall = _integer_weighted_overall(scores, JUDGE_AXIS_WEIGHTS)

    out: dict[str, Any] = {
        **scores,
        "overall": overall,
        "rationale": rationale,
        "reasoning": reasoning,
    }
    return out


class StyleJudge(ABC):
    """Optional LLM-as-a-judge backend."""

    @abstractmethod
    def score(
        self,
        *,
        real_message: str,
        generated_message: str,
        context: str,
        persona_name: str,
        system_instructions: str | None = None,
    ) -> dict[str, Any]:
        pass


class OpenRouterJudge(StyleJudge):
    """OpenAI-compatible chat against OpenRouter (needs ``openai``; use ``[evals]`` extra).

    Set ``OPENROUTER_API_KEY``. Optional: ``OPENROUTER_BASE_URL`` (default
    ``https://openrouter.ai/api/v1``), ``OPENROUTER_HTTP_REFERER``,
    ``OPENROUTER_APP_TITLE`` (sent as ``X-Title``; default ``discord-sft``).
    """

    def __init__(self, model: str = "anthropic/claude-sonnet-4.6") -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install optional dependency: pip install 'discord-sft[evals]' "
                "(includes openai for OpenRouter)."
            ) from e
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY to use OpenRouterJudge.")
        base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        headers: dict[str, str] = {}
        if referer := os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        headers["X-Title"] = os.environ.get("OPENROUTER_APP_TITLE", "discord-sft")
        self._model = model
        self._client = OpenAI(
            base_url=base_url if base_url.endswith("/v1") else f"{base_url}/v1",
            api_key=key,
            default_headers=headers,
        )

    def score(
        self,
        *,
        real_message: str,
        generated_message: str,
        context: str,
        persona_name: str,
        system_instructions: str | None = None,
    ) -> dict[str, Any]:
        prompt = _style_judge_prompt(
            persona_name=persona_name,
            context=context,
            real_message=real_message,
            generated_message=generated_message,
            system_instructions=system_instructions,
        )
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned empty message content.")
        parsed = _parse_judge_json(content)
        return finalize_style_rating(parsed)


def make_judge(backend: str = "openrouter") -> StyleJudge:
    if backend == "openrouter":
        return OpenRouterJudge()
    raise ValueError(f"Unknown judge backend: {backend}")
