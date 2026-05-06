"""Lightweight stylometric features for cross-run style diagnostics."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from discord_sft.analysis.heuristics import _DEFAULT_FILLERS
from discord_sft.analysis.fingerprint import _WORD_RE

_SENT_SPLIT = re.compile(r"[\n.!?]+")


def union_mined_fillers(
    profile_doc: Mapping[str, Any],
    persona_ids: set[str],
    *,
    top_n_per_persona: int = 12,
    cap: int = 40,
) -> list[str]:
    """Union top mined filler tokens from profiles for the given personas."""
    personas = profile_doc.get("personas") or {}
    seen: set[str] = set()
    out: list[str] = []
    for pid in persona_ids:
        prof = personas.get(pid)
        if not isinstance(prof, Mapping):
            continue
        unigrams = (prof.get("top_fillers") or {}).get("1gram") or []
        bigrams = (prof.get("top_fillers") or {}).get("2gram") or []
        for item in (unigrams + bigrams)[:top_n_per_persona]:
            if isinstance(item, dict) and item.get("token"):
                t = str(item["token"]).lower().strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
                    if len(out) >= cap:
                        return out
    return out


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def compute_fingerprint_features(
    text: str,
    *,
    fillers: Sequence[str] | None = None,
) -> dict[str, float]:
    """Scalar stylometric features (non-negative) for one aggregated corpus."""
    t = text.strip()
    if not t:
        return {
            "avg_sentence_len": 0.0,
            "avg_word_len": 0.0,
            "filler_rate": 0.0,
            "punct_comma": 0.0,
            "punct_period": 0.0,
            "punct_excl": 0.0,
            "punct_q": 0.0,
            "word_repeat_rate": 0.0,
            "dup_bigram_rate": 0.0,
        }
    sents = _sentences(t)
    sent_word_counts = [len(_words(s)) for s in sents]
    avg_sentence_len = (
        sum(sent_word_counts) / len(sent_word_counts) if sent_word_counts else 0.0
    )
    words = _words(t)
    n_words = len(words)
    avg_word_len = sum(len(w) for w in words) / n_words if n_words else 0.0
    picks = tuple(fillers) if fillers else _DEFAULT_FILLERS
    chunks = [c.strip() for c in t.split("\n\n") if c.strip()] or ([t.strip()] if t.strip() else [])
    filler_rate = (
        sum(any(f in c.lower() for f in picks) for c in chunks) / len(chunks)
        if picks and chunks
        else 0.0
    )
    n = max(len(t), 1)
    word_repeat_rate = 0.0
    if n_words > 1:
        uniq = len(set(words))
        word_repeat_rate = 1.0 - (uniq / n_words)
    bigrams = (
        [f"{words[i]} {words[i + 1]}" for i in range(n_words - 1)] if n_words > 1 else []
    )
    dup_bigram_rate = 0.0
    if bigrams:
        c = Counter(bigrams)
        dup_bigram_rate = sum(1 for _, v in c.items() if v > 1) / len(c)

    return {
        "avg_sentence_len": float(avg_sentence_len),
        "avg_word_len": float(avg_word_len),
        "filler_rate": float(min(filler_rate, 1.0)),
        "punct_comma": t.count(",") / n,
        "punct_period": t.count(".") / n,
        "punct_excl": t.count("!") / n,
        "punct_q": t.count("?") / n,
        "word_repeat_rate": float(word_repeat_rate),
        "dup_bigram_rate": float(dup_bigram_rate),
    }


_FEATURE_ORDER: tuple[str, ...] = (
    "avg_sentence_len",
    "avg_word_len",
    "filler_rate",
    "punct_comma",
    "punct_period",
    "punct_excl",
    "punct_q",
    "word_repeat_rate",
    "dup_bigram_rate",
)


def features_to_vector(features: Mapping[str, float]) -> list[float]:
    return [float(features.get(k, 0.0)) for k in _FEATURE_ORDER]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
