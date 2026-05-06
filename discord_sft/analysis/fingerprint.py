"""Per-persona style fingerprinting.

Reads a ShareGPT-style JSONL (what ``build-sft`` emits) and produces a
``profiles.json`` with, per persona:

* length distribution (mean, p50, p95)
* burst-structure proxy (mean assistant turn lines; one line per Discord message
  before burst-merge is captured in the ``\\n``-joined text)
* emoji / custom-emoji rate
* lowercase-start rate
* top-K n-gram fillers relative to the global corpus, mined with a log-odds
  Dirichlet prior (Monroe, Colaresi & Quinn 2008). This yields real style
  markers ("imma", "lowkey", "fr", "g") rather than a fixed hand-picked list.

These profiles are used by the ``eval-heuristics --profile ... --persona ...``
flow as the drift reference, replacing the fixed filler list in
:func:`discord_sft.analysis.heuristics.style_heuristics`.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from discord_sft.data_prep.sft import Sample


_EMOJI_UNICODE_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F600-\U0001F64F"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000027BF"
    "]"
)
_CUSTOM_EMOJI_SHORT_RE = re.compile(r":[A-Za-z0-9_]+:")
_WORD_RE = re.compile(r"[A-Za-z']+")


def _assistant_texts_by_persona(samples: Iterable[Sample]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s in samples:
        pid = s.meta.get("persona_id", "unknown")
        final = next(
            (t["value"] for t in reversed(s.conversations) if t["from"] == "assistant"),
            None,
        )
        if final is None:
            continue
        out.setdefault(pid, []).append(final)
    return out


def _length_stats(texts: Sequence[str]) -> dict:
    wcs = [len(t.split()) for t in texts if t]
    if not wcs:
        return {"n": 0, "mean_words": 0.0, "p50": 0, "p95": 0}
    return {
        "n": len(wcs),
        "mean_words": statistics.fmean(wcs),
        "p50": int(statistics.median(wcs)),
        "p95": int(sorted(wcs)[max(0, int(len(wcs) * 0.95) - 1)]),
    }


def _lowercase_start_rate(texts: Sequence[str]) -> float:
    starts = [t.lstrip() for t in texts if t and t.strip()]
    if not starts:
        return 0.0
    return sum(1 for t in starts if t and t[0].isalpha() and t[0].islower()) / len(starts)


def _emoji_rate(texts: Sequence[str]) -> dict:
    if not texts:
        return {"unicode_per_turn": 0.0, "custom_per_turn": 0.0}
    uni = sum(len(_EMOJI_UNICODE_RE.findall(t)) for t in texts)
    custom = sum(len(_CUSTOM_EMOJI_SHORT_RE.findall(t)) for t in texts)
    return {
        "unicode_per_turn": uni / len(texts),
        "custom_per_turn": custom / len(texts),
    }


def _burst_rate(texts: Sequence[str]) -> float:
    """Avg newline-joined segments per turn; proxy for Discord burst-size."""
    if not texts:
        return 0.0
    return sum(t.count("\n") + 1 for t in texts if t) / len(texts)


def _ngram_counts(text: str, n: int) -> Counter:
    toks = [w.lower() for w in _WORD_RE.findall(text)]
    if len(toks) < n:
        return Counter()
    return Counter(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _log_odds(
    persona_counts: Counter,
    global_counts: Counter,
    alpha: float = 0.01,
    top_k: int = 25,
    min_count: int = 3,
) -> list[tuple[str, float, int]]:
    """Informative-Dirichlet log-odds (Monroe et al. 2008).

    Returns top-K n-grams scored by log-odds vs corpus, filtered by min_count.
    """
    if not persona_counts:
        return []
    n_p = sum(persona_counts.values())
    n_g = sum(global_counts.values())
    if n_p == 0 or n_g == 0:
        return []
    vocab = set(persona_counts) | set(global_counts)
    a0 = alpha * len(vocab)
    scored: list[tuple[str, float, int]] = []
    for w in persona_counts:
        if persona_counts[w] < min_count:
            continue
        yp = persona_counts[w]
        yg = global_counts.get(w, 0)
        # Non-persona corpus counts
        yn = yg - yp
        nn = n_g - n_p
        if yn < 0:
            yn = 0
        num = (yp + alpha) / (n_p + a0 - yp - alpha)
        den = (yn + alpha) / (nn + a0 - yn - alpha)
        if den <= 0 or num <= 0:
            continue
        log_odds = math.log(num) - math.log(den)
        var = 1.0 / (yp + alpha) + 1.0 / max(yn + alpha, 1e-9)
        z = log_odds / math.sqrt(var)
        scored.append((w, z, yp))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def build_profiles(
    samples: Sequence[Sample],
    *,
    top_k_ngrams: int = 25,
    ngram_sizes: tuple[int, ...] = (1, 2, 3),
) -> dict:
    by_persona = _assistant_texts_by_persona(samples)
    global_counts: dict[int, Counter] = {n: Counter() for n in ngram_sizes}
    persona_counts: dict[str, dict[int, Counter]] = {}
    for pid, texts in by_persona.items():
        pc = {n: Counter() for n in ngram_sizes}
        for t in texts:
            for n in ngram_sizes:
                c = _ngram_counts(t, n)
                pc[n].update(c)
                global_counts[n].update(c)
        persona_counts[pid] = pc

    profiles: dict[str, dict] = {}
    for pid, texts in by_persona.items():
        fillers = {
            f"{n}gram": [
                {"token": w, "z": round(z, 3), "count": c}
                for w, z, c in _log_odds(
                    persona_counts[pid][n],
                    global_counts[n],
                    top_k=top_k_ngrams,
                )
            ]
            for n in ngram_sizes
        }
        profiles[pid] = {
            "persona_id": pid,
            "length": _length_stats(texts),
            "lowercase_start_rate": _lowercase_start_rate(texts),
            "emoji": _emoji_rate(texts),
            "burst_size": _burst_rate(texts),
            "top_fillers": fillers,
        }
    return {"personas": profiles}


def score_against_profile(
    generated: Sequence[str],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare aggregated generated messages to corpus fingerprint scalars.

    Mirrors the stats computed during :func:`build_profiles` (length /
    lowercase starts / bursts / emoji), then reports absolute deltas plus
    side-by-side values for dashboards.
    """
    stripped = [t.strip() for t in generated if isinstance(t, str) and t.strip()]
    if not stripped:
        return {"error": "empty generated"}

    length = _length_stats(stripped)
    lc = _lowercase_start_rate(stripped)
    emj = _emoji_rate(stripped)
    burst = _burst_rate(stripped)

    plen = profile.get("length") or {}
    ref_len = float(plen.get("mean_words", 0.0))
    ref_lc = float(profile.get("lowercase_start_rate", 0.0))
    ref_burst = float(profile.get("burst_size", 0.0))
    ref_emoji = profile.get("emoji") or {}
    ref_uni = float(ref_emoji.get("unicode_per_turn", 0.0))
    ref_cust = float(ref_emoji.get("custom_per_turn", 0.0))

    gen_mean = float(length["mean_words"])
    gen_uni = float(emj["unicode_per_turn"])
    gen_cust = float(emj["custom_per_turn"])

    return {
        "n_generated": int(length["n"]),
        "mean_words_delta": abs(gen_mean - ref_len),
        "lowercase_start_delta": abs(float(lc) - ref_lc),
        "burst_size_delta": abs(float(burst) - ref_burst),
        "emoji_unicode_per_turn_delta": abs(gen_uni - ref_uni),
        "emoji_custom_per_turn_delta": abs(gen_cust - ref_cust),
        "profile_mean_words": ref_len,
        "generated_mean_words": gen_mean,
        "profile_lowercase_start_rate": ref_lc,
        "generated_lowercase_start_rate": float(lc),
        "profile_burst_size": ref_burst,
        "generated_burst_size": float(burst),
        "profile_emoji_unicode_per_turn": ref_uni,
        "generated_emoji_unicode_per_turn": gen_uni,
        "profile_emoji_custom_per_turn": ref_cust,
        "generated_emoji_custom_per_turn": gen_cust,
    }


def aggregate_profile_drift_for_eval_rows(
    rows: Iterable[Mapping[str, Any]],
    profiles_document: Mapping[str, Any],
    *,
    text_key: str = "generated",
    persona_id_key: str = "persona_id",
    persona_name_key: str = "persona_name",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Group persona eval rows by persona id and score aggregates vs fingerprints.

    ``profiles_document`` is the outer JSON document (must contain a ``personas``
    mapping from id to fingerprint dict from :func:`build_profiles`).
    """
    warnings: list[str] = []
    personas = profiles_document.get("personas")
    if not isinstance(personas, Mapping) or not personas:
        return [], ["profiles document has no ``personas`` entries"]

    by_texts: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, str] = {}
    for row in rows:
        pid = str(row.get(persona_id_key) or "").strip()
        if not pid:
            continue
        raw = row.get(text_key)
        if isinstance(raw, str) and raw.strip():
            by_texts[pid].append(raw)
        pname = str(row.get(persona_name_key) or "").strip()
        if pname and pid not in by_name:
            by_name[pid] = pname

    table: list[dict[str, Any]] = []
    for pid in sorted(by_texts.keys()):
        texts = by_texts[pid]
        prof = personas.get(pid)
        if prof is None or not isinstance(prof, Mapping):
            warnings.append(f"No fingerprint profile for persona_id={pid!r} (skipped).")
            continue
        if not texts:
            warnings.append(f"Persona {pid!r}: no non-empty generations (skipped).")
            continue

        scored = score_against_profile(texts, prof)
        if "error" in scored:
            warnings.append(f"Persona {pid!r}: {scored.get('error')}")
            continue

        pname = by_name.get(pid, "")
        table.append({"persona_id": pid, "persona_name": pname, **scored})

    return table, warnings
