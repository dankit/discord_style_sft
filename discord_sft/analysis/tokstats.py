"""Tokenizer-aware corpus statistics.

Designed for the README prompt "calculate with tokenizer how many tokens the
total training set is before starting" and "compare qwen 3.5 vs 3.6 tokenizers".

The core scoring loop accepts any callable ``encode(str) -> list[int]`` so it's
trivially testable with a fake tokenizer and only pulls in ``transformers`` when
a real HF model id is passed in.
"""
from __future__ import annotations

import json
import statistics
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from discord_sft.data_prep.sft import Sample


Encoder = Callable[[str], Sequence[int]]


class TokenizerLike(Protocol):
    """Minimal surface we need from an HF-style tokenizer (for testing)."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> Sequence[int]: ...

    def tokenize(self, text: str) -> list[str]: ...

    def get_vocab(self) -> Mapping[str, int]: ...


def render_sample_text(sample: Sample) -> str:
    """Canonical text used for token counting -- tokenizer-template-agnostic."""
    parts: list[str] = []
    if sample.system:
        parts.append(sample.system)
    for turn in sample.conversations:
        parts.append(turn.get("value", ""))
    return "\n".join(parts)


def _percentile(xs: Sequence[int], q: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def tokenize_counts(
    samples: Sequence[Sample],
    encode: Encoder,
) -> dict:
    counts: list[int] = []
    per_persona: dict[str, int] = {}
    per_persona_samples: dict[str, int] = {}
    for s in samples:
        n = len(encode(render_sample_text(s)))
        counts.append(n)
        pid = s.meta.get("persona_id", "unknown")
        per_persona[pid] = per_persona.get(pid, 0) + n
        per_persona_samples[pid] = per_persona_samples.get(pid, 0) + 1

    return {
        "n_samples": len(counts),
        "total_tokens": sum(counts),
        "mean_tokens_per_sample": (statistics.fmean(counts) if counts else 0.0),
        "p50_tokens": _percentile(counts, 0.50),
        "p95_tokens": _percentile(counts, 0.95),
        "p99_tokens": _percentile(counts, 0.99),
        "max_tokens": max(counts) if counts else 0,
        "per_persona_tokens": per_persona,
        "per_persona_samples": per_persona_samples,
    }


def load_hf_tokenizer(model_id: str) -> TokenizerLike:
    """Return the raw HF tokenizer (provides encode + tokenize + get_vocab)."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Install optional dependency: pip install 'discord-sft[tokenizers]'"
        ) from e
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)


def encoder_from_tokenizer(tok: TokenizerLike) -> Encoder:
    def _encode(text: str) -> list[int]:
        return list(tok.encode(text, add_special_tokens=False))

    return _encode


def load_hf_encoder(model_id: str) -> Encoder:
    """Return an ``encode`` callable backed by a Hugging Face tokenizer.

    Requires the optional ``tokenizers`` extra (``transformers``).
    """
    return encoder_from_tokenizer(load_hf_tokenizer(model_id))


# ---------- vocab comparison ----------


def _normalize_piece(piece: str) -> str:
    """Strip BPE/SentencePiece space markers so pieces are human-readable."""
    return piece.lstrip("\u2581").lstrip("\u0120")  # SentencePiece and GPT-2 markers.


def compare_vocabs(vocabs: Mapping[str, Iterable[str]]) -> dict:
    """Pairwise vocabulary overlap stats.

    ``vocabs`` maps tokenizer name -> iterable of token strings (typically
    ``tokenizer.get_vocab().keys()``).
    """
    names = list(vocabs.keys())
    sets: dict[str, set[str]] = {n: set(vocabs[n]) for n in names}
    sizes = {n: len(v) for n, v in sets.items()}
    pairs: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            A, B = sets[a], sets[b]
            inter = A & B
            union = A | B
            pairs[f"{a}__vs__{b}"] = {
                "a": a,
                "b": b,
                "size_a": len(A),
                "size_b": len(B),
                "intersection": len(inter),
                "union": len(union),
                "only_in_a": len(A - B),
                "only_in_b": len(B - A),
                "jaccard": (len(inter) / len(union)) if union else 0.0,
            }
    return {"sizes": sizes, "pairs": pairs}


def _readable(piece: str) -> bool:
    """Heuristic for surfacing interesting tokens in unique-token samples."""
    stripped = _normalize_piece(piece)
    if not stripped or len(stripped) > 24 or len(stripped) < 2:
        return False
    if not stripped.isprintable():
        return False
    return all(ch.isalnum() or ch in "-_'" for ch in stripped)


def sample_unique_tokens(
    vocabs: Mapping[str, Iterable[str]],
    *,
    a: str,
    b: str,
    limit: int = 50,
    readable_only: bool = True,
) -> list[str]:
    """Sample tokens that exist in ``a``'s vocab but not ``b``'s."""
    A = set(vocabs[a])
    B = set(vocabs[b])
    diff = A - B
    if readable_only:
        diff = {t for t in diff if _readable(t)}
    # Prefer short pronounceable-looking tokens first; ties broken alphabetically.
    ordered = sorted(diff, key=lambda t: (len(_normalize_piece(t)), t.lower()))
    return ordered[:limit]


def probe_tokenization(
    tokenizers: Mapping[str, TokenizerLike],
    probes: Sequence[str],
) -> list[dict]:
    """For each probe string, show how each tokenizer splits it.

    Returns a list of rows with columns ``probe`` plus per-tokenizer
    ``{name}_pieces`` (list[str], normalized) and ``{name}_count`` (int).
    Useful for eyeballing whether slang like ``imma`` / ``lowkey`` / ``:skull:``
    is a single token or gets shredded into many pieces.
    """
    rows: list[dict] = []
    for probe in probes:
        row: dict[str, Any] = {"probe": probe}
        for name, tok in tokenizers.items():
            pieces = list(tok.tokenize(probe))
            row[f"{name}_pieces"] = [_normalize_piece(p) for p in pieces]
            row[f"{name}_count"] = len(pieces)
        rows.append(row)
    return rows


def compare_tokenizers(
    samples: Sequence[Sample],
    encoders: dict[str, Encoder],
) -> dict:
    """Run ``tokenize_counts`` against each named encoder; report totals + deltas."""
    reports = {name: tokenize_counts(samples, enc) for name, enc in encoders.items()}
    result: dict = {"tokenizers": reports}
    names = list(reports.keys())
    if len(names) >= 2:
        base = names[0]
        base_total = reports[base]["total_tokens"] or 1
        deltas = {}
        for other in names[1:]:
            other_total = reports[other]["total_tokens"]
            deltas[other] = {
                "total_delta": other_total - reports[base]["total_tokens"],
                "pct_change_vs_base": (other_total - reports[base]["total_tokens"])
                / base_total,
            }
        result["vs_" + base] = deltas
    return result


# ---------- tokenizer.json config comparison ----------
#
# ``compare_vocabs`` above only looks at the *set* of tokens in each tokenizer.
# That's fine for "how much overlap is there?" but it cannot answer questions
# like "do both tokenizers map token 'hi' to the same id?" or "are the BPE
# merges applied in the same order?". Those questions require looking at the
# raw ``tokenizer.json`` structure that the Hugging Face ``tokenizers`` library
# serializes; everything below operates on that dict.


def tokenizer_config(tok: TokenizerLike) -> dict:
    """Return the ``tokenizer.json`` config dict for a fast HF tokenizer.

    This is the same document that ships on disk as ``tokenizer.json``. For
    fast tokenizers we can get it without touching the filesystem by calling
    ``backend_tokenizer.to_str()`` (a Rust-side serialize).
    """
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is None or not hasattr(backend, "to_str"):
        raise ValueError(
            "tokenizer_config requires a 'fast' tokenizer (use_fast=True); "
            "slow/Python tokenizers don't expose tokenizer.json directly."
        )
    return json.loads(backend.to_str())


def _ordered_vocab(model: dict) -> list[str]:
    """Return tokens sorted by id, for comparing vocab *ordering*."""
    vocab = model.get("vocab") or {}
    if isinstance(vocab, dict):
        return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    # Unigram models store vocab as a list of [piece, score] pairs.
    if isinstance(vocab, list):
        return [row[0] for row in vocab if row]
    return []


def _vocab_as_mapping(model: dict) -> dict[str, int]:
    """Normalize a tokenizer.json model's vocab into a ``token -> id`` dict."""
    vocab = model.get("vocab") or {}
    if isinstance(vocab, dict):
        return dict(vocab)
    if isinstance(vocab, list):
        return {row[0]: i for i, row in enumerate(vocab) if row}
    return {}


def _added_tokens_by_content(cfg: dict) -> dict[str, dict]:
    return {t.get("content"): t for t in (cfg.get("added_tokens") or []) if t.get("content") is not None}


def diff_tokenizer_configs(
    cfg_a: dict,
    cfg_b: dict,
    *,
    name_a: str = "a",
    name_b: str = "b",
    sample_limit: int = 20,
) -> dict:
    """Deep structural diff of two ``tokenizer.json`` dicts.

    Reports whether the vocabs are identical (same token->id mapping), whether
    their id-ordering matches, whether BPE merges match, whether added/special
    tokens match, and whether each component block (normalizer, pre_tokenizer,
    post_processor, decoder, model type) is identical.
    """
    model_a = cfg_a.get("model") or {}
    model_b = cfg_b.get("model") or {}

    vocab_a = _vocab_as_mapping(model_a)
    vocab_b = _vocab_as_mapping(model_b)
    keys_a, keys_b = set(vocab_a), set(vocab_b)
    shared = keys_a & keys_b
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    id_mismatches = [
        {"token": t, f"id_{name_a}": vocab_a[t], f"id_{name_b}": vocab_b[t]}
        for t in shared
        if vocab_a[t] != vocab_b[t]
    ]
    vocab_identical = not only_a and not only_b and not id_mismatches

    ord_a = _ordered_vocab(model_a)
    ord_b = _ordered_vocab(model_b)
    ordering_identical = ord_a == ord_b
    first_ordering_diff: dict | None = None
    if not ordering_identical:
        n = min(len(ord_a), len(ord_b))
        for i in range(n):
            if ord_a[i] != ord_b[i]:
                first_ordering_diff = {"id": i, name_a: ord_a[i], name_b: ord_b[i]}
                break
        if first_ordering_diff is None:
            first_ordering_diff = {
                "id": n,
                "note": "shorter vocab is a prefix of the longer one",
            }

    merges_a = model_a.get("merges") or []
    merges_b = model_b.get("merges") or []
    merges_identical = merges_a == merges_b
    first_merge_diff: dict | None = None
    if not merges_identical and merges_a and merges_b:
        n = min(len(merges_a), len(merges_b))
        for i in range(n):
            if merges_a[i] != merges_b[i]:
                first_merge_diff = {
                    "rank": i,
                    name_a: merges_a[i],
                    name_b: merges_b[i],
                }
                break
        if first_merge_diff is None:
            first_merge_diff = {
                "rank": n,
                "note": "shorter merges list is a prefix of the longer one",
            }

    added_a = _added_tokens_by_content(cfg_a)
    added_b = _added_tokens_by_content(cfg_b)
    added_only_a = sorted(set(added_a) - set(added_b))
    added_only_b = sorted(set(added_b) - set(added_a))
    added_mismatches: list[dict] = []
    for content in sorted(set(added_a) & set(added_b)):
        if added_a[content] != added_b[content]:
            added_mismatches.append(
                {"content": content, name_a: added_a[content], name_b: added_b[content]}
            )
    added_identical = (
        not added_only_a and not added_only_b and not added_mismatches
    )

    def _eq(key: str) -> bool:
        return cfg_a.get(key) == cfg_b.get(key)

    return {
        "vocab": {
            "identical": vocab_identical,
            f"size_{name_a}": len(vocab_a),
            f"size_{name_b}": len(vocab_b),
            "shared": len(shared),
            f"only_in_{name_a}": len(only_a),
            f"only_in_{name_b}": len(only_b),
            f"only_in_{name_a}_sample": only_a[:sample_limit],
            f"only_in_{name_b}_sample": only_b[:sample_limit],
            "id_mismatches": len(id_mismatches),
            "id_mismatches_sample": id_mismatches[:sample_limit],
        },
        "ordering": {
            "identical": ordering_identical,
            "first_diff": first_ordering_diff,
        },
        "merges": {
            "identical": merges_identical,
            f"count_{name_a}": len(merges_a),
            f"count_{name_b}": len(merges_b),
            "first_diff": first_merge_diff,
        },
        "added_tokens": {
            "identical": added_identical,
            f"count_{name_a}": len(added_a),
            f"count_{name_b}": len(added_b),
            f"only_in_{name_a}": added_only_a,
            f"only_in_{name_b}": added_only_b,
            "mismatches": added_mismatches,
        },
        "model_type_equal": model_a.get("type") == model_b.get("type"),
        "normalizer_equal": _eq("normalizer"),
        "pre_tokenizer_equal": _eq("pre_tokenizer"),
        "post_processor_equal": _eq("post_processor"),
        "decoder_equal": _eq("decoder"),
        "truncation_equal": _eq("truncation"),
        "padding_equal": _eq("padding"),
    }


def summarize_tokenizer_config(cfg: dict) -> dict:
    """One-line-per-tokenizer summary for the UI table."""
    model = cfg.get("model") or {}
    vocab = _vocab_as_mapping(model)

    def _kind(block: Any) -> str | None:
        if isinstance(block, dict):
            return block.get("type")
        return None

    return {
        "model_type": model.get("type"),
        "vocab_size": len(vocab),
        "merges": len(model.get("merges") or []),
        "added_tokens": len(cfg.get("added_tokens") or []),
        "normalizer": _kind(cfg.get("normalizer")),
        "pre_tokenizer": _kind(cfg.get("pre_tokenizer")),
        "post_processor": _kind(cfg.get("post_processor")),
        "decoder": _kind(cfg.get("decoder")),
    }


def compare_tokenizer_configs(configs: Mapping[str, dict]) -> dict:
    """Pairwise deep comparison of a batch of ``tokenizer.json`` configs."""
    per_tokenizer = {name: summarize_tokenizer_config(cfg) for name, cfg in configs.items()}
    names = list(configs.keys())
    pairs: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}__vs__{b}"] = diff_tokenizer_configs(
                configs[a], configs[b], name_a=a, name_b=b
            )
    return {"per_tokenizer": per_tokenizer, "pairs": pairs}
