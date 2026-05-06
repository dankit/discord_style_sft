from __future__ import annotations

from typing import Any, Mapping, Sequence


_DEFAULT_FILLERS: tuple[str, ...] = ("lol", "bro", "ngl", "tbh", "fr", "lowkey")


def style_heuristics(
    generated_msgs: list[str],
    reference_msgs: list[str],
    *,
    fillers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Cheap distributional checks (length, punctuation, filler words).

    Does not measure semantic style match. Pass ``fillers`` to override the
    hand-picked default list with per-persona mined fillers (see
    :mod:`discord_sft.analysis.fingerprint`).
    """
    results: dict[str, Any] = {}

    def _norm(msgs: list[str]) -> list[str]:
        return [m.strip() for m in msgs if isinstance(m, str)]

    def _lens(msgs: list[str]) -> list[int]:
        return [len(m.split()) for m in msgs if m]

    if not generated_msgs or not reference_msgs:
        results["error"] = "empty input"
        return results

    generated_msgs = _norm(generated_msgs)
    reference_msgs = _norm(reference_msgs)
    gen_lens = _lens(generated_msgs)
    ref_lens = _lens(reference_msgs)
    if not gen_lens or not ref_lens:
        results["error"] = "empty or blank input"
        results["avg_length_diff"] = 0.0
        results["exclamation_rate"] = 0.0
        results["caps_rate"] = 0.0
        results["filler_rate"] = 0.0
        picks = tuple(fillers) if fillers else _DEFAULT_FILLERS
        results["fillers_used"] = list(picks)
        return results
    results["avg_length_diff"] = abs(
        sum(gen_lens) / len(gen_lens) - sum(ref_lens) / len(ref_lens)
    )
    results["exclamation_rate"] = sum("!" in m for m in generated_msgs) / len(generated_msgs)
    caps_den = sum(1 for m in generated_msgs if m)
    results["caps_rate"] = (
        sum(m[0].isupper() for m in generated_msgs if m) / caps_den if caps_den else 0.0
    )
    picks = tuple(fillers) if fillers else _DEFAULT_FILLERS
    results["filler_rate"] = sum(
        any(f in m.lower() for f in picks) for m in generated_msgs
    ) / len(generated_msgs)
    results["fillers_used"] = list(picks)
    return results


def profile_heuristics(
    generated_msgs: list[str],
    reference_msgs: list[str],
    *,
    profile: Mapping[str, Any],
    top_n_fillers: int = 15,
) -> dict[str, Any]:
    """Variant that seeds the filler list from a persona profile emitted by
    :func:`discord_sft.analysis.fingerprint.build_profiles`.
    """
    unigrams = profile.get("top_fillers", {}).get("1gram", [])
    bigrams = profile.get("top_fillers", {}).get("2gram", [])
    mined: list[str] = []
    for item in (unigrams + bigrams)[:top_n_fillers]:
        if isinstance(item, dict) and item.get("token"):
            mined.append(str(item["token"]))
    stats = style_heuristics(generated_msgs, reference_msgs, fillers=mined or None)
    stats["profile_persona"] = profile.get("persona_id")
    return stats
