"""Build ShareGPT-style SFT samples from curated sessions.

For each curated session and each emitted persona, at most **one** sample is
built: a window ending on that persona's **last** raw turn in the session.
That turn becomes the final ``assistant`` message; other turns in the window
become ``user`` / merged roles. Emitting every persona still yields disjoint
streams (different ``persona_id`` / system prompts) without sliding duplicate
prefixes along the same thread.
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from discord_sft.data_prep.curate import Session, Turn


@dataclass
class Sample:
    system: str
    conversations: list[dict]
    meta: dict

    def to_json(self) -> dict:
        return {
            "system": self.system,
            "conversations": self.conversations,
            "meta": self.meta,
        }


def _persona_name_lookup(session: Session) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in session.turns:
        out.setdefault(t.author_id, t.author_name)
    return out


def _build_system(
    persona_name: str,
    counterparty_names: Sequence[str],
) -> str:
    others = ", ".join(counterparty_names) if counterparty_names else "someone"
    return f"You are {persona_name} chatting with {others}, continue the discord conversation."


def _merge_consecutive(conv: list[dict]) -> list[dict]:
    """Ensure roles strictly alternate by merging consecutive same-role turns with \\n."""
    out: list[dict] = []
    for turn in conv:
        if out and out[-1]["from"] == turn["from"]:
            out[-1] = {
                "from": turn["from"],
                "value": f"{out[-1]['value']}\n{turn['value']}",
            }
        else:
            out.append(dict(turn))
    return out


def _window_sample_fixed_start(
    session: Session,
    start_idx: int,
    end_idx: int,
    persona_id: str,
) -> Sample | None:
    window: list[Turn] = list(session.turns[start_idx : end_idx + 1])
    # A valid SFT sample needs at least one `user` turn preceding the assistant turn.
    has_user = any(t.author_id != persona_id for t in window[:-1])
    if not has_user:
        return None

    names = _persona_name_lookup(session)
    persona_name = names.get(persona_id, "assistant")
    counterparty_names = [names[a] for a in names if a != persona_id]

    conversations = [
        {
            "from": "assistant" if t.author_id == persona_id else "user",
            "value": t.text,
        }
        for t in window
    ]
    conversations = _merge_consecutive(conversations)
    # After merging, the sample must still end on an assistant turn. It always
    # will because the final window turn is authored by the persona.
    if conversations[-1]["from"] != "assistant":
        return None
    # Trim any leading assistant turns: a valid SFT sample starts with `user`.
    # (Can happen when the window begins at a persona turn.)
    while conversations and conversations[0]["from"] == "assistant":
        conversations.pop(0)
    if len(conversations) < 2 or conversations[-1]["from"] != "assistant":
        return None

    meta = {
        "persona_id": persona_id,
        "persona_name": persona_name,
        "counterparty_ids": sorted(a for a in names if a != persona_id),
        "session_id": session.id,
        "folder": session.folder,
        "first_ts": window[0].start_ts.isoformat(),
        "last_ts": window[-1].end_ts.isoformat(),
        "num_turns": len(conversations),
        "end_source_ids": list(window[-1].source_ids),
    }
    system = _build_system(persona_name, counterparty_names)
    return Sample(system=system, conversations=conversations, meta=meta)


def _window_sample(
    session: Session,
    end_idx: int,
    persona_id: str,
    *,
    window_turns: int = 16,
    min_sharegpt_turns: int = 2,
    max_sharegpt_turns: int | None = 8,
) -> Sample | None:
    """Pick the valid raw window ending at ``end_idx`` with the largest ShareGPT length.

    Scans every raw ``start_idx`` allowed by ``window_turns``, keeps samples whose
    merged ``num_turns`` lie in ``[min_sharegpt_turns, max_sharegpt_turns]`` (when
    ``max_sharegpt_turns`` is not ``None``), and prefers the longest such window;
    ties break toward smaller ``start_idx`` (more left context). Callers typically
    invoke this only for each persona's **last** raw index in the session.
    """
    start_lo = max(0, end_idx - window_turns + 1)
    if max_sharegpt_turns is None:
        return _window_sample_fixed_start(session, start_lo, end_idx, persona_id)

    best: Sample | None = None
    best_nt = -1
    best_start = end_idx + 1
    for start_idx in range(start_lo, end_idx + 1):
        s = _window_sample_fixed_start(session, start_idx, end_idx, persona_id)
        if s is None:
            continue
        nt = int(s.meta["num_turns"])
        if nt < min_sharegpt_turns or nt > max_sharegpt_turns:
            continue
        if nt > best_nt or (nt == best_nt and start_idx < best_start):
            best = s
            best_nt = nt
            best_start = start_idx
    return best


def _last_raw_index_per_persona(session: Session, persona_ids: set[str]) -> dict[str, int]:
    """Index of each persona's last raw turn in ``session`` (among ``persona_ids``)."""
    last: dict[str, int] = {}
    for idx, t in enumerate(session.turns):
        if t.author_id in persona_ids:
            last[t.author_id] = idx
    return last


def build_samples_for_session(
    session: Session,
    personas: Iterable[str] | None = None,
    window_turns: int = 16,
    *,
    min_sharegpt_turns: int = 2,
    max_sharegpt_turns: int | None = 8,
) -> list[Sample]:
    authors = list(session.authors())
    want = set(personas) if personas else set(authors)
    samples: list[Sample] = []
    last_idx = _last_raw_index_per_persona(session, want)
    for idx, t in enumerate(session.turns):
        if t.author_id not in want:
            continue
        if last_idx.get(t.author_id) != idx:
            continue
        s = _window_sample(
            session,
            idx,
            t.author_id,
            window_turns=window_turns,
            min_sharegpt_turns=min_sharegpt_turns,
            max_sharegpt_turns=max_sharegpt_turns,
        )
        if s is not None:
            samples.append(s)
    return samples


def build_samples(
    sessions: Iterable[Session],
    personas: Iterable[str] | None = None,
    window_turns: int = 16,
    *,
    min_sharegpt_turns: int = 2,
    max_sharegpt_turns: int | None = 8,
) -> list[Sample]:
    """Build SFT samples: at most one row per ``(session_id, persona_id)``.

    The window ends on that persona's **last** raw turn in the session; there
    are no intermediate per-assistant training targets along the same thread.
    """
    out: list[Sample] = []
    for s in sessions:
        out.extend(
            build_samples_for_session(
                s,
                personas=personas,
                window_turns=window_turns,
                min_sharegpt_turns=min_sharegpt_turns,
                max_sharegpt_turns=max_sharegpt_turns,
            )
        )
    return out


# ---------- balancing ----------


@dataclass
class BalanceReport:
    policy: str
    k: float
    cap_per_persona: dict[str, int]
    counts_before: dict[str, int]
    counts_after: dict[str, int]


def _parse_balance_policy(policy: str) -> tuple[str, int | None]:
    if policy.startswith("cap:"):
        return "cap", int(policy.split(":", 1)[1])
    return policy, None


def _session_stratified_sample(items: list[Sample], cap: int, rng: random.Random) -> list[Sample]:
    """Take up to ``cap`` samples, round-robin across ``session_id`` buckets (shuffled)."""
    if cap >= len(items):
        return list(items)
    by_sess: dict[str, list[Sample]] = {}
    for it in items:
        by_sess.setdefault(it.meta["session_id"], []).append(it)
    for bucket in by_sess.values():
        rng.shuffle(bucket)
    session_ids = list(by_sess.keys())
    rng.shuffle(session_ids)
    chosen: list[Sample] = []
    idx = 0
    while len(chosen) < cap and session_ids:
        sid = session_ids[idx % len(session_ids)]
        bucket = by_sess[sid]
        if bucket:
            chosen.append(bucket.pop())
        if not bucket:
            session_ids.remove(sid)
            if not session_ids:
                break
            continue
        idx += 1
    return chosen


def balance_samples(
    samples: Sequence[Sample],
    *,
    policy: str = "median",
    k: float = 1.5,
    seed: int = 0,
) -> tuple[list[Sample], BalanceReport]:
    """Cap per-persona sample counts, stratified by session_id."""
    counts_before: dict[str, int] = {}
    by_persona: dict[str, list[Sample]] = {}
    for s in samples:
        pid = s.meta["persona_id"]
        counts_before[pid] = counts_before.get(pid, 0) + 1
        by_persona.setdefault(pid, []).append(s)

    mode, explicit_cap = _parse_balance_policy(policy)
    caps: dict[str, int] = {}
    if mode == "none":
        caps = {p: c for p, c in counts_before.items()}
    elif mode == "min":
        target = min(counts_before.values()) if counts_before else 0
        caps = {p: target for p in counts_before}
    elif mode == "cap":
        target = explicit_cap or 0
        caps = {p: target for p in counts_before}
    elif mode == "median":
        if counts_before:
            med = statistics.median(counts_before.values())
            target = int(round(med * k))
        else:
            target = 0
        caps = {p: target for p in counts_before}
    else:
        raise ValueError(f"Unknown balance policy: {policy}")

    rng = random.Random(seed)
    kept: list[Sample] = []
    counts_after: dict[str, int] = {}
    for pid, items in by_persona.items():
        cap = caps.get(pid, len(items))
        if cap >= len(items) or mode == "none":
            kept.extend(items)
            counts_after[pid] = len(items)
            continue
        chosen = _session_stratified_sample(items, cap, rng)
        kept.extend(chosen)
        counts_after[pid] = len(chosen)

    report = BalanceReport(
        policy=policy,
        k=k,
        cap_per_persona=caps,
        counts_before=counts_before,
        counts_after=counts_after,
    )
    return kept, report


TURN_LENGTH_BINS: tuple[int, ...] = (2, 4, 6, 8)


def _sample_num_turns(s: Sample) -> int | None:
    nt = s.meta.get("num_turns")
    return int(nt) if isinstance(nt, int) else None


def _counts_by_num_turns_str(samples: Sequence[Sample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in samples:
        nt = _sample_num_turns(s)
        if nt is None:
            key = "other"
        else:
            key = str(nt)
        out[key] = out.get(key, 0) + 1
    return out


def parse_turn_mix_weights(policy: str) -> dict[int, float]:
    """Parse ``turn-mix``: ``none`` / empty, ``uniform``, or JSON weights for lengths 2,4,6,8."""
    p = policy.strip()
    if not p or p.lower() == "none":
        return {}
    if p.lower() == "uniform":
        return {b: 1.0 for b in TURN_LENGTH_BINS}
    d = json.loads(p)
    if not isinstance(d, dict):
        raise ValueError("turn-mix JSON must be an object mapping lengths to weights")
    out: dict[int, float] = {}
    for k, v in d.items():
        b = int(k)
        out[b] = float(v)
    return out


@dataclass
class TurnLengthBalanceReport:
    policy: str
    counts_before: dict[str, int]
    counts_after: dict[str, int]
    alpha: float | None
    weights_used: dict[str, float]
    #: Filled after ``split_train_val``; TRAIN vs VALIDATION refer to final jsonl splits.
    post_split_num_turns: dict[str, Any] | None = None


def _sort_num_turn_count_items(counts: dict[str, int]) -> list[tuple[str, int]]:
    def sort_key(kv: tuple[str, int]) -> tuple[int, str]:
        k, _ = kv
        if k == "other":
            return (10_000, k)
        try:
            return (int(k), k)
        except ValueError:
            return (9999, k)

    return sorted(counts.items(), key=sort_key)


def post_split_num_turns_breakdown(
    train: Sequence[Sample],
    val: Sequence[Sample],
) -> dict[str, Any]:
    """Counts and within-split fractions of ``meta.num_turns`` for train vs validation rows."""
    train_counts = dict(_sort_num_turn_count_items(_counts_by_num_turns_str(train)))
    val_counts = dict(_sort_num_turn_count_items(_counts_by_num_turns_str(val)))
    n_train = sum(train_counts.values())
    n_val = sum(val_counts.values())

    def _frac(counts: dict[str, int], n: int) -> dict[str, float]:
        if n <= 0:
            return {}
        return {k: round(v / n, 6) for k, v in _sort_num_turn_count_items(counts)}

    return {
        "note": (
            "TRAIN vs VALIDATION: final session-based splits written to train.jsonl / val.jsonl. "
            "counts_before and counts_after above describe the combined sample list "
            "before split_train_val (after persona balance and optional turn-length mix)."
        ),
        "TRAIN_split": {
            "row_count": n_train,
            "counts_by_num_turns": train_counts,
            "fraction_within_TRAIN_rows": _frac(train_counts, n_train),
        },
        "VALIDATION_split": {
            "row_count": n_val,
            "counts_by_num_turns": val_counts,
            "fraction_within_VALIDATION_rows": _frac(val_counts, n_val),
        },
    }


def balance_turn_length(
    samples: Sequence[Sample],
    *,
    policy: str = "none",
    seed: int = 0,
) -> tuple[list[Sample], TurnLengthBalanceReport]:
    """Optionally downsample within ShareGPT length bins (2/4/6/8) toward target weights.

    Samples whose ``num_turns`` is not in any positive-weight bin are kept unchanged.
    Within bins that participate in the mix, proportional targets use
    ``alpha = min_b count_b / w_b`` so no bin is oversampled past its data.
    """
    counts_before = _counts_by_num_turns_str(samples)
    weights = parse_turn_mix_weights(policy)
    if not weights:
        return list(samples), TurnLengthBalanceReport(
            policy=policy,
            counts_before=counts_before,
            counts_after=dict(counts_before),
            alpha=None,
            weights_used={},
        )

    mix_bins = {b for b, w in weights.items() if w > 0.0}
    if not mix_bins:
        return list(samples), TurnLengthBalanceReport(
            policy=policy,
            counts_before=counts_before,
            counts_after=dict(counts_before),
            alpha=None,
            weights_used={},
        )

    wsum = sum(weights[b] for b in mix_bins)
    if wsum <= 0:
        return list(samples), TurnLengthBalanceReport(
            policy=policy,
            counts_before=counts_before,
            counts_after=dict(counts_before),
            alpha=None,
            weights_used={},
        )
    wnorm = {b: weights[b] / wsum for b in mix_bins}

    by_bin: dict[int, list[Sample]] = {b: [] for b in mix_bins}
    remainder: list[Sample] = []
    for s in samples:
        nt = _sample_num_turns(s)
        if nt is not None and nt in mix_bins:
            by_bin[nt].append(s)
        else:
            remainder.append(s)

    counts: dict[int, int] = {b: len(by_bin[b]) for b in mix_bins}
    total = sum(counts.values())
    if total == 0:
        return list(samples), TurnLengthBalanceReport(
            policy=policy,
            counts_before=counts_before,
            counts_after=dict(counts_before),
            alpha=None,
            weights_used={str(b): wnorm[b] for b in sorted(wnorm)},
        )

    positive_bins = [b for b in mix_bins if counts.get(b, 0) > 0 and wnorm[b] > 0.0]
    if not positive_bins:
        return list(samples), TurnLengthBalanceReport(
            policy=policy,
            counts_before=counts_before,
            counts_after=dict(counts_before),
            alpha=None,
            weights_used={str(b): float(wnorm[b]) for b in sorted(wnorm)},
        )
    alpha = min(counts[b] / wnorm[b] for b in positive_bins)
    keep_n: dict[int, int] = {
        b: min(counts.get(b, 0), int(alpha * wnorm[b])) for b in mix_bins
    }

    rng = random.Random(seed)
    kept_bins: list[Sample] = []
    for b in mix_bins:
        cap = keep_n.get(b, 0)
        kept_bins.extend(_session_stratified_sample(by_bin[b], cap, rng))

    out = remainder + kept_bins
    # Without shuffling, ``remainder`` then length-bin blocks mirror build order
    # (bad for jsonl inspection and any consumer that reads sequentially).
    rng.shuffle(out)
    counts_after = _counts_by_num_turns_str(out)
    return out, TurnLengthBalanceReport(
        policy=policy,
        counts_before=counts_before,
        counts_after=counts_after,
        alpha=float(alpha),
        weights_used={str(b): float(wnorm[b]) for b in sorted(wnorm)},
    )


# ---------- train/val split ----------


def split_train_val(
    samples: Sequence[Sample],
    val_frac: float = 0.10,
    seed: int = 0,
) -> tuple[list[Sample], list[Sample]]:
    """Split by session_id (never by sample) so context leakage is impossible."""
    session_ids = sorted({s.meta["session_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(session_ids)
    n_val = max(1, int(len(session_ids) * val_frac)) if session_ids and val_frac > 0 else 0
    val_ids = set(session_ids[:n_val])
    train, val = [], []
    for s in samples:
        (val if s.meta["session_id"] in val_ids else train).append(s)
    return train, val


def shuffle_samples(samples: list[Sample], seed: int) -> None:
    """Shuffle ``samples`` in place with a fixed seed (e.g. before writing jsonl)."""
    random.Random(seed).shuffle(samples)


# ---------- IO ----------


def write_samples(samples: Iterable[Sample], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_samples(path: Path) -> list[Sample]:
    out: list[Sample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(
            Sample(
                system=d.get("system", ""),
                conversations=list(d.get("conversations", [])),
                meta=dict(d.get("meta", {})),
            )
        )
    return out


def persona_perspective_counts(samples: Sequence[Sample]) -> list[dict[str, str | int]]:
    """How many curated sessions and SFT windows each persona gets as ``assistant``.

    Every sample is built from one author's point of view: that author is the
    trained ``assistant`` and everyone else is ``user``. Rows are sorted by
    ``persona_id``. Samples with a missing ``persona_id`` are skipped.
    """
    accum: dict[str, dict[str, Any]] = {}
    for s in samples:
        pid = str(s.meta.get("persona_id") or "")
        if not pid:
            continue
        row = accum.setdefault(
            pid,
            {"persona_name": "", "sessions": set(), "samples": 0},
        )
        name = s.meta.get("persona_name")
        if name and not row["persona_name"]:
            row["persona_name"] = str(name)
        sid = s.meta.get("session_id")
        if sid is not None:
            row["sessions"].add(sid)
        row["samples"] += 1
    out: list[dict[str, str | int]] = []
    for pid in sorted(accum):
        r = accum[pid]
        out.append(
            {
                "persona_id": pid,
                "persona_name": r["persona_name"],
                "distinct_sessions": len(r["sessions"]),
                "samples": r["samples"],
            }
        )
    return out
