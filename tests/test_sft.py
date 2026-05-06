from datetime import datetime, timedelta, timezone

from discord_sft.data_prep.curate import Session, Turn
from discord_sft.data_prep.sft import (
    Sample,
    balance_samples,
    balance_turn_length,
    build_samples,
    persona_perspective_counts,
    post_split_num_turns_breakdown,
    split_train_val,
)


T0 = datetime(2025, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def _turn(author_id: str, author_name: str, off: int, text: str) -> Turn:
    return Turn(
        author_id=author_id,
        author_name=author_name,
        start_ts=T0 + timedelta(seconds=off),
        end_ts=T0 + timedelta(seconds=off),
        text=text,
        source_ids=[f"{author_id}-{off}"],
    )


def _session(sid: str, turns: list[Turn]) -> Session:
    return Session(id=sid, folder="dm", turns=turns)


def test_two_person_session_produces_disjoint_persona_samples():
    # A/B/A/B session; one row per persona (last raw line each).
    sess = _session(
        "s1",
        [
            _turn("A", "Alice", 0, "hi"),
            _turn("B", "Bob", 10, "yo"),
            _turn("A", "Alice", 20, "sup"),
            _turn("B", "Bob", 30, "chilling"),
        ],
    )
    samples = build_samples([sess], window_turns=3)

    by_persona: dict[str, list] = {}
    for s in samples:
        by_persona.setdefault(s.meta["persona_id"], []).append(s)
    assert len(by_persona["A"]) == 1
    assert len(by_persona["B"]) == 1

    # Every sample ends on an assistant turn.
    for s in samples:
        assert s.conversations[-1]["from"] == "assistant"
        assert s.conversations[0]["from"] == "user"

    # No duplication: (session_id, last source id) is unique.
    keys = [(s.meta["session_id"], tuple(s.meta["end_source_ids"])) for s in samples]
    assert len(keys) == len(set(keys))


def test_system_prompt_includes_persona_and_counterparty():
    sess = _session(
        "s1",
        [
            _turn("A", "Alice", 0, "hi"),
            _turn("B", "Bob", 10, "yo"),
            _turn("A", "Alice", 20, "sup"),
        ],
    )
    samples = build_samples([sess], window_turns=5)
    a_sample = next(s for s in samples if s.meta["persona_id"] == "A")
    assert "Alice" in a_sample.system
    assert "Bob" in a_sample.system


def test_consecutive_same_role_turns_merged():
    # Two A turns in a row: only the last A turn emits a row; merged assistant text.
    sess = _session(
        "s1",
        [
            _turn("B", "Bob", 0, "hey"),
            _turn("A", "Alice", 10, "first"),
            _turn("A", "Alice", 20, "second"),
        ],
    )
    samples = build_samples([sess], window_turns=5)
    a_samples = [s for s in samples if s.meta["persona_id"] == "A"]
    assert len(a_samples) == 1
    a_only = a_samples[0]
    assert [t["from"] for t in a_only.conversations] == ["user", "assistant"]
    assert "first" in a_only.conversations[-1]["value"]
    assert "second" in a_only.conversations[-1]["value"]


def test_balance_median_caps_dominant_persona():
    # Simulate imbalance: A=900, B=100, C=100, spread across many sessions.
    sessions = []
    sample_count = 0
    for i in range(30):
        turns = []
        for j in range(30):
            turns.append(_turn("A", "Alice", i * 1000 + j * 2, f"a{i}_{j}"))
            turns.append(_turn("X", "X", i * 1000 + j * 2 + 1, "stub"))
        sessions.append(_session(f"Asess{i}", turns))
    for i in range(5):
        turns = []
        for j in range(20):
            turns.append(_turn("B", "Bob", 5_000_000 + i * 1000 + j * 2, f"b{i}_{j}"))
            turns.append(_turn("X", "X", 5_000_000 + i * 1000 + j * 2 + 1, "stub"))
        sessions.append(_session(f"Bsess{i}", turns))
    for i in range(5):
        turns = []
        for j in range(20):
            turns.append(_turn("C", "Cat", 9_000_000 + i * 1000 + j * 2, f"c{i}_{j}"))
            turns.append(_turn("X", "X", 9_000_000 + i * 1000 + j * 2 + 1, "stub"))
        sessions.append(_session(f"Csess{i}", turns))

    samples = build_samples(sessions, personas=["A", "B", "C"], window_turns=4)
    counts_before: dict[str, int] = {}
    for s in samples:
        counts_before[s.meta["persona_id"]] = counts_before.get(s.meta["persona_id"], 0) + 1
    assert counts_before["A"] > counts_before["B"]
    assert counts_before["A"] > counts_before["C"]

    kept, report = balance_samples(samples, policy="median", k=1.5, seed=0)
    after: dict[str, int] = {}
    by_persona_sessions: dict[str, set[str]] = {}
    for s in kept:
        pid = s.meta["persona_id"]
        after[pid] = after.get(pid, 0) + 1
        by_persona_sessions.setdefault(pid, set()).add(s.meta["session_id"])

    # Median of {A_big, B, C} scaled by k=1.5 -> cap just above B/C counts, far below A.
    assert after["A"] < counts_before["A"]
    assert after["B"] == counts_before["B"]
    assert after["C"] == counts_before["C"]
    # A samples drawn from multiple sessions (stratified).
    assert len(by_persona_sessions["A"]) >= 2
    # Report contains both snapshots.
    assert report.counts_before == counts_before
    assert report.counts_after == after


def test_balance_none_is_identity():
    sessions = [
        _session(
            "s",
            [
                _turn("A", "Alice", 0, "hi"),
                _turn("B", "Bob", 1, "yo"),
                _turn("A", "Alice", 2, "sup"),
            ],
        )
    ]
    samples = build_samples(sessions, window_turns=3)
    kept, _r = balance_samples(samples, policy="none", seed=0)
    assert len(kept) == len(samples)


def test_train_val_split_by_session():
    sessions = [
        _session(
            f"s{i}",
            [
                _turn("A", "Alice", 0, "hi"),
                _turn("B", "Bob", 1, "yo"),
                _turn("A", "Alice", 2, "sup"),
            ],
        )
        for i in range(20)
    ]
    samples = build_samples(sessions, window_turns=3)
    train, val = split_train_val(samples, val_frac=0.2, seed=0)
    train_sids = {s.meta["session_id"] for s in train}
    val_sids = {s.meta["session_id"] for s in val}
    assert train_sids.isdisjoint(val_sids)
    assert len(train) + len(val) == len(samples)


def test_max_sharegpt_turns_caps_long_alternation():
    # 20 raw turns B,A,... ending on A at index 19 — unconstrained slice can exceed 8 messages.
    turns = []
    for i in range(20):
        aid = "B" if i % 2 == 0 else "A"
        name = "Bob" if aid == "B" else "Alice"
        turns.append(_turn(aid, name, i, f"t{i}"))
    sess = _session("long", turns)
    capped = build_samples([sess], window_turns=32, max_sharegpt_turns=8)
    assert capped
    assert all(s.meta["num_turns"] <= 8 for s in capped)
    assert max(s.meta["num_turns"] for s in capped) == 8

    uncapped = build_samples([sess], window_turns=32, max_sharegpt_turns=None)
    assert max(s.meta["num_turns"] for s in uncapped) > 8


def test_shuffle_samples_is_deterministic():
    from copy import deepcopy

    from discord_sft.data_prep.sft import Sample, shuffle_samples

    base = [Sample("", [], {"i": i}) for i in range(12)]
    a = deepcopy(base)
    b = deepcopy(base)
    shuffle_samples(a, seed=999)
    shuffle_samples(b, seed=999)
    assert [x.meta["i"] for x in a] == [x.meta["i"] for x in b]
    assert [x.meta["i"] for x in a] != list(range(12))


def test_post_split_num_turns_breakdown_train_vs_val():
    train = [
        Sample("", [], {"num_turns": 2, "session_id": "s1", "persona_id": "p"}) for _ in range(10)
    ]
    val = [
        Sample("", [], {"num_turns": 8, "session_id": "s2", "persona_id": "p"}) for _ in range(2)
    ]
    d = post_split_num_turns_breakdown(train, val)
    assert d["TRAIN_split"]["row_count"] == 10
    assert d["VALIDATION_split"]["row_count"] == 2
    assert d["TRAIN_split"]["counts_by_num_turns"]["2"] == 10
    assert d["TRAIN_split"]["fraction_within_TRAIN_rows"]["2"] == 1.0
    assert d["VALIDATION_split"]["counts_by_num_turns"]["8"] == 2
    assert "note" in d


def test_balance_turn_length_uniform_downsamples():
    def _mk(nt: int, tag: str, i: int) -> Sample:
        conv = []
        for j in range(nt // 2):
            conv.append({"from": "user", "value": f"{tag}-{i}-{j}"})
            conv.append({"from": "assistant", "value": f"a{i}-{j}"})
        return Sample(
            "sys",
            conv,
            {
                "persona_id": "p",
                "session_id": f"s_{nt}_{i}",
                "num_turns": nt,
            },
        )

    many8 = [_mk(8, "eight", i) for i in range(40)]
    few2 = [_mk(2, "two", i) for i in range(4)]
    samples = many8 + few2
    out, report = balance_turn_length(samples, policy="uniform", seed=0)
    assert len(out) < len(samples)
    after = {int(k): v for k, v in report.counts_after.items() if k.isdigit()}
    assert after.get(2, 0) == 4
    assert after.get(8, 0) < 40
    assert report.alpha is not None


def test_persona_perspective_counts_sessions_and_samples():
    samples = [
        Sample("", [], {"persona_id": "A", "persona_name": "Alice", "session_id": "s1"}),
        Sample("", [], {"persona_id": "A", "session_id": "s2"}),
        Sample("", [], {"persona_id": "A", "session_id": "s1"}),
        Sample("", [], {"persona_id": "B", "persona_name": "Bob", "session_id": "s1"}),
    ]
    rows = {r["persona_id"]: r for r in persona_perspective_counts(samples)}
    assert rows["A"]["distinct_sessions"] == 2
    assert rows["A"]["samples"] == 3
    assert rows["A"]["persona_name"] == "Alice"
    assert rows["B"]["distinct_sessions"] == 1
    assert rows["B"]["samples"] == 1
