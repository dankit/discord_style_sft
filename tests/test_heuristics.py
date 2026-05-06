from discord_sft.analysis.fingerprint import (
    aggregate_profile_drift_for_eval_rows,
    build_profiles,
    score_against_profile,
)
from discord_sft.analysis.heuristics import profile_heuristics, style_heuristics
from discord_sft.data_prep.sft import Sample


def test_style_heuristics_aligned_lists():
    refs = ["hello there friend", "nice one"]
    gens = ["hi there", "Nice one!!"]
    stats = style_heuristics(gens, refs)
    assert "error" not in stats
    assert stats["exclamation_rate"] == 0.5
    assert stats["filler_rate"] == 0.0
    assert "fillers_used" in stats


def test_style_heuristics_empty():
    assert "error" in style_heuristics([], ["a"])


def test_style_heuristics_custom_filler_list():
    refs = ["anything", "anything"]
    gens = ["imma go soon", "nothing here"]
    stats = style_heuristics(gens, refs, fillers=("imma",))
    assert stats["filler_rate"] == 0.5


def test_profile_heuristics_uses_mined_fillers():
    samples = [
        Sample(
            system="You are A.",
            conversations=[
                {"from": "user", "value": "hi"},
                {"from": "assistant", "value": "imma go soon imma chill imma sleep"},
            ],
            meta={"persona_id": "A", "session_id": "s1"},
        ),
        Sample(
            system="You are A.",
            conversations=[
                {"from": "user", "value": "hey"},
                {"from": "assistant", "value": "imma hit the gym imma lift"},
            ],
            meta={"persona_id": "A", "session_id": "s2"},
        ),
        Sample(
            system="You are B.",
            conversations=[
                {"from": "user", "value": "hi"},
                {"from": "assistant", "value": "going to the gym soon"},
            ],
            meta={"persona_id": "B", "session_id": "s3"},
        ),
        Sample(
            system="You are B.",
            conversations=[
                {"from": "user", "value": "hey"},
                {"from": "assistant", "value": "heading out for lunch"},
            ],
            meta={"persona_id": "B", "session_id": "s4"},
        ),
    ]
    profiles = build_profiles(samples)
    # Persona A's top unigrams should include "imma".
    a_unigrams = [it["token"] for it in profiles["personas"]["A"]["top_fillers"]["1gram"]]
    assert "imma" in a_unigrams

    # profile_heuristics picks up "imma" and flags generated lines containing it.
    stats = profile_heuristics(
        ["imma nap", "normal sentence"],
        ["whatever one", "whatever two"],
        profile=profiles["personas"]["A"],
    )
    assert stats["filler_rate"] == 0.5
    assert stats["profile_persona"] == "A"


def test_score_against_profile_matches_keys():
    samples = [
        Sample(
            system=".",
            conversations=[
                {"from": "user", "value": "h"},
                {"from": "assistant", "value": "hello friend there"},
            ],
            meta={"persona_id": "Z", "session_id": "s1"},
        ),
    ]
    profiles = build_profiles(samples)
    z = profiles["personas"]["Z"]
    out = score_against_profile(["hello friend there"], z)
    assert "error" not in out
    assert out["n_generated"] == 1
    assert out["mean_words_delta"] == 0.0
    assert "emoji_unicode_per_turn_delta" in out
    assert out["profile_mean_words"] == out["generated_mean_words"]


def test_aggregate_profile_drift_for_eval_rows_groups():
    samples = [
        Sample(
            system=".",
            conversations=[
                {"from": "user", "value": "h"},
                {"from": "assistant", "value": "one two three"},
            ],
            meta={"persona_id": "p1", "persona_name": "One", "session_id": "a"},
        ),
        Sample(
            system=".",
            conversations=[
                {"from": "user", "value": "h"},
                {"from": "assistant", "value": "four five six seven"},
            ],
            meta={"persona_id": "p1", "persona_name": "One", "session_id": "b"},
        ),
    ]
    profiles = build_profiles(samples)
    rows = [
        {"persona_id": "p1", "persona_name": "One", "generated": "one two three"},
        {"persona_id": "p1", "persona_name": "One", "generated": "four five six seven"},
    ]
    table, warns = aggregate_profile_drift_for_eval_rows(rows, profiles)
    assert not warns
    assert len(table) == 1
    assert table[0]["persona_id"] == "p1"
    assert table[0]["n_generated"] == 2


def test_aggregate_profile_drift_warns_missing_persona():
    samples = [
        Sample(
            system=".",
            conversations=[
                {"from": "user", "value": "h"},
                {"from": "assistant", "value": "a b"},
            ],
            meta={"persona_id": "only", "session_id": "x"},
        ),
    ]
    profiles = build_profiles(samples)
    rows = [{"persona_id": "missing", "generated": "x y"}]
    table, warns = aggregate_profile_drift_for_eval_rows(rows, profiles)
    assert table == []
    assert any("No fingerprint profile" in w for w in warns)
