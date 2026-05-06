from datetime import datetime, timedelta, timezone

import pytest

from discord_sft.data_prep.curate import CurateReport, record_to_session, curate_messages
from discord_sft.data_prep.ingest import Message


def _m(
    mid: str,
    ts: datetime,
    author_id: str,
    author_name: str,
    content: str,
    *,
    mtype: int = 0,
    referenced_id: str | None = None,
    reply_to_preview: str | None = None,
    reply_to_author_name: str | None = None,
    reply_to_missing: bool = False,
    attachments: tuple[str, ...] = (),
    folder: str = "dm",
) -> Message:
    return Message(
        id=mid,
        ts=ts,
        author_id=author_id,
        author_name=author_name,
        content=content,
        edited_ts=None,
        attachments=attachments,
        num_embeds=0,
        type=mtype,
        referenced_id=referenced_id,
        reply_to_preview=reply_to_preview,
        reply_to_author_id=("B" if reply_to_author_name else None),
        reply_to_author_name=reply_to_author_name,
        reply_to_missing=reply_to_missing,
        folder=folder,
    )


T0 = datetime(2025, 7, 2, 12, 0, 0, tzinfo=timezone.utc)


def _t(off_sec: int) -> datetime:
    return T0 + timedelta(seconds=off_sec)


def test_record_to_session_repairs_year_month_separator_typo():
    rec = {
        "session_id": "dm#bad-ts",
        "folder": "dm",
        "turns": [
            {
                "author_id": "A",
                "author_name": "Alice",
                "start_ts": "2022)12-27T19:23:49.528000+00:00",
                "end_ts": "2022)12-27T19:23:50.528000+00:00",
                "text": "hi",
                "source_ids": ["1"],
            }
        ],
    }

    session = record_to_session(rec)

    assert session.turns[0].start_ts == datetime.fromisoformat(
        "2022-12-27T19:23:49.528000+00:00"
    )


def test_burst_merge_same_author_within_gap():
    msgs = [
        _m("1", _t(0), "A", "Alice", "hello there"),
        _m("2", _t(10), "A", "Alice", "are you around"),
        _m("3", _t(15), "B", "Bob", "yeah I am here whats up"),
        _m("4", _t(30), "A", "Alice", "cool ok"),
    ]
    sessions, _r = curate_messages(msgs, folder="dm", min_turns=2, near_dedup_threshold=None)
    assert len(sessions) == 1
    turns = sessions[0].turns
    assert [t.author_id for t in turns] == ["A", "B", "A"]
    assert turns[0].text == "hello there\nare you around"
    assert len(turns[0].source_ids) == 2


def test_session_split_on_gap():
    msgs = [
        _m("1", _t(0), "A", "Alice", "hi"),
        _m("2", _t(30), "B", "Bob", "hey"),
        _m("3", _t(60), "A", "Alice", "how r u"),
        _m("4", _t(60 + 3 * 3600), "B", "Bob", "new session"),  # +3h
        _m("5", _t(60 + 3 * 3600 + 5), "A", "Alice", "back"),
        _m("6", _t(60 + 3 * 3600 + 10), "B", "Bob", "yo"),
    ]
    sessions, _r = curate_messages(
        msgs, folder="dm", min_turns=2, session_gap_min=120, near_dedup_threshold=None
    )
    assert len(sessions) == 2
    assert len(sessions[0].turns) == 3
    assert len(sessions[1].turns) == 3


def test_deleted_reference_reply_is_dropped():
    msgs = [
        _m("1", _t(0), "A", "Alice", "hi"),
        _m("2", _t(10), "B", "Bob", "yo"),
        _m(
            "3",
            _t(20),
            "A",
            "Alice",
            "this replies to a deleted msg",
            mtype=19,
            referenced_id="999",
            reply_to_missing=True,
        ),
        _m("4", _t(30), "B", "Bob", "huh"),
    ]
    sessions, report = curate_messages(
        msgs, folder="dm", min_turns=2, near_dedup_threshold=None
    )
    assert report.dropped_deleted_ref == 1
    texts = [t.text for s in sessions for t in s.turns]
    assert "this replies to a deleted msg" not in texts


def test_system_type_and_bot_cmd_dropped():
    msgs = [
        _m("1", _t(0), "A", "Alice", "hi"),
        _m("2", _t(5), "A", "Alice", "channel pinned", mtype=6),
        _m("3", _t(10), "A", "Alice", "!ban someone"),
        _m("4", _t(15), "B", "Bob", "yo"),
        _m("5", _t(20), "A", "Alice", "ok"),
    ]
    sessions, report = curate_messages(
        msgs, folder="dm", min_turns=2, near_dedup_threshold=None
    )
    assert report.dropped_system_type == 1
    assert report.dropped_bot_cmd == 1
    all_texts = [t.text for s in sessions for t in s.turns]
    assert "channel pinned" not in all_texts
    assert "!ban someone" not in all_texts


def test_monologue_sessions_dropped():
    msgs = [
        _m("1", _t(0), "A", "Alice", "one two three four five six seven eight"),
        _m("2", _t(30), "A", "Alice", "nine ten eleven twelve thirteen fourteen fifteen"),
        _m("3", _t(60), "A", "Alice", "sixteen seventeen eighteen nineteen twenty"),
        _m("4", _t(90), "B", "Bob", "ok"),
    ]
    sessions, report = curate_messages(
        msgs,
        folder="dm",
        min_turns=2,
        monologue_max_share=0.80,
        near_dedup_threshold=None,
    )
    assert report.dropped_monologue == 1
    assert sessions == []


def test_reply_inline_non_adjacent():
    msgs = [
        _m("1", _t(0), "B", "Bob", "first message"),
        _m("2", _t(30), "A", "Alice", "something else"),
        _m("3", _t(60), "B", "Bob", "another"),
        _m(
            "4",
            _t(90),
            "A",
            "Alice",
            "responding to first",
            mtype=19,
            referenced_id="1",
            reply_to_preview="first message",
            reply_to_author_name="Bob",
        ),
    ]
    sessions, _r = curate_messages(msgs, folder="dm", min_turns=2, near_dedup_threshold=None)
    last_text = sessions[0].turns[-1].text
    assert last_text.startswith("> Bob: first message")
    assert "responding to first" in last_text


def test_pii_scrubbed():
    msgs = [
        _m("1", _t(0), "A", "Alice", "my email is foo@bar.com"),
        _m("2", _t(10), "B", "Bob", "call me at 555-123-4567"),
        _m("3", _t(20), "A", "Alice", "ok"),
    ]
    sessions, _r = curate_messages(
        msgs, folder="dm", min_turns=2, near_dedup_threshold=None
    )
    joined = " ".join(t.text for s in sessions for t in s.turns)
    assert "foo@bar.com" not in joined
    assert "[email]" in joined
    assert "555-123-4567" not in joined
    assert "[phone]" in joined


def test_exact_duplicate_turn_deduped():
    msgs = [
        _m("1", _t(0), "A", "Alice", "lol"),
        _m("2", _t(120), "B", "Bob", "what"),
        _m("3", _t(240), "A", "Alice", "lol"),  # exact dup, > merge gap
        _m("4", _t(360), "B", "Bob", "ok"),
    ]
    sessions, report = curate_messages(
        msgs,
        folder="dm",
        merge_gap_sec=60,
        min_turns=2,
        near_dedup_threshold=None,
    )
    assert report.dropped_duplicate_turn == 1
    texts = [t.text for t in sessions[0].turns]
    assert texts.count("lol") == 1


def test_exact_duplicate_turn_kept_when_dedupe_disabled():
    msgs = [
        _m("1", _t(0), "A", "Alice", "lol"),
        _m("2", _t(120), "B", "Bob", "what"),
        _m("3", _t(240), "A", "Alice", "lol"),
        _m("4", _t(360), "B", "Bob", "ok"),
    ]
    sessions, report = curate_messages(
        msgs,
        folder="dm",
        merge_gap_sec=60,
        min_turns=2,
        near_dedup_threshold=None,
        dedupe_exact_turns=False,
    )
    assert report.dropped_duplicate_turn == 0
    texts = [t.text for t in sessions[0].turns]
    assert texts.count("lol") == 2


def test_exact_duplicate_turn_respects_dup_cap():
    msgs = [
        _m("1", _t(0), "A", "Alice", "lol"),
        _m("2", _t(120), "A", "Alice", "lol"),
        _m("3", _t(240), "A", "Alice", "lol"),
        _m("4", _t(360), "B", "Bob", "ok"),
    ]
    sessions, report = curate_messages(
        msgs,
        folder="dm",
        merge_gap_sec=60,
        min_turns=2,
        near_dedup_threshold=None,
        exact_turn_dup_cap=2,
    )
    assert report.dropped_duplicate_turn == 1
    texts = [t.text for t in sessions[0].turns]
    assert texts.count("lol") == 2


def test_exact_turn_dup_cap_rejects_below_one():
    with pytest.raises(ValueError, match="exact_turn_dup_cap"):
        curate_messages([], exact_turn_dup_cap=0)
