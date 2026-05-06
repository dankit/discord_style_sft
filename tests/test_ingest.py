import json
from pathlib import Path

from discord_sft.data_prep.ingest import iter_messages, messages_to_records


def _msg(mid: str, ts: str, content: str, author_id: str = "A", author_name: str = "Alice", mtype: int = 0):
    return {
        "id": mid,
        "timestamp": ts,
        "content": content,
        "author": {"id": author_id, "username": author_name, "global_name": author_name},
        "type": mtype,
        "attachments": [],
        "embeds": [],
    }


def test_iter_messages_sorts_and_dedupes(tmp_path: Path):
    folder = tmp_path / "dm_alice"
    folder.mkdir()
    # page-1 is "newest", page-2 is "older", plus a duplicate id across pages
    page1 = [
        _msg("3", "2025-07-02T12:00:03+00:00", "c"),
        _msg("2", "2025-07-02T12:00:02+00:00", "b"),
    ]
    page2 = [
        _msg("2", "2025-07-02T12:00:02+00:00", "b"),  # duplicate
        _msg("1", "2025-07-02T12:00:01+00:00", "a"),
    ]
    (folder / "dm_alice-page-1.json").write_text(json.dumps(page1), encoding="utf-8")
    (folder / "dm_alice-page-2.json").write_text(json.dumps(page2), encoding="utf-8")

    msgs = list(iter_messages(folder))
    assert [m.id for m in msgs] == ["1", "2", "3"]
    assert [m.content for m in msgs] == ["a", "b", "c"]
    assert all(m.folder == "dm_alice" for m in msgs)


def test_natural_sort_pages(tmp_path: Path):
    folder = tmp_path / "dm"
    folder.mkdir()
    # With lexical sort, page-10 would come before page-2 and order could matter
    # (id-dedup lets first-seen win). Natural sort keeps pages in numeric order.
    pages = {
        "dm-page-1.json": [_msg("1", "2025-07-02T12:00:01+00:00", "a")],
        "dm-page-2.json": [_msg("2", "2025-07-02T12:00:02+00:00", "b")],
        "dm-page-10.json": [_msg("10", "2025-07-02T12:00:10+00:00", "j")],
        "dm-page-11.json": [_msg("11", "2025-07-02T12:00:11+00:00", "k")],
    }
    for name, data in pages.items():
        (folder / name).write_text(json.dumps(data), encoding="utf-8")
    ids = [m.id for m in iter_messages(folder)]
    assert ids == ["1", "2", "10", "11"]


def test_reply_fields_extracted(tmp_path: Path):
    folder = tmp_path / "dm"
    folder.mkdir()
    page = [
        _msg("1", "2025-07-02T12:00:01+00:00", "original", author_id="B"),
        {
            **_msg("2", "2025-07-02T12:00:02+00:00", "reply", mtype=19),
            "message_reference": {"message_id": "1", "channel_id": "x", "type": 0},
            "referenced_message": {
                "id": "1",
                "content": "original",
                "author": {"id": "B", "username": "bob", "global_name": "Bob"},
            },
        },
    ]
    (folder / "dm-page-1.json").write_text(json.dumps(page), encoding="utf-8")
    msgs = list(iter_messages(folder))
    reply = msgs[1]
    assert reply.type == 19
    assert reply.referenced_id == "1"
    assert reply.reply_to_preview == "original"
    assert reply.reply_to_author_name == "Bob"
    assert reply.reply_to_missing is False


def test_reply_with_deleted_original_flagged(tmp_path: Path):
    folder = tmp_path / "dm"
    folder.mkdir()
    page = [
        {
            **_msg("1", "2025-07-02T12:00:01+00:00", "reply to nothing", mtype=19),
            "message_reference": {"message_id": "999", "channel_id": "x", "type": 0},
            "referenced_message": None,
        }
    ]
    (folder / "dm-page-1.json").write_text(json.dumps(page), encoding="utf-8")
    msgs = list(iter_messages(folder))
    assert msgs[0].reply_to_missing is True
    assert msgs[0].referenced_id == "999"


def test_messages_to_records_roundtrip(tmp_path: Path):
    folder = tmp_path / "dm"
    folder.mkdir()
    page = [_msg("1", "2025-07-02T12:00:01+00:00", "a")]
    (folder / "dm-page-1.json").write_text(json.dumps(page), encoding="utf-8")
    records = messages_to_records(iter_messages(folder))
    assert records[0]["id"] == "1"
    assert records[0]["ts"].startswith("2025-07-02T12:00:01")
    assert records[0]["attachments"] == []
