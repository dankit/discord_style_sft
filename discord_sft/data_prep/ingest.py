"""Load a Discrub DM export folder into a sorted, deduplicated stream of messages.

The Discrub exporter writes one JSON array per *page*, newest-first, with files
named like ``<folder>-page-<N>.json``. This module provides a pure-Python
iterator that flattens all pages, sorts ascending by timestamp, dedupes by id,
and normalizes the subset of Discord ``Message`` fields this pipeline cares
about. Parquet emission is CLI-only and behind an optional ``pyarrow`` import.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Message:
    id: str
    ts: datetime
    author_id: str
    author_name: str
    content: str
    edited_ts: datetime | None
    attachments: tuple[str, ...]
    num_embeds: int
    type: int
    referenced_id: str | None
    reply_to_preview: str | None
    reply_to_author_id: str | None
    reply_to_author_name: str | None
    reply_to_missing: bool
    folder: str


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(p: Path) -> list:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(p.name)]


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    # Discord uses ISO 8601 with offset; Python 3.11+ handles "+00:00" fine
    return datetime.fromisoformat(raw)


def _author_display(raw: dict[str, Any]) -> str:
    return raw.get("global_name") or raw.get("username") or raw.get("id") or "unknown"


def _message_from_raw(raw: dict[str, Any], folder: str) -> Message | None:
    ts = _parse_ts(raw.get("timestamp"))
    if ts is None:
        return None
    author = raw.get("author") or {}
    author_id = str(author.get("id") or "")
    if not author_id:
        return None

    attachments = tuple(
        a.get("url") or a.get("filename") or ""
        for a in (raw.get("attachments") or [])
        if isinstance(a, dict)
    )
    attachments = tuple(a for a in attachments if a)

    ref = raw.get("referenced_message")
    msg_ref = raw.get("message_reference")
    msg_type = int(raw.get("type") or 0)
    referenced_id: str | None = None
    reply_to_preview: str | None = None
    reply_to_author_id: str | None = None
    reply_to_author_name: str | None = None
    reply_to_missing = False

    if msg_type == 19:
        if isinstance(ref, dict):
            referenced_id = str(ref.get("id") or "") or None
            reply_to_preview = ref.get("content") or None
            ref_author = ref.get("author") or {}
            reply_to_author_id = str(ref_author.get("id") or "") or None
            reply_to_author_name = _author_display(ref_author) if ref_author else None
        elif isinstance(msg_ref, dict):
            referenced_id = str(msg_ref.get("message_id") or "") or None
            reply_to_missing = True
        else:
            reply_to_missing = True

    return Message(
        id=str(raw.get("id") or ""),
        ts=ts,
        author_id=author_id,
        author_name=_author_display(author),
        content=raw.get("content") or "",
        edited_ts=_parse_ts(raw.get("edited_timestamp")),
        attachments=attachments,
        num_embeds=len(raw.get("embeds") or []),
        type=msg_type,
        referenced_id=referenced_id,
        reply_to_preview=reply_to_preview,
        reply_to_author_id=reply_to_author_id,
        reply_to_author_name=reply_to_author_name,
        reply_to_missing=reply_to_missing,
        folder=folder,
    )


def iter_messages(folder: Path) -> Iterator[Message]:
    """Yield messages from one DM folder, sorted ascending by timestamp, deduped by id."""
    folder = Path(folder)
    pages = sorted((p for p in folder.glob("*.json")), key=_natural_key)
    seen: set[str] = set()
    buf: list[Message] = []
    for page in pages:
        with page.open("r", encoding="utf-8") as f:
            raw_list = json.load(f)
        if not isinstance(raw_list, list):
            continue
        for raw in raw_list:
            if not isinstance(raw, dict):
                continue
            msg = _message_from_raw(raw, folder=folder.name)
            if msg is None or not msg.id:
                continue
            if msg.id in seen:
                continue
            seen.add(msg.id)
            buf.append(msg)
    buf.sort(key=lambda m: (m.ts, m.id))
    yield from buf


def iter_folders(root: Path) -> Iterator[Path]:
    """Yield DM subfolders that contain at least one .json page."""
    root = Path(root)
    for child in sorted(root.iterdir()):
        if child.is_dir() and any(child.glob("*.json")):
            yield child


def messages_to_records(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """Convert Message objects to plain dicts (ISO timestamps) for serialization."""
    out: list[dict[str, Any]] = []
    for m in messages:
        rec = asdict(m)
        rec["ts"] = m.ts.isoformat()
        rec["edited_ts"] = m.edited_ts.isoformat() if m.edited_ts else None
        rec["attachments"] = list(m.attachments)
        out.append(rec)
    return out


def write_parquet(records: list[dict[str, Any]], out_path: Path) -> None:
    """Write records as parquet. Requires `pyarrow`."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - exercised only in CLI
        raise ImportError("Install pyarrow to write parquet output") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, out_path)


def write_jsonl(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ingest_root(
    source: Path,
    out_dir: Path,
    *,
    fmt: str = "parquet",
) -> dict[str, int]:
    """Ingest every folder under ``source``; write one file per folder to ``out_dir``.

    Returns a per-folder count report.
    """
    source = Path(source)
    out_dir = Path(out_dir)
    report: dict[str, int] = {}
    for folder in iter_folders(source):
        messages = list(iter_messages(folder))
        records = messages_to_records(messages)
        if fmt == "parquet":
            write_parquet(records, out_dir / f"{folder.name}.parquet")
        else:
            write_jsonl(records, out_dir / f"{folder.name}.jsonl")
        report[folder.name] = len(records)
    return report
