"""Curation: raw Discord messages -> cleaned, session-split, merged turns.

Pipeline stages (all pure-Python, iterator-friendly):
    1. Message-level filters: system messages, bot commands, deleted-reply drop.
    2. Reply-inlining: attach quote lines for replies whose target is not the
       immediately preceding turn in the filtered stream.
    3. Text normalization (see :mod:`discord_sft.data_prep.normalize`).
    4. Attachment/empty-content handling.
    5. Burst-merge: consecutive same-author messages within ``merge_gap_sec``
       become one :class:`Turn` joined by ``"\\n"``.
    6. Session split on ``session_gap_min`` silence.
    7. Session-level filters: min turns, min distinct authors, monologue cap,
       exact-duplicate-turn dedup, optional language filter, MinHash near-dup.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from discord_sft.data_prep.ingest import Message
from discord_sft.data_prep.normalize import (
    build_mention_table,
    normalize_text,
)


_BOT_CMD_RE = re.compile(r"^[!?/][A-Za-z][\w-]*(?:\s|$)")
_REF_SNIPPET_MAX = 120


@dataclass
class Turn:
    author_id: str
    author_name: str
    start_ts: datetime
    end_ts: datetime
    text: str
    source_ids: list[str] = field(default_factory=list)

    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Session:
    id: str
    folder: str
    turns: list[Turn]

    @property
    def first_ts(self) -> datetime:
        return self.turns[0].start_ts

    @property
    def last_ts(self) -> datetime:
        return self.turns[-1].end_ts

    def authors(self) -> set[str]:
        return {t.author_id for t in self.turns}


@dataclass
class CurateReport:
    total_in: int = 0
    dropped_system_type: int = 0
    dropped_bot_cmd: int = 0
    dropped_deleted_ref: int = 0
    dropped_empty: int = 0
    dropped_duplicate_turn: int = 0
    sessions_built: int = 0
    sessions_kept: int = 0
    dropped_short_session: int = 0
    dropped_monologue: int = 0
    dropped_single_author: int = 0
    dropped_lang: int = 0
    dropped_near_dup: int = 0


def _is_bot_command(text: str) -> bool:
    if not text:
        return False
    return bool(_BOT_CMD_RE.match(text.strip()))


def _short(s: str | None, n: int = _REF_SNIPPET_MAX) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "..."


def _filter_messages(
    messages: Sequence[Message],
    report: CurateReport,
) -> list[Message]:
    """Stage 1: drop system messages, bot commands, replies to deleted originals."""
    out: list[Message] = []
    for m in messages:
        report.total_in += 1
        if m.type not in (0, 19):
            report.dropped_system_type += 1
            continue
        if m.type == 19 and m.reply_to_missing:
            report.dropped_deleted_ref += 1
            continue
        if _is_bot_command(m.content):
            report.dropped_bot_cmd += 1
            continue
        out.append(m)
    return out


def _apply_reply_inline(messages: Sequence[Message]) -> list[tuple[Message, str]]:
    """Stage 2: pair each message with an optional quote-prefix string."""
    out: list[tuple[Message, str]] = []
    prev_id: str | None = None
    for m in messages:
        prefix = ""
        if (
            m.type == 19
            and m.referenced_id
            and m.reply_to_preview
            and m.referenced_id != prev_id
        ):
            who = m.reply_to_author_name or "user"
            prefix = f"> {who}: {_short(m.reply_to_preview)}\n"
        out.append((m, prefix))
        prev_id = m.id
    return out


def _normalize_and_filter_empty(
    paired: Sequence[tuple[Message, str]],
    mention_table: dict[str, str],
    *,
    url_strip: bool,
    pii_scrub: bool,
    report: CurateReport,
) -> list[tuple[Message, str]]:
    """Stage 3+4: normalize text and drop content-empty / attachment-only stragglers."""
    out: list[tuple[Message, str]] = []
    for m, prefix in paired:
        body = normalize_text(
            m.content,
            mention_table,
            url_strip=url_strip,
            pii_scrub=pii_scrub,
        )
        text = (prefix + body).strip()
        if not text:
            if m.attachments or m.num_embeds:
                text = "[image]"
            else:
                report.dropped_empty += 1
                continue
        out.append((m, text))
    return out


def _burst_merge(
    paired: Sequence[tuple[Message, str]],
    *,
    merge_gap_sec: int,
) -> list[Turn]:
    """Stage 5: merge consecutive same-author messages within ``merge_gap_sec`` seconds."""
    turns: list[Turn] = []
    for m, text in paired:
        if turns:
            last = turns[-1]
            dt = (m.ts - last.end_ts).total_seconds()
            if last.author_id == m.author_id and 0 <= dt <= merge_gap_sec:
                last.text = f"{last.text}\n{text}"
                last.end_ts = m.ts
                last.source_ids.append(m.id)
                continue
        turns.append(
            Turn(
                author_id=m.author_id,
                author_name=m.author_name,
                start_ts=m.ts,
                end_ts=m.ts,
                text=text,
                source_ids=[m.id],
            )
        )
    return turns


def _split_sessions(
    turns: Sequence[Turn],
    *,
    folder: str,
    session_gap_min: int,
) -> list[Session]:
    """Stage 6: cut a new session on any silence > session_gap_min."""
    if not turns:
        return []
    gap = timedelta(minutes=session_gap_min)
    sessions: list[Session] = []
    current: list[Turn] = []
    for t in turns:
        if current and t.start_ts - current[-1].end_ts > gap:
            sessions.append(_make_session(folder, current))
            current = []
        current.append(t)
    if current:
        sessions.append(_make_session(folder, current))
    return sessions


def _make_session(folder: str, turns: list[Turn]) -> Session:
    first_ts = turns[0].start_ts
    sid = f"{folder}#{first_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    return Session(id=sid, folder=folder, turns=turns)


def _dedup_exact_turns(
    session: Session,
    report: CurateReport,
    *,
    max_per_author_text: int = 1,
) -> Session:
    """Drop excess turns sharing the same ``(author_id, text)`` within one session.

    ``max_per_author_text`` is how many identical lines to **keep** per key; the
    default ``1`` matches classic dedup (first only). Use ``2`` to allow one
    intentional repeat (e.g. ``lol`` ``lol``), then trim further spam.
    """
    counts: dict[tuple[str, str], int] = {}
    kept: list[Turn] = []
    for t in session.turns:
        key = (t.author_id, t.text)
        n = counts.get(key, 0)
        if n >= max_per_author_text:
            report.dropped_duplicate_turn += 1
            continue
        counts[key] = n + 1
        kept.append(t)
    session.turns = kept
    return session


def _session_passes(
    session: Session,
    *,
    min_turns: int,
    min_authors: int,
    monologue_max_share: float,
    report: CurateReport,
) -> bool:
    if len(session.turns) < min_turns:
        report.dropped_short_session += 1
        return False
    if len(session.authors()) < min_authors:
        report.dropped_single_author += 1
        return False
    total_words = sum(t.word_count() for t in session.turns) or 1
    per_author: dict[str, int] = {}
    for t in session.turns:
        per_author[t.author_id] = per_author.get(t.author_id, 0) + t.word_count()
    top_share = max(per_author.values()) / total_words
    if top_share > monologue_max_share:
        report.dropped_monologue += 1
        return False
    return True


def _language_filter(
    sessions: list[Session],
    lang: str | None,
    report: CurateReport,
) -> list[Session]:
    if not lang:
        return sessions
    try:
        from langdetect import DetectorFactory, detect  # type: ignore
    except ImportError:
        return sessions
    DetectorFactory.seed = 0
    kept: list[Session] = []
    for s in sessions:
        blob = " ".join(t.text for t in s.turns)[:4000]
        try:
            if detect(blob) != lang:
                report.dropped_lang += 1
                continue
        except Exception:
            pass
        kept.append(s)
    return kept


def _shingles(text: str, n: int = 5) -> set[str]:
    toks = text.lower().split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _near_dedup(
    sessions: list[Session],
    threshold: float,
    report: CurateReport,
) -> list[Session]:
    """Collapse near-duplicate sessions via MinHash LSH; fall back to pure-python Jaccard."""
    if len(sessions) < 2:
        return sessions
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except ImportError:
        return _near_dedup_pure(sessions, threshold, report)

    def _mh(text: str) -> "MinHash":
        mh = MinHash(num_perm=128)
        for sh in _shingles(text):
            mh.update(sh.encode("utf-8"))
        return mh

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept_map: dict[str, Session] = {}
    dropped: set[str] = set()
    # Process longer sessions first so dups collapse onto the longer canonical.
    ordered = sorted(sessions, key=lambda s: -sum(t.word_count() for t in s.turns))
    for s in ordered:
        if s.id in dropped:
            continue
        blob = "\n".join(t.text for t in s.turns)
        mh = _mh(blob)
        dup_ids = lsh.query(mh)
        if dup_ids:
            dropped.add(s.id)
            report.dropped_near_dup += 1
            continue
        lsh.insert(s.id, mh)
        kept_map[s.id] = s
    # Preserve original order
    return [s for s in sessions if s.id in kept_map]


def _near_dedup_pure(
    sessions: list[Session],
    threshold: float,
    report: CurateReport,
) -> list[Session]:
    kept: list[tuple[str, set[str]]] = []
    ordered = sorted(sessions, key=lambda s: -sum(t.word_count() for t in s.turns))
    keep_set: set[str] = set()
    for s in ordered:
        blob = "\n".join(t.text for t in s.turns)
        sh = _shingles(blob)
        dup = False
        for _, prev in kept:
            if not sh or not prev:
                continue
            inter = len(sh & prev)
            union = len(sh | prev)
            if union and inter / union >= threshold:
                dup = True
                break
        if dup:
            report.dropped_near_dup += 1
            continue
        kept.append((s.id, sh))
        keep_set.add(s.id)
    return [s for s in sessions if s.id in keep_set]


def curate_messages(
    messages: Sequence[Message],
    *,
    folder: str | None = None,
    merge_gap_sec: int = 30,
    session_gap_min: int = 60,
    min_turns: int = 2,
    min_authors: int = 2,
    monologue_max_share: float = 0.80,
    url_strip: bool = False,
    pii_scrub: bool = True,
    lang: str | None = None,
    near_dedup_threshold: float | None = 0.85,
    dedupe_exact_turns: bool = True,
    exact_turn_dup_cap: int = 1,
    report: CurateReport | None = None,
) -> tuple[list[Session], CurateReport]:
    """Curate one folder's messages into a list of quality-gated sessions.

    When ``dedupe_exact_turns`` is true (default), excess turns sharing the same
    ``(author_id, text)`` in one session are dropped and counted on
    ``report.dropped_duplicate_turn``. ``exact_turn_dup_cap`` is how many such
    turns to **keep** per key (minimum ``1``); higher values soften dedup.
    """
    if report is None:
        report = CurateReport()
    if exact_turn_dup_cap < 1:
        raise ValueError("exact_turn_dup_cap must be >= 1")
    if not messages:
        return [], report
    folder_name = folder or messages[0].folder

    mention_table = build_mention_table(messages)

    stage1 = _filter_messages(messages, report)
    stage2 = _apply_reply_inline(stage1)
    stage3 = _normalize_and_filter_empty(
        stage2,
        mention_table,
        url_strip=url_strip,
        pii_scrub=pii_scrub,
        report=report,
    )
    turns = _burst_merge(stage3, merge_gap_sec=merge_gap_sec)
    sessions = _split_sessions(turns, folder=folder_name, session_gap_min=session_gap_min)
    report.sessions_built += len(sessions)

    if dedupe_exact_turns:
        sessions = [
            _dedup_exact_turns(s, report, max_per_author_text=exact_turn_dup_cap)
            for s in sessions
        ]
    sessions = [
        s
        for s in sessions
        if _session_passes(
            s,
            min_turns=min_turns,
            min_authors=min_authors,
            monologue_max_share=monologue_max_share,
            report=report,
        )
    ]
    sessions = _language_filter(sessions, lang, report)
    if near_dedup_threshold is not None and near_dedup_threshold > 0:
        sessions = _near_dedup(sessions, near_dedup_threshold, report)
    report.sessions_kept += len(sessions)
    return sessions, report


def session_to_record(session: Session) -> dict:
    return {
        "session_id": session.id,
        "folder": session.folder,
        "first_ts": session.first_ts.isoformat(),
        "last_ts": session.last_ts.isoformat(),
        "authors": sorted(session.authors()),
        "turns": [
            {
                "author_id": t.author_id,
                "author_name": t.author_name,
                "start_ts": t.start_ts.isoformat(),
                "end_ts": t.end_ts.isoformat(),
                "text": t.text,
                "source_ids": t.source_ids,
            }
            for t in session.turns
        ],
    }


def record_to_session(rec: dict) -> Session:
    turns = [
        Turn(
            author_id=t["author_id"],
            author_name=t["author_name"],
            start_ts=_parse_record_ts(t["start_ts"]),
            end_ts=_parse_record_ts(t["end_ts"]),
            text=t["text"],
            source_ids=list(t.get("source_ids", [])),
        )
        for t in rec["turns"]
    ]
    return Session(id=rec["session_id"], folder=rec["folder"], turns=turns)


def _parse_record_ts(raw: str) -> datetime:
    """Parse persisted session timestamps, repairing one known export typo."""
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        if len(raw) > 4 and raw[4] == ")" and raw[:4].isdigit():
            repaired = raw[:4] + "-" + raw[5:]
            return datetime.fromisoformat(repaired)
        raise


def stable_session_id(folder: str, first_ts: datetime) -> str:
    """Deterministic session id; exposed for tests and downstream joins."""
    raw = f"{folder}|{first_ts.isoformat()}"
    return f"{folder}#{hashlib.sha1(raw.encode()).hexdigest()[:10]}"
