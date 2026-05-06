"""Text normalization for Discord messages.

The goal is to strip Discord-specific markup (custom emojis, user/role/channel
mentions) while preserving the casual-chat surface features (casing, slang,
fillers, typos) that define each persona's style.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from discord_sft.data_prep.ingest import Message


_CUSTOM_EMOJI_RE = re.compile(r"<a?:([A-Za-z0-9_]+):\d+>")
_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_CHANNEL_MENTION_RE = re.compile(r"<#\d+>")
_TIMESTAMP_RE = re.compile(r"<t:\d+(?::[a-zA-Z])?>")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_WS_RE = re.compile(r"[ \t\u00A0]+")
_NEWLINES_RE = re.compile(r"\n{3,}")

_EMAIL_RE = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4})"
)
_LONG_DIGITS_RE = re.compile(r"\b\d{13,19}\b")


def build_mention_table(messages: Iterable[Message]) -> dict[str, str]:
    """Build a per-folder id->display_name table so ``<@id>`` can be resolved."""
    table: dict[str, str] = {}
    for m in messages:
        if m.author_id and m.author_id not in table:
            table[m.author_id] = m.author_name
        if m.reply_to_author_id and m.reply_to_author_name:
            table.setdefault(m.reply_to_author_id, m.reply_to_author_name)
    return table


def replace_mentions(text: str, table: Mapping[str, str]) -> str:
    def _sub(match: re.Match[str]) -> str:
        uid = match.group(1)
        name = table.get(uid)
        return f"@{name}" if name else "@user"

    return _USER_MENTION_RE.sub(_sub, text)


def strip_custom_emojis(text: str) -> str:
    return _CUSTOM_EMOJI_RE.sub(r":\1:", text)


def strip_role_and_channel_mentions(text: str) -> str:
    text = _ROLE_MENTION_RE.sub("", text)
    text = _CHANNEL_MENTION_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    return text


def strip_urls(text: str) -> str:
    return _URL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def scrub_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    text = _LONG_DIGITS_RE.sub("[num]", text)
    return text


def normalize_text(
    text: str,
    mention_table: Mapping[str, str],
    *,
    url_strip: bool = False,
    pii_scrub: bool = True,
) -> str:
    """Full normalization pipeline for a single message's content."""
    if not text:
        return ""
    text = strip_custom_emojis(text)
    text = replace_mentions(text, mention_table)
    text = strip_role_and_channel_mentions(text)
    if url_strip:
        text = strip_urls(text)
    if pii_scrub:
        text = scrub_pii(text)
    text = collapse_whitespace(text)
    return text
