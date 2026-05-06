"""Cartesian grid over :func:`curate_messages` for comparing curation settings."""

from __future__ import annotations

import dataclasses
from itertools import product
from pathlib import Path
from typing import Any, Iterator

from discord_sft.data_prep.curate import CurateReport, curate_messages
from discord_sft.data_prep.ingest import iter_folders, iter_messages


def parse_csv_ints(raw: str) -> list[int]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one integer")
    return [int(p, 10) for p in parts]


def parse_csv_floats(raw: str) -> list[float]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("expected at least one float")
    return [float(p) for p in parts]


def iter_curate_sweep_rows(
    source: Path,
    *,
    session_gap_mins: list[int],
    merge_gap_secs: list[int],
    min_turns_list: list[int],
    min_authors_list: list[int],
    monologue_max_shares: list[float],
    url_strip: bool,
    pii_scrub: bool,
    lang: str | None,
    near_dedup_threshold: float | None,
    dedupe_exact_turns: bool,
    exact_turn_dup_cap: int,
) -> Iterator[dict[str, Any]]:
    """Yield one JSON-serializable dict per grid point (no sessions written to disk)."""
    source = Path(source)
    if exact_turn_dup_cap < 1:
        raise ValueError("exact_turn_dup_cap must be >= 1")

    for sgm, mgs, mt, ma, mono in product(
        session_gap_mins,
        merge_gap_secs,
        min_turns_list,
        min_authors_list,
        monologue_max_shares,
    ):
        params = {
            "session_gap_min": sgm,
            "merge_gap_sec": mgs,
            "min_turns": mt,
            "min_authors": ma,
            "monologue_max_share": mono,
            "url_strip": url_strip,
            "pii_scrub": pii_scrub,
            "lang": lang,
            "near_dedup_threshold": near_dedup_threshold,
            "dedupe_exact_turns": dedupe_exact_turns,
            "exact_turn_dup_cap": exact_turn_dup_cap,
        }
        report = CurateReport()
        sessions_kept = 0
        for folder in iter_folders(source):
            messages = list(iter_messages(folder))
            sessions, report = curate_messages(
                messages,
                folder=folder.name,
                merge_gap_sec=mgs,
                session_gap_min=sgm,
                min_turns=mt,
                min_authors=ma,
                monologue_max_share=mono,
                url_strip=url_strip,
                pii_scrub=pii_scrub,
                lang=lang,
                near_dedup_threshold=near_dedup_threshold,
                dedupe_exact_turns=dedupe_exact_turns,
                exact_turn_dup_cap=exact_turn_dup_cap,
                report=report,
            )
            sessions_kept += len(sessions)

        rep_dict = dataclasses.asdict(report)
        yield {
            "params": params,
            "sessions_kept": sessions_kept,
            "sessions_built": rep_dict.get("sessions_built", 0),
            "report": rep_dict,
        }


def default_sweep_lists(
    *,
    session_gap_min: int,
    merge_gap_sec: int,
    min_turns: int,
    min_authors: int,
    monologue_max_share: float,
    sweep_session_gap_min: str | None,
    sweep_merge_gap_sec: str | None,
    sweep_min_turns: str | None,
    sweep_min_authors: str | None,
    sweep_monologue_max_share: str | None,
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """Use explicit sweep CSV strings when set; otherwise single baseline from CLI flags."""
    sgm = (
        parse_csv_ints(sweep_session_gap_min)
        if sweep_session_gap_min
        else [session_gap_min]
    )
    mgs = parse_csv_ints(sweep_merge_gap_sec) if sweep_merge_gap_sec else [merge_gap_sec]
    mt = parse_csv_ints(sweep_min_turns) if sweep_min_turns else [min_turns]
    ma = parse_csv_ints(sweep_min_authors) if sweep_min_authors else [min_authors]
    mono = (
        parse_csv_floats(sweep_monologue_max_share)
        if sweep_monologue_max_share
        else [float(monologue_max_share)]
    )
    return sgm, mgs, mt, ma, mono
