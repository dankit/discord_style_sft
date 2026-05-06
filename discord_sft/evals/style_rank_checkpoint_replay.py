"""Recompute pairwise Elo from ``style_rank_checkpoints/*.jsonl`` and persist to ``style_rank_group_elo.json``."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from discord_sft.evals.pairwise_style import (
    _two_player_near_symmetric_priors,
    shuffle_comparison_rows_for_incremental_elo,
    two_player_elo_ratings_from_win_counts,
    update_elo_pair,
)


def _fallback_group_key(labels: frozenset[str]) -> str:
    from discord_sft.ui.pages.evals.style_rank_groups import serialize_group_key

    h = hashlib.sha256("|".join(sorted(labels)).encode()).hexdigest()[:24]
    return serialize_group_key("checkpoint", h)


def infer_style_rank_group_key(
    evals_root: Path,
    ranked_run_ids: frozenset[str],
) -> tuple[str, str]:
    """Pick the smallest compat group containing all IDs, else synthetic checkpoint-based key."""
    from discord_sft.ui.pages.evals.style_rank_groups import discover_saved_run_dumps, group_dumps

    all_dumps = discover_saved_run_dumps(Path(evals_root))
    groups = group_dumps(all_dumps)
    best_key: tuple[int, str] | None = None
    for gkey, metas in groups.items():
        member = {m.run_id for m in metas}
        if ranked_run_ids <= member:
            size = len(member)
            if best_key is None or size < best_key[0]:
                best_key = (size, str(gkey))
    if best_key is not None:
        return best_key[1], "discovered_group"
    return _fallback_group_key(ranked_run_ids), "checkpoint_hash"


def iter_checkpoint_comparison_rows(checkpoint_jsonl: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects; skip blanks and malformed lines."""
    try:
        with checkpoint_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def load_ok_comparison_rows(checkpoint_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in iter_checkpoint_comparison_rows(checkpoint_jsonl):
        if obj.get("checkpoint_kind") == "error":
            continue
        if "winner_label" not in obj or "loser_label" not in obj:
            continue
        rows.append(dict(obj))
    rows.sort(key=lambda r: (r.get("sample_key", ""), r.get("i", 0), r.get("j", 0)))
    return rows


def load_ok_comparison_rows_in_file_order(
    checkpoint_jsonl: Path,
) -> list[tuple[int, dict[str, Any]]]:
    """OK comparison rows in JSONL line order (for checkpoint-time ordering)."""
    out: list[tuple[int, dict[str, Any]]] = []
    line_no = 0
    for obj in iter_checkpoint_comparison_rows(checkpoint_jsonl):
        line_no += 1
        if obj.get("checkpoint_kind") == "error":
            continue
        if "winner_label" not in obj or "loser_label" not in obj:
            continue
        out.append((line_no, dict(obj)))
    return out


def _path_mtime_ns(path: Path) -> int:
    try:
        st = path.stat()
        mt = getattr(st, "st_mtime_ns", None)
        if isinstance(mt, int):
            return mt
        return int(st.st_mtime * 1e9)
    except OSError:
        return -1


def _comparison_row_dedupe_key(r: dict[str, Any]) -> tuple[Any, ...]:
    """Identity for de-duping reruns without dropping different run pairs on the same prompt.

    Checkpoints from different training runs share the same ``sample_key`` / ``(i, j)`` (same val row)
    but use different ``winner_label`` / ``loser_label``. Those must **not** collide; only the same
    pairwise matchup (same two run ids) should be de-duped across files, keeping the newest file.
    """
    w = str(r.get("winner_label", ""))
    l = str(r.get("loser_label", ""))
    pair = tuple(sorted((w, l)))
    return (str(r.get("sample_key", "")), int(r.get("i", 0)), int(r.get("j", 0)), pair)


def merge_ok_comparison_rows_newest_wins(
    paths_newest_first: list[Path],
) -> tuple[list[dict[str, Any]], tuple[Path, ...]]:
    """Union rows from several JSONL files.

    Same matchup ``(sample_key, i, j, {{winner_label,loser_label}})`` keeps only the **newest**
    file's row (re-runs replace older). Different run pairs on the same prompt are all kept.

    Each kept row is tagged with ``_sr_mtime_ns`` and ``_sr_line`` for chronological replay
    (oldest checkpoint / earliest line first).
    """
    seen: set[tuple[str, int, int]] = set()
    merged: list[dict[str, Any]] = []
    contributors: list[Path] = []
    contrib_set: set[str] = set()

    for path in paths_newest_first:
        mtime_ns = _path_mtime_ns(path)
        pairs = load_ok_comparison_rows_in_file_order(path)
        path_s = str(path.resolve())
        added_from_file = False
        for line_no, r in pairs:
            dk = _comparison_row_dedupe_key(r)
            if dk in seen:
                continue
            seen.add(dk)
            row = dict(r)
            row["_sr_mtime_ns"] = mtime_ns
            row["_sr_line"] = line_no
            merged.append(row)
            added_from_file = True
        if added_from_file and path_s not in contrib_set:
            contrib_set.add(path_s)
            contributors.append(path.resolve())

    return merged, tuple(contributors)


def _sort_comparison_rows_for_replay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if rows and any("_sr_mtime_ns" in r for r in rows):
        return sorted(
            rows,
            key=lambda r: (int(r.get("_sr_mtime_ns", -1)), int(r.get("_sr_line", 0))),
        )
    return sorted(
        rows,
        key=lambda r: (str(r.get("sample_key", "")), int(r.get("i", 0)), int(r.get("j", 0))),
    )


def _initial_elo_from_prior(
    labels_sorted: list[str],
    elo_start: float,
    prior: dict[str, float] | None,
) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for lab in labels_sorted:
        if prior and lab in prior:
            try:
                v = float(prior[lab])
                ratings[lab] = v if math.isfinite(v) else float(elo_start)
            except (TypeError, ValueError):
                ratings[lab] = float(elo_start)
        else:
            ratings[lab] = float(elo_start)
    return ratings


def replay_elo_from_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    elo_k: float = 32.0,
    elo_start: float = 1500.0,
    prior_elo_by_label: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Mirror :func:`~discord_sft.evals.pairwise_style.run_rank_style_eval` Elo accumulation."""
    ordered = _sort_comparison_rows_for_replay(rows)
    label_set = {str(r["winner_label"]) for r in ordered} | {str(r["loser_label"]) for r in ordered}
    labels_sorted = sorted(label_set)
    starters = _initial_elo_from_prior(labels_sorted, elo_start, prior_elo_by_label)

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    games: dict[str, int] = defaultdict(int)

    for r in ordered:
        w = str(r["winner_label"])
        l = str(r["loser_label"])
        wins[w] += 1
        losses[l] += 1
        games[w] += 1
        games[l] += 1

    has_time_prov = bool(ordered) and any("_sr_mtime_ns" in r for r in ordered)

    def _replay_incremental_updates(start: dict[str, float]) -> dict[str, float]:
        elo_rows = list(ordered)
        if not has_time_prov and len(elo_rows) >= 2:
            shuffle_comparison_rows_for_incremental_elo(
                elo_rows,
                seed=int(len(elo_rows)) & 0xFFFFFFFF,
                labels=tuple(labels_sorted),
            )
        rr = dict(start)
        for r in elo_rows:
            w = str(r["winner_label"])
            l = str(r["loser_label"])
            rw_, rl_ = rr[w], rr[l]
            nw, nl = update_elo_pair(rw_, rl_, k=elo_k)
            rr[w], rr[l] = nw, nl
        return rr

    ratings: dict[str, float]
    if len(labels_sorted) == 2:
        pair = (labels_sorted[0], labels_sorted[1])
        if _two_player_near_symmetric_priors(pair, starters):
            ratings = two_player_elo_ratings_from_win_counts(
                pair,
                wins,
                elo_start=float(elo_start),
                prior_by_label=prior_elo_by_label,
            )
        else:
            ratings = _replay_incremental_updates(starters)
    else:
        ratings = _replay_incremental_updates(starters)

    winrate: dict[str, Any] = {}
    for lab in labels_sorted:
        g = games[lab]
        winrate[lab] = (wins[lab] / g) if g else None

    elo_ordered = dict(sorted(ratings.items(), key=lambda kv: -kv[1]))
    return elo_ordered, winrate


@dataclass(frozen=True)
class CheckpointBackfillPick:
    gkey: str
    gkey_source_hint: str
    elo: dict[str, float]
    win_rate: dict[str, Any]
    ranked_run_ids: list[str]
    n_comparisons: int
    source_checkpoint_paths: tuple[Path, ...]
    checkpoint_mtime: float
    checkpoint_mtime_ns: int

    @property
    def checkpoint_path(self) -> Path:
        return self.source_checkpoint_paths[0]


def replay_checkpoint_file(
    path: Path,
    evals_root: Path,
    *,
    elo_k: float = 32.0,
    elo_start: float = 1500.0,
) -> CheckpointBackfillPick | None:
    pairs = load_ok_comparison_rows_in_file_order(path)
    if len(pairs) < 1:
        return None
    mtime_ns = _path_mtime_ns(path)
    rows: list[dict[str, Any]] = []
    for line_no, r in pairs:
        row = dict(r)
        row["_sr_mtime_ns"] = mtime_ns
        row["_sr_line"] = line_no
        rows.append(row)
    elo, wr = replay_elo_from_comparison_rows(rows, elo_k=elo_k, elo_start=elo_start)
    label_set = frozenset(
        str(r["winner_label"]) for r in rows
    ) | frozenset(str(r["loser_label"]) for r in rows)
    ids_sorted = sorted(label_set)
    try:
        st = path.stat()
        mtime = float(st.st_mtime)
        mtime_ns = int(getattr(st, "st_mtime_ns", round(st.st_mtime * 1e9)))
    except OSError:
        mtime = 0.0
        mtime_ns = -1
    gkey, hint = infer_style_rank_group_key(Path(evals_root), label_set)
    rp = path.resolve()
    return CheckpointBackfillPick(
        gkey=gkey,
        gkey_source_hint=hint,
        elo=dict(elo),
        win_rate=dict(wr),
        ranked_run_ids=ids_sorted,
        n_comparisons=len(rows),
        source_checkpoint_paths=(rp,),
        checkpoint_mtime=mtime,
        checkpoint_mtime_ns=mtime_ns,
    )


def backfill_elo_store_from_checkpoints_dir(
    evals_root: Path,
    checkpoints_dir: Path | None = None,
    *,
    elo_k: float = 32.0,
    elo_start: float = 1500.0,
    merge_existing: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Scan JSONL checkpoints; rows for the same group key are merged and replayed once.

    Files are processed **newest mtime first**. For the same inferred ``gkey``, all JSONL rows
    are unioned; duplicate matchups ``(sample_key, i, j, same winner/loser labels)`` keep the
    **newest** file's row so overlapping re-runs do not double-count. Different checkpoints
    that share the same prompts but compare **different** run ids (e.g. epoch-1 vs epoch-4) are
    kept separately and all contribute to the merged Elo table when they map to the same group.
    """

    from discord_sft.ui.pages.evals.style_rank_elo_store import (
        default_elo_store,
        elo_group_store_path,
        get_group_elo_and_winrate,
        load_elo_store,
        save_elo_store,
        upsert_group_elo,
    )

    root = Path(evals_root)
    ck_dir = Path(checkpoints_dir) if checkpoints_dir is not None else root / "style_rank_checkpoints"
    store_path = elo_group_store_path(root)

    if not ck_dir.is_dir():
        return (
            {
                "ok": True,
                "message": "checkpoints directory missing or not a directory",
                "checkpoints_dir": str(ck_dir.resolve()),
                "groups_written": [],
            },
            store_path,
        )

    def _path_sort_key(p: Path) -> tuple[int, str]:
        try:
            st = p.stat()
            mt = getattr(st, "st_mtime_ns", None)
            if isinstance(mt, int):
                kind = mt
            else:
                kind = int(st.st_mtime * 1e9)
        except OSError:
            kind = -1
        return (kind, str(p.resolve()))

    jsonl_paths = [p for p in ck_dir.glob("*.jsonl") if p.is_file()]
    # Newest checkpoints first — avoids same-second mtimes overwriting with an older sibling.
    jsonl_paths.sort(key=_path_sort_key, reverse=True)

    by_gkey_paths: dict[str, list[Path]] = defaultdict(list)
    by_gkey_hint: dict[str, str] = {}
    for path in jsonl_paths:
        rows = load_ok_comparison_rows(path)
        if len(rows) < 1:
            continue
        label_set = frozenset(
            str(r["winner_label"]) for r in rows
        ) | frozenset(str(r["loser_label"]) for r in rows)
        gkey, hint = infer_style_rank_group_key(root, label_set)
        by_gkey_paths[gkey].append(path)
        by_gkey_hint.setdefault(gkey, hint)

    doc = load_elo_store(store_path) if merge_existing else default_elo_store()

    by_gkey: dict[str, CheckpointBackfillPick] = {}
    for gkey, paths in by_gkey_paths.items():
        merged_rows, contributors = merge_ok_comparison_rows_newest_wins(paths)
        if len(merged_rows) < 1:
            continue
        prior_elo_map: dict[str, float] | None = None
        if merge_existing:
            pe, _wr0 = get_group_elo_and_winrate(doc, gkey)
            if pe:
                prior_elo_map = pe
        elo, wr = replay_elo_from_comparison_rows(
            merged_rows,
            elo_k=elo_k,
            elo_start=elo_start,
            prior_elo_by_label=prior_elo_map,
        )
        label_set = frozenset(
            str(r["winner_label"]) for r in merged_rows
        ) | frozenset(str(r["loser_label"]) for r in merged_rows)
        ids_sorted = sorted(label_set)
        mtimes_ns: list[int] = []
        mtimes: list[float] = []
        for p in contributors:
            try:
                st = p.stat()
                mtimes.append(float(st.st_mtime))
                mtimes_ns.append(int(getattr(st, "st_mtime_ns", round(st.st_mtime * 1e9))))
            except OSError:
                pass
        top_mtime_ns = max(mtimes_ns) if mtimes_ns else -1
        top_mtime = max(mtimes) if mtimes else 0.0
        by_gkey[gkey] = CheckpointBackfillPick(
            gkey=gkey,
            gkey_source_hint=by_gkey_hint.get(gkey, "checkpoint_hash"),
            elo=dict(elo),
            win_rate=dict(wr),
            ranked_run_ids=ids_sorted,
            n_comparisons=len(merged_rows),
            source_checkpoint_paths=contributors,
            checkpoint_mtime=top_mtime,
            checkpoint_mtime_ns=top_mtime_ns,
        )

    applied: list[dict[str, Any]] = []
    for _gkey, pick in sorted(by_gkey.items(), key=lambda kv: kv[0]):
        doc = upsert_group_elo(
            doc,
            pick.gkey,
            elo=pick.elo,
            win_rate=pick.win_rate,
            ranked_run_ids=pick.ranked_run_ids,
        )
        cps = [str(p) for p in pick.source_checkpoint_paths]
        applied.append(
            {
                "gkey": pick.gkey,
                "hint": pick.gkey_source_hint,
                "checkpoint": str(pick.checkpoint_path),
                "checkpoints": cps,
                "n_checkpoints_merged": len(cps),
                "mtime": pick.checkpoint_mtime,
                "mtime_ns": pick.checkpoint_mtime_ns,
                "n_comparisons": pick.n_comparisons,
                "run_ids": pick.ranked_run_ids,
            }
        )

    if applied:
        save_elo_store(store_path, doc)

    return (
        {
            "ok": True,
            "checkpoints_dir": str(ck_dir.resolve()),
            "store_path": str(store_path.resolve()),
            "groups_written": applied,
            "n_files_seen": sum(1 for p in jsonl_paths if p.is_file()),
        },
        store_path,
    )
