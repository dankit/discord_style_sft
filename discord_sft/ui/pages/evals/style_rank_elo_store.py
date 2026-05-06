"""Persist last pairwise Elo / win-rate per val-compatibility group for Style rank UI."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_GROUPS_KEY = "groups"
STORE_VERSION = 1


def elo_group_store_path(evals_root: Path) -> Path:
    return Path(evals_root) / "style_rank_group_elo.json"


def default_elo_store() -> dict[str, Any]:
    return {"version": STORE_VERSION, _GROUPS_KEY: {}}


def _normalize_elo_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_elo_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return default_elo_store()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_elo_store()
    if not isinstance(raw, dict):
        return default_elo_store()
    groups_raw = raw.get(_GROUPS_KEY)
    groups: dict[str, Any] = {}
    if isinstance(groups_raw, dict):
        for gk, gv in groups_raw.items():
            if not isinstance(gv, dict):
                continue
            groups[str(gk)] = {
                "updated_utc": str(gv.get("updated_utc") or ""),
                "elo": _normalize_elo_map(gv.get("elo")),
                "win_rate": gv.get("win_rate")
                if isinstance(gv.get("win_rate"), dict)
                else {},
                "ranked_run_ids": [
                    str(x) for x in gv.get("ranked_run_ids") or []
                    if isinstance(x, (str, int))
                ]
                if gv.get("ranked_run_ids") is not None
                else [],
            }
    out = dict(raw)
    out["version"] = int(raw.get("version") or STORE_VERSION)
    out[_GROUPS_KEY] = groups
    return out


def save_elo_store(path: Path, doc: dict[str, Any]) -> None:
    normalized = dict(doc)
    normalized["version"] = int(normalized.get("version") or STORE_VERSION)
    groups_raw = normalized.get(_GROUPS_KEY)
    groups: dict[str, Any] = {}
    if isinstance(groups_raw, dict):
        for gk, gv in groups_raw.items():
            if not isinstance(gv, dict):
                continue
            groups[str(gk)] = {
                "updated_utc": str(gv.get("updated_utc") or ""),
                "elo": _normalize_elo_map(gv.get("elo")),
                "win_rate": gv.get("win_rate")
                if isinstance(gv.get("win_rate"), dict)
                else {},
                "ranked_run_ids": [
                    str(x) for x in (gv.get("ranked_run_ids") or []) if isinstance(x, (str, int))
                ],
            }
    normalized[_GROUPS_KEY] = groups
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def upsert_group_elo(
    doc: dict[str, Any],
    gkey: str,
    *,
    elo: dict[str, float],
    win_rate: dict[str, Any],
    ranked_run_ids: list[str],
) -> dict[str, Any]:
    out = dict(doc)
    grp: dict[str, Any] = dict(out.get(_GROUPS_KEY) or {}) if isinstance(out.get(_GROUPS_KEY), dict) else {}
    grp[str(gkey)] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elo": {str(k): float(v) for k, v in elo.items()},
        "win_rate": {str(k): v for k, v in win_rate.items()},
        "ranked_run_ids": list(ranked_run_ids),
    }
    out[_GROUPS_KEY] = grp
    out["version"] = STORE_VERSION
    return out


def get_group_elo_and_winrate(doc: dict[str, Any], gkey: str) -> tuple[dict[str, float], dict[str, Any]]:
    grp = doc.get(_GROUPS_KEY)
    if not isinstance(grp, dict):
        return {}, {}
    blk = grp.get(str(gkey))
    if not isinstance(blk, dict):
        return {}, {}
    elo = _normalize_elo_map(blk.get("elo"))
    wr_raw = blk.get("win_rate")
    wr = {str(k): v for k, v in wr_raw.items()} if isinstance(wr_raw, dict) else {}
    return elo, wr


_SS_ELO_MT = "_style_rank_elo_store_mtime"
_SS_ELO_DOC = "_style_rank_elo_store_doc"


def streamlit_elo_store(evals_root: Path, session_state: Any) -> dict[str, Any]:
    path = elo_group_store_path(Path(evals_root))
    try:
        mtime = path.stat().st_mtime if path.is_file() else -1.0
    except OSError:
        mtime = -1.0
    if session_state.get(_SS_ELO_MT) != mtime:
        session_state[_SS_ELO_DOC] = load_elo_store(path)
        session_state[_SS_ELO_MT] = mtime
    d = session_state.get(_SS_ELO_DOC)
    return d if isinstance(d, dict) else default_elo_store()


def streamlit_reload_elo_store(evals_root: Path, session_state: Any) -> dict[str, Any]:
    path = elo_group_store_path(Path(evals_root))
    session_state[_SS_ELO_DOC] = load_elo_store(path)
    try:
        session_state[_SS_ELO_MT] = path.stat().st_mtime if path.is_file() else -1.0
    except OSError:
        session_state[_SS_ELO_MT] = -1.0
    return session_state[_SS_ELO_DOC]


def persist_rank_outcome_for_group(
    evals_root: Path,
    session_state: Any,
    gkey: str,
    report: dict[str, Any],
    ranked_run_ids: list[str],
) -> Path:
    """Write Elo/win-rate for this group; refresh Streamlit cache row."""
    path = elo_group_store_path(Path(evals_root))
    doc = load_elo_store(path)
    pw = report.get("pairwise") if isinstance(report, dict) else None
    pw_dict = pw if isinstance(pw, dict) else {}
    elo_raw = pw_dict.get("elo") or {}
    elo = _normalize_elo_map(elo_raw)
    wr_raw = pw_dict.get("win_rate")
    wr: dict[str, Any] = {str(k): v for k, v in wr_raw.items()} if isinstance(wr_raw, dict) else {}
    merged = upsert_group_elo(doc, gkey, elo=elo, win_rate=wr, ranked_run_ids=ranked_run_ids)
    save_elo_store(path, merged)
    streamlit_reload_elo_store(Path(evals_root), session_state)
    return path
