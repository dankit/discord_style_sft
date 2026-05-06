"""Persist per-run alias/notes beside the eval workspace (UI-only, not stored in run.json)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_RUNS_KEY = "runs"
ANN_VERSION = 1


def annotations_file_path(evals_root: Path) -> Path:
    return Path(evals_root) / "run_annotations.json"


def default_annotation_doc() -> dict[str, Any]:
    return {"version": ANN_VERSION, _RUNS_KEY: {}}


def load_annotations_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return default_annotation_doc()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_annotation_doc()
    if not isinstance(raw, dict):
        return default_annotation_doc()
    runs_raw = raw.get(_RUNS_KEY)
    runs: dict[str, Any] = {}
    if isinstance(runs_raw, dict):
        for k, v in runs_raw.items():
            if not isinstance(v, dict):
                continue
            alias = str(v.get("alias", "") or "").strip()
            notes = str(v.get("notes", "") or "").strip()
            runs[str(k)] = {"alias": alias, "notes": notes}
    out = dict(raw)
    out["version"] = int(raw.get("version") or ANN_VERSION)
    out[_RUNS_KEY] = runs
    return out


def save_annotations_file(path: Path, doc: dict[str, Any]) -> None:
    normalized = dict(doc)
    normalized["version"] = int(normalized.get("version") or ANN_VERSION)
    runs_raw = normalized.get(_RUNS_KEY)
    runs: dict[str, Any] = {}
    if isinstance(runs_raw, dict):
        for k, v in runs_raw.items():
            if not isinstance(v, dict):
                continue
            runs[str(k)] = {
                "alias": str(v.get("alias", "") or "").strip(),
                "notes": str(v.get("notes", "") or "").strip(),
            }
    normalized[_RUNS_KEY] = runs
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


def get_run_annotation(doc: dict[str, Any], run_id: str) -> dict[str, str]:
    runs = doc.get(_RUNS_KEY)
    if not isinstance(runs, dict):
        return {"alias": "", "notes": ""}
    blk = runs.get(str(run_id))
    if not isinstance(blk, dict):
        return {"alias": "", "notes": ""}
    return {
        "alias": str(blk.get("alias", "") or "").strip(),
        "notes": str(blk.get("notes", "") or "").strip(),
    }


def truncate_run_id(run_id: str, *, max_chars: int = 48) -> str:
    rid = str(run_id).strip()
    if len(rid) <= max_chars:
        return rid
    return rid[: max_chars - 1] + "…"


def format_eval_run_label(
    run_id: str,
    *,
    yaml_label: str | None = None,
    elo: float | None = None,
    annotations: dict[str, Any] | None = None,
    max_run_id_chars: int = 48,
) -> str:
    """Single-line picker label: alias/YAML label/run id + optional Elo segment."""
    ann = get_run_annotation(annotations or {}, run_id)
    alias = (ann.get("alias") or "").strip()
    ylab = str(yaml_label).strip() if yaml_label else ""
    if alias:
        base = alias
    elif ylab:
        base = ylab
    else:
        base = truncate_run_id(run_id, max_chars=max_run_id_chars)

    parts: list[str] = [base]
    if elo is not None:
        parts.append(f"Elo {round(float(elo), 1)}")
    if alias or ylab:
        tail = truncate_run_id(run_id, max_chars=min(36, max_run_id_chars))
        if tail and tail != base:
            parts.append(tail)
    return " · ".join(parts)


def format_compare_column_header(
    run_id: str,
    *,
    yaml_label: str | None,
    annotations: dict[str, Any],
    max_chars: int = 32,
) -> str:
    """Shorter heading for dataframe columns — keep DF column keys as bare run_id."""
    ann = get_run_annotation(annotations, run_id)
    alias = ann.get("alias") or ""
    if alias:
        return truncate_run_id(alias, max_chars=max_chars)
    if yaml_label and str(yaml_label).strip():
        return truncate_run_id(str(yaml_label).strip(), max_chars=max_chars)
    return truncate_run_id(run_id, max_chars=max_chars)


def merge_run_annotation_rows(doc: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(doc)
    runs: dict[str, Any] = {}
    prev = out.get(_RUNS_KEY)
    if isinstance(prev, dict):
        runs = {str(k): dict(v) if isinstance(v, dict) else {} for k, v in prev.items()}
    for row in rows:
        rid = str(row.get("run_id") or "").strip()
        if not rid:
            continue
        runs[rid] = {
            "alias": str(row.get("alias") or "").strip(),
            "notes": str(row.get("notes") or "").strip(),
        }
    out[_RUNS_KEY] = runs
    out["version"] = ANN_VERSION
    return out


_SS_ANN_MTIME = "_eval_run_ann_mtime"
_SS_ANN_DOC = "_eval_run_ann_doc"


def streamlit_annotations(evals_root: Path, session_state: Any) -> dict[str, Any]:
    """Load ``run_annotations.json`` when the file mtime changes (Streamlit reruns)."""
    path = annotations_file_path(Path(evals_root))
    try:
        mtime = path.stat().st_mtime if path.is_file() else -1.0
    except OSError:
        mtime = -1.0
    if session_state.get(_SS_ANN_MTIME) != mtime:
        session_state[_SS_ANN_DOC] = load_annotations_file(path)
        session_state[_SS_ANN_MTIME] = mtime
    doc = session_state.get(_SS_ANN_DOC)
    return doc if isinstance(doc, dict) else default_annotation_doc()


def streamlit_reload_annotations(evals_root: Path, session_state: Any) -> dict[str, Any]:
    """Call after saving annotations so the next widget read sees fresh data."""
    path = annotations_file_path(Path(evals_root))
    session_state[_SS_ANN_DOC] = load_annotations_file(path)
    try:
        session_state[_SS_ANN_MTIME] = path.stat().st_mtime if path.is_file() else -1.0
    except OSError:
        session_state[_SS_ANN_MTIME] = -1.0
    return session_state[_SS_ANN_DOC]
