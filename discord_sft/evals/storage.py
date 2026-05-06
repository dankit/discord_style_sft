"""Run persistence.

One run = one JSON file at ``<out_dir>/runs/<run_id>.json``. The full
per-sample dumps that lmms-eval produces live alongside under
``<out_dir>/raw/<run_id>/`` and are referenced by ``raw_results_path``.

``out_dir`` should be the eval *root* (e.g. ``out/evals``), not a path that
already ends in ``runs`` — otherwise ``runs_dir`` / ``raw_dir`` double-nest.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def _slug(text: str) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s or "x"


def utc_now_stamp() -> str:
    """UTC timestamp safe for both filesystem paths and JSON ids."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_id_for(model_slug: str, label: str | None = None, *, stamp: str | None = None) -> str:
    """Build a deterministic, sortable run id.

    Format: ``<UTC>__<model_slug>[__<label>]``. Double underscore separates
    the three segments so downstream parsers can split unambiguously.
    """
    ts = stamp or utc_now_stamp()
    parts = [ts, _slug(model_slug)]
    if label:
        parts.append(_slug(label))
    return "__".join(parts)


def runs_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "runs"


def raw_dir(out_dir: str | Path) -> Path:
    return Path(out_dir) / "raw"


def save_run(run: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a run dict as ``<out_dir>/runs/<run_id>.json`` and return the path."""
    if "run_id" not in run:
        raise ValueError("run dict missing 'run_id'")
    dest = runs_dir(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{run['run_id']}.json"
    path.write_text(
        json.dumps(run, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_run(path_or_id: str | Path, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Load a run by filesystem path or bare run_id.

    If ``path_or_id`` is a path that exists, load it directly. Otherwise
    treat it as a run_id and look under ``<out_dir>/runs/<id>.json``.
    """
    p = Path(path_or_id)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    if out_dir is not None:
        candidate = runs_dir(out_dir) / f"{path_or_id}.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Could not resolve run: {path_or_id}")


def list_runs(dir_path: str | Path) -> list[dict[str, Any]]:
    """Return run metadata (not full scores) sorted oldest-first by ``created_utc``.

    Accepts either an ``out_dir`` (we look inside ``runs/``) or a direct
    ``runs/`` path. Unreadable files are skipped silently so a single
    corrupt JSON doesn't break the listing UI.
    """
    p = Path(dir_path)
    if not p.exists():
        return []
    if (p / "runs").is_dir():
        p = p / "runs"
    rows: list[dict[str, Any]] = []
    for f in sorted(p.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "run_id": doc.get("run_id", f.stem),
                "created_utc": doc.get("created_utc", ""),
                "label": doc.get("label"),
                "model": doc.get("model", {}),
                "tasks": (doc.get("config") or {}).get("tasks", []),
                "n_scores": len(doc.get("scores", {})),
                "path": str(f),
            }
        )
    rows.sort(key=lambda r: r.get("created_utc") or "")
    return rows


def resolve_run_paths(
    refs: list[str | Path],
    *,
    out_dir: str | Path | None = None,
) -> list[Path]:
    """Resolve a mix of filesystem paths and bare run_ids to concrete files."""
    resolved: list[Path] = []
    for ref in refs:
        p = Path(ref)
        if p.exists() and p.is_file():
            resolved.append(p)
            continue
        if out_dir is not None:
            cand = runs_dir(out_dir) / f"{ref}.json"
            if cand.exists():
                resolved.append(cand)
                continue
        raise FileNotFoundError(f"Could not resolve run: {ref}")
    return resolved
