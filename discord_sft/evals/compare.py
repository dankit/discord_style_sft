"""Compare two or more saved runs.

Pure-Python so the core install stays dependency-free. We deliberately do
not take a pandas dependency here; the UI layer converts our list-of-dicts
to a ``pandas.DataFrame`` when available.

Terminology
-----------
- ``delta``: signed absolute difference ``value - baseline``.
- ``forgetting_ratio``: ``(baseline - value) / baseline`` for the baseline
  run. Positive means the candidate regressed; negative means it improved.
  Undefined (``None``) when ``baseline == 0`` or ``baseline`` is missing.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Iterable

from discord_sft.evals.storage import load_run, resolve_run_paths


def _flatten_scores(run: dict[str, Any]) -> dict[str, float]:
    """Accept both flat (dotted keys) and nested scores dicts.

    ``scores`` is specified as a flat dict in our schema, but we defensively
    flatten one level in case persona results were inserted as nested dicts.
    """
    out: dict[str, float] = {}
    scores = run.get("scores") or {}
    for k, v in scores.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float)):
                    out[f"{k}.{sub_k}"] = float(sub_v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _match_metrics(all_keys: Iterable[str], patterns: list[str] | None) -> list[str]:
    keys = sorted(all_keys)
    if not patterns:
        return keys
    picked: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for k in keys:
            if k in seen:
                continue
            if fnmatch.fnmatchcase(k, pat):
                picked.append(k)
                seen.add(k)
    return picked


def compare_runs(
    refs: list[str | Path],
    *,
    baseline: int | str | Path | None = 0,
    metrics: list[str] | None = None,
    out_dir: str | Path | None = None,
    omit_persona_metrics: bool = True,
) -> dict[str, Any]:
    """Load runs and return a comparison report.

    Parameters
    ----------
    refs:
        Run paths or run_ids to compare. Order determines column order.
    baseline:
        Either the integer index into ``refs`` (default: 0 = first run),
        or a run path / id explicitly. Pass ``None`` to omit delta columns.
    metrics:
        Optional list of glob patterns (e.g. ``"ifeval.*"``,
        ``"*.forgetting"``). Default: show every metric present in any run.
    out_dir:
        Fallback directory for resolving bare run_ids.
    omit_persona_metrics:
        When ``metrics`` is ``None`` (no glob filter), drop score keys that
        start with ``persona.`` so benchmark tables stay readable. Ignored
        when ``metrics`` is set — your globs then control inclusion exactly.
        Pass ``False`` to list every key when ``metrics`` is ``None``.

    Returns
    -------
    dict with keys ``runs`` (list of run header dicts), ``rows`` (one per
    metric, with the value under each run_id, plus delta/forgetting_ratio
    columns when a baseline is supplied), and ``baseline_run_id``.
    """
    paths = resolve_run_paths(refs, out_dir=out_dir)
    runs = [load_run(p) for p in paths]
    if not runs:
        return {"runs": [], "rows": [], "baseline_run_id": None}

    run_ids = [r.get("run_id", p.stem) for r, p in zip(runs, paths)]
    all_scores = [_flatten_scores(r) for r in runs]

    baseline_idx: int | None
    baseline_scores: dict[str, float] | None
    baseline_run_id: str | None
    if baseline is None:
        baseline_idx = None
        baseline_scores = None
        baseline_run_id = None
    elif isinstance(baseline, int):
        if not 0 <= baseline < len(runs):
            raise IndexError(f"baseline index {baseline} out of range")
        baseline_idx = baseline
        baseline_scores = all_scores[baseline]
        baseline_run_id = run_ids[baseline]
    else:
        baseline_run_id = str(baseline)
        for i, rid in enumerate(run_ids):
            if rid == baseline_run_id or str(paths[i]) == baseline_run_id:
                baseline_idx = i
                baseline_scores = all_scores[i]
                break
        else:
            baseline_run = load_run(baseline, out_dir=out_dir)
            baseline_scores = _flatten_scores(baseline_run)
            baseline_run_id = baseline_run.get("run_id", str(baseline))
            baseline_idx = None

    all_keys: set[str] = set()
    for s in all_scores:
        all_keys.update(s.keys())
    if baseline_scores:
        all_keys.update(baseline_scores.keys())
    keys = _match_metrics(all_keys, metrics)
    if omit_persona_metrics and metrics is None:
        keys = [k for k in keys if not k.startswith("persona.")]

    rows: list[dict[str, Any]] = []
    for k in keys:
        row: dict[str, Any] = {"metric": k}
        for rid, sc in zip(run_ids, all_scores):
            row[rid] = sc.get(k)
        if baseline_scores is not None and baseline_idx is not None:
            base_val = baseline_scores.get(k)
            for i, rid in enumerate(run_ids):
                if i == baseline_idx:
                    continue
                cand = all_scores[i].get(k)
                row[f"delta__{rid}"] = (
                    (cand - base_val) if (base_val is not None and cand is not None) else None
                )
                if base_val is None or cand is None or base_val == 0:
                    row[f"forgetting__{rid}"] = None
                else:
                    row[f"forgetting__{rid}"] = (base_val - cand) / base_val
        rows.append(row)

    run_headers = [
        {
            "run_id": rid,
            "label": r.get("label"),
            "created_utc": r.get("created_utc"),
            "model": r.get("model", {}),
            "is_baseline": baseline_idx is not None and i == baseline_idx,
        }
        for i, (rid, r) in enumerate(zip(run_ids, runs))
    ]
    return {
        "runs": run_headers,
        "rows": rows,
        "baseline_run_id": baseline_run_id,
    }


def render_comparison(report: dict[str, Any], *, fmt: str = "table") -> str:
    """Render ``compare_runs`` output as a string.

    ``fmt``:
        - ``table``: fixed-width ASCII, human-friendly.
        - ``markdown``: GitHub-flavoured table.
        - ``json``: indented JSON dump of the whole report.
    """
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    runs = report["runs"]
    rows = report["rows"]
    if not runs or not rows:
        return "(no runs to compare)"

    run_ids = [r["run_id"] for r in runs]
    baseline_id = report.get("baseline_run_id")
    headers = ["metric"] + run_ids
    if baseline_id:
        for rid in run_ids:
            if rid == baseline_id:
                continue
            headers.append(f"Δ {rid}")
            headers.append(f"forget% {rid}")

    def fmt_val(v: Any, *, pct: bool = False) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v * 100:+.1f}%" if pct else f"{v:.4f}"
        return str(v)

    table: list[list[str]] = [headers]
    for row in rows:
        cells = [row["metric"]]
        cells.extend(fmt_val(row.get(rid)) for rid in run_ids)
        if baseline_id:
            for rid in run_ids:
                if rid == baseline_id:
                    continue
                cells.append(fmt_val(row.get(f"delta__{rid}")))
                cells.append(fmt_val(row.get(f"forgetting__{rid}"), pct=True))
        table.append(cells)

    if fmt == "markdown":
        lines = [
            "| " + " | ".join(table[0]) + " |",
            "| " + " | ".join("---" for _ in table[0]) + " |",
        ]
        for row in table[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    widths = [max(len(r[i]) for r in table) for i in range(len(headers))]
    lines = []
    for i, row in enumerate(table):
        lines.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
