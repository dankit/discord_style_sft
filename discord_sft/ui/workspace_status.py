"""Filesystem-backed workspace metrics for the Streamlit home dashboard.

All summaries are derived from default artifact paths under the working
directory so reopening the app shows the same state as the on-disk pipeline.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from discord_sft.evals.storage import list_runs


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def summarize_messages_dir(messages_dir: Path) -> dict[str, Any] | None:
    """Default ingest output: ``<work>/messages`` with one subfolder per DM."""
    if not messages_dir.is_dir():
        return None
    subdirs = [p for p in messages_dir.iterdir() if p.is_dir()]
    n_parquet = 0
    n_jsonl = 0
    data_bytes = 0
    for d in subdirs:
        try:
            for f in d.iterdir():
                if not f.is_file():
                    continue
                data_bytes += f.stat().st_size
                if f.suffix.lower() == ".parquet":
                    n_parquet += 1
                elif f.suffix.lower() == ".jsonl":
                    n_jsonl += 1
        except OSError:
            continue
    return {
        "dm_folders": len(subdirs),
        "parquet_files": n_parquet,
        "jsonl_files": n_jsonl,
        "data_bytes": data_bytes,
        "data_size": _fmt_bytes(data_bytes),
    }


def aggregate_sessions_jsonl(path: Path) -> dict[str, Any] | None:
    """One pass over ``sessions.jsonl`` for corpus-scale stats."""
    if not path.is_file():
        return None
    sessions = 0
    turns = 0
    words = 0
    authors: set[str] = set()
    folders: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sessions += 1
            folders.add(str(rec.get("folder") or ""))
            for aid in rec.get("authors") or []:
                authors.add(str(aid))
            for t in rec.get("turns") or []:
                turns += 1
                txt = t.get("text") or ""
                if isinstance(txt, str):
                    words += len(txt.split())
    return {
        "sessions": sessions,
        "turns": turns,
        "words": words,
        "distinct_authors": len(authors),
        "distinct_folders": len([x for x in folders if x]),
    }


def aggregate_sft_jsonl(path: Path) -> dict[str, Any] | None:
    """Stream one ShareGPT JSONL (train or val) for sample and persona stats."""
    if not path.is_file():
        return None
    samples = 0
    by_persona: Counter[str] = Counter()
    by_name: dict[str, str] = {}
    sessions: set[str] = set()
    turns_with_meta = 0
    turns_sum = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples += 1
            meta = d.get("meta") or {}
            pid = str(meta.get("persona_id") or "")
            if pid:
                by_persona[pid] += 1
                nm = meta.get("persona_name")
                if isinstance(nm, str) and nm and pid not in by_name:
                    by_name[pid] = nm
            sid = meta.get("session_id")
            if sid is not None and str(sid):
                sessions.add(str(sid))
            nt = meta.get("num_turns")
            if isinstance(nt, int) and nt > 0:
                turns_sum += nt
                turns_with_meta += 1
    avg_turns = (turns_sum / turns_with_meta) if turns_with_meta else None
    top_personas = by_persona.most_common(12)
    return {
        "samples": samples,
        "distinct_sessions": len(sessions),
        "distinct_personas": len(by_persona),
        "avg_turns_per_sample": round(avg_turns, 2) if avg_turns is not None else None,
        "persona_top": [
            {"persona_id": pid, "persona_name": by_name.get(pid, ""), "samples": n}
            for pid, n in top_personas
        ],
    }


def summarize_profiles(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    personas = doc.get("personas") or {}
    if not isinstance(personas, dict):
        return None
    n = len(personas)
    file_bytes = path.stat().st_size
    return {
        "personas": n,
        "file_bytes": file_bytes,
        "file_size": _fmt_bytes(file_bytes),
    }


def summarize_evals(work: Path) -> dict[str, Any]:
    root = work / "evals"
    rows = list_runs(root)
    if not rows:
        return {"runs": 0}
    last = rows[-1]
    return {
        "runs": len(rows),
        "latest_created_utc": last.get("created_utc") or "",
        "latest_run_id": last.get("run_id") or "",
        "latest_label": last.get("label"),
    }


def load_curate_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_balance_report(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# Ordered gates for “how far along is this workspace?” (strict from start).
PIPELINE_STEPS: list[tuple[str, str]] = [
    ("ingest", "Ingest"),
    ("curate", "Curate"),
    ("sft", "Build SFT"),
    ("fingerprint", "Fingerprint"),
    ("evals", "Evals"),
]

PIPELINE_KEYS = [k for k, _ in PIPELINE_STEPS]


def effective_pipeline_flags(flags: dict[str, bool]) -> dict[str, bool]:
    """Derive a consecutive pipeline view from on-disk flags under the working directory.

    If ``curated/``, ``sft/``, etc. exist but ``messages/`` does not (deleted, or
    ingest ran elsewhere), we still treat **later outputs as proof** that earlier
    stages logically completed — including when the **Raw Discord export root**
    is set: progress is always driven by the working tree, not by whether
    ``messages/`` is still present.
    """
    out = dict(flags)
    last_true = -1
    for i, k in enumerate(PIPELINE_KEYS):
        if out.get(k):
            last_true = i
    if last_true < 0:
        return out
    for j in range(last_true + 1):
        out[PIPELINE_KEYS[j]] = True
    return out


def _first_consecutive_gap(flags: dict[str, bool]) -> int:
    for i, (key, _) in enumerate(PIPELINE_STEPS):
        if not flags.get(key):
            return i
    return len(PIPELINE_STEPS)


def pipeline_completion(flags: dict[str, bool], *, raw_flags: dict[str, bool]) -> dict[str, Any]:
    """Progress uses *effective* flags; out-of-order warnings use *raw* on-disk flags."""
    n_total = len(PIPELINE_STEPS)
    first_gap = _first_consecutive_gap(flags)
    n_done = first_gap
    frac = (n_done / n_total) if n_total else 0.0
    first_gap_raw = _first_consecutive_gap(raw_flags)
    chain_broken = False
    if first_gap_raw < n_total:
        # Missing only ``messages/`` while later paths exist is normal — not "broken".
        if first_gap_raw == 0 and any(
            raw_flags.get(PIPELINE_STEPS[j][0]) for j in range(1, n_total)
        ):
            chain_broken = False
        else:
            for j in range(first_gap_raw + 1, n_total):
                k, _ = PIPELINE_STEPS[j]
                if raw_flags.get(k):
                    chain_broken = True
                    break
    if n_done == 0:
        title = "Not started"
        body = (
            "The working directory has no ``messages/``, ``curated/``, or ``sft/`` outputs yet "
            "under the default layout. Set **Working directory** to your ``out`` folder (sidebar), "
            "point **Raw Discord export root** at Discrub JSON, then run **Ingest**."
        )
    elif n_done == n_total:
        title = PIPELINE_STEPS[-1][1]
        body = (
            "Artifacts are present through **Evals** (saved JSON under **evals/runs/**). "
            "You can still add data or re-run earlier stages if you change settings."
        )
    elif n_done == 4 and not flags.get("evals"):
        # Ingest → Fingerprint all satisfied; only optional saved eval runs missing.
        title = "Ready for training"
        body = (
            "**sft/train.jsonl**, **val.jsonl**, and **profiles.json** are in place. "
            "You can fine-tune in your stack of choice. **Evals** (sidebar) is optional: "
            "run benchmarks and store results under **evals/runs/** when you want model QA."
        )
    else:
        done_name = PIPELINE_STEPS[n_done - 1][1]
        next_name = PIPELINE_STEPS[n_done][1]
        title = done_name
        body = (
            f"Latest complete step in the default layout: **{done_name}**. "
            f"Typical next step: **{next_name}**."
        )
    if chain_broken:
        body += (
            " **Note:** on disk, a later stage has files while an earlier default path is missing — "
            "custom output locations or deleted intermediates can cause that."
        )
    return {
        "completed_consecutive": n_done,
        "total": n_total,
        "progress_fraction": frac,
        "status_title": title,
        "status_body": body,
        "chain_broken": chain_broken,
    }


def pipeline_flags(work: Path) -> dict[str, bool]:
    """Which default pipeline stages appear complete on disk."""
    messages = work / "messages"
    has_ingest = messages.is_dir() and any(p.is_dir() for p in messages.iterdir())
    sess = work / "curated" / "sessions.jsonl"
    has_curate = sess.is_file() and sess.stat().st_size > 0
    train = work / "sft" / "train.jsonl"
    has_sft = train.is_file() and train.stat().st_size > 0
    prof = work / "sft" / "profiles.json"
    has_profiles = prof.is_file() and prof.stat().st_size > 0
    runs = work / "evals" / "runs"
    has_evals = runs.is_dir() and any(runs.glob("*.json"))
    return {
        "ingest": bool(has_ingest),
        "curate": bool(has_curate),
        "sft": bool(has_sft),
        "fingerprint": bool(has_profiles),
        "evals": bool(has_evals),
    }


def build_snapshot(work: Path) -> dict[str, Any]:
    """Aggregate everything the home dashboard needs (call from UI with caching)."""
    flags = pipeline_flags(work)
    effective = effective_pipeline_flags(flags)
    snap: dict[str, Any] = {
        "work": str(work.resolve()),
        "flags": flags,
        "pipeline": pipeline_completion(effective, raw_flags=flags),
    }
    snap["messages"] = summarize_messages_dir(work / "messages")
    sess_path = work / "curated" / "sessions.jsonl"
    snap["sessions"] = aggregate_sessions_jsonl(sess_path)
    snap["curate_report"] = load_curate_report(work / "curated" / "curate_report.json")
    train_p = work / "sft" / "train.jsonl"
    val_p = work / "sft" / "val.jsonl"
    snap["train"] = aggregate_sft_jsonl(train_p)
    snap["val"] = aggregate_sft_jsonl(val_p)
    snap["balance_report"] = load_balance_report(work / "sft" / "balance_report.json")
    snap["profiles"] = summarize_profiles(work / "sft" / "profiles.json")
    snap["evals"] = summarize_evals(work)
    return snap
