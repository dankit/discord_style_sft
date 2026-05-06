"""Discover persona_generations.jsonl dumps and group by eval val compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from discord_sft.evals.storage import list_runs, load_run
from discord_sft.ui.persona_compare import sample_key

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_artifact_path(raw: str | Path) -> Path:
    p = Path(str(raw).strip()).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_REPO_ROOT / p).resolve()


def peek_first_jsonl_row(path: Path) -> dict[str, Any] | None:
    """Read the first non-blank JSON object from a JSONL file."""
    first, _fp, _n = scan_persona_generations_jsonl(path)
    return first


def scan_persona_generations_jsonl(path: Path) -> tuple[dict[str, Any] | None, str, int]:
    """Read jsonl once: first row, fingerprint of the eval row-set, line count.

    The fingerprint is ``sha256`` of sorted ``sample_key`` values (same alignment as
    persona compare). Different ``val.jsonl`` slices produce different fingerprints
    even when ``run.json`` still says ``out/sft/val.jsonl`` for every run.
    """
    first: dict[str, Any] | None = None
    keys: list[str] = []
    n = 0
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                n += 1
                if first is None:
                    first = obj
                keys.append(sample_key(obj))
    except OSError:
        return None, "", 0
    keys.sort()
    blob = "\n".join(keys).encode("utf-8")
    fp = hashlib.sha256(blob).hexdigest() if keys else ""
    return first, fp, n


def _resolve_val_path(raw: str) -> str:
    """Normalize val.jsonl path the same way as generation artifacts (repo-rooted)."""
    p = Path(str(raw).strip()).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    try:
        return str(p.resolve(strict=False))
    except OSError:
        return str(p)


def _manifest_val_hint(run_doc: dict[str, Any] | None) -> str:
    cfg = (run_doc or {}).get("config") if run_doc else None
    if not isinstance(cfg, dict):
        return ""
    vj = cfg.get("val_jsonl")
    if not vj:
        return ""
    try:
        rs = _resolve_val_path(str(vj))
        return Path(rs).name
    except Exception:
        return str(vj)


def compat_group_key(
    run_doc: dict[str, Any] | None,
    first_row: dict[str, Any] | None,
    *,
    generations_path: Path,
    eval_set_fingerprint: str,
    n_rows: int,
) -> tuple[str, str, str]:
    """Return (kind, stable_value, display_hint) for val grouping.

    1. **Embedded** ``eval_val_sha256`` / ``eval_val_jsonl`` on rows (truth at write time).
    2. **Eval slice fingerprint** — hash of sorted ``sample_key`` across all rows. Splits
       ``out/sft/val.jsonl`` vs ``out/sft/old/val.jsonl`` when manifests all list the same path.
    3. ``config.val_jsonl`` from ``run.json`` (weak; many runs share an outdated path).
    4. **unknown** per file path.
    """
    if first_row:
        h = str(first_row.get("eval_val_sha256") or "").strip()
        if h:
            return ("valsha", h, f"val file sha256:{h[:12]}…")
        evj = str(first_row.get("eval_val_jsonl") or "").strip()
        if evj:
            rs = _resolve_val_path(evj)
            return ("path", rs, f"{Path(rs).name} ({rs})")

    if eval_set_fingerprint and n_rows > 0:
        man = _manifest_val_hint(run_doc)
        tail = f"{eval_set_fingerprint[:10]}…"
        hint = f"{n_rows} prompts · slice {tail}"
        if man:
            hint = f"manifest {man} · {hint}"
        return ("evalset", eval_set_fingerprint, hint)

    cfg = (run_doc or {}).get("config") if run_doc else None
    if isinstance(cfg, dict):
        vj = cfg.get("val_jsonl")
        if vj:
            rs = _resolve_val_path(str(vj))
            hint = f"{Path(rs).name} ({rs})"
            return ("manifest_path", rs, hint)

    digest = hashlib.sha256(str(generations_path).encode("utf-8")).hexdigest()[:16]
    return ("unknown", digest, f"unknown · {generations_path.name}")


def serialize_group_key(kind: str, value: str) -> str:
    return f"{kind}\x1f{value}"


def infer_run_json_from_generations_path(gens: Path) -> Path | None:
    """If ``…/evals/raw/<run_id>/persona_generations.jsonl``, return ``…/evals/runs/<run_id>.json``."""
    try:
        run_folder = gens.parent
        raw_folder = run_folder.parent
        if raw_folder.name != "raw":
            return None
        eval_root = raw_folder.parent
        rid = run_folder.name
        cand = eval_root / "runs" / f"{rid}.json"
        return cand if cand.is_file() else None
    except (OSError, ValueError):
        return None


@dataclass
class PersonaDumpMeta:
    """One rankable persona_generations.jsonl source."""

    run_id: str
    label: str
    generations_path: Path
    run_json_path: Path | None
    created_utc: str
    group_kind: str
    group_value: str
    display_hint: str
    serialized_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.serialized_key = serialize_group_key(self.group_kind, self.group_value)


def discover_saved_run_dumps(evals_root: Path) -> list[PersonaDumpMeta]:
    """Scan ``list_runs`` for entries with ``persona.generations_path``."""
    out: list[PersonaDumpMeta] = []
    for meta in list_runs(evals_root):
        run_path = Path(str(meta.get("path") or ""))
        if not run_path.is_file():
            continue
        try:
            doc = load_run(run_path)
        except (OSError, json.JSONDecodeError, FileNotFoundError):
            continue
        raw_gp = (doc.get("persona") or {}).get("generations_path")
        if not raw_gp:
            continue
        gens = _resolve_artifact_path(str(raw_gp))
        if not gens.is_file():
            continue
        first, slice_fp, n_rows = scan_persona_generations_jsonl(gens)
        kind, value, hint = compat_group_key(
            doc,
            first,
            generations_path=gens,
            eval_set_fingerprint=slice_fp,
            n_rows=n_rows,
        )
        rid = str(doc.get("run_id") or meta.get("run_id") or run_path.stem)
        label = str(doc.get("label") or rid)
        rj = run_path.resolve()
        out.append(
            PersonaDumpMeta(
                run_id=rid,
                label=label,
                generations_path=gens.resolve(),
                run_json_path=rj,
                created_utc=str(meta.get("created_utc") or doc.get("created_utc") or ""),
                group_kind=kind,
                group_value=value,
                display_hint=hint,
            )
        )
    return out


def parse_manual_dump_lines(
    path_lines: list[str],
    run_json_lines: list[str | None],
) -> list[PersonaDumpMeta]:
    """Build metas from pasted paths (optional parallel run.json paths)."""
    out: list[PersonaDumpMeta] = []
    for idx, pl in enumerate(path_lines):
        gens = _resolve_artifact_path(pl)
        if not gens.is_file():
            continue
        first, slice_fp, n_rows = scan_persona_generations_jsonl(gens)
        run_doc: dict[str, Any] | None = None
        rjp: Path | None = None
        if idx < len(run_json_lines) and run_json_lines[idx]:
            rjp = _resolve_artifact_path(run_json_lines[idx])
            if rjp.is_file():
                try:
                    run_doc = load_run(rjp)
                except (OSError, json.JSONDecodeError, FileNotFoundError):
                    run_doc = None
                    rjp = None
        kind, value, hint = compat_group_key(
            run_doc,
            first,
            generations_path=gens,
            eval_set_fingerprint=slice_fp,
            n_rows=n_rows,
        )
        if rjp is None:
            rjp = infer_run_json_from_generations_path(gens)
        rid = str(run_doc.get("run_id") if run_doc else gens.parent.name)
        label = str((run_doc or {}).get("label") or rid)
        out.append(
            PersonaDumpMeta(
                run_id=rid,
                label=label,
                generations_path=gens.resolve(),
                run_json_path=rjp,
                created_utc=str((run_doc or {}).get("created_utc") or ""),
                group_kind=kind,
                group_value=value,
                display_hint=hint,
            )
        )
    return out


def group_dumps(dumps: list[PersonaDumpMeta]) -> dict[str, list[PersonaDumpMeta]]:
    groups: dict[str, list[PersonaDumpMeta]] = {}
    for d in dumps:
        groups.setdefault(d.serialized_key, []).append(d)
    for lst in groups.values():
        lst.sort(key=lambda x: (x.created_utc, x.run_id))
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def group_title(key: str, metas: list[PersonaDumpMeta]) -> str:
    if not metas:
        return key
    m0 = metas[0]
    n = len(metas)
    if m0.group_kind == "valsha":
        return f"Val file (sha256) {m0.group_value[:12]}… — {n} runs"
    if m0.group_kind == "path":
        return f"{Path(m0.group_value).name} — {n} runs"
    if m0.group_kind == "evalset":
        return f"Eval slice · {m0.display_hint} — {n} runs"
    if m0.group_kind == "manifest_path":
        return f"Manifest only · {Path(m0.group_value).name} — {n} runs"
    return f"Unknown val — {n} runs"
