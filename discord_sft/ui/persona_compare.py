"""Join multi-run ``persona_generations.jsonl`` rows and persist comparison votes."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from discord_sft.evals.paths import resolve_checkpoint_dir
from discord_sft.evals.runner import load_training_config_provenance

VerdictKind = Literal["accepted", "rejected", "neutral"]

VOTE_STORE_VERSION = 2


def sample_key(row: dict[str, Any]) -> str:
    """Stable id for aligning rows across persona generation dumps."""
    payload = {
        "persona_id": str(row.get("persona_id") or ""),
        "context_turns": row.get("context_turns"),
        "reference": str(row.get("reference") or ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def infer_module_badges(target_modules: list[str] | None) -> list[str]:
    """Return badges for MoE/style-relevant projector modules when present."""
    mods = target_modules or []
    badges: list[str] = []
    if "v_proj" in mods:
        badges.append("v_proj")
    if "gate_proj" in mods:
        badges.append("gate_proj")
    return badges


def _mods_from_nested_training_config(run_doc: dict[str, Any]) -> list[str]:
    tc = run_doc.get("training_config")
    if not isinstance(tc, dict):
        return []
    lora = tc.get("lora")
    if not isinstance(lora, dict):
        return []
    mods = lora.get("target_modules")
    if isinstance(mods, list) and mods:
        return [str(x) for x in mods]
    return []


def _adapter_path_from_run_doc(run_doc: dict[str, Any]) -> str | None:
    model = run_doc.get("model")
    if not isinstance(model, dict):
        return None
    ap = model.get("adapter_path")
    if isinstance(ap, str) and ap.strip():
        return ap.strip()
    return None


def target_modules_from_resolved_yaml_near_adapter(
    adapter_path: str | Path,
) -> list[str]:
    """Read ``lora.target_modules`` from ``config.resolved.yaml`` beside the checkpoint."""
    p = resolve_checkpoint_dir(adapter_path)
    if p is None:
        leg = Path(adapter_path).expanduser()
        p = leg.resolve() if leg.exists() and leg.is_dir() else None
    if p is None:
        return []
    for cfg in (p / "config.resolved.yaml", p.parent / "config.resolved.yaml"):
        if not cfg.is_file():
            continue
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            return []
        try:
            raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        lora = raw.get("lora")
        if not isinstance(lora, dict):
            continue
        tm = lora.get("target_modules")
        if isinstance(tm, list) and tm:
            return [str(x) for x in tm]
    return []


def _merged_model_dir_from_run_doc(run_doc: dict[str, Any]) -> str | None:
    model = run_doc.get("model")
    if not isinstance(model, dict):
        return None
    nop = model.get("name_or_path")
    if isinstance(nop, str) and nop.strip():
        return nop.strip()
    return None


def target_modules_from_run(run_doc: dict[str, Any] | None) -> list[str]:
    """Resolve targets from embedded ``training_config``, adapter dir, or local merged ``--model`` path."""
    if not run_doc:
        return []
    mods = _mods_from_nested_training_config(run_doc)
    if mods:
        return mods
    ap = _adapter_path_from_run_doc(run_doc)
    if ap:
        ym = target_modules_from_resolved_yaml_near_adapter(ap)
        if ym:
            return ym
    md = _merged_model_dir_from_run_doc(run_doc)
    if md:
        ym = target_modules_from_resolved_yaml_near_adapter(md)
        if ym:
            return ym
        tc_pb = load_training_config_provenance(md, None)
        mods_pb = (
            _mods_from_nested_training_config({"training_config": tc_pb}) if tc_pb else []
        )
        if mods_pb:
            return mods_pb
    return []


def badge_label(target_modules: list[str] | None) -> str:
    """Short caption: highlight v_proj / gate_proj, else list adapters, else base/no data."""
    mods = target_modules or []
    badges = infer_module_badges(mods)
    if badges:
        return " · ".join(badges)
    if mods:
        shown = mods[:12]
        tail = ", ".join(shown)
        if len(mods) > len(shown):
            tail += ", …"
        return tail
    return "Base · no adapter / YAML"


def variant_stable_id(abs_jsonl: Path, source_index: int) -> str:
    s = f"{abs_jsonl.resolve()}#{source_index}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def comparison_id_from_paths(paths: list[Path]) -> str:
    lines = sorted(str(p.resolve()) for p in paths)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:20]


def vote_store_path(evals_root: Path) -> Path:
    return evals_root / "persona_compare_votes.json"


def load_vote_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": VOTE_STORE_VERSION, "comparisons": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": VOTE_STORE_VERSION, "comparisons": {}}
    if not isinstance(doc, dict):
        return {"version": VOTE_STORE_VERSION, "comparisons": {}}
    doc.setdefault("version", VOTE_STORE_VERSION)
    comp = doc.get("comparisons")
    if not isinstance(comp, dict):
        doc["comparisons"] = {}
    return doc


def save_vote_store(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc["version"] = VOTE_STORE_VERSION
    payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(
        prefix="persona_votes_", suffix=".json", dir=str(path.parent)
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        Path(tmp).replace(path)
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise


@dataclass
class VariantRow:
    variant_id: str
    generated: str
    display_label: str
    target_modules: list[str]
    jsonl_path: str
    run_id: str | None


@dataclass
class MergedPersonaSample:
    sample_key: str
    """Row used for context / reference / system (first source)."""
    base_row: dict[str, Any]
    variants: list[VariantRow]


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = sample_key(r)
        out[k] = r
    return out


def merge_generation_sources(
    indexed_rows: list[tuple[Path, list[dict[str, Any]], dict[str, Any] | None, str, int]],
) -> tuple[list[MergedPersonaSample], list[str]]:
    """Join rows that share ``sample_key`` across all sources (inner join).

    ``indexed_rows`` entries: ``(resolved_jsonl_path, rows, run_doc_or_none, ui_label, source_index)``
    """
    warnings: list[str] = []

    maps: list[dict[str, dict[str, Any]]] = []
    for path, rows, _, _, _ in indexed_rows:
        m = _index_rows(rows)
        if len(m) != len(rows):
            warnings.append(
                f"Duplicate sample keys inside `{path}` — last row wins per key."
            )
        maps.append(m)

    common = set.intersection(*(set(d.keys()) for d in maps)) if maps else set()
    union_all = set()
    for mp in maps:
        union_all |= set(mp.keys())

    if maps:
        missing = union_all - common
        if missing:
            warnings.append(
                f"{len(missing)} sample keys are not present in all sources (inner join dropped them)."
            )

    merged: list[MergedPersonaSample] = []

    variant_meta: list[
        tuple[str, Path, dict[str, Any] | None, str, int]
    ] = []  # (variant_id, path, run_doc, label, idx)
    for path, rows, run_doc, label, source_index in indexed_rows:
        vid = variant_stable_id(path, source_index)
        variant_meta.append((vid, path, run_doc, label, source_index))

    for sk in sorted(common):
        base = maps[0][sk]
        vrows: list[VariantRow] = []
        for (vid, path, run_doc, label, _), mp in zip(variant_meta, maps, strict=True):
            row = mp[sk]
            mods = target_modules_from_run(run_doc)
            run_id = (run_doc or {}).get("run_id") if run_doc else None
            rid = str(run_id) if run_id else None
            short_path = path.name
            display = f"{label}" if label else short_path
            if rid:
                display = f"{display} · `{rid[:48]}{'…' if len(rid) > 48 else ''}`"
            vrows.append(
                VariantRow(
                    variant_id=vid,
                    generated=str(row.get("generated") or ""),
                    display_label=display,
                    target_modules=mods,
                    jsonl_path=str(path.resolve()),
                    run_id=rid,
                )
            )
        merged.append(MergedPersonaSample(sample_key=sk, base_row=base, variants=vrows))

    return merged, warnings


def comparison_variant_metadata(
    indexed_rows: list[tuple[Path, list[dict[str, Any]], dict[str, Any] | None, str, int]],
) -> dict[str, dict[str, str]]:
    """Metadata block stored next to verdicts for bookkeeping."""
    out: dict[str, dict[str, str]] = {}
    for path, _rows, _run_doc, label, source_index in indexed_rows:
        vid = variant_stable_id(path, source_index)
        out[vid] = {
            "path": str(path.resolve()),
            "label": label or path.name,
        }
    return out


def comparison_verdict_map(comp_block: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build merged verdict mapping from stored ``verdicts`` plus legacy ``votes`` lists."""
    merged: dict[str, dict[str, str]] = {}
    raw_v = comp_block.get("verdicts")
    if isinstance(raw_v, dict):
        for sk, row in raw_v.items():
            if not isinstance(sk, str) or not isinstance(row, dict):
                continue
            out_row: dict[str, str] = {}
            for vid, vv in row.items():
                s = str(vid)
                sv = vv if isinstance(vv, str) else str(vv)
                low = sv.lower()
                if low == "accepted":
                    out_row[s] = "accepted"
                elif low == "rejected":
                    out_row[s] = "rejected"
            if out_row:
                merged[str(sk)] = out_row

    legacy = comp_block.get("votes")
    if isinstance(legacy, dict):
        for sk, vlist in legacy.items():
            sks = str(sk)
            if not isinstance(vlist, list):
                continue
            inner = merged.setdefault(sks, {})
            for item in vlist:
                vid = str(item)
                inner.setdefault(vid, "accepted")
    return merged


def reconcile_comparison_block_inplace(comp_block: dict[str, Any]) -> None:
    """Canonicalize to ``verdicts`` only; drop legacy ``votes``."""
    cmap = comparison_verdict_map(comp_block)
    compact: dict[str, dict[str, str]] = {}
    for sk, mp in cmap.items():
        if mp:
            compact[sk] = dict(mp)
    comp_block["verdicts"] = compact
    comp_block.pop("votes", None)


def effective_verdict(
    verdicts: dict[str, dict[str, str]],
    sample_key_hex: str,
    variant_id: str,
) -> VerdictKind:
    row = verdicts.get(sample_key_hex) or {}
    v = row.get(variant_id)
    if v == "accepted" or v == "rejected":
        return v
    return "neutral"


def summarize_verdicts_for_filtered(
    verdicts: dict[str, dict[str, str]],
    merged_filtered: list[MergedPersonaSample],
) -> dict[str, dict[str, int]]:
    """Count accepted / rejected / neutral per variant over (sample, variant) pairs in filter."""
    vids_ordered: list[str] = []
    seen: set[str] = set()
    for m in merged_filtered:
        for vr in m.variants:
            if vr.variant_id not in seen:
                seen.add(vr.variant_id)
                vids_ordered.append(vr.variant_id)

    tally: dict[str, dict[str, int]] = {
        vid: {"accepted": 0, "rejected": 0, "neutral": 0} for vid in vids_ordered
    }

    for m in merged_filtered:
        sk = m.sample_key
        for vr in m.variants:
            kind = effective_verdict(verdicts, sk, vr.variant_id)
            tally[vr.variant_id][kind] += 1

    return tally


def ranked_variant_order(summary: dict[str, dict[str, int]]) -> list[str]:
    """Sort variant ids by accepted desc, rejects asc, then id (deterministic ties)."""
    rows = [(vid, s["accepted"], s["rejected"]) for vid, s in summary.items()]
    rows.sort(key=lambda t: (-t[1], t[2], t[0]))
    return [t[0] for t in rows]


def widget_key_digest(sample_key_hex: str) -> str:
    return hashlib.sha256(sample_key_hex.encode("ascii")).hexdigest()[:16]
