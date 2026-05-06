#!/usr/bin/env python3
"""Summarize PEFT LoRA adapter layout for debugging HF / Unsloth / vLLM issues.

Reads ``adapter_config.json`` and every ``adapter_model*.safetensors`` shard
(tensor names + shapes via ``safetensors`` slices; no full weight load).

Example::

    python scripts/analyze_lora_adapter.py out/lora/<run_name>/epoch-1

    python scripts/analyze_lora_adapter.py --json /path/to/epoch-1 | jq .

vLLM uses ``--enable-lora`` / ``--lora-modules`` and matches LoRA tensors to
live ``nn.Module`` names in the loaded base checkpoint. Typical failures:

- **Wrong base**: ``base_model_name_or_path`` / ``run.json`` ``base_model`` does not
  match ``vllm serve`` ``--model`` (architecture or revision drift).
- **Rank cap**: server's ``max_lora_rank`` must be >= adapter ``r`` (discord-sft
  sweeps derive this from adapters; custom servers may default too low).
- **Target module names**: tensors are saved under the *training stack's* HF
  module prefixes (often multiple ``language_model`` levels for VL / Unsloth).
  Those strings must correspond to submodule names vLLM instantiates; merged
  full checkpoints from ``train merge-peft`` must likewise match the server's
  expected module tree (see vLLM / MoE merge discussions when keys still diverge).
- **MoE / experts**: vLLM's fused MoE LoRA story can disagree with training when
  expert weights differ from what the fused runtime expects.

This script surfaces prefix patterns, ranks, touched layers/modules, vision vs LM
hints, and PEFT bookkeeping so you can diff against errors from vLLM.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load_adapter_config(adapter_dir: Path) -> tuple[dict[str, Any], Path | None]:
    p = adapter_dir / "adapter_config.json"
    if not p.is_file():
        raise SystemExit(f"Not a PEFT adapter directory (missing adapter_config.json): {adapter_dir}")
    return json.loads(p.read_text(encoding="utf-8")), p


def _adapter_safetensor_paths(adapter_dir: Path) -> list[Path]:
    single = adapter_dir / "adapter_model.safetensors"
    if single.is_file():
        return [single]
    idx = adapter_dir / "adapter_model.safetensors.index.json"
    if idx.is_file():
        data = json.loads(idx.read_text(encoding="utf-8"))
        wm = data.get("weight_map")
        if isinstance(wm, dict) and wm:
            names = sorted({str(v) for v in wm.values()})
            return [adapter_dir / n for n in names]
    # glob fallback (some exporters use slightly different filenames)
    found = sorted(adapter_dir.glob("adapter_model*.safetensors"))
    return [x for x in found if x.is_file() and "index.json" not in x.name]


def _read_run_json(adapter_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    for root in (adapter_dir, adapter_dir.parent):
        rj = root / "run.json"
        if not rj.is_file():
            continue
        try:
            d = json.loads(rj.read_text(encoding="utf-8"))
            return (d if isinstance(d, dict) else None), rj
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def _peft_highlight(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fields that affect loading / runtime."""
    keys = (
        "peft_type",
        "task_type",
        "base_model_name_or_path",
        "revision",
        "inference_mode",
        "bias",
        "target_modules",
        "rank_pattern",
        "alpha_pattern",
        "exclude_modules",
        "modules_to_save",
        "r",
        "lora_alpha",
        "lora_dropout",
        "fan_in_fan_out",
        "use_rslora",
        "init_lora_weights",
        "target_parameters",
        "megatron_core",
        "loftq_config",
        "qalora_group_size",
        "qalora_shuffle",
        "qalora_bf16_inference",
        "layer_replication",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    return out


def _layer_indices_from_key(key: str) -> list[int]:
    # Match dotted integers that look like layer indices (cheap heuristic).
    return [int(m.group(1)) for m in re.finditer(r"\.(\d+)\.", key)]


def _stripped_human_prefix(key: str) -> str:
    """Best-effort 'module path' for grouping (drops common PEFT root)."""
    k = key
    for pref in ("base_model.model.", "model."):
        if k.startswith(pref):
            k = k[len(pref) :]
            break
    return k.split(".lora_")[0] if ".lora_" in k else k


def _lora_variant(key: str) -> str | None:
    """PEFT submodule name between lora_A / lora_B and .weight."""
    m = re.search(r"\.lora_(A|B)\.([^.]*)\.weight$", key)
    if not m:
        return None
    return m.group(2)


def _infer_tensor_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    from safetensors import safe_open

    reports: list[dict[str, Any]] = []
    for shard in paths:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for k in sorted(f.keys()):
                sl = f.get_slice(k)
                reports.append(
                    {
                        "key": k,
                        "shard": shard.name,
                        "dtype": str(sl.get_dtype()),
                        "shape": list(sl.get_shape()),
                    }
                )
    return reports


def _analyze_lora_pair_layout(
    reports: list[dict[str, Any]],
    *,
    cfg_r: int | None,
) -> tuple[Counter, Counter, Counter, int]:
    """Classify paired ``lora_A`` / ``lora_B`` 2-D tensors.

    MoE / fused checkpoints often expose ``sh_A[0] == sh_B[1]`` on a huge *reduction*
    axis rather than LoRA rank, so counts matching that corner separately from the
    true rank ``cfg_r``.
    """
    by_key = {r["key"]: r for r in reports}
    rank_canon: Counter = Counter()
    trans: Counter = Counter()
    contraction_corner_other: Counter = Counter()
    exotic = 0
    for key, meta in list(by_key.items()):
        if ".lora_A." not in key or not key.endswith(".weight"):
            continue
        bkey = key.replace(".lora_A.", ".lora_B.")
        other = by_key.get(bkey)
        if not other:
            continue
        sh_a: list[int] = meta["shape"]
        sh_b: list[int] = other["shape"]
        if len(sh_a) != 2 or len(sh_b) != 2:
            continue
        # PEFT default Linear LoRA: lora_A (r, fan_in), lora_B (fan_out, r)
        if sh_a[0] == sh_b[1]:
            shared = sh_a[0]
            if isinstance(cfg_r, int) and shared == cfg_r:
                rank_canon[shared] += 1
            else:
                contraction_corner_other[shared] += 1
        elif sh_a[1] == sh_b[0]:
            trans[sh_a[1]] += 1
        else:
            exotic += 1
    return rank_canon, trans, contraction_corner_other, exotic


def _prefix_buckets(keys: Iterable[str]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for k in keys:
        if k.startswith("base_model.model."):
            c["base_model.model.*"] += 1
        elif k.startswith("model."):
            c["model.*"] += 1
        else:
            c["(other_root)"] += 1

    lm = Counter()
    for k in keys:
        if ".language_model.language_model.language_model." in k:
            lm["triple_language_model_segment"] += 1
        elif ".language_model.language_model." in k:
            lm["double_language_model_segment"] += 1
        elif ".visual." in k or k.startswith("base_model.model.visual.") or ".model.visual." in k:
            lm["visual_segment"] += 1
        if "experts" in k or "moe" in k.lower():
            lm["expert_or_moe_mention"] += 1

    return {**dict(sorted(c.items(), key=lambda x: (-x[1], x[0]))), **_special_counts(lm)}


def _special_counts(inner: Counter) -> dict[str, int]:
    return {k: int(v) for k, v in inner.items() if v}


def analyze(adapter_dir: Path) -> dict[str, Any]:
    adapter_dir = adapter_dir.resolve()
    cfg_full, cfg_path = _load_adapter_config(adapter_dir)
    paths = _adapter_safetensor_paths(adapter_dir)
    if not paths:
        raise SystemExit(
            f"No adapter_model*.safetensors under {adapter_dir} "
            f"(checked adapter_model.safetensors & index)."
        )

    tensor_reports = _infer_tensor_reports(paths)
    keys = [r["key"] for r in tensor_reports]

    layers_flat: list[int] = []
    for k in keys:
        layers_flat.extend(_layer_indices_from_key(k))

    stripped_paths = defaultdict(list)
    for k in keys:
        stripped_paths[_stripped_human_prefix(k)].append(k)

    variant_from_key = Counter()
    for k in keys:
        v = _lora_variant(k)
        if v:
            variant_from_key[v] += 1

    cfg_r_maybe = cfg_full.get("r")
    cfg_r_int = cfg_r_maybe if isinstance(cfg_r_maybe, int) else None
    canon_ranks, trans_ranks, corner_other_ranks, exotic_pairs = _analyze_lora_pair_layout(
        tensor_reports, cfg_r=cfg_r_int
    )

    run_json, run_json_path = _read_run_json(adapter_dir)

    # Config rank vs inferred
    layout_rank_vote = canon_ranks + trans_ranks
    inferred_mode = dict(layout_rank_vote.most_common(5)) if layout_rank_vote else {}

    hints: list[str] = []
    if exotic_pairs > 0:
        hints.append(
            f"{exotic_pairs} LoRA A/B tensor pairs skip textbook PEFT (r,in)x(out,r) layout "
            "(unusual layout)."
        )
    if cfg_r_int is not None and corner_other_ranks:
        dominant = corner_other_ranks.most_common(1)[0]
        hints.append(
            f"{sum(corner_other_ranks.values())} pairs share A[0]==B[1]={dominant[0]} "
            f"but adapter r={cfg_r_int}; dim {dominant[0]} is typically a contraction "
            f"axis inside fused MoE experts, not LoRA rank."
        )
    elif cfg_r_int is not None and trans_ranks and max(trans_ranks.values()) >= 8:
        dominant_t = trans_ranks.most_common(1)[0][0]
        if dominant_t != cfg_r_int:
            hints.append(
                f"Config r={cfg_r_int} but transposed-axis pairs imply r={dominant_t} "
                "(check rank_pattern)."
            )
    triple = sum(
        1 for k in keys if ".language_model.language_model.language_model." in k
    )
    if triple:
        hints.append(
            f"{triple} keys contain triple `.language_model` nesting - "
            "common with Unsloth VL merges; vLLM may expect shorter "
            "`language_model.model.*` prefixes (compare server errors)."
        )
    vision = sum(
        1
        for k in keys
        if ".visual." in k
        or "model.visual." in k
        or k.startswith("base_model.model.visual.")
    )
    if vision:
        hints.append(
            f"{vision} tensors touch `visual`; confirm vLLM LoRA routing "
            "includes vision modules for this model class."
        )
    moe = sum(1 for k in keys if "experts" in k.lower() or ".moe" in k.lower())
    if moe:
        hints.append(
            f"{moe} keys mention MoE/experts; fused expert LoRA mismatches "
            "are common - see repo training README MoE notes."
        )

    return {
        "adapter_dir": str(adapter_dir),
        "adapter_config_path": str(cfg_path),
        "adapter_config_highlight": _peft_highlight(cfg_full),
        "weight_files": [str(p.name) for p in paths],
        "tensor_count": len(tensor_reports),
        "run_json_path": str(run_json_path) if run_json_path else None,
        "run_json_fields": (
            {
                k: run_json[k]
                for k in ("base_model", "run_name", "git_sha", "output_dir")
                if isinstance(run_json, dict) and k in run_json
            }
            or None
        ),
        "key_prefix_buckets": _prefix_buckets(keys),
        "lora submodule suffix counts (…lora_A.<name>.weight)": dict(
            sorted(variant_from_key.items(), key=lambda x: -x[1])
        ),
        "lora_pair_layout": {
            "matching_adapter_r_canonical_corners": dict(canon_ranks.most_common(8)),
            "transposed_axes_count_by_r": dict(trans_ranks.most_common(8)),
            "other_A0_eq_B1_dims_moelike": dict(corner_other_ranks.most_common(8)),
            "exotic_unclassified_pairs": exotic_pairs,
        },
        "combined_inferred_rank_votes_top": inferred_mode if inferred_mode else None,
        "layer_index_histogram": dict(
            sorted(Counter(layers_flat).most_common(), key=lambda x: int(x[0]))
        )
        if layers_flat
        else {},
        "touched_logical_modules_sample": sorted(stripped_paths.keys())[:40],
        "touched_logical_modules_total": len(stripped_paths),
        "vllm_debug_hints": hints,
        "tensors_detail": tensor_reports,
    }


def _print_human(data: dict[str, Any]) -> None:
    h = data["adapter_config_highlight"]
    print(f"Adapter: {data['adapter_dir']}")
    print(f"Weights: {', '.join(data['weight_files'])}  ({data['tensor_count']} tensors)")
    bm = h.get("base_model_name_or_path")
    print(f"PEFT base_model_name_or_path: {bm!r}")
    if data.get("run_json_path"):
        print(f"run.json: {data['run_json_path']}")
    if data.get("run_json_fields", {}).get("base_model"):
        print(f"run.json base_model: {data['run_json_fields']['base_model']!r}")
    print(f"target_modules: {h.get('target_modules')}")
    print(f"r={h.get('r')!r}, lora_alpha={h.get('lora_alpha')!r}, peft_type={h.get('peft_type')!r}")
    lay = data.get("lora_pair_layout") or {}
    if lay:
        print(
            "LoRA pair layout: r-matching_corners="
            f"{lay.get('matching_adapter_r_canonical_corners')} "
            f"A0=B1_other_dims={lay.get('other_A0_eq_B1_dims_moelike')} "
            f"transposed_r={lay.get('transposed_axes_count_by_r')} "
            f"exotic={lay.get('exotic_unclassified_pairs')}"
        )
    if data.get("combined_inferred_rank_votes_top"):
        print(f"Votes (canon+transposed only): {data['combined_inferred_rank_votes_top']}")
    print(f"Prefix buckets: {data['key_prefix_buckets']}")
    print(f"Touched logical modules (total {data['touched_logical_modules_total']}):")
    for line in data["touched_logical_modules_sample"][:24]:
        print(f"  {line}")
    rem = max(0, data["touched_logical_modules_total"] - 24)
    if rem:
        print(f"  ... ({rem} more)")
    hints = data.get("vllm_debug_hints") or []
    print("vLLM-oriented hints:")
    if hints:
        for t in hints:
            print(f"  - {t}")
    else:
        print("  (none heuristic - compare tensor prefixes to server's module names)")
    print("\nRe-run with --json for tensors_detail (+ pipe to jq / a file diff).")


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument(
        "adapter_dir",
        type=Path,
        help="Checkpoint dir containing adapter_config.json + adapter weights.",
    )
    ap.add_argument(
        "--no-tensors",
        action="store_true",
        help="Omit per-tensors_detail (smaller --json output).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report to stdout.",
    )
    args = ap.parse_args()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    report = analyze(adapter_dir)
    if args.no_tensors:
        report.pop("tensors_detail", None)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())