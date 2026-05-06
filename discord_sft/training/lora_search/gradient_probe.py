"""Collect LoRA gradient norms to pick high-leverage modules/layers quickly.

Run a few forward/backward steps on a training config and record where
gradients concentrate. This gives a low-cost signal for choosing LoRA targets
before expensive full runs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from discord_sft.training import load_config
from discord_sft.training.common import (
    ASSISTANT_TURN_PREFIX,
    USER_TURN_PREFIX,
    num_hidden_layers as _shared_num_hidden_layers,
    set_unsloth_env,
)
from discord_sft.training.config import resolve_layers_to_transform
from discord_sft.training.data import load_sharegpt_dataset

MODULE_PATTERN = re.compile(
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|gate_up_proj)"
)
LAYER_PATTERN = re.compile(r"\.(\d+)\.")
IGNORE_INDEX = -100


def _num_hidden_layers(model: Any) -> int:
    return _shared_num_hidden_layers(model)


def _extract_layer_and_module(param_name: str) -> tuple[int | None, str | None]:
    layer_match = LAYER_PATTERN.search(param_name)
    module_match = MODULE_PATTERN.search(param_name)
    layer = int(layer_match.group(1)) if layer_match else None
    module = module_match.group(1) if module_match else None
    return layer, module


def _batchify_text_rows(
    rows: list[dict[str, str]], tokenizer: Any, max_seq_length: int
) -> dict[str, Any]:
    texts = [row["text"] for row in rows]
    toks = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    instruction_ids = tokenizer.encode(USER_TURN_PREFIX, add_special_tokens=False)
    response_ids = tokenizer.encode(ASSISTANT_TURN_PREFIX, add_special_tokens=False)
    labels: list[list[int]] = []
    supervised_total = 0
    for seq_ids, seq_mask in zip(toks["input_ids"].tolist(), toks["attention_mask"].tolist()):
        seq_labels, supervised = _build_response_only_labels(
            seq_ids,
            seq_mask,
            instruction_ids=instruction_ids,
            response_ids=response_ids,
        )
        labels.append(seq_labels)
        supervised_total += supervised
    toks["labels"] = toks["input_ids"].new_tensor(labels)
    toks["supervised_token_count"] = supervised_total
    return toks


def _find_subsequence(seq: list[int], sub: list[int], start: int, end: int) -> int:
    if not sub:
        return -1
    max_i = end - len(sub) + 1
    for i in range(start, max_i):
        if seq[i : i + len(sub)] == sub:
            return i
    return -1


def _build_response_only_labels(
    input_ids: list[int],
    attention_mask: list[int],
    *,
    instruction_ids: list[int],
    response_ids: list[int],
) -> tuple[list[int], int]:
    """Mirror response-only masking used by training.

    Only assistant response spans are supervised.
    """
    seq_len = sum(1 for x in attention_mask if int(x) != 0)
    labels = [IGNORE_INDEX] * len(input_ids)
    supervised = 0
    cursor = 0
    while True:
        response_start = _find_subsequence(input_ids, response_ids, cursor, seq_len)
        if response_start < 0:
            break
        start = response_start + len(response_ids)
        next_instruction = _find_subsequence(input_ids, instruction_ids, start, seq_len)
        end = seq_len if next_instruction < 0 else next_instruction
        for idx in range(start, end):
            labels[idx] = input_ids[idx]
            supervised += 1
        cursor = end
    return labels, supervised


def _aggregate_gradient_records(raw_params: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate captured grad norm records into global/module/layer rankings."""
    by_module: dict[str, list[float]] = defaultdict(list)
    by_layer: dict[int, list[float]] = defaultdict(list)
    by_layer_module: dict[tuple[int, str], list[float]] = defaultdict(list)

    for record in raw_params:
        grad_norm = float(record["grad_norm"])
        layer = record.get("layer")
        module = record.get("module")
        if module is not None:
            by_module[module].append(grad_norm)
        if layer is not None:
            by_layer[layer].append(grad_norm)
        if layer is not None and module is not None:
            by_layer_module[(layer, module)].append(grad_norm)

    def _summ(values: list[float]) -> dict[str, float]:
        return {
            "sum": float(sum(values)),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
            "count": float(len(values)),
        }

    modules_ranked = sorted(
        ({"module": mod, **_summ(vals)} for mod, vals in by_module.items()),
        key=lambda x: x["sum"],
        reverse=True,
    )
    layers_ranked = sorted(
        ({"layer": layer, **_summ(vals)} for layer, vals in by_layer.items()),
        key=lambda x: x["sum"],
        reverse=True,
    )
    layer_module_ranked = sorted(
        (
            {"layer": layer, "module": mod, **_summ(vals)}
            for (layer, mod), vals in by_layer_module.items()
        ),
        key=lambda x: x["sum"],
        reverse=True,
    )
    ranked_params = sorted(raw_params, key=lambda x: x["grad_norm"], reverse=True)

    return {
        "modules_ranked": modules_ranked,
        "layers_ranked": layers_ranked,
        "layer_module_ranked": layer_module_ranked,
        "params_ranked": ranked_params,
    }


def aggregate_gradient_norms(named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    """Aggregate grad norms into global/module/layer rankings."""

    raw_params: list[dict[str, Any]] = []
    for name, param in named_parameters:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_norm = float(grad.norm().item())
        layer, module = _extract_layer_and_module(name)
        raw_params.append(
            {
                "name": name,
                "grad_norm": grad_norm,
                "layer": layer,
                "module": module,
            }
        )
    return _aggregate_gradient_records(raw_params)


def _pick_device() -> str:
    import torch  # type: ignore[import-not-found]

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _fast_model_probe_load_kw(device: str) -> dict[str, Any]:
    """Kwargs passed through to FastModel/HF ``from_pretrained``.

    Do **not** pass ``device_map`` here: Transformers runs ``caching_allocator_warmup``
    whenever ``device_map`` is set, which preallocates tens of GB on CUDA and routinely
    OOMs on single-GPU 80–96GB cards that already fit the bf16 weights for Qwen3.5-A3B MoE.

    ``low_cpu_mem_usage=False`` skips the Accelerate/meta init path when possible so
    real tensors load before ``model.to(device)``.
    """
    if device != "cuda":
        return {}
    return {"low_cpu_mem_usage": False}


def collect_gradient_probe(
    *,
    config_path: str | Path,
    steps: int,
    output_path: str | Path,
    probe_batch_size: int | None = None,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
) -> dict[str, Any]:
    """Run a short no-update gradient probe and write aggregated norms.

    Optional ``lora_r`` / ``lora_alpha`` override the training YAML without editing it.
    """

    cfg = load_config(config_path)
    if lora_r is not None:
        cfg.lora.r = int(lora_r)
    if lora_alpha is not None:
        cfg.lora.alpha = int(lora_alpha)
    set_unsloth_env(cfg.model.moe_backend)

    from unsloth import FastModel  # type: ignore[import-not-found]

    import torch  # type: ignore[import-not-found]

    device = _pick_device()
    load_kw = _fast_model_probe_load_kw(device)
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=cfg.model.name,
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=False,
            load_in_16bit=bool(cfg.model.load_in_16bit),
            full_finetuning=False,
            **load_kw,
        )
    except TypeError:
        if load_kw:
            model, tokenizer = FastModel.from_pretrained(
                model_name=cfg.model.name,
                max_seq_length=cfg.model.max_seq_length,
                load_in_4bit=False,
                load_in_16bit=bool(cfg.model.load_in_16bit),
                full_finetuning=False,
            )
            load_kw = {}  # fall back to ``.to(device)`` below
        else:
            raise
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    num_hidden_layers = _num_hidden_layers(model)
    layers = resolve_layers_to_transform(cfg.lora, num_hidden_layers)

    peft_kwargs: dict[str, Any] = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha or cfg.lora.r,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
    )
    if layers is not None:
        peft_kwargs["layers_to_transform"] = layers

    model = FastModel.get_peft_model(model, **peft_kwargs)

    ds = load_sharegpt_dataset(
        cfg.data.train_path,
        tokenizer,
        system_prompt_override=cfg.data.system_prompt_override,
        enable_thinking=cfg.model.enable_thinking,
        shuffle=True,
        shuffle_seed=cfg.train.seed,
    )
    if len(ds) == 0:
        raise ValueError("Training dataset is empty; cannot run gradient probe.")

    batch_size = probe_batch_size or cfg.train.per_device_train_batch_size
    model.to(device)
    model.train()

    losses: list[float] = []
    supervised_token_counts: list[int] = []
    step_records: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    max_steps = min(int(steps), len(ds))

    for idx in range(max_steps):
        start = idx * batch_size
        if start >= len(ds):
            break
        end = min(start + batch_size, len(ds))
        rows = [ds[i] for i in range(start, end)]
        batch = _batchify_text_rows(rows, tokenizer, cfg.model.max_seq_length)
        supervised_count = int(batch.pop("supervised_token_count", 0))
        if supervised_count <= 0 and not cfg.train.allow_full_sequence_loss_fallback:
            raise RuntimeError(
                "Gradient probe produced zero response-only supervised tokens; "
                "this mirrors train.allow_full_sequence_loss_fallback=false safety."
            )
        supervised_token_counts.append(supervised_count)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        losses.append(float(loss.item()))

        agg = aggregate_gradient_norms(list(model.named_parameters()))
        aggregate_records.extend(
            {**record, "step": idx + 1} for record in agg["params_ranked"]
        )
        step_records.append(
            {
                "step": idx + 1,
                "loss": float(loss.item()),
                "supervised_tokens": supervised_count,
                "top_modules": agg["modules_ranked"][:5],
                "top_layers": agg["layers_ranked"][:8],
            }
        )

    final_agg = _aggregate_gradient_records(aggregate_records)
    result = {
        "config_path": str(Path(config_path).resolve()),
        "base_model": cfg.model.name,
        "train_path": cfg.data.train_path,
        "steps_requested": int(steps),
        "steps_executed": len(step_records),
        "batch_size": int(batch_size),
        "max_seq_length": int(cfg.model.max_seq_length),
        "lora": {
            "r": cfg.lora.r,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": list(cfg.lora.target_modules),
            "layers_to_transform": layers,
        },
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "mean": (sum(losses) / len(losses)) if losses else None,
        },
        "supervised_tokens": {
            "total": sum(supervised_token_counts),
            "mean_per_step": (
                sum(supervised_token_counts) / len(supervised_token_counts)
                if supervised_token_counts
                else 0.0
            ),
        },
        "step_records": step_records,
        "aggregate": final_agg,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m discord_sft.training.lora_search.gradient_probe",
        description="Run a short LoRA gradient-norm probe on a training YAML.",
    )
    p.add_argument("--config", required=True, help="Training YAML to probe.")
    p.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of no-update backward steps to run.",
    )
    p.add_argument(
        "--output",
        default="out/lora/probes/gradient-probe",
        help=(
            "Output JSON path, or directory (writes gradient_norms.json inside). "
            "Default: out/lora/probes/gradient-probe."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional probe batch size override.",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=None,
        metavar="R",
        help="Override lora.r from the config (optional).",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        metavar="A",
        help="Override lora.alpha from the config (optional).",
    )
    return p


def _single_output_path(output: str | Path) -> Path:
    path = Path(output)
    return path if path.suffix else path / "gradient_norms.json"


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out_path = _single_output_path(args.output)
    result = collect_gradient_probe(
        config_path=args.config,
        steps=args.steps,
        output_path=out_path,
        probe_batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    print(
        json.dumps(
            {
                "output": str(out_path.resolve()),
                "steps_executed": result["steps_executed"],
                "top_modules": result["aggregate"]["modules_ranked"][:5],
                "lora": result["lora"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
