"""Typed config for a single LoRA training run.

The YAML on disk is intentionally a thin mirror of these dataclasses: one
YAML file = one reproducible run. Two serialised forms exist at runtime:

* ``config.source.yaml`` — verbatim copy of the user's YAML (intent).
* ``config.resolved.yaml`` — after defaults are filled in and
  ``layers_last_pct`` has been expanded to a concrete index list
  (reproducibility, see :func:`resolve_layers_to_transform`).

The resolver lives here rather than in ``trainer.py`` so the eval side can
load a checkpoint's ``config.resolved.yaml`` without importing torch.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# Default LoRA targets for Qwen3.5 MoE per Unsloth docs: all attention +
# MLP projections, including the fused MoE ``gate_up_proj``. Callers can
# override to e.g. ``[o_proj, down_proj]`` for pure style fine-tuning.
DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "gate_up_proj",
)


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen3.5-35B-A3B"
    max_seq_length: int = 2048
    load_in_16bit: bool = True
    # Qwen3.5 defaults to thinking mode in the chat template. For Discord
    # style SFT we train clean assistant turns, not empty <think> scaffolds.
    enable_thinking: bool = False
    # ``None`` means let Unsloth pick its default MoE backend. Set to e.g.
    # "triton" / "cutlass" to pin it; surfaced as ``UNSLOTH_MOE_BACKEND``.
    moe_backend: str | None = None


@dataclass
class DataConfig:
    train_path: str = "out/sft/train.jsonl"
    val_path: str | None = "out/sft/val.jsonl"
    # Override the stored ``system`` field on every sample if set. ``None``
    # keeps each sample's per-persona system prompt from build-sft.
    system_prompt_override: str | None = None


@dataclass
class LoraConfig:
    r: int = 16
    # Per Unsloth guidance for Qwen3.5: ``alpha == r`` (not ``2*r``).
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    # Exactly one of ``layers_to_transform`` / ``layers_last_pct`` may be set;
    # if neither is set, LoRA is attached to every layer (PEFT default).
    layers_to_transform: list[int] | None = None
    layers_last_pct: float | None = None
    use_gradient_checkpointing: str | bool = "unsloth"
    random_state: int = 3407
    bias: str = "none"


@dataclass
class TrainLoopConfig:
    epochs: int = 1
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    seed: int = 3407
    # HF Trainer ``eval_strategy``: "no" | "steps" | "epoch".
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    logging_steps: int = 10
    max_grad_norm: float = 1.0
    # Optional cap on training steps; overrides ``epochs`` if set.
    max_steps: int | None = None
    # If false, a failure to mask user spans aborts training rather than
    # silently switching to full-sequence loss.
    allow_full_sequence_loss_fallback: bool = False
    # Optional Weights & Biases run tracking.
    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    # Optional checkpoint-triggered tertiary evals during training, typically
    # used to stream extra metrics to W&B. Training loop defaults to
    # persona-only tasks so it does not depend on lmms-eval.
    tertiary_eval_enabled: bool = False
    tertiary_eval_tasks_checkpoint: list[str] = field(
        default_factory=lambda: ["persona"]
    )
    tertiary_eval_tasks_final: list[str] | None = None
    tertiary_eval_out_dir: str = "out/evals"
    tertiary_eval_profile_json: str | None = "out/sft/profiles.json"
    tertiary_eval_limit: int | None = None
    tertiary_eval_batch_size: int = 1
    # Optional adapter directory (epoch-N/step-S/final) to continue training
    # from. This loads existing LoRA weights as trainable parameters before
    # the trainer starts.
    resume_adapter_path: str | None = None


@dataclass
class CheckpointConfig:
    output_dir: str = "out/lora/run"
    # "epoch" | "steps" | "no".
    save_strategy: str = "epoch"
    save_steps: int | None = None
    # How many in-run epoch/step dumps to keep (oldest evicted). ``None``
    # keeps everything. The ``final/`` dump is never evicted.
    save_total_limit: int | None = 3
    # Also save a ``final/`` adapter at end of training.
    save_final: bool = True


@dataclass
class TrainConfig:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- YAML I/O ----------


def _require_yaml():
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "pyyaml is required. Install with `pip install 'discord-sft[train]'` "
            "or `pip install pyyaml`."
        ) from e
    return yaml


def load_config(path: str | Path) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file.

    Unknown keys trigger a ``ValueError`` so typos don't silently fall back
    to defaults (this matters a lot for hyperparameters). Known sections are
    merged with dataclass defaults.
    """
    yaml = _require_yaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return from_dict(raw)


def load_config_for_merge_parity(path: str | Path) -> TrainConfig:
    """Load training YAML like :func:`load_config` for merge-time metadata only.

    Older ``config.resolved.yaml`` files could list both ``lora.layers_last_pct``
    and materialised ``lora.layers_to_transform`` (before the trainer cleared
    the pct field). Drop the pct when both are present so parity load matches
    :class:`LoraConfig` validation.
    """
    yaml = _require_yaml()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    lora_section = raw.get("lora")
    if isinstance(lora_section, dict):
        lt = lora_section.get("layers_to_transform")
        lp = lora_section.get("layers_last_pct")
        if lt is not None and lp is not None:
            raw = {**raw, "lora": {**lora_section, "layers_last_pct": None}}
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> TrainConfig:
    """Validate + coerce a plain dict into a :class:`TrainConfig`."""

    def _section(name: str, cls: type) -> Any:
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"Section '{name}' must be a mapping")
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(section) - valid
        if unknown:
            raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")
        return cls(**section)

    top_known = {"run_name", "model", "data", "lora", "train", "checkpoint"}
    unknown = set(raw) - top_known
    if unknown:
        raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")

    lora = _section("lora", LoraConfig)
    # Validate layer-selection mutual exclusion and bounds here so errors fire
    # at config-load time, before a model is ever touched.
    if lora.layers_to_transform is not None and lora.layers_last_pct is not None:
        raise ValueError(
            "lora.layers_to_transform and lora.layers_last_pct are mutually "
            "exclusive; set at most one."
        )
    if lora.layers_last_pct is not None:
        p = float(lora.layers_last_pct)
        if p <= 0.0:
            raise ValueError(
                "layers_last_pct=0 would attach no LoRA adapters; "
                "either raise it above 0 or omit the field."
            )
        if p > 1.0:
            raise ValueError(f"layers_last_pct must be <= 1.0, got {p}")
    if lora.alpha is None:
        lora.alpha = lora.r

    return TrainConfig(
        run_name=str(raw.get("run_name", "run")),
        model=_section("model", ModelConfig),
        data=_section("data", DataConfig),
        lora=lora,
        train=_section("train", TrainLoopConfig),
        checkpoint=_section("checkpoint", CheckpointConfig),
    )


def dump_yaml(cfg: TrainConfig, path: str | Path) -> None:
    """Write ``cfg`` to ``path`` as human-readable YAML."""
    yaml = _require_yaml()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


# ---------- Layer resolution ----------


def resolve_layers_to_transform(
    lora: LoraConfig, num_hidden_layers: int
) -> list[int] | None:
    """Expand the LoRA layer spec into a concrete index list (or ``None``).

    Rules (plan-locked):

    * Explicit ``layers_to_transform`` wins if set — validated against
      ``num_hidden_layers``.
    * ``layers_last_pct`` in ``(0, 1]`` expands to the last ``p * N`` layers.
    * Neither set → return ``None`` (PEFT applies LoRA to every layer).

    The ``pct == 1.0`` case deliberately expands to the full ``[0..N-1]``
    list rather than ``None`` so the resolved config records exactly which
    layers were trained.
    """
    if lora.layers_to_transform is not None:
        layers = list(lora.layers_to_transform)
        for idx in layers:
            if not (0 <= int(idx) < num_hidden_layers):
                raise ValueError(
                    f"layers_to_transform index {idx} out of range "
                    f"[0, {num_hidden_layers})"
                )
        return [int(i) for i in layers]
    if lora.layers_last_pct is None:
        return None
    p = float(lora.layers_last_pct)
    start = int(math.floor(num_hidden_layers * (1.0 - p)))
    start = max(0, min(start, num_hidden_layers - 1))
    return list(range(start, num_hidden_layers))
