"""Main ``run_training`` entrypoint for Unsloth + LoRA SFT of Qwen3.5 MoE.

This module is the only one that imports ``unsloth`` / ``trl`` / ``torch``.
We push all other training-deps work into ``config.py`` / ``data.py`` so
the rest of the package (and the CLI) can be imported without paying for
them.

Key details (documented in the plan):

* ``UNSLOTH_COMPILE_DISABLE=1`` is set *before* any Unsloth import to
  sidestep the MoE+LoRA mixed-dtype crash that ``torch.compile`` triggers.
* Model is loaded in bf16 via ``FastModel.from_pretrained`` —
  ``load_in_4bit=False`` because Qwen3.5 QLoRA is not recommended.
* ``layers_last_pct`` is resolved to a concrete index list *after* the
  model is loaded (needs ``num_hidden_layers``), then the resolved spec is
  written to ``config.resolved.yaml``.
* ``train_on_responses_only`` restricts loss to assistant spans, matching
  the ShareGPT multi-turn targets in ``out/sft/train.jsonl``.
* Adapter-only checkpoints land in ``epoch-N/`` / ``step-S/`` / ``final/``
  subdirs via :class:`LoRACheckpointCallback`; the trainer's own
  ``save_strategy`` is set to ``"no"`` to avoid double-writing the huge
  ``checkpoint-<global_step>/`` tree.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from discord_sft.training.callbacks import RunManifest, make_checkpoint_callback
from discord_sft.training.config import (
    TrainConfig,
    dump_yaml,
    resolve_layers_to_transform,
)
from discord_sft.training.common import (
    ASSISTANT_TURN_PREFIX,
    USER_TURN_PREFIX,
    num_hidden_layers,
    set_unsloth_env,
)
from discord_sft.training.data import load_sharegpt_dataset


def _preflight_tertiary_eval(cfg: TrainConfig) -> None:
    """Fail fast if configured tertiary evals cannot run in-train.

    Training supports persona tertiary evals only; lmms-eval tasks should be
    run out-of-band via ``discord-sft eval run`` after checkpoints are saved.
    """
    if not cfg.train.tertiary_eval_enabled:
        return
    from discord_sft.evals.benchmarks import split_tasks

    def _check(task_list: list[str], label: str) -> None:
        lmms_tasks, include_persona = split_tasks(task_list)
        if lmms_tasks:
            raise RuntimeError(
                f"{label} includes lmms-eval tasks {lmms_tasks}. "
                "Training tertiary evals are persona-only; run lmms tasks "
                "separately with `discord-sft eval run`."
            )
        if include_persona and (not cfg.data.val_path or not Path(cfg.data.val_path).exists()):
            raise FileNotFoundError(
                f"persona tertiary eval requires val_path, but {cfg.data.val_path!r} "
                "is missing."
            )

    _check(list(cfg.train.tertiary_eval_tasks_checkpoint), "tertiary_eval_tasks_checkpoint")
    if cfg.train.tertiary_eval_tasks_final is not None:
        _check(list(cfg.train.tertiary_eval_tasks_final), "tertiary_eval_tasks_final")


def _to_wandb_tertiary_metrics(scores: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in scores.items():
        if key.startswith("persona."):
            out[f"eval/{key.replace('.', '/')}"] = float(value)
    return out


def _make_wandb_log_callback(wandb_module: Any):
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class WandbMetricCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
            self._train_start_time = time.monotonic()
            self._last_step_time = self._train_start_time
            self._last_step = int(state.global_step or 0)
            self._last_train_loss: float | None = None
            self._best_eval_loss: float | None = None
            if wandb_module.run is not None:
                wandb_module.define_metric("train/global_step")
                wandb_module.define_metric("train/loss", step_metric="train/global_step")
                wandb_module.define_metric("eval/loss", step_metric="train/global_step")
                wandb_module.define_metric("eval/ppl", step_metric="train/global_step")
                wandb_module.define_metric(
                    "eval/generalization_gap", step_metric="train/global_step"
                )
                wandb_module.define_metric(
                    "train/step_throughput_steps_per_sec",
                    step_metric="train/global_step",
                )
                wandb_module.define_metric("eval/best_loss", step_metric="train/global_step")
                wandb_module.define_metric(
                    "train/grad_norm_preclip", step_metric="train/global_step"
                )
                wandb_module.define_metric("train/lr", step_metric="train/global_step")
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
            if not logs or wandb_module.run is None:
                return control
            payload: dict[str, float] = {"train/global_step": float(state.global_step)}
            now = time.monotonic()
            cur_step = int(state.global_step or 0)
            if cur_step > self._last_step:
                dt = max(now - self._last_step_time, 1e-6)
                dstep = cur_step - self._last_step
                payload["train/step_throughput_steps_per_sec"] = float(dstep / dt)
                self._last_step = cur_step
                self._last_step_time = now
            if "loss" in logs:
                self._last_train_loss = float(logs["loss"])
                payload["train/loss"] = self._last_train_loss
            if "eval_loss" in logs:
                eval_loss = float(logs["eval_loss"])
                payload["eval/loss"] = eval_loss
                if eval_loss < 80.0:
                    payload["eval/ppl"] = float(math.exp(eval_loss))
                if self._last_train_loss is not None:
                    payload["eval/generalization_gap"] = float(eval_loss - self._last_train_loss)
                if self._best_eval_loss is None or eval_loss < self._best_eval_loss:
                    self._best_eval_loss = eval_loss
                payload["eval/best_loss"] = float(self._best_eval_loss)
            if "grad_norm" in logs:
                payload["train/grad_norm_preclip"] = float(logs["grad_norm"])
            if "learning_rate" in logs:
                payload["train/lr"] = float(logs["learning_rate"])
            if len(payload) > 1:
                wandb_module.log(payload, step=state.global_step)
            return control

    return WandbMetricCallback


def run_training(
    cfg: TrainConfig,
    *,
    source_yaml: str | Path | None = None,
) -> dict[str, Any]:
    """Load base model, attach LoRA, train, checkpoint, save ``final/``.

    Parameters
    ----------
    cfg:
        The validated :class:`TrainConfig` describing this run.
    source_yaml:
        Path to the YAML file the user passed to the CLI. If provided,
        it's copied verbatim into ``{output_dir}/config.source.yaml`` so
        the recorded intent survives editing the YAML after launch.

    Returns
    -------
    dict
        Summary with ``run_name``, ``output_dir``, and the manifest data.
    """
    set_unsloth_env(cfg.model.moe_backend)
    _preflight_tertiary_eval(cfg)

    # Lazy heavy imports.
    try:
        from unsloth import FastModel  # type: ignore[import-not-found]
        from unsloth.chat_templates import (
            train_on_responses_only,  # type: ignore[import-not-found]
        )
    except ImportError as e:
        raise ImportError(
            "unsloth is not installed. Install with `pip install 'discord-sft[train]'`."
        ) from e
    try:
        from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "trl is not installed. Install with `pip install 'discord-sft[train]'`."
        ) from e

    output_dir = Path(cfg.checkpoint.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_module = None
    if cfg.train.wandb_enabled:
        try:
            import wandb as wandb_module  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb`."
            ) from e

        wandb_init: dict[str, Any] = {
            "project": cfg.train.wandb_project or "discord-sft-training",
            "name": cfg.train.wandb_run_name or cfg.run_name,
            "config": cfg.to_dict(),
            "tags": list(cfg.train.wandb_tags),
            "reinit": True,
        }
        if cfg.train.wandb_entity:
            wandb_init["entity"] = cfg.train.wandb_entity
        wandb_module.init(**wandb_init)

    # --- 1. Load base model + tokenizer ------------------------------------
    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=bool(cfg.model.load_in_16bit),
        full_finetuning=False,
    )
    # Some FastModel variants return a processor; unwrap to the tokenizer.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

    num_layers = num_hidden_layers(model)

    # --- 2. Resolve LoRA layer spec into concrete indices ------------------
    resolved_layers = resolve_layers_to_transform(cfg.lora, num_layers)
    # Persist the resolution back into the config so config.resolved.yaml is
    # self-describing (no need to re-run the layers_last_pct math to know
    # which layers were trained). Drop layers_last_pct once indices are
    # materialised — keeping both violates LoraConfig validation downstream
    # (merge-peft, strict YAML reload).
    if resolved_layers is not None and cfg.lora.layers_last_pct is not None:
        resolved_lora = replace(
            cfg.lora,
            layers_to_transform=resolved_layers,
            layers_last_pct=None,
        )
    else:
        resolved_lora = replace(cfg.lora, layers_to_transform=resolved_layers)
    resolved_cfg = replace(cfg, lora=resolved_lora)

    # --- 3. Write the config trio + manifest stub --------------------------
    manifest = RunManifest(resolved_cfg, source_yaml=source_yaml)
    manifest.initialize(num_hidden_layers=num_layers)

    # --- 4. Attach LoRA adapters -------------------------------------------
    peft_kwargs: dict[str, Any] = dict(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha or cfg.lora.r,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        use_gradient_checkpointing=cfg.lora.use_gradient_checkpointing,
        random_state=cfg.lora.random_state,
    )
    if resolved_layers is not None:
        peft_kwargs["layers_to_transform"] = resolved_layers

    model = FastModel.get_peft_model(model, **peft_kwargs)
    if cfg.train.resume_adapter_path:
        resume_path = Path(cfg.train.resume_adapter_path)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"resume_adapter_path {cfg.train.resume_adapter_path!r} not found"
            )
        if not (resume_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                "resume_adapter_path must point to a LoRA adapter directory "
                f"containing adapter_config.json, got {cfg.train.resume_adapter_path!r}"
            )
        try:
            from peft import PeftModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "peft is not installed. Install with `pip install 'discord-sft[train]'`."
            ) from e
        model = PeftModel.from_pretrained(
            model,
            str(resume_path),
            is_trainable=True,
        )
        manifest.data.setdefault("events", []).append(
            {
                "type": "resume_adapter_loaded",
                "path": str(resume_path.resolve()),
            }
        )
        manifest._flush()

    # --- 5. Datasets -------------------------------------------------------
    chat_template_metadata: dict[str, Any] = {}
    train_ds = load_sharegpt_dataset(
        cfg.data.train_path,
        tokenizer,
        system_prompt_override=cfg.data.system_prompt_override,
        enable_thinking=cfg.model.enable_thinking,
        template_metadata=chat_template_metadata,
        shuffle=True,
        shuffle_seed=cfg.train.seed,
    )
    eval_ds = None
    if cfg.data.val_path and Path(cfg.data.val_path).exists():
        eval_ds = load_sharegpt_dataset(
            cfg.data.val_path,
            tokenizer,
            system_prompt_override=cfg.data.system_prompt_override,
            enable_thinking=cfg.model.enable_thinking,
            shuffle=False,
        )
    elif cfg.train.eval_strategy != "no" and cfg.data.val_path:
        raise FileNotFoundError(
            f"val_path {cfg.data.val_path} not found but eval_strategy={cfg.train.eval_strategy!r}"
        )
    manifest.data["chat_template"] = chat_template_metadata
    manifest._flush()

    # --- 6. Build SFTConfig / SFTTrainer -----------------------------------
    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir / "hf_trainer"),
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_length,
        packing=False,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        seed=cfg.train.seed,
        logging_steps=cfg.train.logging_steps,
        max_grad_norm=cfg.train.max_grad_norm,
        save_strategy="no",  # handled by LoRACheckpointCallback
        eval_strategy=cfg.train.eval_strategy if eval_ds is not None else "no",
        report_to=["wandb"] if cfg.train.wandb_enabled else [],
    )
    if cfg.train.eval_steps:
        sft_kwargs["eval_steps"] = cfg.train.eval_steps
    if cfg.train.max_steps is not None:
        sft_kwargs["max_steps"] = cfg.train.max_steps

    sft_args = SFTConfig(**sft_kwargs)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    # trl 0.12+ accepts processing_class/tokenizer depending on version;
    # pass both under names the current release understands.
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
    if wandb_module is not None:
        trainer.add_callback(_make_wandb_log_callback(wandb_module)())

    # --- 7. Loss masking: only assistant spans are supervised --------------
    # Qwen3.5 chat template uses ``<|im_start|>user`` / ``<|im_start|>assistant``
    # as turn delimiters. ``train_on_responses_only`` masks everything up to
    # and including the ``response_part`` of each turn.
    try:
        trainer = train_on_responses_only(
            trainer,
            instruction_part=USER_TURN_PREFIX,
            response_part=ASSISTANT_TURN_PREFIX,
        )
    except Exception as e:
        msg = (
            f"train_on_responses_only failed: {e}; loss would be computed over "
            "the full sequence."
        )
        if not cfg.train.allow_full_sequence_loss_fallback:
            manifest.finish(status="failed", error=msg)
            raise RuntimeError(
                msg
                + " Set train.allow_full_sequence_loss_fallback=true to opt into "
                "full-sequence loss."
            ) from e

        fallback_msg = msg + " Continuing because fallback was explicitly enabled."
        manifest.data.setdefault("warnings", []).append(fallback_msg)
        manifest._flush()
        print(f"WARNING: {fallback_msg}", file=sys.stderr)

        try:
            from transformers import TrainerCallback  # type: ignore[import-not-found]

            class FullSequenceLossWarningCallback(TrainerCallback):
                def on_log(self, args, state, control, **kwargs):  # noqa: D401
                    print(f"WARNING: {fallback_msg}", file=sys.stderr)
                    return control

            trainer.add_callback(FullSequenceLossWarningCallback())
        except Exception:
            # The startup warning and manifest entry are the hard guarantees.
            pass

    # --- 8. Checkpointing callback -----------------------------------------
    def _run_tertiary_evals(
        *,
        kind: str,
        path: Path,
        epoch: float | None,
        global_step: int | None,
        metrics: dict[str, float],
    ) -> None:
        if not cfg.train.tertiary_eval_enabled:
            return
        try:
            from discord_sft.evals.benchmarks import split_tasks
            from discord_sft.evals.model import ModelSpec
            from discord_sft.evals.persona import default_hf_generate_fn
            from discord_sft.evals.qwen35_sampling import DEFAULT_QWEN_SAMPLING
            from discord_sft.evals.runner import run_evals

            tasks = (
                list(cfg.train.tertiary_eval_tasks_final or cfg.train.tertiary_eval_tasks_checkpoint)
                if kind == "final"
                else list(cfg.train.tertiary_eval_tasks_checkpoint)
            )
            lmms_tasks, include_persona = split_tasks(tasks)
            # Persona-only in-train evals are preflight-guaranteed. Reuse the
            # live trainer model so we do not load a second 35B (+ LoRA) copy
            # onto the GPU (that duplicate load caused CUDA OOM on one-GPU runs).
            generate_fn = None
            if include_persona and not lmms_tasks:
                generate_fn = default_hf_generate_fn(
                    model,
                    tokenizer,
                    qwen_sampling=DEFAULT_QWEN_SAMPLING,
                )

            spec = ModelSpec(
                name_or_path=cfg.model.name,
                backend="hf",
                adapter_path=str(path),
                dtype="bfloat16" if cfg.train.bf16 else "float16",
            )
            was_training = bool(model.training)
            if generate_fn is not None:
                model.eval()
            try:
                run = run_evals(
                    spec,
                    tasks=tasks,
                    val_jsonl=cfg.data.val_path,
                    profile_json=cfg.train.tertiary_eval_profile_json,
                    out_dir=cfg.train.tertiary_eval_out_dir,
                    limit=cfg.train.tertiary_eval_limit,
                    batch_size=cfg.train.tertiary_eval_batch_size,
                    label=f"{cfg.run_name}-{kind}-{global_step}",
                    generate_fn=generate_fn,
                )
            finally:
                if was_training:
                    model.train()
            ter_metrics = _to_wandb_tertiary_metrics(dict(run.get("scores", {})))
            if wandb_module is not None and ter_metrics:
                ter_metrics["train/global_step"] = float(global_step or 0)
                wandb_module.log(ter_metrics, step=global_step or 0)
            manifest.data.setdefault("events", []).append(
                {
                    "type": "tertiary_eval",
                    "status": "ok",
                    "kind": kind,
                    "checkpoint_path": str(path),
                    "global_step": global_step,
                    "epoch": epoch,
                    "run_id": run.get("run_id"),
                    "n_scores": len(run.get("scores", {})),
                }
            )
            manifest._flush()
        except Exception as e:
            warn = (
                f"tertiary eval failed for {kind} checkpoint at {path}: "
                f"{type(e).__name__}: {e}"
            )
            manifest.data.setdefault("warnings", []).append(warn)
            manifest.data.setdefault("events", []).append(
                {
                    "type": "tertiary_eval",
                    "status": "failed",
                    "kind": kind,
                    "checkpoint_path": str(path),
                    "global_step": global_step,
                    "epoch": epoch,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            manifest._flush()
            print(f"WARNING: {warn}", file=sys.stderr)

    ckpt_cb = make_checkpoint_callback(
        output_dir=output_dir,
        manifest=manifest,
        save_strategy=cfg.checkpoint.save_strategy,
        save_steps=cfg.checkpoint.save_steps,
        save_total_limit=cfg.checkpoint.save_total_limit,
        on_checkpoint_saved=_run_tertiary_evals,
    )
    trainer.add_callback(ckpt_cb)

    # --- 9. Train ----------------------------------------------------------
    try:
        trainer.train()
    except Exception as e:
        manifest.finish(status="failed", error=f"{type(e).__name__}: {e}")
        if wandb_module is not None and wandb_module.run is not None:
            wandb_module.finish(exit_code=1)
        raise

    # --- 10. Final adapter dump -------------------------------------------
    if cfg.checkpoint.save_final:
        final_path = output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_path))
        try:
            tokenizer.save_pretrained(str(final_path))
        except Exception:
            pass
        # Pull the most recent losses for the final entry.
        metrics: dict[str, float] = {}
        for row in reversed(trainer.state.log_history):
            if "loss" in row and "train_loss" not in metrics:
                metrics["train_loss"] = float(row["loss"])
            if "eval_loss" in row and "eval_loss" not in metrics:
                metrics["eval_loss"] = float(row["eval_loss"])
            if "train_loss" in metrics and "eval_loss" in metrics:
                break
        manifest.record_checkpoint(
            kind="final",
            path=final_path,
            epoch=trainer.state.epoch,
            global_step=trainer.state.global_step,
            metrics=metrics,
        )
        _run_tertiary_evals(
            kind="final",
            path=final_path,
            epoch=trainer.state.epoch,
            global_step=trainer.state.global_step,
            metrics=metrics,
        )

    manifest.finish(status="succeeded")
    if wandb_module is not None and wandb_module.run is not None:
        wandb_module.finish()

    # Re-dump the resolved config with expanded layers for final provenance.
    dump_yaml(resolved_cfg, output_dir / "config.resolved.yaml")

    return {
        "run_name": cfg.run_name,
        "output_dir": str(output_dir.resolve()),
        "manifest": manifest.data,
    }


__all__ = ["run_training"]
