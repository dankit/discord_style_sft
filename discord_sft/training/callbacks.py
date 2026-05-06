"""Checkpointing + run-manifest logic for a LoRA fine-tuning run.

Two responsibilities:

* :class:`LoRACheckpointCallback` — adapter-only dumps at epoch boundaries
  and optional fixed step intervals, with a rolling ``save_total_limit``
  eviction policy that never touches the ``final/`` dir.
* :class:`RunManifest` — an append-only ``run.json`` that records every
  checkpoint alongside the resolved config, so the eval side can answer
  "which config produced this adapter?" from a single file.

The callback is intentionally independent of Hugging Face's built-in
``save_strategy`` — we run alongside it, writing nicely-named copies to
``{output_dir}/epoch-{N}/`` and ``step-{S}/`` so the eval CLI can point at
them directly (``discord-sft eval run --adapter out/lora/<run>/epoch-3``)
without having to decode the trainer's ``checkpoint-<global_step>/``
naming.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from discord_sft.training.config import TrainConfig, dump_yaml

if TYPE_CHECKING:  # heavy imports are deferred to trainer.py
    from transformers import (
        TrainerCallback,  # type: ignore[import-not-found]
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return None


def _find_uv_lock(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` looking for a ``uv.lock`` at the repo root.

    Returns ``None`` if no lockfile is found (e.g. user installed with plain
    pip, or this repo was vendored into a larger tree without the lock).
    """
    cur = (start or Path.cwd()).resolve()
    for parent in (cur, *cur.parents):
        cand = parent / "uv.lock"
        if cand.is_file():
            return cand
    return None


def _uv_lock_fingerprint() -> dict[str, str | None]:
    """Return ``{"path": ..., "sha256": ...}`` for the active ``uv.lock``.

    Best-effort: any IO or hashing error downgrades to ``{"path": None,
    "sha256": None}`` rather than aborting training. The point is to snapshot
    the frozen dependency set alongside the git SHA so a checkpoint can be
    reproduced byte-for-byte (or at minimum, mismatches can be flagged at
    eval time).
    """
    lock = _find_uv_lock()
    if lock is None:
        return {"path": None, "sha256": None}
    try:
        h = hashlib.sha256()
        with lock.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return {"path": str(lock), "sha256": h.hexdigest()}
    except OSError:
        return {"path": str(lock), "sha256": None}


class RunManifest:
    """Append-only ``run.json`` + per-checkpoint config mirroring.

    Not a ``TrainerCallback`` — the trainer calls into it explicitly so the
    manifest captures both Trainer-driven events (checkpoints, losses) and
    trainer-external events (setup start/finish, fatal errors).
    """

    def __init__(self, cfg: TrainConfig, source_yaml: str | Path | None):
        self.cfg = cfg
        self.output_dir = Path(cfg.checkpoint.output_dir)
        self.source_yaml = Path(source_yaml) if source_yaml else None
        self.data: dict[str, Any] = {
            "run_name": cfg.run_name,
            "started_at": None,
            "finished_at": None,
            "git_sha": _git_sha(),
            "uv_lock": _uv_lock_fingerprint(),
            "base_model": cfg.model.name,
            "train_path": cfg.data.train_path,
            "val_path": cfg.data.val_path,
            "config_source": None,
            "config_resolved": None,
            "num_hidden_layers": None,
            "status": "pending",
            "error": None,
            "checkpoints": [],
            "best_checkpoint": None,
        }

    # ---- filesystem helpers -------------------------------------------------

    def _resolved_path(self) -> Path:
        return self.output_dir / "config.resolved.yaml"

    def _source_path(self) -> Path:
        return self.output_dir / "config.source.yaml"

    def _manifest_path(self) -> Path:
        return self.output_dir / "run.json"

    def _flush(self) -> None:
        self._manifest_path().write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ---- public API ---------------------------------------------------------

    def initialize(self, num_hidden_layers: int) -> None:
        """Write the resolved config + source copy + manifest stub."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        dump_yaml(self.cfg, self._resolved_path())
        if self.source_yaml and self.source_yaml.exists():
            shutil.copyfile(self.source_yaml, self._source_path())
        else:
            # No separate source YAML (e.g. CLI-built config); resolved == source.
            dump_yaml(self.cfg, self._source_path())

        self.data["started_at"] = _utc_now()
        self.data["status"] = "running"
        self.data["config_source"] = str(self._source_path().relative_to(self.output_dir))
        self.data["config_resolved"] = str(self._resolved_path().relative_to(self.output_dir))
        self.data["num_hidden_layers"] = num_hidden_layers
        self._flush()

    def record_checkpoint(
        self,
        *,
        kind: str,
        path: Path,
        epoch: float | None,
        global_step: int | None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Copy the config trio into ``path`` and append a manifest entry."""
        path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._resolved_path(), path / "config.resolved.yaml")
        shutil.copyfile(self._source_path(), path / "config.source.yaml")

        entry = {
            "kind": kind,  # "epoch" | "step" | "final"
            "path": str(path.resolve()),
            "rel_path": str(path.relative_to(self.output_dir)),
            "epoch": epoch,
            "global_step": global_step,
            "saved_at": _utc_now(),
            "metrics": dict(metrics or {}),
        }
        self.data["checkpoints"].append(entry)
        eval_loss = (metrics or {}).get("eval_loss")
        if isinstance(eval_loss, (int, float)):
            best = self.data.get("best_checkpoint")
            best_loss = None
            if isinstance(best, dict):
                best_loss = best.get("eval_loss")
            if not isinstance(best_loss, (int, float)) or float(eval_loss) < float(best_loss):
                self.data["best_checkpoint"] = {
                    "kind": kind,
                    "path": str(path.resolve()),
                    "rel_path": str(path.relative_to(self.output_dir)),
                    "epoch": epoch,
                    "global_step": global_step,
                    "saved_at": entry["saved_at"],
                    "eval_loss": float(eval_loss),
                }
        self._flush()

    def finish(self, *, status: str = "succeeded", error: str | None = None) -> None:
        self.data["finished_at"] = _utc_now()
        self.data["status"] = status
        self.data["error"] = error
        self._flush()


def _build_callback_class():
    """Lazily construct the callback class against HF's real base class.

    We import ``transformers`` inside the factory so ``config.py`` /
    ``data.py`` stay importable in the lightweight core install path (and
    in unit tests that stub the trainer).
    """
    from transformers import TrainerCallback  # type: ignore[import-not-found]

    class LoRACheckpointCallback(TrainerCallback):
        """Writes adapter-only dumps to named directories.

        Parameters
        ----------
        output_dir:
            Root run directory (``out/lora/<run_name>/``). Subdirs
            ``epoch-N/`` and ``step-S/`` are created inside it.
        manifest:
            The owning :class:`RunManifest`, used to mirror config files
            and append checkpoint rows.
        save_strategy:
            ``"epoch"`` | ``"steps"`` | ``"no"``. With ``"no"``, only the
            final dump (handled by ``trainer.py``) is written.
        save_steps:
            Required when ``save_strategy == "steps"``.
        save_total_limit:
            Keep at most N of the rolling dumps; the ``final/`` dir is
            excluded from the rolling set.
        """

        def __init__(
            self,
            *,
            output_dir: str | Path,
            manifest: RunManifest,
            save_strategy: str = "epoch",
            save_steps: int | None = None,
            save_total_limit: int | None = 3,
            on_checkpoint_saved: Callable[..., None] | None = None,
        ) -> None:
            self.output_dir = Path(output_dir)
            self.manifest = manifest
            self.save_strategy = save_strategy
            self.save_steps = save_steps
            self.save_total_limit = save_total_limit
            self.on_checkpoint_saved = on_checkpoint_saved
            if save_strategy == "steps" and not save_steps:
                raise ValueError(
                    "save_strategy='steps' requires a positive save_steps value"
                )
            self._rolling: list[Path] = []
            self._model = None  # set by on_train_begin via kwargs

        # ---- internal helpers ----------------------------------------------

        def _save_adapter(
            self,
            model: Any,
            tokenizer: Any,
            path: Path,
            *,
            kind: str,
            epoch: float | None,
            global_step: int | None,
            metrics: dict[str, float] | None = None,
        ) -> None:
            path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(path))
            if tokenizer is not None:
                try:
                    tokenizer.save_pretrained(str(path))
                except Exception:
                    # Tokenizer saves are a nicety; not all Unsloth-wrapped
                    # tokenizers implement ``save_pretrained`` fully.
                    pass
            self.manifest.record_checkpoint(
                kind=kind,
                path=path,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
            )
            if self.on_checkpoint_saved is not None:
                self.on_checkpoint_saved(
                    kind=kind,
                    path=path,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=dict(metrics or {}),
                )

        def _evict_old(self) -> None:
            if self.save_total_limit is None or self.save_total_limit <= 0:
                return
            while len(self._rolling) > self.save_total_limit:
                victim = self._rolling.pop(0)
                if victim.exists():
                    shutil.rmtree(victim, ignore_errors=True)

        # ---- TrainerCallback hooks -----------------------------------------

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
            self._model = kwargs.get("model")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.save_strategy != "epoch":
                return control
            model = kwargs.get("model") or self._model
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model is None:
                return control
            epoch = int(round(state.epoch)) if state.epoch is not None else None
            tag = f"epoch-{epoch}" if epoch is not None else f"epoch-step-{state.global_step}"
            path = self.output_dir / tag
            # Pull latest train/eval loss from the log history for this row.
            metrics: dict[str, float] = {}
            for row in reversed(state.log_history):
                if "loss" in row and "train_loss" not in metrics:
                    metrics["train_loss"] = float(row["loss"])
                if "eval_loss" in row and "eval_loss" not in metrics:
                    metrics["eval_loss"] = float(row["eval_loss"])
                if "train_loss" in metrics and "eval_loss" in metrics:
                    break
            self._save_adapter(
                model,
                tokenizer,
                path,
                kind="epoch",
                epoch=state.epoch,
                global_step=state.global_step,
                metrics=metrics,
            )
            self._rolling.append(path)
            self._evict_old()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.save_strategy != "steps":
                return control
            if not self.save_steps or state.global_step <= 0:
                return control
            if state.global_step % self.save_steps != 0:
                return control
            model = kwargs.get("model") or self._model
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model is None:
                return control
            path = self.output_dir / f"step-{state.global_step}"
            metrics: dict[str, float] = {}
            for row in reversed(state.log_history):
                if "loss" in row:
                    metrics["train_loss"] = float(row["loss"])
                    break
            self._save_adapter(
                model,
                tokenizer,
                path,
                kind="step",
                epoch=state.epoch,
                global_step=state.global_step,
                metrics=metrics,
            )
            self._rolling.append(path)
            self._evict_old()
            return control

    return LoRACheckpointCallback


# Public re-export: resolved lazily so importing this module doesn't require
# transformers to be installed.
def make_checkpoint_callback(**kwargs):
    cls = _build_callback_class()
    return cls(**kwargs)


__all__ = ["RunManifest", "make_checkpoint_callback"]
