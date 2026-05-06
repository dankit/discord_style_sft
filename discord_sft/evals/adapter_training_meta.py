"""Read training metadata saved next to LoRA checkpoints (for eval parity)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_training_base_model_from_adapter(adapter_dir: str | os.PathLike[str]) -> str | None:
    """Return ``base_model`` from the training ``run.json`` for this checkpoint.

    Checkpoints live under ``<output_dir>/epoch-N/``, ``<output_dir>/final/``, etc.
    The manifest ``run.json`` is written at ``<output_dir>/run.json``.

    Returns ``None`` if the path is not a checkpoint dir or ``run.json`` is missing.
    """
    root = Path(adapter_dir).resolve()
    marker = root / "adapter_config.json"
    if not marker.is_file():
        return None
    run_json = root.parent / "run.json"
    if not run_json.is_file():
        return None
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    name = data.get("base_model")
    return str(name).strip() if isinstance(name, str) and name.strip() else None


__all__ = ["read_training_base_model_from_adapter"]
