from __future__ import annotations

import os
from typing import Any

USER_TURN_PREFIX = "<|im_start|>user\n"
ASSISTANT_TURN_PREFIX = "<|im_start|>assistant\n"


def set_unsloth_env(moe_backend: str | None) -> None:
    """Apply Unsloth env flags before importing heavy training libraries."""
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
    if moe_backend:
        os.environ["UNSLOTH_MOE_BACKEND"] = str(moe_backend)


def num_hidden_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("Loaded model has no .config attribute")
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None:
        inner = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None)
        if inner is not None:
            n = getattr(inner, "num_hidden_layers", None)
    if n is None:
        raise RuntimeError("Could not determine num_hidden_layers from model config")
    return int(n)
