"""Merge PEFT/Unsloth LoRA checkpoints into dense Hugging Face model weights."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from discord_sft.evals.adapter_training_meta import read_training_base_model_from_adapter
from discord_sft.training.common import set_unsloth_env
from discord_sft.training.config import ModelConfig, TrainConfig, load_config_for_merge_parity


def resolve_base_model_id(
    adapter_dir: Path,
    *,
    override: str | None = None,
) -> str:
    """Prefer explicit override, then ``run.json`` ``base_model``, then PEFT config."""
    if override and override.strip():
        return override.strip()
    run_json = read_training_base_model_from_adapter(adapter_dir)
    if run_json:
        return run_json
    marker = adapter_dir / "adapter_config.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"No adapter_config.json under {adapter_dir}; not a PEFT checkpoint."
        )
    try:
        peft_data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid adapter_config.json: {e}") from e
    raw = peft_data.get("base_model_name_or_path")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    raise ValueError(
        "Could not resolve base model id: set --base-model, or ensure "
        "parent run.json has base_model, or adapter_config.json has "
        "base_model_name_or_path."
    )


def resolve_train_config_for_merge(
    adapter_dir: Path,
    config_path: Path | None,
) -> TrainConfig | None:
    """Load training YAML for dtype / seq length / MoE parity when available."""
    paths: list[Path] = []
    if config_path is not None:
        paths.append(config_path)
    parent = adapter_dir.parent
    paths.append(parent / "config.resolved.yaml")
    for p in paths:
        if p.is_file():
            return load_config_for_merge_parity(p)
    return None


def _read_run_json_sidecar(adapter_dir: Path) -> dict[str, Any] | None:
    run_json = adapter_dir.parent / "run.json"
    if not run_json.is_file():
        return None
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _ensure_multimodal_config_from_hub(repo_id: str, out_dir: Path) -> list[str]:
    """Copy small HF preprocessor files vLLM needs for Qwen3-VL checkpoints.

    Merged checkpoints only write ``tokenizer.*`` via ``tokenizer.save_pretrained``;
    multimodal repos also ship ``preprocessor_config.json`` / ``processor_config.json``
    beside the tokenizer. Without them local ``--model merged/`` startup fails:
    ``Can't load image processor … preprocessor_config.json``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []

    copied: list[str] = []
    for name in (
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ):
        dest = out_dir / name
        if dest.is_file():
            continue
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=name)
            shutil.copy2(cached, dest)
            copied.append(name)
        except Exception:
            continue
    return copied


def _infer_merge_torch_dtype(resolved_16bit: bool) -> Any:
    import torch

    if not resolved_16bit:
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_base_unsloth(
    base_id: str,
    resolved_max: int,
    resolved_16bit: bool,
) -> tuple[Any, Any]:
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_id,
        max_seq_length=resolved_max,
        load_in_4bit=False,
        load_in_16bit=resolved_16bit,
        full_finetuning=False,
    )
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return model, tok


def _load_base_transformers(
    base_id: str,
    resolved_16bit: bool,
) -> tuple[Any, Any]:
    """Load ~plain HF checkpoint when Unsloth is not installed ([evals] venv)."""
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    dtype = _infer_merge_torch_dtype(resolved_16bit)
    load_kw: dict[str, Any] = {"trust_remote_code": True}
    if dtype != torch.float32:
        # transformers>=5.5 prefers ``dtype``; ``torch_dtype`` logs deprecation.
        load_kw["dtype"] = dtype
    try:
        __import__("accelerate")  # noqa: PLC0415
        load_kw["device_map"] = "auto"
    except ImportError:
        pass

    loaders: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("AutoModelForCausalLM", AutoModelForCausalLM.from_pretrained),
        ("AutoModel", AutoModel.from_pretrained),
    )
    last_exc: BaseException | None = None
    model = None
    for _, from_pretrained in loaders:
        try:
            model = from_pretrained(base_id, **load_kw)
            break
        except Exception as exc:  # noqa: BLE001 — try next loader shape
            last_exc = exc
            continue
    if model is None:
        hints = ", ".join(l[0] for l in loaders)
        raise RuntimeError(
            f"Could not load {base_id!r} with {hints}; last error: {last_exc!r}"
        ) from last_exc

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    return model, tokenizer


def _library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    try:
        import transformers  # type: ignore[import-not-found]

        out["transformers"] = getattr(transformers, "__version__", None)
    except Exception:
        out["transformers"] = None
    try:
        import unsloth  # type: ignore[import-not-found]

        out["unsloth"] = getattr(unsloth, "__version__", None)
    except Exception:
        out["unsloth"] = None
    try:
        import peft  # type: ignore[import-not-found]

        out["peft"] = getattr(peft, "__version__", None)
    except Exception:
        out["peft"] = None
    try:
        import torch  # type: ignore[import-not-found]

        out["torch"] = getattr(torch, "__version__", None)
    except Exception:
        out["torch"] = None
    return out


def merge_peft_adapter_to_hf_dir(
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
    config_path: str | Path | None = None,
    max_seq_length: int | None = None,
    load_in_16bit: bool | None = None,
    max_shard_size: str = "5GB",
) -> dict[str, Any]:
    """Load base model, attach PEFT weights, merge, save dense HF tree.

    Prefers **Unsloth ``FastModel``** when installed (parity with ``train run``).
    Otherwise uses **Transformers** only so ``discord-sft[evals]`` (torch + peft +
    transformers) can merge adapters without pulling ``[train]``.

    Returns
    -------
    dict
        Summary including paths, base model id, and optional manifest path.
    """
    adapter_path = Path(adapter_dir).resolve()
    out_path = Path(output_dir).resolve()
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Missing adapter_config.json under {adapter_path}"
        )
    out_path.mkdir(parents=True, exist_ok=True)

    cfg_yaml = Path(config_path).resolve() if config_path else None
    train_cfg = resolve_train_config_for_merge(adapter_path, cfg_yaml)
    model_defaults = train_cfg.model if train_cfg is not None else ModelConfig()
    moe_backend = model_defaults.moe_backend
    set_unsloth_env(moe_backend)

    base_id = resolve_base_model_id(adapter_path, override=base_model)

    resolved_max = (
        int(max_seq_length)
        if max_seq_length is not None
        else int(model_defaults.max_seq_length)
    )
    resolved_16bit = (
        bool(load_in_16bit) if load_in_16bit is not None else bool(model_defaults.load_in_16bit)
    )

    try:
        from peft import PeftModel  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "merge-peft requires peft (+ torch + transformers): "
            "`pip install -e '.[evals]'` or `pip install -e '.[train]'`."
        ) from e

    merge_backend: str
    try:
        __import__("unsloth")
    except ImportError:
        model, tokenizer = _load_base_transformers(base_id, resolved_16bit)
        merge_backend = "transformers"
    else:
        model, tokenizer = _load_base_unsloth(
            base_id, resolved_max, resolved_16bit
        )
        merge_backend = "unsloth"

    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_path), max_shard_size=max_shard_size)
    tokenizer.save_pretrained(str(out_path))
    hub_mm_files = _ensure_multimodal_config_from_hub(base_id, out_path)

    copied_config_resolved: str | None = None
    cfg_src = adapter_path.parent / "config.resolved.yaml"
    cfg_dst = out_path / "config.resolved.yaml"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, cfg_dst)
        copied_config_resolved = str(cfg_dst.resolve())

    manifest: dict[str, Any] = {
        "adapter_dir": str(adapter_path),
        "output_dir": str(out_path),
        "base_model": base_id,
        "max_seq_length": resolved_max,
        "load_in_16bit": resolved_16bit,
        "moe_backend": moe_backend,
        "copied_config_resolved": copied_config_resolved,
        "train_config_yaml": (
            str(cfg_yaml.resolve()) if cfg_yaml and cfg_yaml.is_file() else None
        ),
        "auto_loaded_config_resolved": (
            str((adapter_path.parent / "config.resolved.yaml").resolve())
            if train_cfg is not None and cfg_yaml is None
            else None
        ),
        "hub_multimodal_files_copied": hub_mm_files,
        "merge_backend": merge_backend,
    }
    run_side = _read_run_json_sidecar(adapter_path)
    if run_side:
        manifest["training_git_sha"] = run_side.get("git_sha")
        manifest["training_run_name"] = run_side.get("run_name")
    manifest["library_versions"] = _library_versions()

    mf_path = out_path / "merge_manifest.json"
    mf_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest["merge_manifest_path"] = str(mf_path)
    return manifest


__all__ = [
    "merge_peft_adapter_to_hf_dir",
    "resolve_base_model_id",
    "resolve_train_config_for_merge",
]
