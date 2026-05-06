"""Resolve filesystem paths embedded in eval artifacts against the git repo root.

Relative paths like ``out/merged/foo`` appear in saved run JSON; they must match
the layout under the repository checkout, not the process ``cwd``.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def eval_repo_root() -> Path:
    """Root of the ``discord-sft`` checkout (parent of ``discord_sft`` package)."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_artifact(raw: str | Path | None) -> Path | None:
    """If ``raw`` resolves to an existing filesystem path, return it; else ``None``.

    Anchors relative paths at :func:`eval_repo_root` instead of ``Path.cwd()`` so
    eval tooling resolves saved run paths consistently regardless of shell cwd.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    p = Path(s).expanduser()
    cand = p.resolve() if p.is_absolute() else (eval_repo_root() / p).resolve()
    return cand if cand.exists() else None


def resolve_local_dir(raw: str | Path | None) -> Path | None:
    """``resolve_repo_artifact`` narrowed to directories only."""
    c = resolve_repo_artifact(raw)
    return c if c is not None and c.is_dir() else None


def resolve_checkpoint_dir(raw: str | Path | None) -> Path | None:
    """Best-effort directory for a checkpoint: repo-relative, then plain absolute path."""
    loc = resolve_local_dir(raw)
    if loc is not None:
        return loc
    s = str(raw or "").strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        return None
    p = p.resolve()
    return p if p.is_dir() else None


def merge_training_run_id_from_model_path(raw: str | Path | None) -> str | None:
    """Training run folder name baked into merged model paths such as ``out/merged/<id>/epoch-…``.

    For ``out/merged/style-late-r16-a16-5epochs/epoch-4-vllm-keys`` returns
    ``style-late-r16-a16-5epochs``. Used to fall back to ``out/lora/<id>/``
    when the merged tree was deleted or never synced.
    """
    s = str(raw or "").strip().replace("\\", "/")
    if not s:
        return None
    parts = PurePosixPath(s).parts
    rid: str | None = None
    if len(parts) >= 3 and parts[0] == "out" and parts[1] == "merged":
        rid = parts[2]
    elif len(parts) >= 2 and parts[0] == "merged":
        rid = parts[1]
    if rid is None or not rid or rid in (".", ".."):
        return None
    if len(PurePosixPath(rid).parts) != 1:
        return None
    return rid


def resolve_lora_training_run_dir(run_name: str) -> Path | None:
    """``out/lora/<run_name>/`` under the repo root, if present."""
    rid = str(run_name or "").strip()
    if not rid or rid in (".", "..") or len(PurePosixPath(rid).parts) != 1:
        return None
    return resolve_local_dir(f"out/lora/{rid}")


__all__ = [
    "eval_repo_root",
    "merge_training_run_id_from_model_path",
    "resolve_checkpoint_dir",
    "resolve_lora_training_run_dir",
    "resolve_repo_artifact",
    "resolve_local_dir",
]
