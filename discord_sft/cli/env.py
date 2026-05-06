from __future__ import annotations

from pathlib import Path


def _load_local_dotenv() -> None:
    """Best-effort load of repository .env without overriding shell exports."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except Exception:
        return
    load_dotenv(dotenv_path=Path(".env"), override=False)
