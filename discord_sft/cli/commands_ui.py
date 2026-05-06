from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "Install the UI extra: pip install 'discord-sft[ui]'\n"
        )
        return 1
    import discord_sft

    app_path = Path(discord_sft.__file__).resolve().parent / "ui.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]
    proc = subprocess.Popen(cmd)
    try:
        return int(proc.wait())
    except KeyboardInterrupt:
        sys.stderr.write("\nStopping Streamlit\n")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 130
