"""Shared subprocess log streaming for Streamlit (train / eval run)."""

from __future__ import annotations

import subprocess
import sys
import threading
from queue import Empty, Queue
from typing import Any

import streamlit as st


def stream_subprocess(cmd: list[str], log_area: Any) -> int:
    """Run *cmd*, stream combined stdout/stderr into *log_area*, return exit code."""
    q: Queue[str] = Queue()

    def _reader(stream: Any) -> None:
        for line in stream:
            q.put(line.rstrip())
        q.put("__EOF__")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    threading.Thread(target=_reader, args=(proc.stdout,), daemon=True).start()

    buf: list[str] = []
    while True:
        try:
            line = q.get(timeout=0.5)
        except Empty:
            if proc.poll() is not None and q.empty():
                break
            continue
        if line == "__EOF__":
            break
        buf.append(line)
        log_area.code("\n".join(buf[-200:]), language="text")

    rc = proc.wait()
    if rc == 0:
        st.success("Run complete.")
    else:
        st.error(f"Run failed with exit code {rc}.")
    return int(rc)


def shlex_join_cmd(cmd: list[str]) -> str:
    import shlex

    return shlex.join([str(part) for part in cmd if str(part)])


def build_train_cmd(
    *,
    config: str,
    output_dir: str | None,
    run_name: str | None,
) -> list[str]:
    parts = [sys.executable, "-m", "discord_sft.cli", "train", "--config", config]
    if output_dir and str(output_dir).strip():
        parts += ["--output-dir", str(output_dir).strip()]
    if run_name and str(run_name).strip():
        parts += ["--run-name", str(run_name).strip()]
    return parts
