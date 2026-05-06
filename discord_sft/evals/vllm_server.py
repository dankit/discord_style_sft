"""Launch + manage a local vLLM OpenAI-compatible server for eval runs.

Why a server instead of in-process vLLM?
----------------------------------------
At 35B-A3B scale the base model load dominates wall-clock. A single
``discord-sft eval run`` invocation typically wants to:

1. Run lmms-eval for the standard benchmarks (as a subprocess, because
   lmms-eval's internal Python API churns between versions).
2. Run the native persona evals in-process.
3. Repeat (1)+(2) once per LoRA adapter in a sweep.

If each of those loaded the base model separately we'd pay the load cost
3+ times. Instead we stand up one vLLM server per ``eval run`` with every
adapter registered via ``--lora-modules alias=path``. Both lmms-eval
(``--model openai --model_args base_url=...,model_version=<alias>``) and
the persona eval (``openai.chat.completions.create`` against the same
URL) hit the same process. vLLM's continuous batching + paged attention
then do the heavy lifting.

Design notes
------------
- We deliberately shell out (``python -m vllm.entrypoints.openai.api_server``)
  rather than import ``vllm`` as a library. Keeps ``discord_sft.evals``
  importable without torch, survives vLLM internal refactors, and makes
  stdout easy to forward into the UI live log.
- Readiness is detected by polling ``GET /v1/models`` until every
  expected alias shows up in the ``data`` list, with a hard timeout so
  we don't hang forever on a bad config.
- On Windows we don't have SIGTERM, so ``stop()`` uses
  ``Popen.terminate()`` which maps to ``TerminateProcess`` on win32;
  that's the standard vLLM shutdown path in the CLI too.

- Spawned subprocess uses :func:`python_for_vllm_subprocess`: prefers CLI
  ``--vllm-python``, then ``DISCORD_SFT_VLLM_PYTHON``, then (when
  ``VIRTUAL_ENV`` disagrees with ``sys.executable``) the activated venv
  interpreter so vLLM matches the intended torch stack.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


def _venv_python_binary(venv_root: Path) -> str | None:
    """Return ``…/bin/python`` for *this* venv, or None.

    **Must** be the path inside the venv (e.g. ``.venv-evals/bin/python3``), not
    the resolved system interpreter: on Linux ``bin/python3`` is often a
    symlink to ``/usr/bin/python3.*`` — invoking the symlink preserves
    ``pyvenv.cfg``/``site-packages``; spawning ``/usr/bin/python3.*`` alone does not.
    """
    if sys.platform == "win32":
        bindir = venv_root / "Scripts"
        names = ("python.exe", "python3.exe")
    else:
        bindir = venv_root / "bin"
        names = ("python3", "python")
    for name in names:
        cand = bindir / name
        try:
            if cand.is_file():
                return os.fspath(cand.expanduser())
        except OSError:
            continue
    return None


def python_for_vllm_subprocess(*, explicit: str | None = None) -> str:
    """Python binary for ``python -m vllm.entrypoints.openai.api_server``.

    Resolution order:

    1. ``explicit`` (from ``--vllm-python`` / :attr:`VLLMServerConfig.python_executable`).
    2. ``DISCORD_SFT_VLLM_PYTHON`` when set in the environment.
    3. If ``VIRTUAL_ENV`` points at an existing venv, its ``bin/python`` (verbatim path,
       no symlink collapsing — see :func:`_venv_python_binary`).
    4. Else if ``sys.prefix != sys.base_prefix`` (running inside *some* venv), same
       for :data:`sys.prefix` (covers ``uv run`` without exporting ``VIRTUAL_ENV``).
    5. Resolved :data:`sys.executable`.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    override = os.environ.get("DISCORD_SFT_VLLM_PYTHON")
    if override and override.strip():
        return override.strip()

    roots: list[Path] = []
    venv_override = os.environ.get("VIRTUAL_ENV")
    if venv_override:
        roots.append(Path(venv_override).expanduser().resolve())
    base_pf = getattr(sys, "base_prefix", sys.prefix)
    if sys.prefix != base_pf:
        roots.append(Path(sys.prefix).resolve())

    seen: set[Path] = set()
    for root in roots:
        key = root.resolve()
        if key in seen:
            continue
        seen.add(key)
        picked = _venv_python_binary(key)
        if picked:
            return picked

    return str(Path(sys.executable).resolve())


def alias_for(path: str | os.PathLike[str]) -> str:
    """Derive a vLLM LoRA alias from an adapter path.

    vLLM's server uses the alias as the ``model`` in request bodies and in
    ``/v1/models``. It should match ``[A-Za-z0-9._-]+``; anything else
    becomes ``-``. We keep it close to the ``ModelSpec.slug()`` style so
    run ids and LoRA aliases line up in logs.
    """
    p = Path(str(path))
    base = p.name or p.parent.name or "lora"
    if base == "final" and p.parent.name:
        base = p.parent.name
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "-" for c in base.lower())
    cleaned = cleaned.strip("-._") or "lora"
    return cleaned


def _looks_like_alias(s: str) -> bool:
    """``True`` if ``s`` is a plausible bare alias (no path separators)."""
    if not s:
        return False
    for ch in s:
        if not (ch.isalnum() or ch in "._-"):
            return False
    return True


def adapters_from_cli(raw: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``--adapter`` values into ``(alias, abs_path)`` pairs.

    Supports both bare ``/abs/path`` (alias auto-derived) and explicit
    ``alias=/abs/path``. The ``alias=`` form is only recognised when the
    left-hand side is a plain identifier (``[A-Za-z0-9._-]+``) so Windows
    drive letters in paths like ``C:\\foo`` don't get mis-split.

    Duplicate aliases are disambiguated with a ``-2``, ``-3`` suffix since
    some vLLM versions silently drop a later LoRA with a colliding name.
    """
    out: list[tuple[str, str]] = []
    used: set[str] = set()
    for item in raw:
        item = item.strip()
        if not item:
            continue
        alias: str
        path: str
        if "=" in item:
            head, tail = item.split("=", 1)
            if _looks_like_alias(head.strip()):
                alias = head.strip()
                path = tail.strip()
            else:
                alias = alias_for(item)
                path = item
        else:
            alias = alias_for(item)
            path = item
        abspath = str(Path(path).resolve())
        base = alias
        i = 2
        while alias in used:
            alias = f"{base}-{i}"
            i += 1
        used.add(alias)
        out.append((alias, abspath))
    return out


def discover_adapter_dirs(root: str | os.PathLike[str]) -> list[str]:
    """Discover LoRA adapter directories under ``root``.

    Training runs often save both ``epoch-N/`` and ``final/`` adapters. For
    sweep evals, prefer final adapters when any are present; otherwise return
    every directory containing an ``adapter_config.json``.
    """

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {root}")
    if root_path.is_file():
        raise ValueError(f"Adapter discovery expects a directory, got file: {root}")

    adapter_dirs = sorted(
        {p.parent.resolve() for p in root_path.rglob("adapter_config.json")},
        key=lambda p: str(p),
    )
    final_dirs = [p for p in adapter_dirs if p.name == "final"]
    picks = final_dirs or adapter_dirs
    return [str(p) for p in picks]


def _free_port() -> int:
    """Ask the OS for a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _detect_lora_rank(path: str) -> int | None:
    """Read ``adapter_config.json`` to learn the adapter's rank.

    Returns ``None`` if the file is missing or unparseable so the caller
    can fall back to vLLM's default (16).
    """
    cfg = Path(path) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        doc = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    r = doc.get("r") or doc.get("lora_r")
    if isinstance(r, int) and r > 0:
        return r
    return None


def max_lora_rank(adapters: Iterable[tuple[str, str]], *, default: int = 16) -> int:
    """Max adapter rank across a sweep. Over-provisioning wastes GPU mem."""
    best = 0
    for _, path in adapters:
        r = _detect_lora_rank(path)
        if r is not None and r > best:
            best = r
    return best or default


@dataclass
class VLLMServerConfig:
    """Tunables for the local vLLM server.

    Defaults target a single H100 80GB running Qwen3.5-35B-A3B in bf16.
    Override via CLI flags / ``discord_sft.cli`` if you swap models.
    """

    model: str
    adapters: list[tuple[str, str]] = field(default_factory=list)
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    max_lora_rank: int = 16
    reasoning_parser: str | None = "qwen3"
    quantization: str | None = None
    trust_remote_code: bool = True
    port: int | None = None
    host: str = "127.0.0.1"
    extra_args: list[str] = field(default_factory=list)
    startup_timeout_sec: float = 900.0
    readiness_poll_sec: float = 2.0
    python_executable: str | None = None


def build_server_argv(cfg: VLLMServerConfig, port: int) -> list[str]:
    """Build the argv list for ``python -m vllm.entrypoints.openai.api_server``.

    Deliberately skips ``--tensor-parallel-size`` — we are single-GPU only
    in this harness, and vLLM defaults to TP=1. Passing it explicitly
    would have no effect but clutters the command log.
    """
    argv: list[str] = [
        python_for_vllm_subprocess(explicit=cfg.python_executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        str(port),
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    if cfg.trust_remote_code:
        argv.append("--trust-remote-code")
    if cfg.quantization:
        argv.extend(["--quantization", cfg.quantization])
    if cfg.reasoning_parser:
        argv.extend(["--reasoning-parser", cfg.reasoning_parser])
    if cfg.adapters:
        argv.append("--enable-lora")
        argv.extend(["--max-lora-rank", str(cfg.max_lora_rank)])
        argv.append("--lora-modules")
        for alias, path in cfg.adapters:
            argv.append(f"{alias}={path}")
    argv.extend(cfg.extra_args)
    return argv


class VLLMServer:
    """Context manager around a local vLLM OpenAI-compatible server.

    Usage::

        cfg = VLLMServerConfig(model="Qwen/Qwen3.5-35B-A3B",
                               adapters=[("r8", "/abs/out/lora/r8")])
        with VLLMServer(cfg) as server:
            print(server.base_url)  # http://127.0.0.1:XXXXX/v1
            # run lmms-eval pointing at server.base_url
            # run persona gen pointing at server.base_url

    The context manager always reclaims the process on exit, even if
    startup fails mid-load. Output lines are forwarded to ``on_line`` as
    they arrive so the caller (CLI / UI) can stream progress live.
    """

    def __init__(
        self,
        cfg: VLLMServerConfig,
        *,
        on_line: Callable[[str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_line = on_line
        self._proc: subprocess.Popen[str] | None = None
        self._log_thread: threading.Thread | None = None
        self._log_lines: list[str] = []
        self._port = cfg.port or _free_port()
        self._argv: list[str] = []

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        """The ``/v1`` base URL OpenAI clients expect."""
        return f"http://{self.cfg.host}:{self._port}/v1"

    @property
    def argv(self) -> list[str]:
        return list(self._argv)

    def aliases(self) -> list[str]:
        """Expected LoRA aliases (used for readiness)."""
        return [a for a, _ in self.cfg.adapters]

    def start(self) -> "VLLMServer":
        self._argv = build_server_argv(self.cfg, self._port)
        env = os.environ.copy()
        # vLLM v1 can run DeepGEMM warmup during kernel init; without a working
        # `deep_gemm` wheel this raises (common on GH200 nightlies). Eval README
        # uses VLLM_USE_DEEP_GEMM=0; default here so eval runs work without a
        # manual export. Override by setting VLLM_USE_DEEP_GEMM in the shell.
        if "VLLM_USE_DEEP_GEMM" not in env:
            env["VLLM_USE_DEEP_GEMM"] = "0"
            self._emit(
                "[vllm-server] VLLM_USE_DEEP_GEMM=0 (default; skip DeepGEMM warmup)"
            )
        # Surface LoRA runtime updates if the caller pre-set them; we don't
        # force it on, since the docs flag it as production-risky.
        self._emit(f"[vllm-server] starting: {' '.join(self._argv)}")
        self._proc = subprocess.Popen(
            self._argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._log_thread = threading.Thread(
            target=self._drain_stdout, name="vllm-server-log", daemon=True
        )
        self._log_thread.start()
        self._wait_for_ready()
        return self

    def _drain_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            self._log_lines.append(line)
            if len(self._log_lines) > 2000:
                # cap memory: keep last 2000 lines
                del self._log_lines[:500]
            if self._on_line is not None:
                try:
                    self._on_line(line)
                except Exception:
                    pass

    def _emit(self, line: str) -> None:
        self._log_lines.append(line)
        if self._on_line is not None:
            try:
                self._on_line(line)
            except Exception:
                pass

    def _wait_for_ready(self) -> None:
        deadline = time.monotonic() + self.cfg.startup_timeout_sec
        url = f"http://{self.cfg.host}:{self._port}/v1/models"
        expected = set(self.aliases())
        last_err: str | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                tail = "\n".join(self._log_lines[-40:])
                raise RuntimeError(
                    f"vLLM server exited early with code "
                    f"{self._proc.returncode}. Last output:\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                data = body.get("data") or []
                ids = {row.get("id") for row in data if isinstance(row, dict)}
                if expected and not expected.issubset(ids):
                    missing = expected - ids
                    last_err = f"waiting for aliases {sorted(missing)}"
                else:
                    self._emit(f"[vllm-server] ready at {self.base_url}")
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last_err = str(e)
            time.sleep(self.cfg.readiness_poll_sec)
        tail = "\n".join(self._log_lines[-40:])
        raise TimeoutError(
            f"vLLM server did not become ready within "
            f"{self.cfg.startup_timeout_sec:.0f}s ({last_err}). "
            f"Last output:\n{tail}"
        )

    def stop(self, *, timeout: float = 30.0) -> None:
        if self._proc is None:
            return
        proc = self._proc
        if proc.poll() is None:
            self._emit("[vllm-server] stopping")
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=timeout)
                except Exception:
                    pass
        self._proc = None
        if self._log_thread is not None:
            self._log_thread.join(timeout=2.0)
            self._log_thread = None

    def log_tail(self, n: int = 50) -> list[str]:
        return list(self._log_lines[-n:])

    def __enter__(self) -> "VLLMServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _is_qwen35_model(model_name_or_path: str | None) -> bool:
    if not model_name_or_path:
        return False
    s = model_name_or_path.lower()
    return "qwen3.5" in s or "qwen3_5" in s


def _ensure_qwen35_transformers_support(model_name_or_path: str | None) -> None:
    """Fail fast when transformers cannot parse qwen3_5_moe configs.

    This catches the common GH200 failure mode early with a concise fix
    instead of letting the vLLM server crash during startup.
    """
    if not _is_qwen35_model(model_name_or_path):
        return
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as e:
        raise ImportError(
            "Transformers is required for Qwen3.5 config parsing. "
            "Install a recent build, e.g. "
            "`uv pip install -U \"transformers[serving] @ "
            "git+https://github.com/huggingface/transformers.git@main\"`."
        ) from e
    if "qwen3_5_moe" not in CONFIG_MAPPING:
        cur = "unknown"
        try:
            cur = importlib_metadata.version("transformers")
        except importlib_metadata.PackageNotFoundError:
            pass
        raise ImportError(
            "This environment's transformers build does not support "
            "Qwen3.5 (`qwen3_5_moe`). "
            f"Found transformers=={cur}. "
            "Create a fresh env and install a recent transformers + vLLM "
            "stack with uv (Qwen model card recommendation), then rerun eval."
        )


def ensure_vllm_available(
    model_name_or_path: str | None = None,
    *,
    spawn_python: str | None = None,
) -> None:
    """Fail fast with a friendly error if ``vllm`` isn't importable.

    vLLM is a heavy optional dep; we don't want to crash on module import
    of the harness core, so callers invoke this right before ``start()``.
    """
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Install the evals extra with vLLM on Linux aarch64: "
            "`uv sync --extra evals`, then on GH200 run `bash scripts/setup_gh200_evals.sh` "
            "if you use transformers@main + lmms-eval overlays."
        ) from e
    _ensure_qwen35_transformers_support(model_name_or_path)
    # Also sanity-check we can actually spawn the api_server entrypoint.
    spawn_py = Path(
        python_for_vllm_subprocess(explicit=spawn_python)
    ).resolve()
    if not spawn_py.is_file():
        raise RuntimeError(
            f"Python interpreter {spawn_py} is missing; "
            "cannot launch vllm.entrypoints.openai.api_server."
        )


__all__ = [
    "VLLMServer",
    "VLLMServerConfig",
    "adapters_from_cli",
    "alias_for",
    "build_server_argv",
    "discover_adapter_dirs",
    "ensure_vllm_available",
    "max_lora_rank",
    "python_for_vllm_subprocess",
]
