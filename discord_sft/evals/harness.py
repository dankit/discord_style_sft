"""Drive ``python -m lmms_eval`` as a subprocess and parse its results JSON.

Design notes
------------
- We intentionally shell out rather than import ``lmms_eval`` as a library.
  Their per-version Python API (task manager internals, model wrappers)
  churns more than the stable CLI + ``--output_path`` JSON contract. This
  keeps the harness robust to upstream refactors and makes the UI's live
  progress output trivial (just tail stdout).
- ``parse_lmms_results`` is split out so we can unit-test the schema parser
  against a captured fixture without invoking lmms-eval.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from discord_sft.evals.model import ModelSpec


def build_command(
    spec: ModelSpec,
    tasks: list[str],
    *,
    output_path: str | Path,
    limit: int | None = None,
    batch_size: int | str = 1,
    num_fewshot: int | None = None,
    log_samples: bool = True,
    extra_cli: Iterable[str] = (),
    server_base_url: str | None = None,
) -> list[str]:
    """Build the argv list for ``python -m lmms_eval ...``.

    We favour the built-in ``hf``/``vllm``/``sglang``/dedicated-VLM backends
    that ship with lmms-eval (see their ``examples/models/*.sh``). The
    ``--model_args`` payload is produced by :meth:`ModelSpec.to_lmms_eval_args`.

    When ``spec.backend == "openai"`` (or ``server_base_url`` is supplied),
    we render the command for lmms-eval's openai backend pointed at a
    local vLLM OpenAI-compatible server. That path is how a multi-LoRA
    sweep avoids reloading the 35B base between runs: one server,
    ``--lora-modules`` pre-registers each alias, and each lmms-eval
    invocation differs only in ``model_version=<alias>``.
    """
    if not tasks:
        raise ValueError("build_command: empty tasks list")

    use_openai = spec.backend == "openai" or server_base_url is not None
    if use_openai:
        model_arg = "openai"
        payload = spec.to_openai_lmms_args(base_url=server_base_url)
    else:
        model_arg = spec.backend
        payload = spec.to_lmms_eval_args()

    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        model_arg,
        "--model_args",
        payload,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if log_samples:
        cmd.append("--log_samples")
    cmd.extend(extra_cli)
    return cmd


def run_lmms_eval(
    spec: ModelSpec,
    tasks: list[str],
    *,
    output_path: str | Path,
    limit: int | None = None,
    batch_size: int | str = 1,
    num_fewshot: int | None = None,
    on_line: Callable[[str], None] | None = None,
    extra_cli: Iterable[str] = (),
    server_base_url: str | None = None,
) -> dict[str, Any]:
    """Run lmms-eval, stream its stdout, then parse the results JSON it wrote.

    ``on_line`` receives each decoded stdout line — use it to forward live
    progress to a Streamlit code block. Returns the harness result dict:
    ``{"scores": {"ifeval.prompt_level_strict_acc": 0.64, ...},
       "raw_results_path": "<dir>", "tasks": [...], "returncode": 0}``.
    Raises ``RuntimeError`` on non-zero exit (after surfacing the tail of
    stdout in the exception message).
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        spec,
        tasks,
        output_path=output_dir,
        limit=limit,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        extra_cli=extra_cli,
        server_base_url=server_base_url,
    )
    lines: list[str] = []
    # lmms-eval constructs openai.OpenAI(); recent SDKs reject missing keys even
    # for self-hosted vLLM. Our model_args include api_key=EMPTY, but some stacks
    # only read OPENAI_API_KEY — set a harmless default for the managed-server path.
    proc_env = os.environ.copy()
    if server_base_url is not None and not proc_env.get("OPENAI_API_KEY", "").strip():
        proc_env["OPENAI_API_KEY"] = "EMPTY"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=proc_env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            if on_line is not None:
                on_line(line)
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
    if rc != 0:
        tail = "\n".join(lines[-40:])
        raise RuntimeError(
            f"lmms-eval exited with code {rc}\nLast output:\n{tail}"
        )

    results_path = _find_results_json(output_dir)
    scores = parse_lmms_results(results_path) if results_path else {}
    return {
        "scores": scores,
        "raw_results_path": str(output_dir),
        "results_json": str(results_path) if results_path else None,
        "tasks": list(tasks),
        "returncode": rc,
        "cmd": cmd,
    }


def _find_results_json(output_dir: Path) -> Path | None:
    """Locate the latest aggregated results JSON lmms-eval wrote under ``output_dir``.

    Older layouts used ``results_<ts>.json`` directly under ``--output_path``.
    Newer lmms-eval (``evaluation_tracker.save_results_aggregated``) writes
    ``<output_path>/<model_sanitized>/<date_id>_results.json``, which does not
    match a ``results*.json`` prefix glob. We therefore accept any ``*.json``
    whose name starts with ``results`` or ends with ``_results.json``, and
    skip per-task sample sidecars that embed ``_samples_`` in the filename.
    """
    if not output_dir.exists():
        return None
    candidates: list[Path] = []
    for p in output_dir.rglob("*.json"):
        name = p.name
        if "_samples_" in name:
            continue
        if name.startswith("results") or name.endswith("_results.json"):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_lmms_results(path: str | Path) -> dict[str, float]:
    """Flatten lmms-eval's results.json into our dotted-key score dict.

    Schema (post-v0.4): ``{"results": {"<task>": {"<metric>": value, ...}}}``.
    Non-numeric metrics (e.g. submission placeholders, stderr strings) are
    skipped. Metric names that already contain a comma (lm-eval convention,
    e.g. ``exact_match,strict-match``) are flattened to a single dotted key.
    """
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    results = doc.get("results") or {}
    out: dict[str, float] = {}
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if metric in {"alias", "samples"}:
                continue
            if not isinstance(value, (int, float)):
                continue
            clean = metric.replace(",", ".").replace(" ", "")
            out[f"{task_name}.{clean}"] = float(value)
    return out


def pick_primary_metric(
    scores: dict[str, float],
    task_key: str,
    primary_metric: str,
) -> tuple[str, float] | None:
    """Pick the primary scalar for a benchmark out of lmms-eval's full dump.

    lmms-eval emits several metrics per task; we surface the one named in
    :data:`discord_sft.evals.benchmarks.BENCHMARKS`. Falls back to the first
    key starting with ``<task_key>.`` if the exact metric name isn't present
    (common when the metric is split like ``exact_match.strict-match``).
    """
    exact = f"{task_key}.{primary_metric}"
    if exact in scores:
        return exact, scores[exact]
    prefix = f"{task_key}.{primary_metric}"
    for k, v in scores.items():
        if k.startswith(prefix):
            return k, v
    for k, v in scores.items():
        if k.startswith(f"{task_key}."):
            return k, v
    return None
