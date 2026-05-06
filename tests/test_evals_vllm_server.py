"""Unit tests for the local vLLM server helpers.

We never actually launch vLLM in tests — the module is imported purely
for its pure functions (alias parsing, argv construction, rank probing).
That keeps the test suite fast and independent of whether the heavy
``vllm`` package is installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from discord_sft.evals.vllm_server import (
    VLLMServerConfig,
    adapters_from_cli,
    alias_for,
    build_server_argv,
    discover_adapter_dirs,
    max_lora_rank,
    python_for_vllm_subprocess,
)


def test_python_for_vllm_subprocess_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISCORD_SFT_VLLM_PYTHON", "/opt/py/bin/python")
    assert python_for_vllm_subprocess() == "/opt/py/bin/python"


def test_python_for_vllm_subprocess_explicit_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DISCORD_SFT_VLLM_PYTHON", "/env/python")
    assert python_for_vllm_subprocess(explicit="/opt/x/bin/python3") == "/opt/x/bin/python3"


def test_python_for_vllm_subprocess_prefers_activation_when_executable_differs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Regression: launcher shebang targets .venv but user activated .venv-evals."""
    monkeypatch.delenv("DISCORD_SFT_VLLM_PYTHON", raising=False)
    fake_evals = tmp_path / ".venv-evals"
    if sys.platform == "win32":
        bin_dir = fake_evals / "Scripts"
        py_interp = bin_dir / "python.exe"
        wrong = tmp_path / "wrong" / "python.exe"
    else:
        bin_dir = fake_evals / "bin"
        py_interp = bin_dir / "python3"
        wrong = tmp_path / "wrong" / "python"
    bin_dir.mkdir(parents=True)
    py_interp.write_bytes(b"# fake\n")
    py_interp.chmod(0o755)
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_evals))
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"# fake\n")
    wrong.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(wrong))

    assert Path(python_for_vllm_subprocess()) == py_interp.resolve()


def test_python_for_vllm_subprocess_keeps_venv_path_when_symlinks_to_system_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """bin/python3 → system binary must still spawn the venv path (pyvenv + site-packages)."""
    monkeypatch.delenv("DISCORD_SFT_VLLM_PYTHON", raising=False)
    if sys.platform == "win32":
        pytest.skip("Unix venv symlink layout")
    fake_evals = tmp_path / ".venv-evals"
    bin_dir = fake_evals / "bin"
    bin_dir.mkdir(parents=True)
    system_stub = tmp_path / "fake-system-python"
    system_stub.write_bytes(b"# fake system python\n")
    system_stub.chmod(0o755)
    py_in_venv = bin_dir / "python3"
    py_in_venv.symlink_to(system_stub)
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_evals.resolve()))
    monkeypatch.setattr(sys, "executable", str(system_stub.resolve()))

    assert python_for_vllm_subprocess() == str(py_in_venv)
    assert Path(python_for_vllm_subprocess()).resolve() == system_stub.resolve()


def test_alias_for_sanitises_basename():
    assert alias_for("/abs/out/lora/r8") == "r8"
    assert alias_for("out/lora/style late r32") == "style-late-r32"
    assert alias_for("/abs/out/lora/style-late-r32/final") == "style-late-r32"
    # Trailing slash shouldn't collapse to empty string.
    assert alias_for("/abs/out/lora/r8/") == "r8"


def test_adapters_from_cli_handles_bare_path_and_explicit_alias(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "something"
    a.mkdir()
    b.mkdir()
    pairs = adapters_from_cli([str(a), f"myname={b}"])
    assert pairs[0][0] == "a"
    assert Path(pairs[0][1]).name == "a"
    assert pairs[1][0] == "myname"
    assert Path(pairs[1][1]).name == "something"


def test_adapters_from_cli_disambiguates_colliding_aliases(tmp_path: Path):
    # Two directories whose basename would collide must get -2/-3 suffixes,
    # because some vLLM versions silently drop a later --lora-modules entry
    # when an alias repeats.
    d1 = tmp_path / "sub1" / "r8"
    d2 = tmp_path / "sub2" / "r8"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    pairs = adapters_from_cli([str(d1), str(d2)])
    aliases = [alias for alias, _ in pairs]
    assert aliases == ["r8", "r8-2"]


def test_adapters_from_cli_does_not_split_windows_drive_letter_paths(tmp_path: Path):
    # Regression guard: a bare ``C:\path`` (no ``=``) must not crash and
    # must keep the whole thing as the path. Our heuristic only splits on
    # ``=`` when the LHS is a plain identifier (``[A-Za-z0-9._-]+``), so a
    # drive-letter path like ``C:\foo`` never gets mis-parsed as alias=C.
    p = tmp_path / "lora"
    p.mkdir()
    pairs = adapters_from_cli([str(p)])
    assert len(pairs) == 1
    alias, path = pairs[0]
    assert alias == "lora"
    assert Path(path) == p.resolve()


def test_discover_adapter_dirs_prefers_final_adapters(tmp_path: Path):
    arm_a_final = tmp_path / "arm-a" / "final"
    arm_a_epoch = tmp_path / "arm-a" / "epoch-1"
    arm_b_final = tmp_path / "arm-b" / "final"
    for d in [arm_a_final, arm_a_epoch, arm_b_final]:
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(json.dumps({"r": 32}), encoding="utf-8")

    discovered = [Path(p) for p in discover_adapter_dirs(tmp_path)]

    assert discovered == [arm_a_final.resolve(), arm_b_final.resolve()]


def test_discover_adapter_dirs_falls_back_to_all_checkpoints(tmp_path: Path):
    step = tmp_path / "run" / "step-200"
    step.mkdir(parents=True)
    (step / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")

    assert discover_adapter_dirs(tmp_path) == [str(step.resolve())]


def test_max_lora_rank_probes_adapter_config_json(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    (b / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")
    assert max_lora_rank([("a", str(a)), ("b", str(b))]) == 64


def test_max_lora_rank_falls_back_to_default(tmp_path: Path):
    d = tmp_path / "noconfig"
    d.mkdir()
    # No adapter_config.json + no probe fallback.
    assert max_lora_rank([("x", str(d))], default=16) == 16


def test_build_server_argv_respects_python_executable():
    cfg = VLLMServerConfig(
        model="Qwen/Qwen3.5-35B-A3B",
        python_executable="/srv/.venv-evals/bin/python3",
    )
    argv = build_server_argv(cfg, port=1)
    assert argv[0] == "/srv/.venv-evals/bin/python3"


def test_build_server_argv_minimal_command_for_single_h100():
    cfg = VLLMServerConfig(
        model="Qwen/Qwen3.5-35B-A3B",
        adapters=[],
        max_model_len=16384,
        gpu_memory_utilization=0.9,
    )
    argv = build_server_argv(cfg, port=12345)
    # We deliberately do NOT pass --tensor-parallel-size: single GPU,
    # vLLM default TP=1. Regression test that this stays omitted.
    assert "--tensor-parallel-size" not in argv
    assert "vllm.entrypoints.openai.api_server" in argv
    assert argv[argv.index("--model") + 1] == "Qwen/Qwen3.5-35B-A3B"
    assert argv[argv.index("--port") + 1] == "12345"
    assert argv[argv.index("--max-model-len") + 1] == "16384"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--trust-remote-code" in argv
    # No adapters → no LoRA flags.
    assert "--enable-lora" not in argv
    assert "--lora-modules" not in argv


def test_build_server_argv_with_adapters_includes_lora_modules():
    cfg = VLLMServerConfig(
        model="Qwen/Qwen3.5-35B-A3B",
        adapters=[("r8", "/abs/out/lora/r8"), ("style", "/abs/out/lora/style")],
        max_lora_rank=64,
        reasoning_parser="qwen3",
    )
    argv = build_server_argv(cfg, port=8001)
    assert "--enable-lora" in argv
    assert argv[argv.index("--max-lora-rank") + 1] == "64"
    idx = argv.index("--lora-modules")
    assert argv[idx + 1] == "r8=/abs/out/lora/r8"
    assert argv[idx + 2] == "style=/abs/out/lora/style"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"


def test_build_server_argv_quantization_escape_hatch():
    cfg = VLLMServerConfig(
        model="Qwen/Qwen3.5-35B-A3B",
        quantization="fp8",
        reasoning_parser=None,
    )
    argv = build_server_argv(cfg, port=9999)
    assert argv[argv.index("--quantization") + 1] == "fp8"
    # reasoning_parser=None/"" must omit the flag entirely.
    assert "--reasoning-parser" not in argv


def test_build_server_argv_appends_extra_args():
    cfg = VLLMServerConfig(
        model="Qwen/Qwen3.5-35B-A3B",
        extra_args=["--language-model-only", "--max-num-seqs", "64"],
    )
    argv = build_server_argv(cfg, port=7000)
    assert argv[-3:] == ["--language-model-only", "--max-num-seqs", "64"]
