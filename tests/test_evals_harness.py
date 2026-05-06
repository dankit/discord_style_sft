from __future__ import annotations

import json
from pathlib import Path

import pytest

from discord_sft.evals.harness import (
    _find_results_json,
    build_command,
    parse_lmms_results,
    pick_primary_metric,
)
from discord_sft.evals.model import ModelSpec


FIXTURE = Path(__file__).parent / "fixtures" / "lmms_eval_results_sample.json"


def test_parse_lmms_results_flattens_dotted_keys():
    scores = parse_lmms_results(FIXTURE)
    assert scores["ifeval.prompt_level_strict_acc.none"] == pytest.approx(0.641)
    assert scores["mmmu_val.mmmu_acc.none"] == pytest.approx(0.503)
    assert scores["mmstar.average.none"] == pytest.approx(0.572)
    assert scores["screenspot_v2.accuracy.none"] == pytest.approx(0.612)
    assert "ifeval.alias" not in scores


def test_pick_primary_metric_exact_and_prefix():
    scores = parse_lmms_results(FIXTURE)
    got = pick_primary_metric(scores, "ifeval", "prompt_level_strict_acc")
    assert got is not None
    key, val = got
    assert key.startswith("ifeval.prompt_level_strict_acc")
    assert val == pytest.approx(0.641)

    got_mmmu = pick_primary_metric(scores, "mmmu_val", "mmmu_acc")
    assert got_mmmu is not None
    assert got_mmmu[1] == pytest.approx(0.503)


def test_pick_primary_metric_missing_returns_none():
    scores = {"foo.bar": 1.0}
    assert pick_primary_metric(scores, "notfound", "x") is None


def test_build_command_surfaces_model_args_and_tasks():
    spec = ModelSpec(
        name_or_path="Qwen/Qwen3.5-35B-A3B",
        backend="vllm",
        adapter_path="out/lora/r8",
        dtype="bfloat16",
        lora_rank=32,
    )
    cmd = build_command(
        spec,
        ["ifeval", "mmmu_val"],
        output_path="/tmp/out",
        limit=200,
        batch_size="auto",
    )
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "vllm"
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "model=Qwen/Qwen3.5-35B-A3B" in model_args
    # vllm backend must use lora_local_path, not peft (the latter is hf-only)
    assert "lora_local_path=out/lora/r8" in model_args
    assert "enable_lora=true" in model_args
    assert "max_lora_rank=32" in model_args
    assert "peft=" not in model_args
    assert "dtype=bfloat16" in model_args
    assert cmd[cmd.index("--tasks") + 1] == "ifeval,mmmu_val"
    assert "--limit" in cmd
    assert cmd[cmd.index("--limit") + 1] == "200"
    assert "--log_samples" in cmd


def test_model_spec_hf_backend_uses_pretrained_key():
    spec = ModelSpec(name_or_path="foo/bar", backend="hf")
    rendered = spec.to_lmms_eval_args()
    assert rendered.startswith("pretrained=foo/bar")
    assert "trust_remote_code=true" in rendered


def test_model_spec_hf_with_adapter_uses_peft_key():
    # Adapter on the hf backend is the lm-eval ``peft=`` convention; we
    # keep that path unchanged. The bug was only for the vllm backend.
    spec = ModelSpec(
        name_or_path="foo/bar", backend="hf", adapter_path="/tmp/lora"
    )
    rendered = spec.to_lmms_eval_args()
    assert "peft=/tmp/lora" in rendered
    assert "lora_local_path" not in rendered


def test_model_spec_vllm_with_adapter_uses_lora_local_path():
    # Key regression guard: pre-fix code emitted ``peft=`` for the vllm
    # backend too, which lm-eval's VLLM wrapper ignores — the adapter
    # silently wasn't applied. We now emit the correct key.
    spec = ModelSpec(
        name_or_path="Qwen/Qwen3.5-35B-A3B",
        backend="vllm",
        adapter_path="/abs/out/lora/r8",
        lora_rank=16,
    )
    rendered = spec.to_lmms_eval_args()
    assert "model=Qwen/Qwen3.5-35B-A3B" in rendered
    assert "lora_local_path=/abs/out/lora/r8" in rendered
    assert "enable_lora=true" in rendered
    assert "max_lora_rank=16" in rendered
    assert "peft=" not in rendered


def test_model_spec_openai_args_point_at_server():
    spec = ModelSpec(
        name_or_path="Qwen/Qwen3.5-35B-A3B",
        backend="openai",
        lora_alias="r8",
    )
    rendered = spec.to_openai_lmms_args(base_url="http://127.0.0.1:8000/v1")
    assert "model_version=r8" in rendered
    # lmms-eval appends /chat/completions; base_url must be the API root (.../v1).
    assert "base_url=http://127.0.0.1:8000/v1" in rendered
    assert "chat/completions/chat/completions" not in rendered
    assert "api_key=EMPTY" in rendered
    assert "max_retries=5" in rendered


def test_model_spec_openai_args_does_not_stub_external_api_key():
    spec = ModelSpec(name_or_path="gpt-4.1-mini", backend="openai")
    rendered = spec.to_openai_lmms_args(base_url="https://api.openai.com/v1")
    assert "base_url=https://api.openai.com/v1" in rendered
    assert "api_key=EMPTY" not in rendered


def test_model_spec_openai_args_stubs_non_localhost_servers():
    """LAN / container hostnames used for vLLM must still get a placeholder key."""
    spec = ModelSpec(name_or_path="x", backend="openai")
    for base in (
        "http://ubuntu:8000/v1",
        "http://gpu-0:8000/v1",
        "http://172.18.0.2:8000/v1",
        "http://[::1]:8000/v1",
    ):
        rendered = spec.to_openai_lmms_args(base_url=base)
        assert "api_key=EMPTY" in rendered, base


def test_build_command_routes_to_openai_when_server_base_url_set():
    spec = ModelSpec(
        name_or_path="Qwen/Qwen3.5-35B-A3B",
        backend="openai",
        lora_alias="r8",
    )
    cmd = build_command(
        spec,
        ["ifeval"],
        output_path="/tmp/out",
        server_base_url="http://127.0.0.1:12345/v1",
    )
    assert cmd[cmd.index("--model") + 1] == "openai"
    args = cmd[cmd.index("--model_args") + 1]
    assert "model_version=r8" in args
    assert "base_url=http://127.0.0.1:12345/v1" in args
    assert "api_key=EMPTY" in args


def test_build_command_forwards_extra_cli_gen_kwargs():
    spec = ModelSpec(name_or_path="Qwen/Qwen3.5-35B-A3B", backend="vllm")
    cmd = build_command(
        spec,
        ["ifeval"],
        output_path="/tmp/out",
        extra_cli=["--gen_kwargs", "temperature=1.0,top_p=0.95,top_k=20"],
    )
    assert "--gen_kwargs" in cmd
    assert cmd[cmd.index("--gen_kwargs") + 1] == "temperature=1.0,top_p=0.95,top_k=20"


def test_find_results_json_newest_match(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    older = tmp_path / "sub" / "results_1.json"
    newer = tmp_path / "sub" / "results_2.json"
    older.write_text(json.dumps({"results": {}}), encoding="utf-8")
    newer.write_text(json.dumps({"results": {}}), encoding="utf-8")
    import os
    import time

    os.utime(older, (time.time() - 120, time.time() - 120))
    picked = _find_results_json(tmp_path)
    assert picked == newer


def test_find_results_json_none_when_missing(tmp_path: Path):
    assert _find_results_json(tmp_path) is None


def test_find_results_json_lmms_dated_results_filename(tmp_path: Path):
    """lmms-eval writes ``<date_id>_results.json`` under a model subfolder."""
    nested = tmp_path / "Qwen__Qwen3.5-35B-A3B"
    nested.mkdir(parents=True)
    results_file = nested / "20260426_062800_results.json"
    results_file.write_text(json.dumps({"results": {"ifeval": {}}}), encoding="utf-8")
    assert _find_results_json(tmp_path) == results_file


def test_find_results_json_skips_sample_json_sidecars(tmp_path: Path):
    """Do not treat unrelated JSON as aggregated results."""
    nested = tmp_path / "Model"
    nested.mkdir(parents=True)
    bad = nested / "20260426_062800_samples_ifeval.json"
    bad.write_text("[]", encoding="utf-8")
    good = nested / "20260426_070000_results.json"
    good.write_text(json.dumps({"results": {}}), encoding="utf-8")
    assert _find_results_json(tmp_path) == good
