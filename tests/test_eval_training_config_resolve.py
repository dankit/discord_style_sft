"""Tests for resolving ``training_config`` from adapter dirs, merged dirs, manifests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from discord_sft.evals.paths import merge_training_run_id_from_model_path
from discord_sft.evals.runner import (
    load_training_config_provenance,
    training_config_provenance_report,
)


MIN_YAML = (
    "run_name: rn\nmodel:\n  name: hub/M\n  max_seq_length: 512\n"
    "data: {}\nlora:\n  target_modules:\n  - v_proj\ntrain: {}\ncheckpoint: {}\n"
)


@pytest.fixture(autouse=True)
def _needs_yaml():
    pytest.importorskip("yaml")


def test_provenance_prefers_adapter_path(tmp_path: Path) -> None:
    train = tmp_path / "train"
    ckpt_a = train / "a"
    ckpt_b = train / "b"
    ckpt_a.mkdir(parents=True)
    ckpt_b.mkdir(parents=True)
    (train / "config.resolved.yaml").write_text(MIN_YAML, encoding="utf-8")

    tc, src = training_config_provenance_report(str(ckpt_b), str(ckpt_a))
    assert src == "adapter_path"
    assert isinstance(tc, dict)
    mods = (((tc.get("lora") or {}).get("target_modules")) if isinstance(tc.get("lora"), dict) else []) or []
    assert "v_proj" in mods


def test_provenance_model_dir_yaml(tmp_path: Path) -> None:
    merged = tmp_path / "merged-model"
    merged.mkdir()
    (merged / "config.resolved.yaml").write_text(MIN_YAML, encoding="utf-8")
    tc, src = training_config_provenance_report(str(merged), None)
    assert src == "model_dir"
    assert isinstance(tc, dict)


def test_provenance_merge_manifest_adapter_dir(tmp_path: Path) -> None:
    train = tmp_path / "train"
    ckpt = train / "final"
    ckpt.mkdir(parents=True)
    (train / "config.resolved.yaml").write_text(MIN_YAML, encoding="utf-8")

    merged = tmp_path / "merged-empty"
    merged.mkdir()
    (merged / "merge_manifest.json").write_text(
        json.dumps({"adapter_dir": str(ckpt)}),
        encoding="utf-8",
    )
    tc, src = training_config_provenance_report(str(merged), None)
    assert src == "merge_manifest.adapter_dir"
    assert isinstance(tc, dict)

    loose = load_training_config_provenance("Qwen/not-a-local-dir", None)
    assert loose is None


def test_provenance_repo_relative_model_path(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    mdir = fake_root / "out" / "merged" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "config.resolved.yaml").write_text(MIN_YAML, encoding="utf-8")
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report("out/merged/m1", None)
    assert src == "model_dir"
    assert isinstance(tc, dict)
    assert isinstance(tc.get("lora"), dict)


def test_provenance_tags_missing_model_dir(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    (fake_root / "out").mkdir(parents=True)
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report("out/merged/absent-merge", None)
    assert tc is None
    assert src == "model_path_missing"

    merged = fake_root / "out" / "merged" / "no_yaml"
    merged.mkdir(parents=True)
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report("out/merged/no_yaml", None)
    assert tc is None
    assert src == "model_dir_no_yaml"

    (merged / "merge_manifest.json").write_text(json.dumps({"adapter_dir": ""}), encoding="utf-8")
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report("out/merged/no_yaml", None)
    assert tc is None
    assert src == "merge_manifest_bad_adapter_dir"


def test_merge_training_run_id_from_model_path() -> None:
    assert (
        merge_training_run_id_from_model_path(
            r"out/merged/style-late-r16-a16-5epochs/epoch-4-vllm-keys"
        )
        == "style-late-r16-a16-5epochs"
    )
    assert merge_training_run_id_from_model_path("merged/foo/bar") == "foo"
    assert merge_training_run_id_from_model_path("Qwen/Qwen3-8B") is None


def test_provenance_lora_via_merge_path_hint_when_merged_absent(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    ldir = fake_root / "out" / "lora" / "style-late-r16-a16-5epochs"
    ldir.mkdir(parents=True)
    yaml_body = (
        "run_name: style-late-r16-a16-5epochs\n"
        + MIN_YAML.split("\n", 1)[1]  # drop first run_name line from MIN_YAML
    )
    (ldir / "config.resolved.yaml").write_text(yaml_body, encoding="utf-8")
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report(
            "out/merged/style-late-r16-a16-5epochs/epoch-4-vllm-keys", None
        )
    assert src == "lora_run_dir.merge_path_hint"
    assert isinstance(tc, dict)
    assert tc.get("run_name") == "style-late-r16-a16-5epochs"


def test_provenance_lora_hint_skips_run_name_mismatch(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    ldir = fake_root / "out" / "lora" / "style-late-r16-a16-5epochs"
    ldir.mkdir(parents=True)
    (ldir / "config.resolved.yaml").write_text(
        "run_name: other-run\n" + MIN_YAML.split("\n", 1)[1],
        encoding="utf-8",
    )
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report(
            "out/merged/style-late-r16-a16-5epochs/epoch-4-vllm-keys", None
        )
    assert tc is None
    assert src == "model_path_missing"


def test_provenance_merge_dir_wins_over_lora_hint(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    mdir = fake_root / "out" / "merged" / "my-run" / "epoch-1"
    mdir.mkdir(parents=True)
    yaml_merged = MIN_YAML.replace("v_proj", "gate_proj")
    (mdir / "config.resolved.yaml").write_text(yaml_merged, encoding="utf-8")

    ldir = fake_root / "out" / "lora" / "my-run"
    ldir.mkdir(parents=True)
    (ldir / "config.resolved.yaml").write_text(MIN_YAML, encoding="utf-8")

    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report(
            "out/merged/my-run/epoch-1", None
        )
    assert src == "model_dir"
    mods = (((tc or {}).get("lora") or {}).get("target_modules")) if isinstance(tc, dict) else []
    assert "gate_proj" in (mods or [])


def test_provenance_manifest_adapter_dir_missing_on_disk(tmp_path: Path) -> None:
    fake_root = tmp_path / "checkout"
    merged = fake_root / "out" / "merged" / "mmanifest"
    merged.mkdir(parents=True)
    (merged / "merge_manifest.json").write_text(
        json.dumps({"adapter_dir": "out/lora/not-there"}),
        encoding="utf-8",
    )
    with patch("discord_sft.evals.paths.eval_repo_root", return_value=fake_root):
        tc, src = training_config_provenance_report("out/merged/mmanifest", None)
    assert tc is None
    assert src == "merge_manifest_adapter_missing"
