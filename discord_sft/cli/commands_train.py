from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_train(args: argparse.Namespace) -> int:
    action = getattr(args, "train_action", None) or "run"
    if action == "merge-peft":
        return _cmd_train_merge_peft(args)
    return _cmd_train_run(args)


def _cmd_train_merge_peft(args: argparse.Namespace) -> int:
    from discord_sft.training.merge_peft import merge_peft_adapter_to_hf_dir

    adapter = getattr(args, "merge_adapter", None)
    out_dir = getattr(args, "merge_output", None)
    if not adapter:
        sys.stderr.write("merge-peft requires --adapter <checkpoint_dir>.\n")
        return 2
    if not out_dir:
        sys.stderr.write("merge-peft requires --output <merged_dir>.\n")
        return 2
    if getattr(args, "output_dir", None):
        sys.stderr.write("--output-dir applies to `train run`, not merge-peft.\n")
        return 2

    cfg_path = Path(args.config) if getattr(args, "config", None) else None

    try:
        summary = merge_peft_adapter_to_hf_dir(
            adapter,
            out_dir,
            base_model=getattr(args, "merge_base_model", None),
            config_path=cfg_path,
            max_seq_length=getattr(args, "merge_max_seq_length", None),
            load_in_16bit=getattr(args, "merge_load_in_16bit", None),
            max_shard_size=str(getattr(args, "merge_max_shard_size", "5GB") or "5GB"),
        )
    except Exception as e:
        sys.stderr.write(f"merge-peft failed: {e}\n")
        return 1

    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


def _cmd_train_run(args: argparse.Namespace) -> int:
    from discord_sft.training import load_config, run_training

    if getattr(args, "merge_output", None):
        sys.stderr.write("`--output` is only valid for `train merge-peft`.\n")
        return 2
    if getattr(args, "merge_adapter", None):
        sys.stderr.write("`--adapter` is only valid for `train merge-peft`.\n")
        return 2
    cfg_arg = getattr(args, "config", None)
    if not cfg_arg:
        sys.stderr.write("`train run` requires --config <training.yaml>.\n")
        return 2

    cfg_path = Path(cfg_arg)
    if not cfg_path.exists():
        sys.stderr.write(f"Config not found: {cfg_path}\n")
        return 1
    cfg = load_config(cfg_path)

    # CLI overrides that are common enough to wire without editing YAML.
    if getattr(args, "output_dir", None):
        cfg.checkpoint.output_dir = args.output_dir
    if getattr(args, "run_name", None):
        cfg.run_name = args.run_name
    if getattr(args, "resume_adapter", None):
        cfg.train.resume_adapter_path = args.resume_adapter

    result = run_training(cfg, source_yaml=cfg_path)
    sys.stdout.write(
        json.dumps(
            {
                "run_name": result["run_name"],
                "output_dir": result["output_dir"],
                "checkpoints": [
                    c.get("rel_path") for c in result["manifest"].get("checkpoints", [])
                ],
                "status": result["manifest"].get("status"),
            },
            indent=2,
        )
        + "\n"
    )
    return 0
