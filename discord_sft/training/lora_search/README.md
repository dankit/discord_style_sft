# LoRA Search CLI Matrix

Use canonical training configs from
[`discord_sft/training/configs/README.md`](../configs/README.md).

For full training setup (env, merge-peft, recipes), see [`../README.md`](../README.md).

## Build Data

```bash
discord-sft curate --source discord_messages --out out/curated
discord-sft build-sft --input out/curated/sessions.jsonl --out out/sft
discord-sft fingerprint --input out/sft/train.jsonl --out out/sft/profiles.json
```

Token-length sanity check (1024-token training length):

```bash
discord-sft stats \
  --input out/sft/train.jsonl \
  --tokenizer Qwen/Qwen3.5-35B-A3B
```

## Baseline Eval

```bash
discord-sft eval run \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend vllm \
  --include-baseline \
  --baseline-prompt profile \
  --val out/sft/val.jsonl \
  --profiles out/sft/profiles.json \
  --out out/evals
```

## Gradient Probe Commands

Gradient probe runs forward/backward passes only. It does **not** perform
optimizer updates. It records per-module/layer gradient norms so you can pick
LoRA targets before expensive full training.

Base config:
`discord_sft/training/configs/qwen35_a3b_full_r16.yaml`

```bash
# r=16, alpha=16
python -m discord_sft.training.lora_search.gradient_probe \
  --config discord_sft/training/configs/qwen35_a3b_full_r16.yaml \
  --steps 10 \
  --lora-r 16 --lora-alpha 16 \
  --output out/lora/probes/gradient-probe/gradient_norms_r16_alpha16.json

# r=16, alpha=32 (2r)
python -m discord_sft.training.lora_search.gradient_probe \
  --config discord_sft/training/configs/qwen35_a3b_full_r16.yaml \
  --steps 10 \
  --lora-r 16 --lora-alpha 32 \
  --output out/lora/probes/gradient-probe/gradient_norms_r16_alpha32.json

# r=32, alpha=32
python -m discord_sft.training.lora_search.gradient_probe \
  --config discord_sft/training/configs/qwen35_a3b_full_r16.yaml \
  --steps 10 \
  --lora-r 32 --lora-alpha 32 \
  --output out/lora/probes/gradient-probe/gradient_norms_r32_alpha32.json

# r=32, alpha=64 (2r)
python -m discord_sft.training.lora_search.gradient_probe \
  --config discord_sft/training/configs/qwen35_a3b_full_r16.yaml \
  --steps 10 \
  --lora-r 32 --lora-alpha 64 \
  --output out/lora/probes/gradient-probe/gradient_norms_r32_alpha64.json
```

Default output behavior:

- If `--output` is a directory, probe writes `gradient_norms.json` inside it.
- If `--output` is a `.json` path, probe writes exactly that file.

Each probe JSON includes:

- `aggregate.modules_ranked`: modules ranked by total gradient norm.
- `aggregate.layers_ranked`: layers ranked by total gradient norm.
- `aggregate.layer_module_ranked`: top layer+module pairs.
- `step_records`: per-step loss, supervised token count, top modules/layers.

Reading signal quickly:

- `o_proj` / `down_proj` high: style/register adaptation likely dominates.
- `q_proj` / `v_proj` high: context/instruction handling likely needs help.
- `gate_proj` high: consider MoE gate-path experiments.
- Late layers high: late-layer style recipe is plausible.
- Flat/noisy ranking: keep broader modules/layers until eval narrows candidates.

## Visualize Probe Runs

```bash
python -m discord_sft.training.lora_search.visualize_probe \
  --input out/lora/probes \
  --out-dir out/lora/probes/analysis
```

Expected outputs:

- `out/lora/probes/analysis/probe_ranking.json`
- `out/lora/probes/analysis/layer_contrib.png`
- `out/lora/probes/analysis/module_contrib.png`
- `out/lora/probes/analysis/layer_module_heatmap_best.png`

## Train Canonical Configs

```bash
discord-sft train --config discord_sft/training/configs/qwen35_a3b_style_late.yaml
discord-sft train --config discord_sft/training/configs/qwen35_a3b_full_r16.yaml
```

## Evaluate All Probe Runs

```bash
discord-sft eval run \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend vllm \
  --tasks persona,ifeval \
  --adapter-dir out/lora/probes \
  --include-baseline \
  --baseline-prompt profile \
  --val out/sft/val.jsonl \
  --profiles out/sft/profiles.json \
  --out out/evals
```

## Guardrails

- Keep `model.enable_thinking: false` for Discord-style SFT consistency.
- Keep `train.allow_full_sequence_loss_fallback: false` so masking failures stop early.
- Treat short probe loss changes as noisy; trust module/layer rankings plus eval.
- Use probe ranking to shortlist directions; choose final configs by eval + manual quality check.
