# `discord_sft.training`

LoRA SFT for `unsloth/Qwen3.5-35B-A3B` on the ShareGPT JSONL produced by
`discord-sft build-sft`. One YAML file = one reproducible run. Adapter-only
checkpoints land in `out/lora/<run_name>/` and slot directly into the
existing `discord-sft eval run --adapter ...` path.

**Data alignment:** curation **`--min-turns`** should be compatible with the
ShareGPT lengths you want from **`build-sft`** (e.g. `--min-turns 2` on curate if you
rely on short `num_turns: 2` rows). After `build-sft`, check **`turn_length_report.json`**
next to `train.jsonl` / `val.jsonl` for the final length distribution. See the root
**README** *Tuning curation* for `curate_report.json` and **`curate-sweep`**.

## Install

The committed [`uv.lock`](../../uv.lock) at the repo root pins exact
versions for every dep in `[train]` (276 packages total across base +
every extra). A run launched today will resolve to the same Unsloth / TRL
/ torch / transformers versions six months from now, which is the whole
point of saving checkpoints.

```bash
pip install uv                  # one-time
uv sync --extra train           # installs the frozen set into ./.venv
```

The lock targets **Linux aarch64** (same `tool.uv.environments` as `[evals]`).
Unsloth → Triton is Linux-only. `uv sync --extra train` will fail on macOS or Windows.
Train on a Linux GH200 (or other aarch64 GPU host). **`[train]` and `[evals]` cannot be combined in one `uv sync`** — use separate venvs if you need both on one machine.

`pyproject.toml` **`[train]`** pins **Transformers v5 only** (`>=5.0.1`, **`<=5.5.0`**, **`!=5.1.0`**) so a lock/sync cannot pick legacy **4.x** for Qwen3.5–style stacks; **`setup_gh200_training.sh`** still uses **`transformers==5.5.0`**.

**Upstream incompatibility:** Unsloth `2026.4.x` requires **`torch<2.11`** and **`transformers<=5.5.0`**; **`[evals]`** uses **vLLM 0.19.x** with **`torch<2.11`** but needs **`transformers>=5.5.1`** (often **`@main`** on GH200). This is deliberate pin friction, not a uv quirk—you still need **two venvs** (suggested dirs: **`.venv-train`** vs **`.venv-evals`**; set `GH200_VENV_DIR` for the bootstrap scripts).

If you need to modify deps, edit `[project.optional-dependencies.train]`
in `pyproject.toml` and re-run `uv lock`. Commit the updated `uv.lock`
alongside your code change so every future checkpoint carries a lock
fingerprint that matches a real committed state.

Plain `pip install -e ".[train]"` still works but ignores the lockfile;
use only for one-off experiments, never for artifacts you intend to
evaluate later.

## GH200 (ARM64) runbook

GH200 hosts are `aarch64` (`uname -m`). The committed `uv.lock` resolves **`[train]`**
for **Linux aarch64**. If you ever see CPU-only `torch` wheels or Unsloth failing with
"cannot find any torch accelerator", your resolver or index selection drifted from
that lock — reset the venv and re-run **`uv sync --extra train`** from a clean checkout.

Typical failure modes when the environment drifts from the lock:

- Wrong `pip`/venv so installs land outside the project venv.
- Resolver picks CPU-only `torch` on ARM.
- Mismatched `torch` / `torchao` builds (`AttributeError: torch has no attribute int1`).

Prefer **`uv sync --extra train`** from a clean checkout when the committed `uv.lock` matches your host. Fall back to manual installs only for index drift (CPU torch, wrong CUDA build) or custom pins.

After changing `[project.dependencies]` or `[project.optional-dependencies]`, run **`uv lock`** from the repo root and commit the updated `uv.lock`. Use **separate** venv directories for `uv sync --extra train` vs `uv sync --extra evals` (see [eval runbook](../evals/README.md#2-install)).

Validated baseline for this repo on GH200:

- `torch==2.10.0+cu128`
- `torch.version.cuda == "12.8"`
- `torch.cuda.is_available() == True`
- `hasattr(torch, "_grouped_mm") == True`

### One-time setup on GH200

One-click bootstrap (recommended):

```bash
cd ~/discord_sft
GH200_VENV_DIR=.venv-train bash scripts/setup_gh200_training.sh   # optional: keeps eval venv (.venv-evals) separate from train
```

This script handles virtualenv creation/reset, CUDA torch pinning, pinned training
deps, Unsloth install, FlashAttention wheel install, CUDA preflight checks, and
`.env` template creation.
It also prints the exact next training command when setup completes.

### Preflight: val loss + persona evals

To ensure training reports validation loss **and** runs persona tertiary evals:

- `data.val_path` must point to an existing `val.jsonl` (typically `out/sft/val.jsonl`).
- `train.eval_strategy` must not be `no` (use `epoch` or `steps`) so `eval/loss` is computed.
- `train.tertiary_eval_enabled: true`.
- `train.tertiary_eval_tasks_checkpoint: [persona]`.
- `train.tertiary_eval_tasks_final: [persona]` (or omit to reuse checkpoint tasks).
- `train.tertiary_eval_profile_json` should point to `out/sft/profiles.json` for profile-conditioned persona heuristics.

Minimal YAML snippet:

```yaml
data:
  val_path: out/sft/val.jsonl

train:
  eval_strategy: epoch
  tertiary_eval_enabled: true
  tertiary_eval_tasks_checkpoint: [persona]
  tertiary_eval_tasks_final: [persona]
  tertiary_eval_profile_json: out/sft/profiles.json
```

## One-liner

```bash
discord-sft train --config discord_sft/training/configs/qwen35_a3b_style_late.yaml
```

Before your first training run (or after changing curation / template flags), you can verify that `train.jsonl` / `val.jsonl` round-trip cleanly through the target tokenizer’s chat template:

```bash
discord-sft validate-chat-template --input out/sft/train.jsonl --tokenizer Qwen/Qwen3.5-35B-A3B
```

Same HF tokenizer dependency as `discord-sft stats` / training: install `[tokenizers]` or use the full `[train]` extra.

Overrides:

- `--output-dir out/lora/style-late-r16-v2` — change where checkpoints go.
- `--run-name style-late-r16-v2` — change the name recorded in `run.json`.
- `--resume-adapter out/lora/style-late-r16/epoch-2` — continue training from an existing adapter checkpoint.

Equivalent explicit form:

```bash
discord-sft train run --config discord_sft/training/configs/qwen35_a3b_style_late.yaml
```

## merge-peft: bake LoRA into base weights (vLLM without adapters)

[vLLM’s MoE LoRA path](https://github.com/vllm-project/vllm/issues) can be unreliable for Qwen3.5-style fused experts. Adapter checkpoints from this repo are **standard PEFT** on disk (`adapter_config.json`, `adapter_model.safetensors`); merging them removes the `--lora-modules` requirement so **eval sees a plain HF folder**.

Requirements:

- **Dependencies**: **`[evals]`** (`torch` + **`peft>=0.18`** + `transformers`, no Unsloth) is enough — older PEFT with **transformers 5.5+** can raise **`WeightConverter` / `distributed_operation`** at merge time; re-sync deps from this repo’s lock/pyproject if you see that. Merge falls back to **Transformers‑only** base load plus **`PeftModel.merge_and_unload`**. With **Unsloth** (`[train]` venv), merge prefers **`FastModel.from_pretrained`** for parity with `train run`. If Transformers-only load fails for your base, use **`[train]`** on that host.
- **VRAM and disk** comparable to hosting the dense MoE checkpoint (much larger than adapter-only dirs). Merges can OOM mid-save on undersized GPUs; run on training-class hardware.

Resolve **base model id**: `--base-model`, else `run.json` next to `final/` / `epoch-N/`, else `adapter_config.json` → `base_model_name_or_path`. Match **`config.model.name` from training** (e.g. `unsloth/Qwen3.5-35B-A3B`).

**Unsloth path** uses the same **`FastModel.from_pretrained`** + **`set_unsloth_env`** knobs as [`trainer.py`](trainer.py). **Transformers path** uses **`AutoModelForCausalLM`** / **`AutoModel`** with optional **`device_map="auto"`** when `accelerate` is available. Parity comes from `<output_dir>/config.resolved.yaml`, or **`--config path/to/recipe.yaml`**, plus **`--max-seq-length`** / **`--merge-load-in-16bit`** / **`--no-merge-load-in-16bit`** overrides.

Writes **`merge_manifest.json`** into the merged tree (paths, resolved base id, **`merge_backend`** `unsloth` vs `transformers`, library versions, optional `training_git_sha`). Also copies **`config.resolved.yaml`** from the training run directory (next to **`--adapter`**) into the merged folder so **`discord-sft eval run --model`** on that merged folder (without **`--adapter`**) still embeds **`training_config`** in the eval JSON (offline tools read **`lora.target_modules`**). For **Qwen3.5 multimodal** bases, merge also pulls **`preprocessor_config.json`** (and **`processor_config.json`** / **`video_preprocessor_config.json`** when present on the Hub) from the resolved **`base_model`** id so **vLLM** can instantiate the HF image processor (“Can’t load image processor … **preprocessor_config.json**”). If you merged **before** that behavior existed, copy those JSON files manually from `merge_manifest.json`’s **`base_model`** snapshot or rerun merge after upgrading.

```bash
discord-sft train merge-peft \
  --adapter out/lora/full-r16/final \
  --output out/merged/full-r16-final
```

Then point **eval’s separate vLLM env** at the merged folder (**no `--adapter`**):

```bash
discord-sft eval run \
  --model /abs/path/to/out/merged/full-r16-final --backend vllm \
  ...
```

Each adapter ⇒ one merged dir ⇒ one **`eval run`** (no multi-adapter `--adapter-dir` sweep on those weights).

See also [§7](../evals/README.md#7-lora-sweep-and-optional-base-in-one-shot) in the eval runbook.

## W&B + checkpoint eval telemetry

Training can stream metrics to Weights & Biases and run tertiary evals at each
saved checkpoint.

Set in YAML under `train`:

- `wandb_enabled: true` turns on W&B reporting (shipped example YAMLs default to
  `false` so you opt in explicitly).
- `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags` set run metadata.
- `tertiary_eval_enabled: true` turns on best-effort checkpoint evals.
- `tertiary_eval_tasks_checkpoint: [persona]` controls per-checkpoint evals.
- `tertiary_eval_tasks_final: [persona]` keeps final in-train eval persona-only.
- Training loop tertiary evals are persona-only (no lmms-eval dependency in-train).
  Run lmms tasks separately with `discord-sft eval run`.

Expected W&B metrics for each run:

- Primary: `train/loss`, `eval/loss`, `eval/ppl`, `eval/generalization_gap`.
- Throughput: `train/step_throughput_steps_per_sec`.
- Best-so-far: `eval/best_loss` (running minimum `eval/loss`).
- Secondary: `train/grad_norm_preclip`, `train/lr`.
- Tertiary: `eval/persona/*`.

If tertiary eval fails (missing deps, data, runtime error), training continues.
The failure is recorded in `run.json` warnings/events for auditability.

## Hardware

`Qwen3.5-35B-A3B` in bf16 LoRA needs ~74 GB VRAM (Unsloth docs). A single
H100 80 GB is enough with `max_seq_length: 2048`. 4-bit QLoRA is not
recommended for this model — per Unsloth, the quantization error is higher
than usual and degrades training quality.

## Two reference configs

- [`configs/qwen35_a3b_full_r16.yaml`](configs/qwen35_a3b_full_r16.yaml) —
  all 7 attention + MLP projections (including MoE-fused `gate_up_proj`),
  every layer, `r=16`. Closest to Unsloth's stock recipe; use as baseline.
- [`configs/qwen35_a3b_style_late.yaml`](configs/qwen35_a3b_style_late.yaml)
  — style-oriented recipe (`up_proj`, `down_proj`, `v_proj`) on the last 25 %
  of layers, `r=16`, `alpha=32`.

Both configs set `model.enable_thinking: false`. Qwen3.5's chat template
defaults to thinking mode, which can inject an empty `<think></think>`
scaffold before every assistant reply. For Discord style SFT, that scaffold
is not part of the target behavior, so training renders clean chat turns and
leaves reasoning-mode decisions to eval/serving. Both configs also keep
`train.allow_full_sequence_loss_fallback: false`; if Unsloth cannot apply
response-only masking, training fails instead of silently supervising user
tokens.

## What gets written

```
out/lora/<run_name>/
├── config.source.yaml       # verbatim copy of --config
├── config.resolved.yaml     # same, after defaults + layers_last_pct expansion
├── run.json                 # manifest: started_at, finished_at, git_sha,
│                            # uv_lock: {path, sha256}, base_model,
│                            # checkpoints: [{kind, path, epoch,
│                            # global_step, metrics, saved_at}, ...],
│                            # best_checkpoint: {path, eval_loss, ...}
├── epoch-1/                 # adapter_config.json, adapter_model.safetensors,
│   ├── config.source.yaml   # + a copy of both YAMLs so this dir is
│   └── config.resolved.yaml # fully self-describing in isolation
├── epoch-2/
├── epoch-3/
├── final/                   # always written at end of training
└── hf_trainer/              # HF Trainer scratch dir (logs, rng state). Not
                             # used by eval; safe to delete after a run.
```

### Reproducibility triple

Every checkpoint can be traced back to an exact environment by three
fields in `run.json`:

- `git_sha` — the commit that produced this adapter. Combined with
  `uv.lock` at the same SHA, reconstruct the training env with
  `git checkout <sha> && uv sync --extra train`.
- `uv_lock.sha256` — hash of the lockfile used at training time. Eval
  compares this against the live repo's `uv.lock`; mismatches surface in
  both stderr and the eval run JSON.
- `config.resolved.yaml` — the LoRA recipe, with `layers_last_pct`
  already expanded to explicit layer indices.

`save_total_limit` in the YAML controls how many `epoch-N/` / `step-S/`
dirs are kept. `final/` is never evicted.

## How evals pick up training configs

The eval runner ([`discord_sft/evals/runner.py`](../evals/runner.py)) reads
`config.resolved.yaml` **and** `run.json` from each `--adapter` path and
embeds them inline in the run JSON at `out/evals/runs/<run_id>.json`
under the `training_config` key — including a
`uv_lock: {training, current, matches}` block for environment-parity
checks. A mismatch logs one line to stderr and stores `matches: false`
structurally, so later analysis can distinguish "this adapter lost
points because the recipe is worse" from "this adapter lost points
because the torch build drifted". So:

```bash
discord-sft train --config .../style_late.yaml
# -> out/lora/style-late-r16/{epoch-1,epoch-2,epoch-3,final}/

discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --adapter out/lora/style-late-r16/epoch-3 \
    --include-baseline --baseline-prompt profile \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label style-late
# (Default --out out/evals — do not set --out to …/evals/runs or paths double-nest.)
# -> out/evals/runs/<rid>.json, with run["training_config"] = {...} mirroring
#    the exact lora.r / target_modules / layers_to_transform used, and the
#    base-model variant prompted with per-persona style bullets derived
#    from profiles.json (strongest no-LoRA control).
```

`discord-sft eval compare` can then filter or group by any training
hyperparameter without extra plumbing.

### Baseline-prompt modes

The `--baseline-prompt {minimal,style,profile}` flag controls what system
prompt the **no-adapter** (base-model) variant sees. LoRA variants always
get the val.jsonl training prompt because that's what they were trained
under, so the comparison is adapter-vs-prompted-base apples-to-apples:

- `minimal` (default) — identical prompt to the LoRA variant. Fair
  control for "how much does the LoRA add over zero prompt help?"
- `style` — adds generic Discord-DM bullets (short replies, lowercase
  start, informal fillers, no "As an AI"). Use when no `profiles.json`
  exists.
- `profile` — adds per-persona bullets mined from `profiles.json`:
  length distribution, lowercase-start rate, top n-gram fillers, emoji
  density, burst rate. Directly targets each judge axis and heuristic.

The chosen mode is recorded in the eval run JSON as
`persona.baseline_prompt_mode`, and each persona-generation log entry
stores the exact system string it was evaluated with, so eval-compare
can attribute score deltas to prompt strength rather than LoRA strength
(or vice versa).

## Writing your own config

All top-level keys: `run_name`, `model`, `data`, `lora`, `train`,
`checkpoint`. Unknown keys raise a `ValueError` at load time so typos
surface immediately.

Layer selection:

- `lora.layers_to_transform: [40, 41, 42, 43, 44, 45, 46, 47]` — explicit
  indices, validated against `model.config.num_hidden_layers`.
- `lora.layers_last_pct: 0.5` — shortcut for "last 50 % of layers".
  Expanded to a concrete index list in `config.resolved.yaml`.
- Neither → LoRA on every layer (PEFT default).
- Setting both is rejected.

Target modules:

- Default = all 7 attention + MLP projections + `gate_up_proj`.
- `["o_proj", "down_proj"]` = style / personality recipe.
- `["q_proj", "k_proj", "v_proj", "o_proj"]` = attention-only.

`lora.alpha` defaults to `r` per Unsloth guidance (not `2*r`).

Resume/continue training:

- Set `train.resume_adapter_path` to an existing adapter checkpoint directory
  (for example `out/lora/style-late-r16/final`).
- Or pass `--resume-adapter ...` on the CLI to override YAML.
- This loads prior LoRA weights as trainable before the new run starts; it is
  a continuation of adapter weights, not a full optimizer-state resume.

## LoRA Search Commands

For the rank/alpha sweep command matrix and probe workflow, use
[`lora_search/README.md`](lora_search/README.md). This keeps search-specific
instructions in one place and keeps this page focused on canonical training
configs.

## Known gotchas

- **`UNSLOTH_COMPILE_DISABLE=1`** is set automatically before the Unsloth
  import; this avoids the `expected mat1 and mat2 to have the same dtype`
  crash `torch.compile` triggers on MoE + LoRA.
- **MoE + ParamWrapper dropout constraint**: if PEFT routes through
  `lora.ParamWrapper`, non-zero `lora.dropout` can fail with
  `ValueError: ... does not work with lora_dropout != 0`. Set
  `lora.dropout: 0.0` for those runs.
- **Router layers** stay frozen (Unsloth default for stability).
- **Context length** above 2048 tokens tends to OOM on a single H100 80 GB
  once the backward pass allocates activation memory. Raise at your own
  risk.
