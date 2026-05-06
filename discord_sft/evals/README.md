# Evaluation runbook

Step-by-step commands for `discord-sft eval`: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) benchmarks plus native persona evals. Each run is saved as one JSON under `out/evals/runs/<run_id>.json`.

**`training_config` in saved runs:** Each `eval run` embeds the LoRA recipe (`config.resolved.yaml`) when possible: from **`--adapter`**, from a **local `--model`** path (merged checkpoints include YAML after **`train merge-peft`**), from **`merge_manifest.json` → `adapter_dir`**, or—when the merged tree is missing YAML but `model.name_or_path` looks like **`out/merged/<training-run>/…`**—from **`out/lora/<training-run>/config.resolved.yaml`** (when **`run_name`** in YAML matches `<training-run>`). It may be **`null`** for Hub-only base runs or when local artifacts are missing.

Relative paths in the run JSON (**`model.name_or_path`**, **`adapter_path`**, manifest **`adapter_dir`**) are resolved against the **git checkout root** (parent of the **`discord_sft`** package), not the process working directory, as long as artifacts live under the repo (see `discord_sft.evals.paths`).

**`--out` (default `out/evals`)** is the *eval root*, not the `runs` folder. The CLI always creates `runs/` and `raw/` under that root. If you pass `--out out/evals/runs`, you will get `out/evals/runs/runs/` and `out/evals/runs/raw/` by mistake.

**vLLM + Qwen3.5 MoE + `--adapter` troubleshooting:** vLLM often logs **“LoRA module `…experts.base_layer…` is not in the model's supported LoRA target modules”** for PEFT/Unsloth MoE checkpoints: expert LoRA tensors are **ignored** (only compatible targets like `q_proj` / `gate` may apply), which **skips the behavioral change you trained**. Fix: in the **training** venv run **`discord-sft train merge-peft --adapter … --output …`** and eval with **`--model <merged_dir>`** only—no `--adapter` / `--lora-modules` (see [training merge-peft](../training/README.md#merge-peft-bake-lora-into-base-weights-vllm-without-adapters)).

**Subcommands (in addition to `eval run` documented below):** `discord-sft eval doctor`, `eval list`, `eval compare`, `eval judge-persona` (score an existing `persona_generations.jsonl`), and `eval rank-style` (pairwise style ranking + stylometric fingerprint across multiple generation dumps; see `--help`).

## 1. Prerequisites

- **Data**: `out/sft/val.jsonl` from `discord-sft build-sft`. For the strongest base-model persona baseline, also run `discord-sft fingerprint` and keep `out/sft/profiles.json`.
- **Separate venv from training**: `[train]` and `[evals]` pin incompatible **`transformers`** ranges (Unsloth vs Qwen3.5 / vLLM tooling). Use one virtualenv for `uv sync --extra train` and another for `uv sync --extra evals` on the same machine. Versions and rationale: [training README — Install](../training/README.md#install).
- **Hardware / OS**: The committed `uv.lock` resolves **`[evals]` on Linux aarch64** (GH200-style hosts). `pyproject.toml` installs **vLLM 0.19.x** + **`torch<2.11`** with that extra on aarch64 (**CUDA 12.8 ``cu128``**, aligned with typical ``nvidia-smi`` **CUDA Version: 12.8**). **Bootstrap does not install vLLM** — it patches/overlays lmms-eval, NLTK, and runs a health check when vLLM is already importable. On GH200, run **`scripts/setup_gh200_evals.sh`** after `uv sync --active` to re-pin **`transformers@main`**, **`lmms-eval`** (`--no-deps`), and install **vLLM pre-built wheels** per [upstream GPU docs](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#pre-built-wheels) (default **`--torch-backend=cu128`**; **`GH200_VLLM_NIGHTLY=1`** uses **`wheels.vllm.ai/nightly`** unless overridden).
- **Optional judge**: Set `OPENROUTER_API_KEY` in the environment (see [`.env.example`](../../.env.example)) if you pass `--judge openrouter`. Default judge model is **`anthropic/claude-sonnet-4.6`** (see `discord_sft.evals.judge.OpenRouterJudge`).
- **HF auth token (recommended)**: Some `lmms-eval` tasks require authenticated dataset/model downloads. Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) in your environment or `.env` (see [`.env.example`](../../.env.example)).
- **IFEval + NLTK**: The `ifeval` task scores responses with NLTK tokenizers. NLTK 3.9+ expects the **`punkt_tab`** data package (not included in the `nltk` pip wheel). The **bootstrap** script (below) downloads it; or run `uv run python scripts/download_nltk_for_ifeval.py` once. If you see `Resource 'punkt_tab' not found`, run either and retry the eval.
- **Shared vLLM + `max_new_tokens` > 4096**: The default `--backend vllm` path uses lmms-eval’s OpenAI-compatible model code. Some `lmms-eval` releases cap `max_new_tokens` at 4096 in that layer; the bootstrap applies `scripts/patch_lmms_openai_max_new_tokens.py` so `--gen-kwargs max_new_tokens=8192` (and similar) is honored. Re-run bootstrap after `uv sync` or any reinstall that overwrites `site-packages`.

## 2. Install

| Goal | Command | Notes |
|------|---------|-------|
| Portable eval (PyPI stack, matches `uv.lock`) | `uv sync --extra evals` | Re-run **bootstrap** (below) after every reinstall that touches `lmms-eval`. |
| Git **transformers@main** + pinned **lmms-eval** (Qwen3.5 MoE eval) | `uv sync --extra evals --extra evals-gpu` | Cannot combine with `--extra train` (`tool.uv.conflicts`). Use a separate venv from training; see [training README — Install](../training/README.md#install). |
| Train + eval in one venv | _(not supported)_ | Use **`--extra train`** in one venv and **`--extra evals`** / **`evals-gpu`** in another ([training README](../training/README.md#install)). |

From the repository root:

```bash
uv sync --extra evals
```

Fallback (no lockfile reproducibility):

```bash
pip install -e ".[evals]"
```

### Bootstrap (one post-install entry)

Default `discord-sft eval run --backend vllm` uses a managed OpenAI-compatible vLLM server and lmms-eval’s `openai` model backend. After installing extras, run **once per virtualenv** (and again after any `uv sync` or reinstall that overwrites `lmms-eval`):

```bash
uv run python scripts/bootstrap_lmms_eval_env.py
```

This clones/overlays lmms-eval **tasks** at pin `LMMS_REF` (override with env vars), applies the **task.py** None-safe strip hotfix, the **OpenAI `max_new_tokens`** patch, downloads **NLTK** punkt data, and runs a short **health** import check (including `vllm._C` when vLLM is installed). Environment variables match `scripts/setup_gh200_evals.sh`: `LMMS_SRC_DIR`, `LMMS_REPO_URL`, `LMMS_REF`, `EVAL_GIT_TOKEN`.

**Linux aarch64** without a vLLM wheel (HF-only persona eval, etc.):

```bash
uv run python scripts/bootstrap_lmms_eval_env.py --no-vllm-health
```

Patch **only** the OpenAI backend (no clone/NLTK/health):

```bash
bash scripts/apply_lmms_openai_max_tokens_patch.sh
```

`bash scripts/setup_gh200_evals.sh` ends with the same bootstrap after aligning torch/vLLM and `transformers@main` (default **`vllm==0.19.1` + `--torch-backend=cu128`**; nightly if **`GH200_VLLM_NIGHTLY=1`**; override backend with **`UV_TORCH_BACKEND`** / **`GH200_VLLM_TORCH_BACKEND`**). For `--gen-kwargs` behaviour, see **§3** below.

### vLLM vs newer `transformers` (resolver timing)

**`uv lock` / `uv pip install --upgrade`** may pull **transformers** minor versions newer than each **vLLM** wheel was QA’d against—the declared **`Requires-Dist`** ranges are wide on purpose.

If upgrades break evals or model startup, treat it as ordinary API drift: **pin `transformers` to a last-known-good** compatible with **`[evals]`’s floor**, rerun **`uv lock`**, smoke **`discord-sft eval doctor --require-vllm`** plus a tiny **IFEval** run, then widen pins cautiously.

### `discord-sft eval doctor`

Prints library versions, lmms-eval **patch** status, and (by default) a transformers **qwen3_5_moe** check without requiring vLLM. On a GPU server:

```bash
uv run discord-sft eval doctor --require-vllm
```

#### IFEval-only smoke (Qwen3.5 35B + vLLM, verified)

Text-only IFEval does not need `--val` or `--profiles`. This matches a working post-patch, post–NLTK-download flow (adjust `--gpu-memory-utilization` / `--vllm-extra` for your GPU):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --tasks ifeval \
    --gen-kwargs temperature=1.0,top_p=0.95,top_k=20,max_new_tokens=8192 \
    --max-model-len 16384 --gpu-memory-utilization 0.93 \
    --vllm-extra "--max-num-seqs 512" \
    --reasoning-parser qwen3 \
    --label ifeval-smoke --limit 10 --lmms-verbosity DEBUG
```

### GH200 known-good fresh environment (Qwen model-card aligned)

When using `Qwen/Qwen3.5-35B-A3B` with `--backend vllm` on GH200, use a fresh venv from the repo root and run the helper (default **PyPI `vllm==0.19.1`** + lockfile-aligned **``cu128``** torch; overrides below):

```bash
python3 -m venv .venv
source .venv/bin/activate
uv pip install -U pip
git clone <this-repo> && cd <this-repo>   # or your checkout
bash scripts/setup_gh200_evals.sh
```

Optional: **`GH200_VLLM_NIGHTLY=1`** — vLLM nightly from **`wheels.vllm.ai/nightly`** (override with **`GH200_VLLM_NIGHTLY_URL`**). **`VLLM_PIP_SPEC='vllm==0.19.1'`** (default) can be changed to another **`0.19.x`** pin. **`GH200_VLLM_TORCH_BACKEND=cu129`** (or **`UV_TORCH_BACKEND=auto`**) if you deliberately move off **`cu128`**.

Manual equivalent (not recommended; the script orders installs to avoid ABI skew):

```bash
uv sync --extra evals --extra lang
uv pip install --torch-backend=cu128 -U "vllm==0.19.1"
uv pip install --no-deps -U \
    "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
uv pip install -U huggingface_hub safetensors "tokenizers>=0.22.0,<=0.23.0"
uv pip install -U --no-deps lmms-eval
uv pip install --torch-backend=cu128 -U "vllm==0.19.1"
uv pip install --no-deps -U \
    "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
uv pip install "tokenizers>=0.22.0,<=0.23.0"
```

Preflight checks (must succeed before `discord-sft eval run`):

```bash
python - <<'PY'
import torch
import transformers
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B")
print("OK model_type:", cfg.model_type)
print("transformers:", transformers.__version__)
print("torch.cuda:", torch.version.cuda)
PY
```

If this fails, do **not** continue to eval. Recreate the env and repeat the exact
order above. On GH200, `torch.version.cuda` should report `12.8` (or a compatible
CUDA 12.x build), not a CUDA 13-only stack.

One-command alternative (same steps scripted):

```bash
source .venv/bin/activate
bash scripts/setup_gh200_evals.sh
```

The script pins `tokenizers` to a range compatible with `transformers @ main`, reapplies it after `lmms-eval` install, and verifies `import vllm._C`, `qwen3_5_moe`, and config load in a single health check. If Hugging Face widens tokenizers bounds, set `TOKENIZERS_SPEC='…'` when invoking the script (documented in `scripts/setup_gh200_evals.sh`).

The script also applies an `lmms-eval` tasks overlay from source (pinned commit)
to avoid missing `_default_template_yaml` packaging issues seen on some builds.
It also installs task-side extras used by that overlay (for example
`langdetect` and `immutabledict` for IFEval imports).

### GH200 startup tuning (`max_num_seqs` / Mamba cache)

On Qwen3.5 MoE, vLLM can fail during startup with an error like:

`ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (...)`

This means the default vLLM concurrency is too high for your current
`--max-model-len` and memory budget. Lower `max_num_seqs` with `--vllm-extra`.
On some nightly stacks, vLLM may also fail DeepGEMM warmup on GH200; set
`VLLM_USE_DEEP_GEMM=0` to use fallback kernels. The managed vLLM server started
by `discord-sft eval run --backend vllm` sets `VLLM_USE_DEEP_GEMM=0` when unset,
so standalone `python -m vllm.entrypoints.openai.api_server` still needs the
export if you bypass the CLI.

Known-good starting points on GH200:

```bash
# Full eval, safer startup
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --max-model-len 65536 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra="--max-num-seqs 512" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label full-gh200
```

```bash
# If keeping longer context
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --max-model-len 131072 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 384" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label full-gh200-131k
```

You can raise `--max-num-seqs` gradually after a successful startup. If startup
fails again, lower it further.

## 3. Qwen3.5-35B-A3B sampling, lmms-eval `--gen_kwargs`, and vLLM extras

The [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) documents sampling **by mode** (thinking vs instruct / non-thinking). Persona eval defaults to **`instruct_general`** (non-thinking general tasks), aligned with the card:

| Mode | temperature | top_p | top_k | presence_penalty | repetition_penalty |
| ---- | ----------- | ----- | ----- | ---------------- | ------------------ |
| Thinking — general | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Thinking — precise coding | 0.6 | 0.95 | 20 | 0.0 | 1.0 |
| Instruct — general (persona default) | 0.7 | 0.8 | 20 | 1.5 | 1.0 |
| Instruct — reasoning | 1.0 | 0.95 | 20 | 1.5 | 1.0 |

- **`--qwen-sampling`**: one of `instruct_general`, `thinking_general`, `thinking_coding`, `instruct_reasoning`. Controls persona generation only (OpenAI path to vLLM and HF `default_hf_generate_fn`). Instruct presets explicitly disable Qwen thinking in the chat template; thinking presets leave it enabled. vLLM-only request fields such as `top_k` are sent through OpenAI `extra_body`. The chosen preset is stored in run JSON under `config.qwen_sampling` and `persona.qwen_sampling`.
- **`--gen-kwargs`**: forwarded to lmms-eval as `--gen_kwargs` (comma-separated, same format as `--model_args`). For the default shared vLLM path, use **`max_new_tokens=...`** in this string (it flows into lmms-eval’s OpenAI request path). Use this when you want custom sampling for generative benchmarks, e.g. `--gen-kwargs temperature=1.0,top_p=0.95,top_k=20,max_new_tokens=8192`. **After** the `patch_lmms_openai_max_new_tokens.py` hotfix, values **above 4096** are honored; without that patch, some `lmms-eval` versions still clamp at 4096 in `openai.py`. Task YAMLs are overridden by CLI `generation_kwargs` when set (see lmms-eval log: “generation_kwargs specified through cli…”). This flag does **not** change persona sampling (use `--qwen-sampling` for that).
- **Shared vLLM path and the 4096 cap**: the default backend uses lmms-eval’s OpenAI-compatible model code, which in some releases **silently** capped `max_new_tokens` at **4096** regardless of `--gen-kwargs` until patched. After installing `lmms-eval`, from the repo root run `uv run python scripts/patch_lmms_openai_max_new_tokens.py`, or `bash scripts/apply_lmms_openai_max_tokens_patch.sh` (same; `setup_gh200_evals.sh` also runs the patch). Do not paste a line that uses `"$UV_BIN"` from that script by itself. Optional: set `LMMS_EVAL_OPENAI_MAX_TOKENS` to a positive integer to use `min(requested, env)` instead of removing the cap. If you rely on upstream fixes, consider [filing an issue or PR](https://github.com/EvolvingLMMs-Lab/lmms-eval) so the clamp is removed or tied to model context and logged when applied.
- **`--vllm-extra`**: repeatable; each value is split with `shlex` and appended to the shared vLLM server argv. Example: `--vllm-extra '--language-model-only'` for text-only serving (saves memory). **Do not** use `--language-model-only` with the default task set (`mmmu_val`, `mmstar`, `screenspot_v2` need the vision stack).
- **Context length**: the model card recommends long context (e.g. ≥128k) for full thinking behavior; this repo’s defaults (`--max-model-len 16384`) target a single H100-class GPU. Raise when you have headroom.
- **vLLM version**: `[evals]` pins **vLLM 0.19.x** on Linux aarch64; compare with the [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html). Use **`GH200_VLLM_NIGHTLY=1`** in `setup_gh200_evals.sh` if you need main/nightly instead.
- **`--lmms-disable-thinking` / `--no-lmms-disable-thinking`**: Set `LMMS_EVAL_DISABLE_THINKING` to `1` or `0` for **lmms-eval benchmark** HTTP requests on the shared OpenAI path. Omit both flags to leave the environment unchanged.
- **`--apply-chat-template` (default on) vs shared vLLM**: For `--backend vllm` **without** `--no-shared-server`, the CLI **disables** forwarding `--apply_chat_template` to lmms-eval (the OpenAI-compatible backend does not support it) and prints a stderr notice.

Example combining persona preset and lmms-eval generation kwargs (default tasks include VLM; requires `--val` / `--profiles` when `persona` is in the task set):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --qwen-sampling thinking_general \
    --gen-kwargs temperature=1.0,top_p=0.95,top_k=20,max_new_tokens=8192 \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra="--max-num-seqs 512" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label smoke-thinking --limit 200
```

### Non-thinking instruct preset (no adapter)

IFEval + persona on the **base** model with **`instruct_general`**, lmms-eval thinking disabled via **`--lmms-disable-thinking`**, and model-card-aligned instruct **`--gen-kwargs`** (shorter **`--max-model-len`** than long-thinking runs):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
  --model Qwen/Qwen3.5-35B-A3B \
  --backend vllm \
  --tasks ifeval,persona \
  --val out/sft/val.jsonl \
  --profiles out/sft/profiles.json \
  --baseline-prompt profile \
  --qwen-sampling instruct_general \
  --lmms-disable-thinking \
  --gen-kwargs temperature=0.7,top_p=0.8,top_k=20,max_new_tokens=2048 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.93 \
  --vllm-extra "--max-num-seqs 512" \
  --reasoning-parser qwen3 \
  --label ifeval-persona-base-profile-nothinking
```

## 4. Discover benchmarks

```bash
uv run discord-sft eval run --list-benchmarks
```

## 5. Smoke: default task set (fast)

**`discord-sft eval run` defaults to `--backend hf`.** All vLLM examples in this doc pass **`--backend vllm`** explicitly for the shared-server + GPU path; keep the default only when you want local HF/PEFT.

Default tasks: `ifeval`, `mmmu_val`, `mmstar`, `screenspot_v2`, `persona`. `--limit` caps lmms-eval work; persona uses a derived per-persona cap when set.
For Qwen3.5 + vLLM on GH200, include `--vllm-extra="--max-num-seqs 512"` by default to avoid Mamba cache startup failures.
All **`--backend vllm` command blocks in §5–7** set `VLLM_USE_DEEP_GEMM=0` and use `uv run` (see [GH200 startup tuning](#gh200-startup-tuning-max_num_seqs--mamba-cache)); you can drop the export if your vLLM build does not need it.

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra="--max-num-seqs 512" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label smoke --limit 200
```

Linux aarch64 fallback (no vLLM in `[evals]` by default):

```bash
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend hf \
    --tasks ifeval,persona \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label smoke-hf --limit 200
```

Add optional OpenRouter judge:

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra="--max-num-seqs 512" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --label smoke --limit 200 \
    --judge openrouter
```

## 6. Base model only (no LoRA)

With **no** `--adapter`, the runner evaluates the base checkpoint. When `persona` is in `--tasks`, a shared vLLM server is used and the base variant is included automatically.

When **`persona` is the only task** (e.g. `--tasks persona`), there are **no** lmms-eval subprocesses; only the native persona path runs, with the same **`out/evals/runs/<run_id>.json`** artifact for `eval list` / `eval compare`. If you combine **`persona`** with benchmarks (default task list), lmms-eval runs as well.

**Minimal** system prompt (default; matches training prompt in `val.jsonl`):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm --tasks persona \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --label ctrl-minimal
```

**Profile**-augmented baseline (uses `profiles.json` per persona):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm --tasks persona \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --baseline-prompt profile \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --label ctrl-profile
```

**Style** baseline (generic Discord-style bullets; useful without fingerprints):

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm --tasks persona \
    --val out/sft/val.jsonl \
    --baseline-prompt style \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --label ctrl-style
```

LoRA runs always use the training system prompt from `val.jsonl`; `--baseline-prompt` applies only to no-adapter runs. The chosen mode is stored in the run JSON under `persona.baseline_prompt_mode`.

## 7. LoRA sweep and optional base in one shot

With **`--backend vllm`** and **without** `--no-shared-server` (the default), the CLI starts **one** OpenAI-compatible vLLM server, registers every adapter as `--lora-modules alias=path`, and runs **one saved run JSON per variant** (each full task set for that LoRA or baseline). **`--include-baseline`** prepends a **no-adapter** variant (label `baseline`, or `{--label}-baseline` when `--label` is set).

**Multi-adapter sweeps require this shared-server path.** If you use **`--backend hf`** or **`--no-shared-server`**, only the **first** `--adapter` / first path from `--adapter-dir` is used—pass **`--help`** and avoid multiple adapters in those modes.

**Discovery / naming:** **`--adapter`** is repeatable: bare path (alias = directory name) or `alias=/abs/path` when the left side matches `[A-Za-z0-9._-]+` (so Windows drive letters are not split). Duplicate aliases get `-2`, `-3`, … suffixes. **`--adapter-dir`** is repeatable; under each root the runner collects directories containing `adapter_config.json`. If **any** `final/` adapter exists anywhere under that tree, **only** `…/final` directories are kept; otherwise every adapter directory is included.

**Output:** After all variants finish, **`eval run` prints a JSON object to stdout** with `run_ids`, `out_files`, `n_scores`, and `tasks` (useful for scripts).

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --adapter out/lora/r8 \
    --adapter style-late=out/lora/late-mlp-r32 \
    --include-baseline \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --baseline-prompt profile \
    --label qwen35-sweep --limit 500
```

You can also point at a training output directory and let the runner discover
adapters. If any `final/` adapters exist below the directory, only those are
included; otherwise every directory with an `adapter_config.json` is included.

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm \
    --tasks persona,ifeval \
    --adapter-dir out/lora/probes \
    --include-baseline --baseline-prompt profile \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --label probe-arms
```

Named adapter alias:

```bash
export VLLM_USE_DEEP_GEMM=0
uv run discord-sft eval run \
    --model Qwen/Qwen3.5-35B-A3B --backend vllm --tasks persona \
    --adapter style-late=out/lora/style-late-r32/epoch-3 \
    --include-baseline --baseline-prompt profile \
    --val out/sft/val.jsonl --profiles out/sft/profiles.json \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --reasoning-parser qwen3 \
    --vllm-extra "--max-num-seqs 512" \
    --label qwen35-v1
```

## 8. List runs and compare

```bash
uv run discord-sft eval list
```

Replace the run IDs with those printed after your runs (or filenames under `out/evals/runs/*.json` without `.json`):

```bash
uv run discord-sft eval compare --baseline qwen35-sweep-baseline \
    qwen35-sweep-r8 qwen35-sweep-style-late
```

Filter metrics with shell globs, for example:

```bash
uv run discord-sft eval compare --baseline qwen35-sweep-baseline \
    qwen35-sweep-r8 \
    --metrics 'ifeval.*,mmmu_val.*'
```

With **no** `--metrics` filter, **`persona.*` score keys are omitted by default** so the table focuses on lmms-eval benchmarks. Add **`--include-persona-metrics`** to list them, or pass an explicit glob such as `--metrics 'persona.*'`.

## 9. Judge persona generations only (outside main eval loop)

If you already have `persona_generations.jsonl` from a previous run and only
want LLM-as-judge scores (no regeneration, no lmms-eval benchmarks), run:

```bash
uv run discord-sft eval judge-persona \
    --generations out/evals/raw/<run_id>/persona_generations.jsonl \
    --judge openrouter \
    --out out/evals/judge/<run_id>__judge.json
```

This command reads existing rows (`reference`, `generated`, `persona_id`,
`persona_name`, `system`; `context` when present). When both `context` and `system`
exist, the judge prompt uses `context` as the thread and `system` as the model’s
instructions (otherwise `system` fills context when `context` is absent). The OpenRouter
rubric judge emits scores on vocabulary / tone / length / authentic persona (plus per-sample
chain-of-thought in `reasoning`; aggregates average numeric axes only). Output JSON reports
`overall` and `per_persona` averages.

## Reference: benchmark keys

| Key | Modality | What it measures |
| --- | -------- | ---------------- |
| `ifeval` | text | Instruction following with verifiable constraints. High regression risk for chat-style SFT. |
| `mmmu_val` | image | Multi-discipline college-level VQA (validation split; use `mmmu_val` not `mmmu_test` for local scores). |
| `mmstar` | image | VQA requiring vision; text-only models cannot solve from prompt alone. |
| `screenspot_v2` | image | GUI grounding from natural-language instructions. |
| `persona` | text | Native persona evals on held-out `val.jsonl` (heuristics + optional OpenRouter rubric judge with reasoning-first JSON). |

Subset example: `--tasks ifeval,persona`. Omit `--limit` for full runs.

## Reference: `--baseline-prompt` (base model / no adapter only)

| Mode | What the base model sees | When to use |
| ---- | ------------------------ | ----------- |
| `minimal` | Exact system string from `val.jsonl` (default). | Fair control vs LoRA. |
| `style` | `minimal` plus generic Discord-DM style bullets. | No `profiles.json` or missing fingerprints. |
| `profile` | `minimal` plus per-persona bullets from `profiles.json`. | Strongest prompted baseline. |

## Reference: vLLM and LoRA

- **Base id for LoRA parity**: Shipped training YAML (e.g. `qwen35_a3b_style_late.yaml`) uses **`--model unsloth/Qwen3.5-35B-A3B`**. For vLLM evals with `--adapter`, pass **that same** HF id so the server tokenizer matches training (vLLM does not load tokenizer from the LoRA folder by default). `discord-sft eval run` warns if `--model` disagrees with `run.json`’s `base_model` next to the adapter. Base-only runs may still use `Qwen/Qwen3.5-35B-A3B` when you intend that snapshot. See also [`discord_sft/training/configs/README.md`](../training/configs/README.md) — vLLM eval parity.
- **Memory**: Qwen3.5-35B-A3B in bf16 is large on one GPU; add `--quantization fp8` if you need more KV headroom or higher `--persona-max-concurrency`. Single GPU uses vLLM default tensor parallel size 1.
- **`--max-lora-rank`**: Optional ceiling passed to vLLM for the LoRA stack. Default: **auto** from each `adapter_config.json` (`r` / `lora_r`), maximum across the sweep, fallback **16**. Set explicitly if auto-detect fails or you want a lower cap to save KV memory.
- **`--vllm-python`**: Optional Python executable used to spawn `python -m vllm.entrypoints.openai.api_server` when vLLM is installed in a different venv than `discord-sft` (see CLI help).
- **Backends**: **`--backend vllm`** (default shared server): one server, lmms-eval talks OpenAI API; LoRAs are `lora_local_path` / module routing per variant. **`--backend hf`**: local Transformers + PEFT; **`--backend vllm` + `--no-shared-server`**: legacy lmms-eval vLLM wrapper — **single adapter only**, see §7.
- **Aliases**: `--adapter /path/to/ckpt` derives the alias from the directory name; use `--adapter myname=/path` to set it. Collisions get `-2`, `-3`, … suffixes.
- **Throughput**: With the shared server, persona requests run concurrently up to `--persona-max-concurrency` (default 16).
- **Qwen3.5 auto-tuning**: If the model id looks like Qwen3.5 and `--vllm-extra` does not already set `--max-num-seqs`, the CLI appends **`--max-num-seqs 512`** (Mamba cache / startup stability); stderr notes when it does.
- **Debugging**: `--no-shared-server` avoids the managed server and uses the **single-shot** path (at most **one** `--adapter`); not for multi-LoRA sweeps.

## Artifacts

- **Run summary**: `out/evals/runs/<run_id>.json` (model spec, config, flattened `scores`, persona metadata).
- **Raw lmms-eval output**: `out/evals/raw/<run_id>/` (including parsed results paths recorded in the run JSON).
- **Persona generations**: logged under the raw run directory (path in run JSON, typically `persona_generations.jsonl`).
  Rows include `context_turns` (structured conversation history fed to the model)
  and `context` (rendered text form) in addition to `reference`, `generated`,
  `persona_id`, `persona_name`, and `system`.

Scores use dotted keys, for example:

```json
{
  "scores": {
    "ifeval.prompt_level_strict_acc.none": 0.641,
    "mmmu_val.mmmu_acc.none": 0.503,
    "mmstar.average.none": 0.572,
    "screenspot_v2.accuracy.none": 0.612,
    "persona.heuristics.123456789012345678.avg_length_diff": 1.23,
    "persona.judge.123456789012345678.overall": 4.1
  }
}
```

Use `uv run discord-sft eval compare --metrics` with `fnmatch` patterns against these keys.




## Scope and limitations

This harness does **not** currently cover long-context **agentic tool-use** benchmarks (e.g. WebArena, VisualWebArena, OSWorld, GAIA), which need live environments and a larger evaluation commitment. It also does not target broad **multilingual** retention (e.g. MMMU-Pro multilingual, MGSM, Belebele, C-Eval); this project assumes English-dominant Discord DMs. Adding tasks is done in `discord_sft/evals/benchmarks.py` without changing the run JSON schema.