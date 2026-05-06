# discord-sft

Turn a **Discrub** Discord DM export into a clean, multi-persona, ShareGPT-style
SFT corpus for Qwen (or any chat LLM), with high quality data heuristics and
an optional LLM-as-judge for persona fit. Related work:
[Llama 3.1 8B SFT](https://github.com/dankit/llama_3.1_8b_base_sft),
[computer-use agents research](https://github.com/dankit/adversarial-agents-research-outline).

Place raw Discrub exports under a folder you choose (the UI defaults to `discord_messages/`
at the repo root). Generated artifacts — `out/` LoRA checkpoints, eval logs, curated JSONL —
stay on disk locally and are ignored by git (see `.gitignore`).

## Privacy and responsibility

Discord exports and anything derived from them (curated logs, ShareGPT JSONL,
checkpoints, eval artifacts) can include **identifiers, conversation content,
and inferred relationships**. You—not this repository—are responsible for how
you collect, process, store, and share that data.

- Respect **Discord’s Terms of Service**, **Community Guidelines**, and
  applicable privacy law. Only process exports you are **allowed** to use
  (typically your own data, or with clear consent).
- Do **not** commit raw exports, checkpoints, or datasets that identify people to
  a public fork or publication without permission and a clear lawful basis.
- Heuristics and optional judges **reduce risk** (PII scrub, dedup gates) but are
  not a guarantee of anonymity or suitability for redistribution.
- API keys (`OPENROUTER_API_KEY`, `HF_TOKEN`, W&B credentials, etc.) belong in
  your environment or private config—never in git (see `.env.example`).

This software is provided as a tool; use it in compliance with your obligations
to Discord, conversation participants, and third-party services you call.

## Documentation

| Topic | Where |
| ----- | ----- |
| Training (Unsloth, `uv`/lockfile, GH200, merge-peft, W&B, config keys, gotchas) | [`discord_sft/training/README.md`](discord_sft/training/README.md) |
| Evaluation (`lmms-eval`, bootstrap, vLLM, benchmarks, artifacts, scope limits) | [`discord_sft/evals/README.md`](discord_sft/evals/README.md) |
| Shipped YAML recipes and LoRA module guide | [`discord_sft/training/configs/README.md`](discord_sft/training/configs/README.md) |
| Gradient probe matrix and probe → eval workflow | [`discord_sft/training/lora_search/README.md`](discord_sft/training/lora_search/README.md) |

## Install

This project uses [`uv`](https://github.com/astral-sh/uv); a committed [`uv.lock`](uv.lock) pins dependencies for reproducible training and eval on Linux **aarch64** (heavy extras such as `[train]` and `[evals]` are oriented to that target).

```bash
pip install uv                      # one-time
uv sync --extra dev                 # base + pytest
```

Stack extras as needed:

```bash
uv sync --extra dev --extra ui --extra tokenizers --extra lang
```

Common optional groups: `ui` (Streamlit `discord-sft ui`), `tokenizers` (`stats`, `validate-chat-template`), `lang` (`langdetect` in curation), `train` (Unsloth SFT), `evals` / `evals-gpu` (benchmark harness). **`[train]` and `[evals]` cannot be combined in one `uv sync`** (transformers pin conflict)—use separate virtualenvs on the same machine. Full matrix, GH200 bootstrap scripts, bootstrap for `lmms-eval`, lockfile regeneration, and `pip install -e` fallbacks: **[training README](discord_sft/training/README.md#install)** and **[evals README](discord_sft/evals/README.md#2-install)**.

On macOS/Windows, the data pipeline (`ingest` → `build-sft`, `stats`, `ui`) works via editable install without the lockfile, for example `uv pip install -e "."`; see the training readme for cross-platform testing notes.

## Pipeline

```
Discrub JSON pages
      │
      ▼  discord-sft ingest
 normalized message log (parquet/jsonl per folder)
      │
      ▼  discord-sft curate
 sessions.jsonl  (burst-merged, reply-inlined, gap-split, PII-scrubbed, near-dup-collapsed)
      │  (optional: discord-sft curate-sweep → JSONL of metrics per setting grid)
      ▼  discord-sft build-sft
   train.jsonl  /  val.jsonl  /  balance_report.json  /  turn_length_report.json  (ShareGPT; one sample per session per persona, ending on that persona's last line)
      │
      ├──► discord-sft stats         (total tokens; compare Qwen3 vs Qwen3.5 tokenizers)
      ├──► discord-sft validate-chat-template  (optional; needs `[tokenizers]` or `[train]`)
      ├──► discord-sft fingerprint   (per-persona style profile; feeds eval-heuristics)
      └──► discord-sft eval          (benchmarks + persona evals; [runbook](discord_sft/evals/README.md))
```

## Quick start

```bash
discord-sft ingest    --source discord_messages --out out/messages --format parquet
discord-sft curate    --source discord_messages --out out/curated
discord-sft curate-sweep --source discord_messages --out out/curate_sweep.jsonl \
    --sweep-session-gap-min 30,60,120 --no-near-dedup
discord-sft build-sft --input out/curated/sessions.jsonl --out out/sft
discord-sft validate-chat-template --input out/sft/train.jsonl --tokenizer Qwen/Qwen3.5-35B-A3B
discord-sft stats     --input out/sft/train.jsonl --tokenizer Qwen/Qwen3-8B --tokenizer Qwen/Qwen3.5-35B-A3B
discord-sft fingerprint --input out/sft/train.jsonl --out out/sft/profiles.json
discord-sft eval-heuristics \
    --references refs.txt --generated gen.txt \
    --profile out/sft/profiles.json --persona <snowflake_id_from_profiles.json>
```

### UI

`uv sync --extra ui` (or `pip install -e ".[ui]"`), then `discord-sft ui`. Press Ctrl+C once to stop; the launcher terminates the Streamlit child. Sections: **Home**, **Data** (ingest / curate / build SFT), **Analyze** (fingerprint, tokens, browsers), **Train**, **Evaluate**.

## Output sample (one line of `train.jsonl`)

```json
{
  "system": "You are user1 chatting with user2.",
  "conversations": [
    {"from": "user", "value": "i can trade on US hours"},
    {"from": "assistant", "value": "go set alarm for 6h\nthats when markets open"}
  ],
  "meta": {
    "persona_id": "123123123020202",
    "persona_name": "user1",
    "counterparty_ids": ["09123912039123"],
    "session_id": "user2#2025-07-02T17:09:14Z",
    "num_turns": 2
  }
}
```

## Quality heuristics (summary)

Implementation lives under `discord_sft.data_prep.*`: drop non-chat message types and bad replies; strip emojis/mentions and optional URLs; PII scrub; burst-merge and reply-inline; gap-based sessions; session quality gates (min turns, author mix); exact and MinHash near-deduplication; optional `langdetect` filter with `[lang]`.

## Tuning curation

After **curate**, read `curate_report.json` (or the Data → Curate UI): compare `sessions_built` vs `sessions_kept` and use `dropped_*` counts to choose which gate to adjust. Change one knob per run and a new output directory. Align **`--min-turns`** with desired ShareGPT lengths; for many `num_turns: 2` rows, use `curate … --min-turns 2`, then check `turn_length_report.json` / `balance_report.json` after `build-sft`. Metrics-only grid: `discord-sft curate-sweep` (see `--help`; output JSONL has `params` + full `report` per combo).

## Multi-persona training data

`build-sft` emits at most one ShareGPT row per session per speaking author (window ends on that persona’s last turn). Persona balance (`--balance median` by default), ShareGPT turn caps (`--min-sharegpt-turns` / `--max-sharegpt-turns`), and optional `--turn-mix` are documented in `--help`. **LoRA training, recipes, checkpoints, `run.json`, merge-for-vLLM:** [`discord_sft/training/README.md`](discord_sft/training/README.md).

## Training (pointer)

```bash
uv sync --extra train    # in a train-only venv; see training README

discord-sft train --config discord_sft/training/configs/qwen35_a3b_style_late.yaml
```

Checkpoints live under `out/lora/<run_name>/`. For eval without MoE adapter quirks, merge adapters then eval the dense folder—see **merge-peft** in the [training README](discord_sft/training/README.md#merge-peft-bake-lora-into-base-weights-vllm-without-adapters). Shipped YAML reference: [`discord_sft/training/configs/README.md`](discord_sft/training/configs/README.md).

## Evaluation (pointer)

`discord-sft eval` runs `lmms-eval` benchmarks plus native persona evals; each run is `out/evals/runs/<run_id>.json`. **Install, commands, tables, vLLM, `--out` layout (use the eval root, not `…/runs`), and what is *out of scope* for this harness:** [`discord_sft/evals/README.md`](discord_sft/evals/README.md).

## Library surface

- `discord_sft.data_prep` — ingest, normalize, curate, SFT build/split/balance.
- `discord_sft.analysis` — `tokstats`, `fingerprint`, `heuristics`.
- `discord_sft.evals` — runner, judge, benchmarks ([runbook](discord_sft/evals/README.md)).
