#!/usr/bin/env bash
set -euo pipefail

# One-click GH200 training environment bootstrap (Unsloth stack).
#
# Stack policy (distinct from eval / vLLM — see README "GH200: why two virtualenvs"):
#   torch 2.10.x+cu128, transformers==5.5.0 (Unsloth max on PyPI; vLLM eval needs >=5.5.1),
#   unsloth==2026.4.8, unsloth-zoo==2026.4.9, datasets==4.3.0 (Unsloth requires <4.4).
#
# Usage:
#   GH200_VENV_DIR=.venv-train bash scripts/setup_gh200_training.sh
#
# Optional environment overrides:
#   GH200_RESET_VENV=0      keep existing venv (default: 1 / recreate)
#   GH200_VENV_DIR=.venv   venv path (default: .venv; use .venv-train if sharing the host with `.venv-evals`)
#   GH200_REPO_ROOT=...     explicit repo root (default: current directory)
#   GH200_FLASH_ATTN_WHEEL  override FlashAttention wheel URL

GH200_RESET_VENV="${GH200_RESET_VENV:-1}"
GH200_VENV_DIR="${GH200_VENV_DIR:-.venv}"
GH200_REPO_ROOT="${GH200_REPO_ROOT:-$PWD}"
GH200_FLASH_ATTN_WHEEL="${GH200_FLASH_ATTN_WHEEL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_aarch64.whl}"

cd "$GH200_REPO_ROOT"

if [[ ! -f "pyproject.toml" ]]; then
  echo "error: run this from the repo root (missing pyproject.toml)." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: GH200 setup requires Linux." >&2
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "error: expected aarch64 host, got '$(uname -m)'." >&2
  exit 1
fi

deactivate 2>/dev/null || true

if [[ "$GH200_RESET_VENV" == "1" ]]; then
  echo "[gh200-train] resetting ${GH200_VENV_DIR}"
  rm -rf "$GH200_VENV_DIR"
fi

if [[ ! -d "$GH200_VENV_DIR" ]]; then
  echo "[gh200-train] creating virtualenv at ${GH200_VENV_DIR}"
  python3 -m venv "$GH200_VENV_DIR"
fi

# shellcheck disable=SC1091
source "${GH200_VENV_DIR}/bin/activate"

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "[gh200-train] uv not found; installing standalone uv to ~/.local/bin"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    echo "error: need curl or wget to install uv automatically" >&2
    exit 1
  fi
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ ! -x "$UV_BIN" ]]; then
  echo "error: uv binary not executable at '$UV_BIN'" >&2
  exit 1
fi

PY_BIN="${GH200_VENV_DIR}/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  echo "error: missing python in ${GH200_VENV_DIR}" >&2
  exit 1
fi

echo "[gh200-train] upgrading pip build toolchain"
"$PY_BIN" -m ensurepip --upgrade
"$UV_BIN" pip install --python "$PY_BIN" -U pip setuptools wheel

echo "[gh200-train] pinning CUDA torch stack (cu128)"
"$UV_BIN" pip install --python "$PY_BIN" \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" "torchaudio==2.10.0+cu128"

echo "[gh200-train] installing project train extras without torch resolver drift"
"$UV_BIN" pip install --python "$PY_BIN" -e ".[train]" --no-deps

echo "[gh200-train] installing pinned training dependencies"
"$UV_BIN" pip install --python "$PY_BIN" -U \
  "datasketch>=1.6" "pyarrow>=15.0" \
  "accelerate==1.13.0" "datasets==4.3.0" "peft==0.19.1" \
  "pyyaml==6.0.3" "trl==0.24.0" "transformers==5.5.0" \
  "bitsandbytes==0.49.2" "wandb>=0.16.0"

echo "[gh200-train] pinning unsloth stack"
"$UV_BIN" pip install --python "$PY_BIN" -U \
  "unsloth==2026.4.8" "unsloth-zoo==2026.4.9" --no-deps

echo "[gh200-train] installing FlashAttention wheel (default path)"
"$UV_BIN" pip install --python "$PY_BIN" --no-deps "$GH200_FLASH_ATTN_WHEEL"
"$UV_BIN" pip install --python "$PY_BIN" -U "einops"

echo "[gh200-train] running CUDA preflight"
"$PY_BIN" - <<'PY'
import platform
import flash_attn
import torch

print("arch:", platform.machine())
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("has _grouped_mm:", hasattr(torch, "_grouped_mm"))
print("flash_attn:", getattr(flash_attn, "__version__", "ok"))

assert platform.machine() == "aarch64", "not running on aarch64"
assert torch.cuda.is_available(), "CUDA not available"
assert hasattr(torch, "_grouped_mm"), "torch._grouped_mm missing"
PY

if [[ ! -f ".env" ]]; then
  cat > .env <<'EOF'
# Populate before tracked runs.
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=discord-sft
# WANDB_ENTITY=your_user_or_team
EOF
  echo "[gh200-train] created .env template at repo root"
fi

echo
echo "[gh200-train] setup complete."
echo "[gh200-train] next run command:"
echo "  source ${GH200_VENV_DIR}/bin/activate"
echo "  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  discord-sft train --config discord_sft/training/configs/qwen35_a3b_style_late.yaml"
