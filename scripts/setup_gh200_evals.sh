#!/usr/bin/env bash
set -euo pipefail

# GH200 helper: sync eval deps, then align torch/vLLM/transformers/lmms-eval for Qwen3.5 evals.
#
# vLLM + PyTorch installs follow upstream “Pre-built wheels”:
#   https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#pre-built-wheels
# — use ``uv pip install`` with ``--torch-backend=…`` (or ``UV_TORCH_BACKEND``) so CUDA
# wheels resolve from PyTorch’s index; prefer the vLLM-pulled torch ABI over a mismatched PyPI torch.
#
# Usage:
#   python3 -m venv .venv-evals   # recommended: separate from training (.venv / .venv-train)
#   source .venv-evals/bin/activate
#   bash scripts/setup_gh200_evals.sh
#
# If your venv is not named `.venv`, set GH200_VENV_DIR (does not need to match the dirname
# of VIRTUAL_ENV if you pass an explicit absolute path).
# Example without relying on cwd defaults:
#   GH200_REPO_ROOT=/path/to/discord_sft GH200_VENV_DIR=.venv-evals \\
#     bash scripts/setup_gh200_evals.sh   # activate that venv first
#
# Defaults: Linux aarch64 + CUDA 12.8 wheels (matches common GH200 ``nvidia-smi`` “CUDA Version: 12.8”).
#   VLLM_PIP_SPEC                     stable requirement (default: vllm==0.19.1, matches ``uv.lock`` / [evals])
#   GH200_VLLM_TORCH_BACKEND /        ``uv pip`` ``--torch-backend`` (default: cu128).
#   UV_TORCH_BACKEND                  If set, overrides GH200_VLLM_TORCH_BACKEND (``auto`` = driver-guided).
#
# Other CUDA wheel lines (override only): e.g. ``GH200_VLLM_TORCH_BACKEND=cu129``.
#
# Nightly:
#   GH200_VLLM_NIGHTLY=1             ``uv pip install -U --pre vllm`` from wheels.vllm.ai
#   GH200_VLLM_NIGHTLY_URL           Override index (default: ``https://wheels.vllm.ai/nightly``; see vLLM docs for variants)
#
# Optional GitHub release wheel (per vLLM “Pre-built wheels” URL pattern):
#   GH200_VLLM_GITHUB_RELEASE=1      ``curl`` + ``jq`` latest tag → ``vllm-…+cu${CUDA_TAG}.whl``
#   GH200_VLLM_CUDA_TAG               default 128 → ``https://download.pytorch.org/whl/cu128``
#
# Other overrides:
#   GH200_VENV_DIR   venv path (default: active $VIRTUAL_ENV, else `.venv`) — use `.venv-evals` with README flow
#   GH200_REPO_ROOT   repo root (default: $PWD); script `cd`s here before resolving relative venv paths
#   TOKENIZERS_SPEC   default tokenizers>=0.22.0,<=0.23.0 (transformers@main upper bound)
#   GH200_SKIP_VERIFY   set to 1 to skip end-of-script import/Qwen config check (offline/no-HF hosts)

GH200_REPO_ROOT="${GH200_REPO_ROOT:-$PWD}"
cd "$GH200_REPO_ROOT"

if [[ -n "${GH200_VENV_DIR:-}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  GH200_VENV_DIR="$VIRTUAL_ENV"
else
  GH200_VENV_DIR=".venv"
fi

if [[ ! -d "$GH200_VENV_DIR" ]]; then
  echo "error: venv not found: $GH200_VENV_DIR" >&2
  echo "  Create it: python3 -m venv .venv-evals && source .venv-evals/bin/activate" >&2
  echo "  Or point at it: GH200_VENV_DIR=.venv-evals bash scripts/setup_gh200_evals.sh (with that venv activated)" >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "error: activate your virtualenv first (must match GH200_VENV_DIR when set), e.g.:" >&2
  echo "  source ${GH200_VENV_DIR}/bin/activate" >&2
  exit 1
fi

_venv_canon="$(cd "$GH200_VENV_DIR" && pwd)"
_virt_canon="$(cd "$VIRTUAL_ENV" && pwd)"
if [[ "$_venv_canon" != "$_virt_canon" ]]; then
  echo "error: active venv ($VIRTUAL_ENV) does not match GH200_VENV_DIR ($GH200_VENV_DIR)" >&2
  echo "  source ${GH200_VENV_DIR}/bin/activate   # or unset GH200_VENV_DIR to use the active env only" >&2
  exit 1
fi
unset _venv_canon _virt_canon

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  echo "[gh200-setup] uv not found; installing standalone uv to ~/.local/bin"
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

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

TOKENIZERS_SPEC="${TOKENIZERS_SPEC:-tokenizers>=0.22.0,<=0.23.0}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm==0.19.1}"
# uv: https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection
_VLLM_TORCH_BACKEND="${UV_TORCH_BACKEND:-${GH200_VLLM_TORCH_BACKEND:-cu128}}"

effective_nightly_url() {
  if [[ -n "${GH200_VLLM_NIGHTLY_URL:-}" ]]; then
    echo "${GH200_VLLM_NIGHTLY_URL}"
  else
    echo "https://wheels.vllm.ai/nightly"
  fi
}

install_vllm_github_release_wheel() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "error: GH200_VLLM_GITHUB_RELEASE=1 requires jq (e.g. apt install jq)" >&2
    exit 1
  fi
  local cuda_tag="${GH200_VLLM_CUDA_TAG:-128}"
  local vllm_ver arch url pti
  vllm_ver="$(curl -sSf https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')"
  arch="$(uname -m)"
  url="https://github.com/vllm-project/vllm/releases/download/v${vllm_ver}/vllm-${vllm_ver}+cu${cuda_tag}-cp38-abi3-manylinux_2_35_${arch}.whl"
  pti="https://download.pytorch.org/whl/cu${cuda_tag}"
  echo "[gh200-setup] vLLM GitHub release wheel per docs:"
  echo "  $url"
  echo "[gh200-setup] PyTorch index: $pti"
  # Same pattern as vLLM docs: explicit cu-tagged wheel + PyTorch CUDA index only.
  "$UV_BIN" pip install -U "$url" --extra-index-url "$pti" \
    --index-strategy unsafe-best-match
}

install_stable_torch_vllm() {
  echo "[gh200-setup] vLLM pre-built wheels (stable): $VLLM_PIP_SPEC + --torch-backend=${_VLLM_TORCH_BACKEND}"
  echo "[gh200-setup] refs: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#pre-built-wheels"
  "$UV_BIN" pip install --torch-backend="${_VLLM_TORCH_BACKEND}" -U "$VLLM_PIP_SPEC"
}

install_nightly_torch_vllm() {
  local nightly_url
  nightly_url="$(effective_nightly_url)"
  echo "[gh200-setup] vLLM nightly (uv + --pre — see vLLM “Install the latest code”):"
  echo "  --extra-index-url ${nightly_url}"
  echo "  --torch-backend ${_VLLM_TORCH_BACKEND}"
  "$UV_BIN" pip install --torch-backend="${_VLLM_TORCH_BACKEND}" -U --pre \
    vllm \
    --extra-index-url "${nightly_url}" \
    --index-strategy unsafe-best-match
}

reconcile_torch_vllm() {
  if [[ "${GH200_VLLM_GITHUB_RELEASE:-}" == "1" ]]; then
    echo "[gh200-setup] reconciling GitHub release vLLM wheel after lmms-eval"
    install_vllm_github_release_wheel
  elif [[ "${GH200_VLLM_NIGHTLY:-}" == "1" ]]; then
    echo "[gh200-setup] reconciling vLLM nightly after lmms-eval"
    install_nightly_torch_vllm
  else
    echo "[gh200-setup] reconciling $VLLM_PIP_SPEC + torch-backend after lmms-eval (ABI / deps)"
    "$UV_BIN" pip install --torch-backend="${_VLLM_TORCH_BACKEND}" -U "$VLLM_PIP_SPEC"
  fi
}

echo "[gh200-setup] syncing project extras (evals + lang) into active venv"
# Without ``--active``, ``uv sync`` installs into the project default (usually ``.venv``),
# while this script expects the *activated* env (e.g. ``.venv-evals``) — then ``discord_sft``
# is missing and bootstrap fails with No module named 'discord_sft'.
"$UV_BIN" sync --active --extra evals --extra lang

if [[ "${GH200_VLLM_GITHUB_RELEASE:-}" == "1" ]]; then
  install_vllm_github_release_wheel
elif [[ "${GH200_VLLM_NIGHTLY:-}" == "1" ]]; then
  install_nightly_torch_vllm
else
  install_stable_torch_vllm
fi

echo "[gh200-setup] re-pinning transformers main for qwen3_5_moe support"
"$UV_BIN" pip install --no-deps -U \
  "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
"$UV_BIN" pip install -U huggingface_hub safetensors
"$UV_BIN" pip install "$TOKENIZERS_SPEC"

echo "[gh200-setup] ensuring lmms-eval + task libs (avoid pulling torch/CUDA/transformers chaos)"
"$UV_BIN" pip install -U langdetect immutabledict
# ``pip install -U lmms-eval`` resolves a full dependency tree and may replace CUDA torch stacks
# on aarch64 → nccl ABI errors and broken Qwen configs.
"$UV_BIN" pip install -U --no-deps lmms-eval

echo "[gh200-setup] re-pinning tokenizers (lmms-eval may upgrade it past transformers-main bounds)"
"$UV_BIN" pip install "$TOKENIZERS_SPEC"

reconcile_torch_vllm

echo "[gh200-setup] re-pinning transformers@main after vLLM (resolver often drops git transformers)"
"$UV_BIN" pip install --no-deps -U \
  "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
"$UV_BIN" pip install -U huggingface_hub safetensors
"$UV_BIN" pip install "$TOKENIZERS_SPEC"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LMMS_SRC_DIR="${LMMS_SRC_DIR:-../lmms-eval-main}"
export LMMS_REPO_URL="${LMMS_REPO_URL:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
export LMMS_REF="${LMMS_REF:-4caa4a67ee03640734e824449ea10afa60c71719}"

echo "[gh200-setup] lmms-eval bootstrap (task overlay + patches + NLTK + health)"
BOOTSTRAP_ARGS=()
if [[ "${BOOTSTRAP_QUIET:-}" == "1" ]]; then
  BOOTSTRAP_ARGS+=(--quiet)
fi
if [[ "${BOOTSTRAP_NO_VLLM_HEALTH:-}" == "1" ]]; then
  BOOTSTRAP_ARGS+=(--no-vllm-health)
fi
VENV_PYTHON="${VIRTUAL_ENV}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: missing $VENV_PYTHON (broken venv?)" >&2
  exit 1
fi
"$VENV_PYTHON" "$SCRIPT_DIR/bootstrap_lmms_eval_env.py" "${BOOTSTRAP_ARGS[@]}"

echo
echo "[gh200-setup] export this in your current shell before eval:"
echo "  export VLLM_USE_DEEP_GEMM=0"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "  # and set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) for gated dataset downloads"
fi
echo
GH200_MODE="stable vLLM + uv --torch-backend=${_VLLM_TORCH_BACKEND}"
if [[ "${GH200_VLLM_GITHUB_RELEASE:-}" == "1" ]]; then
  GH200_MODE="GitHub release vLLM wheel + cu${GH200_VLLM_CUDA_TAG:-128} PyTorch index"
elif [[ "${GH200_VLLM_NIGHTLY:-}" == "1" ]]; then
  GH200_MODE="vLLM nightly $(effective_nightly_url) + uv --torch-backend=${_VLLM_TORCH_BACKEND}"
fi
echo "[gh200-setup] mode: ${GH200_MODE}"
echo "[gh200-setup]       GH200_VLLM_NIGHTLY=1 for nightly; GH200_VLLM_GITHUB_RELEASE=1 for docs’ release .whl URL"
echo "[gh200-setup]       GH200_VLLM_NIGHTLY_URL=https://wheels.vllm.ai/nightly/cu129 (example) when you need another CUDA line"
echo "[gh200-setup] vLLM spec: ${VLLM_PIP_SPEC}"
echo
if [[ "${GH200_SKIP_VERIFY:-}" == "1" ]]; then
  echo "[gh200-setup] skipping verification (GH200_SKIP_VERIFY=1); manual check:"
  echo "  $VENV_PYTHON -c \"import torch, vllm; print('torch cuda:', torch.version.cuda); print('vllm:', vllm.__version__)\""
else
  echo "[gh200-setup] verification (imports + Qwen/Qwen3.5-35B-A3B config)"
  "$VENV_PYTHON" <<'PY'
import torch, vllm, lmms_eval, transformers
from transformers import AutoConfig

print("torch cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("model_type:", AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").model_type)
PY
fi
echo
echo "[gh200-setup] torch uses torch.version.cuda (not version_cuda)."
echo "[gh200-setup] If torch still hits undefined symbol ncclCommResume:"
echo "  unset LD_LIBRARY_PATH  # avoids older system libnccl shadowing pip nvidia-nccl"
echo "  ldd \"$VIRTUAL_ENV\"/lib/*/site-packages/torch/lib/libtorch_cuda.so | grep -i nccl"
