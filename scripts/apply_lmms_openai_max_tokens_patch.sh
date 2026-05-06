#!/usr/bin/env bash
# Run from any cwd: patch installed lmms-eval (see patch_lmms_openai_max_new_tokens.py).
# Usage: bash scripts/apply_lmms_openai_max_tokens_patch.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_PY="$REPO_ROOT/scripts/patch_lmms_openai_max_new_tokens.py"
if [[ ! -f "$PATCH_PY" ]]; then
  echo "error: missing $PATCH_PY" >&2
  exit 1
fi
if command -v uv >/dev/null 2>&1; then
  (cd "$REPO_ROOT" && uv run python "$PATCH_PY")
else
  (cd "$REPO_ROOT" && python3 "$PATCH_PY")
fi
