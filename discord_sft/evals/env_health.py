"""Shared lmms-eval stack checks: patches, optional GPU health, doctor output.

Used by ``scripts/bootstrap_lmms_eval_env.py`` and ``discord-sft eval doctor``.
"""
from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Pinned overlay / clone ref (keep in sync with scripts/bootstrap_lmms_eval_env.py defaults).
DEFAULT_LMMS_REPO_URL = "https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
DEFAULT_LMMS_REF = "4caa4a67ee03640734e824449ea10afa60c71719"

_TASK_PY_OLD = "results = [res.strip() for res in results]"
_TASK_PY_NEW = "results = [(res or '').strip() for res in results]"

# Mirrors scripts/patch_lmms_openai_max_new_tokens.py — if unpatched, this matches.
_RE_MIN_4096 = re.compile(
    r"min\(\s*"
    r"request_gen_kwargs\.get\(\s*"
    r"(['\"])max_new_tokens\1\s*,\s*(\d+)\s*\)\s*"
    r",\s*4096\s*\)",
    re.MULTILINE,
)


def lmms_eval_site_dir() -> Path:
    import lmms_eval  # type: ignore[import-untyped]

    return Path(lmms_eval.__file__).resolve().parent


def openai_chat_py_path() -> Path:
    return lmms_eval_site_dir() / "models" / "chat" / "openai.py"


def task_py_path() -> Path:
    return lmms_eval_site_dir() / "api" / "task.py"


def patch_task_py_none_safe_strip() -> tuple[bool, str]:
    """Idempotent task.py hotfix for None completions. Returns (wrote_file, message)."""
    task_py = task_py_path()
    if not task_py.is_file():
        return False, f"task.py not found: {task_py}"
    text = task_py.read_text(encoding="utf-8")
    if _TASK_PY_NEW in text:
        return False, f"already patched: {task_py}"
    if _TASK_PY_OLD not in text:
        return False, f"expected line not found (upstream changed?): {task_py}"
    task_py.write_text(text.replace(_TASK_PY_OLD, _TASK_PY_NEW, 1), encoding="utf-8")
    return True, f"patched: {task_py}"


def openai_max_tokens_clamp_present() -> bool:
    """True if the known 4096 clamp pattern is still present (patch not applied)."""
    p = openai_chat_py_path()
    if not p.is_file():
        return False
    return _RE_MIN_4096.search(p.read_text(encoding="utf-8")) is not None


def patch_openai_max_new_tokens_via_script(scripts_dir: Path) -> int:
    """Run ``patch_lmms_openai_max_new_tokens.py`` in the same interpreter."""
    script = scripts_dir / "patch_lmms_openai_max_new_tokens.py"
    if not script.is_file():
        print(f"error: patch script missing: {script}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(script)])


def openai_disable_thinking_patch_present() -> bool:
    """True if lmms-eval openai.py has env-gated disable-thinking support."""
    p = openai_chat_py_path()
    if not p.is_file():
        return False
    text = p.read_text(encoding="utf-8")
    return 'LMMS_EVAL_DISABLE_THINKING' in text and "enable_thinking" in text


def patch_openai_disable_thinking_via_script(scripts_dir: Path) -> int:
    """Run ``patch_lmms_openai_disable_thinking.py`` in the same interpreter."""
    script = scripts_dir / "patch_lmms_openai_disable_thinking.py"
    if not script.is_file():
        print(f"error: patch script missing: {script}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(script)])


def overlay_tasks_from_clone(src_root: Path) -> None:
    """Copy ``lmms_eval/tasks`` from a git checkout into site-packages."""
    import shutil

    src = src_root / "lmms_eval" / "tasks"
    dst = lmms_eval_site_dir() / "tasks"
    if not src.is_dir():
        raise FileNotFoundError(f"lmms source tasks dir not found: {src}")
    if not dst.parent.is_dir():
        raise FileNotFoundError(f"lmms installed package dir missing: {dst.parent}")
    for name in os.listdir(src):
        s = src / name
        d = dst / name
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)


def ensure_lmms_eval_clone(
    *,
    src_dir: Path,
    repo_url: str,
    ref: str,
    git_token: str | None,
) -> None:
    """Clone or fetch/checkout ``ref`` into ``src_dir``."""
    src_dir = src_dir.resolve()
    clone_url = repo_url
    if git_token and repo_url.startswith("https://"):
        clone_url = f"https://x-access-token:{git_token}@{repo_url.removeprefix('https://')}"

    if not (src_dir / ".git").is_dir():
        src_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", clone_url, str(src_dir)],
            check=True,
        )
    subprocess.run(["git", "-C", str(src_dir), "fetch", "--all", "--tags", "--prune"], check=True)
    subprocess.run(["git", "-C", str(src_dir), "checkout", ref], check=True)


def download_nltk_ifeval(*, quiet: bool) -> int:
    try:
        import nltk  # noqa: F401
    except ImportError:
        print(
            "error: nltk is not installed. Install eval extras: uv sync --extra evals",
            file=sys.stderr,
        )
        return 1
    import nltk as nltk_mod

    names = ("punkt_tab", "punkt")
    for name in names:
        nltk_mod.download(name, quiet=quiet)
    if not quiet:
        print("NLTK resources for IFEval:", ", ".join(names))
    return 0


def patch_report() -> dict[str, Any]:
    """Structured patch / overlay hints for ``eval doctor``."""
    report: dict[str, Any] = {}
    task_py = task_py_path()
    report["lmms_eval_site"] = str(lmms_eval_site_dir())
    if task_py.is_file():
        text = task_py.read_text(encoding="utf-8")
        report["task_py_none_safe"] = _TASK_PY_NEW in text
    else:
        report["task_py_none_safe"] = None
    oa = openai_chat_py_path()
    if oa.is_file():
        report["openai_4096_clamp_still_present"] = openai_max_tokens_clamp_present()
        report["openai_disable_thinking_patch_present"] = openai_disable_thinking_patch_present()
    else:
        report["openai_4096_clamp_still_present"] = None
        report["openai_disable_thinking_patch_present"] = None
    return report


def library_versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for mod in ("lmms_eval", "transformers", "torch", "peft", "accelerate", "vllm", "tokenizers"):
        try:
            if mod == "tokenizers":
                out["tokenizers"] = importlib.metadata.version("tokenizers")
            else:
                m = __import__(mod)
                out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            continue
    return out


def run_gpu_stack_health(
    *,
    require_vllm: bool = True,
    qwen_config_smoke: bool = True,
) -> int:
    """Import torch/vllm/lmms_eval/transformers and verify qwen3_5_moe mapping.

    If ``require_vllm`` is False, skip vLLM and CUDA checks (e.g. CPU-only doctor).
    """
    try:
        import torch
    except Exception as e:
        exe = sys.executable
        print(
            "error: cannot import torch "
            f"({type(e).__name__}: {e})\n"
            f"  interpreter: {exe}",
            file=sys.stderr,
        )
        print(
            "  Torch is listed under `[evals]`; if ``import torch`` fails here your venv "
            "never got a CUDA build or the resolver dropped it.",
            file=sys.stderr,
        )
        print(
            "  Typical fixes:\n"
            "    uv sync --extra evals --extra lang\n"
            "  GH200 stack (matches scripts/setup_gh200_evals.sh — see vLLM GPU install docs):\n"
            "    bash scripts/setup_gh200_evals.sh\n"
            "    # or uv pip install vllm ... --torch-backend=cu128\n"
            "  If you intentionally have no GPU stack yet (overlay/NLTK only):\n"
            "    python scripts/bootstrap_lmms_eval_env.py --skip-health",
            file=sys.stderr,
        )
        return 1

    if require_vllm:
        try:
            import vllm
            import vllm._C  # noqa: F401
        except ImportError as e:
            print(f"error: vllm not importable ({e})", file=sys.stderr)
            return 1
    else:
        vllm = None  # type: ignore[assignment]

    try:
        import lmms_eval  # noqa: F401
        import transformers
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    if vllm is not None:
        print("vllm:", vllm.__version__)
    print("transformers:", transformers.__version__)
    try:
        print("tokenizers:", importlib.metadata.version("tokenizers"))
    except importlib.metadata.PackageNotFoundError:
        print("tokenizers: (not installed)")

    if qwen_config_smoke:
        if "qwen3_5_moe" not in CONFIG_MAPPING:
            print(
                "error: transformers build missing qwen3_5_moe in CONFIG_MAPPING; "
                "reinstall transformers@main (or use evals-gpu extra) and keep "
                "tokenizers within transformers' declared upper bound.",
                file=sys.stderr,
            )
            return 1
        try:
            cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B")
        except Exception as e:
            print(
                f"warning: could not download Qwen/Qwen3.5-35B-A3B config ({e}); "
                "qwen3_5_moe is registered locally.",
                file=sys.stderr,
            )
        else:
            print("model_type:", cfg.model_type)

    return 0


def print_doctor_report(*, require_vllm: bool = False) -> int:
    """Human-readable one-shot for ``discord-sft eval doctor``."""
    try:
        import lmms_eval  # noqa: F401
    except ImportError:
        print(
            "error: lmms-eval is not installed. Try: uv sync --extra evals",
            file=sys.stderr,
        )
        return 1

    vers = library_versions()
    print("versions:")
    for k in sorted(vers):
        print(f"  {k}: {vers[k]}")

    pr = patch_report()
    print("lmms-eval site:", pr.get("lmms_eval_site"))
    t_ok = pr.get("task_py_none_safe")
    if t_ok is True:
        print("task.py None-safe strip: ok")
    elif t_ok is False:
        print("task.py None-safe strip: MISSING — run bootstrap_lmms_eval_env.py")
    else:
        print("task.py None-safe strip: unknown (task.py missing)")

    clamp = pr.get("openai_4096_clamp_still_present")
    if clamp is True:
        print("openai max_new_tokens 4096 clamp: STILL PRESENT — run bootstrap or patch_lmms_openai_max_new_tokens.py")
    elif clamp is False:
        print("openai max_new_tokens 4096 clamp: removed or upstream changed (ok)")
    else:
        print("openai patch status: unknown (openai.py missing)")

    no_think = pr.get("openai_disable_thinking_patch_present")
    if no_think is True:
        print("openai no-thinking patch for lmms tasks: present (env-gated by LMMS_EVAL_DISABLE_THINKING)")
    elif no_think is False:
        print("openai no-thinking patch for lmms tasks: MISSING — run bootstrap or patch_lmms_openai_disable_thinking.py")
    else:
        print("openai no-thinking patch status: unknown (openai.py missing)")

    return run_gpu_stack_health(require_vllm=require_vllm, qwen_config_smoke=True)
