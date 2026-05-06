"""Model specification shared by the standard harness and persona runner.

`ModelSpec` is a pure-data dataclass: it has no HF / torch imports at module
load time, so importing `discord_sft.evals` does not pull torch into the
lightweight core install.

Two consumers:

1. ``harness.run_lmms_eval`` — passes ``to_lmms_eval_args()`` straight through
   to the ``python -m lmms_eval --model_args ...`` CLI.
2. ``persona.run_persona_evals`` — calls ``load_hf(spec)`` to open the model
   in-process for generation. HF / torch / peft are only imported here.

Unsloth-trained models round-trip through this path unchanged: Unsloth's
``FastLanguageModel.from_pretrained`` returns an HF-compatible
``AutoModelForCausalLM`` / ``AutoTokenizer`` pair, which is what the persona
runner consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse


@dataclass
class ModelSpec:
    """Backend-agnostic description of a model to evaluate.

    Attributes:
        name_or_path: HF hub id or local path.
        backend: lmms-eval model backend. Common values:
            ``hf`` (default, works out of the box for text-only tasks),
            ``vllm`` (recommended for 35B-A3B scale),
            ``sglang``,
            ``qwen2_5_vl`` / ``qwen3_vl`` (dedicated VLM wrappers),
            ``openai`` (used internally when we proxy lmms-eval through a
            local vLLM OpenAI-compatible server for a multi-adapter sweep).
        revision: Optional HF revision / git sha pin.
        adapter_path: Optional PEFT / LoRA adapter directory. Serialised as
            ``peft=<path>`` for the ``hf`` backend and
            ``lora_local_path=<path>`` for the ``vllm`` backend (lmms-eval's
            vLLM wrapper inherits lm-eval's ``VLLM`` class which uses the
            latter key and internally builds a single
            ``LoRARequest("finetuned", 1, lora_local_path)``). Applied in
            ``load_hf`` for persona generation.
        lora_alias: Name under which a LoRA adapter has been registered
            with a running vLLM server (via ``--lora-modules alias=path``).
            When set together with ``backend == "openai"``, the persona
            generator and lmms-eval proxy target this alias instead of the
            base ``name_or_path``.
        lora_rank: Max rank across the adapters on this run. Forwarded to
            lmms-eval as ``max_lora_rank=`` on the vllm backend. If ``None``
            and an adapter is set, we fall back to ``16`` (vLLM's default).
        dtype: Torch dtype string; forwarded verbatim to lmms-eval.
        device: Device for in-process persona generation. Ignored by
            lmms-eval (it manages its own device placement).
        trust_remote_code: Needed for Qwen custom modelling code.
        extra_args: Free-form extra ``--model_args`` key/value pairs
            (e.g. ``{"max_model_len": "16384", "reasoning_parser": "qwen3"}``
            for vLLM).
    """

    name_or_path: str
    backend: str = "hf"
    revision: str | None = None
    adapter_path: str | None = None
    lora_alias: str | None = None
    lora_rank: int | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = True
    extra_args: dict[str, str] = field(default_factory=dict)

    def to_lmms_eval_args(self) -> str:
        """Render as a single ``--model_args`` string for lmms-eval.

        Notes
        -----
        - lmms-eval's vLLM / SGLang / dedicated VLM wrappers spell the model
          key ``model=`` instead of ``pretrained=``. We key off ``backend``
          so the user doesn't have to remember.
        - Adapter serialisation differs by backend:
          * ``hf`` -> ``peft=<path>`` (HF loader uses peft.PeftModel).
          * ``vllm`` -> ``lora_local_path=<path>`` plus ``enable_lora=true``
            and ``max_lora_rank=<R>`` (mirrors lm-eval's VLLM wrapper;
            lmms-eval inherits from it).
        - The ``openai`` backend (server-proxied mode) is rendered by
          :meth:`to_openai_lmms_args` instead.
        - Booleans are lower-cased (``True`` -> ``true``) to match lmms-eval's
          argument parser expectations.
        """
        if self.backend == "openai":
            # Callers should use to_openai_lmms_args directly; this fallback
            # keeps to_lmms_eval_args total so older callers don't explode.
            return self.to_openai_lmms_args()

        key = "model" if self.backend in {"vllm", "sglang"} else "pretrained"
        parts: list[tuple[str, str]] = [(key, self.name_or_path)]
        if self.revision:
            parts.append(("revision", self.revision))
        if self.adapter_path:
            if self.backend == "vllm":
                parts.append(("lora_local_path", self.adapter_path))
                parts.append(("enable_lora", "true"))
                parts.append(("max_lora_rank", str(self.lora_rank or 16)))
            else:
                parts.append(("peft", self.adapter_path))
        if self.dtype:
            parts.append(("dtype", self.dtype))
        if self.trust_remote_code:
            parts.append(("trust_remote_code", "true"))
        for k, v in self.extra_args.items():
            parts.append((k, v))
        return ",".join(f"{k}={v}" for k, v in parts)

    def to_openai_lmms_args(self, base_url: str | None = None) -> str:
        """Render ``--model_args`` for lmms-eval's ``openai`` backend.

        When we run a local vLLM OpenAI-compatible server and want to sweep
        several LoRAs against it, each sweep variant points lmms-eval at the
        server with a different ``model_version=<alias>``. ``base_url`` is
        read from ``extra_args["base_url"]`` if not passed explicitly, so
        the CLI can stash it there once per run.

        lmms-eval's OpenAI chat backend appends ``/chat/completions`` itself.
        We therefore pass an API root (typically ``.../v1``), not a full
        endpoint path, to avoid malformed URLs like
        ``.../v1/chat/completions/chat/completions``.
        """
        url = base_url or self.extra_args.get("base_url")
        model_id = self.lora_alias or self.name_or_path
        parts: list[tuple[str, str]] = [("model_version", model_id)]
        if url:
            url = _normalize_openai_base_url(url)
            parts.append(("base_url", url))
            parsed = urlparse(url)
            hostname = parsed.hostname.lower() if parsed.hostname else ""
            # vLLM's OpenAI server accepts any placeholder key; omitting api_key breaks
            # recent openai SDK versions (they require OPENAI_API_KEY / client api_key).
            # Only skip the stub when targeting OpenAI-hosted APIs (callers must pass a
            # real key via --model-arg api_key=... or OPENAI_API_KEY).
            if "api_key" not in self.extra_args and _should_stub_openai_api_key(
                hostname
            ):
                parts.append(("api_key", "EMPTY"))
        parts.append(("max_retries", "5"))
        for k, v in self.extra_args.items():
            if k in {"base_url"}:
                continue
            parts.append((k, v))
        return ",".join(f"{k}={v}" for k, v in parts)

    def slug(self) -> str:
        """Filesystem- and URL-safe identifier for this model (+ adapter)."""
        base = self.name_or_path.rsplit("/", 1)[-1]
        base = base.lower().replace(".", "p").replace("_", "-")
        base = "".join(c if (c.isalnum() or c == "-") else "-" for c in base)
        if self.adapter_path:
            adapter = self.adapter_path.rstrip("/\\").rsplit("/", 1)[-1]
            adapter = adapter.rstrip("\\").rsplit("\\", 1)[-1]
            adapter = "".join(c if (c.isalnum() or c == "-") else "-" for c in adapter.lower())
            if adapter:
                base = f"{base}__{adapter}"
        return base.strip("-") or "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_or_path": self.name_or_path,
            "backend": self.backend,
            "revision": self.revision,
            "adapter_path": self.adapter_path,
            "lora_alias": self.lora_alias,
            "lora_rank": self.lora_rank,
            "dtype": self.dtype,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "extra_args": dict(self.extra_args),
        }


def _should_stub_openai_api_key(hostname: str) -> bool:
    """True local / self-hosted proxies; false for api.openai.com / Azure OAI."""
    h = hostname.strip("[]").lower()
    if h in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return True
    if h == "api.openai.com":
        return False
    if h.endswith(".openai.azure.com"):
        return False
    return True


def _normalize_openai_base_url(url: str) -> str:
    """Normalize OpenAI API root for lmms-eval chat backend.

    Accepts historical values such as ``.../v1/chat/completions`` and rewrites
    them to ``.../v1``.
    """
    u = url.rstrip("/")
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


_TORCH_DTYPES: dict[str, Any] = {}


def _resolve_torch_dtype(dtype: str):
    import torch

    if not _TORCH_DTYPES:
        _TORCH_DTYPES.update(
            {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
                "auto": "auto",
            }
        )
    return _TORCH_DTYPES.get(dtype, dtype)


def load_hf(spec: ModelSpec) -> tuple[Any, Any]:
    """Load an HF causal LM + tokenizer for in-process persona generation.

    Applies a PEFT / LoRA adapter if ``spec.adapter_path`` is set. Returns the
    pair ``(model, tokenizer)``. Heavy imports happen lazily so that the rest
    of ``discord_sft.evals`` remains importable without torch.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Install the evals extra: pip install 'discord-sft[evals]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(
        spec.name_or_path,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.name_or_path,
        revision=spec.revision,
        torch_dtype=_resolve_torch_dtype(spec.dtype),
        trust_remote_code=spec.trust_remote_code,
        device_map=spec.device if spec.device != "cpu" else None,
    )
    if spec.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "Install peft for adapter loading: pip install 'discord-sft[evals]'"
            ) from e
        model = PeftModel.from_pretrained(model, spec.adapter_path)
    model.eval()
    return model, tokenizer
