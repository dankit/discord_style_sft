"""Registry of MVP benchmarks.

Each entry maps a short key (what we use in our own CLI / JSON) to:

- ``task``: the lmms-eval task name to pass to ``--tasks`` on its CLI.
- ``metric``: the primary scalar metric we surface in our unified run JSON
  (lmms-eval may report several; we pick one canonical number per benchmark
  for the comparison grid).
- ``modality``: ``text`` or ``image`` — purely informational, used by the UI
  to warn if you request a vision benchmark with a text-only backend.
- ``description``: one-line human summary used in CLI help, UI tooltips,
  and the README.

Task-name verification: as of lmms-eval 0.7 these are registered:
``ifeval``, ``mmmu_val`` (``mmmu_test`` is submission-only so we skip it),
``mmstar``, ``screenspot_v2``. Re-verify with
``python -m lmms_eval --tasks list`` if you upgrade lmms-eval.

The ``persona`` key is a special pseudo-benchmark handled by
:mod:`discord_sft.evals.persona` (no lmms-eval task behind it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Modality = Literal["text", "image", "video", "audio"]


@dataclass(frozen=True)
class BenchmarkSpec:
    task: str
    metric: str
    modality: Modality
    description: str
    category: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "ifeval": BenchmarkSpec(
        task="ifeval",
        metric="prompt_level_strict_acc",
        modality="text",
        category="instruction-following",
        description=(
            "Instruction following on verifiable constraints "
            "(e.g. 'respond in exactly 3 bullets', 'include the word X'). "
            "The single biggest regression risk for persona SFT, since "
            "chat-style Discord data rarely contains strict formatting "
            "directives."
        ),
    ),
    "mmmu_val": BenchmarkSpec(
        task="mmmu_val",
        metric="mmmu_acc",
        modality="image",
        category="multimodal-reasoning",
        description=(
            "Multi-discipline college-level VQA across 30 subjects "
            "(MMMU validation split). VLM analogue of MMLU but harder: "
            "answers require interpreting the image (charts, diagrams, "
            "medical / scientific figures), not just reading the prompt."
        ),
    ),
    "mmstar": BenchmarkSpec(
        task="mmstar",
        metric="average",
        modality="image",
        category="vqa",
        description=(
            "Visual QA curated to filter out questions a text-only model "
            "could solve from the prompt alone. Score reflects real "
            "perception + reasoning rather than language priors."
        ),
    ),
    "screenspot_v2": BenchmarkSpec(
        task="screenspot_v2",
        metric="accuracy",
        modality="image",
        category="gui-grounding",
        description=(
            "GUI grounding: given a UI screenshot and a natural-language "
            "target ('click the login button'), localize the element. "
            "The core low-level skill any web-browsing agent needs; also "
            "directly relevant to interacting with CAPTCHA challenges."
        ),
    ),
    "persona": BenchmarkSpec(
        task="__persona__",
        metric="overall",
        modality="text",
        category="persona-fit",
        description=(
            "Native persona evals (not an lmms-eval task): "
            "style_heuristics / profile_heuristics on generated replies "
            "to held-out val.jsonl, plus optional OpenRouter rubric LLM-as-judge "
            "(reasoning-first JSON; vocabulary / tone / length / authentic persona)."
        ),
    ),
}


PERSONA_KEY = "persona"
DEFAULT_TASKS = ["ifeval", "mmmu_val", "mmstar", "screenspot_v2", "persona"]


def split_tasks(tasks: list[str]) -> tuple[list[str], bool]:
    """Split a user-supplied task list into lmms-eval tasks + persona flag.

    Returns ``(lmms_tasks, include_persona)``. Raises ``KeyError`` on any
    unknown key, so CLI typos surface immediately with the full list of
    known benchmarks in the message.
    """
    lmms: list[str] = []
    include_persona = False
    for t in tasks:
        if t not in BENCHMARKS:
            raise KeyError(
                f"Unknown benchmark '{t}'. Known: {sorted(BENCHMARKS)}"
            )
        if t == PERSONA_KEY:
            include_persona = True
        else:
            lmms.append(BENCHMARKS[t].task)
    return lmms, include_persona


def describe(keys: list[str] | None = None) -> str:
    """Render a bullet list of benchmark descriptions (for CLI / README)."""
    picks = keys or list(BENCHMARKS.keys())
    lines = []
    for k in picks:
        if k not in BENCHMARKS:
            continue
        b = BENCHMARKS[k]
        lines.append(f"- {k} [{b.modality}, {b.category}]: {b.description}")
    return "\n".join(lines)
