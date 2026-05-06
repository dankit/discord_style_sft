"""Sparse pairwise LLM style judging + Elo + stylometric fingerprints across runs."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import random
import zlib
import re
import threading
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from discord_sft.analysis.stylometry import (
    compute_fingerprint_features,
    cosine_similarity,
    features_to_vector,
    union_mined_fillers,
)
from discord_sft.ui.persona_compare import MergedPersonaSample, merge_generation_sources


def _stable_u64(*parts: object) -> int:
    """Deterministic 64-bit mixer for RNG seeds (not Python's per-process hash())."""

    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "little")


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_val_jsonl_from_run_json(path: Path) -> str | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cfg = doc.get("config")
    if not isinstance(cfg, dict):
        return None
    v = cfg.get("val_jsonl")
    return str(v).strip() if v else None


def check_val_provenance(
    run_json_paths: list[Path],
    *,
    use_hash: bool = False,
) -> dict[str, Any]:
    """Ensure all runs used the same eval val.jsonl (path or content hash)."""
    paths_resolved: list[str] = []
    hashes: list[str | None] = []
    for p in run_json_paths:
        vj = read_val_jsonl_from_run_json(p)
        if not vj:
            raise ValueError(f"No config.val_jsonl in run JSON: {p}")
        rp = Path(vj).expanduser()
        try:
            paths_resolved.append(str(rp.resolve(strict=False)))
        except OSError:
            paths_resolved.append(str(rp))
        if use_hash:
            h = file_sha256(rp)
            if h is None:
                raise ValueError(f"val_jsonl not a readable file for hash: {vj} (from {p})")
            hashes.append(h)

    if use_hash:
        first_h = hashes[0]
        for p, h in zip(run_json_paths, hashes, strict=True):
            if h != first_h:
                raise ValueError(
                    "Eval val.jsonl content hash mismatch between runs: "
                    f"{p} sha256={h[:16]}… vs expected {first_h[:16]}…"
                )
        return {
            "mode": "sha256",
            "val_sha256": first_h,
            "val_jsonl_paths": paths_resolved,
        }

    first = paths_resolved[0]
    for p, rp in zip(run_json_paths, paths_resolved, strict=True):
        if rp != first:
            raise ValueError(
                "Eval val_jsonl path mismatch after resolve: "
                f"{p} -> {rp!r} vs first {first!r}"
            )
    return {"mode": "path", "val_jsonl": first, "val_jsonl_paths": paths_resolved}


def check_val_provenance_from_rows(
    rows_meta: list[dict[str, Any] | None],
    *,
    use_hash: bool = False,
) -> dict[str, Any]:
    """Use eval_val_jsonl / eval_val_sha256 embedded in generation rows (first row per file)."""
    n = len(rows_meta)
    jsonl_paths: list[str | None] = []
    hashes: list[str | None] = []
    for meta in rows_meta:
        if not meta:
            jsonl_paths.append(None)
            hashes.append(None)
            continue
        jsonl_paths.append(str(meta.get("eval_val_jsonl") or "").strip() or None)
        h = meta.get("eval_val_sha256")
        hashes.append(str(h).strip() if h else None)

    if use_hash:
        if any(h is None for h in hashes):
            raise ValueError(
                "Provenance via hash requires eval_val_sha256 on every generations file "
                "(re-run eval with updated discord-sft or pass --run-json)."
            )
        first_h = hashes[0]
        assert first_h is not None
        if any(h != first_h for h in hashes):
            raise ValueError(f"eval_val_sha256 mismatch across inputs: {hashes!r}")
        return {"mode": "sha256", "val_sha256": first_h, "source": "jsonl_rows"}

    if any(p is None for p in jsonl_paths):
        raise ValueError(
            "Provenance via path requires eval_val_jsonl on every generations file "
            "(re-run eval with updated discord-sft or pass --run-json)."
        )
    try:
        resolved = [str(Path(p).expanduser().resolve(strict=False)) for p in jsonl_paths]
    except OSError:
        resolved = [str(Path(p).expanduser()) for p in jsonl_paths]
    if len(resolved) != n:
        raise ValueError("eval_val_jsonl: missing path on at least one row")
    first = resolved[0]
    if any(r != first for r in resolved):
        raise ValueError(f"eval_val_jsonl mismatch after resolve: {resolved!r}")
    return {"mode": "path", "val_jsonl": first, "source": "jsonl_rows"}


def _user_prompt_from_row(base_row: dict[str, Any]) -> str:
    ctx = str(base_row.get("context") or "").strip()
    if ctx:
        return ctx
    turns = base_row.get("context_turns")
    if isinstance(turns, list):
        lines = []
        for turn in turns:
            if isinstance(turn, dict):
                role = turn.get("from", "user")
                lines.append(f"[{role}] {turn.get('value', '')}")
        return "\n".join(lines)
    return ""


def build_pairwise_prompt(
    *,
    user_prompt: str,
    reference_style: str | None,
    output_a: str,
    output_b: str,
    include_reference_style: bool,
) -> str:
    ref_block = ""
    if include_reference_style and (reference_style or "").strip():
        ref_block = (
            "\nReference style example (optional — match style and voice only, "
            "not factual content; ignore factual differences vs outputs):\n"
            f"{reference_style.strip()}\n"
        )
    return f"""# Preference-based style judge (LLM-as-a-judge)

## Objective
Pick which of **Output A** or **Output B** better matches the **target writing style** implied by the user prompt and any reference style snippet—**voice, tone, phrasing, rhythm, and casual authenticity**. Prefer outputs that feel like a real person in a Discord-style chat when that is what the reference suggests.

## What to compare
- **Tone** (energy, attitude, sarcasm vs sincerity)
- **Phrasing** (slang, fillers, contractions, micro-style)
- **Personality / voice** (consistent character vs generic assistant voice)
- **Rhythm** (short bursts vs walls of text, pacing)

Ignore factual correctness unless it breaks style (e.g. lecturing or hedging that clashes with the voice). The topical answer can differ from the reference style example.

## How to respond
1. Write a **short reasoning** paragraph (a few sentences): cite concrete style cues from A vs B.
2. End with **exactly one** final line of the form: `Answer: A` or `Answer: B` (nothing after that on the line).

User prompt:
{user_prompt}
{ref_block}
Output A:
{output_a}

Output B:
{output_b}
"""


def parse_pairwise_answer(text: str) -> str | None:
    raw = text.strip()
    if not raw:
        return None
    for line in reversed(raw.splitlines()):
        s = line.strip()
        low = s.lower()
        if low.startswith("answer:"):
            tail = low.split(":", 1)[1].strip()
            if tail in ("a", "b"):
                return tail.upper()
    for line in raw.splitlines():
        s = line.strip()
        low = s.lower()
        if low in ("a", "b"):
            return s[0].upper()
        if low.startswith("answer:"):
            tail = low.split(":", 1)[1].strip()
            if tail in ("a", "b"):
                return tail.upper()
    low = raw.lower()
    m = re.search(r"\b([ab])\b", low, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if low.startswith("a") and not low.startswith("assistant"):
        return "A"
    if low.startswith("b"):
        return "B"
    return None


def expected_elo_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_elo_pair(
    r_winner: float,
    r_loser: float,
    *,
    k: float = 32.0,
) -> tuple[float, float]:
    """Winner gets score 1, loser 0."""
    e_w = expected_elo_score(r_winner, r_loser)
    e_l = expected_elo_score(r_loser, r_winner)
    new_w = r_winner + k * (1.0 - e_w)
    new_l = r_loser + k * (0.0 - e_l)
    return new_w, new_l


_TWO_PLAYER_SYM_PRIOR_TOLERANCE = 8.0  # starters within this ⇒ treat as "fresh" pairwise ladder


def _two_player_near_symmetric_priors(pair: tuple[str, str], starters: dict[str, float]) -> bool:
    a, b = pair
    return abs(float(starters[a]) - float(starters[b])) <= _TWO_PLAYER_SYM_PRIOR_TOLERANCE


def two_player_elo_ratings_from_win_counts(
    labels: tuple[str, str],
    wins: Mapping[str, int],
    *,
    elo_start: float,
    prior_by_label: dict[str, float] | None,
) -> dict[str, float]:
    """Head-to-head Elo aligned with empirical win fraction (Bradley-Terry inversion at 400 scale).

    For exactly two opponents, incremental updates in arbitrary row order can rank the loser
    above the winner despite priors matching; this maps batch win-rate to an Elo gap monotonically.

    Ratings are anchored on the midpoint of the two starters (``elo_start`` or prior).
    """
    a, b = labels[0], labels[1]
    wa_int = int(wins.get(a, 0))
    wb_int = int(wins.get(b, 0))
    n = wa_int + wb_int
    if n == 0:
        return _initial_elo_ratings_for_labels(list(labels), elo_start, prior_by_label)
    starters = _initial_elo_ratings_for_labels(list(labels), elo_start, prior_by_label)
    ra0, rb0 = float(starters[a]), float(starters[b])
    center = 0.5 * (ra0 + rb0)
    pa = wa_int / float(n)
    eps = max(1.0 / (2 * n), 1e-9)
    pa_c = max(min(pa, 1.0 - eps), eps)
    magnitude = 400.0 * math.log10(pa_c / (1.0 - pa_c))
    return {a: center + magnitude / 2.0, b: center - magnitude / 2.0}


def shuffle_comparison_rows_for_incremental_elo(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    labels: tuple[str, ...],
) -> None:
    """Shuffle in place so incremental Elo updates are not biased by row sort order.

    Applying :func:`update_elo_pair` in a fixed order (e.g. lexicographic ``sample_key``) can
    leave the higher-rated player as the one who *lost* more games, even from 1500/1500 starts.
    A deterministic shuffle keyed by ``seed`` and run labels matches a reproducible random match
    order while keeping win/loss counts unchanged.
    """
    if len(rows) < 2:
        return
    tag = "\0".join(labels).encode("utf-8")
    mix = int(zlib.adler32(tag)) & 0xFFFFFFFF
    rng_seed = (int(seed) ^ mix ^ (len(rows) * 0x9E3779B9)) & 0xFFFFFFFFFFFFFFFF
    random.Random(rng_seed).shuffle(rows)


def sample_pairs(k_variants: int, n_pairs: int, rng: random.Random) -> list[tuple[int, int]]:
    if k_variants < 2:
        return []
    all_pairs = list(combinations(range(k_variants), 2))
    n = min(max(1, n_pairs), len(all_pairs))
    return rng.sample(all_pairs, n)


@runtime_checkable
class PairwiseJudgeFn(Protocol):
    """Return ``\"A\"|\"B\"`` or ``(choice, raw_model_text)`` for logging."""

    def __call__(
        self,
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> str | tuple[str, str | None]: ...


class OpenRouterPairwiseJudge:
    """A/B style judge via OpenRouter (same env vars as OpenRouterJudge)."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 384,
        max_retries: int = 5,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install optional dependency: pip install 'discord-sft[evals]' "
                "(includes openai for OpenRouter)."
            ) from e
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY to use OpenRouterPairwiseJudge.")
        base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        headers: dict[str, str] = {}
        if referer := os.environ.get("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = referer
        headers["X-Title"] = os.environ.get("OPENROUTER_APP_TITLE", "discord-sft")
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._max_retries = int(max_retries)
        url = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self._client = OpenAI(base_url=url, api_key=key, default_headers=headers)

    def compare(
        self,
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
        include_reference_style: bool,
    ) -> tuple[str, str]:
        prompt = build_pairwise_prompt(
            user_prompt=user_prompt,
            reference_style=reference_style,
            output_a=output_a,
            output_b=output_b,
            include_reference_style=include_reference_style,
        )
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = resp.choices[0].message.content
                if not content:
                    raise RuntimeError("empty judge content")
                raw_text = content.strip()
                parsed = parse_pairwise_answer(content)
                if parsed in ("A", "B"):
                    return parsed, raw_text
                last = RuntimeError(f"unparseable judge output: {content!r}")
            except Exception as e:
                last = e
        assert last is not None
        raise last


def _wrap_pairwise_judge(
    judge: OpenRouterPairwiseJudge,
    *,
    include_reference_style: bool,
) -> PairwiseJudgeFn:
    def _fn(
        *,
        user_prompt: str,
        reference_style: str | None,
        output_a: str,
        output_b: str,
    ) -> tuple[str, str]:
        return judge.compare(
            user_prompt=user_prompt,
            reference_style=reference_style,
            output_a=output_a,
            output_b=output_b,
            include_reference_style=include_reference_style,
        )

    return _fn


@dataclass
class RankStyleConfig:
    generations_paths: list[Path]
    labels: list[str]
    run_json_paths: list[Path | None]
    skip_provenance: bool = False
    provenance_from_rows: bool = False
    use_val_hash: bool = False
    pairs_per_prompt: int = 6
    seed: int = 0
    judge_model: str = "google/gemini-3-flash-preview"
    judge_temperature: float = 0.2
    max_concurrency: int = 16
    include_reference_style: bool = True
    profiles_path: Path | None = None
    emit_comparisons: bool = False
    # Append one JSON object per line (ok + error rows) as comparisons finish.
    comparisons_checkpoint_path: Path | None = None
    pairwise_weight: float = 0.7
    fingerprint_weight: float = 0.3
    elo_k: float = 32.0
    elo_start: float = 1500.0
    # Seed pairwise Elo from persisted ratings (e.g. style_rank_group_elo.json) before this run.
    prior_elo_by_label: dict[str, float] | None = None


def _initial_elo_ratings_for_labels(
    labels: list[str],
    elo_start: float,
    prior: dict[str, float] | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lab in labels:
        if prior and lab in prior:
            try:
                v = float(prior[lab])
                out[lab] = v if math.isfinite(v) else float(elo_start)
            except (TypeError, ValueError):
                out[lab] = float(elo_start)
        else:
            out[lab] = float(elo_start)
    return out


def _first_row_meta(path: Path) -> dict[str, Any] | None:
    rows = load_jsonl_objects(path)
    return rows[0] if rows else None


def _append_jsonl_checkpoint(
    path: Path, lock: threading.Lock, obj: dict[str, Any]
) -> None:
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def run_rank_style_eval(
    cfg: RankStyleConfig,
    judge_fn: PairwiseJudgeFn | None,
) -> dict[str, Any]:
    """Merge generation dumps, optional provenance, pairwise judging, Elo, fingerprints."""
    if len(cfg.generations_paths) < 2:
        raise ValueError("rank-style needs at least two --generations files")
    if len(cfg.labels) != len(cfg.generations_paths):
        raise ValueError("labels length must match generations_paths")
    if len(cfg.run_json_paths) != len(cfg.generations_paths):
        raise ValueError("run_json_paths length must match generations_paths")

    prov: dict[str, Any] | None = None
    if not cfg.skip_provenance:
        if cfg.provenance_from_rows:
            metas = [_first_row_meta(p) for p in cfg.generations_paths]
            prov = check_val_provenance_from_rows(metas, use_hash=cfg.use_val_hash)
        else:
            rjs = [p for p in cfg.run_json_paths if p is not None]
            if len(rjs) != len(cfg.generations_paths):
                raise ValueError(
                    "Provenance: pass one --run-json per --generations, "
                    "or use --skip-provenance-check, "
                    "or embed eval metadata in jsonl (--provenance-from-rows)."
                )
            prov = check_val_provenance(rjs, use_hash=cfg.use_val_hash)
    else:
        prov = {"skipped": True}

    indexed: list[tuple[Path, list[dict[str, Any]], dict[str, Any] | None, str, int]] = []
    for i, (gp, label) in enumerate(zip(cfg.generations_paths, cfg.labels, strict=True)):
        rows = load_jsonl_objects(gp)
        run_doc: dict[str, Any] | None = None
        rjp = cfg.run_json_paths[i]
        if rjp is not None and rjp.is_file():
            try:
                run_doc = json.loads(rjp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run_doc = None
        run_doc = dict(run_doc or {})
        run_doc["run_id"] = label
        indexed.append((gp.resolve(), rows, run_doc, label, i))

    merged, merge_warnings = merge_generation_sources(indexed)
    if not merged:
        raise ValueError(
            "No overlapping samples between generation files (inner join is empty). "
            "Check that all dumps share the same val.jsonl slice and sample keys."
        )

    persona_ids: set[str] = set()
    for m in merged:
        pid = m.base_row.get("persona_id")
        if pid is not None:
            persona_ids.add(str(pid))

    fillers: list[str] | None = None
    if cfg.profiles_path and cfg.profiles_path.is_file():
        prof_doc = json.loads(cfg.profiles_path.read_text(encoding="utf-8"))
        fillers = union_mined_fillers(prof_doc, persona_ids)

    target_blob = "\n\n".join(
        str(m.base_row.get("reference") or "").strip() for m in merged if m.base_row
    )
    target_features = compute_fingerprint_features(target_blob, fillers=fillers)
    target_vec = features_to_vector(target_features)

    k_variants = len(merged[0].variants) if merged else 0
    fp_per_run: dict[str, dict[str, Any]] = {}
    for vi in range(k_variants):
        label = cfg.labels[vi]
        blob = "\n\n".join(m.variants[vi].generated for m in merged)
        feats = compute_fingerprint_features(blob, fillers=fillers)
        vec = features_to_vector(feats)
        fp_per_run[label] = {
            "features": feats,
            "similarity_to_target": cosine_similarity(vec, target_vec),
        }

    ratings: dict[str, float] = _initial_elo_ratings_for_labels(
        cfg.labels,
        cfg.elo_start,
        cfg.prior_elo_by_label,
    )
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    games: dict[str, int] = defaultdict(int)
    comparisons_out: list[dict[str, Any]] = []

    if judge_fn is None:
        pairwise_block = {
            "skipped": True,
            "reason": "no judge_fn (tests or dry)",
        }
        report = {
            "merge_warnings": merge_warnings,
            "n_merged_samples": len(merged),
            "k_variants": k_variants,
            "provenance": prov,
            "pairwise": pairwise_block,
            "fingerprint": {
                "target_features": target_features,
                "per_run": fp_per_run,
                "fillers_source": str(cfg.profiles_path) if cfg.profiles_path else "default",
            },
        }
        report["combined"] = _combined_scores(
            ratings, fp_per_run, cfg, skip_pairwise=True
        )
        report["diagnostics"] = _diagnostics(ratings, fp_per_run, cfg.labels)
        if cfg.emit_comparisons:
            report["comparisons"] = []
        return report

    jobs: list[tuple[MergedPersonaSample, int, int, bool]] = []
    for m in merged:
        row_rng = random.Random(
            (_stable_u64(m.sample_key) ^ int(cfg.seed)) & 0xFFFFFFFFFFFFFFFF
        )
        pairs = sample_pairs(k_variants, cfg.pairs_per_prompt, row_rng)
        for i, j in pairs:
            swap_seed = (_stable_u64(m.sample_key, i, j) ^ int(cfg.seed)) & 0xFFFFFFFFFFFFFFFF
            swap = random.Random(swap_seed).random() < 0.5
            jobs.append((m, i, j, swap))

    def _one(job: tuple[MergedPersonaSample, int, int, bool]) -> dict[str, Any]:
        m, i, j, swap = job
        vi_a, vi_b = (j, i) if swap else (i, j)
        text_a = m.variants[vi_a].generated
        text_b = m.variants[vi_b].generated
        ref = str(m.base_row.get("reference") or "").strip() or None
        up = _user_prompt_from_row(m.base_row)
        j_ret = judge_fn(
            user_prompt=up,
            reference_style=ref,
            output_a=text_a,
            output_b=text_b,
        )
        judge_response: str | None = None
        if isinstance(j_ret, tuple):
            choice, judge_response = j_ret[0], j_ret[1]
        else:
            choice = j_ret
        if choice not in ("A", "B"):
            raise RuntimeError(f"judge_fn returned invalid choice: {choice!r}")
        winner_vi = vi_a if choice == "A" else vi_b
        loser_vi = vi_b if choice == "A" else vi_a
        winner_label = cfg.labels[winner_vi]
        loser_label = cfg.labels[loser_vi]
        out: dict[str, Any] = {
            "sample_key": m.sample_key,
            "i": i,
            "j": j,
            "swap": swap,
            "choice": choice,
            "winner_label": winner_label,
            "loser_label": loser_label,
            "parse_error": False,
        }
        # Full judge payload (optional): Elo path only needs labels + choice above.
        if cfg.emit_comparisons:
            out["user_prompt"] = up
            out["reference_style"] = ref
            out["output_a"] = text_a
            out["output_b"] = text_b
            out["label_a"] = cfg.labels[vi_a]
            out["label_b"] = cfg.labels[vi_b]
            if judge_response is not None:
                out["judge_response"] = judge_response
        return out

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ckpt_lock = threading.Lock()
    ckpt_path = cfg.comparisons_checkpoint_path
    workers = max(1, min(cfg.max_concurrency, len(jobs) or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {pool.submit(_one, job): job for job in jobs}
        for fut in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[fut]
            m, i, j, swap = job
            try:
                r = fut.result()
                results.append(r)
                if ckpt_path is not None:
                    _append_jsonl_checkpoint(ckpt_path, ckpt_lock, dict(r))
            except Exception as e:
                err: dict[str, Any] = {
                    "checkpoint_kind": "error",
                    "sample_key": m.sample_key,
                    "i": i,
                    "j": j,
                    "swap": swap,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                errors.append(err)
                if ckpt_path is not None:
                    _append_jsonl_checkpoint(ckpt_path, ckpt_lock, err)

    results.sort(key=lambda r: (r["sample_key"], r["i"], r["j"]))

    merge_warnings_out = list(merge_warnings)
    if errors:
        hint = (
            f" Checkpoint: {ckpt_path}"
            if ckpt_path is not None
            else " Enable comparisons checkpoint to retain every row on disk."
        )
        merge_warnings_out.append(
            f"{len(errors)} pairwise judge call(s) failed; "
            f"Elo uses {len(results)} successful comparison(s).{hint}"
        )

    for r in results:
        w = str(r["winner_label"])
        l = str(r["loser_label"])
        wins[w] += 1
        losses[l] += 1
        games[w] += 1
        games[l] += 1
        if cfg.emit_comparisons:
            comparisons_out.append(dict(r))

    if k_variants == 2:
        pair = (str(cfg.labels[0]), str(cfg.labels[1]))
        starters_two = _initial_elo_ratings_for_labels(
            list(pair), float(cfg.elo_start), cfg.prior_elo_by_label
        )
        if _two_player_near_symmetric_priors(pair, starters_two):
            ratings = two_player_elo_ratings_from_win_counts(
                pair,
                wins,
                elo_start=float(cfg.elo_start),
                prior_by_label=cfg.prior_elo_by_label,
            )
        else:
            ratings = dict(starters_two)
            elo_rows = list(results)
            shuffle_comparison_rows_for_incremental_elo(
                elo_rows, seed=int(cfg.seed), labels=tuple(cfg.labels)
            )
            for r in elo_rows:
                w = str(r["winner_label"])
                l = str(r["loser_label"])
                rw, rl = ratings[w], ratings[l]
                nw, nl = update_elo_pair(rw, rl, k=cfg.elo_k)
                ratings[w] = nw
                ratings[l] = nl
    else:
        elo_rows = list(results)
        shuffle_comparison_rows_for_incremental_elo(
            elo_rows, seed=int(cfg.seed), labels=tuple(cfg.labels)
        )
        for r in elo_rows:
            w = str(r["winner_label"])
            l = str(r["loser_label"])
            rw, rl = ratings[w], ratings[l]
            nw, nl = update_elo_pair(rw, rl, k=cfg.elo_k)
            ratings[w] = nw
            ratings[l] = nl
    winrate = {}
    for lab in cfg.labels:
        g = games[lab]
        winrate[lab] = (wins[lab] / g) if g else None

    err_cap = 80
    pairwise_block: dict[str, Any] = {
        "elo": dict(sorted(ratings.items(), key=lambda kv: -kv[1])),
        "win_rate": winrate,
        "wins": dict(wins),
        "losses": dict(losses),
        "games": dict(games),
        "n_comparisons": len(results),
        "n_comparison_errors": len(errors),
        "pairs_per_prompt_cap": cfg.pairs_per_prompt,
        "seed": cfg.seed,
        "judge_model": cfg.judge_model,
        "temperature": cfg.judge_temperature,
        "comparisons_checkpoint": str(ckpt_path) if ckpt_path is not None else None,
    }
    if errors:
        pairwise_block["errors"] = errors[:err_cap]
        pairwise_block["errors_truncated"] = len(errors) > err_cap
        pairwise_block["partial"] = bool(results)

    report: dict[str, Any] = {
        "merge_warnings": merge_warnings_out,
        "n_merged_samples": len(merged),
        "k_variants": k_variants,
        "provenance": prov,
        "pairwise": pairwise_block,
        "fingerprint": {
            "target_features": target_features,
            "per_run": fp_per_run,
            "fillers_source": str(cfg.profiles_path) if cfg.profiles_path else "default",
        },
    }
    report["combined"] = _combined_scores(ratings, fp_per_run, cfg, skip_pairwise=False)
    report["diagnostics"] = _diagnostics(ratings, fp_per_run, cfg.labels)
    if cfg.emit_comparisons:
        report["comparisons"] = comparisons_out
    return report


def _norm_pairwise_scores(ratings: dict[str, float]) -> dict[str, float]:
    vals = list(ratings.values())
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 1e-9:
        return {k: 0.5 for k in ratings}
    return {k: (v - lo) / span for k, v in ratings.items()}


def _combined_scores(
    ratings: dict[str, float],
    fp_per_run: dict[str, dict[str, Any]],
    cfg: RankStyleConfig,
    *,
    skip_pairwise: bool,
) -> dict[str, Any]:
    sims = {k: float(v.get("similarity_to_target", 0.0)) for k, v in fp_per_run.items()}
    lo, hi = min(sims.values(), default=0.0), max(sims.values(), default=1.0)
    span = hi - lo
    fp_n = {
        k: (sims[k] - lo) / span if span > 1e-9 else 0.5 for k in cfg.labels
    }
    if skip_pairwise:
        return {
            "weights": {"pairwise": 0.0, "fingerprint": 1.0},
            "per_run": dict(fp_n),
            "normalized_pairwise": None,
            "normalized_fingerprint": fp_n,
        }
    pw = _norm_pairwise_scores(ratings)
    combined = {}
    for lab in cfg.labels:
        combined[lab] = (
            cfg.pairwise_weight * pw.get(lab, 0.5)
            + cfg.fingerprint_weight * fp_n.get(lab, 0.5)
        )
    return {
        "weights": {"pairwise": cfg.pairwise_weight, "fingerprint": cfg.fingerprint_weight},
        "per_run": combined,
        "normalized_pairwise": pw,
        "normalized_fingerprint": fp_n,
    }


def _diagnostics(
    ratings: dict[str, float],
    fp_per_run: dict[str, dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    elos = [ratings[l] for l in labels]
    sims = [float(fp_per_run[l]["similarity_to_target"]) for l in labels]
    med_e = sorted(elos)[len(elos) // 2] if elos else 0.0
    med_s = sorted(sims)[len(sims) // 2] if sims else 0.0
    per_run: dict[str, dict[str, Any]] = {}
    for lab in labels:
        hi_e = ratings[lab] >= med_e
        hi_s = float(fp_per_run[lab]["similarity_to_target"]) >= med_s
        quadrant = "low_both"
        if hi_s and hi_e:
            quadrant = "high_fingerprint_high_elo"
        elif hi_s and not hi_e:
            quadrant = "high_fingerprint_low_elo"
        elif not hi_s and hi_e:
            quadrant = "low_fingerprint_high_elo"
        per_run[lab] = {
            "high_elo_vs_median": hi_e,
            "high_fingerprint_vs_median": hi_s,
            "quadrant": quadrant,
        }
    return {
        "median_elo": med_e,
        "median_fingerprint_similarity": med_s,
        "per_run": per_run,
        "quadrant_meaning": {
            "high_fingerprint_high_elo": "strong surface + judge signal",
            "high_fingerprint_low_elo": "possible shallow mimicry / overfitting",
            "low_fingerprint_high_elo": "judge prefers style beyond surface stats",
            "low_both": "weak on both signals",
        },
    }

