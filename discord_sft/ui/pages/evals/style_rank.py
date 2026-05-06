"""Style rank tab: pairwise + fingerprint ranking with val-compatible grouping."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from discord_sft.evals.pairwise_style import (
    OpenRouterPairwiseJudge,
    RankStyleConfig,
    _wrap_pairwise_judge,
    run_rank_style_eval,
)
from discord_sft.ui.common import resolve_repo_path, session_work_dir
from discord_sft.ui.pages.evals.paths import evals_root
from discord_sft.ui.pages.evals.run_annotations import (
    annotations_file_path,
    format_eval_run_label,
    get_run_annotation,
    merge_run_annotation_rows,
    save_annotations_file,
    streamlit_annotations,
    streamlit_reload_annotations,
)
from discord_sft.ui.pages.evals.style_rank_elo_store import (
    elo_group_store_path,
    get_group_elo_and_winrate,
    persist_rank_outcome_for_group,
    streamlit_elo_store,
    streamlit_reload_elo_store,
)
from discord_sft.ui.pages.evals.style_rank_groups import (
    PersonaDumpMeta,
    discover_saved_run_dumps,
    group_dumps,
    group_title,
    parse_manual_dump_lines,
)

# OpenRouter slugs for the pairwise style judge (curated list).
STYLE_RANK_JUDGE_MODELS: tuple[str, ...] = (
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-4o",
)


def _default_profiles_path() -> Path | None:
    p = session_work_dir() / "sft" / "profiles.json"
    return p if p.is_file() else None


def _elo_dropdown_order(
    metas: list[PersonaDumpMeta],
    elo_by_run_id: dict[str, float],
) -> list[PersonaDumpMeta]:
    """Highest Elo first; runs without a stored rating sort after (chrono tie-break)."""

    def key(m: PersonaDumpMeta) -> tuple[float, str, str]:
        r = elo_by_run_id.get(m.run_id)
        if r is None:
            return (float("inf"), m.created_utc, m.run_id)
        return (-float(r), m.created_utc, m.run_id)

    return sorted(metas, key=key)


def _normalize_elo_map_from_report(pw: dict[str, Any]) -> dict[str, float]:
    if not isinstance(pw, dict):
        return {}
    elo_raw = pw.get("elo") or {}
    if not isinstance(elo_raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in elo_raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _win_rate_from_pairwise(pw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pw, dict):
        return {}
    wr = pw.get("win_rate") or {}
    return {str(k): v for k, v in wr.items()} if isinstance(wr, dict) else {}


def _elo_and_wr_for_group(gkey: str, *, eval_root: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Session report wins for this group when present; else disk store (survives reload)."""
    last_gkey = st.session_state.get("style_rank_last_group_key")
    last_rep = st.session_state.get("style_rank_last_report")
    if last_gkey == gkey and isinstance(last_rep, dict):
        pw = last_rep.get("pairwise") or {}
        pwd = pw if isinstance(pw, dict) else {}
        return _normalize_elo_map_from_report(pwd), _win_rate_from_pairwise(pwd)

    doc = streamlit_elo_store(eval_root, st.session_state)
    return get_group_elo_and_winrate(doc, gkey)


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    pw = report.get("pairwise") or {}
    elo = pw.get("elo") or {}
    wr = pw.get("win_rate") or {}
    fp = (report.get("fingerprint") or {}).get("per_run") or {}
    comb = (report.get("combined") or {}).get("per_run") or {}
    rows: list[dict[str, Any]] = []
    for run_id, rating in sorted(elo.items(), key=lambda kv: -kv[1]):
        sim = (fp.get(run_id) or {}).get("similarity_to_target")
        rows.append(
            {
                "run": run_id,
                "elo": round(float(rating), 1),
                "win_rate": wr.get(run_id),
                "fingerprint_sim": None if sim is None else round(float(sim), 4),
                "combined": None if comb.get(run_id) is None else round(float(comb[run_id]), 4),
            }
        )
    return rows


def render_style_rank_tab() -> None:
    st.subheader("Style rank")
    st.caption(
        "Compare **two** persona runs head-to-head. Pairwise Elo is **seeded from** "
        f"``{elo_group_store_path(evals_root()).name}`` when present so ratings accumulate over sessions."
    )
    with st.expander("How this works", expanded=False):
        st.markdown(
            "Pick runs that share the **same val set** (grouped automatically). "
            "An LLM judge picks better style per prompt; scores update Elo plus a light fingerprint. "
            "Requires **`OPENROUTER_API_KEY`**."
        )

    root = evals_root()

    manual_metas: list[PersonaDumpMeta] = []
    with st.expander("Manual JSONL paths (optional)", expanded=False):
        st.caption(
            "One ``persona_generations.jsonl`` per line. Optional: matching ``run.json`` paths on the right column "
            "for val grouping when rows lack ``eval_val_*``."
        )
        paths_ta = st.text_area(
            "JSONL paths",
            height=100,
            key="style_rank_manual_jsonl",
            label_visibility="collapsed",
            placeholder="out/evals/raw/<run_id>/persona_generations.jsonl",
        )
        runs_ta = st.text_area(
            "Optional run.json paths (same order)",
            height=80,
            key="style_rank_manual_runjson",
            label_visibility="collapsed",
            placeholder="out/evals/runs/<run_id>.json",
        )
        path_lines = [ln.strip() for ln in paths_ta.splitlines() if ln.strip()]
        run_lines_raw = [ln.strip() for ln in runs_ta.splitlines()]
        if path_lines:
            run_lines_ml: list[str | None] = []
            for i in range(len(path_lines)):
                run_lines_ml.append(
                    run_lines_raw[i] if i < len(run_lines_raw) and run_lines_raw[i] else None
                )
            manual_metas = parse_manual_dump_lines(path_lines, run_lines_ml)

    saved = discover_saved_run_dumps(root)

    by_gens: dict[str, PersonaDumpMeta] = {}
    for m in saved + manual_metas:
        k = str(m.generations_path)
        by_gens[k] = m
    all_dumps = list(by_gens.values())
    groups = group_dumps(all_dumps)

    eligible = {k: v for k, v in groups.items() if len(v) >= 2}
    if not eligible:
        st.info(
            "No **compatible group** with at least two persona generation dumps. "
            "Save eval runs under the configured working directory, or add manual paths."
        )
        return

    options = list(eligible.keys())
    labels_ui = [group_title(k, eligible[k]) for k in options]
    ix = st.selectbox(
        "Val compatibility group",
        range(len(options)),
        format_func=lambda i: labels_ui[i],
        key="style_rank_group_ix",
        help=(
            "Order: embedded ``eval_val_*`` on jsonl rows; else **eval slice fingerprint** "
            "(hash of all ``sample_key`` rows — splits old vs new ``val.jsonl`` when every "
            "``run.json`` still says ``out/sft/val.jsonl``); else ``config.val_jsonl`` only."
        ),
    )
    gkey = options[int(ix)]
    metas = eligible[gkey]
    hint = metas[0].display_hint if metas else ""
    st.caption(f"Val source: {hint}")

    ann_doc = streamlit_annotations(root, st.session_state)

    elo_order, _ = _elo_and_wr_for_group(gkey, eval_root=root)
    metas_for_dropdown = _elo_dropdown_order(metas, elo_order)
    meta_by_rid: dict[str, PersonaDumpMeta] = {m.run_id: m for m in metas}
    option_run_ids = [m.run_id for m in metas_for_dropdown]

    def _run_label(rid: str) -> str:
        m = meta_by_rid.get(rid)
        ylab = m.label if m else None
        return format_eval_run_label(
            rid,
            yaml_label=ylab,
            elo=elo_order.get(rid),
            annotations=ann_doc,
        )

    ranked_for_leader = (
        sorted(elo_order.keys(), key=lambda rid: (-elo_order.get(rid, 0.0), rid))
        if elo_order
        else []
    )
    if ranked_for_leader:
        lead = ranked_for_leader[0]
        lead_el = round(float(elo_order[lead]), 1)
        lead_nm = format_eval_run_label(
            lead,
            yaml_label=meta_by_rid[lead].label,
            elo=None,
            annotations=ann_doc,
            max_run_id_chars=40,
        )
        st.markdown(f"**Leader (stored):** {lead_nm} · Elo **{lead_el}**")
    else:
        st.caption(
            f"No stored Elo for this group yet. After a comparison, ratings are saved to "
            f"``{elo_group_store_path(root).name}``."
        )

    if len(metas) > 2:
        tbl_rows: list[dict[str, Any]] = []
        for rid in option_run_ids:
            e = elo_order.get(rid)
            tbl_rows.append(
                {
                    "run_id": rid,
                    "elo": round(float(e), 1) if e is not None else None,
                    "label": format_eval_run_label(
                        rid,
                        yaml_label=meta_by_rid[rid].label,
                        elo=None,
                        annotations=ann_doc,
                        max_run_id_chars=36,
                    ),
                }
            )
        with st.expander("All runs in this group (stored Elo)", expanded=False):
            st.dataframe(tbl_rows, width="stretch", hide_index=True)

    st.markdown("##### Compare")

    chrono_metas = sorted(metas, key=lambda x: (x.created_utc, x.run_id))
    da = chrono_metas[-2].run_id if len(chrono_metas) >= 2 else option_run_ids[0]
    db = chrono_metas[-1].run_id if len(chrono_metas) >= 2 else option_run_ids[-1]
    ix_a = option_run_ids.index(da) if da in option_run_ids else 0
    ix_b = option_run_ids.index(db) if db in option_run_ids else min(1, len(option_run_ids) - 1)

    run_a = st.selectbox(
        "Run A",
        option_run_ids,
        index=ix_a,
        format_func=_run_label,
        key="style_rank_run_a",
        help=f"Shows stored Elo from ``{elo_group_store_path(root).name}`` when available.",
    )
    options_b = [rid for rid in option_run_ids if rid != run_a]
    ix_b2 = options_b.index(db) if db in options_b else 0
    run_b = st.selectbox(
        "Run B",
        options_b,
        index=min(ix_b2, len(options_b) - 1) if options_b else 0,
        format_func=_run_label,
        key="style_rank_run_b",
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        ea_st = elo_order.get(run_a)
        st.metric(
            "Run A · stored Elo",
            round(float(ea_st), 1) if ea_st is not None else "—",
        )
    with mc2:
        eb_st = elo_order.get(run_b)
        st.metric(
            "Run B · stored Elo",
            round(float(eb_st), 1) if eb_st is not None else "—",
        )

    picked_run_ids = [run_a, run_b]
    run_by_id = meta_by_rid

    with st.expander("More options (backfill, run notes)", expanded=False):
        st.caption(
            "Replay ``style_rank_checkpoints/*.jsonl`` into ``style_rank_group_elo.json``. "
            "Same-group checkpoints are merged. The **same** pairwise matchup (prompt slot + two run ids) "
            "keeps the **newest** file only; different run pairs on the same prompt are all included."
        )
        fresh_store = st.checkbox(
            "Ignore existing elo store for backfill (write only replayed groups)",
            value=False,
            key="style_rank_bfill_fresh_store",
        )
        if st.button("Run backfill from checkpoints", key="style_rank_bfill_ckpt"):
            from discord_sft.evals.style_rank_checkpoint_replay import (
                backfill_elo_store_from_checkpoints_dir,
            )

            summ, store_p = backfill_elo_store_from_checkpoints_dir(
                root,
                merge_existing=not fresh_store,
            )
            streamlit_reload_elo_store(root, st.session_state)
            nw = summ.get("groups_written") or []
            if nw:
                st.success(f"Updated **{len(nw)}** group(s); store: `{store_p}`")
            else:
                st.info(str(summ.get("message") or "No pairwise rows found — nothing written."))

        st.divider()
        st.caption(f"Aliases and notes apply across Evaluate tabs. File: `{annotations_file_path(root)}`")
        edit_rows: list[dict[str, Any]] = []
        for m in sorted(metas, key=lambda x: (x.created_utc, x.run_id)):
            a = get_run_annotation(ann_doc, m.run_id)
            edit_rows.append(
                {
                    "run_id": m.run_id,
                    "alias": str(a.get("alias", "") or ""),
                    "notes": str(a.get("notes", "") or ""),
                }
            )
        edited = st.data_editor(
            edit_rows,
            num_rows="fixed",
            key="style_rank_ann_editor",
            column_config={
                "run_id": st.column_config.TextColumn("run_id", disabled=True, width="large"),
                "alias": st.column_config.TextColumn("Alias", width="medium"),
                "notes": st.column_config.TextColumn("Notes", width="large"),
            },
            hide_index=True,
        )
        if st.button("Save run names & notes", key="style_rank_ann_save"):
            merged = merge_run_annotation_rows(ann_doc, edited)
            save_annotations_file(annotations_file_path(root), merged)
            streamlit_reload_annotations(root, st.session_state)
            st.success("Saved.")
            st.rerun()

    with st.expander("Advanced", expanded=False):
        pairs_pp = st.number_input(
            "Pairwise comparisons per prompt (cap)",
            min_value=1,
            max_value=12,
            value=6,
            help="Each prompt compares random pairs of runs (capped by all unordered pairs).",
            key="style_rank_pairs",
        )
        seed = st.number_input("RNG seed", value=0, step=1, help="Reproducible pair sampling.", key="style_rank_seed")
        judge_model = st.selectbox(
            "Judge model (OpenRouter)",
            options=list(STYLE_RANK_JUDGE_MODELS),
            index=0,
            help="OpenRouter model id for the pairwise A/B judge.",
            key="style_rank_judge_model_sb",
        )
        temperature = st.slider(
            "Judge temperature",
            0.0,
            1.0,
            0.2,
            0.05,
            help="Lower = more deterministic A/B choices.",
            key="style_rank_temp",
        )
        max_conc = st.number_input(
            "Max concurrent judge requests",
            min_value=1,
            max_value=64,
            value=16,
            key="style_rank_conc",
        )
        prof_default = _default_profiles_path()
        profiles_in = st.text_input(
            "profiles.json (optional, for fingerprint fillers)",
            value=str(prof_default) if prof_default else "",
            help="From ``discord-sft fingerprint``; improves filler-rate features.",
            key="style_rank_profiles",
        )
        c1, c2 = st.columns(2)
        with c1:
            pw = st.slider("Combined: pairwise weight", 0.0, 1.0, 0.7, 0.05, key="style_rank_pw")
        with c2:
            fw = st.slider("Combined: fingerprint weight", 0.0, 1.0, 0.3, 0.05, key="style_rank_fw")
        no_ref = st.checkbox(
            "Omit reference style block from judge prompt",
            value=False,
            help="If checked, the judge only sees the user/context prompt, not the gold reply.",
            key="style_rank_no_ref",
        )
        st.checkbox(
            "Compact report (omit per-comparison payloads from JSON / checkpoint)",
            value=False,
            help=(
                "When checked, downloads omit prompt/candidate/judge-response bodies; pairwise Elo unchanged."
            ),
            key="style_rank_compact_report",
        )

    emit_cmp = not bool(st.session_state.get("style_rank_compact_report", False))

    if not os.environ.get("OPENROUTER_API_KEY"):
        st.error("Set **OPENROUTER_API_KEY** in the environment before running the judge.")

    st.checkbox(
        "Preview only — run judge and fingerprints but **do not** save Elo, session leaderboard, or checkpoint JSONL",
        value=False,
        key="style_rank_preview_no_persist",
        help=(
            "Uses the same pairwise pipeline; the report below shows this run’s results. "
            "Your ``style_rank_group_elo.json`` and checkpoint files stay unchanged (no backfill noise)."
        ),
    )
    preview_no_persist = bool(st.session_state.get("style_rank_preview_no_persist", False))

    if st.button("Run comparison", type="primary", key="style_rank_go"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            return
        selected = [run_by_id[r] for r in picked_run_ids if r in run_by_id]
        if len(selected) < 2:
            st.warning("Select **at least two** valid runs.")
            return
        gens_paths = [m.generations_path for m in selected]
        labels = [m.run_id for m in selected]
        rj_paths: list[Path | None] = [m.run_json_path for m in selected]

        prof_path: Path | None = None
        ps = profiles_in.strip()
        if ps:
            p = resolve_repo_path(ps)
            if p.is_file():
                prof_path = p

        ckpt_path: Path | None
        if preview_no_persist:
            ckpt_path = None
        else:
            ckpt_dir = evals_root() / "style_rank_checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            ckpt_path = ckpt_dir / f"{stamp}_{uuid.uuid4().hex[:8]}.jsonl"

        doc_for_prior = streamlit_elo_store(root, st.session_state)
        prior_map, _ = get_group_elo_and_winrate(doc_for_prior, gkey)
        prior_subset: dict[str, float] = {}
        for rid in labels:
            if rid in prior_map:
                try:
                    prior_subset[rid] = float(prior_map[rid])
                except (TypeError, ValueError):
                    pass

        cfg = RankStyleConfig(
            generations_paths=gens_paths,
            labels=labels,
            run_json_paths=rj_paths,
            skip_provenance=True,
            pairs_per_prompt=int(pairs_pp),
            seed=int(seed),
            judge_model=str(judge_model),
            judge_temperature=float(temperature),
            max_concurrency=int(max_conc),
            include_reference_style=not bool(no_ref),
            profiles_path=prof_path,
            emit_comparisons=bool(emit_cmp),
            comparisons_checkpoint_path=ckpt_path,
            pairwise_weight=float(pw),
            fingerprint_weight=float(fw),
            prior_elo_by_label=prior_subset if prior_subset else None,
        )
        try:
            judge = OpenRouterPairwiseJudge(
                cfg.judge_model,
                temperature=cfg.judge_temperature,
            )
            fn = _wrap_pairwise_judge(judge, include_reference_style=cfg.include_reference_style)
            with st.spinner("Running pairwise judge and fingerprints…"):
                report = run_rank_style_eval(cfg, fn)
        except Exception as e:
            st.exception(e)
            try:
                if (
                    ckpt_path is not None
                    and ckpt_path.is_file()
                    and ckpt_path.stat().st_size > 0
                ):
                    st.warning(
                        f"Some pairwise rows were written before the failure: `{ckpt_path}`"
                    )
            except OSError:
                pass
            return

        if not preview_no_persist:
            st.session_state["style_rank_last_report"] = report
            st.session_state["style_rank_last_group_key"] = gkey
            store_path = persist_rank_outcome_for_group(
                root,
                st.session_state,
                gkey,
                report,
                ranked_run_ids=labels,
            )
        else:
            store_path = elo_group_store_path(root)

        rows = _summary_rows(report)
        st.success("Done (preview)." if preview_no_persist else "Done.")
        pwrep = report.get("pairwise") or {}
        seeded = bool(prior_subset)
        seed_note = (
            " Elo in the report was **seeded** from stored ratings for this simulation."
            if seeded
            else ""
        )
        if preview_no_persist:
            st.caption(
                "**Preview:** no changes to saved Elo, leaderboard cache, or ``style_rank_checkpoints/``. "
                f"Download the JSON below if you want a snapshot.{seed_note}"
            )
        else:
            st.caption(
                f"Pairwise checkpoint (JSONL, ok + error rows): `{ckpt_path}` · "
                f"Elo for this group persisted to `{store_path}`.{seed_note}"
            )
        n_err = int(pwrep.get("n_comparison_errors") or 0)
        if n_err:
            st.warning(
                f"{n_err} judge call(s) failed; Elo uses successful calls only. "
                f"Details in `report.pairwise.errors` and in the checkpoint file."
            )
        st.dataframe(rows, width="stretch", hide_index=True)
        for w in report.get("merge_warnings") or []:
            st.warning(w)
        comps = report.get("comparisons") or []
        if isinstance(comps, list) and comps:
            with st.expander("Comparison log", expanded=True):
                log_rows: list[dict[str, Any]] = []
                for c in comps:
                    if not isinstance(c, dict):
                        continue
                    log_rows.append(
                        {
                            "sample_key": str(c.get("sample_key", ""))[:20] + "…"
                            if len(str(c.get("sample_key", ""))) > 20
                            else c.get("sample_key"),
                            "winner": c.get("winner_label"),
                            "loser": c.get("loser_label"),
                            "choice": c.get("choice"),
                        }
                    )
                try:
                    import pandas as pd
                except ImportError:
                    pd = None
                if pd is not None and log_rows:
                    st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)
                elif log_rows:
                    st.json(log_rows)
                st.caption("Full prompts, outputs, and judge text are in the downloaded JSON.")
        diag = report.get("diagnostics") or {}
        pr = diag.get("per_run") or {}
        if pr:
            elo_pw = (report.get("pairwise") or {}).get("elo") or {}
            with st.expander("Diagnostics (fingerprints quadrants)", expanded=False):
                st.caption(
                    "Per run vs median Elo and fingerprint. With two runs, medians match the better value on each axis."
                )
                ordered = sorted(
                    pr.items(),
                    key=lambda kv: -float(elo_pw.get(kv[0], 0.0)),
                )
                st.json({k: v.get("quadrant") for k, v in ordered})

        payload = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            "Download full report (JSON)",
            data=payload.encode("utf-8"),
            file_name="style_rank_report.json",
            mime="application/json",
            key="style_rank_dl",
        )
