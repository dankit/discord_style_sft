"""Browse persona_generations.jsonl and compare runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from discord_sft.analysis.fingerprint import aggregate_profile_drift_for_eval_rows
from discord_sft.analysis.heuristics import profile_heuristics, style_heuristics
from discord_sft.evals.storage import list_runs, load_run
from discord_sft.ui.common import existing_file, resolve_repo_path, session_work_path
from discord_sft.ui.persona_compare import (
    badge_label,
    comparison_id_from_paths,
    comparison_variant_metadata,
    comparison_verdict_map,
    load_vote_store,
    merge_generation_sources,
    ranked_variant_order,
    reconcile_comparison_block_inplace,
    save_vote_store,
    summarize_verdicts_for_filtered,
    vote_store_path,
)

from discord_sft.ui.pages.evals.paths import evals_root
from discord_sft.ui.pages.evals.persona_render import (
    read_persona_generations_jsonl,
    render_persona_compare_detail_sample,
    render_persona_detail_view,
    truncate_display,
)


def render_persona_single_file_browser() -> None:
    ev_root = evals_root()
    runs = list_runs(ev_root)
    source = st.radio(
        "Source",
        ["Saved run", "Path"],
        horizontal=True,
        key="persona_gen_source",
    )

    gens_path_resolved: Path | None = None
    if source == "Saved run":
        if not runs:
            st.info(
                "No runs under the configured eval output dir — use **Path**, "
                "or complete a run first."
            )
            return
        run_labels = list(reversed([r["run_id"] for r in runs]))

        picked_id = st.selectbox(
            "Run (full run_id)",
            run_labels,
            key="persona_gen_run",
        )
        picked = next((r for r in runs if r["run_id"] == picked_id), None)
        if not picked:
            return
        try:
            doc = load_run(Path(picked["path"]))
        except OSError as e:
            st.error(f"Could not load run JSON: {e}")
            return
        persona_blob = doc.get("persona") or {}
        raw_path = persona_blob.get("generations_path")
        if not raw_path:
            st.warning(
                "This run has no ``persona.generations_path`` (persona task may have been skipped)."
            )
            return
        gens_path_resolved = resolve_repo_path(raw_path)
        st.caption(f"Resolved path: `{gens_path_resolved}`")
    else:
        default_guess = ""
        if runs:
            try:
                latest = load_run(Path(runs[-1]["path"]))
                gp = (latest.get("persona") or {}).get("generations_path")
                if gp:
                    default_guess = str(gp)
            except OSError:
                pass
        raw_in = st.text_input(
            "persona_generations.jsonl path",
            value=default_guess,
            placeholder="out/evals/raw/<run_id>/persona_generations.jsonl",
            key="persona_gen_manual_path",
        ).strip()
        if not raw_in:
            return
        gens_path_resolved = resolve_repo_path(raw_in)

    if gens_path_resolved is None:
        return
    if not gens_path_resolved.is_file():
        st.error(f"File not found: {gens_path_resolved}")
        return

    rows, err = read_persona_generations_jsonl(gens_path_resolved)
    if err:
        st.error(err)
        return
    if not rows:
        st.warning("No JSON lines in file.")
        return

    personas = sorted(
        {(str(r.get("persona_id")), str(r.get("persona_name") or "")) for r in rows},
        key=lambda x: (x[0], x[1]),
    )
    persona_options = ["(all)"] + [f"{pid} — {pn or '?'}" for pid, pn in personas]
    pick = st.selectbox("Persona filter", persona_options, key="persona_gen_persona_sb")
    if pick == "(all)":
        filtered = rows
    else:
        target_id = pick.split(" ", 1)[0]
        filtered = [r for r in rows if str(r.get("persona_id")) == target_id]

    needle = st.text_input(
        "Text filter (substring in reference or generated)", "", key="persona_gen_needle"
    )
    if needle.strip():
        n = needle.lower()
        filtered = [
            r
            for r in filtered
            if n in str(r.get("reference", "")).lower()
            or n in str(r.get("generated", "")).lower()
        ]

    st.caption(f"Showing **{len(filtered)}** rows (of {len(rows)} in file).")
    if not filtered:
        return

    view = st.radio(
        "View", ["Detail", "Table (truncated)"], horizontal=True, key="persona_gen_view"
    )
    if view == "Table (truncated)":
        preview = []
        for r in filtered:
            pid = str(r.get("persona_id", ""))
            pname = str(r.get("persona_name", ""))
            preview.append(
                {
                    "persona_id": pid,
                    "persona_name": pname,
                    "reference": truncate_display(r.get("reference"), 200),
                    "generated": truncate_display(r.get("generated"), 200),
                }
            )
        try:
            import pandas as pd

            st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)
        except ImportError:
            st.json(preview)
        return

    idx = st.slider(
        "Row index (within filtered list)", 0, len(filtered) - 1, 0, key="persona_gen_idx"
    )
    render_persona_detail_view(filtered[idx])


def render_persona_compare_runs_browser() -> None:
    ev_root = evals_root()
    runs = list_runs(ev_root)
    st.caption(
        "Join runs on **persona + context + reference**. "
        "Accept / Neutral / Reject judgments save under ``evals/persona_compare_votes.json``."
    )
    src = st.radio(
        "Compare sources",
        ["Saved runs", "JSONL paths"],
        horizontal=True,
        key="persona_cmp_src_mode",
    )

    indexed: list[tuple[Path, list[dict[str, Any]], dict[str, Any] | None, str, int]] = []

    if src == "Saved runs":
        if not runs:
            st.info("No saved runs — use **JSONL paths** or finish an eval first.")
            return
        run_labels = list(reversed([r["run_id"] for r in runs]))

        default_n = min(2, len(run_labels))
        default_pick = run_labels[-default_n:] if default_n else []
        picked = st.multiselect(
            "Runs (full run_ids; select at least two)",
            run_labels,
            default=default_pick,
            key="persona_cmp_run_ms",
            help="Shows full run ids.",
        )
        if len(picked) < 2:
            st.warning("Select **at least two** runs.")
            return
        for idx, rid in enumerate(picked):
            meta = next((r for r in runs if r["run_id"] == rid), None)
            if not meta:
                continue
            try:
                doc = load_run(Path(meta["path"]))
            except OSError as e:
                st.error(f"Could not load run `{rid}`: {e}")
                return
            raw_path = (doc.get("persona") or {}).get("generations_path")
            if not raw_path:
                st.error(f"Run `{rid}` has no ``persona.generations_path``.")
                return
            gens_path = resolve_repo_path(raw_path)
            if not gens_path.is_file():
                st.error(f"Generations file missing for `{rid}`: `{gens_path}`")
                return
            rows, err = read_persona_generations_jsonl(gens_path)
            if err:
                st.error(f"{rid}: {err}")
                return
            if not rows:
                st.error(f"Run `{rid}` has an empty generations file.")
                return
            label = str(doc.get("label") or rid)
            indexed.append((gens_path, rows, doc, label, idx))
    else:
        paths_raw = st.text_area(
            "One ``persona_generations.jsonl`` path per line",
            placeholder="out/evals/raw/<run_a>/persona_generations.jsonl\nout/evals/raw/<run_b>/persona_generations.jsonl",
            key="persona_cmp_path_lines",
        )
        runs_raw = st.text_area(
            "Optional matching **run** JSON paths (same order; leave line blank to skip)",
            placeholder="out/evals/runs/<run_a>.json\nout/evals/runs/<run_b>.json",
            key="persona_cmp_run_json_lines",
        )
        path_lines = [ln.strip() for ln in paths_raw.splitlines() if ln.strip()]
        run_lines = [ln.strip() for ln in runs_raw.splitlines()]
        if len(path_lines) < 2:
            st.warning("Enter **at least two** JSONL paths (one per line).")
            return
        for idx, pl in enumerate(path_lines):
            gens_path = resolve_repo_path(pl)
            if not gens_path.is_file():
                st.error(f"File not found: `{gens_path}`")
                return
            rows, err = read_persona_generations_jsonl(gens_path)
            if err:
                st.error(f"`{gens_path}`: {err}")
                return
            if not rows:
                st.error(f"Empty file: `{gens_path}`")
                return
            run_doc: dict[str, Any] | None = None
            display_label = gens_path.parent.name
            if idx < len(run_lines) and run_lines[idx]:
                try:
                    run_doc = load_run(resolve_repo_path(run_lines[idx]))
                    display_label = str(run_doc.get("label") or run_doc.get("run_id") or display_label)
                except OSError as e:
                    st.error(f"Could not load run JSON `{run_lines[idx]}`: {e}")
                    return
            indexed.append((gens_path, rows, run_doc, display_label, idx))

    merged, merge_warns = merge_generation_sources(indexed)
    for w in merge_warns:
        st.warning(w)
    if not merged:
        st.error("No samples present in **all** sources after joining.")
        return

    paths_for_id = [t[0] for t in indexed]
    cid = comparison_id_from_paths(paths_for_id)
    ev_root = evals_root()
    vpath = vote_store_path(ev_root)
    doc_v = load_vote_store(vpath)
    blk = doc_v.setdefault("comparisons", {}).setdefault(cid, {})
    reconcile_comparison_block_inplace(blk)
    vm = blk.setdefault("variants", {})
    if isinstance(vm, dict):
        vm.update(comparison_variant_metadata(indexed))
    else:
        blk["variants"] = dict(comparison_variant_metadata(indexed))
    save_vote_store(vpath, doc_v)

    st.caption(
        f"**{len(merged)}** aligned samples · comparison id `{cid}` · "
        f"store `{vpath}`"
    )

    pfx_pbtn = f"pbtn__{cid}__"
    pfx_pver = f"pver__{cid}__"
    pfx_pvote = f"pvote__{cid}__"
    pfx_pgen = f"pgen__{cid}__"
    if st.button("Clear judgments for this comparison", key="persona_cmp_reset_votes"):
        doc2 = load_vote_store(vpath)
        comps = doc2.get("comparisons")
        if isinstance(comps, dict) and cid in comps:
            del comps[cid]
        save_vote_store(vpath, doc2)
        for k in list(st.session_state.keys()):
            if not isinstance(k, str):
                continue
            if k.startswith(pfx_pbtn) or k.startswith(pfx_pver) or k.startswith(
                pfx_pvote
            ) or k.startswith(pfx_pgen):
                del st.session_state[k]
        st.rerun()

    personas_cmp = sorted(
        {
            (str(m.base_row.get("persona_id")), str(m.base_row.get("persona_name") or ""))
            for m in merged
        },
        key=lambda x: (x[0], x[1]),
    )
    persona_opts = ["(all)"] + [f"{pid} — {pn or '?'}" for pid, pn in personas_cmp]
    persona_pick_cmp = st.selectbox(
        "Persona filter", persona_opts, key="persona_cmp_persona_sb"
    )
    if persona_pick_cmp == "(all)":
        merged_f = list(merged)
    else:
        tid = persona_pick_cmp.split(" ", 1)[0]
        merged_f = [m for m in merged if str(m.base_row.get("persona_id")) == tid]

    needle_cmp = st.text_input(
        "Text filter (reference or any variant)", "", key="persona_cmp_needle"
    )
    if needle_cmp.strip():
        n_lc = needle_cmp.lower()
        tmp: list[Any] = []
        for mm in merged_f:
            if n_lc in str(mm.base_row.get("reference") or "").lower():
                tmp.append(mm)
                continue
            hits = False
            for vr in mm.variants:
                if n_lc in (vr.generated or "").lower():
                    hits = True
                    break
            if hits:
                tmp.append(mm)
        merged_f = tmp

    st.caption(f"Showing **{len(merged_f)}** joined rows (filtered).")
    if not merged_f:
        return

    vid_to_disp: dict[str, str] = {}
    for mm in merged_f:
        for vr in mm.variants:
            vid_to_disp.setdefault(
                vr.variant_id,
                f"{vr.display_label} · {badge_label(vr.target_modules)}",
            )

    def render_scores_expander() -> None:
        doc_sv = load_vote_store(vpath)
        raw_cmp = (doc_sv.get("comparisons") or {}).get(cid) or {}
        verdicts_for_scores = comparison_verdict_map(raw_cmp)
        summary = summarize_verdicts_for_filtered(verdicts_for_scores, merged_f)
        rank = ranked_variant_order(summary)
        with st.expander(f"Scores ({len(merged_f)} prompts in filter)", expanded=True):
            if not summary:
                st.caption("No variants in the current filter.")
            else:
                try:
                    import pandas as pd

                    rows_lb: list[dict[str, Any]] = []
                    for vid in rank:
                        stt = summary[vid]
                        rows_lb.append(
                            {
                                "label": truncate_display(vid_to_disp.get(vid, vid), 64),
                                "accept": stt["accepted"],
                                "reject": stt["rejected"],
                                "neutral": stt["neutral"],
                                "variant_id": vid,
                            }
                        )
                    df = pd.DataFrame(rows_lb)
                    st.dataframe(df, width="stretch", hide_index=True)
                except ImportError:
                    for vid in rank:
                        stt = summary[vid]
                        st.write(
                            f"{vid_to_disp.get(vid, vid)} — "
                            f"+{stt['accepted']} / −{stt['rejected']} / neutral {stt['neutral']}"
                        )
                if rank:
                    top = rank[0]
                    stt_top = summary[top]
                    if stt_top["accepted"] > 0 or stt_top["rejected"] > 0:
                        st.success(
                            "**Leading (accepts ↓ rejects ↑):** "
                            + truncate_display(vid_to_disp.get(top, top), 120)
                        )

    view_cmp = st.radio(
        "View", ["Detail", "Table (truncated)"], horizontal=True, key="persona_cmp_view"
    )
    if view_cmp == "Table (truncated)":
        preview_cmp: list[dict[str, Any]] = []
        for mm in merged_f:
            row0: dict[str, Any] = {
                "persona_id": str(mm.base_row.get("persona_id") or ""),
                "persona_name": str(mm.base_row.get("persona_name") or ""),
                "reference": truncate_display(mm.base_row.get("reference"), 200),
            }
            for j, vr in enumerate(mm.variants, start=1):
                row0[f"gen_{j}"] = truncate_display(vr.generated, 200)
                row0[f"gen_{j}_style"] = badge_label(vr.target_modules)
            preview_cmp.append(row0)
        try:
            import pandas as pd

            st.dataframe(pd.DataFrame(preview_cmp), width="stretch", hide_index=True)
        except ImportError:
            st.json(preview_cmp)
        render_scores_expander()
        return

    caps = max(0, len(merged_f) - 1)
    if "persona_cmp_slider" not in st.session_state:
        st.session_state.persona_cmp_slider = 0
    if int(st.session_state.persona_cmp_slider) > caps:
        st.session_state.persona_cmp_slider = caps

    def _persona_cmp_step_prev() -> None:
        st.session_state.persona_cmp_slider = max(
            0, int(st.session_state.persona_cmp_slider) - 1
        )

    def _persona_cmp_step_next() -> None:
        st.session_state.persona_cmp_slider = min(
            caps, int(st.session_state.persona_cmp_slider) + 1
        )

    # Prev / Next cannot assign ``persona_cmp_slider`` inside ``with`` blocks that run after
    # ``st.slider`` for the same key — use ``on_click`` so updates run before the widget tree.
    b_prev, b_mid, b_next = st.columns([1, 6, 1])
    with b_prev:
        st.button("← Prev", key="persona_cmp_prev", on_click=_persona_cmp_step_prev)
    with b_mid:
        st.slider(
            "Sample index (filtered)",
            min_value=0,
            max_value=caps,
            step=1,
            key="persona_cmp_slider",
        )
    with b_next:
        st.button("Next →", key="persona_cmp_next", on_click=_persona_cmp_step_next)

    idx_cmp = int(st.session_state.persona_cmp_slider)
    render_persona_compare_detail_sample(
        merged_f[idx_cmp],
        comparison_id=cid,
        store_path=vpath,
        sample_index=idx_cmp,
        sample_total=len(merged_f),
    )
    render_scores_expander()


def _render_profile_drift_vs_corpus() -> None:
    """Compare ``persona_generations.jsonl`` aggregates to ``profiles.json`` scalars."""
    st.caption(
        "Fingerprint rows are corpus-level stats from ``discord-sft fingerprint``; "
        "generations are summed over the selected eval dump. "
        "Deltas are absolute differences (lower is closer to the mined training style)."
    )

    prof_fp = existing_file(
        st.text_input(
            "profiles.json (from ``discord-sft fingerprint``)",
            value=session_work_path("sft", "profiles.json"),
            key="profdrift_profiles_json",
        ).strip()
    )
    if prof_fp is None:
        st.warning("Set a valid path to ``profiles.json``.")
        return

    try:
        profiles_doc = json.loads(prof_fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        st.error(f"Could not read profiles JSON: {e}")
        return

    ev_root = evals_root()
    runs = list_runs(ev_root)

    source = st.radio(
        "Generations source",
        ["Saved run", "Path"],
        horizontal=True,
        key="profdrift_gens_source",
    )
    gens_path_resolved: Path | None = None
    if source == "Saved run":
        if not runs:
            st.info("No saved runs — pick **Path** or finish a persona eval first.")
            return
        run_labels = list(reversed([r["run_id"] for r in runs]))
        picked_id = st.selectbox(
            "Run (full run_id)",
            run_labels,
            key="profdrift_gens_run",
        )
        picked = next((r for r in runs if r["run_id"] == picked_id), None)
        if picked is None:
            return
        try:
            doc = load_run(Path(picked["path"]))
        except OSError as e:
            st.error(f"Could not load run JSON: {e}")
            return
        raw_path = (doc.get("persona") or {}).get("generations_path")
        if not raw_path:
            st.warning("This run has no ``persona.generations_path``.")
            return
        gens_path_resolved = resolve_repo_path(str(raw_path))
        st.caption(f"Resolved generations: `{gens_path_resolved}`")
    else:
        default_guess = ""
        if runs:
            try:
                latest = load_run(Path(runs[-1]["path"]))
                gp = (latest.get("persona") or {}).get("generations_path")
                if gp:
                    default_guess = str(gp)
            except OSError:
                pass
        raw_in = st.text_input(
            "persona_generations.jsonl path",
            value=default_guess,
            key="profdrift_gens_path",
        ).strip()
        if not raw_in:
            st.warning("Enter a JSONL path.")
            return
        gens_path_resolved = resolve_repo_path(raw_in)

    if gens_path_resolved is None or not gens_path_resolved.is_file():
        st.error(f"Generations file not found: {gens_path_resolved}")
        return

    go = st.button("Compute drift vs fingerprint", type="primary", key="profdrift_compute")
    if not go:
        return

    rows, read_err = read_persona_generations_jsonl(gens_path_resolved)
    if read_err:
        st.error(read_err)
        return
    if not rows:
        st.warning("No rows in generations file.")
        return

    table, warns = aggregate_profile_drift_for_eval_rows(rows, profiles_doc)
    for w in warns:
        st.warning(w)
    if not table:
        st.error("No personas could be scored — check ``persona_id`` keys match profiles.")
        return

    table.sort(key=lambda r: float(r.get("mean_words_delta", 0.0)), reverse=True)

    display_cols = [
        "persona_id",
        "persona_name",
        "n_generated",
        "mean_words_delta",
        "profile_mean_words",
        "generated_mean_words",
        "lowercase_start_delta",
        "profile_lowercase_start_rate",
        "generated_lowercase_start_rate",
        "burst_size_delta",
        "profile_burst_size",
        "generated_burst_size",
        "emoji_unicode_per_turn_delta",
        "profile_emoji_unicode_per_turn",
        "generated_emoji_unicode_per_turn",
        "emoji_custom_per_turn_delta",
        "profile_emoji_custom_per_turn",
        "generated_emoji_custom_per_turn",
    ]
    slim = []
    for row in table:
        slim.append({k: row[k] for k in display_cols if k in row})

    try:
        import pandas as pd

        df = pd.DataFrame(slim).round(4)
        st.dataframe(df, width="stretch", hide_index=True)
    except ImportError:
        st.json(slim)

    with st.expander("Metric guide", expanded=False):
        st.markdown(
            "- **mean_words** · average word count per message (assistant style length).\n"
            "- **lowercase_start_rate** · share of messages whose first alphabetic char is lowercase.\n"
            "- **burst_size** · average ``lines per message`` proxy (newline segments).\n"
            "- **emoji_*_per_turn** · unicode vs ``:custom:`` emoji counts per assistant message.\n"
        )

    csv_lines = ["\t".join(display_cols)]
    for row in slim:
        csv_lines.append("\t".join(str(row.get(c, "")) for c in display_cols))
    csv_blob = "\n".join(csv_lines) + "\n"
    st.download_button(
        "Download drift table (TSV)",
        data=csv_blob.encode("utf-8"),
        file_name="profile_drift_vs_run.tsv",
        mime="text/tab-separated-values",
        key="profdrift_dl_tsv",
    )


def _render_paste_heuristics() -> None:
    st.caption("Cheap style drift vs references; optional profile-guided scores.")
    c1, c2 = st.columns(2)
    with c1:
        refs_text = st.text_area("References (one per line)", height=200, key="eval_persona_heur_refs")
    with c2:
        gens_text = st.text_area("Generations (one per line, aligned)", height=200, key="eval_persona_heur_gens")
    profile_path = existing_file(
        st.text_input(
            "Optional profiles.json path",
            value=session_work_path("sft", "profiles.json"),
            key="eval_persona_heur_prof_path",
        )
    )
    profile_doc: dict | None = None
    persona_id: str | None = None
    if profile_path:
        profile_doc = json.loads(profile_path.read_text(encoding="utf-8"))
        personas = list(profile_doc.get("personas", {}).keys())
        if personas:
            persona_id = st.selectbox("Persona", personas, key="eval_persona_heur_persona")
    if st.button("Score lines", type="secondary", key="eval_persona_heur_btn"):
        refs = [ln for ln in refs_text.splitlines() if ln.strip()]
        gens = [ln for ln in gens_text.splitlines() if ln.strip()]
        if not refs or not gens or len(refs) != len(gens):
            st.error("Provide same-number non-empty references and generations.")
            return
        stats = (
            profile_heuristics(gens, refs, profile=profile_doc["personas"][persona_id])
            if profile_doc and persona_id
            else style_heuristics(gens, refs)
        )
        st.json(stats)


def render_persona_generations_tab() -> None:
    st.subheader("Persona outputs")
    st.caption(
        "Browse ``persona_generations.jsonl`` from a saved run "
        "(``persona.generations_path``), compare checkpoints, and vote on variants."
    )
    with st.expander("Run vs corpus fingerprint", expanded=False):
        _render_profile_drift_vs_corpus()
    with st.expander("Paste aligned lines (quick heuristics)", expanded=False):
        _render_paste_heuristics()
    st.divider()
    mode = st.radio(
        "Browse mode",
        ["Single file", "Compare runs"],
        horizontal=True,
        key="persona_gen_browse_mode",
    )
    if mode == "Single file":
        render_persona_single_file_browser()
    else:
        render_persona_compare_runs_browser()
