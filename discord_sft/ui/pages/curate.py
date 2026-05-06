"""Curate page."""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone
import streamlit as st

from discord_sft.data_prep.curate import CurateReport, curate_messages, session_to_record
from discord_sft.data_prep.curate_sweep import default_sweep_lists, iter_curate_sweep_rows
from discord_sft.data_prep.ingest import Message, iter_folders, iter_messages
from discord_sft.ui.common import existing_dir, report_to_df, resolve_repo_path, session_work_path


def _csv_or_none(raw: str) -> str | None:
    t = (raw or "").strip()
    return t if t else None


def _sweep_rows_to_table(rows: list[dict]) -> list[dict]:
    """Flatten sweep JSONL rows for a compact dataframe."""
    out: list[dict] = []
    for row in rows:
        p = row["params"]
        r = row["report"]
        flat: dict = {
            "gap_min": p["session_gap_min"],
            "merge_s": p["merge_gap_sec"],
            "min_turns": p["min_turns"],
            "min_authors": p["min_authors"],
            "mono_share": p["monologue_max_share"],
            "sessions_kept": row["sessions_kept"],
            "sessions_built": row["sessions_built"],
        }
        for k, v in r.items():
            if k.startswith("dropped_") and isinstance(v, int) and v:
                flat[k] = v
        out.append(flat)
    return out


def _synthetic_curate_messages() -> list[Message]:
    """Tiny hand-crafted DM used by the Curate page's live example.

    The messages are arranged so every Curate control visibly changes something:
      - `bob`'s ``/help`` is a bot-command (drops on any settings).
      - `alice` sends two messages 20s apart (burst-merge gap).
      - URLs and an email appear (strip_urls / pii_scrub effects).
      - A 3.5h silence between "session A" and "session B" (session_gap_min).
      - Session B is near-identical to session A (near-dup dedup).
      - `alice` dominates word count in session A (monologue cap).
    """
    base_tz = timezone.utc

    def _msg(
        mid: str,
        ts: datetime,
        author_id: str,
        author_name: str,
        content: str,
        *,
        msg_type: int = 0,
    ) -> Message:
        return Message(
            id=mid,
            ts=ts,
            author_id=author_id,
            author_name=author_name,
            content=content,
            edited_ts=None,
            attachments=(),
            num_embeds=0,
            type=msg_type,
            referenced_id=None,
            reply_to_preview=None,
            reply_to_author_id=None,
            reply_to_author_name=None,
            reply_to_missing=False,
            folder="demo",
        )

    out: list[Message] = []
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=base_tz)
    out.append(_msg("1", t0.replace(minute=0, second=0), "bob", "bob", "hey alice"))
    out.append(
        _msg(
            "2",
            t0.replace(minute=0, second=30),
            "alice",
            "alice",
            "hi bob check out https://example.com/post very cool thing",
        )
    )
    out.append(
        _msg(
            "3",
            t0.replace(minute=0, second=50),
            "alice",
            "alice",
            "also my email is alice@example.com ping me later please",
        )
    )
    out.append(_msg("4", t0.replace(minute=1, second=30), "bob", "bob", "cool lol"))
    out.append(
        _msg(
            "5",
            t0.replace(minute=2, second=0),
            "alice",
            "alice",
            "yeah agreed totally makes sense i think so too honestly",
        )
    )
    out.append(_msg("6", t0.replace(minute=2, second=15), "bob", "bob", "/help"))
    out.append(_msg("7", t0.replace(minute=2, second=45), "bob", "bob", "anyway ttyl"))

    t1 = datetime(2024, 1, 1, 13, 30, 0, tzinfo=base_tz)
    out.append(_msg("8", t1, "bob", "bob", "hey alice"))
    out.append(
        _msg(
            "9",
            t1.replace(second=10),
            "alice",
            "alice",
            "hi bob check out https://example.com/post very cool thing alice@example.com",
        )
    )
    out.append(_msg("10", t1.replace(second=30), "bob", "bob", "cool lol"))
    out.append(
        _msg(
            "11",
            t1.replace(second=40),
            "alice",
            "alice",
            "yeah agreed totally makes sense i think so too honestly",
        )
    )
    out.append(_msg("12", t1.replace(second=55), "bob", "bob", "anyway ttyl"))
    return out


def _render_sample_preview(
    *,
    session_gap_min: int,
    merge_gap_sec: int,
    min_turns: int,
    min_authors: int,
    monologue: float,
    near_dedup_thr: float,
    do_near_dedup: bool,
    dedupe_exact_turns: bool,
    exact_turn_dup_cap: int,
    strip_urls: bool,
    pii_scrub: bool,
    lang: str,
) -> None:
    """Sample preview: one `curate_messages` run shared by both tabs (toggles stay live)."""
    st.subheader("Sample preview")
    st.caption(
        "Uses a **fixed 12-message** demo (bob / alice, burst-merge, URLs, PII, 3.5h gap, "
        "near-duplicate second chat). **Strip URLs**, **PII scrub**, gates, and dedup apply in "
        "the pipeline — the **Pipeline peek** and **Metrics & sessions** tabs both refresh on "
        "every control change. Your real export is not read here."
    )
    raw = _synthetic_curate_messages()
    report = CurateReport()
    sessions = []
    preview_error: str | None = None
    try:
        sessions, report = curate_messages(
            raw,
            folder="demo",
            merge_gap_sec=merge_gap_sec,
            session_gap_min=session_gap_min,
            min_turns=min_turns,
            min_authors=min_authors,
            monologue_max_share=monologue,
            url_strip=strip_urls,
            pii_scrub=pii_scrub,
            lang=lang or None,
            near_dedup_threshold=near_dedup_thr if do_near_dedup else None,
            dedupe_exact_turns=dedupe_exact_turns,
            exact_turn_dup_cap=int(exact_turn_dup_cap),
            report=report,
        )
    except Exception as exc:  # noqa: BLE001 - live preview must not crash the page
        preview_error = str(exc)
        sessions = []

    tab_demo, tab_out = st.tabs(["Demo input, tips & pipeline peek", "Metrics & sessions"])

    with tab_demo:
        if preview_error:
            st.error(f"Preview failed: {preview_error}")
        st.markdown(
            "**Source messages (fixed demo)** — what Discrub would export; "
            "URL/PII scrub and stripping apply **after** this in the real pipeline."
        )
        st.dataframe(
            [
                {
                    "#": i + 1,
                    "Time": m.ts.strftime("%H:%M:%S"),
                    "Author": m.author_name,
                    "Content": m.content,
                }
                for i, m in enumerate(raw)
            ],
            hide_index=True,
            width="stretch",
        )

        st.divider()
        st.markdown("**Exact-repeat dedup** (after burst-merge, same session)")

        st.markdown(
            """
| Speaker | Text |
|---------|------|
| Bob | ready? |
| Alice | yep |
| Alice | yep |
| Alice | yep |
| Bob | go |

Same **speaker + identical line** three times.

- **Cap = 1** — keep one `yep`, drop two.
- **Cap = 2** — keep two, drop one.
- **Dedupe off** — all three stay.

Fixed **keep budget** per repeated line (not random). Matches **Exact-repeat keep cap** above.
            """.strip()
        )

        st.divider()
        st.markdown(
            "**Pipeline peek** — effective **merged turn text** after burst-merge, "
            "URL strip, PII scrub, and exact-repeat dedup for your current settings."
        )
        if preview_error:
            st.caption("No pipeline output while preview is in error.")
        elif not sessions:
            st.info(
                "No sessions survived the gates — nothing to peek. Loosen **min turns**, "
                "**min authors**, or **monologue**."
            )
        else:
            peek_rows: list[dict] = []
            for si, sess in enumerate(sessions):
                rec = session_to_record(sess)
                for ti, t in enumerate(rec["turns"]):
                    peek_rows.append(
                        {
                            "Session": si + 1,
                            "Turn": ti + 1,
                            "Author": t["author_name"],
                            "Effective text": t["text"],
                        }
                    )
            st.dataframe(peek_rows, hide_index=True, width="stretch")

    with tab_out:
        if preview_error:
            st.error(f"Preview failed: {preview_error}")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Messages in", f"{report.total_in:,}")
            m2.metric("Sessions built", f"{report.sessions_built:,}")
            m3.metric("Sessions kept", f"{len(sessions):,}")
            turns_kept = sum(len(s.turns) for s in sessions)
            m4.metric("Turns kept", f"{turns_kept:,}")

            drops = {
                k: v
                for k, v in dataclasses.asdict(report).items()
                if k.startswith("dropped_") and isinstance(v, int) and v > 0
            }
            if drops:
                st.caption("Non-zero drops this run")
                st.bar_chart(drops, horizontal=True)

            if not sessions:
                st.info(
                    "These settings remove every session. Loosen **min turns**, **min authors**, "
                    "or **monologue**."
                )
            else:
                st.markdown("**Sessions kept** — one block per session (merged turns).")
                for i, sess in enumerate(sessions):
                    rec = session_to_record(sess)
                    with st.container(border=True):
                        st.markdown(
                            f"**Session {i + 1}** · `{rec['session_id']}` · {len(rec['turns'])} turns · "
                            f"{', '.join(rec['authors'])}"
                        )
                        st.dataframe(
                            [
                                {
                                    "start": t["start_ts"].split("T", 1)[-1][:8],
                                    "author": t["author_name"],
                                    "text": t["text"],
                                }
                                for t in rec["turns"]
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                        with st.popover("JSON record"):
                            st.json(rec, expanded=False)


def render_curate() -> None:
    st.title("🧼 Curate")
    st.caption(
        "Turn raw Discord exports into session-split, quality-gated chats for SFT. "
        "Use **Sample preview** to learn the knobs, **Grid search** to compare metrics on "
        "your real data, then **Run curate** to write `sessions.jsonl`."
    )
    src = existing_dir(st.session_state["source_dir"])
    if src is None:
        st.warning("Set **Raw Discord export root** in the sidebar first.")
        return

    with st.expander("Advanced: output directory", expanded=False):
        out_dir = st.text_input(
            "Output directory",
            value=session_work_path("curated"),
            help="Where `sessions.jsonl` and `curate_report.json` will be written.",
        )

    st.markdown("##### Session shape")
    c1, c2 = st.columns(2)
    with c1:
        session_gap_min = st.slider(
            "Session gap (min)",
            15,
            720,
            60,
            15,
            help=(
                "If silence between two messages exceeds this many minutes, they "
                "start a new session. Larger values → fewer, longer sessions; "
                "smaller values → more, shorter sessions."
            ),
        )
        merge_gap_sec = st.slider(
            "Burst merge gap (sec)",
            0,
            300,
            30,
            5,
            help=(
                "Consecutive messages from the same author within this many "
                "seconds are merged into a single turn, joined by newlines. "
                "Set to 0 to disable merging (every message stays its own turn)."
            ),
        )

    st.markdown("##### Quality gates")
    c1, c2 = st.columns(2)
    with c1:
        min_turns = st.number_input(
            "Min turns per session",
            2,
            50,
            2,
            help=(
                "Sessions with fewer kept turns than this are dropped "
                "(`dropped_short_session`)."
            ),
        )
        min_authors = st.number_input(
            "Min distinct authors",
            1,
            4,
            2,
            help=(
                "Sessions with fewer distinct authors are dropped "
                "(`dropped_single_author`). Keep at 2+ to exclude solo rambles."
            ),
        )
    with c2:
        monologue = st.slider(
            "Max single-author word share",
            0.50,
            1.00,
            0.80,
            0.05,
            help=(
                "If one author contributes more than this fraction of all words "
                "in a session, the whole session is dropped (`dropped_monologue`). "
                "Lower = stricter."
            ),
        )
        near_dedup_thr = st.slider(
            "Near-dup Jaccard threshold",
            0.50,
            1.00,
            0.85,
            0.01,
            help=(
                "Two sessions whose 5-gram Jaccard similarity exceeds this are "
                "collapsed into one (`dropped_near_dup`). Lower = more "
                "aggressive deduplication."
            ),
        )

    st.markdown("##### Text cleanup & filters")
    c1, c2 = st.columns(2)
    with c1:
        lang = st.text_input(
            "Language filter (ISO code; blank = off)",
            value="",
            help=(
                "Keep only sessions whose detected language matches this "
                "ISO-639-1 code (e.g. `en`). Blank keeps all languages. "
                "Requires the optional `langdetect` package."
            ),
        )
    with c2:
        st.write("")  # vertical alignment with left column
    t1, t2, t3, t4 = st.columns(4)
    strip_urls = t1.toggle(
        "Strip URLs",
        value=False,
        help=(
            "If on, `http(s)://…` URLs are removed from message text before "
            "burst-merging. If off, URLs are kept verbatim."
        ),
    )
    pii_scrub = t2.toggle(
        "PII scrub",
        value=True,
        help=(
            "Mask email addresses, phone numbers, and long digit runs "
            "(e.g. card/account-looking numbers) with placeholder tokens."
        ),
    )
    do_near_dedup = t3.toggle(
        "Near-dup dedup",
        value=True,
        help=(
            "Enable the near-duplicate session collapse using the Jaccard "
            "threshold above. Turn off to keep all sessions verbatim."
        ),
    )
    dedupe_exact_turns = t4.toggle(
        "Dedupe exact repeats",
        value=True,
        help=(
            "If on, trim excess turns in a session that share the same author "
            "and exact text (`dropped_duplicate_turn`). Use the cap below to "
            "allow a few intentional repeats."
        ),
    )

    exact_turn_dup_cap = st.number_input(
        "Exact-repeat keep cap (N)",
        min_value=1,
        max_value=20,
        value=1,
        disabled=not dedupe_exact_turns,
        help=(
            "Per session, per identical (author + text): keep the first N "
            "occurrences; drop the rest. N=1 is strictest; raise N to soften."
        ),
    )

    _render_sample_preview(
        session_gap_min=int(session_gap_min),
        merge_gap_sec=int(merge_gap_sec),
        min_turns=int(min_turns),
        min_authors=int(min_authors),
        monologue=float(monologue),
        near_dedup_thr=float(near_dedup_thr),
        do_near_dedup=bool(do_near_dedup),
        dedupe_exact_turns=bool(dedupe_exact_turns),
        exact_turn_dup_cap=int(exact_turn_dup_cap),
        strip_urls=bool(strip_urls),
        pii_scrub=bool(pii_scrub),
        lang=lang,
    )

    st.divider()
    with st.expander(
        "Grid search (optional — compare settings, no `sessions.jsonl`)",
        expanded=False,
    ):
        st.caption(
            "Cartesian product of the comma-separated lists below. Uses **your sidebar export** "
            "and the **Text cleanup & filters** toggles above (PII, URLs, near-dup, exact dedup). "
            "Blank field = use the **single value from the sliders** for that axis. "
            "Same idea as `discord-sft curate-sweep`."
        )
        g1, g2 = st.columns(2)
        with g1:
            sw_sgm = st.text_input(
                "Session gap (min)",
                value="",
                placeholder="30,60,120",
                help="Comma-separated integers. Empty → use Session gap slider only.",
            )
            sw_mgs = st.text_input(
                "Burst merge (sec)",
                value="",
                placeholder="15,30,60",
                help="Comma-separated integers. Empty → use Burst merge slider only.",
            )
            sw_mt = st.text_input(
                "Min turns",
                value="",
                placeholder="2,4,6",
                help="Comma-separated integers. Empty → use Min turns above only.",
            )
        with g2:
            sw_ma = st.text_input(
                "Min authors",
                value="",
                placeholder="1,2",
                help="Comma-separated integers. Empty → use Min distinct authors only.",
            )
            sw_mono = st.text_input(
                "Max monologue share",
                value="",
                placeholder="0.75,0.80,0.90",
                help="Comma-separated floats. Empty → use Max single-author word share slider only.",
            )
            max_combos = st.number_input(
                "Max grid points",
                min_value=1,
                max_value=5000,
                value=500,
                help="Safety cap: refuse to run if the product of list lengths exceeds this.",
            )

        st.caption(
            "When you click **Run grid search**, a **status** box appears below this form with a "
            "spinner until the run finishes. Last results stay visible under the expander."
        )
        sweep_btn = st.button("Run grid search", type="secondary", key="curate_sweep_btn")

    if sweep_btn:
        st.session_state.pop("curate_sweep_error", None)
        try:
            sgm, mgs, mt, ma, mono = default_sweep_lists(
                session_gap_min=int(session_gap_min),
                merge_gap_sec=int(merge_gap_sec),
                min_turns=int(min_turns),
                min_authors=int(min_authors),
                monologue_max_share=float(monologue),
                sweep_session_gap_min=_csv_or_none(sw_sgm),
                sweep_merge_gap_sec=_csv_or_none(sw_mgs),
                sweep_min_turns=_csv_or_none(sw_mt),
                sweep_min_authors=_csv_or_none(sw_ma),
                sweep_monologue_max_share=_csv_or_none(sw_mono),
            )
            n_combo = math.prod(len(x) for x in (sgm, mgs, mt, ma, mono))
            if n_combo > int(max_combos):
                st.session_state["curate_sweep_error"] = (
                    f"Grid has {n_combo} points (product of list lengths), "
                    f"above Max grid points ({int(max_combos)}). Shorten lists or raise the cap."
                )
            else:
                folders = list(iter_folders(src))
                if not folders:
                    st.session_state["curate_sweep_error"] = (
                        "No DM subfolders with JSON pages found under the export root."
                    )
                else:
                    thr = near_dedup_thr if do_near_dedup else None
                    with st.status(
                        f"Running grid search — {n_combo} setting(s) × {len(folders)} folder(s)…",
                        expanded=True,
                    ) as sweep_status:
                        rows = list(
                            iter_curate_sweep_rows(
                                src,
                                session_gap_mins=sgm,
                                merge_gap_secs=mgs,
                                min_turns_list=mt,
                                min_authors_list=ma,
                                monologue_max_shares=mono,
                                url_strip=bool(strip_urls),
                                pii_scrub=bool(pii_scrub),
                                lang=lang or None,
                                near_dedup_threshold=thr,
                                dedupe_exact_turns=bool(dedupe_exact_turns),
                                exact_turn_dup_cap=int(exact_turn_dup_cap),
                            )
                        )
                        st.session_state["curate_sweep_rows"] = rows
                        st.session_state["curate_sweep_folders"] = len(folders)
                        sweep_status.update(
                            label=f"Grid search finished — {len(rows)} combination(s)",
                            state="complete",
                        )
        except ValueError as exc:
            st.session_state["curate_sweep_error"] = str(exc)

    err = st.session_state.get("curate_sweep_error")
    if err:
        st.error(err)

    rows = st.session_state.get("curate_sweep_rows")
    if rows:
        st.success(
            f"Last grid search: **{len(rows)}** setting(s) × **{st.session_state.get('curate_sweep_folders', '?')}** folder(s)."
        )
        if len(rows) == 1:
            st.caption(
                "Only one combination ran (all sweep fields blank = current sliders). "
                "Enter comma-separated values in one or more fields to compare multiple settings."
            )
        table = _sweep_rows_to_table(rows)
        st.dataframe(table, hide_index=True, width="stretch", height=min(420, 60 + len(table) * 38))
        jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        st.download_button(
            "Download results as JSONL",
            data=jsonl.encode("utf-8"),
            file_name="curate_grid_search.jsonl",
            mime="application/json",
            key="curate_sweep_download",
        )

    submitted = st.button("Run curate", type="primary")
    if not submitted:
        return

    out_path = resolve_repo_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sessions_path = out_path / "sessions.jsonl"
    report = CurateReport()
    total_sessions = 0
    prog = st.progress(0.0, text="Curating folders...")
    folders = list(iter_folders(src))
    thr = near_dedup_thr if do_near_dedup else None
    with sessions_path.open("w", encoding="utf-8") as f:
        for i, folder in enumerate(folders):
            messages = list(iter_messages(folder))
            sessions, report = curate_messages(
                messages,
                folder=folder.name,
                merge_gap_sec=merge_gap_sec,
                session_gap_min=session_gap_min,
                min_turns=int(min_turns),
                min_authors=int(min_authors),
                monologue_max_share=monologue,
                url_strip=strip_urls,
                pii_scrub=pii_scrub,
                lang=lang or None,
                near_dedup_threshold=thr,
                dedupe_exact_turns=dedupe_exact_turns,
                exact_turn_dup_cap=int(exact_turn_dup_cap),
                report=report,
            )
            for s in sessions:
                f.write(json.dumps(session_to_record(s), ensure_ascii=False) + "\n")
            total_sessions += len(sessions)
            prog.progress((i + 1) / len(folders), text=f"Curated {folder.name}")
    prog.empty()

    (out_path / "curate_report.json").write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")
    st.success(f"Wrote {total_sessions:,} sessions to {sessions_path}.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Messages in", f"{report.total_in:,}")
    c2.metric("Sessions built", f"{report.sessions_built:,}")
    c3.metric("Sessions kept", f"{report.sessions_kept:,}")
    drops = {k: v for k, v in report_to_df(report).items() if k.startswith("dropped_") and v > 0}
    if drops:
        st.bar_chart(drops, horizontal=True)
    with st.expander("How to read drops vs sessions kept"):
        st.markdown(
            """
Compare **`sessions_built`** (after gap-split and burst-merge) to **`sessions_kept`**
(after quality gates and optional near-dedup). Non-zero **`dropped_*`** keys explain
the gap; the largest bar is usually the first knob to change.

- **`dropped_short_session`** — raise **`min_turns`** to require longer threads, or
  lower it / widen **`session_gap_min`** / **`merge_gap_sec`** if you need more volume.
- **`dropped_single_author`** / **`dropped_monologue`** — conversation shape; adjust
  **`min_authors`** or **`monologue_max_share`**.
- **`dropped_near_dup`** — duplicate-looking chats removed by MinHash; raise the
  Jaccard threshold or disable near-dup in the controls above.

Change **one** control per iteration and re-run curate to a **new output folder** so
you can diff `curate_report.json` files. For a grid search without writing
`sessions.jsonl`, use **Grid search** on this page or **`discord-sft curate-sweep`**
(see repo README).
            """.strip()
        )
