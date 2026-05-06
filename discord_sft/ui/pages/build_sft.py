"""Build SFT page."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

import streamlit as st

from discord_sft.data_prep.curate import Session, Turn
from discord_sft.data_prep.sft import (
    balance_samples,
    balance_turn_length,
    build_samples,
    post_split_num_turns_breakdown,
    split_train_val,
    shuffle_samples,
    write_samples,
)
from discord_sft.ui.common import existing_file, load_sessions, resolve_repo_path, session_work_path

# Block-count mix preset (ShareGPT lengths 2/4/6/8). Select in UI or paste the same JSON.
TURN_MIX_PRESET_153025_JSON = '{"2": 0.15, "4": 0.30, "6": 0.30, "8": 0.25}'
TURN_MIX_PRESET_153025_LABEL = "15 / 30 / 30 / 25 mix"


def _synthetic_sft_session() -> Session:
    """Tiny curated session used by the Build SFT page's live example.

    Author turn counts are deliberately lopsided (alice=6, bob=3, carol=1) so
    that the Balance policy and k sliders visibly reshape the per-persona
    sample counts.
    """
    tz = timezone.utc
    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=tz)

    def _turn(i: int, author: str, text: str) -> Turn:
        ts = t0.replace(minute=i)
        return Turn(
            author_id=author,
            author_name=author,
            start_ts=ts,
            end_ts=ts,
            text=text,
            source_ids=[str(i)],
        )

    turns = [
        _turn(0, "bob", "yo"),
        _turn(1, "alice", "hey"),
        _turn(2, "bob", "how's it going"),
        _turn(3, "alice", "good, you?"),
        _turn(4, "bob", "nice"),
        _turn(5, "alice", "yeah same"),
        _turn(6, "bob", "cool"),
        _turn(7, "alice", "haha"),
        _turn(8, "bob", "indeed"),
        _turn(9, "alice", "lol"),
        _turn(10, "alice", "same"),
        _turn(11, "carol", "hi everyone"),
    ]
    return Session(id="demo#2024-01-01T10:00:00Z", folder="demo", turns=turns)


def _render_sft_live_example(
    *,
    window_turns: int,
    min_sharegpt_turns: int,
    max_sharegpt_turns: int | None,
    turn_mix: str,
    personas: list[str],
    val_frac: float,
    seed: int,
    balance_mode: str,
    balance_k: float,
    cap_n: int,
    expanded: bool = True,
) -> None:
    """Run the SFT builder on a tiny synthetic session and show the output live."""
    with st.expander(
        "Live example — watch the output change as you tweak options",
        expanded=expanded,
    ):
        st.caption(
            "A synthetic one-session mini-corpus with lopsided activity "
            "(alice=6 raw lines, bob=3, carol=1). Every control above visibly "
            "reshapes the sample stream."
        )
        demo_session = _synthetic_sft_session()

        with st.expander("Synthetic session (fixed)", expanded=False):
            st.dataframe(
                [
                    {
                        "start": t.start_ts.strftime("%H:%M"),
                        "author": t.author_name,
                        "text": t.text,
                    }
                    for t in demo_session.turns
                ],
                hide_index=True,
                width="stretch",
            )

        policy = balance_mode if balance_mode != "cap" else f"cap:{int(cap_n)}"
        try:
            raw_samples = build_samples(
                [demo_session],
                personas=personas,
                window_turns=window_turns,
                min_sharegpt_turns=min_sharegpt_turns,
                max_sharegpt_turns=max_sharegpt_turns,
            )
            balanced, balance_report = balance_samples(
                raw_samples, policy=policy, k=balance_k, seed=seed
            )
            mixed, _turn_report = balance_turn_length(
                balanced, policy=turn_mix.strip() or "none", seed=seed
            )
            train, val = split_train_val(mixed, val_frac=val_frac, seed=seed)
        except Exception as exc:  # noqa: BLE001 - live preview must not crash the page
            st.error(f"Preview failed: {exc}")
            return

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Samples built", f"{len(raw_samples):,}")
        m2.metric("After persona balance", f"{len(balanced):,}")
        m3.metric("After turn-mix", f"{len(mixed):,}")
        m4.metric("Train", f"{len(train):,}")
        m5.metric("Val", f"{len(val):,}")

        counts_before = balance_report.counts_before or {}
        counts_after = balance_report.counts_after or {}
        caps = balance_report.cap_per_persona or {}
        if counts_before:
            rows = [
                {
                    "persona": pid,
                    "before": int(counts_before.get(pid, 0)),
                    "cap": int(caps.get(pid, counts_before.get(pid, 0))),
                    "after": int(counts_after.get(pid, 0)),
                }
                for pid in sorted(counts_before)
            ]
            st.caption(f"Per-persona balance (policy = `{policy}`)")
            st.dataframe(rows, hide_index=True, width="stretch")

        if not mixed:
            st.info(
                "Current settings produce zero samples. Try selecting more "
                "personas or lowering the cap."
            )
            return

        preview = mixed[: min(3, len(mixed))]
        tabs = st.tabs([f"Sample {i + 1} ({s.meta.get('persona_id')})" for i, s in enumerate(preview)])
        for tab, sample in zip(tabs, preview):
            with tab:
                st.caption(f"**system:** {sample.system}")
                st.dataframe(
                    [{"from": c["from"], "value": c["value"]} for c in sample.conversations],
                    hide_index=True,
                    width="stretch",
                )
                with st.expander("Raw JSON sample", expanded=False):
                    st.json(sample.to_json(), expanded=False)


def render_build_sft() -> None:
    st.title("🧱 Build SFT")
    st.caption(
        "Turn each curated chat session into ShareGPT training rows (one row per "
        "session per selected persona—final reply only), optionally balance personas "
        "and lengths, then split whole sessions into train and validation. "
        "The live example below mirrors your settings."
    )

    default_in = session_work_path("curated", "sessions.jsonl")
    default_out = session_work_path("sft")
    with st.expander("Advanced: input/output paths", expanded=False):
        in_str = st.text_input("Sessions JSONL path", value=str(default_in))
        out_dir = st.text_input(
            "Output directory",
            value=str(default_out),
            help=(
                "Where `train.jsonl`, `val.jsonl`, `balance_report.json`, and "
                "`turn_length_report.json` are written. The turn report includes "
                "`post_split_num_turns` (TRAIN_split vs VALIDATION_split) after the session split."
            ),
        )
    sess_path = existing_file(in_str)
    if sess_path is None:
        st.warning("Run **Curate** first, or point to an existing sessions.jsonl.")
        return
    sessions = load_sessions(sess_path)
    all_personas = sorted({a for s in sessions for a in s.authors()})

    st.markdown("##### Sample shape")
    with st.expander("New here? How raw lines differ from ShareGPT blocks", expanded=False):
        st.markdown(
            """
Two separate ideas control how long each **training row** can be. The command-line
flags reuse the word “turn” for both; this page uses clearer words.

**Raw lines (Discord messages in the slice)**  
Each line in your curated session: one person, one piece of text in order. For each
training row, the builder only looks at a **contiguous chunk** of the session ending
on that persona's **last** raw line in that session. **Max raw lines in the slice**
bounds how far back that chunk may reach (CLI: `--window-turns`).

**ShareGPT blocks (what one training row contains)**  
The slice is mapped to alternating `user` and `assistant`. If the same side speaks
several raw lines in a row, those lines are **merged** into a single block (one
`{from, value}`). **Min / max blocks** count those merged entries—the length of the
`conversations` list—not “one block per raw line” (CLI: `--min-sharegpt-turns`,
`--max-sharegpt-turns`). Allowed values are even (2, 4, …) because a valid row ends on
assistant.

**Order of operations**  
The builder emits **one row per session per selected persona** (final assistant line
only), picks a slice (bounded by raw lines), merges to blocks, then keeps a row only
if its block count is between your min and max. **Block-count mix** (below) optionally
downsamples rows toward target proportions of block counts 2 / 4 / 6 / 8 *after*
persona balancing.
            """.strip()
        )

    c1, c2 = st.columns(2)
    with c1:
        window_turns = st.slider(
            "Max raw lines in the slice",
            2,
            64,
            16,
            help=(
                "Upper bound on how many consecutive curated session lines may appear "
                "in the context ending at each persona's **last** line in that session. "
                "Independent of the block limits below. CLI: --window-turns."
            ),
        )
        min_sg = st.select_slider(
            "Min ShareGPT blocks per row",
            options=[2, 4, 6, 8],
            value=2,
            help=(
                "Smallest allowed number of merged user/assistant entries in one "
                "training row (even). Shorter rows are dropped. CLI: --min-sharegpt-turns."
            ),
        )
        no_sg_cap = st.checkbox(
            "No cap on ShareGPT blocks (max)",
            value=False,
            help=(
                "When checked, there is no upper limit on block count: only the raw-line "
                "slider above bounds how much history is considered, and the row uses a "
                "single fixed slice (CLI: --max-sharegpt-turns none)."
            ),
        )
        max_sg_options = [m for m in (2, 4, 6, 8) if m >= int(min_sg)]
        # Always render the same widget type here so Streamlit session state for widgets
        # below (block-mix JSON, etc.) is not remapped when min/max options collapse to one.
        max_sg = st.select_slider(
            "Max ShareGPT blocks per row",
            options=max_sg_options,
            value=max(max_sg_options),
            disabled=no_sg_cap or len(max_sg_options) == 1,
            help=(
                "Among valid slices, prefer the longest row whose block count is at "
                "most this value (and at least the min above). CLI: --max-sharegpt-turns. "
                "Disabled when it must match min, or when the no-cap checkbox is on."
            ),
            key="build_sft_max_sharegpt_blocks",
        )
        if len(max_sg_options) == 1:
            st.caption("Max blocks are fixed to match your minimum (only one valid choice).")
        turn_mix_ui = st.selectbox(
            "Block-count mix (after persona balance)",
            ["none", "uniform", TURN_MIX_PRESET_153025_LABEL],
            help=(
                "none: no length downsampling. uniform: equalize counts for block "
                f"counts 2/4/6/8. {TURN_MIX_PRESET_153025_LABEL}: fixed weights "
                f"{TURN_MIX_PRESET_153025_JSON}. Custom JSON below overrides this dropdown."
            ),
            key="build_sft_turn_mix_mode",
        )
        turn_mix_custom = st.text_input(
            "Custom block-mix JSON (overrides select if non-empty)",
            value="",
            help=(
                'Optional weights for block counts 2/4/6/8, e.g. {"2":1,"4":2,"6":1,"8":1}. '
                "If this box has any text, the dropdown above is ignored. CLI: --turn-mix."
            ),
            key="build_sft_turn_mix_json",
        )
        personas = st.multiselect(
            "Personas to emit",
            all_personas,
            default=all_personas,
            help=(
                "Each selected author yields **one** training row per session (their "
                "last line is the final `assistant`); others in the window are `user`. "
                "Unselected authors are never trained on."
            ),
        )
    with c2:
        val_frac = st.slider(
            "Validation fraction",
            0.0,
            0.30,
            0.10,
            0.01,
            help=(
                "Fraction of **sessions** (not samples) held out for validation. "
                "Splitting by session id guarantees zero context leakage from "
                "train into val."
            ),
        )
        seed = st.number_input(
            "Seed",
            value=0,
            step=1,
            help=(
                "Controls the validation split and the stratified round-robin "
                "used during balancing. Same seed = identical outputs."
            ),
        )

    st.markdown("##### Balancing")
    c1, c2, c3 = st.columns(3)
    with c1:
        balance_mode = st.selectbox(
            "Balance policy",
            ["median", "none", "min", "cap"],
            help=(
                "`median`: cap each persona at round(k × median). "
                "`none`: keep everything as-is. "
                "`min`: hard-cap every persona to the smallest count. "
                "`cap`: explicit fixed cap (use the field on the right)."
            ),
        )
    with c2:
        balance_k = st.slider(
            "Balance k (median)",
            0.5,
            5.0,
            1.5,
            0.1,
            help=(
                "Multiplier on the median persona-sample count; only used when "
                "policy = `median`. k=1.0 caps at the median exactly; higher k "
                "lets heavy personas keep more samples."
            ),
        )
    with c3:
        cap_n = st.number_input(
            "Explicit cap (policy=cap)",
            value=1000,
            step=100,
            help=(
                "Per-persona hard cap used when policy = `cap`. Ignored for all "
                "other policies."
            ),
        )

    _mix_custom = (turn_mix_custom or "").strip()
    if _mix_custom:
        turn_mix_eff = _mix_custom
    elif turn_mix_ui == TURN_MIX_PRESET_153025_LABEL:
        turn_mix_eff = TURN_MIX_PRESET_153025_JSON
    else:
        turn_mix_eff = turn_mix_ui
    max_sharegpt_eff: int | None = None if no_sg_cap else int(max_sg)

    _live_expanded = not st.session_state.get("ui_seen_build_sft_example", False)
    _render_sft_live_example(
        window_turns=int(window_turns),
        min_sharegpt_turns=int(min_sg),
        max_sharegpt_turns=max_sharegpt_eff,
        turn_mix=str(turn_mix_eff),
        personas=list(personas) if personas else ["alice", "bob", "carol"],
        val_frac=float(val_frac),
        seed=int(seed),
        balance_mode=str(balance_mode),
        balance_k=float(balance_k),
        cap_n=int(cap_n),
        expanded=_live_expanded,
    )

    submitted = st.button("Build SFT", type="primary")
    if not submitted or not personas:
        return

    policy = balance_mode if balance_mode != "cap" else f"cap:{int(cap_n)}"
    with st.spinner("Building samples..."):
        samples = build_samples(
            sessions,
            personas=personas,
            window_turns=int(window_turns),
            min_sharegpt_turns=int(min_sg),
            max_sharegpt_turns=max_sharegpt_eff,
        )
        samples, balance_report = balance_samples(
            samples, policy=policy, k=float(balance_k), seed=int(seed)
        )
        try:
            samples, turn_report = balance_turn_length(
                samples, policy=str(turn_mix_eff), seed=int(seed)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Invalid turn-mix: {exc}")
            return
        train, val = split_train_val(samples, val_frac=float(val_frac), seed=int(seed))
        turn_report = dataclasses.replace(
            turn_report,
            post_split_num_turns=post_split_num_turns_breakdown(train, val),
        )
        shuffle_samples(train, int(seed))
        shuffle_samples(val, int(seed))
    out_path = resolve_repo_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_train = write_samples(train, out_path / "train.jsonl")
    n_val = write_samples(val, out_path / "val.jsonl")
    (out_path / "balance_report.json").write_text(json.dumps(dataclasses.asdict(balance_report), indent=2), encoding="utf-8")
    (out_path / "turn_length_report.json").write_text(
        json.dumps(dataclasses.asdict(turn_report), indent=2), encoding="utf-8"
    )
    st.session_state["ui_seen_build_sft_example"] = True
    st.success(f"Wrote {n_train:,} train / {n_val:,} val samples to {out_path}.")
