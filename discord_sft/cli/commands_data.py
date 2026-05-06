from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from discord_sft.analysis.heuristics import profile_heuristics, style_heuristics


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _cmd_eval_heuristics(args: argparse.Namespace) -> int:
    ref_path = Path(args.references)
    gen_path = Path(args.generated)
    refs = _read_lines(ref_path)
    gens = _read_lines(gen_path)
    if len(refs) != len(gens):
        sys.stderr.write(
            f"Line count mismatch: references={len(refs)} generated={len(gens)}\n"
        )
        return 1

    if args.profile and args.persona:
        profile_doc = json.loads(Path(args.profile).read_text(encoding="utf-8"))
        persona_profile = profile_doc.get("personas", {}).get(args.persona)
        if not persona_profile:
            sys.stderr.write(
                f"Persona '{args.persona}' not in profile file {args.profile}\n"
            )
            return 2
        stats = profile_heuristics(gens, refs, profile=persona_profile)
    else:
        stats = style_heuristics(gens, refs)
    sys.stdout.write(json.dumps(stats, indent=2) + "\n")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from discord_sft.data_prep.ingest import ingest_root

    fmt = args.format
    report = ingest_root(Path(args.source), Path(args.out), fmt=fmt)
    sys.stdout.write(json.dumps({"format": fmt, "per_folder": report}, indent=2) + "\n")
    return 0


def _cmd_curate(args: argparse.Namespace) -> int:
    import json as _json
    from discord_sft.data_prep.curate import (
        CurateReport,
        curate_messages,
        session_to_record,
    )
    from discord_sft.data_prep.ingest import Message, iter_folders, iter_messages

    source = Path(args.source)
    if not source.exists():
        sys.stderr.write(f"Source not found: {source}\n")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_path = out_dir / "sessions.jsonl"
    report_path = out_dir / "curate_report.json"

    report = CurateReport()
    total_sessions = 0
    near_dedup_thr = None if args.no_near_dedup else args.near_dedup_threshold
    cap = int(args.exact_turn_dup_cap)
    if cap < 1:
        sys.stderr.write("--exact-turn-dup-cap must be >= 1\n")
        return 2
    with sessions_path.open("w", encoding="utf-8") as f:
        for folder in iter_folders(source):
            messages: list[Message] = list(iter_messages(folder))
            sessions, report = curate_messages(
                messages,
                folder=folder.name,
                merge_gap_sec=args.merge_gap_sec,
                session_gap_min=args.session_gap_min,
                min_turns=args.min_turns,
                min_authors=args.min_authors,
                monologue_max_share=args.monologue_max_share,
                url_strip=args.strip_urls,
                pii_scrub=not args.no_pii_scrub,
                lang=args.lang,
                near_dedup_threshold=near_dedup_thr,
                dedupe_exact_turns=not args.no_dedupe_exact_turns,
                exact_turn_dup_cap=cap,
                report=report,
            )
            for s in sessions:
                f.write(_json.dumps(session_to_record(s), ensure_ascii=False) + "\n")
            total_sessions += len(sessions)

    report_path.write_text(
        _json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8"
    )
    sys.stdout.write(
        _json.dumps(
            {
                "sessions_written": total_sessions,
                "sessions_file": str(sessions_path),
                "report_file": str(report_path),
                "report": dataclasses.asdict(report),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _cmd_curate_sweep(args: argparse.Namespace) -> int:
    import json as _json
    from math import prod

    from discord_sft.data_prep.curate_sweep import (
        default_sweep_lists,
        iter_curate_sweep_rows,
    )

    source = Path(args.source)
    if not source.exists():
        sys.stderr.write(f"Source not found: {source}\n")
        return 1

    cap = int(args.exact_turn_dup_cap)
    if cap < 1:
        sys.stderr.write("--exact-turn-dup-cap must be >= 1\n")
        return 2

    near_dedup_thr = None if args.no_near_dedup else args.near_dedup_threshold

    sgm, mgs, mt, ma, mono = default_sweep_lists(
        session_gap_min=int(args.session_gap_min),
        merge_gap_sec=int(args.merge_gap_sec),
        min_turns=int(args.min_turns),
        min_authors=int(args.min_authors),
        monologue_max_share=float(args.monologue_max_share),
        sweep_session_gap_min=args.sweep_session_gap_min,
        sweep_merge_gap_sec=args.sweep_merge_gap_sec,
        sweep_min_turns=args.sweep_min_turns,
        sweep_min_authors=args.sweep_min_authors,
        sweep_monologue_max_share=args.sweep_monologue_max_share,
    )

    n_combo = prod(len(x) for x in (sgm, mgs, mt, ma, mono))
    if n_combo > int(args.max_combos):
        sys.stderr.write(
            f"Sweep size {n_combo} exceeds --max-combos={args.max_combos}. "
            "Narrow comma lists or raise --max-combos.\n"
        )
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for row in iter_curate_sweep_rows(
            source,
            session_gap_mins=sgm,
            merge_gap_secs=mgs,
            min_turns_list=mt,
            min_authors_list=ma,
            monologue_max_shares=mono,
            url_strip=bool(args.strip_urls),
            pii_scrub=not args.no_pii_scrub,
            lang=args.lang,
            near_dedup_threshold=near_dedup_thr,
            dedupe_exact_turns=not args.no_dedupe_exact_turns,
            exact_turn_dup_cap=cap,
        ):
            out_f.write(_json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    sys.stdout.write(
        _json.dumps(
            {
                "rows_written": n_written,
                "out_file": str(out_path),
                "grid": {
                    "session_gap_min": sgm,
                    "merge_gap_sec": mgs,
                    "min_turns": mt,
                    "min_authors": ma,
                    "monologue_max_share": mono,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _parse_max_sharegpt_turns(value: str) -> int | None:
    t = str(value).strip().lower()
    if t in ("none", "off", "0"):
        return None
    n = int(t, 10)
    if n < 0:
        return None
    return n


def _validate_sharegpt_turn_bounds(mn: int, mx: int | None) -> None:
    if mn < 2 or mn % 2 != 0:
        raise ValueError("--min-sharegpt-turns must be an even integer >= 2")
    if mx is not None:
        if mx < 2 or mx % 2 != 0:
            raise ValueError("--max-sharegpt-turns must be an even integer >= 2, or none")
        if mx < mn:
            raise ValueError("--max-sharegpt-turns must be >= --min-sharegpt-turns")


def _cmd_build_sft(args: argparse.Namespace) -> int:
    from discord_sft.data_prep.curate import record_to_session
    from discord_sft.data_prep.sft import (
        balance_samples,
        balance_turn_length,
        build_samples,
        post_split_num_turns_breakdown,
        shuffle_samples,
        split_train_val,
        write_samples,
    )

    sessions = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sessions.append(record_to_session(json.loads(line)))

    personas: list[str] | None = None
    if args.personas and args.personas != "all":
        personas = [p.strip() for p in args.personas.split(",") if p.strip()]

    max_sg = _parse_max_sharegpt_turns(str(args.max_sharegpt_turns))
    try:
        _validate_sharegpt_turn_bounds(int(args.min_sharegpt_turns), max_sg)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    samples = build_samples(
        sessions,
        personas=personas,
        window_turns=args.window_turns,
        min_sharegpt_turns=int(args.min_sharegpt_turns),
        max_sharegpt_turns=max_sg,
    )
    samples, balance_report = balance_samples(
        samples,
        policy=args.balance,
        k=args.balance_k,
        seed=args.seed,
    )
    try:
        samples, turn_report = balance_turn_length(
            samples, policy=str(args.turn_mix or "none"), seed=int(args.seed)
        )
    except (json.JSONDecodeError, ValueError) as exc:
        sys.stderr.write(f"turn-mix: {exc}\n")
        return 2
    train, val = split_train_val(samples, val_frac=args.val_frac, seed=args.seed)
    turn_report = dataclasses.replace(
        turn_report,
        post_split_num_turns=post_split_num_turns_breakdown(train, val),
    )
    shuffle_samples(train, int(args.seed))
    shuffle_samples(val, int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_samples(train, out_dir / "train.jsonl")
    n_val = write_samples(val, out_dir / "val.jsonl")

    report_path = out_dir / "balance_report.json"
    report_path.write_text(
        json.dumps(dataclasses.asdict(balance_report), indent=2), encoding="utf-8"
    )
    turn_path = out_dir / "turn_length_report.json"
    turn_path.write_text(
        json.dumps(dataclasses.asdict(turn_report), indent=2), encoding="utf-8"
    )

    sys.stdout.write(
        json.dumps(
            {
                "train": n_train,
                "val": n_val,
                "balance": dataclasses.asdict(balance_report),
                "turn_length": dataclasses.asdict(turn_report),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from discord_sft.analysis.tokstats import compare_tokenizers, load_hf_encoder
    from discord_sft.data_prep.sft import read_samples

    samples = read_samples(Path(args.input))
    if not args.tokenizer:
        def _ws_encode(text: str) -> list[int]:
            return [0] * len(text.split())

        encoders = {"whitespace": _ws_encode}
    else:
        encoders = {name: load_hf_encoder(name) for name in args.tokenizer}

    report = compare_tokenizers(samples, encoders)
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


def _cmd_validate_chat_template(args: argparse.Namespace) -> int:
    from discord_sft.analysis.tokstats import load_hf_tokenizer
    from discord_sft.training.data import validate_chat_template_dataset

    tok = load_hf_tokenizer(args.tokenizer)
    report = validate_chat_template_dataset(
        args.input,
        tok,
        enable_thinking=args.enable_thinking,
        max_samples=args.max_samples,
        max_errors=args.max_errors,
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    from discord_sft.analysis.fingerprint import build_profiles
    from discord_sft.data_prep.sft import read_samples

    samples = read_samples(Path(args.input))
    profiles = build_profiles(samples, top_k_ngrams=args.top_k)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "profiles_written": len(profiles.get("personas", {})),
                "out_file": str(out_path),
            },
            indent=2,
        )
        + "\n"
    )
    return 0
