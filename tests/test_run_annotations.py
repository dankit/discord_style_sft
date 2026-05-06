from __future__ import annotations

from pathlib import Path

from discord_sft.ui.pages.evals.run_annotations import (
    annotations_file_path,
    default_annotation_doc,
    format_compare_column_header,
    format_eval_run_label,
    load_annotations_file,
    merge_run_annotation_rows,
    save_annotations_file,
    truncate_run_id,
)


def test_truncate_run_id() -> None:
    s = "a" * 50
    t = truncate_run_id(s, max_chars=48)
    assert len(t) == 48
    assert t.endswith("…")


def test_format_eval_run_label_prefers_alias_and_elo() -> None:
    ann = {"version": 1, "runs": {"rid-1": {"alias": "My run", "notes": "n"}}}
    s = format_eval_run_label(
        "rid-1",
        yaml_label="yaml_fallback",
        elo=1543.2,
        annotations=ann,
    )
    assert "My run" in s
    assert "Elo 1543.2" in s


def test_format_eval_run_label_truncates_plain_run_id() -> None:
    long_id = "x" * 60
    s = format_eval_run_label(long_id, annotations={"version": 1, "runs": {}})
    assert "…" in s
    assert len(s) <= 80


def test_format_compare_column_header() -> None:
    ann = {"version": 1, "runs": {"a": {"alias": "ShortNick", "notes": ""}}}
    assert format_compare_column_header("a", yaml_label=None, annotations=ann) == "ShortNick"


def test_load_save_round_trip_atomic(tmp_path: Path) -> None:
    path = annotations_file_path(tmp_path)
    assert path.name == "run_annotations.json"
    doc_in = default_annotation_doc()
    doc_in["runs"] = {"r1": {"alias": "one", "notes": "hello"}}
    save_annotations_file(path, doc_in)
    doc_out = load_annotations_file(path)
    assert doc_out["runs"]["r1"]["alias"] == "one"
    assert doc_out["runs"]["r1"]["notes"] == "hello"


def test_merge_preserves_other_runs() -> None:
    base = {"version": 1, "runs": {"keep": {"alias": "stay", "notes": ""}}}
    rows = [{"run_id": "new", "alias": "n", "notes": "note"}]
    merged = merge_run_annotation_rows(base, rows)
    assert merged["runs"]["keep"]["alias"] == "stay"
    assert merged["runs"]["new"]["alias"] == "n"
    assert merged["runs"]["new"]["notes"] == "note"


def test_invalid_json_returns_default(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{", encoding="utf-8")
    assert load_annotations_file(bad) == default_annotation_doc()
