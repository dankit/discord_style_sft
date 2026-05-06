from __future__ import annotations

import json
from pathlib import Path

from discord_sft.evals.storage import save_run
from discord_sft.ui.pages.evals.style_rank_groups import (
    compat_group_key,
    discover_saved_run_dumps,
    group_dumps,
    infer_run_json_from_generations_path,
    peek_first_jsonl_row,
    scan_persona_generations_jsonl,
    serialize_group_key,
)


def test_embedded_eval_val_overrides_run_config(tmp_path: Path) -> None:
    new_val = tmp_path / "sft" / "val.jsonl"
    old_val = tmp_path / "sft" / "old" / "val.jsonl"
    new_val.parent.mkdir(parents=True)
    old_val.parent.mkdir(parents=True)
    new_val.write_text("{}\n", encoding="utf-8")
    old_val.write_text("{}\n", encoding="utf-8")
    run_doc = {"config": {"val_jsonl": str(new_val)}}
    first = {"eval_val_jsonl": str(old_val)}
    kind, val, _hint = compat_group_key(
        run_doc,
        first,
        generations_path=tmp_path / "g.jsonl",
        eval_set_fingerprint="unused",
        n_rows=1,
    )
    assert kind == "path"
    assert Path(val).resolve() == old_val.resolve()


def test_embedded_sha_overrides_run_config(tmp_path: Path) -> None:
    run_doc = {"config": {"val_jsonl": str(tmp_path / "manifest_only.jsonl")}}
    first = {"eval_val_sha256": "deadbeef" * 8, "eval_val_jsonl": str(tmp_path / "row.jsonl")}
    kind, val, _hint = compat_group_key(
        run_doc,
        first,
        generations_path=tmp_path / "g.jsonl",
        eval_set_fingerprint="unused",
        n_rows=1,
    )
    assert kind == "valsha"
    assert val == "deadbeef" * 8


def test_compat_sha256_from_row(tmp_path: Path) -> None:
    gens = tmp_path / "x.jsonl"
    first = {"eval_val_sha256": "deadbeef" * 8}
    kind, val, hint = compat_group_key(
        None, first, generations_path=gens, eval_set_fingerprint="x", n_rows=1
    )
    assert kind == "valsha"
    assert val == "deadbeef" * 8
    assert "sha256" in hint


def test_evalset_splits_when_manifest_path_is_same(tmp_path: Path) -> None:
    """Same ``config.val_jsonl`` string but different eval rows → different ``evalset`` groups."""
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("{}\n", encoding="utf-8")
    run_doc = {"config": {"val_jsonl": str(val_path)}}

    def _row(ref: str) -> dict:
        return {
            "persona_id": "1",
            "reference": ref,
            "generated": "g",
            "context_turns": [{"from": "user", "value": "u"}],
        }

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("old-ref")) + "\n", encoding="utf-8")
    b.write_text(json.dumps(_row("new-ref")) + "\n", encoding="utf-8")
    _fa, fpa, na = scan_persona_generations_jsonl(a)
    _fb, fpb, nb = scan_persona_generations_jsonl(b)
    assert fpa != fpb
    ka, va, _ = compat_group_key(run_doc, _fa, generations_path=a, eval_set_fingerprint=fpa, n_rows=na)
    kb, vb, _ = compat_group_key(run_doc, _fb, generations_path=b, eval_set_fingerprint=fpb, n_rows=nb)
    assert ka == "evalset" and kb == "evalset"
    assert va != vb


def test_compat_unknown(tmp_path: Path) -> None:
    gens = tmp_path / "orphan.jsonl"
    kind, val, _h = compat_group_key(
        None, None, generations_path=gens, eval_set_fingerprint="", n_rows=0
    )
    assert kind == "unknown"
    assert len(val) == 16


def test_serialize_group_key_roundtrip() -> None:
    s = serialize_group_key("path", "/tmp/val.jsonl")
    assert "\x1f" in s


def test_infer_run_json_from_generations_path(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    raw_foo = eval_root / "raw" / "myrunid"
    raw_foo.mkdir(parents=True)
    gens = raw_foo / "persona_generations.jsonl"
    gens.write_text('{"x":1}\n', encoding="utf-8")
    runs_dir = eval_root / "runs"
    runs_dir.mkdir(parents=True)
    rj = runs_dir / "myrunid.json"
    rj.write_text('{"run_id":"myrunid"}\n', encoding="utf-8")
    assert infer_run_json_from_generations_path(gens) == rj.resolve()


def test_discover_and_group(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    raw = eval_root / "raw"
    runs_dir = eval_root / "runs"
    runs_dir.mkdir(parents=True)
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("{}\n", encoding="utf-8")

    def _write_run(rid: str, label: str) -> None:
        gdir = raw / rid
        gdir.mkdir(parents=True)
        gp = gdir / "persona_generations.jsonl"
        gp.write_text(
            json.dumps(
                {
                    "persona_id": "1",
                    "reference": "r",
                    "generated": "g",
                    "context_turns": [{"from": "user", "value": "u"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = {
            "run_id": rid,
            "label": label,
            "created_utc": f"2026-01-0{rid[-1]}T00:00:00Z" if rid[-1].isdigit() else "2026-01-10T00:00:00Z",
            "config": {"val_jsonl": str(val_path)},
            "persona": {"generations_path": str(gp.resolve())},
        }
        save_run(run, eval_root)

    _write_run("run_a", "A")
    _write_run("run_b", "B")

    dumps = discover_saved_run_dumps(eval_root)
    assert len(dumps) == 2
    groups = group_dumps(dumps)
    assert len(groups) == 1
    assert dumps[0].group_kind == "evalset"
    assert len(next(iter(groups.values()))) == 2


def test_discover_splits_same_manifest_different_slice(tmp_path: Path) -> None:
    eval_root = tmp_path / "evals"
    raw = eval_root / "raw"
    runs_dir = eval_root / "runs"
    runs_dir.mkdir(parents=True)
    val_path = tmp_path / "val.jsonl"
    val_path.write_text("{}\n", encoding="utf-8")

    def _write_run(rid: str, ref: str) -> None:
        gdir = raw / rid
        gdir.mkdir(parents=True)
        gp = gdir / "persona_generations.jsonl"
        gp.write_text(
            json.dumps(
                {
                    "persona_id": "1",
                    "reference": ref,
                    "generated": "g",
                    "context_turns": [{"from": "user", "value": "u"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        run = {
            "run_id": rid,
            "label": rid,
            "created_utc": "2026-01-01T00:00:00Z",
            "config": {"val_jsonl": str(val_path)},
            "persona": {"generations_path": str(gp.resolve())},
        }
        save_run(run, eval_root)

    _write_run("run_a", "ref-a")
    _write_run("run_b", "ref-b")
    groups = group_dumps(discover_saved_run_dumps(eval_root))
    assert len(groups) == 2


def test_peek_first_row(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text("\n\n{\"a\": 1}\n", encoding="utf-8")
    assert peek_first_jsonl_row(p) == {"a": 1}
