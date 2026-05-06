from discord_sft.ui.common import (
    default_discord_export_root,
    default_work_dir,
    load_sessions,
    repo_root,
    resolve_repo_path,
    resolve_work_dir,
)


def test_resolve_work_dir_relative_out_is_repo_out() -> None:
    assert resolve_work_dir("out") == (repo_root() / "out").resolve()


def test_resolve_work_dir_empty_defaults_to_repo_out() -> None:
    assert resolve_work_dir("") == (repo_root() / "out").resolve()
    assert resolve_work_dir("   ") == (repo_root() / "out").resolve()


def test_default_discord_export_root_is_repo_discord_messages() -> None:
    assert default_discord_export_root() == "discord_messages"
    assert resolve_repo_path(default_discord_export_root()) == (
        repo_root() / "discord_messages"
    ).resolve()


def test_default_work_dir_is_relative_out() -> None:
    assert default_work_dir() == "out"


def test_resolve_repo_path_anchors_relative_values_at_repo_root() -> None:
    assert resolve_repo_path("out/sft/train.jsonl") == (
        repo_root() / "out" / "sft" / "train.jsonl"
    ).resolve()


def test_load_sessions_skips_malformed_rows(tmp_path, monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr("discord_sft.ui.common.st.warning", warnings.append)
    path = tmp_path / "sessions.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"session_id":"ok","folder":"dm","turns":[{"author_id":"A","author_name":"Alice","start_ts":"2022-12-27T19:23:49.528000+00:00","end_ts":"2022-12-27T19:23:50.528000+00:00","text":"hi","source_ids":["1"]}]}',
                '{"session_id":"bad","folder":"dm","turns":[{"author_id":"A","author_name":"Alice","start_ts":"not-a-date","end_ts":"2022-12-27T19:23:50.528000+00:00","text":"oops","source_ids":["2"]}]}',
            ]
        ),
        encoding="utf-8",
    )

    sessions = load_sessions(path)

    assert [s.id for s in sessions] == ["ok"]
    assert warnings
