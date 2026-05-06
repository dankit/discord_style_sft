"""Tests for :mod:`scripts.patch_lmms_openai_disable_thinking`."""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "patch_lmms_openai_disable_thinking.py"


@pytest.fixture(scope="module")
def thinking_patch():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tg", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_ensure_import_os_inserts_after_future(thinking_patch) -> None:
    gated = thinking_patch._GATED_SNIPPET
    blob = (
        'from __future__ import annotations\n\n'
        f'{gated}\n'
        '    payload["extra_body"] = {{}}\n'
    )
    fixed, touched = thinking_patch.ensure_import_os_for_gated_env(blob)
    assert touched is True
    assert "import os\n" in fixed
    assert fixed.index("import os") < fixed.index("if os.environ")


def test_ensure_import_os_skips_when_present(thinking_patch) -> None:
    gated = thinking_patch._GATED_SNIPPET
    blob = "import os\n\n" + gated + "\n"
    fixed, touched = thinking_patch.ensure_import_os_for_gated_env(blob)
    assert fixed == blob and touched is False
