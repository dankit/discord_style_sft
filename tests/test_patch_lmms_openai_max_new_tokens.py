from scripts.patch_lmms_openai_max_new_tokens import patch_text


def test_patch_text_removes_4096_min_double_quotes() -> None:
    s = 'x = min(request_gen_kwargs.get("max_new_tokens", 1024), 4096)'
    t, n = patch_text(s, ceiling=None)
    assert n == 1
    assert t == 'x = request_gen_kwargs.get("max_new_tokens", 1024)'
    assert "4096" not in t


def test_patch_text_removes_4096_min_single_quotes() -> None:
    s = "x = min(request_gen_kwargs.get('max_new_tokens', 1024), 4096)"
    t, n = patch_text(s, ceiling=None)
    assert n == 1
    assert "4096" not in t
    assert "min(" not in t


def test_patch_text_idempotent() -> None:
    s = 'x = request_gen_kwargs.get("max_new_tokens", 1024)'
    _, n = patch_text(s, ceiling=None)
    assert n == 0


def test_patch_text_respects_env_ceiling() -> None:
    t, n = patch_text(
        'm = min(request_gen_kwargs.get("max_new_tokens", 512), 4096)',
        ceiling=8000,
    )
    assert n == 1
    assert t == 'm = min(request_gen_kwargs.get("max_new_tokens", 512), 8000)'
