from discord_sft.analysis.tokstats import (
    compare_tokenizer_configs,
    compare_tokenizers,
    compare_vocabs,
    diff_tokenizer_configs,
    probe_tokenization,
    render_sample_text,
    sample_unique_tokens,
    summarize_tokenizer_config,
    tokenize_counts,
)
from discord_sft.data_prep.sft import Sample


def _sample(persona: str, text: str) -> Sample:
    return Sample(
        system=f"You are {persona} chatting with other.",
        conversations=[
            {"from": "user", "value": "hello"},
            {"from": "assistant", "value": text},
        ],
        meta={"persona_id": persona, "session_id": f"{persona}-s"},
    )


def test_render_includes_system_and_turns():
    s = _sample("A", "hi there")
    rendered = render_sample_text(s)
    assert "You are A" in rendered
    assert "hello" in rendered
    assert "hi there" in rendered


def test_tokenize_counts_with_whitespace_encoder():
    samples = [_sample("A", "one two three"), _sample("B", "four five")]
    encode = lambda t: [0] * len(t.split())
    rep = tokenize_counts(samples, encode)
    assert rep["n_samples"] == 2
    # system (6 words) + hello (1) + body
    assert rep["total_tokens"] == (6 + 1 + 3) + (6 + 1 + 2)
    assert set(rep["per_persona_tokens"].keys()) == {"A", "B"}
    assert rep["per_persona_samples"]["A"] == 1


def test_compare_tokenizers_reports_delta():
    samples = [_sample("A", "one two three four")]
    encoders = {
        "char": lambda t: list(t),
        "word": lambda t: t.split(),
    }
    report = compare_tokenizers(samples, encoders)
    assert "tokenizers" in report
    assert "char" in report["tokenizers"]
    assert "word" in report["tokenizers"]
    assert "vs_char" in report
    # word count must be smaller than char count for the same sample.
    assert report["tokenizers"]["word"]["total_tokens"] < report["tokenizers"]["char"]["total_tokens"]
    assert report["vs_char"]["word"]["total_delta"] < 0


# ---------- vocab comparison ----------


class _FakeTok:
    """Minimal stand-in for an HF tokenizer used in pure-python tests."""

    def __init__(self, vocab: dict[str, int], space: str = "\u0120"):
        self._vocab = dict(vocab)
        self._space = space

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def tokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        for i, word in enumerate(text.split()):
            marker = self._space if i > 0 else ""
            candidate = marker + word
            if candidate in self._vocab or word in self._vocab:
                pieces.append(candidate if candidate in self._vocab else word)
            else:
                pieces.extend(marker + ch if j == 0 else ch for j, ch in enumerate(word))
        return pieces

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:  # noqa: ARG002
        out: list[int] = []
        for piece in self.tokenize(text):
            out.append(self._vocab.get(piece, 0))
        return out


def test_compare_vocabs_reports_overlap_and_jaccard():
    vocabs = {
        "a": ["hi", "bye", "lol", "imma"],
        "b": ["hi", "bye", "lmao", "ngl"],
    }
    report = compare_vocabs(vocabs)
    assert report["sizes"] == {"a": 4, "b": 4}
    pair = report["pairs"]["a__vs__b"]
    assert pair["intersection"] == 2
    assert pair["only_in_a"] == 2
    assert pair["only_in_b"] == 2
    assert pair["union"] == 6
    assert abs(pair["jaccard"] - (2 / 6)) < 1e-9


def test_sample_unique_tokens_filters_readable_and_strips_space_markers():
    vocabs = {
        "a": ["\u0120imma", "\u0120lol", "!!!", "ok", "\u0120hi"],
        "b": ["\u0120hi", "\u0120bye"],
    }
    uniq = sample_unique_tokens(vocabs, a="a", b="b", limit=10)
    assert "\u0120imma" in uniq
    assert "\u0120lol" in uniq
    assert "ok" in uniq
    # Non-readable punctuation-only token is filtered.
    assert "!!!" not in uniq


# ---------- tokenizer.json config comparison ----------


def _cfg(
    vocab: dict[str, int],
    merges: list[str] | None = None,
    added: list[dict] | None = None,
    *,
    model_type: str = "BPE",
    normalizer: dict | None = None,
    pre_tokenizer: dict | None = None,
    post_processor: dict | None = None,
    decoder: dict | None = None,
) -> dict:
    """Build a minimal tokenizer.json-shaped dict for tests."""
    return {
        "version": "1.0",
        "added_tokens": added or [],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": {"type": model_type, "vocab": dict(vocab), "merges": list(merges or [])},
    }


def test_diff_identical_configs_reports_all_identical():
    cfg = _cfg(
        {"hi": 0, "bye": 1, "ok": 2},
        merges=["h i", "b y"],
        added=[{"id": 0, "content": "<s>", "special": True}],
        normalizer={"type": "NFC"},
        pre_tokenizer={"type": "ByteLevel"},
    )
    diff = diff_tokenizer_configs(cfg, cfg, name_a="x", name_b="y")
    assert diff["vocab"]["identical"]
    assert diff["vocab"]["id_mismatches"] == 0
    assert diff["vocab"]["only_in_x"] == 0
    assert diff["ordering"]["identical"]
    assert diff["ordering"]["first_diff"] is None
    assert diff["merges"]["identical"]
    assert diff["added_tokens"]["identical"]
    assert diff["model_type_equal"]
    assert diff["normalizer_equal"]
    assert diff["pre_tokenizer_equal"]


def test_diff_detects_id_mismatch_with_same_token_set():
    a = _cfg({"hi": 0, "bye": 1, "ok": 2})
    b = _cfg({"hi": 0, "bye": 2, "ok": 1})
    diff = diff_tokenizer_configs(a, b, name_a="a", name_b="b")
    # same keys, so set difference is empty...
    assert diff["vocab"]["only_in_a"] == 0
    assert diff["vocab"]["only_in_b"] == 0
    # ...but ids disagree, so vocab is not identical.
    assert not diff["vocab"]["identical"]
    assert diff["vocab"]["id_mismatches"] == 2
    tokens = {row["token"] for row in diff["vocab"]["id_mismatches_sample"]}
    assert tokens == {"bye", "ok"}
    assert not diff["ordering"]["identical"]
    assert diff["ordering"]["first_diff"]["id"] == 1


def test_diff_detects_tokens_only_in_one_vocab():
    a = _cfg({"hi": 0, "bye": 1})
    b = _cfg({"hi": 0, "bye": 1, "lol": 2})
    diff = diff_tokenizer_configs(a, b, name_a="a", name_b="b")
    assert diff["vocab"]["only_in_b"] == 1
    assert diff["vocab"]["only_in_b_sample"] == ["lol"]
    assert not diff["vocab"]["identical"]


def test_diff_detects_merges_and_component_differences():
    a = _cfg(
        {"hi": 0, "bye": 1},
        merges=["h i", "b y"],
        normalizer={"type": "NFC"},
        pre_tokenizer={"type": "ByteLevel", "add_prefix_space": False},
    )
    b = _cfg(
        {"hi": 0, "bye": 1},
        merges=["b y", "h i"],  # reordered
        normalizer={"type": "NFKC"},  # different
        pre_tokenizer={"type": "ByteLevel", "add_prefix_space": True},  # different cfg
    )
    diff = diff_tokenizer_configs(a, b, name_a="a", name_b="b")
    assert not diff["merges"]["identical"]
    assert diff["merges"]["first_diff"]["rank"] == 0
    assert not diff["normalizer_equal"]
    assert not diff["pre_tokenizer_equal"]


def test_diff_detects_added_token_differences():
    a = _cfg(
        {"a": 0},
        added=[{"id": 0, "content": "<s>", "special": True, "lstrip": False}],
    )
    b = _cfg(
        {"a": 0},
        added=[
            {"id": 0, "content": "<s>", "special": True, "lstrip": True},  # mismatch
            {"id": 1, "content": "</s>", "special": True},  # only in b
        ],
    )
    diff = diff_tokenizer_configs(a, b, name_a="a", name_b="b")
    assert not diff["added_tokens"]["identical"]
    assert diff["added_tokens"]["only_in_b"] == ["</s>"]
    mismatches = diff["added_tokens"]["mismatches"]
    assert len(mismatches) == 1
    assert mismatches[0]["content"] == "<s>"


def test_summarize_tokenizer_config_extracts_block_types():
    cfg = _cfg(
        {"a": 0, "b": 1},
        merges=["a b"],
        added=[{"id": 0, "content": "<s>"}],
        normalizer={"type": "NFC"},
        pre_tokenizer={"type": "ByteLevel"},
        decoder={"type": "ByteLevel"},
    )
    summary = summarize_tokenizer_config(cfg)
    assert summary["vocab_size"] == 2
    assert summary["merges"] == 1
    assert summary["added_tokens"] == 1
    assert summary["normalizer"] == "NFC"
    assert summary["pre_tokenizer"] == "ByteLevel"
    assert summary["decoder"] == "ByteLevel"
    assert summary["post_processor"] is None


def test_compare_tokenizer_configs_does_pairwise():
    a = _cfg({"hi": 0, "bye": 1})
    b = _cfg({"hi": 0, "bye": 1})
    c = _cfg({"hi": 1, "bye": 0})
    report = compare_tokenizer_configs({"a": a, "b": b, "c": c})
    assert set(report["per_tokenizer"].keys()) == {"a", "b", "c"}
    assert report["pairs"]["a__vs__b"]["vocab"]["identical"]
    assert not report["pairs"]["a__vs__c"]["vocab"]["identical"]
    assert not report["pairs"]["a__vs__c"]["ordering"]["identical"]


def test_probe_tokenization_shows_per_tokenizer_counts():
    # Tokenizer "big" knows "imma" natively; tokenizer "small" does not.
    big = _FakeTok({"\u0120imma": 1, "\u0120hi": 2, "hi": 3, "imma": 4})
    small = _FakeTok({"\u0120hi": 2, "hi": 3})
    rows = probe_tokenization({"big": big, "small": small}, ["hi imma"])
    row = rows[0]
    assert row["probe"] == "hi imma"
    # "big" has native imma -> 2 pieces; "small" must fall back to chars -> >2.
    assert row["big_count"] == 2
    assert row["small_count"] > row["big_count"]
    # Normalized pieces should no longer carry the Ġ space marker.
    assert all("\u0120" not in p for p in row["big_pieces"])
