from __future__ import annotations

from pathlib import Path

from discord_sft.training import load_config
from discord_sft.training.lora_search.gradient_probe import (
    IGNORE_INDEX,
    _aggregate_gradient_records,
    _build_parser,
    _num_hidden_layers,
    _single_output_path,
    aggregate_gradient_norms,
    _build_response_only_labels,
)
from discord_sft.training.lora_search.visualize_probe import (
    _annotate_if_qwen35_a3b,
    _discover_probe_files,
    _qwen35_a3b_layer_annotation,
    _run_label,
    _score_probe,
)


class _Scalar:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _Grad:
    def __init__(self, norm_value: float):
        self._norm_value = norm_value

    def norm(self) -> _Scalar:
        return _Scalar(self._norm_value)


class _Param:
    def __init__(self, norm_value: float):
        self.grad = _Grad(norm_value)


class _Obj:
    pass


def test_aggregate_gradient_norms_ranks_modules_and_layers():
    named_parameters = [
        ("model.layers.1.self_attn.q_proj.lora_A.default.weight", _Param(1.5)),
        ("model.layers.1.mlp.down_proj.lora_A.default.weight", _Param(4.0)),
        ("model.layers.20.self_attn.o_proj.lora_B.default.weight", _Param(3.0)),
        ("model.layers.20.mlp.down_proj.lora_B.default.weight", _Param(2.0)),
    ]

    agg = aggregate_gradient_norms(named_parameters)

    assert agg["modules_ranked"][0]["module"] == "down_proj"
    assert agg["layers_ranked"][0]["layer"] == 1
    assert agg["params_ranked"][0]["name"].endswith("down_proj.lora_A.default.weight")
    assert agg["params_ranked"][0]["grad_norm"] == 4.0


def test_aggregate_gradient_records_combines_probe_steps():
    records = [
        {"name": "model.layers.1.self_attn.o_proj.lora_A.weight", "grad_norm": 1.0, "layer": 1, "module": "o_proj"},
        {"name": "model.layers.1.self_attn.o_proj.lora_A.weight", "grad_norm": 3.0, "layer": 1, "module": "o_proj"},
        {"name": "model.layers.2.mlp.down_proj.lora_A.weight", "grad_norm": 2.0, "layer": 2, "module": "down_proj"},
    ]

    agg = _aggregate_gradient_records(records)

    assert agg["modules_ranked"][0]["module"] == "o_proj"
    assert agg["modules_ranked"][0]["sum"] == 4.0
    assert agg["modules_ranked"][0]["count"] == 2.0


def test_canonical_training_configs_load():
    root = Path(__file__).resolve().parents[1]
    style = load_config(root / "discord_sft" / "training" / "configs" / "qwen35_a3b_style_late.yaml")
    full = load_config(root / "discord_sft" / "training" / "configs" / "qwen35_a3b_full_r16.yaml")

    assert style.run_name
    assert full.run_name
    assert style.lora.r == 16
    assert style.lora.alpha == 16
    assert style.lora.layers_last_pct == 0.25
    assert full.lora.r == 16
    assert full.lora.alpha == 16
    assert full.lora.layers_last_pct is None


def test_response_only_labels_mask_user_tokens():
    instruction_ids = [10, 11]
    response_ids = [20, 21]
    input_ids = [10, 11, 1, 1, 20, 21, 7, 8, 10, 11, 2, 20, 21, 9]
    attention_mask = [1] * len(input_ids)

    labels, supervised = _build_response_only_labels(
        input_ids,
        attention_mask,
        instruction_ids=instruction_ids,
        response_ids=response_ids,
    )

    assert supervised == 3
    assert labels[0] == IGNORE_INDEX
    assert labels[2] == IGNORE_INDEX
    assert labels[6] == 7
    assert labels[7] == 8
    assert labels[12] == IGNORE_INDEX
    assert labels[13] == 9


def test_qwen35_a3b_probe_notes_forty_blocks_zero_indexed():
    assert "40" in _qwen35_a3b_layer_annotation()
    assert "0" in _qwen35_a3b_layer_annotation()
    assert (
        _annotate_if_qwen35_a3b(
            [(Path("g.json"), {"base_model": "unsloth/Qwen3.5-35B-A3B"})]
        )
        == _qwen35_a3b_layer_annotation()
    )
    assert (
        _annotate_if_qwen35_a3b([(Path("g.json"), {"base_model": "Qwen/Qwen2.5-7B"})]) is None
    )


def test_probe_scoring_prefers_stable_improving_signal():
    probe = {
        "aggregate": {
            "layers_ranked": [{"layer": i, "sum": 1.0 / (i + 1)} for i in range(8)],
            "modules_ranked": [
                {"module": "down_proj", "sum": 3.0},
                {"module": "o_proj", "sum": 2.0},
                {"module": "v_proj", "sum": 1.0},
            ],
        },
        "step_records": [
            {"top_modules": [{"module": "down_proj"}]},
            {"top_modules": [{"module": "down_proj"}]},
            {"top_modules": [{"module": "down_proj"}]},
        ],
        "supervised_tokens": {"mean_per_step": 128},
        "loss": {"first": 2.0, "last": 1.6},
    }
    scored = _score_probe(probe)
    assert scored["score"] > 0
    assert scored["stable_top_module"] == 1.0
    assert scored["loss_delta"] > 0


def test_gradient_probe_parser_default_output():
    args = _build_parser().parse_args(["--config", "probe.yaml"])

    assert args.output == "out/lora/probes/gradient-probe"
    assert args.lora_r is None and args.lora_alpha is None


def test_gradient_probe_parser_lora_overrides():
    args = _build_parser().parse_args(
        ["--config", "probe.yaml", "--lora-r", "64", "--lora-alpha", "128"]
    )
    assert args.lora_r == 64
    assert args.lora_alpha == 128


def test_single_output_path_accepts_directory_or_file(tmp_path):
    assert _single_output_path(tmp_path / "probe").name == "gradient_norms.json"
    assert _single_output_path(tmp_path / "probe.json").name == "probe.json"


def test_num_hidden_layers_supports_nested_model_configs():
    model = _Obj()
    model.config = _Obj()
    model.config.text_config = _Obj()
    model.config.text_config.num_hidden_layers = 48

    assert _num_hidden_layers(model) == 48


def test_discover_probe_files_includes_sweep_outputs(tmp_path):
    single = tmp_path / "gradient_norms.json"
    sweep = tmp_path / "sweep" / "gradient_norms_r16_alpha32.json"
    index = tmp_path / "gradient_probe_sweep_index.json"
    sweep.parent.mkdir()
    single.write_text("{}", encoding="utf-8")
    sweep.write_text("{}", encoding="utf-8")
    index.write_text("{}", encoding="utf-8")

    discovered = {p.name for p in _discover_probe_files(tmp_path)}

    assert discovered == {"gradient_norms.json", "gradient_norms_r16_alpha32.json"}


def test_run_label_distinguishes_rank_alpha_sweep_outputs(tmp_path):
    path = tmp_path / "gradient_norms_r16_alpha32.json"
    probe = {
        "config_path": "discord_sft/training/configs/qwen35_a3b_full_r16.yaml",
        "lora": {"r": 16, "alpha": 32},
    }

    assert _run_label(path, probe) == "qwen35_a3b_full_r16-r16-a32"
