"""Visualize and rank LoRA gradient probe outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_probe(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_probe_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("**/gradient_norms*.json"))


def _run_label(path: Path, probe: dict[str, Any]) -> str:
    lora = probe.get("lora", {})
    config_path = probe.get("config_path")
    if isinstance(config_path, str) and config_path:
        base = Path(config_path).stem
    else:
        base = path.parent.name if path.name.startswith("gradient_norms") else path.stem

    parts = [base]
    r = lora.get("r")
    alpha = lora.get("alpha")
    if r is not None:
        parts.append(f"r{r}")
    if alpha is not None:
        parts.append(f"a{alpha}")
    return "-".join(parts)


def _gini(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(max(0.0, float(v)) for v in values)
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum) / (n * total) - (n + 1) / n


def _qwen35_a3b_layer_annotation() -> str:
    """HF/Unsloth Qwen3.5-35B-A3B configs use num_hidden_layers=40 → indices 0–39."""

    return "Architecture: Qwen3.5-35B-A3B · 40 transformer blocks (indices 0–39)."


def _annotate_if_qwen35_a3b(probes: list[tuple[Path, dict[str, Any]]]) -> str | None:
    for _, data in probes:
        bm = data.get("base_model")
        if isinstance(bm, str) and "35b-a3b" in bm.lower():
            return _qwen35_a3b_layer_annotation()
    return None


def _score_probe(probe: dict[str, Any]) -> dict[str, float]:
    agg = probe.get("aggregate", {})
    layers = agg.get("layers_ranked", [])
    modules = agg.get("modules_ranked", [])
    steps = probe.get("step_records", [])
    sup = probe.get("supervised_tokens", {})
    mean_sup = float(sup.get("mean_per_step", 0.0) or 0.0)
    loss = probe.get("loss", {})
    first = loss.get("first")
    last = loss.get("last")
    if first is None or last is None or float(first) == 0.0:
        loss_delta = 0.0
    else:
        loss_delta = (float(first) - float(last)) / abs(float(first))

    top_modules = [m.get("module") for m in modules[:3]]
    top_module_concentration = _gini([m.get("sum", 0.0) for m in modules])
    top_layer_concentration = _gini([l.get("sum", 0.0) for l in layers])

    stable_top_module = 0.0
    if steps:
        tops = [s.get("top_modules", [{}])[0].get("module") for s in steps if s.get("top_modules")]
        if tops:
            mode = max(set(tops), key=tops.count)
            stable_top_module = tops.count(mode) / len(tops)

    # Heuristic score, higher is better for a "clean signal" probe.
    # - prefers non-trivial supervised tokens
    # - prefers stable signal across steps
    # - prefers some concentration (clearer attribution), but not only one source
    # - prefers improving loss during short probe
    score = (
        0.30 * min(loss_delta, 1.0)
        + 0.30 * stable_top_module
        + 0.25 * min(top_layer_concentration, 0.8)
        + 0.15 * min(top_module_concentration, 0.8)
    )
    if mean_sup <= 0:
        score -= 1.0

    return {
        "score": float(score),
        "loss_delta": float(loss_delta),
        "stable_top_module": float(stable_top_module),
        "layer_concentration_gini": float(top_layer_concentration),
        "module_concentration_gini": float(top_module_concentration),
        "mean_supervised_tokens": float(mean_sup),
        "top_modules": top_modules,
    }


def _render_plots(probes: list[tuple[Path, dict[str, Any]]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        import seaborn as sns  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Plotting requires pandas, matplotlib, and seaborn. "
            'Install them (e.g. `pip install "discord-sft[probe_visualization]"`) '
            "or rerun with `--no-plots` to write only probe_ranking.json."
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    layer_note = _annotate_if_qwen35_a3b(probes)

    # 1) Layer contributions per run (line plot)
    layer_rows: list[dict[str, Any]] = []
    for p, data in probes:
        run = data.get("lora", {}).get("target_modules", [])
        run_label = _run_label(p, data)
        for row in data.get("aggregate", {}).get("layers_ranked", []):
            layer_rows.append(
                {
                    "run": run_label,
                    "layer": int(row["layer"]),
                    "sum_grad_norm": float(row["sum"]),
                    "modules": ",".join(run),
                }
            )
    if layer_rows:
        df = pd.DataFrame(layer_rows)
        plt.figure(figsize=(11, 5))
        sns.lineplot(data=df, x="layer", y="sum_grad_norm", hue="run", marker="o")
        if layer_note:
            plt.title(f"Gradient contribution per layer\n{layer_note}", fontsize=10)
        else:
            plt.title("Gradient contribution per layer")
        plt.xlabel("Block index")
        plt.tight_layout()
        plt.savefig(out_dir / "layer_contrib.png", dpi=180, bbox_inches="tight")
        plt.close()

    # 2) Module contributions per run (grouped bar)
    mod_rows: list[dict[str, Any]] = []
    for p, data in probes:
        run_label = _run_label(p, data)
        for row in data.get("aggregate", {}).get("modules_ranked", []):
            mod_rows.append(
                {
                    "run": run_label,
                    "module": str(row["module"]),
                    "sum_grad_norm": float(row["sum"]),
                }
            )
    if mod_rows:
        df = pd.DataFrame(mod_rows)
        plt.figure(figsize=(12, 5))
        sns.barplot(data=df, x="module", y="sum_grad_norm", hue="run")
        plt.xticks(rotation=30, ha="right")
        if layer_note:
            plt.title(f"Gradient contribution by module\n{layer_note}", fontsize=10)
        else:
            plt.title("Gradient contribution by module")
        plt.tight_layout()
        plt.savefig(out_dir / "module_contrib.png", dpi=180, bbox_inches="tight")
        plt.close()

    # 3) Heatmap for best-scoring run (layer x module)
    scored = [(p, d, _score_probe(d)) for p, d in probes]
    if scored:
        best_path, best_probe, _ = sorted(scored, key=lambda x: x[2]["score"], reverse=True)[0]
        hm_rows = []
        for row in best_probe.get("aggregate", {}).get("layer_module_ranked", []):
            hm_rows.append(
                {
                    "layer": int(row["layer"]),
                    "module": str(row["module"]),
                    "sum_grad_norm": float(row["sum"]),
                }
            )
        if hm_rows:
            df = pd.DataFrame(hm_rows)
            piv = df.pivot(index="layer", columns="module", values="sum_grad_norm").fillna(0.0)
            piv = piv.sort_index(axis=0)
            piv = piv.sort_index(axis=1)
            plt.figure(figsize=(10, 7))
            # robust=True trims color limits to ~2–98th pct. so outliers do not flatten contrast
            sns.heatmap(
                piv,
                cmap="rocket",
                robust=True,
                linewidths=0.35,
                linecolor="white",
                cbar_kws={
                    "label": "Σ grad norms (within robust percentile color range)",
                    "shrink": 0.78,
                },
            )
            hm_lines = [
                f"Layer x module heatmap — best run: {best_path.parent.name}",
            ]
            if layer_note:
                hm_lines.append(layer_note)
            hm_lines.append("Color scale: robust percentile band (warm = stronger).")
            plt.title("\n".join(hm_lines), fontsize=10)
            plt.xlabel("Projection / expert block")
            plt.ylabel("Block index (row)")
            plt.tight_layout()
            plt.savefig(out_dir / "layer_module_heatmap_best.png", dpi=180, bbox_inches="tight")
            plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m discord_sft.training.lora_search.visualize_probe",
        description="Visualize and rank gradient probe outputs.",
    )
    parser.add_argument(
        "--input",
        default="out/lora/probes",
        help="Probe JSON file or directory containing gradient_norms.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default="out/lora/probes/analysis",
        help="Directory for PNGs and ranking JSON.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and only emit ranking JSON.",
    )
    args = parser.parse_args(argv)

    probe_paths = _discover_probe_files(Path(args.input))
    if not probe_paths:
        raise FileNotFoundError(f"No probe files found under: {args.input}")

    probes = [(p, _load_probe(p)) for p in probe_paths]
    scored = []
    for path, probe in probes:
        metrics = _score_probe(probe)
        scored.append(
            {
                "probe_path": str(path.resolve()),
                "run_dir": str(path.parent.resolve()),
                "run_label": _run_label(path, probe),
                **metrics,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = out_dir / "probe_ranking.json"
    ranking_path.write_text(json.dumps({"ranked": scored}, indent=2), encoding="utf-8")

    if not args.no_plots:
        _render_plots(probes, out_dir)

    print(
        json.dumps(
            {
                "ranking_json": str(ranking_path.resolve()),
                "winner": scored[0] if scored else None,
                "num_runs": len(scored),
                "plots_dir": str(out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
