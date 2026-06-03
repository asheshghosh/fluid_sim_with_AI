from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "fluid_ai_sim_matplotlib"
mpl_config = cache_root / "mpl"
xdg_cache = cache_root / "xdg"
mpl_config.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_metadata(run_dir: Path, mode: str) -> dict:
    return _load_json(run_dir / mode / "metadata.json")


def make_plot(stride_run: Path, fno_run: Path, out: Path) -> None:
    stride_metrics = _load_json(stride_run / "comparison_metrics.json")
    fno_metrics = _load_json(fno_run / "comparison_metrics.json")

    stride_meta = {mode: _mode_metadata(stride_run, mode) for mode in ["solver", "ai", "hybrid"]}
    fno_meta = {mode: _mode_metadata(fno_run, mode) for mode in ["solver", "ai", "hybrid"]}

    grid_size = int(fno_meta["solver"]["solver_config"]["n"])
    steps = int(fno_meta["solver"]["steps"])
    stride = int(fno_meta["ai"].get("surrogate_step_size", 1))
    correction_interval = int(_load_json(fno_run / "metadata.json").get("correction_interval", 0))

    solver_speed = (
        float(stride_meta["solver"]["steps_per_second"]) + float(fno_meta["solver"]["steps_per_second"])
    ) / 2.0
    speed_items = [
        ("Exact solver", solver_speed, "#2563eb"),
        ("CNN stride-8 AI", float(stride_meta["ai"]["steps_per_second"]), "#dc2626"),
        ("CNN stride-8 hybrid", float(stride_meta["hybrid"]["steps_per_second"]), "#0f766e"),
        ("FNO stride-8 AI", float(fno_meta["ai"]["steps_per_second"]), "#7c3aed"),
        ("FNO stride-8 hybrid", float(fno_meta["hybrid"]["steps_per_second"]), "#0891b2"),
    ]
    error_items = [
        ("CNN stride-8 AI", float(stride_metrics["ai"]["final_relative_l2"]), "#dc2626"),
        ("CNN stride-8 hybrid", float(stride_metrics["hybrid"]["final_relative_l2"]), "#0f766e"),
        ("FNO stride-8 AI", float(fno_metrics["ai"]["final_relative_l2"]), "#7c3aed"),
        ("FNO stride-8 hybrid", float(fno_metrics["hybrid"]["final_relative_l2"]), "#0891b2"),
    ]

    fig, (speed_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(10.6, 7.6),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.42},
    )
    fig.patch.set_facecolor("white")

    labels = [item[0] for item in speed_items]
    speeds = [item[1] for item in speed_items]
    colors = [item[2] for item in speed_items]
    x = list(range(len(speed_items)))
    bars = speed_axis.bar(x, speeds, color=colors, width=0.62)
    for bar, speed in zip(bars, speeds):
        speed_axis.annotate(
            f"{speed:,.0f}\n{speed / solver_speed:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#111827",
        )
    speed_axis.axhline(solver_speed, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.75)
    speed_axis.set_title("N=128 FNO Rollout vs Strided CNN Surrogate", fontsize=15, pad=14)
    speed_axis.set_ylabel("Solver-equivalent steps/sec")
    speed_axis.set_xticks(x)
    speed_axis.set_xticklabels(labels, rotation=12, ha="right")
    speed_axis.set_ylim(0.0, max(speeds) * 1.24)
    speed_axis.grid(True, axis="y", alpha=0.25)
    speed_axis.spines["top"].set_visible(False)
    speed_axis.spines["right"].set_visible(False)

    error_labels = [item[0] for item in error_items]
    errors = [item[1] for item in error_items]
    error_colors = [item[2] for item in error_items]
    ex = list(range(len(error_items)))
    error_bars = error_axis.bar(ex, errors, color=error_colors, width=0.62)
    for bar, error in zip(error_bars, errors):
        error_axis.annotate(
            f"{error:.3g}",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#111827",
        )
    error_axis.set_yscale("log")
    error_axis.set_ylabel("Final relative L2 error")
    error_axis.set_xticks(ex)
    error_axis.set_xticklabels(error_labels, rotation=12, ha="right")
    error_axis.grid(True, axis="y", which="both", alpha=0.25)
    error_axis.spines["top"].set_visible(False)
    error_axis.spines["right"].set_visible(False)

    note = (
        f"n={grid_size}, {steps} solver-equivalent steps, stride={stride}, "
        f"correction interval={correction_interval}; speed baseline is the mean exact-solver timing "
        f"from both comparison runs."
    )
    fig.text(0.01, 0.012, note, fontsize=8.5, color="#4b5563")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FNO-vs-strided-CNN n=128 comparison SVG.")
    parser.add_argument("--stride-run", default="runs/accelerated_stride8_n128_compare_ci16")
    parser.add_argument("--fno-run", default="runs/fno_stride8_n128_compare_ci16")
    parser.add_argument("--out", default="docs/fno_vs_stride_n128.svg")
    args = parser.parse_args()
    make_plot(Path(args.stride_run), Path(args.fno_run), Path(args.out))


if __name__ == "__main__":
    main()
