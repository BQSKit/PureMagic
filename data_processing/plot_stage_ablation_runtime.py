#!/usr/bin/env python3
"""Chart how much each Clifford+T pipeline stage costs in compile time, for
the largest circuit in each benchmark family.

Same five `compile_cliffordt` (rust) run logs as plot_stage_ablation.py (see
its docstring for what each configuration means), but plotting each run's
`total:` wall-clock time instead of T count, normalized to that circuit's
qiskit-backend total time (data/all_compiled_qiskit/out).

Runtime spans far more orders of magnitude than T count did: the gridsynth
configs (no_opt/no_resynth/basic) mostly run in a few percent to two-thirds
of qiskit's own time, while cyclosynth (both the full pipeline and the
cyclosynth-alone probe) runs 7-80x *slower* than qiskit on every family. A
linear axis (even broken into two panels) can't hold a range that wide
without crushing the fast end flat, so this uses a log-scale y-axis instead.

Usage: ./plot_stage_ablation_runtime.py [-o OUTPUT.png]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPE_DIR = REPO_ROOT / "data/all_compiled_pipe-testing"

RUNS = {
    "no_opt": PIPE_DIR / "out-no-gauge-collapse-no-resynthesis-no-phase",
    "no_resynth": PIPE_DIR / "out-no-resynthesis",
    "basic": PIPE_DIR / "out-basic",
    "cyclosynth": PIPE_DIR / "out-cyclosynth",
    "cyclosynth_only": PIPE_DIR / "out-no-gauge-collapse-no-resynthesis-cyclosynth",
}
CONFIG_ORDER = ["no_opt", "no_resynth", "basic", "cyclosynth", "cyclosynth_only"]
CONFIG_LABELS = {
    "no_opt": "no optimization",
    "no_resynth": "+ phase merge & gauge collapse",
    "basic": "+ windowed resynthesis",
    "cyclosynth": "+ cyclosynth final synthesis",
    "cyclosynth_only": "cyclosynth alone",
}
QISKIT_OUT = REPO_ROOT / "data/all_compiled_qiskit/out"

# Same family/circuit picks as plot_stage_ablation.py -- see its comment for
# why dnn_n51/qft_n160 are each the largest same-prefixed circuit but not a
# bigger version of their smaller siblings.
FAMILIES = {
    "dnn": "dnn_n51",
    "hubbard": "hubbard_18",
    "knn": "knn_n129",
    "qaoa": "qaoa_barabasi_albert_N149_3reps",
    "qft": "qft_n160",
    "qugan": "qugan_n111",
    "qv": "qv_N036_12345",
    "heisenberg": "square_heisenberg_N225",
    "ising": "ising_n98",
}

QUBIT_RE = re.compile(r"(\d+) qubits,")
TOTAL_TIME_RE = re.compile(r"total:\s*([\d.]+)s")

CONFIG_COLORS = {
    "no_opt": "#c9daf8",
    "no_resynth": "#7aa8e8",
    "basic": "#2a78d6",
    "cyclosynth": "#0d3472",
    "cyclosynth_only": "#eda100",
}
SURFACE = "#fcfcfb"
INK = "#1a1a19"
BAR_WIDTH = 0.16


def parse_total_times(path: Path) -> dict[str, float]:
    text = path.read_text()
    blocks = re.split(r"^=== ", text, flags=re.M)[1:]
    times = {}
    for block in blocks:
        name = block.split("\n", 1)[0].strip().removesuffix(".qasm")
        m = TOTAL_TIME_RE.search(block)
        if m:
            times[name] = float(m.group(1))
    return times


def parse_qubits(path: Path) -> dict[str, int]:
    text = path.read_text()
    blocks = re.split(r"^=== ", text, flags=re.M)[1:]
    qubits = {}
    for block in blocks:
        name = block.split("\n", 1)[0].strip().removesuffix(".qasm")
        m = QUBIT_RE.search(block)
        if m:
            qubits[name] = int(m.group(1))
    return qubits


def draw_groups(ax, values, n_families):
    for ci, config in enumerate(CONFIG_ORDER):
        offset = (ci - (len(CONFIG_ORDER) - 1) / 2) * BAR_WIDTH
        xs = [fi + offset for fi in range(n_families)]
        heights = values[config]
        ax.bar(
            xs, heights, width=BAR_WIDTH * 0.92, color=CONFIG_COLORS[config],
            edgecolor=SURFACE, linewidth=0.5,
        )
        for xi, h in zip(xs, heights):
            label = f"{h:.0f}" if h >= 10 else f"{h:.1f}"
            ax.annotate(
                label, (xi, h), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="stage_ablation_runtime.png")
    args = parser.parse_args()

    runs = {key: parse_total_times(path) for key, path in RUNS.items()}
    qiskit_time = parse_total_times(QISKIT_OUT)
    qubits = parse_qubits(QISKIT_OUT)

    labels = []
    rows = []
    values = {config: [] for config in CONFIG_ORDER}
    for family, circuit in FAMILIES.items():
        baseline = qiskit_time[circuit]
        pcts = {config: runs[config][circuit] / baseline * 100 for config in CONFIG_ORDER}
        labels.append(f"{family}\n({qubits[circuit]}q)")
        rows.append((family, circuit, baseline, pcts))
        for config in CONFIG_ORDER:
            values[config].append(pcts[config])

    print(
        f"{'family':11s} {'circuit':32s} {'qiskit_s':>9s} "
        + " ".join(f"{c:>16s}" for c in CONFIG_ORDER)
    )
    for family, circuit, baseline, pcts in rows:
        print(
            f"{family:11s} {circuit:32s} {baseline:9.2f} "
            + " ".join(f"{pcts[c]:15.0f}%" for c in CONFIG_ORDER)
        )

    n_families = len(labels)
    fig, ax = plt.subplots(figsize=(17, 8))

    draw_groups(ax, values, n_families)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(axis="y", which="major", color=INK, alpha=0.15, linewidth=0.6)
    top = max(max(values[c]) for c in CONFIG_ORDER)
    ax.set_ylim(top=top * 4)  # headroom so the legend box doesn't sit over any bar

    ax.axhline(100, color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(range(n_families))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlim(-0.6, n_families - 0.4)
    ax.set_ylabel("Total compile time (% of qiskit backend, log scale)")
    ax.set_title(
        "Pipeline stage contribution to compile time, largest circuit per family\n"
        "(one bar per configuration, increasing functionality left to right)"
    )

    fig.text(
        0.5, 0.015, "  ".join(f"{f}={c}" for f, c in FAMILIES.items()), ha="center", fontsize=6.5
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CONFIG_COLORS[c]) for c in CONFIG_ORDER
    ] + [plt.Line2D([0], [0], color=INK, linewidth=0.8, linestyle="--", alpha=0.6)]
    ax.legend(
        legend_handles,
        [CONFIG_LABELS[c] for c in CONFIG_ORDER] + ["qiskit baseline (100%)"],
        loc="upper left", fontsize=8, framealpha=0.9,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
