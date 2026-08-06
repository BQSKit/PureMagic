#!/usr/bin/env python3
"""Chart how much each Clifford+T pipeline stage contributes to the final T
count, for the largest circuit in each benchmark family.

Reads four `compile_cliffordt` (rust) run logs from
data/all_compiled_pipe-testing/, covering increasing pipeline functionality:

    no_opt      --skip-gauge-collapse --skip-windowed-resynthesis --skip-phase-merge
    no_resynth  --skip-windowed-resynthesis
    basic       (all stages, gridsynth final synthesis)
    cyclosynth  (all stages, cyclosynth final synthesis)

For each family's largest circuit, T counts are normalized to that circuit's
qiskit-backend T count (data/all_compiled_qiskit/out), matching the indexing
convention plot_gate_counts.py already uses -- circuits span too wide a range
of absolute T counts to compare on one linear axis otherwise.

Each family gets a cluster of four side-by-side bars, one per configuration,
so a stage's occasional regression (it makes that particular circuit
slightly worse) just shows up as a taller bar to its left neighbor's right,
with no special-casing needed the way a stacked/waterfall rendering would.

qft_n160 and ising_n98's no_opt bars run far above the rest (dropping
phase-merge lets a huge number of otherwise-identical small-angle Rz
rotations go unmerged), so the y-axis is broken into two panels rather than
crushing every other bar flat.

Usage: ./plot_stage_ablation.py [-o OUTPUT.png]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPE_DIR = REPO_ROOT / "data/all_compiled_pipe-testing"

RUNS = {
    "no_opt": PIPE_DIR / "out-no-gauge-collapse-no-resynthesis-no-phase",
    "no_resynth": PIPE_DIR / "out-no-resynthesis",
    "basic": PIPE_DIR / "out-basic",
    "cyclosynth": PIPE_DIR / "out-cyclosynth",
}
CONFIG_ORDER = ["no_opt", "no_resynth", "basic", "cyclosynth"]
CONFIG_LABELS = {
    "no_opt": "no optimization",
    "no_resynth": "+ phase merge & gauge collapse",
    "basic": "+ windowed resynthesis",
    "cyclosynth": "+ cyclosynth final synthesis",
}
QISKIT_OUT = REPO_ROOT / "data/all_compiled_qiskit/out"

# The largest circuit in each family, picked by qubit count. dnn_n51 and
# qft_n160 are each structurally distinct from their lower-numbered,
# same-prefixed siblings (different generator, different gate mix) -- they're
# still the largest circuit sharing that name prefix, just not a "bigger
# version" of the smaller ones in that family.
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

SUMMARY_RE = re.compile(r"(\d+) qubits, \d+ gates -> \d+ gates \(T=(\d+),")

# Sequential ramp (palette.md "Blue" scale), lightest (least optimized) to
# darkest (most optimized), in CONFIG_ORDER.
CONFIG_COLORS = {
    "no_opt": "#c9daf8",
    "no_resynth": "#7aa8e8",
    "basic": "#2a78d6",
    "cyclosynth": "#0d3472",
}
SURFACE = "#fcfcfb"
INK = "#1a1a19"
BAR_WIDTH = 0.2


def parse_t_counts(path: Path) -> dict[str, int]:
    text = path.read_text()
    blocks = re.split(r"^=== ", text, flags=re.M)[1:]
    counts = {}
    for block in blocks:
        name = block.split("\n", 1)[0].strip().removesuffix(".qasm")
        m = SUMMARY_RE.search(block)
        if m:
            counts[name] = int(m.group(2))
    return counts


def parse_qubits(path: Path) -> dict[str, int]:
    text = path.read_text()
    blocks = re.split(r"^=== ", text, flags=re.M)[1:]
    qubits = {}
    for block in blocks:
        name = block.split("\n", 1)[0].strip().removesuffix(".qasm")
        m = SUMMARY_RE.search(block)
        if m:
            qubits[name] = int(m.group(1))
    return qubits


def draw_groups(ax, values, ylim):
    """values[config] is a list of per-family percentages. Draws one cluster
    of bars per family and labels bars whose top lands within ylim."""
    n_families = len(next(iter(values.values())))
    for ci, config in enumerate(CONFIG_ORDER):
        offset = (ci - (len(CONFIG_ORDER) - 1) / 2) * BAR_WIDTH
        xs = [fi + offset for fi in range(n_families)]
        heights = values[config]
        bars = ax.bar(
            xs, heights, width=BAR_WIDTH * 0.92, color=CONFIG_COLORS[config],
            edgecolor=SURFACE, linewidth=0.5,
        )
        for xi, h, bar in zip(xs, heights, bars):
            if not (ylim[0] < h <= ylim[1]):
                continue
            ax.text(xi, h + (ylim[1] - ylim[0]) * 0.012, f"{h:.0f}", ha="center",
                     va="bottom", fontsize=6.5, color=INK, rotation=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="stage_ablation_tcount.png")
    args = parser.parse_args()

    runs = {key: parse_t_counts(path) for key, path in RUNS.items()}
    qiskit_t = parse_t_counts(QISKIT_OUT)
    qubits = parse_qubits(QISKIT_OUT)

    labels = []
    rows = []  # for the printed summary table
    values = {config: [] for config in CONFIG_ORDER}
    for family, circuit in FAMILIES.items():
        baseline = qiskit_t[circuit]
        pcts = {config: runs[config][circuit] / baseline * 100 for config in CONFIG_ORDER}
        labels.append(f"{family}\n({qubits[circuit]}q)")
        rows.append((family, circuit, pcts))
        for config in CONFIG_ORDER:
            values[config].append(pcts[config])

    print(f"{'family':11s} {'circuit':32s} " + " ".join(f"{c:>11s}" for c in CONFIG_ORDER))
    for family, circuit, pcts in rows:
        print(f"{family:11s} {circuit:32s} " + " ".join(f"{pcts[c]:10.1f}%" for c in CONFIG_ORDER))

    n_families = len(labels)
    bottom_max = 115
    top_min = 115
    top_max = max(max(values[c]) for c in CONFIG_ORDER) * 1.08

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08}
    )

    draw_groups(ax_bottom, values, (0, bottom_max))
    draw_groups(ax_top, values, (top_min, top_max))

    ax_bottom.set_ylim(0, bottom_max)
    ax_top.set_ylim(top_min, top_max)

    baseline_line_kwargs = dict(color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bottom.axhline(100, **baseline_line_kwargs)
    if top_min <= 100 <= top_max:
        ax_top.axhline(100, **baseline_line_kwargs)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    d = 0.5
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)], markersize=10, linestyle="none", color=INK, mec=INK, mew=1,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **break_kwargs)

    ax_bottom.set_xticks(range(n_families))
    ax_bottom.set_xticklabels(labels, fontsize=10)
    ax_bottom.set_xlim(-0.6, n_families - 0.4)
    fig.text(0.5, 0.015, "  ".join(f"{f}={c}" for f, c in FAMILIES.items()), ha="center", fontsize=6.5)
    fig.text(0.02, 0.5, "T gates (% of qiskit backend)", va="center", rotation="vertical", fontsize=10)
    ax_top.set_title(
        "Pipeline stage contribution to T count, largest circuit per family\n"
        "(one bar per configuration, increasing functionality left to right)"
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CONFIG_COLORS[c]) for c in CONFIG_ORDER
    ] + [plt.Line2D([0], [0], **baseline_line_kwargs)]
    ax_top.legend(
        legend_handles,
        [CONFIG_LABELS[c] for c in CONFIG_ORDER] + ["qiskit baseline (100%)"],
        loc="upper left", fontsize=8, framealpha=0.9,
    )

    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
