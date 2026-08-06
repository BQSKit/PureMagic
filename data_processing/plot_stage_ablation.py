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

Each bar is a waterfall collapsed into one stacked column: the bottom segment
is the final (cyclosynth) result, and each segment above it is the extra cost
of *not* having that stage. A stage's cost is occasionally negative (it makes
that particular circuit slightly worse) -- those segments are drawn hatched in
the regression color rather than clipped to zero.

qft_n160's "phase merge + gauge collapse" segment and ising_n98's total both
run far above the rest (dropping those two stages lets a huge number of
otherwise-identical small-angle Rz rotations go unmerged), so the y-axis is
broken into two panels rather than crushing every other bar flat.

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

# Sequential ramp (palette.md "Blue" scale), bottom (final result) darkest to
# top (largest remaining baseline) lightest.
SEGMENT_COLORS = ["#0d3472", "#2a78d6", "#7aa8e8", "#c9daf8"]
REGRESSION_COLOR = "#d5341f"
SURFACE = "#fcfcfb"
INK = "#1a1a19"
LABEL_MIN = 4.0  # suppress on-bar text for segments smaller than this (% points)

SEGMENT_LABELS = [
    "cyclosynth final synthesis",
    "gridsynth vs cyclosynth",
    "windowed resynthesis",
    "phase merge + gauge collapse",
]


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


def draw_stack(ax, x, segments):
    """Draw the stacked/waterfall bars on one axis; returns the per-family
    (bottom, height) pairs for each segment, for shared text placement."""
    bottoms = [0.0] * len(x)
    placed = []
    for i, heights in enumerate(segments):
        colors = [REGRESSION_COLOR if h < 0 and i > 0 else SEGMENT_COLORS[i] for h in heights]
        bars = ax.bar(
            x, heights, bottom=bottoms, color=colors, edgecolor=SURFACE, linewidth=0.6, width=0.62
        )
        for bar, h in zip(bars, heights):
            if h < 0 and i > 0:
                bar.set_hatch("//")
        placed.append(list(zip(bottoms, heights)))
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    return placed


def label_stack(ax, x, placed, ylim):
    for i, seg in enumerate(placed):
        for xi, (b, h) in zip(x, seg):
            mid = b + h / 2
            if abs(h) < LABEL_MIN or not (ylim[0] <= mid <= ylim[1]):
                continue
            text = f"{h:.0f}%" if i == 0 else f"{h:+.0f}%"
            color = "white" if i < 2 else INK
            ax.text(xi, mid, text, ha="center", va="center", fontsize=8, color=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="stage_ablation_tcount.png")
    args = parser.parse_args()

    runs = {key: parse_t_counts(path) for key, path in RUNS.items()}
    qiskit_t = parse_t_counts(QISKIT_OUT)

    labels = []
    rows = []  # for the printed summary table
    segments = [[] for _ in range(4)]
    for family, circuit in FAMILIES.items():
        baseline = qiskit_t[circuit]
        s4 = runs["cyclosynth"][circuit] / baseline * 100
        s3 = runs["basic"][circuit] / baseline * 100
        s2 = runs["no_resynth"][circuit] / baseline * 100
        s1 = runs["no_opt"][circuit] / baseline * 100
        heights = [s4, s3 - s4, s2 - s3, s1 - s2]
        labels.append(family)
        rows.append((family, circuit, s4, s3, s2, s1))
        for i, h in enumerate(heights):
            segments[i].append(h)

    print(f"{'family':11s} {'circuit':32s} {'cyclosynth':>10s} {'basic':>8s} {'no_resynth':>10s} {'no_opt':>8s}")
    for family, circuit, s4, s3, s2, s1 in rows:
        print(f"{family:11s} {circuit:32s} {s4:9.1f}% {s3:7.1f}% {s2:9.1f}% {s1:7.1f}%")

    x = list(range(len(labels)))
    bottom_max = 115
    top_min = 115

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08}
    )

    placed_top = draw_stack(ax_top, x, segments)
    placed_bottom = draw_stack(ax_bottom, x, segments)

    ax_bottom.set_ylim(0, bottom_max)
    tops = [sum(h for _, h in [seg[fi] for seg in placed_top]) for fi in range(len(x))]
    ax_top.set_ylim(top_min, max(tops) * 1.05)

    label_stack(ax_bottom, x, placed_bottom, (0, bottom_max))
    label_stack(ax_top, x, placed_top, (top_min, max(tops) * 1.05))

    baseline_line_kwargs = dict(color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bottom.axhline(100, **baseline_line_kwargs)
    if top_min <= 100 <= max(tops) * 1.05:
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

    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, fontsize=10)
    fig.text(0.5, 0.015, "  ".join(f"{f}={c}" for f, c in FAMILIES.items()), ha="center", fontsize=6.5)
    fig.text(0.02, 0.5, "T gates (% of qiskit backend)", va="center", rotation="vertical", fontsize=10)
    ax_top.set_title(
        "Pipeline stage contribution to T count, largest circuit per family\n"
        "(stacked bottom-up: final result, then each stage's cost if skipped)"
    )

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[i]) for i in range(4)] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=REGRESSION_COLOR, hatch="//", edgecolor=SURFACE),
        plt.Line2D([0], [0], **baseline_line_kwargs),
    ]
    ax_top.legend(
        legend_handles,
        SEGMENT_LABELS + ["stage regression (T increased)", "qiskit baseline (100%)"],
        loc="upper left", fontsize=8, framealpha=0.9,
    )

    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
