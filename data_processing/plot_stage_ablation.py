#!/usr/bin/env python3
"""Chart how much each Clifford+T pipeline stage contributes to the final T
count, for the largest circuit in each benchmark family.

Reads five `compile_cliffordt` (rust) run logs from
data/all_compiled_pipe-testing/, covering increasing pipeline functionality:

    no_opt          --skip-gauge-collapse --skip-windowed-resynthesis --skip-phase-merge
    no_resynth      --skip-windowed-resynthesis
    basic           (all stages, gridsynth final synthesis)
    cyclosynth      (all stages, cyclosynth final synthesis)
    cyclosynth_only --skip-gauge-collapse --skip-windowed-resynthesis --cyclosynth

cyclosynth_only isn't a further step in that progression -- it isolates
cyclosynth's own contribution, with the same stages skipped as no_opt, just
gridsynth swapped for cyclosynth in Stage 4. Comparing those two bars
directly shows what cyclosynth alone is worth, before gauge collapse or
windowed resynthesis get a chance to help, which is why it's drawn last in
each cluster and off the blue progression, in its own color. (Its log
predates --skip-phase-merge, so Stage 0 still ran there, unlike no_opt -- a
small, mostly negligible mismatch except on qft/ising, see below.)

For each family's largest circuit, T counts are normalized to that circuit's
qiskit-backend T count (data/all_compiled_qiskit/out), matching the indexing
convention plot_gate_counts.py already uses -- circuits span too wide a range
of absolute T counts to compare on one linear axis otherwise.

Each family gets a cluster of five side-by-side bars, one per configuration,
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
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPE_DIR = REPO_ROOT / "data/all_compiled_pipe-testing"

RUNS = {
    "no_opt": PIPE_DIR / "out-no-gauge-collapse-no-resynthesis-no-phase",
    "cyclosynth_only": PIPE_DIR / "out-no-gauge-collapse-no-resynthesis-cyclosynth",
    "no_resynth": PIPE_DIR / "out-no-resynthesis",
    "basic": PIPE_DIR / "out-basic",
    "cyclosynth": PIPE_DIR / "out-cyclosynth",
}
CONFIG_ORDER = ["no_opt", "no_resynth", "basic", "cyclosynth", "cyclosynth_only"]
CONFIG_LABELS = {
    "no_opt": "no optimization",
    "no_resynth": "+ phase merge & gauge collapse",
    "basic": "+ windowed resynthesis",
    "cyclosynth": "+ cyclosynth final synthesis",
    "cyclosynth_only": "cyclosynth alone",
}
QISKIT_OUT = PIPE_DIR / "out-qiskit"

# The largest circuit in each family, picked by qubit count. dnn_n51 and
# qft_n160 are each structurally distinct from their lower-numbered,
# same-prefixed siblings (different generator, different gate mix) -- they're
# still the largest circuit sharing that name prefix, just not a "bigger
# version" of the smaller ones in that family.
FAMILIES = {
    "dnn": "dnn_n51",
    "knn": "knn_n129",
    "qaoa": "qaoa_barabasi_albert_N149_3reps",
    "qft": "qft_n160",
    "qugan": "qugan_n111",
    "qv": "qv_N064_12345",
    "heisenberg": "square_heisenberg_N225",
    "ising": "ising_n98",
    "adder": "cdkm_ripple_carry_adder_indep_opt0_200",
    "fermihubbard": "fermi_hubbard_1d_128q",
    "grover": "grover_n100_i10",
}

SUMMARY_RE = re.compile(r"(\d+) qubits, \d+ gates -> \d+ gates \(T=(\d+),")

# Sequential ramp (palette.md "Blue" scale) for the four progression steps,
# lightest (least optimized) to darkest (most optimized). cyclosynth_only
# isn't a step in that progression, so it gets its own categorical color --
# the same amber plot_gate_counts.py already uses for cyclosynth's identity.
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
    """values[config] is a list of per-family percentages, `None` where that
    (family, config) hasn't been run yet. Draws one cluster of bars per
    family -- missing entries just leave a gap in the cluster -- and labels
    bars whose top lands within ylim."""
    n_families = len(next(iter(values.values())))
    for ci, config in enumerate(CONFIG_ORDER):
        offset = (ci - (len(CONFIG_ORDER) - 1) / 2) * BAR_WIDTH
        pairs = [(fi + offset, h) for fi, h in enumerate(values[config]) if h is not None]
        if not pairs:
            continue
        xs, heights = zip(*pairs)
        bars = ax.bar(
            xs,
            heights,
            width=BAR_WIDTH * 0.92,
            color=CONFIG_COLORS[config],
            edgecolor=SURFACE,
            linewidth=0.5,
        )
        for xi, h, bar in zip(xs, heights, bars):
            if not (ylim[0] < h <= ylim[1]):
                continue
            ax.text(
                xi,
                h + (ylim[1] - ylim[0]) * 0.012,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
                rotation=90,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="stage_ablation_tcount.png")
    args = parser.parse_args()

    runs = {key: parse_t_counts(path) for key, path in RUNS.items()}
    qiskit_t = parse_t_counts(QISKIT_OUT)
    qubits = parse_qubits(QISKIT_OUT)

    labels = []
    rows = []  # for the printed summary table
    values: dict[str, list[float | None]] = {config: [] for config in CONFIG_ORDER}
    skipped = []
    for family, circuit in sorted(FAMILIES.items()):
        baseline = qiskit_t.get(circuit)
        if baseline is None:
            skipped.append((family, circuit))
            continue
        pcts = {}
        for config in CONFIG_ORDER:
            t = runs[config].get(circuit)
            pcts[config] = (t / baseline * 100) if t is not None else None
        labels.append(f"{family}\n({qubits[circuit]}q)")
        rows.append((family, circuit, pcts))
        for config in CONFIG_ORDER:
            values[config].append(pcts[config])

    if skipped:
        print("Skipping (no qiskit baseline yet): " + ", ".join(f"{f} ({c})" for f, c in skipped))

    def fmt_pct(v: float | None) -> str:
        return f"{v:10.1f}%" if v is not None else f"{'n/a':>10s} "

    print(f"{'family':11s} {'circuit':32s} " + " ".join(f"{c:>11s}" for c in CONFIG_ORDER))
    for family, circuit, pcts in rows:
        print(f"{family:11s} {circuit:32s} " + " ".join(fmt_pct(pcts[c]) for c in CONFIG_ORDER))
    missing = [
        (family, config)
        for family, _, pcts in rows
        for config in CONFIG_ORDER
        if pcts[config] is None
    ]
    if missing:
        print("Missing (not yet run): " + ", ".join(f"{f}/{c}" for f, c in missing))

    n_families = len(labels)
    bottom_max = 115
    top_min = 115
    all_vals = [v for c in CONFIG_ORDER for v in values[c] if v is not None]
    above_bottom = [v for v in all_vals if v > bottom_max]
    top_max = max(above_bottom) * 1.08 if above_bottom else bottom_max * 1.2

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(17, 8), sharex=True, gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08}
    )

    draw_groups(ax_bottom, values, (0, bottom_max))
    draw_groups(ax_top, values, (top_min, top_max))

    ax_bottom.set_ylim(0, bottom_max)
    ax_top.set_ylim(top_min, top_max)

    baseline_line_kwargs: dict[str, Any] = dict(color=INK, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_bottom.axhline(100, **baseline_line_kwargs)
    if top_min <= 100 <= top_max:
        ax_top.axhline(100, **baseline_line_kwargs)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False)
    d = 0.5
    break_kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=10,
        linestyle="none",
        color=INK,
        mec=INK,
        mew=1,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **break_kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **break_kwargs)

    ax_bottom.set_xticks(range(n_families))
    ax_bottom.set_xticklabels(labels, fontsize=10)
    ax_bottom.set_xlim(-0.6, n_families - 0.4)
    fig.text(
        0.5,
        0.015,
        "  ".join(f"{family}={circuit}" for family, circuit, _ in rows),
        ha="center",
        fontsize=6.5,
    )
    fig.text(
        0.02, 0.5, "T gates (% of qiskit backend)", va="center", rotation="vertical", fontsize=10
    )
    ax_top.set_title(
        "Pipeline stage contribution to T count, largest circuit per family\n"
        "(one bar per configuration, increasing functionality left to right)"
    )

    legend_handles = [Rectangle((0, 0), 1, 1, color=CONFIG_COLORS[c]) for c in CONFIG_ORDER] + [
        Line2D([0], [0], **baseline_line_kwargs)
    ]
    ax_top.legend(
        legend_handles,
        [CONFIG_LABELS[c] for c in CONFIG_ORDER] + ["qiskit baseline (100%)"],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
