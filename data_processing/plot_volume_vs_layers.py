#!/usr/bin/env python3
"""
Plot scheduled volume vs. pre-scheduling DAG layer count across a max-weight
sweep (out-pm-trans0 .. out-pm-transN), one point per (circuit, weight).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_puremagic import parse_output_file, prettify_circuit_name  # noqa: E402

# 11 visually-distinct (colour, marker) pairs -- assigned in a fixed order per
# circuit, never cycled, so no two circuits ever share the same combination.
_STYLES = [
    ("steelblue", "o"),
    ("darkorange", "s"),
    ("forestgreen", "^"),
    ("crimson", "D"),
    ("mediumpurple", "v"),
    ("saddlebrown", "P"),
    ("deeppink", "X"),
    ("teal", "*"),
    ("gray", "o"),
    ("olive", "s"),
    ("royalblue", "^"),
]


def load_sweep(directory, max_weight):
    frames = []
    for w in range(max_weight + 1):
        path = os.path.join(directory, f"out-pm-trans{w}")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        df = parse_output_file(path)
        if df.empty:
            continue
        df["weight"] = w
        frames.append(df)
    if not frames:
        print("Error: no data loaded", file=sys.stderr)
        sys.exit(1)
    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=["layers", "volume"])
    full["circuit_pretty"] = full["circuit"].apply(prettify_circuit_name)
    return full


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-d", "--directory", default="results/circuits")
    ap.add_argument("-w", "--max-weight", type=int, default=10)
    ap.add_argument("-o", "--output", default="volume_v_layers.png")
    args = ap.parse_args()

    df = load_sweep(args.directory, args.max_weight)

    log_x = np.log10(df["layers"])
    log_y = np.log10(df["volume"])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    r = np.corrcoef(log_x, log_y)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 6))

    circuits = sorted(df["circuit_pretty"].unique())
    for i, name in enumerate(circuits):
        colour, marker = _STYLES[i % len(_STYLES)]
        sub = df[df["circuit_pretty"] == name].sort_values("weight")
        ax.plot(
            sub["layers"],
            sub["volume"],
            color=colour,
            linewidth=1,
            alpha=0.6,
            zorder=1,
        )
        ax.scatter(
            sub["layers"],
            sub["volume"],
            color=colour,
            marker=marker,
            s=28,
            label=name,
            zorder=2,
        )

    xs = np.array([df["layers"].min(), df["layers"].max()])
    ax.plot(
        xs,
        10 ** (intercept) * xs**slope,
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=0,
        label=f"fit: volume $\\propto$ layers$^{{{slope:.2f}}}$ ($R$={r:.2f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Layers (pre-scheduling DAG depth)")
    ax.set_ylabel("Volume (lcycles $\\times$ total qubits)")
    ax.set_title(f"Volume vs. Layers across weight 0-{args.max_weight}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.output}")
    print(f"Pooled log-log correlation: r={r:.3f}, slope={slope:.3f}")


if __name__ == "__main__":
    main()
