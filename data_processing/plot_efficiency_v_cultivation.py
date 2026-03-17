#!/usr/bin/env python3
"""
Scatter plot of scheduling Efficiency (y) vs 1/magic_state_lambda (x) from
PureMagic output files.  Each input file becomes one series on the plot.

A single file may contain multiple concatenated runs (one per lambda value),
which is the typical output when sweeping over lambda values for one circuit.

"""

import argparse
import math
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Colours cycled through for successive series
_COLOURS = [
    "steelblue",
    "darkorange",
    "forestgreen",
    "crimson",
    "mediumpurple",
    "saddlebrown",
    "deeppink",
    "teal",
]


def parse_output_file(filepath):
    """
    Parse a PureMagic output file and return a list of dicts, one per run:
        {"circuit": str, "efficiency": float, "inv_lambda": float}

    A single file may contain multiple concatenated runs (one per lambda value).

    magic_state_lambda is read from the Args debug block, e.g.:
        magic_state_lambda: 0.0387396,

    Scheduling efficiency is printed directly as:
        Scheduling efficiency: <value>
    """
    results = []
    current_circuit = None
    current_lambda = None
    current_efficiency = None

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # magic_state_lambda from the Args debug block
            m = re.match(r"magic_state_lambda:\s*([0-9.eE+\-]+),?", line_clean)
            if m:
                current_lambda = float(m.group(1))
                # Reset per-run state when a new lambda is seen
                current_circuit = None
                current_efficiency = None

            # Circuit name
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                current_circuit = m.group(1)
                current_efficiency = None

            m = re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                current_efficiency = float(m.group(1))

            # Record once we have enough data for this run
            if (
                current_circuit is not None
                and current_lambda is not None
                and current_efficiency is not None
            ):
                assert current_efficiency is not None
                assert current_lambda is not None
                results.append(
                    {
                        "circuit": current_circuit,
                        "efficiency": current_efficiency,
                        "inv_lambda": 1.0 / current_lambda,
                    }
                )
                current_circuit = None
                current_efficiency = None

    return results


def node_count_label(name):
    """Return the node count (number after N or n) as a string, or the full name if not found."""
    m = re.search(r"[Nn](\d+)", name)
    return m.group(1) if m else name


def split_file_label(arg, default_label):
    """Split 'path:label' into (path, label). Uses the last colon as separator."""
    if ":" in arg:
        idx = arg.rfind(":")
        return arg[:idx], arg[idx + 1 :] or default_label
    return arg, default_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of Efficiency vs 1/magic_state_lambda from "
            "PureMagic output files.  Each -f argument is one series.  Append "
            ":<label> to set a display label, e.g. results/out:PureMagic."
        )
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        required=True,
        metavar="FILE[:LABEL]",
        help="PureMagic output file, optionally as path:label.  Repeat for multiple series.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image file name",
    )
    parser.add_argument(
        "-s",
        "--select",
        default=None,
        metavar="SUBSTRING",
        help="Only plot circuits whose name contains this substring.",
    )
    args = parser.parse_args()

    n = len(args.files)
    _, ax = plt.subplots(figsize=(max(8, n * 2), 6))

    for i, file_arg in enumerate(args.files):
        filepath, label = split_file_label(file_arg, f"file{i + 1}")
        data = parse_output_file(filepath)
        if not data:
            print(f"Warning: no data found in {filepath}", file=sys.stderr)
            continue

        if args.select:
            data = [d for d in data if args.select in d["circuit"]]
        if not data:
            print(f"Warning: no matching circuits in {filepath}", file=sys.stderr)
            continue

        inv_lambdas = [d["inv_lambda"] for d in data]
        efficiencies = [d["efficiency"] for d in data]
        colour = _COLOURS[i % len(_COLOURS)]

        ax.scatter(
            inv_lambdas,
            efficiencies,
            label=label,
            color=colour,
            edgecolors="black",
            linewidths=0.5,
            s=60,
            zorder=3,
        )

        # Connect points with a line (sorted by x) to show the trend
        sorted_pairs = sorted(zip(inv_lambdas, efficiencies))
        xs, ys = zip(*sorted_pairs)
        ax.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.6, zorder=2)

    ax.set_ylim(0, 1)
    ax.set_xscale("log", base=2)
    # Label every power-of-2 tick as "2^k"
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"$2^{{{int(round(math.log2(x)))}}}$")
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Expected cultivation time (cycles)")
    ax.set_ylabel("Efficiency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
