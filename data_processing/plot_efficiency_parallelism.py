#!/usr/bin/env python3
"""
Scatter plot of Scheduling Efficiency (y) vs Parallelism (x) from PureMagic
output files.  Each input file becomes one series on the plot.

Usage:
    python plot_efficiency_parallelism.py -f <file>[:<label>] [-f <file>[:<label>] ...] -o output.png

Example:
    python plot_efficiency_parallelism.py \\
        -f ../results/cliffordt/puremagic/out:PureMagic \\
        -f ../results/cliffordt/bus/out:Bus \\
        -o scatter.png
"""

import argparse
import re
import sys
import matplotlib.pyplot as plt


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
    Parse a PureMagic output file and return a list of dicts, one per circuit:
        {"circuit": str, "efficiency": float, "parallelism": float}
    """
    results = []
    current_circuit = None
    current_efficiency = None

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Circuit name
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                current_circuit = m.group(1)
                current_efficiency = None

            # Scheduling efficiency
            m = re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                current_efficiency = float(m.group(1))

            # Parallelism  (value followed by 'x', e.g. "1.657x")
            m = re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", line_clean)
            if m and current_circuit is not None and current_efficiency is not None:
                parallelism = float(m.group(1))
                results.append(
                    {
                        "circuit": current_circuit,
                        "efficiency": current_efficiency,
                        "parallelism": parallelism,
                    }
                )
                current_circuit = None
                current_efficiency = None

    return results


def prettify_circuit_name(name):
    """Apply display-friendly substitutions to a circuit name."""
    if name.startswith("qaoa_barabasi_albert"):
        name = "QAOA" + name[len("qaoa_barabasi_albert") :]
    if name.startswith("square_"):
        name = name[len("square_") :]
    return name


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
            "Scatter plot of Scheduling Efficiency vs Parallelism from PureMagic "
            "output files.  Each -f argument is one series.  Append :<label> to "
            "set a display label, e.g. results/out:PureMagic."
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

        parallelisms = [d["parallelism"] for d in data]
        efficiencies = [d["efficiency"] for d in data]
        display_names = [node_count_label(d["circuit"]) for d in data]
        colour = _COLOURS[i % len(_COLOURS)]

        ax.scatter(
            parallelisms,
            efficiencies,
            label=label,
            color=colour,
            edgecolors="black",
            linewidths=0.5,
            s=60,
            zorder=3,
        )

        # Annotate each point with the circuit name
        for x_val, y_val, name in zip(parallelisms, efficiencies, display_names):
            ax.annotate(
                name,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=colour,
            )

    ax.set_xlabel("Parallelism")
    ax.set_ylabel("Scheduling Efficiency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
