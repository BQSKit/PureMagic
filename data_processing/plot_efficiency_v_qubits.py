#!/usr/bin/env python3
"""
Scatter plot of Parallel Efficiency (y) vs ancilla qubits (x) from PureMagic
output files.  Ancilla qubits = total qubits - data qubits.  Each -f input
file becomes one series on the plot.

A single file may contain multiple concatenated runs (one per circuit), which
is the typical output when running PureMagic over a benchmark suite.

Usage:
    python plot_efficiency_v_qubits.py \\
        -f results/cliffordt/puremagic/out:PureMagic \\
        -f results/cliffordt/bus/out:Bus \\
        -o efficiency_v_qubits.png
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
    Parse a PureMagic output file and return a list of dicts, one per run:
        {"circuit": str, "parallel_efficiency": float, "ancilla_qubits": int}

    ancilla_qubits = total_qubits - data_qubits, parsed from the block:
        Number of qubits:
          data:         8 (0.222)
          ...
          total:        36

    Parallel efficiency is read from:
        Parallel efficiency: <value>
    or computed from:
        Parallelism: <speedup>x
        Optimal timesteps <n> (<optimal_speedup> speedup) volume <v>
    as  parallel_efficiency = speedup / optimal_speedup.
    """
    results = []
    current_circuit = None
    current_data_qubits = None
    current_total_qubits = None
    current_parallelism = None
    current_optimal_speedup = None
    current_parallel_efficiency = None
    in_qubit_block = False

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Start of qubit block
            if line_clean == "Number of qubits:":
                in_qubit_block = True
                current_data_qubits = None
                current_total_qubits = None
                continue

            if in_qubit_block:
                # "  data:         8 (0.222)"
                m = re.match(r"data:\s+(\d+)", line_clean)
                if m:
                    current_data_qubits = int(m.group(1))
                # "  total:        36"
                m = re.match(r"total:\s+(\d+)", line_clean)
                if m:
                    current_total_qubits = int(m.group(1))
                    in_qubit_block = False  # total is the last line of the block

            # Circuit name — reset per-circuit state
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                current_circuit = m.group(1)
                current_parallelism = None
                current_optimal_speedup = None
                current_parallel_efficiency = None

            # Optimal speedup: "Optimal timesteps 15216 (5.245 speedup) volume ..."
            m = re.match(r"Optimal timesteps \d+ \(([0-9.eE+\-]+) speedup\)", line_clean)
            if m and current_circuit is not None:
                current_optimal_speedup = float(m.group(1))

            # Parallelism: "Parallelism: 4.010x"
            m = re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", line_clean)
            if m and current_circuit is not None:
                current_parallelism = float(m.group(1))

            # Parallel efficiency printed directly (newer builds)
            m = re.match(r"Parallel efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                current_parallel_efficiency = float(m.group(1))

            # Record once we have all required values
            have_efficiency = current_parallel_efficiency is not None or (
                current_parallelism is not None and current_optimal_speedup is not None
            )
            if (
                current_circuit is not None
                and current_data_qubits is not None
                and current_total_qubits is not None
                and have_efficiency
            ):
                if current_parallel_efficiency is None:
                    current_parallel_efficiency = (
                        current_parallelism / current_optimal_speedup  # type: ignore[operator]
                    )
                results.append(
                    {
                        "circuit": current_circuit,
                        "parallel_efficiency": current_parallel_efficiency,
                        "ancilla_qubits": current_total_qubits - current_data_qubits,
                    }
                )
                current_circuit = None
                current_parallelism = None
                current_optimal_speedup = None
                current_parallel_efficiency = None

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
            "Scatter plot of Parallel Efficiency vs ancilla qubits "
            "(total - data) from PureMagic output files.  Each -f argument "
            "is one series.  Append :<label> to set a display label."
        )
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        required=True,
        metavar="FILE[:LABEL]",
        help=("PureMagic output file, optionally as path:label.  " "Repeat for multiple series."),
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

        ancilla_counts = [d["ancilla_qubits"] for d in data]
        parallel_efficiencies = [d["parallel_efficiency"] for d in data]
        colour = _COLOURS[i % len(_COLOURS)]

        ax.plot(
            ancilla_counts,
            parallel_efficiencies,
            label=label,
            color=colour,
            linewidth=1.0,
            marker="o",
        )

    # ax.set_xscale("log")
    ax.set_xlabel("Ancilla qubits (total − data)")
    ax.set_ylabel("Parallel Efficiency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
