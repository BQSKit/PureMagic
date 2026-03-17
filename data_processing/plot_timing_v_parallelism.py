#!/usr/bin/env python3
"""
Scatter plot of average schedule_timestep timing (y, in μs) vs Parallelism or
number of data qubits (x) from PureMagic output files.  Each -f input file
becomes one series on the plot.

A single file may contain multiple concatenated runs (one per circuit), which
is the typical output when running PureMagic over a benchmark suite.

Usage:
    python plot_timing_v_parallelism.py \\
        -f results/cliffordt/puremagic/out:PureMagic \\
        -f results/cliffordt/bus/out:Bus \\
        -o timing_v_parallelism.png

    # Use data qubit count on x-axis
    python plot_timing_v_parallelism.py \\
        -f results/cliffordt/puremagic/out:PureMagic \\
        --x-axis qubits \\
        -o timing_v_qubits.png
"""

import argparse
import re
import sys
import numpy as np
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

# Conversion factors to microseconds
_TO_US = {"μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}


def parse_avg_timing(value_str, unit_str):
    """Convert a timing value + unit string to microseconds."""
    factor = _TO_US.get(unit_str, 1.0)
    return float(value_str) * factor


def parse_output_file(filepath):
    """
    Parse a PureMagic output file and return a list of dicts, one per run:
        {"circuit": str, "parallelism": float, "data_qubits": int,
         "avg_timestep_us": float}

    Parallelism is read from:
        Parallelism: <value>x

    Data qubit count is read from the Number of qubits block:
        data:         8 (0.222)

    Average schedule_timestep timing is read from the accumulated timings block:
        schedule_timestep         total: ...  avg:  <value> <unit>  max: ...
    """
    results = []
    current_circuit = None
    current_parallelism = None
    current_data_qubits = None
    in_qubit_block = False

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Start of qubit block
            if line_clean == "Number of qubits:":
                in_qubit_block = True
                continue

            if in_qubit_block:
                m = re.match(r"data:\s+(\d+)", line_clean)
                if m:
                    current_data_qubits = int(m.group(1))
                # End of qubit block at "total:" line
                if re.match(r"total:\s+\d+", line_clean):
                    in_qubit_block = False

            # Circuit name
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                current_circuit = m.group(1)
                current_parallelism = None

            # Parallelism: "Parallelism: 4.010x"
            m = re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", line_clean)
            if m and current_circuit is not None:
                current_parallelism = float(m.group(1))

            # schedule_timestep avg timing:
            # "  schedule_timestep   total: 23.0 ms  avg:  0.70 μs  max: ..."
            m = re.match(
                r"schedule_timestep\s+total:.*avg:\s*([0-9.eE+\-]+)\s*(\S+)\s+max:",
                line_clean,
            )
            if m and current_circuit is not None and current_parallelism is not None:
                avg_us = parse_avg_timing(m.group(1), m.group(2))
                results.append(
                    {
                        "circuit": current_circuit,
                        "parallelism": current_parallelism,
                        "data_qubits": current_data_qubits,
                        "avg_timestep_us": avg_us,
                    }
                )
                current_circuit = None
                current_parallelism = None

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
            "Scatter plot of average schedule_timestep timing vs Parallelism "
            "or data qubit count from PureMagic output files.  Each -f "
            "argument is one series.  Append :<label> to set a display label."
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
    parser.add_argument(
        "-x",
        "--x-axis",
        dest="x_axis",
        choices=["parallelism", "qubits"],
        default="parallelism",
        help="Variable to use on the x-axis: 'parallelism' (default) or 'qubits' (data qubit count).",
    )
    args = parser.parse_args()

    use_qubits = args.x_axis == "qubits"

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

        if use_qubits:
            data = [d for d in data if d["data_qubits"] is not None]
            if not data:
                print(f"Warning: no data qubit counts found in {filepath}", file=sys.stderr)
                continue
            x_values = [d["data_qubits"] for d in data]
        else:
            x_values = [d["parallelism"] for d in data]

        avg_timings = [d["avg_timestep_us"] for d in data]
        colour = _COLOURS[i % len(_COLOURS)]

        ax.scatter(
            x_values,
            avg_timings,
            label=label,
            color=colour,
            edgecolors="black",
            linewidths=0.5,
            s=60,
            zorder=3,
        )

        # Power-law trendline: fit log(y) = a*log(x) + b in log-log space
        if len(x_values) >= 2:
            log_x = np.log(x_values)
            log_y = np.log(avg_timings)
            a, b = np.polyfit(log_x, log_y, 1)
            x_fit = np.linspace(min(x_values), max(x_values), 200)
            y_fit = np.exp(b) * x_fit**a
            ax.plot(
                x_fit,
                y_fit,
                color=colour,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
                zorder=2,
                label=f"{label} fit ($x^{{{a:.2f}}}$)",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Use decimal tick labels at 0.5, 1, 2, 5, 10, 20, 50, ... instead of powers of 10
    decimal_fmt = ticker.FuncFormatter(lambda x, _: f"{x:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(ticker.LogLocator(base=10, subs=[0.5, 1.0, 2.0, 5.0], numticks=20))
        axis.set_major_formatter(decimal_fmt)
        axis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("Data qubits" if use_qubits else "Parallelism")
    ax.set_ylabel("Avg schedule_timestep (μs)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
