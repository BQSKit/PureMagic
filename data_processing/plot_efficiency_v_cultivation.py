#!/usr/bin/env python3
"""
Scatter plot of scheduling Efficiency (y) vs 1/magic_state_lambda (x) from
PureMagic output files.  Each -f input becomes one series on the plot.

A single file may contain multiple concatenated runs (one per lambda value),
which is the typical output when sweeping over lambda values for one circuit.

Two forms for -f:
  -f file:label          Plain efficiency plot for that file.
  -f file1,file2:label   Ratio mode: plots efficiency(file1) / efficiency(file2)
                         for each matching inv_lambda value.

If any series uses the comma form, the y-axis label changes to "Efficiency ratio".
Series can mix plain and ratio forms freely.

Usage:
    # Plain efficiency plot
    python plot_efficiency_v_cultivation.py \\
        -f results/out-N25:N25 -f results/out-N49:N49 -o out.png

    # Ratio plot
    python plot_efficiency_v_cultivation.py \\
        -f results/puremagic/out-N25,results/bus/out-N25:PureMagic/Bus \\
        -f results/puremagic/out-N49,results/bus/out-N49:PureMagic/Bus \\
        -o ratio.png
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


def build_lookup(data):
    """Return a dict mapping inv_lambda -> efficiency for fast ratio lookup."""
    return {d["inv_lambda"]: d["efficiency"] for d in data}


def parse_file_arg(arg, default_label):
    """
    Parse a -f argument into (path1, path2_or_None, label).

    Accepted forms:
      file:label           -> (file, None, label)
      file1,file2:label    -> (file1, file2, label)
      file                 -> (file, None, default_label)
      file1,file2          -> (file1, file2, default_label)

    The label is always the part after the last colon.
    The file part (before the last colon) may contain a comma separating two paths.
    """
    if ":" in arg:
        idx = arg.rfind(":")
        file_part = arg[:idx]
        label = arg[idx + 1 :] or default_label
    else:
        file_part = arg
        label = default_label

    if "," in file_part:
        parts = file_part.split(",", 1)
        return parts[0].strip(), parts[1].strip(), label
    return file_part, None, label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of Efficiency (or efficiency ratio) vs "
            "1/magic_state_lambda from PureMagic output files.  Each -f "
            "argument is one series.  Use 'file:label' for plain efficiency or "
            "'file1,file2:label' for the ratio file1/file2."
        )
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        required=True,
        metavar="FILE[:LABEL] or FILE1,FILE2[:LABEL]",
        help=(
            "PureMagic output file(s) for one series.  "
            "Use 'file:label' for plain efficiency, or "
            "'file1,file2:label' to plot the ratio file1/file2.  "
            "Repeat for multiple series."
        ),
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

    any_ratio = False

    for i, file_arg in enumerate(args.files):
        path1, path2, label = parse_file_arg(file_arg, f"file{i + 1}")
        is_ratio = path2 is not None
        if is_ratio:
            any_ratio = True

        data1 = parse_output_file(path1)
        if not data1:
            print(f"Warning: no data found in {path1}", file=sys.stderr)
            continue

        if args.select:
            data1 = [d for d in data1 if args.select in d["circuit"]]
        if not data1:
            print(f"Warning: no matching circuits in {path1}", file=sys.stderr)
            continue

        if is_ratio:
            data2 = parse_output_file(path2)
            if not data2:
                print(f"Warning: no data found in {path2}", file=sys.stderr)
                continue
            denom = build_lookup(data2)
            paired = [
                (d["inv_lambda"], d["efficiency"] / denom[d["inv_lambda"]])
                for d in data1
                if d["inv_lambda"] in denom and denom[d["inv_lambda"]] != 0.0
            ]
            if not paired:
                print(
                    f"Warning: no matching inv_lambda values between {path1} and {path2}",
                    file=sys.stderr,
                )
                continue
            inv_lambdas, y_values = zip(*paired)
            inv_lambdas = list(inv_lambdas)
            y_values = list(y_values)
        else:
            inv_lambdas = [d["inv_lambda"] for d in data1]
            y_values = [d["efficiency"] for d in data1]

        colour = _COLOURS[i % len(_COLOURS)]

        ax.scatter(
            inv_lambdas,
            y_values,
            label=label,
            color=colour,
            edgecolors="black",
            linewidths=0.5,
            s=60,
            zorder=3,
        )

        # Connect points with a line (sorted by x) to show the trend
        sorted_pairs = sorted(zip(inv_lambdas, y_values))
        xs, ys = zip(*sorted_pairs)
        ax.plot(xs, ys, color=colour, linewidth=0.8, alpha=0.6, zorder=2)

    if not any_ratio:
        ax.set_ylim(0, 1)
    ax.set_xscale("log", base=2)
    # Label every power-of-2 tick as "2^k"
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"$2^{{{int(round(math.log2(x)))}}}$")
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Expected cultivation time (cycles)")
    ax.set_ylabel("Efficiency ratio" if any_ratio else "Efficiency")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
