#!/usr/bin/env python3
"""
Plot of scheduled timesteps (left y-axis) and number of Cliffords (right y-axis)
vs circuit weight (x-axis) from PureMagic output files.  Each -f input file
becomes one series on the plot.

A single file may contain multiple concatenated runs (one per weight value),
which is the typical output when sweeping over weight values for one circuit.

The weight is read from a bare "weight N" line that precedes each run's Args block.
Scheduled timesteps come from "Scheduled N in M timesteps".
Number of Cliffords comes from "Number of Cliffords: N" in the circuit statistics.

Usage:
    python plot_timesteps_v_weight.py \\
        -f results/vary-weight/puremagic/out-square_heisenberg_N64:PureMagic \\
        -f results/vary-weight/bus/out-square_heisenberg_N64:Bus \\
        -o timesteps_v_weight.png
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
        {"weight": int, "timesteps": int, "num_cliffords": int}

    weight         — from a bare "weight N" line
    timesteps      — from "Scheduled N in M timesteps, ..."
    num_cliffords  — from "Number of Cliffords: N"
    """
    results = []
    current_weight = None
    current_cliffords = None
    current_timesteps = None

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Weight marker: bare "weight N" line
            m = re.match(r"^weight\s+(\d+)$", line_clean)
            if m:
                current_weight = int(m.group(1))
                current_cliffords = None
                current_timesteps = None

            # Number of Cliffords from circuit statistics
            m = re.match(r"Number of Cliffords:\s+(\d+)", line_clean)
            if m and current_weight is not None:
                current_cliffords = int(m.group(1))

            # Scheduled timesteps: "Scheduled 212340 in 130070 timesteps, volume ..."
            m = re.match(r"Scheduled \d+ in (\d+) timesteps", line_clean)
            if m and current_weight is not None:
                current_timesteps = int(m.group(1))

            # Record once we have all three values
            if (
                current_weight is not None
                and current_cliffords is not None
                and current_timesteps is not None
            ):
                results.append(
                    {
                        "weight": current_weight,
                        "timesteps": current_timesteps,
                        "num_cliffords": current_cliffords,
                    }
                )
                current_weight = None
                current_cliffords = None
                current_timesteps = None

    return results


def split_file_label(arg, default_label):
    """Split 'path:label' into (path, label). Uses the last colon as separator."""
    if ":" in arg:
        idx = arg.rfind(":")
        return arg[:idx], arg[idx + 1 :] or default_label
    return arg, default_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot scheduled timesteps (left y) and number of Cliffords (right y) "
            "vs circuit weight (x) from PureMagic output files.  Each -f argument "
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
        help="Only plot circuits whose name contains this substring (not used here, reserved).",
    )
    args = parser.parse_args()

    n = len(args.files)
    fig, ax_left = plt.subplots(figsize=(max(8, n * 2), 6))
    ax_right = ax_left.twinx()

    # Track handles/labels for a unified legend
    handles, labels = [], []

    for i, file_arg in enumerate(args.files):
        filepath, label = split_file_label(file_arg, f"file{i + 1}")
        data = parse_output_file(filepath)
        if not data:
            print(f"Warning: no data found in {filepath}", file=sys.stderr)
            continue

        data.sort(key=lambda d: d["weight"])
        weights = [d["weight"] for d in data]
        timesteps = [d["timesteps"] for d in data]
        cliffords = [d["num_cliffords"] for d in data]
        colour = _COLOURS[i % len(_COLOURS)]

        # Timesteps on left axis — solid line + filled markers
        (h1,) = ax_left.plot(
            weights,
            timesteps,
            color=colour,
            linewidth=1.5,
            marker="o",
            markersize=5,
            zorder=3,
            label=f"{label} timesteps",
        )
        # Cliffords on right axis — dashed line + open markers
        (h2,) = ax_right.plot(
            weights,
            cliffords,
            color=colour,
            linewidth=1.5,
            linestyle="--",
            marker="s",
            markersize=5,
            markerfacecolor="none",
            zorder=3,
            label=f"{label} Cliffords",
        )
        handles += [h1, h2]
        labels += [h1.get_label(), h2.get_label()]

    ax_left.set_xlabel("Weight")
    ax_left.set_ylabel("Scheduled timesteps")
    ax_right.set_ylabel("Number of Cliffords")

    # Single legend combining both axes
    ax_left.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
