#!/usr/bin/env python3
"""
Plot the ratio of scheduling efficiencies between two PureMagic output files.

Usage:
    python plot_efficiency_ratio.py -f1 <file1>[:<label1>] -f2 <file2>[:<label2>]

The bar plot shows efficiency(file1) / efficiency(file2) for each circuit.
An optional label can be appended after a colon, e.g. "results/out:PureMagic".
The labels are used in the y-axis title and plot title.
"""

import argparse
import re
import sys
import matplotlib.pyplot as plt


def parse_output_file(filepath):
    """
    Parse a PureMagic output file and return a dict mapping
    circuit name -> scheduling efficiency.
    """
    results = {}
    current_circuit = None

    with open(filepath, "r") as f:
        for line in f:
            # Strip ANSI escape codes
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Match circuit name from "Scheduled products written to <name>.schedule"
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                current_circuit = m.group(1)

            # Match scheduling efficiency
            m = re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                efficiency = float(m.group(1))
                results[current_circuit] = efficiency
                current_circuit = None  # reset until next circuit block

    return results


def prettify_circuit_name(name):
    """Apply display-friendly substitutions to a circuit name."""
    if name.startswith("qaoa_barabasi_albert"):
        name = "QAOA" + name[len("qaoa_barabasi_albert") :]
    if name.startswith("square_"):
        name = name[len("square_") :]
    return name


def split_file_label(arg, default_label):
    """Split 'path:label' into (path, label). Colon inside the path is not supported."""
    if ":" in arg:
        # Split on the LAST colon so Windows drive letters (C:\...) still work
        idx = arg.rfind(":")
        return arg[:idx], arg[idx + 1 :] or default_label
    return arg, default_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the ratio of scheduling efficiencies between two PureMagic output files. "
            "Append :<label> to a file argument to set a custom label, e.g. results/out:PureMagic."
        )
    )
    parser.add_argument(
        "-f1",
        "--file1",
        required=True,
        help="First PureMagic output file (numerator), optionally as path:label",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        required=True,
        help="Second PureMagic output file (denominator), optionally as path:label",
    )
    args = parser.parse_args()

    file1, label1 = split_file_label(args.file1, "file1")
    file2, label2 = split_file_label(args.file2, "file2")

    data1 = parse_output_file(file1)
    data2 = parse_output_file(file2)

    # Only plot circuits present in both files, preserving file1 order
    common_circuits = [c for c in data1 if c in data2]
    if not common_circuits:
        print("No common circuits found between the two files.", file=sys.stderr)
        sys.exit(1)

    ratios = [data1[c] / data2[c] for c in common_circuits]

    # --- Plot ---
    # fig, ax = plt.subplots(figsize=(max(8, len(common_circuits) * 0.9), 6))
    _, ax = plt.subplots(figsize=(12, 6))

    x = range(len(common_circuits))
    bars = ax.bar(x, ratios, color="steelblue", edgecolor="black", linewidth=0.7)

    # Draw a horizontal reference line at ratio = 1
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, label="ratio = 1")

    display_names = [prettify_circuit_name(c) for c in common_circuits]
    ax.set_xticks(list(x))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"Scheduling Efficiency Ratio ({label1} / {label2})")
    ax.set_xlabel("Circuit")
    ax.legend()

    # Annotate each bar with its value
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{ratio:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    plt.savefig("efficiency_ratio.png", dpi=150)
    print("Plot saved to efficiency_ratio.png")
    plt.show()


if __name__ == "__main__":
    main()
