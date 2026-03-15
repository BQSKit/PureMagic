#!/usr/bin/env python3
"""
Plot the ratio of scheduling efficiencies between PureMagic output files.

Usage (one series):
    python plot_efficiency_ratio.py -f1 <file>[:<label>] -f2 <file>[:<label>]

Usage (two series, grouped bars):
    python plot_efficiency_ratio.py -f1 <file>[:<label>] -f2 <file>[:<label>] \\
                                    -f3 <file>[:<label>] -f4 <file>[:<label>]

Each series plots efficiency(numerator) / efficiency(denominator).
Append :<label> to any file argument to set a display label,
e.g. "results/out:PureMagic".
"""

import argparse
import re
import sys
import numpy as np
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
    """Split 'path:label' into (path, label). Uses the last colon as separator."""
    if ":" in arg:
        idx = arg.rfind(":")
        return arg[:idx], arg[idx + 1 :] or default_label
    return arg, default_label


def compute_series(file_num, file_den, label_num, label_den, circuit_order=None):
    """
    Load two files, compute ratios, and return (circuits, ratios, series_label).
    If circuit_order is given, use that ordering (and only those circuits).
    """
    data_num = parse_output_file(file_num)
    data_den = parse_output_file(file_den)

    if circuit_order is None:
        circuits = [c for c in data_num if c in data_den]
    else:
        circuits = [c for c in circuit_order if c in data_num and c in data_den]

    if not circuits:
        print(
            f"No common circuits found for series {label_num}/{label_den}.",
            file=sys.stderr,
        )
        sys.exit(1)

    ratios = [data_num[c] / data_den[c] for c in circuits]
    series_label = f"{label_num} / {label_den}"
    return circuits, ratios, series_label


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot scheduling efficiency ratios from PureMagic output files. "
            "Append :<label> to any file argument, e.g. results/out:PureMagic. "
            "Add -f3/-f4 for a second series plotted as grouped bars."
        )
    )
    parser.add_argument(
        "-f1",
        "--file1",
        required=True,
        help="Numerator file for series 1, optionally as path:label",
    )
    parser.add_argument(
        "-f2",
        "--file2",
        required=True,
        help="Denominator file for series 1, optionally as path:label",
    )
    parser.add_argument(
        "-f3",
        "--file3",
        default=None,
        help="Numerator file for series 2, optionally as path:label",
    )
    parser.add_argument(
        "-f4",
        "--file4",
        default=None,
        help="Denominator file for series 2, optionally as path:label",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="efficiency_ratio.png",
        help="Output image file name (default: efficiency_ratio.png)",
    )
    args = parser.parse_args()

    if (args.file3 is None) != (args.file4 is None):
        parser.error("Both -f3 and -f4 must be provided together for a second series.")

    file1, label1 = split_file_label(args.file1, "file1")
    file2, label2 = split_file_label(args.file2, "file2")

    # Series 1 — defines circuit order
    circuits1, ratios1, series_label1 = compute_series(file1, file2, label1, label2)

    two_series = args.file3 is not None
    ratios2, series_label2 = None, None  # set below when two_series is True
    if two_series:
        file3, label3 = split_file_label(args.file3, "file3")
        file4, label4 = split_file_label(args.file4, "file4")
        # Series 2 — use series-1 circuit order
        circuits2, ratios2, series_label2 = compute_series(
            file3, file4, label3, label4, circuit_order=circuits1
        )
        # Restrict to circuits present in both series
        common = [c for c in circuits1 if c in circuits2]
        idx1 = [circuits1.index(c) for c in common]
        idx2 = [circuits2.index(c) for c in common]
        ratios1 = [ratios1[i] for i in idx1]
        ratios2 = [ratios2[i] for i in idx2]
        circuits = common
    else:
        circuits = circuits1

    display_names = [prettify_circuit_name(c) for c in circuits]
    n = len(circuits)
    x = np.arange(n)

    # --- Plot ---
    _, ax = plt.subplots(figsize=(max(12, n * 0.5), 6))

    # Draw a horizontal reference line at ratio = 1
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2, label="_nolegend_")

    if two_series:
        assert ratios2 is not None and series_label2 is not None
        width = 0.4
        bars1 = ax.bar(
            x - width / 2,
            ratios1,
            width,
            label=series_label1,
            color="steelblue",
            edgecolor="black",
            linewidth=0.7,
        )
        bars2 = ax.bar(
            x + width / 2,
            ratios2,
            width,
            label=series_label2,
            color="darkorange",
            edgecolor="black",
            linewidth=0.7,
        )
        for bar, ratio in list(zip(bars1, ratios1)) + list(zip(bars2, ratios2)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{ratio:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_ylabel("Scheduling Efficiency Ratio")
    else:
        bars = ax.bar(
            x, ratios1, color="steelblue", edgecolor="black", linewidth=0.7, label=series_label1
        )
        for bar, ratio in zip(bars, ratios1):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{ratio:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylabel(f"Scheduling Efficiency Ratio ({series_label1})")

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Circuit")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
