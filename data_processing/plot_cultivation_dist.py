#!/usr/bin/env python3
"""
Plot cultivation-time distributions.

Reads one or more directories, collects all *.cultivation_dist files from each,
combines them into a single distribution per directory (summing counts, then
normalizing), and plots all distributions on one figure.

Each .cultivation_dist file has lines of the form:
    <cultivation_time_int>  <normalized_fraction_float>

Usage:
    python plot_cultivation_dist.py DIR1 [DIR2 ...] -o output.pdf [options]
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants (consistent with plot_puremagic.py)
# ---------------------------------------------------------------------------
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

_LABEL_FONTSIZE = 15
_TICK_FONTSIZE = 15
_LEGEND_FONTSIZE = 15


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def read_cultivation_dist(filepath: str) -> dict[int, float]:
    """
    Parse a single .cultivation_dist file.

    Returns a dict mapping cultivation_time (int) -> raw count weight (float).
    The values in the file are already normalized per-file fractions; we treat
    them as unnormalized weights so that files with more events contribute
    proportionally when combined.  (If you want equal-weight combination,
    pass --equal-weight.)
    """
    dist: dict[int, float] = {}
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(
                    f"Warning: {filepath}:{lineno}: expected 2 columns, got {len(parts)} — skipping",
                    file=sys.stderr,
                )
                continue
            try:
                t = int(parts[0])
                frac = float(parts[1])
            except ValueError:
                print(
                    f"Warning: {filepath}:{lineno}: cannot parse '{line}' — skipping",
                    file=sys.stderr,
                )
                continue
            dist[t] = dist.get(t, 0.0) + frac
    return dist


def load_directory(directory: str, equal_weight: bool) -> dict[int, float] | None:
    """
    Read all *.cultivation_dist files in *directory* and combine them into a
    single distribution.

    If equal_weight=True each file contributes equally (its fractions are
    re-normalized to sum to 1 before accumulation).  Otherwise the raw
    fractions are summed directly (files with more events dominate).

    Returns a dict mapping cultivation_time -> combined_fraction (normalized
    so that values sum to 1), or None if no files were found.
    """
    combined: dict[int, float] = defaultdict(float)
    n_files = 0

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".cultivation_dist"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            dist = read_cultivation_dist(fpath)
        except OSError as exc:
            print(f"Warning: cannot read {fpath}: {exc}", file=sys.stderr)
            continue
        if not dist:
            continue

        if equal_weight:
            # Re-normalize this file's distribution to sum to 1 before adding.
            total = sum(dist.values())
            if total > 0:
                dist = {t: v / total for t, v in dist.items()}

        for t, v in dist.items():
            combined[t] += v
        n_files += 1

    if n_files == 0:
        print(f"Warning: no .cultivation_dist files found in '{directory}'", file=sys.stderr)
        return None

    # Normalize the combined distribution so fractions sum to 1.
    total = sum(combined.values())
    if total <= 0:
        print(f"Warning: combined distribution for '{directory}' is empty", file=sys.stderr)
        return None

    return {t: v / total for t, v in combined.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cultivation-time distributions. "
            "Reads all *.cultivation_dist files from each input directory, "
            "combines them into one normalized distribution per directory, "
            "and plots all distributions on a single figure."
        )
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        default=None,
        metavar="DIR[:LABEL]",
        help=(
            "Directory to read (may be given multiple times, one series per directory). "
            "Optionally append :LABEL to override the legend label (default: directory path)."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Output plot file path (e.g. cultivation_dist.pdf or .png).",
    )
    parser.add_argument(
        "--equal-weight",
        dest="equal_weight",
        action="store_true",
        default=False,
        help=(
            "Give each .cultivation_dist file equal weight when combining "
            "(re-normalize each file to sum to 1 before accumulation). "
            "Default: sum raw fractions (files with more events dominate)."
        ),
    )
    parser.add_argument(
        "--bar",
        action="store_true",
        default=False,
        help="Draw distributions as bar charts instead of step lines.",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        default=False,
        help="Use a log-2 scale on the x-axis.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        default=False,
        help="Use a log scale on the y-axis.",
    )
    parser.add_argument(
        "--xlim",
        default=None,
        metavar="MIN,MAX",
        help="Set the x-axis range, e.g. --xlim 0,100.",
    )
    parser.add_argument(
        "--ylim",
        default=None,
        metavar="MIN,MAX",
        help="Set the y-axis range.",
    )
    parser.add_argument(
        "--xlabel",
        default="Cultivation Time (cycles)",
        metavar="LABEL",
        help="X-axis label. Default: 'Cultivation Time (cycles)'.",
    )
    parser.add_argument(
        "--ylabel",
        default="Fraction",
        metavar="LABEL",
        help="Y-axis label. Default: 'Fraction'.",
    )
    parser.add_argument(
        "--figsize",
        default=None,
        metavar="W,H",
        help="Figure size in inches, e.g. --figsize 10,5. Default: 8,4.5.",
    )
    parser.add_argument(
        "--no-legend",
        dest="no_legend",
        action="store_true",
        default=False,
        help="Suppress the legend.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        metavar="ALPHA",
        help="Opacity for bar charts (0–1). Default: 0.7.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate -f
    # -----------------------------------------------------------------------
    if not args.files:
        parser.error("-f/--file is required (may be given multiple times).")

    # -----------------------------------------------------------------------
    # Parse DIR[:LABEL] arguments
    # -----------------------------------------------------------------------
    dir_entries: list[tuple[str, str]] = []  # (directory, label)
    for arg in args.files:
        if ":" in arg:
            idx = arg.rfind(":")
            directory, label = arg[:idx].strip(), arg[idx + 1 :].strip() or arg[:idx].strip()
        else:
            directory, label = arg.strip(), arg.strip()
        dir_entries.append((directory, label))

    # -----------------------------------------------------------------------
    # Load distributions
    # -----------------------------------------------------------------------
    distributions: list[tuple[str, dict[int, float]]] = []  # (label, dist)
    for directory, label in dir_entries:
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
            sys.exit(1)
        dist = load_directory(directory, equal_weight=args.equal_weight)
        if dist is not None:
            distributions.append((label, dist))

    if not distributions:
        print("Error: no data to plot.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\nCombined distribution summary:")
    header = (
        f"{'Label':<40}  {'Min':>6}  {'Max':>6}  {'P10':>8}  {'Median':>8}  {'P90':>8}  {'Files'}"
    )
    print(header)
    print("-" * len(header))
    for directory, label in dir_entries:
        if not os.path.isdir(directory):
            continue
        # Count files
        try:
            n_files = sum(1 for f in os.listdir(directory) if f.endswith(".cultivation_dist"))
        except OSError:
            n_files = "?"
        # Find the matching distribution
        dist_match = next((d for lbl, d in distributions if lbl == label), None)
        if dist_match is None:
            continue
        times_s = sorted(dist_match.keys())
        # Compute p10, median, p90 from the discrete CDF.
        cumsum = 0.0
        p10_s = times_s[0]
        median_s = times_s[-1]
        p90_s = times_s[-1]
        p10_found = median_found = False
        for t in times_s:
            cumsum += dist_match[t]
            if not p10_found and cumsum >= 0.10:
                p10_s = t
                p10_found = True
            if not median_found and cumsum >= 0.50:
                median_s = t
                median_found = True
            if cumsum >= 0.90:
                p90_s = t
                break
        print(
            f"{label:<40}  {times_s[0]:>6}  {times_s[-1]:>6}"
            f"  {p10_s:>8.2f}  {median_s:>8.2f}  {p90_s:>8.2f}  {n_files}"
        )
    print()

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    figsize = (8, 4.5)
    if args.figsize:
        try:
            fw, fh = args.figsize.split(",", 1)
            figsize = (float(fw), float(fh))
        except ValueError:
            print(
                f"Warning: invalid --figsize '{args.figsize}', using default 8,4.5.",
                file=sys.stderr,
            )

    fig, ax = plt.subplots(figsize=figsize)

    for idx, (label, dist) in enumerate(distributions):
        colour = _COLOURS[idx % len(_COLOURS)]
        times = np.array(sorted(dist.keys()), dtype=float)
        fracs = np.array([dist[int(t)] for t in times], dtype=float)

        # Compute the median (p50) of this distribution.
        cumsum = 0.0
        median_val = times[-1]
        for t, frac in zip(times, fracs):
            cumsum += frac
            if cumsum >= 0.50:
                median_val = float(t)
                break

        if args.bar:
            # Bar width: 1 cycle (or scaled for log axis)
            ax.bar(
                times,
                fracs,
                width=0.8,
                label=label,
                color=colour,
                edgecolor="black",
                linewidth=0.3,
                alpha=args.alpha,
                align="center",
            )
        else:
            # Step plot: extend one step beyond the last value for a clean right edge.
            step_x = np.append(times, times[-1] + 1) if len(times) > 0 else times
            step_y = np.append(fracs, 0.0)
            ax.step(
                step_x,
                step_y,
                where="pre",
                color=colour,
                linewidth=2.0,
                alpha=0.85,
                label=label,
            )

        # Compute p10, p90 from the discrete distribution.
        cumsum = 0.0
        p10_val = times[0]
        p90_val = times[-1]
        p10_found = False
        for t, frac in zip(times, fracs):
            cumsum += frac
            if not p10_found and cumsum >= 0.10:
                p10_val = float(t)
                p10_found = True
            if cumsum >= 0.90:
                p90_val = float(t)
                break

        # Vertical line at the median (dashed, same colour, no separate legend entry).
        ax.axvline(
            x=median_val,
            color=colour,
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
            label="_nolegend_",
        )
        # Label the median line near the bottom of the axes.
        ax.text(
            median_val,
            0.0,
            f"  median={median_val:.2f}",
            transform=ax.get_xaxis_transform(),  # x in data coords, y in axes [0,1]
            color=colour,
            fontsize=_TICK_FONTSIZE,
            va="bottom",
            ha="left",
            rotation=90,
        )

        # Vertical line at the 10th percentile (dotted, same colour).
        ax.axvline(
            x=p10_val,
            color=colour,
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="_nolegend_",
        )
        ax.text(
            p10_val,
            0.0,
            f"  p10={p10_val:.2f}",
            transform=ax.get_xaxis_transform(),
            color=colour,
            fontsize=_TICK_FONTSIZE,
            va="bottom",
            ha="left",
            rotation=90,
        )

        # Vertical line at the 90th percentile (dotted, same colour).
        ax.axvline(
            x=p90_val,
            color=colour,
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="_nolegend_",
        )
        ax.text(
            p90_val,
            0.0,
            f"  p90={p90_val:.2f}",
            transform=ax.get_xaxis_transform(),
            color=colour,
            fontsize=_TICK_FONTSIZE,
            va="bottom",
            ha="left",
            rotation=90,
        )

    # -----------------------------------------------------------------------
    # Axes formatting
    # -----------------------------------------------------------------------
    ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(args.xlabel, fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel(args.ylabel, fontsize=_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)

    if args.logx:
        ax.set_xscale("log", base=2)
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if args.logy:
        ax.set_yscale("log")

    if args.xlim:
        try:
            xmin, xmax = args.xlim.split(",", 1)
            ax.set_xlim(float(xmin), float(xmax))
        except ValueError:
            print(f"Warning: invalid --xlim '{args.xlim}', ignoring.", file=sys.stderr)
    if args.ylim:
        try:
            ymin, ymax = args.ylim.split(",", 1)
            ax.set_ylim(float(ymin), float(ymax))
        except ValueError:
            print(f"Warning: invalid --ylim '{args.ylim}', ignoring.", file=sys.stderr)

    if not args.no_legend:
        ax.legend(fontsize=_LEGEND_FONTSIZE)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
