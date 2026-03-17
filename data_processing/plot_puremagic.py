#!/usr/bin/env python3
"""
Unified PureMagic results plotter.

Specify what to plot on each axis:

  -x  circuit | cultivation | parallelism | ancilla_qubits | data_qubits | weight
  -y  scheduling_efficiency | parallel_efficiency | cliffords | timesteps

Each -f argument is one series.  Forms accepted:

  file:label               plain value plot, series labelled 'label'
  file1:l1,file2:l2        ratio file1/file2; y-axis shows 'l1/l2'; series label 'l1/l2'
  file1,file2:label        ratio file1/file2; series labelled 'label'
  file1:l1,file2:l2:label  ratio with per-file labels and explicit series label

When x=circuit a grouped bar chart is produced; all other x choices produce a
scatter plot with a connecting line (cultivation/weight) or plain scatter
(parallelism/ancilla_qubits/data_qubits).

Usage examples:
    # Scheduling efficiency vs circuit name (bar chart)
    python plot_puremagic.py -x circuit -y scheduling_efficiency \\
        -f results/cliffordt/puremagic/out:PureMagic \\
        -f results/cliffordt/bus/out:Bus -o out.png

    # Ratio of scheduling efficiency vs circuit (bar chart)
    python plot_puremagic.py -x circuit -y scheduling_efficiency \\
        -f "results/puremagic/out:PureMagic,results/bus/out:Bus" -o ratio.png

    # Parallel efficiency vs cultivation time (line+scatter)
    python plot_puremagic.py -x cultivation -y parallel_efficiency \\
        -f results/cliffordt/puremagic-vary-cultivation/out-N25:N25 -o out.png
"""

import argparse
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Colours
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

# Conversion factors to microseconds (for schedule_timestep timing)
_TO_US = {"μs": 1.0, "us": 1.0, "ms": 1e3, "s": 1e6}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_output_file(filepath):
    """
    Parse a PureMagic output file and return a list of dicts, one per run.

    Each dict contains all fields that could be extracted; missing fields are
    None.  Fields:
        circuit              str   – circuit name (stem of .schedule file)
        weight               int   – from bare "weight N" line
        inv_lambda           float – 1 / magic_state_lambda from Args block
        scheduling_efficiency float
        parallel_efficiency  float – direct or computed from parallelism/optimal
        parallelism          float – from "Parallelism: Nx"
        cliffords            int   – from "Number of Cliffords: N"
        timesteps            int   – from "Scheduled N in M timesteps"
        data_qubits          int   – from "Number of qubits: / data: N"
        total_qubits         int   – from "Number of qubits: / total: N"
        ancilla_qubits       int   – total_qubits - data_qubits
    """
    results = []

    current_weight = None
    current_lambda = None
    current_circuit = None
    current_cliffords = None
    current_timesteps = None
    current_data_qubits = None
    current_total_qubits = None
    current_parallelism = None
    current_optimal_speedup = None
    current_parallel_efficiency = None
    current_scheduling_efficiency = None
    current_avg_timestep_us = None
    in_qubit_block = False

    def _flush():
        """Emit a record if we have at least a circuit name."""
        nonlocal current_circuit, current_parallelism, current_optimal_speedup
        nonlocal current_parallel_efficiency, current_scheduling_efficiency
        nonlocal current_timesteps, current_cliffords, current_avg_timestep_us
        if current_circuit is None:
            return
        pe = current_parallel_efficiency
        if pe is None and current_parallelism is not None and current_optimal_speedup is not None:
            pe = current_parallelism / current_optimal_speedup
        anc = None
        if current_data_qubits is not None and current_total_qubits is not None:
            anc = current_total_qubits - current_data_qubits
        results.append(
            {
                "circuit": current_circuit,
                "weight": current_weight,
                "inv_lambda": (1.0 / current_lambda) if current_lambda is not None else None,
                "scheduling_efficiency": current_scheduling_efficiency,
                "parallel_efficiency": pe,
                "parallelism": current_parallelism,
                "cliffords": current_cliffords,
                "timesteps": current_timesteps,
                "data_qubits": current_data_qubits,
                "total_qubits": current_total_qubits,
                "ancilla_qubits": anc,
                "timing": current_avg_timestep_us,
            }
        )
        current_circuit = None
        current_parallelism = None
        current_optimal_speedup = None
        current_parallel_efficiency = None
        current_scheduling_efficiency = None
        current_timesteps = None
        current_cliffords = None
        current_avg_timestep_us = None

    with open(filepath, "r") as f:
        for line in f:
            line_clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            # Weight marker (resets per-run state)
            m = re.match(r"^weight\s+(\d+)$", line_clean)
            if m:
                _flush()
                current_weight = int(m.group(1))
                in_qubit_block = False
                continue

            # magic_state_lambda from Args debug block (resets per-run state)
            m = re.match(r"magic_state_lambda:\s*([0-9.eE+\-]+),?", line_clean)
            if m:
                _flush()
                current_lambda = float(m.group(1))
                in_qubit_block = False
                continue

            # Qubit block
            if line_clean == "Number of qubits:":
                in_qubit_block = True
                current_data_qubits = None
                current_total_qubits = None
                continue
            if in_qubit_block:
                m = re.match(r"data:\s+(\d+)", line_clean)
                if m:
                    current_data_qubits = int(m.group(1))
                m = re.match(r"total:\s+(\d+)", line_clean)
                if m:
                    current_total_qubits = int(m.group(1))
                    in_qubit_block = False

            # Number of Cliffords (circuit statistics block)
            m = re.match(r"Number of Cliffords:\s+(\d+)", line_clean)
            if m:
                current_cliffords = int(m.group(1))

            # Circuit name / schedule file
            m = re.match(r"Scheduled products written to (.+)\.schedule", line_clean)
            if m:
                # If we already have a circuit pending, flush it first
                if current_circuit is not None:
                    _flush()
                current_circuit = m.group(1)

            # Scheduled timesteps (appears before "written to" in some output formats)
            m = re.match(r"Scheduled \d+ in (\d+) timesteps", line_clean)
            if m:
                current_timesteps = int(m.group(1))

            # Optimal speedup
            m = re.match(r"Optimal timesteps \d+ \(([0-9.eE+\-]+) speedup\)", line_clean)
            if m and current_circuit is not None:
                current_optimal_speedup = float(m.group(1))

            # Parallelism
            m = re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", line_clean)
            if m and current_circuit is not None:
                current_parallelism = float(m.group(1))

            # Scheduling efficiency
            m = re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                current_scheduling_efficiency = float(m.group(1))

            # Parallel efficiency (direct)
            m = re.match(r"Parallel efficiency:\s+([0-9.eE+\-]+)", line_clean)
            if m and current_circuit is not None:
                current_parallel_efficiency = float(m.group(1))

            # schedule_timestep avg timing (from accumulated timings block)
            m = re.match(
                r"schedule_timestep\s+total:.*avg:\s*([0-9.eE+\-]+)\s*(\S+)\s+max:",
                line_clean,
            )
            if m and current_circuit is not None:
                unit = m.group(2)
                factor = _TO_US.get(unit, 1.0)
                current_avg_timestep_us = float(m.group(1)) * factor

            # End-of-run: flush when we see the timing summary line
            m = re.match(r"Timing: main took", line_clean)
            if m:
                _flush()

    _flush()  # catch last run if no "Timing: main" line
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_lookup(data, x_field, y_field):
    """Return dict mapping x_value -> y_value."""
    return {
        d[x_field]: d[y_field] for d in data if d[x_field] is not None and d[y_field] is not None
    }


def split_file_arg(arg, default_label):
    """
    Parse a -f argument.  Returns (path1, path2_or_None, series_label, ratio_label_or_None).

    ratio_label is a string like "l1/l2" derived from per-file labels; it is used
    to annotate the y-axis when plotting ratios.

    Accepted forms (colon separates file from its per-file label, comma separates
    the two files, an optional trailing :series_label overrides the series label):

      file                         -> (file, None, default_label, None)
      file:label                   -> (file, None, label, None)
      file1,file2                  -> (file1, file2, default_label, None)
      file1,file2:label            -> (file1, file2, label, None)
      file1:l1,file2:l2            -> (file1, file2, "l1/l2", "l1/l2")
      file1:l1,file2:l2:label      -> (file1, file2, label, "l1/l2")

    The rule for the part after the comma:
      - If part1 has a per-file label (contains ':'), then the part after the comma
        is parsed as "file2[:label2][:series_label]" where an extra trailing colon
        segment is the explicit series label.
      - If part1 has no per-file label, the part after the comma is parsed as
        "file2[:series_label]" (one optional trailing colon segment = series label).
    """
    # Split on comma first to detect ratio mode
    if "," in arg:
        comma_idx = arg.index(",")
        part1 = arg[:comma_idx].strip()
        rest = arg[comma_idx + 1 :].strip()

        # part1 may be "file" or "file:label1"
        if ":" in part1:
            colon1 = part1.rfind(":")
            path1 = part1[:colon1].strip()
            label1 = part1[colon1 + 1 :].strip() or None
        else:
            path1 = part1
            label1 = None

        # Parse rest depending on whether part1 had a per-file label.
        # If label1 is set: rest = "file2[:label2[:series_label]]"
        #   - 0 colons: path2=rest, label2=None, series_label=None
        #   - 1 colon:  path2, label2, series_label=None
        #   - 2+ colons: path2, label2, series_label (last colon segment)
        # If label1 is not set: rest = "file2[:series_label]"
        #   - 0 colons: path2=rest, series_label=None
        #   - 1 colon:  path2, series_label
        colon_count = rest.count(":")
        series_label = None
        label2 = None

        if label1 is not None:
            # Expect "file2[:label2[:series_label]]"
            if colon_count >= 2:
                last_colon = rest.rfind(":")
                series_label = rest[last_colon + 1 :].strip() or None
                rest = rest[:last_colon].strip()
            if ":" in rest:
                colon2 = rest.rfind(":")
                path2 = rest[:colon2].strip()
                label2 = rest[colon2 + 1 :].strip() or None
            else:
                path2 = rest
        else:
            # Expect "file2[:series_label]"
            if ":" in rest:
                colon2 = rest.rfind(":")
                path2 = rest[:colon2].strip()
                series_label = rest[colon2 + 1 :].strip() or None
            else:
                path2 = rest

        # Build ratio_label from per-file labels
        if label1 and label2:
            ratio_label = f"{label1}/{label2}"
        else:
            ratio_label = None

        # Build series label: explicit > ratio_label > default
        if series_label is None:
            series_label = ratio_label or default_label

        return path1, path2, series_label, ratio_label

    # No comma — plain single-file form
    if ":" in arg:
        idx = arg.rfind(":")
        path = arg[:idx].strip()
        label = arg[idx + 1 :].strip() or default_label
    else:
        path = arg.strip()
        label = default_label
    return path, None, label, None


def prettify_circuit_name(name):
    """Apply display-friendly substitutions to a circuit name."""
    if name.startswith("qaoa_barabasi_albert"):
        name = "QAOA" + name[len("qaoa_barabasi_albert") :]
    if name.startswith("square_"):
        name = name[len("square_") :]
    return name


def node_count_label(name):
    m = re.search(r"[Nn](\d+)", name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------
# X-axis field mapping
# ---------------------------------------------------------------------------
_X_FIELD = {
    "circuit": "circuit",
    "cultivation": "inv_lambda",
    "parallelism": "parallelism",
    "ancilla_qubits": "ancilla_qubits",
    "data_qubits": "data_qubits",
    "weight": "weight",
}

_X_LABEL = {
    "circuit": "Circuit",
    "cultivation": "Expected Cultivation Time (cycles)",
    "parallelism": "Parallelism",
    "ancilla_qubits": "Routing and Cultivation Overhead (Logical Qubits)",
    "data_qubits": "Logical Qubits",
    "weight": "Max. Transpilation Weight",
}

_Y_FIELD = {
    "scheduling_efficiency": "scheduling_efficiency",
    "parallel_efficiency": "parallel_efficiency",
    "cliffords": "cliffords",
    "timesteps": "timesteps",
    "parallelism": "parallelism",
    "timing": "timing",
}

_Y_LABEL = {
    "scheduling_efficiency": "Scheduling Efficiency",
    "parallel_efficiency": "Parallel Efficiency",
    "cliffords": "Number of Cliffords",
    "timesteps": "Scheduled Cycles",
    "parallelism": "Parallelism",
    "timing": "Average Time per Cycle (μs)",
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified PureMagic results plotter.  Choose x and y axes and "
            "supply one or more -f series files."
        )
    )
    parser.add_argument(
        "-x",
        "--xaxis",
        dest="x_axis",
        choices=list(_X_FIELD),
        required=True,
        help="Variable for the x-axis.",
    )
    parser.add_argument(
        "-y",
        "--yaxis",
        dest="y_axis",
        choices=list(_Y_FIELD),
        required=True,
        help="Variable for the y-axis.",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        required=True,
        metavar="FILE[:LABEL] or FILE1,FILE2[:LABEL]",
        help=(
            "PureMagic output file(s) for one series.  Use 'file:label' for a "
            "plain value plot, or 'file1,file2:label' to plot the ratio "
            "file1/file2.  Repeat for multiple series."
        ),
    )
    parser.add_argument("-o", "--output", required=True, help="Output image file name.")
    parser.add_argument(
        "-s",
        "--select",
        default=None,
        metavar="SUBSTRING",
        help="Only include records whose circuit name contains this substring.",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        default=False,
        help="Connect scatter-plot data points with a line (sorted by x value).",
    )
    parser.add_argument(
        "--hline",
        action="store_true",
        default=False,
        help="Draw a solid black horizontal reference line at y=1 (useful for ratio plots).",
    )
    args = parser.parse_args()

    x_key = args.x_axis
    y_key = args.y_axis
    x_field = _X_FIELD[x_key]
    y_field = _Y_FIELD[y_key]
    is_circuit_x = x_key == "circuit"
    is_cultivation_x = x_key == "cultivation"
    is_weight_x = x_key == "weight"

    # -----------------------------------------------------------------------
    # Load all series
    # -----------------------------------------------------------------------
    series_list = []  # list of (label, x_values, y_values, is_ratio, ratio_label)
    any_ratio = False
    ratio_labels = []  # collect ratio_label strings for y-axis annotation

    for i, file_arg in enumerate(args.files):
        path1, path2, label, ratio_label = split_file_arg(file_arg, f"file{i + 1}")
        is_ratio = path2 is not None

        data1 = parse_output_file(path1)
        if not data1:
            print(f"Warning: no data found in {path1}", file=sys.stderr)
            continue
        if args.select:
            data1 = [d for d in data1 if d.get("circuit") and args.select in d["circuit"]]
        if not data1:
            print(f"Warning: no matching records in {path1}", file=sys.stderr)
            continue

        if is_ratio:
            any_ratio = True
            if ratio_label:
                ratio_labels.append(ratio_label)
            data2 = parse_output_file(path2)
            if not data2:
                print(f"Warning: no data found in {path2}", file=sys.stderr)
                continue
            if args.select:
                data2 = [d for d in data2 if d.get("circuit") and args.select in d["circuit"]]
            lookup2 = build_lookup(data2, x_field, y_field)
            pairs = [
                (d[x_field], d[y_field] / lookup2[d[x_field]])
                for d in data1
                if d[x_field] is not None
                and d[y_field] is not None
                and d[x_field] in lookup2
                and lookup2[d[x_field]] != 0.0
            ]
            if not pairs:
                print(f"Warning: no matching x values between {path1} and {path2}", file=sys.stderr)
                continue
            xs, ys = zip(*pairs)
            series_list.append((label, list(xs), list(ys), True, ratio_label))
        else:
            pairs = [
                (d[x_field], d[y_field])
                for d in data1
                if d[x_field] is not None and d[y_field] is not None
            ]
            if not pairs:
                print(f"Warning: no usable ({x_key}, {y_key}) data in {path1}", file=sys.stderr)
                continue
            xs, ys = zip(*pairs)
            series_list.append((label, list(xs), list(ys), False, None))

    if not series_list:
        print("Error: no data to plot.", file=sys.stderr)
        sys.exit(1)

    # Ensure all series are consistently ratio or non-ratio
    ratio_flags = [s[3] for s in series_list]
    if any(ratio_flags) and not all(ratio_flags):
        print("Error: mix of ratio and non-ratio -f arguments is not allowed.", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    n_series = len(series_list)
    fig, ax = plt.subplots(figsize=(8, 6))

    if is_circuit_x:
        # --- Grouped bar chart ---
        # Collect union of all circuit names, preserving first-seen order
        seen = {}
        for _, xs, _, _, _ in series_list:
            for name in xs:
                if name not in seen:
                    seen[name] = len(seen)
        all_circuits = list(seen.keys())
        n_circuits = len(all_circuits)
        bar_width = 0.8 / max(n_series, 1)
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * bar_width

        for i, (label, xs, ys, _, _) in enumerate(series_list):
            colour = _COLOURS[i % len(_COLOURS)]
            lookup = dict(zip(xs, ys))
            heights = [lookup.get(c, 0.0) for c in all_circuits]
            positions = np.arange(n_circuits) + offsets[i]
            ax.bar(
                positions,
                heights,
                width=bar_width * 0.9,
                label=label,
                color=colour,
                edgecolor="black",
                linewidth=0.4,
            )

        x_pos = np.arange(n_circuits)
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [prettify_circuit_name(c) for c in all_circuits],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_xlabel(_X_LABEL[x_key])

    else:
        # --- Scatter / line plot ---
        draw_lines = args.lines or is_cultivation_x or is_weight_x
        is_timing_y = y_key == "timing"
        for i, (label, xs, ys, is_ratio_series, _) in enumerate(series_list):
            colour = _COLOURS[i % len(_COLOURS)]

            if draw_lines:
                pairs_sorted = sorted(zip(xs, ys))
                xs_plot, ys_plot = zip(*pairs_sorted)
                ax.scatter(
                    xs_plot,
                    ys_plot,
                    color=colour,
                    edgecolors="black",
                    linewidths=0.5,
                    s=60,
                    zorder=3,
                )
                ax.plot(
                    xs_plot, ys_plot, color=colour, linewidth=0.8, alpha=0.6, zorder=2, label=label
                )
            else:
                ax.scatter(
                    xs,
                    ys,
                    label=label,
                    color=colour,
                    edgecolors="black",
                    linewidths=0.5,
                    s=60,
                    zorder=3,
                )

            # Power-law trendline for timing y-axis
            if is_timing_y and len(xs) >= 2:
                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                # Filter out non-positive values (log requires > 0)
                mask = (xs_arr > 0) & (ys_arr > 0)
                if mask.sum() >= 2:
                    log_x = np.log(xs_arr[mask])
                    log_y = np.log(ys_arr[mask])
                    a, b = np.polyfit(log_x, log_y, 1)
                    # R² in log-log space
                    log_y_pred = a * log_x + b
                    ss_res = np.sum((log_y - log_y_pred) ** 2)
                    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                    x_fit = np.linspace(xs_arr[mask].min(), xs_arr[mask].max(), 200)
                    y_fit = np.exp(b) * x_fit**a
                    ax.plot(
                        x_fit,
                        y_fit,
                        color=colour,
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.7,
                        zorder=2,
                        label=f"{label} fit ($x^{{{a:.2f}}}$, R²={r2:.3f})",
                    )

        # x-axis scale for cultivation (log base-2, plain integer labels)
        if is_cultivation_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xlabel(_X_LABEL[x_key])

    # Optional reference line at y=1
    if args.hline:
        ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.0, label="_nolegend_")

    # y-axis label
    y_label = _Y_LABEL[y_key]
    if any_ratio:
        # If all ratio series share the same ratio_label, include it
        unique_ratio_labels = list(dict.fromkeys(ratio_labels))  # deduplicated, order-preserving
        if len(unique_ratio_labels) == 1:
            y_label += f" Ratio ({unique_ratio_labels[0]})"
        else:
            y_label += " Ratio"
    ax.set_ylabel(y_label)

    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
