#!/usr/bin/env python3
"""
Unified PureMagic results plotter.

Specify what to plot on each axis:

  -x  circuit | cultivation | parallelism | ancilla_qubits | data_qubits | weight
  -y  KEY  or  KEY_LEFT,KEY_RIGHT
      where KEY is one of: scheduling_efficiency | parallel_efficiency | cliffords
                           | timesteps | parallelism | timing

When two comma-separated y-keys are given, the first is plotted on the left
y-axis and the second on the right y-axis (twin axes).  Each -f file
contributes one series per y-axis.

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

    # Dual y-axes: weight (left) and cliffords (right) vs circuit
    python plot_puremagic.py -x circuit -y weight,cliffords \\
        -f results/cliffordt/puremagic/out:PureMagic -o out.png
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
    current_loaded_qubits = None
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
        nonlocal current_timesteps, current_cliffords, current_avg_timestep_us, current_loaded_qubits
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
                "loaded_qubits": current_loaded_qubits,
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
        current_loaded_qubits = None

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

            # Loaded circuit qubit count
            m = re.match(r"Loaded circuit with \d+ products and (\d+) qubits", line_clean)
            if m:
                current_loaded_qubits = int(m.group(1))

            # Scheduled timesteps (appears before "written to" in some output formats)
            m = re.match(r"Scheduled \d+ in (\d+) timesteps", line_clean)
            if m:
                current_timesteps = int(m.group(1))

            # Optimal speedup (still needed for parallel_efficiency computation)
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
        required=True,
        metavar=f"{'|'.join(_Y_FIELD)} [,{'|'.join(_Y_FIELD)}]",
        help=(
            "Variable(s) for the y-axis.  Supply one key for a single y-axis, "
            "or two comma-separated keys (e.g. 'weight,cliffords') to plot the "
            "first on the left axis and the second on the right axis."
        ),
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
        help="Connect scatter-plot data points with a line (sorted by x value), no markers.",
    )
    parser.add_argument(
        "--lines-with-markers",
        dest="lines_with_markers",
        action="store_true",
        default=False,
        help="Connect scatter-plot data points with a line and also show markers.",
    )
    parser.add_argument(
        "--hline",
        action="store_true",
        default=False,
        help="Draw a solid black horizontal reference line at y=1 (useful for ratio plots).",
    )
    parser.add_argument(
        "--label-data-qubits",
        dest="label_data_qubits",
        action="store_true",
        default=False,
        help=("When x=parallelism, annotate each data point with its number of data qubits."),
    )
    args = parser.parse_args()

    x_key = args.x_axis

    # Parse y-axis: single key or "key_left,key_right"
    y_raw = args.y_axis.strip()
    if "," in y_raw:
        y_parts = [p.strip() for p in y_raw.split(",", 1)]
        if len(y_parts) != 2:
            print("Error: -y accepts at most two comma-separated keys.", file=sys.stderr)
            sys.exit(1)
        for p in y_parts:
            if p not in _Y_FIELD:
                print(
                    f"Error: unknown y-axis key '{p}'. Choose from: {', '.join(_Y_FIELD)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        y_keys = y_parts  # [left_key, right_key]
        dual_y = True
    else:
        if y_raw not in _Y_FIELD:
            print(
                f"Error: unknown y-axis key '{y_raw}'. Choose from: {', '.join(_Y_FIELD)}",
                file=sys.stderr,
            )
            sys.exit(1)
        y_keys = [y_raw]
        dual_y = False

    x_field = _X_FIELD[x_key]
    is_circuit_x = x_key == "circuit"
    is_cultivation_x = x_key == "cultivation"
    is_weight_x = x_key == "weight"
    is_ancilla_x = x_key == "ancilla_qubits"

    # -----------------------------------------------------------------------
    # Helper: load series for one y-key
    # -----------------------------------------------------------------------
    is_parallelism_x = x_key == "parallelism"

    def load_series(y_key, label_suffix=None):
        """
        Returns (series_list, any_ratio, ratio_labels) for the given y_key.
        series_list entries: (label, xs, ys, is_ratio, ratio_label, point_labels)
          point_labels is a list of strings (one per point) or None.

        If label_suffix is given, each series label is appended with " (label_suffix)".
        """
        y_field = _Y_FIELD[y_key]
        series_list = []
        any_ratio = False
        ratio_labels = []

        for i, file_arg in enumerate(args.files):
            path1, path2, label, ratio_label = split_file_arg(file_arg, f"file{i + 1}")
            if label_suffix:
                label = f"{label} ({label_suffix})"
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
                if is_circuit_x:
                    # Match by circuit name (one record per circuit per file)
                    lookup2 = {
                        d["circuit"]: d[y_field]
                        for d in data2
                        if d.get("circuit") is not None and d[y_field] is not None
                    }
                    pairs = [
                        (d[x_field], d[y_field] / lookup2[d["circuit"]])
                        for d in data1
                        if d[x_field] is not None
                        and d[y_field] is not None
                        and d.get("circuit") in lookup2
                        and lookup2[d["circuit"]] != 0.0
                    ]
                else:
                    # Match by x-value (e.g. cultivation time, weight) —
                    # the same circuit appears multiple times with different x-values.
                    lookup2 = {
                        d[x_field]: d[y_field]
                        for d in data2
                        if d[x_field] is not None and d[y_field] is not None
                    }
                    pairs = [
                        (d[x_field], d[y_field] / lookup2[d[x_field]])
                        for d in data1
                        if d[x_field] is not None
                        and d[y_field] is not None
                        and d[x_field] in lookup2
                        and lookup2[d[x_field]] != 0.0
                    ]
                if not pairs:
                    print(
                        f"Warning: no matching data points between {path1} and {path2}",
                        file=sys.stderr,
                    )
                    continue
                xs, ys = zip(*pairs)
                series_list.append((label, list(xs), list(ys), True, ratio_label, None))
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
                # Optionally annotate each point with loaded_qubits for parallelism x-axis
                point_labels = None
                if args.label_data_qubits and is_parallelism_x:
                    pt_map = {
                        d[x_field]: d.get("loaded_qubits") for d in data1 if d[x_field] is not None
                    }
                    point_labels = [str(pt_map[x]) if pt_map.get(x) is not None else "" for x in xs]
                series_list.append((label, list(xs), list(ys), False, None, point_labels))

        if not series_list:
            print(f"Error: no data to plot for y={y_key}.", file=sys.stderr)
            sys.exit(1)

        ratio_flags = [s[3] for s in series_list]
        if any(ratio_flags) and not all(ratio_flags):
            print("Error: mix of ratio and non-ratio -f arguments is not allowed.", file=sys.stderr)
            sys.exit(1)

        return series_list, any_ratio, ratio_labels

    # -----------------------------------------------------------------------
    # Helper: draw series onto an axes object
    # -----------------------------------------------------------------------
    def draw_series(ax, series_list, y_key, colour_offset=0):
        """
        Draw all series in series_list onto ax.
        colour_offset shifts the colour palette so left/right axes use different colours.
        Returns the colour_idx after drawing (for twinx colour continuity).
        """
        # Determine line/marker drawing mode:
        #   draw_lines=True  → connect points with a line
        #   show_markers     → also show scatter markers on top of the line
        draw_lines = args.lines or args.lines_with_markers or is_cultivation_x or is_weight_x
        show_markers = args.lines_with_markers or (
            not args.lines and (is_cultivation_x or is_weight_x)
        )
        is_timing_y = y_key == "timing"
        colour_idx = colour_offset

        for i, (label, xs, ys, is_ratio_series, _, point_labels) in enumerate(series_list):
            colour = _COLOURS[colour_idx % len(_COLOURS)]
            colour_idx += 1

            if draw_lines:
                pairs_sorted = sorted(zip(xs, ys))
                xs_plot, ys_plot = zip(*pairs_sorted)
                if show_markers:
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
                    xs_plot,
                    ys_plot,
                    color=colour,
                    linewidth=1.8,
                    alpha=0.8,
                    linestyle="-",
                    zorder=2,
                    label=label,
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

            # Annotate each point with its data_qubits label if requested
            if point_labels is not None:
                for xv, yv, lbl in zip(xs, ys, point_labels):
                    if lbl:
                        ax.annotate(
                            lbl,
                            (xv, yv),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=7,
                            color=colour,
                        )

            # Power-law trendline for timing y-axis
            if is_timing_y and len(xs) >= 2:
                xs_arr = np.array(xs, dtype=float)
                ys_arr = np.array(ys, dtype=float)
                mask = (xs_arr > 0) & (ys_arr > 0)
                if mask.sum() >= 2:
                    log_x = np.log(xs_arr[mask])
                    log_y = np.log(ys_arr[mask])
                    a, b = np.polyfit(log_x, log_y, 1)
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

        return colour_idx

    # -----------------------------------------------------------------------
    # Load data for each y-axis
    # -----------------------------------------------------------------------
    all_series = []  # list of (series_list, any_ratio, ratio_labels) per y-key
    for yk in y_keys:
        suffix = _Y_LABEL[yk] if dual_y else None
        all_series.append(load_series(yk, label_suffix=suffix))

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = None  # second y-axis (twinx), created only in dual_y mode

    if is_circuit_x:
        # --- Grouped bar chart ---
        # For dual y-axes on a bar chart, we use twinx and alternate bar groups
        axes = [ax]
        if dual_y:
            ax2 = ax.twinx()
            axes.append(ax2)
        all_circuits: list = []
        n_circuits: int = 0

        for axis_idx, (yk, (series_list, any_ratio, ratio_labels)) in enumerate(
            zip(y_keys, all_series)
        ):
            cur_ax = axes[axis_idx]
            n_series = len(series_list)

            # Collect union of circuit names across all series for this axis
            seen = {}
            for _, xs, _, _, _, _ in series_list:
                for name in xs:
                    if name not in seen:
                        seen[name] = len(seen)
            all_circuits = list(seen.keys())
            n_circuits = len(all_circuits)

            # In dual mode, offset bars so left/right don't overlap
            total_series = sum(len(s[0]) for s in all_series)
            bar_width = 0.8 / max(total_series, 1)
            left_count = len(all_series[0][0]) if dual_y else n_series
            right_count = len(all_series[1][0]) if dual_y else 0
            total_count = left_count + right_count
            all_offsets = (
                np.linspace(-(total_count - 1) / 2, (total_count - 1) / 2, total_count) * bar_width
            )
            if axis_idx == 0:
                offsets = all_offsets[:left_count]
                colour_start = 0
            else:
                offsets = all_offsets[left_count:]
                colour_start = left_count

            for j, (label, xs, ys, _, _, _) in enumerate(series_list):
                colour = _COLOURS[(colour_start + j) % len(_COLOURS)]
                lookup = dict(zip(xs, ys))
                heights = [lookup.get(c, 0.0) for c in all_circuits]
                positions = np.arange(n_circuits) + offsets[j]
                cur_ax.bar(
                    positions,
                    heights,
                    width=bar_width * 0.9,
                    label=label,
                    color=colour,
                    edgecolor="black",
                    linewidth=0.4,
                )

            # y-axis label
            y_label = _Y_LABEL[yk]
            if any_ratio:
                unique_ratio_labels = list(dict.fromkeys(ratio_labels))
                if len(unique_ratio_labels) == 1:
                    y_label += f" Ratio ({unique_ratio_labels[0]})"
                else:
                    y_label += " Ratio"
            cur_ax.set_ylabel(y_label)

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

        # Combined legend
        handles, labels_leg = ax.get_legend_handles_labels()
        if dual_y and ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels_leg += l2
        ax.legend(handles, labels_leg)

    else:
        # --- Scatter / line plot ---
        axes = [ax]
        if dual_y:
            ax2 = ax.twinx()
            axes.append(ax2)

        colour_offset = 0
        for axis_idx, (yk, (series_list, any_ratio, ratio_labels)) in enumerate(
            zip(y_keys, all_series)
        ):
            cur_ax = axes[axis_idx]
            colour_offset = draw_series(cur_ax, series_list, yk, colour_offset)

            # y-axis label
            y_label = _Y_LABEL[yk]
            if any_ratio:
                unique_ratio_labels = list(dict.fromkeys(ratio_labels))
                if len(unique_ratio_labels) == 1:
                    y_label += f" Ratio ({unique_ratio_labels[0]})"
                else:
                    y_label += " Ratio"
            cur_ax.set_ylabel(y_label)

        # x-axis scale for cultivation (log base-2, plain integer labels)
        if is_cultivation_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        # x-axis scale for ancilla_qubits (log base-10, plain integer labels)
        # if is_ancilla_x:
        #    ax.set_xscale("log")
        #    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
        #    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        #    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=8))

        ax.set_xlabel(_X_LABEL[x_key])

        # Optional reference line at y=1
        if args.hline:
            ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.0, label="_nolegend_")

        # Combined legend from all axes
        handles, labels_leg = ax.get_legend_handles_labels()
        if dual_y and ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels_leg += l2
        ax.legend(handles, labels_leg)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
