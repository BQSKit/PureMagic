#!/usr/bin/env python3
"""
Unified PureMagic results plotter.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Constants
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

# X-axis: key -> (DataFrame column name, display label)
_X_AXES = {
    "circuit": ("circuit", "Circuit"),
    "cultivation": ("inv_lambda", "Expected Cultivation Time (cycles)"),
    "parallelism": ("parallelism", "Parallelism"),
    "ancilla_qubits": ("ancilla_qubits", "Routing and Cultivation Overhead (Logical Qubits)"),
    "data_qubits": ("data_qubits", "Logical Qubits"),
    "weight": ("weight", "Max. Transpilation Weight"),
}

# Y-axis: key -> display label  (column name == key)
_Y_AXES = {
    "scheduling_efficiency": "Scheduling Efficiency",
    "parallel_efficiency": "Parallel Efficiency",
    "cliffords": "Number of Cliffords",
    "timesteps": "Scheduled Cycles",
    "parallelism": "Parallelism",
    "timing": "Average Time per Cycle (μs)",
    "total_qubits": "Total Qubits",
    "volume": "Volume (Cycles × Qubits)",
}


# ---------------------------------------------------------------------------
# Series dataclass
# ---------------------------------------------------------------------------
@dataclass
class Series:
    label: str
    xs: list
    ys: list
    circuits: list
    is_ratio: bool = False
    ratio_label: Optional[str] = None
    point_labels: Optional[list] = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_output_file(filepath):
    """
    Parse a PureMagic output file and return a DataFrame, one row per run.

    Columns (NaN where missing):
        circuit, weight, magic_state_lambda, scheduling_efficiency,
        parallel_efficiency, parallelism, cliffords, timesteps,
        data_qubits, total_qubits, loaded_qubits, timing,
        inv_lambda, ancilla_qubits, volume
    """
    rows = []
    cur = {}  # mutable state for the current run
    in_qubit_block = False

    def _flush():
        nonlocal in_qubit_block
        if not cur.get("circuit"):
            return
        pe = cur.get("parallel_efficiency")
        if pe is None and cur.get("parallelism") and cur.get("optimal_speedup"):
            pe = cur["parallelism"] / cur["optimal_speedup"]
        rows.append(
            {
                "circuit": cur.get("circuit"),
                "weight": cur.get("weight"),
                "magic_state_lambda": cur.get("lambda"),
                "scheduling_efficiency": cur.get("scheduling_efficiency"),
                "parallel_efficiency": pe,
                "parallelism": cur.get("parallelism"),
                "cliffords": cur.get("cliffords"),
                "timesteps": cur.get("timesteps"),
                "data_qubits": cur.get("data_qubits"),
                "total_qubits": cur.get("total_qubits"),
                "loaded_qubits": cur.get("loaded_qubits"),
                "timing": cur.get("timing"),
            }
        )
        for k in (
            "circuit",
            "parallelism",
            "optimal_speedup",
            "parallel_efficiency",
            "scheduling_efficiency",
            "timesteps",
            "cliffords",
            "timing",
            "loaded_qubits",
        ):
            cur.pop(k, None)
        in_qubit_block = False

    with open(filepath) as f:
        for line in f:
            s = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()

            if m := re.match(r"^weight\s+(\d+)$", s):
                _flush()
                cur["weight"] = int(m.group(1))
                in_qubit_block = False
                continue

            if m := re.match(r"magic_state_lambda:\s*([0-9.eE+\-]+),?", s):
                _flush()
                cur["lambda"] = float(m.group(1))
                in_qubit_block = False
                continue

            if s == "Number of qubits:":
                in_qubit_block = True
                cur.pop("data_qubits", None)
                cur.pop("total_qubits", None)
                continue

            if in_qubit_block:
                if m := re.match(r"data:\s+(\d+)", s):
                    cur["data_qubits"] = int(m.group(1))
                if m := re.match(r"total:\s+(\d+)", s):
                    cur["total_qubits"] = int(m.group(1))
                    in_qubit_block = False

            if m := re.match(r"Number of Cliffords:\s+(\d+)", s):
                cur["cliffords"] = int(m.group(1))

            if m := re.match(r"Scheduled products written to (.+)\.schedule", s):
                if cur.get("circuit"):
                    _flush()
                cur["circuit"] = m.group(1)

            if m := re.match(r"Loaded circuit with \d+ products and (\d+) qubits", s):
                cur["loaded_qubits"] = int(m.group(1))

            if m := re.match(r"Scheduled \d+ in (\d+) timesteps", s):
                cur["timesteps"] = int(m.group(1))

            if cur.get("circuit"):
                if m := re.match(r"Optimal timesteps \d+ \(([0-9.eE+\-]+) speedup\)", s):
                    cur["optimal_speedup"] = float(m.group(1))
                if m := re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", s):
                    cur["parallelism"] = float(m.group(1))
                if m := re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", s):
                    cur["scheduling_efficiency"] = float(m.group(1))
                if m := re.match(r"Parallel efficiency:\s+([0-9.eE+\-]+)", s):
                    cur["parallel_efficiency"] = float(m.group(1))
                if m := re.match(
                    r"schedule_timestep\s+total:.*avg:\s*([0-9.eE+\-]+)\s*(\S+)\s+max:", s
                ):
                    cur["timing"] = float(m.group(1)) * _TO_US.get(m.group(2), 1.0)

            if re.match(r"Timing: main took", s):
                _flush()

    _flush()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["inv_lambda"] = 1.0 / df["magic_state_lambda"]
    df["ancilla_qubits"] = df["total_qubits"] - df["data_qubits"]
    df["volume"] = df["timesteps"] * df["total_qubits"]
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def split_file_arg(arg, default_label):
    """
    Parse a -f argument.  Returns (path1, path2_or_None, series_label, ratio_label_or_None).

    Accepted forms:
      file                     -> (file, None, default_label, None)
      file:label               -> (file, None, label, None)
      file1,file2              -> (file1, file2, default_label, None)
      file1,file2:label        -> (file1, file2, label, None)
      file1:l1,file2:l2        -> (file1, file2, "l1/l2", "l1/l2")
      file1:l1,file2:l2:label  -> (file1, file2, label, "l1/l2")
    """
    if "," not in arg:
        if ":" in arg:
            idx = arg.rfind(":")
            return arg[:idx].strip(), None, arg[idx + 1 :].strip() or default_label, None
        return arg.strip(), None, default_label, None

    comma_idx = arg.index(",")
    part1, rest = arg[:comma_idx].strip(), arg[comma_idx + 1 :].strip()

    if ":" in part1:
        c = part1.rfind(":")
        path1, label1 = part1[:c].strip(), part1[c + 1 :].strip() or None
    else:
        path1, label1 = part1, None

    series_label = label2 = None
    if label1 is not None:
        if rest.count(":") >= 2:
            lc = rest.rfind(":")
            series_label, rest = rest[lc + 1 :].strip() or None, rest[:lc].strip()
        if ":" in rest:
            c2 = rest.rfind(":")
            path2, label2 = rest[:c2].strip(), rest[c2 + 1 :].strip() or None
        else:
            path2 = rest
    else:
        if ":" in rest:
            c2 = rest.rfind(":")
            path2, series_label = rest[:c2].strip(), rest[c2 + 1 :].strip() or None
        else:
            path2 = rest

    ratio_label = f"{label1}/{label2}" if label1 and label2 else None
    if series_label is None:
        series_label = ratio_label or default_label
    return path1, path2, series_label, ratio_label


def prettify_circuit_name(name):
    """Apply display-friendly substitutions to a circuit name."""
    if name.startswith("qaoa_barabasi_albert"):
        name = "QAOA" + name[len("qaoa_barabasi_albert") :]
    if name.startswith("square_"):
        name = name[len("square_") :]
    return name


def _y_axis_label(y_key, any_ratio, ratio_labels):
    """Return the y-axis display label, appending a ratio suffix when needed."""
    label = _Y_AXES[y_key]
    if any_ratio:
        unique = list(dict.fromkeys(ratio_labels))
        label += f" Ratio ({unique[0]})" if len(unique) == 1 else " Ratio"
    return label


def _combine_legend(ax, ax2=None):
    """Collect handles+labels from ax (and optionally ax2) and attach to ax."""
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified PureMagic results plotter.")
    parser.add_argument("-x", "--xaxis", dest="x_axis", choices=list(_X_AXES), required=True)
    parser.add_argument(
        "-y",
        "--yaxis",
        dest="y_axis",
        required=True,
        metavar=f"{'|'.join(_Y_AXES)} [,{'|'.join(_Y_AXES)}]",
        help="One or two comma-separated y-axis keys (left[,right]).",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        required=True,
        metavar="FILE[:LABEL] or FILE1,FILE2[:LABEL]",
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-s", "--select", default=None, metavar="SUBSTRING")
    parser.add_argument("--lines", action="store_true", default=False)
    parser.add_argument(
        "--lines-with-markers", dest="lines_with_markers", action="store_true", default=False
    )
    parser.add_argument("--hline", action="store_true", default=False)
    parser.add_argument(
        "--logx", action="store_true", default=False, help="Use a log scale on the x-axis."
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        default=False,
        help="Use a log scale on the left y-axis (and right y-axis in dual mode).",
    )
    parser.add_argument(
        "--xlim", default=None, metavar="MIN,MAX", help="Set the x-axis range, e.g. --xlim 0,100."
    )
    parser.add_argument("--ylim", default=None, metavar="MIN,MAX")
    parser.add_argument("--y2lim", default=None, metavar="MIN,MAX")
    parser.add_argument(
        "--label-data-qubits", dest="label_data_qubits", action="store_true", default=False
    )
    args = parser.parse_args()

    x_key = args.x_axis
    x_field, x_label = _X_AXES[x_key]
    is_circuit_x = x_key == "circuit"
    is_cultivation_x = x_key == "cultivation"
    is_weight_x = x_key == "weight"
    is_parallelism_x = x_key == "parallelism"

    # Parse y-axis keys
    y_raw = args.y_axis.strip()
    if "," in y_raw:
        y_keys = [p.strip() for p in y_raw.split(",", 1)]
        if len(y_keys) != 2 or any(k not in _Y_AXES for k in y_keys):
            print(f"Error: invalid -y value '{y_raw}'.", file=sys.stderr)
            sys.exit(1)
        dual_y = True
    else:
        if y_raw not in _Y_AXES:
            print(f"Error: unknown y-axis key '{y_raw}'.", file=sys.stderr)
            sys.exit(1)
        y_keys = [y_raw]
        dual_y = False

    # -----------------------------------------------------------------------
    # load_series: build Series objects for one y-key
    # -----------------------------------------------------------------------
    def _load_df(path):
        df = parse_output_file(path)
        if df.empty:
            print(f"Warning: no data found in {path}", file=sys.stderr)
            return None
        if args.select:
            df = df[df["circuit"].str.contains(args.select, na=False)]
        if df.empty:
            print(f"Warning: no matching records in {path}", file=sys.stderr)
            return None
        return df

    def load_series(y_key, label_suffix=None):
        series_list, any_ratio, ratio_labels = [], False, []

        for i, file_arg in enumerate(args.files):
            path1, path2, label, ratio_label = split_file_arg(file_arg, f"file{i + 1}")
            if label_suffix:
                label = f"{label} ({label_suffix})"
            is_ratio = path2 is not None

            df1 = _load_df(path1)
            if df1 is None:
                continue
            df1 = df1.dropna(subset=[x_field, y_key])

            if is_ratio:
                any_ratio = True
                if ratio_label:
                    ratio_labels.append(ratio_label)
                df2 = _load_df(path2)
                if df2 is None:
                    continue
                df2 = df2.dropna(subset=[x_field, y_key])

                merge_keys = ["circuit"] if is_circuit_x else ["circuit", x_field]
                merged = df1.merge(df2[merge_keys + [y_key]], on=merge_keys, suffixes=("_1", "_2"))
                merged = merged[merged[f"{y_key}_2"] != 0.0]
                if merged.empty:
                    print(
                        f"Warning: no matching points between {path1} and {path2}", file=sys.stderr
                    )
                    continue
                merged["_ratio"] = merged[f"{y_key}_1"] / merged[f"{y_key}_2"]
                series_list.append(
                    Series(
                        label=label,
                        xs=merged[x_field].tolist(),
                        ys=merged["_ratio"].tolist(),
                        circuits=merged["circuit"].tolist(),
                        is_ratio=True,
                        ratio_label=ratio_label,
                    )
                )
            else:
                if df1.empty:
                    print(f"Warning: no usable ({x_key}, {y_key}) data in {path1}", file=sys.stderr)
                    continue
                pt_labels = None
                if args.label_data_qubits and is_parallelism_x:
                    pt_map = df1.set_index(x_field)["loaded_qubits"].to_dict()
                    pt_labels = [
                        str(int(pt_map[x])) if x in pt_map and pd.notna(pt_map[x]) else ""
                        for x in df1[x_field]
                    ]
                series_list.append(
                    Series(
                        label=label,
                        xs=df1[x_field].tolist(),
                        ys=df1[y_key].tolist(),
                        circuits=df1["circuit"].fillna("").tolist(),
                        point_labels=pt_labels,
                    )
                )

        if not series_list:
            print(f"Error: no data to plot for y={y_key}.", file=sys.stderr)
            sys.exit(1)
        ratio_flags = [s.is_ratio for s in series_list]
        if any(ratio_flags) and not all(ratio_flags):
            print("Error: mix of ratio and non-ratio -f arguments.", file=sys.stderr)
            sys.exit(1)
        return series_list, any_ratio, ratio_labels

    # -----------------------------------------------------------------------
    # draw_series: scatter/line plot onto an axes object
    # -----------------------------------------------------------------------
    def draw_series(ax, series_list, y_key, colour_offset=0):
        draw_lines = args.lines or args.lines_with_markers or is_cultivation_x or is_weight_x
        show_markers = args.lines_with_markers or (
            not args.lines and (is_cultivation_x or is_weight_x)
        )
        is_timing_y = y_key == "timing"
        is_total_qubits_y = y_key == "total_qubits"
        is_data_qubits_x = x_key == "data_qubits"
        colour_idx = colour_offset

        for s in series_list:
            colour = _COLOURS[colour_idx % len(_COLOURS)]
            colour_idx += 1

            if draw_lines:
                xs_plot, ys_plot = zip(*sorted(zip(s.xs, s.ys)))
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
                    label=s.label,
                )
            else:
                ax.scatter(
                    s.xs,
                    s.ys,
                    label=s.label,
                    color=colour,
                    edgecolors="black",
                    linewidths=0.5,
                    s=60,
                    zorder=3,
                )

            if s.point_labels:
                for xv, yv, lbl in zip(s.xs, s.ys, s.point_labels):
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
            if is_timing_y and len(s.xs) >= 2:
                xa, ya = np.array(s.xs, float), np.array(s.ys, float)
                mask = (xa > 0) & (ya > 0)
                if mask.sum() >= 2:
                    lx, ly = np.log(xa[mask]), np.log(ya[mask])
                    a, b = np.polyfit(lx, ly, 1)
                    ly_pred = a * lx + b
                    ss_res = np.sum((ly - ly_pred) ** 2)
                    ss_tot = np.sum((ly - ly.mean()) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                    xf = np.linspace(xa[mask].min(), xa[mask].max(), 200)
                    ax.plot(
                        xf,
                        np.exp(b) * xf**a,
                        color=colour,
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.7,
                        zorder=2,
                        label=f"{s.label} fit ($x^{{{a:.2f}}}$, R²={r2:.3f})",
                    )

        # Sqrt ratio fit for data_qubits × total_qubits
        if is_data_qubits_x and is_total_qubits_y:
            rx = ry = None
            if len(series_list) == 1 and series_list[0].is_ratio:
                rx = np.array(series_list[0].xs, float)
                ry = np.array(series_list[0].ys, float)
            elif (
                len(series_list) >= 2
                and not series_list[0].is_ratio
                and not series_list[1].is_ratio
            ):
                s0, s1 = series_list[0], series_list[1]
                mf = pd.DataFrame({"x": s0.xs, "y": s0.ys}).merge(
                    pd.DataFrame({"x": s1.xs, "y": s1.ys}), on="x", suffixes=("_0", "_1")
                )
                mf = mf[mf["y_1"] != 0.0]
                if len(mf) >= 2:
                    rx = mf["x"].to_numpy(float)
                    ry = (mf["y_0"] / mf["y_1"]).to_numpy(float)
            if rx is not None and ry is not None and len(rx) >= 2:
                mask = (rx > 0) & (ry > 0)
                if mask.sum() >= 2:
                    c, r2 = 2.16, 0.99
                    xf = np.linspace(rx[mask].min(), rx[mask].max(), 200)
                    ax.plot(
                        xf,
                        c / np.sqrt(xf) + 1,
                        color="black",
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.8,
                        zorder=2,
                        label=f"ratio fit ($c/\\sqrt{{x}}$, c={c:.2f}, R²={r2:.3f})",
                    )

        return colour_idx

    # -----------------------------------------------------------------------
    # Load all series
    # -----------------------------------------------------------------------
    all_series = [load_series(yk, label_suffix=_Y_AXES[yk] if dual_y else None) for yk in y_keys]

    # -----------------------------------------------------------------------
    # Print data table
    # -----------------------------------------------------------------------
    table_frames, col_names = [], []
    for yk, (series_list, any_ratio, ratio_labels) in zip(y_keys, all_series):
        y_label_base = _y_axis_label(yk, any_ratio, ratio_labels)
        for s in series_list:
            col_name = (
                f"{y_label_base} [{s.label}]" if len(args.files) > 1 or dual_y else y_label_base
            )
            col_names.append(col_name)
            table_frames.append(
                pd.DataFrame({"circuit": s.circuits, x_field: s.xs, col_name: s.ys})
            )

    if table_frames:
        df_table = table_frames[0]
        for tf in table_frames[1:]:
            df_table = df_table.merge(tf, on=["circuit", x_field], how="outer")
        df_table = df_table.sort_values(
            by=[x_field, "circuit"],
            key=lambda col: (
                pd.to_numeric(col, errors="coerce").fillna(col.astype(str))
                if col.name == x_field
                else col
            ),
        )
        for cn in col_names:
            df_table[cn] = df_table[cn].apply(lambda v: f"{v:.6g}" if pd.notna(v) else "N/A")
        df_display = df_table.rename(columns={x_field: x_label, "circuit": "Circuit"})
        if is_circuit_x:
            df_display = df_display.drop(columns=["Circuit"], errors="ignore")
        print("\nData table:")
        print(df_display.to_string(index=False))
        print()

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2 = ax.twinx() if dual_y else None
    axes = [ax, ax2] if dual_y else [ax]

    if is_circuit_x:
        # --- Grouped bar chart ---
        all_circuits: list = []
        for axis_idx, (yk, (series_list, any_ratio, ratio_labels)) in enumerate(
            zip(y_keys, all_series)
        ):
            cur_ax = axes[axis_idx]

            seen: dict = {}
            for s in series_list:
                for name in s.xs:
                    seen.setdefault(name, len(seen))
            all_circuits = list(seen.keys())
            n_circuits = len(all_circuits)

            left_count = len(all_series[0][0]) if dual_y else len(series_list)
            right_count = len(all_series[1][0]) if dual_y else 0
            total_count = left_count + right_count
            bar_width = 0.8 / max(total_count, 1)
            all_offsets = (
                np.linspace(-(total_count - 1) / 2, (total_count - 1) / 2, total_count) * bar_width
            )
            offsets = all_offsets[:left_count] if axis_idx == 0 else all_offsets[left_count:]
            colour_start = 0 if axis_idx == 0 else left_count

            for j, s in enumerate(series_list):
                colour = _COLOURS[(colour_start + j) % len(_COLOURS)]
                lookup = dict(zip(s.xs, s.ys))
                heights = [lookup.get(c, 0.0) for c in all_circuits]
                cur_ax.bar(
                    np.arange(n_circuits) + offsets[j],
                    heights,
                    width=bar_width * 0.9,
                    label=s.label,
                    color=colour,
                    edgecolor="black",
                    linewidth=0.4,
                )

            cur_ax.set_ylabel(_y_axis_label(yk, any_ratio, ratio_labels))

        x_pos = np.arange(len(all_circuits))
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [prettify_circuit_name(c) for c in all_circuits], rotation=45, ha="right", fontsize=8
        )
        ax.set_xlabel(x_label)
        if args.hline:
            ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.0, label="_nolegend_")
        _combine_legend(ax, ax2)

    else:
        # --- Scatter / line plot ---
        colour_offset = 0
        for axis_idx, (yk, (series_list, any_ratio, ratio_labels)) in enumerate(
            zip(y_keys, all_series)
        ):
            cur_ax = axes[axis_idx]
            colour_offset = draw_series(cur_ax, series_list, yk, colour_offset)
            cur_ax.set_ylabel(_y_axis_label(yk, any_ratio, ratio_labels))

        if is_cultivation_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        elif args.logx:
            ax.set_xscale("log")

        if args.logy:
            ax.set_yscale("log")
            if ax2 is not None:
                ax2.set_yscale("log")

        ax.set_xlabel(x_label)
        if args.hline:
            ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=1.0, label="_nolegend_")
        _combine_legend(ax, ax2)

    # Y-axis limits
    if args.xlim:
        ax.set_xlim(*map(float, args.xlim.split(",", 1)))
    if args.ylim:
        ax.set_ylim(*map(float, args.ylim.split(",", 1)))
    if ax2 is not None and args.y2lim:
        ax2.set_ylim(*map(float, args.y2lim.split(",", 1)))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
