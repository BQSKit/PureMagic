#!/usr/bin/env python3
"""
Unified PureMagic results plotter.
"""

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
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

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Conversion factors to microseconds (for schedule_lcycle timing)
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

# Y-axis: key -> display label  (column name == key, unless overridden in _Y_FIELD)
_Y_AXES = {
    "scheduling_efficiency": "Scheduling Efficiency",
    "scheduling_efficiency_loss": "Scheduling Efficiency Loss",
    "parallel_efficiency": "Parallel Efficiency",
    "cliffords": "Number of Cliffords",
    "lcycles": "Scheduled Logical Cycles",
    "parallelism": "Parallelism",
    "timing": "Average Time per Cycle (μs)",
    "total_qubits": "Total Qubits",
    "ancilla_qubits": "Area",
    "volume": "Volume",
    "volume_loss": "Volume Loss",
    "max_parallelism": "Max Parallelism",
    "cultivation": "Average Cultivation Time (cycles)",
}

# Y-axis keys whose DataFrame column name differs from the key itself
_Y_FIELD = {
    "cultivation": "avg_cultivation_time",
    # scheduling_efficiency_loss reads the same column but applies 1-x (see _Y_INVERT)
    "scheduling_efficiency_loss": "scheduling_efficiency",
    "volume_loss": "volume",
}

# Y-axis keys whose plotted value is (1 - raw_value) in non-ratio mode, or
# (1 - ratio) in ratio mode.
_Y_INVERT = {"scheduling_efficiency_loss", "volume_loss"}


# ---------------------------------------------------------------------------
# Series dataclass
# ---------------------------------------------------------------------------
@dataclass
class Series:
    label: Optional[str]
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
        parallel_efficiency, parallelism, cliffords, lcycles,
        data_qubits, total_qubits, magic_qubits, loaded_qubits, timing,
        avg_cultivation_time, inv_lambda, ancilla_qubits, volume, max_parallelism
    """
    rows = []
    cur = {}  # mutable state for the current run
    in_qubit_block = False
    in_cultivation_block = False

    def _flush():
        nonlocal in_qubit_block, in_cultivation_block
        if not cur.get("circuit"):
            return
        pe = cur.get("parallel_efficiency")
        if pe is None and cur.get("parallelism") and cur.get("optimal_speedup"):
            pe = cur["parallelism"] / cur["optimal_speedup"]
        # Compute scheduling efficiency as optimal_lcycles / lcycles (ignore parsed value).
        se = None
        if cur.get("optimal_lcycles") is not None and cur.get("lcycles"):
            se = cur["optimal_lcycles"] / cur["lcycles"]
        rows.append(
            {
                "circuit": cur.get("circuit"),
                "weight": cur.get("weight"),
                "magic_state_lambda": cur.get("lambda"),
                "scheduling_efficiency": se,
                "parallel_efficiency": pe,
                "parallelism": cur.get("parallelism"),
                "cliffords": cur.get("cliffords"),
                "lcycles": cur.get("lcycles"),
                "data_qubits": cur.get("data_qubits"),
                "total_qubits": cur.get("total_qubits"),
                "magic_qubits": cur.get("magic_qubits"),
                "loaded_qubits": cur.get("loaded_qubits"),
                "timing": cur.get("timing"),
                "avg_cultivation_time": cur.get("avg_cultivation_time"),
            }
        )
        for k in (
            "circuit",
            "parallelism",
            "optimal_speedup",
            "optimal_lcycles",
            "parallel_efficiency",
            "scheduling_efficiency",
            "lcycles",
            "cliffords",
            "timing",
            "loaded_qubits",
            "avg_cultivation_time",
        ):
            cur.pop(k, None)
        in_qubit_block = False
        in_cultivation_block = False

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
                if m := re.match(r"magic:\s+(\d+)", s):
                    cur["magic_qubits"] = int(m.group(1))
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

            if m := re.match(r"Scheduled \d+ in (\d+) logical cycles", s):
                cur["lcycles"] = int(m.group(1))

            if s == "Magic state cultivation time:":
                in_cultivation_block = True
                continue

            if in_cultivation_block:
                if m := re.match(r"average:\s+([0-9.eE+\-]+)", s):
                    cur["avg_cultivation_time"] = float(m.group(1))
                    in_cultivation_block = False

            if cur.get("circuit"):
                if m := re.match(r"Optimal logical cycles (\d+) \(([0-9.eE+\-]+) speedup\)", s):
                    cur["optimal_lcycles"] = int(m.group(1))
                    cur["optimal_speedup"] = float(m.group(2))
                if m := re.match(r"Parallelism:\s+([0-9.eE+\-]+)x", s):
                    cur["parallelism"] = float(m.group(1))
                if m := re.match(r"Scheduling efficiency:\s+([0-9.eE+\-]+)", s):
                    cur["scheduling_efficiency"] = float(m.group(1))
                if m := re.match(r"Parallel efficiency:\s+([0-9.eE+\-]+)", s):
                    cur["parallel_efficiency"] = float(m.group(1))
                if m := re.match(
                    r"schedule_lcycle\s+total:.*avg:\s*([0-9.eE+\-]+)\s*(\S+)\s+max:", s
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
    df["volume"] = df["lcycles"] * df["total_qubits"]
    df["max_parallelism"] = df["magic_qubits"] * df["magic_state_lambda"]
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
            return arg[:idx].strip(), None, arg[idx + 1 :].strip() or None, None
        return arg.strip(), None, None, None

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
        series_label = ratio_label if ratio_label is not None else default_label
    return path1, path2, series_label, ratio_label


def prettify_circuit_name(name):
    """Apply display-friendly substitutions to a circuit name."""
    # Strip leading path components (keep only the basename without extension)
    name = os.path.basename(name)
    # Strip all known intermediate extensions (e.g. "foo.pkl" -> "foo")
    while True:
        root, ext = os.path.splitext(name)
        if ext.lower() in (".pkl", ".qasm", ".trans", ".schedule", ".txt"):
            name = root
        else:
            break
    # Strip leading weight prefix "mN." (e.g. "m1.", "m23.")
    name = re.sub(r"^m\d+\.", "", name)

    # qaoa_barabasi_albert_N<n>_3reps -> QAOA(<n>)
    m = re.match(r"qaoa_barabasi_albert_N(\d+)_3reps(.*)", name, re.IGNORECASE)
    if m:
        return f"QAOA({m.group(1)}){m.group(2)}"

    # qv_N<n>_12345 -> QV(<n>)
    m = re.match(r"qv_N(\d+)_\d+(.*)", name, re.IGNORECASE)
    if m:
        return f"QV({m.group(1)}){m.group(2)}"

    # strip square_ prefix
    if name.startswith("square_"):
        name = name[len("square_") :]

    # Simple prefix/whole-name uppercasing substitutions (case-insensitive)
    _PREFIX_MAP = [
        (r"dnn", "DNN"),
        (r"qft", "QFT"),
        (r"knn", "KNN"),
        (r"ghz(?:_state)?", "GHZ"),
        (r"vqe_uccsd", "VQE"),
    ]
    for pattern, replacement in _PREFIX_MAP:
        name = re.sub(rf"(?i)^{pattern}(?=_|$)", replacement, name)

    # heisenberg -> heis
    name = re.sub(r"(?i)heisenberg", "heis", name)

    # separate trailing _n<digits> or _N<digits> qubit count: foo_n8 -> foo(8)
    name = re.sub(r"[_-][nN](\d+)$", lambda mo: f"({mo.group(1)})", name)

    return name


def _y_axis_label(y_key, any_ratio, pct_improvement=False):
    label = _Y_AXES[y_key]
    if any_ratio:
        if pct_improvement:
            label += " % Improvement"
        else:
            label += " Ratio"
    return label


def _axis_label(yk_list, any_ratio, pct_improvement=False):
    return " / ".join(_y_axis_label(yk, any_ratio, pct_improvement) for yk in yk_list)


def _ordered_union_xs(series_list):
    """Return the union of all xs across series, in first-encounter order."""
    seen = {}
    for s in series_list:
        for xv in s.xs:
            seen.setdefault(xv, len(seen))
    return list(seen.keys())


_LABEL_FONTSIZE = 15  # ~50 % larger than the default 10 pt
_TICK_FONTSIZE = 15  # ~50 % larger than the default 10 pt
_LEGEND_FONTSIZE = 15  # ~50 % larger than the default 10 pt


def _combine_legend(ax, ax2=None):
    """Collect handles+labels from ax (and optionally ax2) and attach to ax."""
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    # Filter out entries explicitly marked as no-legend
    filtered = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, fontsize=_LEGEND_FONTSIZE)
    else:
        ax.legend(handles, labels, fontsize=_LEGEND_FONTSIZE)


# ---------------------------------------------------------------------------
# WISQ helpers
# ---------------------------------------------------------------------------


def _dascot_square_sparse_total(n: int) -> int:
    """Return the total logical qubit count for the Square Sparse layout with n data qubits."""
    s = math.ceil(math.sqrt(n))
    inner_side = 2 * s + 1
    final_side = inner_side + 2  # = 2s + 3
    return final_side * final_side


def parse_wisq_dir(directory: str) -> list:
    """
    Read all *.json files in *directory* and return a list of
    (circuit_name, volume) tuples, where:
      circuit_name  = filename stem stripped of '-wisq' suffix
      volume        = dascot_square_sparse_total(n_data) * len(steps)
    """
    entries = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            n_data = len(data["map"])
            n_steps = len(data["steps"])
            area = _dascot_square_sparse_total(n_data)
            volume = area * n_steps
            # Derive a canonical circuit name from the filename
            stem = fname[: -len(".json")]
            # Strip trailing "-wisq" or "_wisq" suffix if present
            stem = re.sub(r"[-_]wisq$", "", stem, flags=re.IGNORECASE)
            entries.append((stem, volume))
        except Exception as exc:
            print(f"Warning: could not parse WISQ file {fpath}: {exc}", file=sys.stderr)
    return entries


# ---------------------------------------------------------------------------
# FLASQ plot
# ---------------------------------------------------------------------------


def parse_flasq_file(filepath):
    """
    Parse a concatenated FLASQ output file (produced by flasq_lower_bound.py)
    and return a list of (circuit_name, vol_conservative, vol_optimistic, n_data_qubits) tuples.

    The file consists of blocks of the form:
        ========================================================================
        FLASQ Lower Bound for <filepath>
          Layout: ...
        ========================================================================
          Max simultaneous qubit usage (Q)          : <Q>
          ...
          FLASQ spacetime volume (S, blocks)        :      <cons>       <opt>
        ========================================================================

    The circuit name is taken from the "FLASQ Lower Bound for" header line,
    Q from the "Max simultaneous qubit usage" row, and the two volumes from
    the "FLASQ spacetime volume" row.
    """
    entries = []
    current_circuit = None
    current_q = None
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            # Header: "FLASQ Lower Bound for <path>"
            m = re.match(r"^FLASQ Lower Bound for\s+(.+)$", s)
            if m:
                current_circuit = m.group(1).strip()
                current_q = None
                continue
            # Q row: "Max simultaneous qubit usage (Q)  :  <Q>"
            m = re.match(r"Max simultaneous qubit usage \(Q\)\s*:\s*(\d+)", s)
            if m and current_circuit is not None:
                current_q = int(m.group(1))
                continue
            # Volume row: "FLASQ spacetime volume (S, blocks)  :  <cons>  <opt>"
            m = re.match(
                r"FLASQ spacetime volume \(S, blocks\)\s*:\s*([0-9.eE+\-]+)\s+([0-9.eE+\-]+)",
                s,
            )
            if m and current_circuit is not None:
                try:
                    vol_cons = float(m.group(1))
                    vol_opt = float(m.group(2))
                    entries.append((current_circuit, vol_cons, vol_opt, current_q))
                except ValueError:
                    pass
                current_circuit = None
                current_q = None
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified PureMagic results plotter.")
    parser.add_argument(
        "--flasq",
        dest="flasq_file",
        default=None,
        metavar="FLASQ_FILE",
        help=(
            "Path to a concatenated FLASQ output file (produced by running "
            "flasq_lower_bound.py on multiple circuits and concatenating the output). "
            "When given, a FLASQ volume plot is produced and all other arguments "
            "except -o / --output, --logy, and --ylabel are ignored."
        ),
    )
    parser.add_argument(
        "--wisq",
        dest="wisq_dir",
        default=None,
        metavar="WISQ_DIR",
        help=(
            "Path to a directory containing WISQ *.json output files. "
            "When given, the WISQ volume (square-sparse area × steps) is computed "
            "for each circuit and plotted as an additional series. "
            "Forces the y-axis to 'volume'."
        ),
    )
    parser.add_argument("-x", "--xaxis", dest="x_axis", choices=list(_X_AXES))
    parser.add_argument(
        "-y",
        "--yaxis",
        dest="y_axis",
        default=None,
        metavar=f"KEY[/KEY...] or KEY,KEY",
        help=(
            "Y-axis key(s).  Three forms are accepted:\n"
            "  key              – single series on one axis\n"
            "  key1/key2[/...]  – multiple keys on the same (left) axis\n"
            "  key1,key2        – key1 on left axis, key2 on right axis (dual-y)\n"
            f"Valid keys: {', '.join(_Y_AXES)}"
        ),
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="files",
        action="append",
        default=None,
        metavar="FILE[:LABEL] or FILE1,FILE2[:LABEL]",
    )
    parser.add_argument("-o", "--output", required=True, help="Output plot file path.")
    parser.add_argument(
        "-c",
        "--circuits",
        dest="circuits_file",
        default=None,
        metavar="CIRCUITS_FILE",
        help="Path to a file listing allowed circuit names (one per line, basenames without extension). Only data for circuits in this file will be plotted.",
    )
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
    parser.add_argument(
        "--xlabel", default=None, metavar="LABEL", help="Override the x-axis label."
    )
    parser.add_argument(
        "--ylabel", default=None, metavar="LABEL", help="Override the left y-axis label."
    )
    parser.add_argument("--ylim", default=None, metavar="MIN,MAX")
    parser.add_argument("--y2lim", default=None, metavar="MIN,MAX")
    parser.add_argument(
        "--label-data-qubits", dest="label_data_qubits", action="store_true", default=False
    )
    parser.add_argument(
        "--percent-improvement",
        dest="percent_improvement",
        action="store_true",
        default=False,
        help=(
            "When plotting a ratio (file1,file2 form), show %% improvement on the y-axis "
            "instead of the raw ratio. Computed as (1 - file1/file2) * 100, so positive "
            "values mean file2 is better (smaller metric)."
        ),
    )
    parser.add_argument(
        "--stackedbar",
        action="store_true",
        default=False,
        help=(
            "When -y is slash-separated and -f uses ratio (file1,file2) form, "
            "render a stacked bar chart instead of a scatter/line plot."
        ),
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate -c (always required)
    # -----------------------------------------------------------------------
    if args.circuits_file is None:
        parser.error("-c/--circuits is required.")

    # -----------------------------------------------------------------------
    # FLASQ mode: force -y volume, default -x to circuit if not given
    # -----------------------------------------------------------------------
    if args.flasq_file is not None:
        if not args.files:
            parser.error("-f/--file is required when --flasq is given (provides scheduled volume).")
        if args.x_axis is None:
            args.x_axis = "circuit"
        args.y_axis = "volume"

    # -----------------------------------------------------------------------
    # WISQ mode: force -y volume
    # -----------------------------------------------------------------------
    if args.wisq_dir is not None:
        args.y_axis = "volume"
        if args.x_axis is None:
            args.x_axis = "circuit"

    # -----------------------------------------------------------------------
    # Normal mode — validate remaining required args
    # -----------------------------------------------------------------------
    if args.x_axis is None:
        parser.error("-x/--xaxis is required when --flasq/--wisq is not given.")
    if args.y_axis is None:
        parser.error("-y/--yaxis is required when --flasq/--wisq is not given.")
    if not args.files and args.wisq_dir is None:
        parser.error("-f/--file is required (or use --wisq to supply a WISQ directory).")
    # Allow --wisq-only mode (no -f files)
    if not args.files:
        args.files = []

    # Load the allowed circuits set from the -c file
    try:
        with open(args.circuits_file) as _cf:
            _allowed_circuits = {
                line.strip() for line in _cf if line.strip() and not line.startswith("#")
            }
    except OSError as e:
        print(f"Error: cannot read circuits file '{args.circuits_file}': {e}", file=sys.stderr)
        sys.exit(1)

    x_key = args.x_axis
    x_field, x_label = _X_AXES[x_key]
    if args.xlabel:
        x_label = args.xlabel
    is_circuit_x = x_key == "circuit"
    is_cultivation_x = x_key == "cultivation"
    is_weight_x = x_key == "weight"
    is_parallelism_x = x_key == "parallelism"

    y_raw = args.y_axis.strip()
    dual_y = False
    if "," in y_raw:
        y_keys = [p.strip() for p in y_raw.split(",", 1)]
        if len(y_keys) != 2 or any(k not in _Y_AXES for k in y_keys):
            print(
                f"Error: invalid -y value '{y_raw}'. Dual-y requires exactly two valid keys separated by ','.",
                file=sys.stderr,
            )
            sys.exit(1)
        multi_y_keys = [[y_keys[0]], [y_keys[1]]]
        dual_y = True
    elif "/" in y_raw:
        # multi-key same-axis mode
        parts = [p.strip() for p in y_raw.split("/")]
        bad = [p for p in parts if p not in _Y_AXES]
        if bad:
            print(
                f"Error: unknown y-axis key(s): {', '.join(bad)}. Valid: {', '.join(_Y_AXES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        multi_y_keys = [parts]
    else:
        if y_raw not in _Y_AXES:
            print(
                f"Error: unknown y-axis key '{y_raw}'. Valid: {', '.join(_Y_AXES)}", file=sys.stderr
            )
            sys.exit(1)
        multi_y_keys = [[y_raw]]

    # When -y uses slash form (multiple keys on the same axis), each key is paired
    # with its own -f argument.  Validate that the counts match.
    slash_y = not dual_y and len(multi_y_keys) == 1 and len(multi_y_keys[0]) > 1
    if slash_y:
        n_keys = len(multi_y_keys[0])
        n_files = len(args.files)
        if n_files != n_keys:
            print(
                f"Error: -y has {n_keys} slash-separated keys but {n_files} -f argument(s) "
                f"were given.  Provide exactly one -f per key.",
                file=sys.stderr,
            )
            sys.exit(1)

    def _load_df(path):
        df = parse_output_file(path)
        if df.empty:
            print(f"Warning: no data found in {path}", file=sys.stderr)
            return None
        if args.select:
            df = df[df["circuit"].str.contains(args.select, na=False)]

        # Filter to only circuits listed in the -c circuits file.
        # Strip all extensions (e.g. "m1.square_heisenberg_N64.pkl" -> strip ".pkl" -> "m1.square_heisenberg_N64")
        # and also strip a leading weight prefix like "m1." or "m23." before comparing.
        def _canonical_circuit_name(c):
            name = os.path.basename(str(c))
            # Strip all dot-separated extensions that are not part of the circuit name
            while True:
                root, ext = os.path.splitext(name)
                if ext.lower() in (".pkl", ".qasm", ".trans", ".schedule", ".txt"):
                    name = root
                else:
                    break
            # Strip leading weight prefix "mN." (e.g. "m1.", "m23.")
            name = re.sub(r"^m\d+\.", "", name)
            return name

        df = df[df["circuit"].apply(lambda c: _canonical_circuit_name(c) in _allowed_circuits)]
        if df.empty:
            print(f"Warning: no matching records in {path}", file=sys.stderr)
            return None
        return df

    def load_series(y_keys_for_axis, label_suffix=None):
        """
        Build Series objects for one axis.  y_keys_for_axis is a list of one or more
        y-axis column names.

        Normal mode (single key, or dual-y): each (file × key) pair produces its own
        Series on the same axis.

        Slash mode (multiple keys on the same axis, slash_y=True): each key is paired
        with its own -f argument (key[i] reads from files[i]).  The number of -f
        arguments must equal the number of keys (validated above).
        """
        series_list, any_ratio, ratio_labels = [], False, []

        if slash_y:
            # Pair each key with its corresponding file argument.
            pairs = list(zip(y_keys_for_axis, args.files))
        else:
            # Original behaviour: every file × every key.
            pairs = [(y_key, file_arg) for file_arg in args.files for y_key in y_keys_for_axis]

        for y_key, file_arg in pairs:
            path1, path2, file_label, ratio_label = split_file_arg(file_arg, "")
            is_ratio = path2 is not None

            df1 = _load_df(path1)
            if df1 is None:
                continue

            df2 = None
            if is_ratio:
                df2 = _load_df(path2)
                if df2 is None:
                    continue

            # Build a per-series label: include y-key name when multiple keys share an axis,
            # but in slash mode each key has its own file label so just use that label directly.
            multi_key = len(y_keys_for_axis) > 1
            if file_label is None:
                label = None
            elif slash_y:
                # In slash mode each key is paired with its own -f; use the file label as-is.
                if label_suffix:
                    label = f"{file_label} ({label_suffix})"
                else:
                    label = file_label
            elif multi_key and label_suffix:
                label = f"{file_label} {_Y_AXES[y_key]} ({label_suffix})"
            elif multi_key:
                label = f"{file_label} {_Y_AXES[y_key]}"
            elif label_suffix:
                label = f"{file_label} ({label_suffix})"
            else:
                label = file_label

            y_field = _Y_FIELD.get(y_key, y_key)
            d1 = df1.dropna(subset=[x_field, y_field])

            if is_ratio:
                any_ratio = True
                if ratio_label and ratio_label not in ratio_labels:
                    ratio_labels.append(ratio_label)
                assert (
                    df2 is not None
                )  # guaranteed: is_ratio=True and _load_df returned non-None (else continued)
                d2 = df2.dropna(subset=[x_field, y_field])
                merge_keys = ["circuit"] if is_circuit_x else ["circuit", x_field]
                merged = d1.merge(d2[merge_keys + [y_field]], on=merge_keys, suffixes=("_1", "_2"))
                merged = merged[merged[f"{y_field}_2"] != 0.0]
                if merged.empty:
                    print(
                        f"Warning: no matching points between {path1} and {path2} for y={y_key}",
                        file=sys.stderr,
                    )
                    continue
                merged["_ratio"] = merged[f"{y_field}_1"] / merged[f"{y_field}_2"]
                if y_key in _Y_INVERT:
                    # For loss keys: plot 1 - ratio
                    ys_values = (1.0 - merged["_ratio"]).tolist()
                elif args.percent_improvement:
                    # ys_values = ((1.0 - merged["_ratio"]) * 100.0).tolist()
                    ys_values = ((merged["_ratio"] - 1) * 100.0).tolist()
                else:
                    ys_values = merged["_ratio"].tolist()
                series_list.append(
                    Series(
                        label=label,
                        xs=merged[x_field].tolist(),
                        ys=ys_values,
                        circuits=merged["circuit"].tolist(),
                        is_ratio=True,
                        ratio_label=ratio_label,
                    )
                )
            else:
                if d1.empty:
                    print(
                        f"Warning: no usable ({x_key}, {y_key}) data in {path1}",
                        file=sys.stderr,
                    )
                    continue
                pt_labels = None
                if args.label_data_qubits and is_parallelism_x:
                    pt_map = d1.set_index(x_field)["loaded_qubits"].to_dict()
                    pt_labels = [
                        str(int(pt_map[x])) if x in pt_map and pd.notna(pt_map[x]) else ""
                        for x in d1[x_field]
                    ]
                raw_ys = d1[y_field].tolist()
                if y_key in _Y_INVERT:
                    ys_vals = [1.0 - v for v in raw_ys]
                else:
                    ys_vals = raw_ys
                series_list.append(
                    Series(
                        label=label,
                        xs=d1[x_field].tolist(),
                        ys=ys_vals,
                        circuits=d1["circuit"].fillna("").tolist(),
                        point_labels=pt_labels,
                    )
                )

        if not series_list:
            if args.wisq_dir is not None:
                print(
                    f"Warning: no -f data to plot for y={y_keys_for_axis}; will plot WISQ only.",
                    file=sys.stderr,
                )
                return [], False, []
            print(f"Error: no data to plot for y={y_keys_for_axis}.", file=sys.stderr)
            sys.exit(1)
        ratio_flags = [s.is_ratio for s in series_list]
        if any(ratio_flags) and not all(ratio_flags):
            print("Error: mix of ratio and non-ratio -f arguments.", file=sys.stderr)
            sys.exit(1)
        return series_list, any_ratio, ratio_labels

    def draw_series(ax, series_list, yk_list, colour_offset=0):
        y_key = yk_list[0] if isinstance(yk_list, list) else yk_list
        draw_lines = args.lines or args.lines_with_markers or is_cultivation_x or is_weight_x
        show_markers = args.lines_with_markers or (
            not args.lines and (is_cultivation_x or is_weight_x)
        )
        is_timing_y = y_key == "timing"
        is_total_qubits_y = y_key == "total_qubits"
        is_ancilla_qubits_y = "ancilla_qubits" in (
            yk_list if isinstance(yk_list, list) else [yk_list]
        )
        is_data_qubits_x = x_key == "data_qubits"
        colour_idx = colour_offset

        for s in series_list:
            colour = _COLOURS[colour_idx % len(_COLOURS)]
            marker = _MARKERS[colour_idx % len(_MARKERS)]
            colour_idx += 1

            if draw_lines:
                xs_plot, ys_plot = zip(*sorted(zip(s.xs, s.ys)))
                if show_markers:
                    ax.scatter(
                        xs_plot,
                        ys_plot,
                        marker=marker,
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
                    linewidth=2.7 if not show_markers else 1.8,
                    alpha=0.8,
                    linestyle="-",
                    marker=marker if show_markers else None,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=2,
                    label=s.label if s.label is not None else "_nolegend_",
                )
            else:
                ax.scatter(
                    s.xs,
                    s.ys,
                    label=s.label if s.label is not None else "_nolegend_",
                    marker=marker,
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
                            fontsize=12,
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
                    print(
                        f"{s.label} fit ($c \\cdot x^{{{a:.2f}}}$, c={np.exp(b):.2f}, R²={r2:.3f})"
                    )
                    ax.plot(
                        xf,
                        np.exp(b) * xf**a,
                        color=colour,
                        linewidth=1.2,
                        linestyle="--",
                        alpha=0.7,
                        zorder=2,
                        label=f"Fit ($c \\cdot x^{{{a:.2f}}}$, c={np.exp(b):.2f}, R²={r2:.3f})",
                    )

        # Fit y = A * x^(-0.5) on the ancilla_qubits ratio for data_qubits x-axis.
        # Works for any y-axis that includes ancilla_qubits (e.g. ancilla_qubits, volume/ancilla_qubits).
        # The fit is always applied to the ratio of ancilla_qubits between two series (or the
        # already-computed ratio/percent-improvement values when a ratio series is present).
        if is_data_qubits_x and is_ancilla_qubits_y:
            rx = ry = None
            # Case 1: a pre-computed ratio series exists (file1,file2 form) — ys are already
            # the ratio or percent-improvement values.
            ratio_series = [s for s in series_list if s.is_ratio]
            non_ratio_series = [s for s in series_list if not s.is_ratio]
            if ratio_series:
                # Use the first ratio series (there is typically only one per axis)
                s = ratio_series[0]
                rx = np.array(s.xs, float)
                ry = np.array(s.ys, float)
            elif len(non_ratio_series) >= 2:
                # Compute the ancilla_qubits ratio from the first two non-ratio series,
                # matching on x (data_qubits).
                s0, s1 = non_ratio_series[0], non_ratio_series[1]
                mf = pd.DataFrame({"x": s0.xs, "y": s0.ys}).merge(
                    pd.DataFrame({"x": s1.xs, "y": s1.ys}), on="x", suffixes=("_0", "_1")
                )
                mf = mf[mf["y_1"] != 0.0]
                if len(mf) >= 2:
                    rx = mf["x"].to_numpy(float)
                    raw_ratio = (mf["y_0"] / mf["y_1"]).to_numpy(float)
                    if args.percent_improvement:
                        ry = (raw_ratio - 1.0) * 100.0
                    else:
                        ry = raw_ratio
            if rx is not None and ry is not None and len(rx) >= 2:
                mask = (rx > 0) & np.isfinite(ry)
                if mask.sum() >= 2:
                    sx, sy = rx[mask], ry[mask]
                    # Least-squares fit for y = A / sqrt(x):
                    # minimise sum((y - A/sqrt(x))^2)  =>  A = sum(y/sqrt(x)) / sum(1/x)
                    A = np.sum(sy / np.sqrt(sx)) / np.sum(1.0 / sx)
                    # A *= 0.48
                    y_pred = A / np.sqrt(sx)
                    ss_res = np.sum((sy - y_pred) ** 2)
                    ss_tot = np.sum((sy - sy.mean()) ** 2)
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                    xf = np.linspace(sx.min(), sx.max(), 200)
                    print(f"fit ($A/\\sqrt{{x}}$, A={A:.2f}, R²={r2:.3f})")
                    ax.plot(
                        xf,
                        A / np.sqrt(xf),
                        color="black",
                        linewidth=2.0,
                        linestyle="--",
                        alpha=0.8,
                        zorder=2,
                        label=f"Area fit $\\alpha/\\sqrt{{x}}$",
                        # label="_nolegend_",
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

    # multi_y_keys[axis_idx] = list of y-keys for that axis;
    # for dual-y each list has one key; for multi-key same-axis the list has many keys.
    # When --wisq is used without any -f files, skip load_series entirely.
    if args.files:
        all_series = [
            load_series(yk_list, label_suffix=_Y_AXES[yk_list[0]] if dual_y else None)
            for yk_list in multi_y_keys
        ]
    else:
        # wisq-only mode: no -f series; create empty placeholders
        all_series = [([], False, []) for _ in multi_y_keys]

    # Print data table
    table_frames, col_names = [], []
    for yk_list, (series_list, any_ratio, ratio_labels) in zip(multi_y_keys, all_series):
        # Use a combined label for the axis when multiple keys share it
        y_label_base = _axis_label(yk_list, any_ratio)
        for s in series_list:
            col_name = (
                f"{y_label_base} [{s.label}]"
                if len(args.files) > 1 or dual_y or len(yk_list) > 1
                else y_label_base
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

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx() if dual_y else None
    axes = [ax, ax2] if dual_y else [ax]
    all_circuits: list = []  # populated in the circuit-x branch; used by FLASQ overlay

    if is_circuit_x:
        # --- Grouped bar chart ---
        for axis_idx, (yk_list, (series_list, any_ratio, ratio_labels)) in enumerate(
            zip(multi_y_keys, all_series)
        ):
            cur_ax = axes[axis_idx]

            if series_list:
                all_circuits = _ordered_union_xs(series_list)
            # (if series_list is empty, all_circuits stays [] until WISQ overlay fills it)

            left_count = len(all_series[0][0]) if dual_y else len(series_list)
            right_count = len(all_series[1][0]) if dual_y else 0
            total_count = left_count + right_count
            bar_width = 0.8 / max(total_count, 1)
            all_offsets = (
                np.linspace(-(total_count - 1) / 2, (total_count - 1) / 2, total_count) * bar_width
                if total_count > 0
                else np.array([0.0])
            )
            offsets = all_offsets[:left_count] if axis_idx == 0 else all_offsets[left_count:]
            colour_start = 0 if axis_idx == 0 else left_count

            for j, s in enumerate(series_list):
                colour = _COLOURS[(colour_start + j) % len(_COLOURS)]
                lookup = dict(zip(s.xs, s.ys))
                heights = [lookup.get(c, 0.0) for c in all_circuits]
                cur_ax.bar(
                    np.arange(len(all_circuits)) + offsets[j],
                    heights,
                    width=bar_width * 0.9,
                    label=s.label if s.label is not None else "_nolegend_",
                    color=colour,
                    edgecolor="black",
                    linewidth=0.4,
                )

            cur_ax.set_ylabel(
                (
                    args.ylabel
                    if (axis_idx == 0 and args.ylabel)
                    else _axis_label(yk_list, any_ratio, args.percent_improvement)
                ),
                fontsize=_LABEL_FONTSIZE,
            )
            cur_ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)

        # x-axis ticks are set after WISQ overlay (which may populate all_circuits)
        _deferred_circuit_x_setup = True

    else:
        # Check whether to use a stacked bar chart:
        # condition: slash-separated y-keys (multi-key, single axis, not dual-y) + ratio mode.
        _yk_list_0 = multi_y_keys[0]
        _series_0, _any_ratio_0, _ratio_labels_0 = all_series[0]
        is_stacked_ratio = args.stackedbar and (not dual_y) and len(_yk_list_0) > 1 and _any_ratio_0

        if is_stacked_ratio:
            # --- Stacked bar chart for multi-key ratio, non-circuit x-axis ---
            # In normal mode series_list is ordered [file0_key0, file0_key1, ..., file1_key0, ...].
            # In slash mode each key is paired with its own file, so series_list is ordered
            # [key0_file0, key1_file1, ...] — one entry per key, treated as n_files=1 group.
            # Each Series has xs = x-axis values, ys = ratio values.
            # We draw one group of stacked bars per unique x value.
            series_list = _series_0
            yk_list = _yk_list_0
            any_ratio = _any_ratio_0
            ratio_labels = _ratio_labels_0

            if slash_y:
                # One series per key; treat as a single file-group with all keys stacked.
                n_files = 1
                n_keys = len(yk_list)
            else:
                n_files = len(args.files)
                n_keys = len(yk_list)

            all_x_vals = _ordered_union_xs(series_list)
            n_x = len(all_x_vals)

            bar_width = 0.8 / max(n_files, 1)
            file_offsets = (
                np.linspace(-(n_files - 1) / 2, (n_files - 1) / 2, n_files) * bar_width
                if n_files > 1
                else np.array([0.0])
            )
            _HATCHES = ["", "//", "xx", ".."]

            for fi in range(n_files):
                bottoms = np.zeros(n_x)
                prev_heights = np.zeros(n_x)
                for ki, y_key in enumerate(yk_list):
                    si = fi * n_keys + ki
                    if si >= len(series_list):
                        break
                    s = series_list[si]
                    colour = _COLOURS[ki % len(_COLOURS)]
                    lookup = dict(zip(s.xs, s.ys))
                    heights = np.array([lookup.get(xv, 0.0) for xv in all_x_vals])
                    # Each segment's drawn height = this key's value minus the previous key's
                    # value, so the total bar top equals the last key's ratio value.
                    seg_heights = heights - prev_heights
                    seg_label = (
                        _y_axis_label(y_key, any_ratio, args.percent_improvement)
                        if fi == 0
                        else "_nolegend_"
                    )
                    ax.bar(
                        np.arange(n_x) + file_offsets[fi],
                        seg_heights,
                        bottom=bottoms,
                        width=bar_width * 0.9,
                        label=seg_label,
                        color=colour,
                        edgecolor="black",
                        linewidth=0.4,
                        hatch=_HATCHES[fi % len(_HATCHES)],
                    )
                    bottoms += seg_heights
                    prev_heights = heights

                if n_files > 1:
                    file_label = split_file_arg(args.files[fi], f"file{fi + 1}")[2]
                    ax.bar(
                        [],
                        [],
                        color="white",
                        edgecolor="black",
                        linewidth=0.4,
                        hatch=_HATCHES[fi % len(_HATCHES)],
                        label=file_label if file_label is not None else "_nolegend_",
                    )

            ax.set_ylabel(
                (
                    args.ylabel
                    if args.ylabel
                    else _axis_label(yk_list, any_ratio, args.percent_improvement)
                ),
                fontsize=_LABEL_FONTSIZE,
            )
            ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)

            # X-axis: use the actual x values as tick labels.
            ax.set_xticks(np.arange(n_x))
            x_tick_labels = [str(xv) for xv in all_x_vals]
            ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=_TICK_FONTSIZE)
            ax.set_xlim(-0.6, n_x - 0.4)
            ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
            if args.hline:
                hline_y = 0.0 if args.percent_improvement else 1.0
                ax.axhline(
                    y=hline_y, color="grey", linestyle=":", linewidth=1.0, label="_nolegend_"
                )
            _combine_legend(ax, ax2)

        else:
            # --- Scatter / line plot ---
            colour_offset = 0
            for axis_idx, (yk_list, (series_list, any_ratio, ratio_labels)) in enumerate(
                zip(multi_y_keys, all_series)
            ):
                cur_ax = axes[axis_idx]
                # draw_series uses y_key only for special trendline logic; pass the first key
                # (trendlines are per-series and keyed by the series' own y_key name)
                colour_offset = draw_series(cur_ax, series_list, yk_list, colour_offset)
                cur_ax.set_ylabel(
                    (
                        args.ylabel
                        if (axis_idx == 0 and args.ylabel)
                        else _axis_label(yk_list, any_ratio, args.percent_improvement)
                    ),
                    fontsize=_LABEL_FONTSIZE,
                )
                cur_ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)

            if is_cultivation_x:
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(round(x))}"))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            elif args.logx:
                ax.set_xscale("log")

            # Apply log2 scale to any y-axis whose key is "cultivation";
            # otherwise honour --logy for that axis.
            for axis_idx, yk_list in enumerate(multi_y_keys):
                cur_ax = axes[axis_idx]
                if "cultivation" in yk_list:
                    cur_ax.set_yscale("log", base=2)
                    cur_ax.yaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda y, _: f"{int(round(y))}")
                    )
                    cur_ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                elif args.logy:
                    cur_ax.set_yscale("log")

            ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
            ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
            if args.hline:
                hline_y = 0.0 if args.percent_improvement else 1.0
                ax.axhline(
                    y=hline_y, color="grey", linestyle=":", linewidth=1.0, label="_nolegend_"
                )
            _combine_legend(ax, ax2)
        _deferred_circuit_x_setup = False

    # -----------------------------------------------------------------------
    # WISQ overlay / ratio mode
    # -----------------------------------------------------------------------
    if args.wisq_dir is not None:
        wisq_entries = parse_wisq_dir(args.wisq_dir)

        # Filter by allowed circuits
        def _wisq_canonical(stem):
            name = stem
            name = re.sub(r"^m\d+\.", "", name)
            return name

        wisq_entries = [e for e in wisq_entries if _wisq_canonical(e[0]) in _allowed_circuits]
        if args.select:
            wisq_entries = [e for e in wisq_entries if args.select in _wisq_canonical(e[0])]

        if not wisq_entries:
            print("Warning: no WISQ entries matched the allowed circuits.", file=sys.stderr)
        else:
            wisq_map = {_wisq_canonical(e[0]): e[1] for e in wisq_entries}

            has_f_series = any(len(sl) > 0 for sl, _, _ in all_series)

            if has_f_series:
                # -----------------------------------------------------------------
                # Ratio mode: replace each -f series with wisq_volume / series_volume
                # Only circuits present in both the series and wisq_map are shown.
                # Clear the axes and redraw.
                # -----------------------------------------------------------------
                ax.cla()
                if ax2 is not None:
                    ax2.cla()

                if args.percent_improvement:
                    ratio_y_label = args.ylabel if args.ylabel else "WISQ Volume % Overhead"
                else:
                    ratio_y_label = args.ylabel if args.ylabel else "WISQ Volume / Volume"

                def _apply_pct(ratio):
                    """Convert raw ratio to percent improvement if requested."""
                    return (ratio - 1.0) * 100.0 if args.percent_improvement else ratio

                if is_circuit_x:
                    # Collect ratio series: for each -f series, compute per-circuit ratios
                    ratio_series_data = []  # list of (label, {circuit: ratio})
                    for yk_list, (series_list, any_ratio, ratio_labels) in zip(
                        multi_y_keys, all_series
                    ):
                        for s in series_list:
                            lookup = dict(zip(s.xs, s.ys))
                            ratio_map = {}
                            for c, vol in lookup.items():
                                wisq_vol = wisq_map.get(c)
                                if wisq_vol is not None and vol and vol != 0:
                                    ratio_map[c] = _apply_pct(wisq_vol / vol)
                            ratio_series_data.append((s.label, ratio_map))

                    # all_circuits: union of circuits present in all ratio series AND wisq_map
                    ratio_circuits = [
                        c
                        for c in (all_circuits if all_circuits else list(wisq_map.keys()))
                        if any(c in rd for _, rd in ratio_series_data)
                    ]
                    all_circuits[:] = ratio_circuits  # update in-place for deferred setup

                    n_c = len(ratio_circuits)
                    n_s = len(ratio_series_data)
                    bar_width_r = 0.8 / max(n_s, 1)
                    offsets_r = (
                        np.linspace(-(n_s - 1) / 2, (n_s - 1) / 2, n_s) * bar_width_r
                        if n_s > 1
                        else np.array([0.0])
                    )
                    for j, (lbl, ratio_map) in enumerate(ratio_series_data):
                        colour = _COLOURS[j % len(_COLOURS)]
                        heights_r = [ratio_map.get(c, 0.0) for c in ratio_circuits]
                        ax.bar(
                            np.arange(n_c) + offsets_r[j],
                            heights_r,
                            width=bar_width_r * 0.9,
                            label=lbl if lbl is not None else "_nolegend_",
                            color=colour,
                            edgecolor="black",
                            linewidth=0.4,
                        )

                    ax.set_ylabel(ratio_y_label, fontsize=_LABEL_FONTSIZE)
                    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
                    if args.logy:
                        ax.set_yscale("log")
                    if args.hline:
                        hline_y = 0.0 if args.percent_improvement else 1.0
                        ax.axhline(
                            y=hline_y,
                            color="black",
                            linestyle="-",
                            linewidth=1.0,
                            label="_nolegend_",
                        )

                else:
                    # Scatter / line ratio mode for non-circuit x-axis
                    draw_lines_r = args.lines or args.lines_with_markers
                    colour_idx = 0
                    for yk_list, (series_list, any_ratio_f, ratio_labels) in zip(
                        multi_y_keys, all_series
                    ):
                        for s in series_list:
                            colour = _COLOURS[colour_idx % len(_COLOURS)]
                            marker = _MARKERS[colour_idx % len(_MARKERS)]
                            colour_idx += 1
                            xs_r, ys_r = [], []
                            for xv, yv, circ in zip(s.xs, s.ys, s.circuits):
                                wisq_vol = wisq_map.get(circ)
                                if wisq_vol is not None and yv and yv != 0:
                                    xs_r.append(xv)
                                    ys_r.append(_apply_pct(wisq_vol / yv))
                            if not xs_r:
                                continue
                            if draw_lines_r:
                                pairs_r = sorted(zip(xs_r, ys_r))
                                xp_r, yp_r = zip(*pairs_r)
                                ax.plot(
                                    xp_r,
                                    yp_r,
                                    color=colour,
                                    linewidth=2.7 if not args.lines_with_markers else 1.8,
                                    alpha=0.8,
                                    linestyle="-",
                                    marker=marker if args.lines_with_markers else None,
                                    markersize=6,
                                    markeredgecolor="black",
                                    markeredgewidth=0.5,
                                    zorder=2,
                                    label=s.label if s.label is not None else "_nolegend_",
                                )
                            else:
                                ax.scatter(
                                    xs_r,
                                    ys_r,
                                    label=s.label if s.label is not None else "_nolegend_",
                                    marker=marker,
                                    color=colour,
                                    edgecolors="black",
                                    linewidths=0.5,
                                    s=60,
                                    zorder=3,
                                )

                    ax.set_ylabel(ratio_y_label, fontsize=_LABEL_FONTSIZE)
                    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
                    ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
                    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
                    if args.logy:
                        ax.set_yscale("log")
                    if args.hline:
                        hline_y = 0.0 if args.percent_improvement else 1.0
                        ax.axhline(
                            y=hline_y,
                            color="grey",
                            linestyle=":",
                            linewidth=1.0,
                            label="_nolegend_",
                        )

                _combine_legend(ax, ax2)

            else:
                # -----------------------------------------------------------------
                # WISQ-only mode: plot absolute WISQ volumes
                # -----------------------------------------------------------------
                wisq_colour = _COLOURS[0]
                wisq_marker = _MARKERS[0]

                if is_circuit_x:
                    if not all_circuits:
                        all_circuits = [_wisq_canonical(e[0]) for e in wisq_entries]
                    heights = [wisq_map.get(c, 0.0) for c in all_circuits]
                    ax.bar(
                        np.arange(len(all_circuits)),
                        heights,
                        width=0.8,
                        label="WISQ",
                        color=wisq_colour,
                        edgecolor="black",
                        linewidth=0.4,
                    )
                    ax.set_ylabel(
                        args.ylabel if args.ylabel else _Y_AXES["volume"],
                        fontsize=_LABEL_FONTSIZE,
                    )
                    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
                    if args.logy:
                        ax.set_yscale("log")
                    if args.hline:
                        ax.axhline(
                            y=1.0, color="black", linestyle="-", linewidth=1.0, label="_nolegend_"
                        )
                else:
                    wisq_entries_full = []
                    for fname in sorted(os.listdir(args.wisq_dir)):
                        if not fname.endswith(".json"):
                            continue
                        fpath = os.path.join(args.wisq_dir, fname)
                        try:
                            with open(fpath) as f:
                                data = json.load(f)
                            stem = fname[: -len(".json")]
                            stem = re.sub(r"[-_]wisq$", "", stem, flags=re.IGNORECASE)
                            if _wisq_canonical(stem) not in _allowed_circuits:
                                continue
                            if args.select and args.select not in _wisq_canonical(stem):
                                continue
                            n_data = len(data["map"])
                            n_steps = len(data["steps"])
                            area = _dascot_square_sparse_total(n_data)
                            volume = area * n_steps
                            wisq_entries_full.append((stem, n_data, volume))
                        except Exception:
                            pass

                    if x_key == "data_qubits":
                        xs_w = [e[1] for e in wisq_entries_full]
                        ys_w = [e[2] for e in wisq_entries_full]
                    else:
                        xs_w = [e[0] for e in wisq_entries_full]
                        ys_w = [e[2] for e in wisq_entries_full]

                    draw_lines_wisq = args.lines or args.lines_with_markers
                    if draw_lines_wisq:
                        pairs_w = sorted(zip(xs_w, ys_w))
                        xp_w, yp_w = zip(*pairs_w) if pairs_w else ([], [])
                        ax.plot(
                            xp_w,
                            yp_w,
                            color=wisq_colour,
                            linewidth=2.7,
                            alpha=0.8,
                            linestyle="-",
                            marker=wisq_marker if args.lines_with_markers else None,
                            markersize=6,
                            markeredgecolor="black",
                            markeredgewidth=0.5,
                            zorder=2,
                            label="WISQ",
                        )
                    else:
                        ax.scatter(
                            xs_w,
                            ys_w,
                            label="WISQ",
                            marker=wisq_marker,
                            color=wisq_colour,
                            edgecolors="black",
                            linewidths=0.5,
                            s=60,
                            zorder=3,
                        )
                    ax.set_ylabel(
                        args.ylabel if args.ylabel else _Y_AXES["volume"],
                        fontsize=_LABEL_FONTSIZE,
                    )
                    ax.tick_params(axis="y", labelsize=_TICK_FONTSIZE)
                    ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
                    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
                    if args.logy:
                        ax.set_yscale("log")

                _combine_legend(ax, ax2)

    # -----------------------------------------------------------------------
    # Deferred circuit-x axis setup (applied after WISQ overlay so all_circuits is final)
    # -----------------------------------------------------------------------
    if _deferred_circuit_x_setup and all_circuits:
        x_pos = np.arange(len(all_circuits))
        ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [prettify_circuit_name(c) for c in all_circuits], rotation=45, ha="right", fontsize=12
        )
        ax.set_xlabel(x_label, fontsize=_LABEL_FONTSIZE)
        if args.logy:
            ax.set_yscale("log")
            if ax2 is not None:
                ax2.set_yscale("log")
        if args.hline:
            hline_y = 0.0 if args.percent_improvement else 1.0
            ax.axhline(y=hline_y, color="black", linestyle="-", linewidth=1.0, label="_nolegend_")
        _combine_legend(ax, ax2)

    # -----------------------------------------------------------------------
    # FLASQ overlay: add conservative and optimistic series on top of the plot
    # -----------------------------------------------------------------------
    if args.flasq_file is not None:

        def _flasq_canonical(path):
            name = os.path.basename(path)
            while True:
                root, ext = os.path.splitext(name)
                if ext.lower() in (
                    ".pkl",
                    ".qasm",
                    ".trans",
                    ".schedule",
                    ".txt",
                    ".cliffordt",
                    ".compiled",
                ):
                    name = root
                else:
                    break
            name = re.sub(r"^m\d+\.", "", name)
            return name

        flasq_entries = parse_flasq_file(args.flasq_file)
        # Filter by allowed circuits and --select
        try:
            with open(args.circuits_file) as _cf:
                _flasq_allowed = {l.strip() for l in _cf if l.strip() and not l.startswith("#")}
        except OSError:
            _flasq_allowed = set()
        flasq_entries = [e for e in flasq_entries if _flasq_canonical(e[0]) in _flasq_allowed]
        if args.select:
            flasq_entries = [e for e in flasq_entries if args.select in _flasq_canonical(e[0])]

        colour_cons = _COLOURS[len(args.files) % len(_COLOURS)]
        colour_opt = _COLOURS[(len(args.files) + 1) % len(_COLOURS)]

        def _plot_flasq_series(xs, ys, label, colour):
            """Draw FLASQ series as a dashed line (no markers), sorted by x."""
            if not xs:
                return
            pairs = sorted(zip(xs, ys))
            xp, yp = zip(*pairs)
            ax.plot(
                xp,
                yp,
                color=colour,
                linewidth=2.0,
                alpha=0.85,
                linestyle="--",
                zorder=3,
                label=label,
            )

        if flasq_entries and is_circuit_x:
            flasq_map = {_flasq_canonical(e[0]): (e[1], e[2]) for e in flasq_entries}
            xs_cons, ys_cons, xs_opt, ys_opt = [], [], [], []
            for xi, c in enumerate(all_circuits):
                entry = flasq_map.get(c) or flasq_map.get(_flasq_canonical(c))
                if entry:
                    xs_cons.append(xi)
                    ys_cons.append(entry[0])
                    xs_opt.append(xi)
                    ys_opt.append(entry[1])
            _plot_flasq_series(xs_cons, ys_cons, "FLASQ conservative", colour_cons)
            _plot_flasq_series(xs_opt, ys_opt, "FLASQ optimistic", colour_opt)
            _combine_legend(ax, ax2)

        elif flasq_entries and x_key == "data_qubits":
            xs_cons, ys_cons, xs_opt, ys_opt = [], [], [], []
            for e in flasq_entries:
                q = e[3]
                if q is not None:
                    xs_cons.append(q)
                    ys_cons.append(e[1])
                    xs_opt.append(q)
                    ys_opt.append(e[2])
            _plot_flasq_series(xs_cons, ys_cons, "FLASQ conservative", colour_cons)
            _plot_flasq_series(xs_opt, ys_opt, "FLASQ optimistic", colour_opt)
            _combine_legend(ax, ax2)

        elif flasq_entries:
            print(
                f"Warning: FLASQ overlay is not supported for x-axis '{x_key}'. "
                "Use 'circuit' or 'data_qubits'.",
                file=sys.stderr,
            )

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
