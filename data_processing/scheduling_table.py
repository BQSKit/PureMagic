#!/usr/bin/env python3
"""
Generate a LaTeX table comparing two scheduling results.

Columns:
  Circuit
  Volume (split: label1 | label2 | % Improvement)
  Magic Cultivation Average (split: label1 | label2)

Usage:
    python scheduling_table.py --benchmarks <benchmarks_file>
                               -d file1:label1,file2:label2
                               [--pdf <output.pdf>]

Arguments:
    --benchmarks   File containing a list of circuit names (one per line)
    -d             Two data files with labels: file1:label1,file2:label2
                   Volume (lcycles * total_qubits) from file1 goes in the
                   first column (label1 header), file2 in the second
                   (label2 header).
                   % Improvement is (file2 - file1) / file1 * 100.
                   The same labels are used for the Magic Cultivation columns.
    --pdf          If given, also render the table to a PDF at this path
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_puremagic_output(filepath):
    """
    Parse a PureMagic output file and return a dict mapping circuit name to
    a dict with keys:
        volume                 (int or None)    -- from "Scheduled N in L logical cycles, volume V"
        magic_cultivation_avg  (float or None)  -- from "average: <x>" under
                                                   "Magic state cultivation time:"
        data_qubits            (int or None)    -- from "Loaded circuit with N products and Q qubits"
        parallel_efficiency    (float or None)  -- from "Parallel efficiency: <x>" or computed as
                                                   parallelism / optimal_speedup

    The file structure per run is:
        magic_state_lambda: <value>   (in Args block)
        Loaded circuit with <n> products and <q> qubits
        Magic state cultivation time:
          ...
          average: <value>
          ...
        Scheduled <n> in <lcycles> logical cycles, volume <volume>
        Scheduled products written to <name>.schedule
        Timing: main took ...          <- end of run
    """
    result = {}
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    # Per-run state
    cur_circuit = None
    cur_qubits = None  # from "Loaded circuit" line
    cur_volume = None  # from "Scheduled N in L logical cycles, volume V"
    cur_cult_avg = None
    cur_parallel_eff = None
    cur_parallelism = None
    cur_optimal_speedup = None
    in_cult_block = False

    lambda_re = re.compile(r"magic_state_lambda:\s*([0-9.eE+\-]+),?")
    loaded_re = re.compile(r"Loaded circuit with \d+ products and (\d+) qubits")
    cult_block_re = re.compile(r"Magic state cultivation time:")
    cult_avg_re = re.compile(r"average:\s*([0-9.eE+\-]+)")
    volume_re = re.compile(r"Scheduled \d+ in \d+ logical cycles, volume (\d+)")
    wrote_re = re.compile(r"Scheduled products written to (.+?)\.schedule")
    flush_re = re.compile(r"Timing: main took")
    parallel_eff_re = re.compile(r"Parallel efficiency:\s*([0-9.eE+\-]+)")
    parallelism_re = re.compile(r"Parallelism:\s*([0-9.eE+\-]+)x")
    optimal_speedup_re = re.compile(r"Optimal logical cycles \d+ \(([0-9.eE+\-]+) speedup\)")

    def _flush():
        nonlocal cur_circuit, cur_qubits, cur_volume, cur_cult_avg, in_cult_block
        nonlocal cur_parallel_eff, cur_parallelism, cur_optimal_speedup
        if cur_circuit is not None:
            pe = cur_parallel_eff
            if pe is None and cur_parallelism is not None and cur_optimal_speedup is not None:
                pe = cur_parallelism / cur_optimal_speedup
            result[cur_circuit] = {
                "volume": cur_volume,
                "magic_cultivation_avg": cur_cult_avg,
                "data_qubits": cur_qubits,
                "parallel_efficiency": pe,
            }
        cur_circuit = None
        cur_qubits = None
        cur_volume = None
        cur_cult_avg = None
        cur_parallel_eff = None
        cur_parallelism = None
        cur_optimal_speedup = None
        in_cult_block = False

    with open(filepath, "r") as f:
        for line in f:
            s = ansi_escape.sub("", line).strip()

            # New lambda signals start of a new run — flush previous
            if lambda_re.match(s):
                _flush()
                continue

            # Qubit count — appears before the circuit name line
            m = loaded_re.match(s)
            if m:
                cur_qubits = int(m.group(1))
                continue

            # Enter cultivation time block
            if cult_block_re.search(s):
                in_cult_block = True
                continue

            # Cultivation average (inside the block)
            if in_cult_block:
                m = cult_avg_re.match(s)
                if m:
                    cur_cult_avg = float(m.group(1))
                    in_cult_block = False
                    continue

            # Volume — "Scheduled N in L logical cycles, volume V"
            m = volume_re.match(s)
            if m:
                cur_volume = int(m.group(1))
                continue

            # Parallel efficiency — "Parallel efficiency: <x>"
            m = parallel_eff_re.match(s)
            if m:
                cur_parallel_eff = float(m.group(1))
                continue

            # Parallelism — "Parallelism: <x>x"
            m = parallelism_re.match(s)
            if m:
                cur_parallelism = float(m.group(1))
                continue

            # Optimal speedup — "Optimal logical cycles N (<x> speedup)"
            m = optimal_speedup_re.match(s)
            if m:
                cur_optimal_speedup = float(m.group(1))
                continue

            # Circuit name — appears after the volume line
            m = wrote_re.search(s)
            if m:
                if cur_circuit is not None:
                    _flush()
                cur_circuit = m.group(1)
                continue

            # End-of-run marker
            if flush_re.match(s):
                _flush()

    _flush()
    return result


# ---------------------------------------------------------------------------
# Circuit name prettifier (identical rules to circuit_table.py)
# ---------------------------------------------------------------------------

_UPPERCASE_NAMES = {"dnn", "knn", "qft", "qv"}


def pretty_name(name, num_qubits=None):
    """
    Apply human-readable substitutions to raw circuit names, appending the
    qubit/size count in parentheses.

    Rules (applied in order):
      square_heisenberg_N<k>  ->  Heis.(<k>)
      qaoa_barabasi_albert_N<k>_3reps  ->  QAOA(<k>)
      Truncate at first '_' or ' '
      dnn/knn/qft/qv  ->  uppercase
      everything else ->  title-case first letter
    Then append (<size>) extracted from the name, or (<num_qubits>) as fallback.
    """
    m = re.fullmatch(r"square_heisenberg_[Nn](\d+)", name)
    if m:
        return f"Heis.({m.group(1)})"

    m = re.fullmatch(r"qaoa_barabasi_albert_[Nn](\d+)_3reps", name)
    if m:
        return f"QAOA({m.group(1)})"

    # Try to extract a size number: prefer _N<digits> or _n<digits> segment
    m = re.search(r"_[Nn](\d+)", name)
    if m:
        size = str(int(m.group(1)))  # strip leading zeros
    elif num_qubits is not None:
        size = str(num_qubits)
    else:
        size = None

    # Truncate at first underscore or space
    prefix = re.split(r"[_ ]", name)[0]

    if prefix.lower() in _UPPERCASE_NAMES:
        base = prefix.upper()
    else:
        base = prefix.capitalize()

    if size is not None:
        return f"{base}({size})"
    return base


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def latex_escape(s):
    """Escape special LaTeX characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def fmt_pct(value):
    """Format a percentage value to 1 decimal place (no % sign)."""
    return f"{value:.1f}"


def fmt_volume(value):
    """Format a volume in thousands (value / 1000) rounded to nearest integer."""
    return f"{round(value / 1000)}"


def fmt_cultivation(value):
    """Format cultivation average (cycles) as a float with 2 decimal places."""
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# PDF rendering (same as circuit_table.py)
# ---------------------------------------------------------------------------


def generate_pdf(latex_table: str, pdf_path: str) -> None:
    """
    Wrap *latex_table* in a minimal standalone document and render it to
    *pdf_path*.  Tries pdflatex / xelatex / lualatex first, then pandoc.
    Raises RuntimeError if no suitable tool is found.
    """
    standalone_doc = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\usepackage{makecell}" + "\n"
        r"\usepackage{geometry}" + "\n"
        r"\geometry{margin=1in}" + "\n"
        r"\begin{document}" + "\n"
        r"\pagestyle{empty}" + "\n" + latex_table + "\n"
        r"\end{document}" + "\n"
    )

    pdf_path = os.path.abspath(pdf_path)

    for engine in ("pdflatex", "xelatex", "lualatex"):
        if not shutil.which(engine):
            continue
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "table.tex")
            with open(tex_path, "w") as f:
                f.write(standalone_doc)
            try:
                subprocess.run(
                    [engine, "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                    check=True,
                    capture_output=True,
                )
                shutil.copy(os.path.join(tmpdir, "table.pdf"), pdf_path)
                print(f"PDF written to {pdf_path} (via {engine})", file=sys.stderr)
                return
            except subprocess.CalledProcessError as e:
                print(f"{engine} failed: {e.stderr.decode()}", file=sys.stderr)

    if shutil.which("pandoc"):
        with tempfile.NamedTemporaryFile(suffix=".tex", mode="w", delete=False) as tmp:
            tmp.write(standalone_doc)
            tex_path = tmp.name
        try:
            subprocess.run(
                ["pandoc", tex_path, "-o", pdf_path, "--pdf-engine=pdflatex", "--from=latex"],
                check=True,
                capture_output=True,
            )
            print(f"PDF written to {pdf_path} (via pandoc)", file=sys.stderr)
            return
        except subprocess.CalledProcessError as e:
            print(f"pandoc failed: {e.stderr.decode()}", file=sys.stderr)
        finally:
            os.unlink(tex_path)

    raise RuntimeError("No PDF renderer found. Install pdflatex, xelatex, lualatex, or pandoc.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a LaTeX table comparing scheduling results.\n\n"
            "Each -d entry adds a triple of columns (label1 | label2 | %% Impr.) "
            "under 'Volume'. When only one -d is given, two Magic "
            "Cultivation Avg. columns (label1 | label2) are also appended."
        )
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        required=True,
        metavar="FILE",
        help="File containing a list of circuit names (one per line)",
    )
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        action="append",
        metavar="FILE1:LABEL1,FILE2:LABEL2",
        help=(
            "Two PureMagic output files with column labels, comma-separated. "
            "E.g. bus.out:BUS,pm.out:PureMagic. May be repeated; each repetition "
            "adds another triple of Scheduling Efficiency columns. When more than "
            "one -d is given, Magic Cultivation columns are omitted."
        ),
    )
    parser.add_argument(
        "-p",
        "--pdf",
        metavar="FILE",
        default=None,
        help="If given, also render the table to a PDF at this path",
    )
    args = parser.parse_args()

    # Parse each -d argument: "file1:label1,file2:label2"
    def _parse_d(spec):
        parts = spec.split(",")
        if len(parts) != 2:
            parser.error(f"-d {spec!r} requires exactly two entries: file1:label1,file2:label2")
        result = []
        for part in parts:
            if ":" not in part:
                parser.error(f"-d entry {part!r} must be in the form file:label")
            filepath, label = part.rsplit(":", 1)
            result.append((filepath, label))
        return result

    groups = [_parse_d(spec) for spec in args.data]
    # groups: list of [(file1, label1), (file2, label2)]

    show_cultivation = len(groups) == 1

    # Load all data files (cache by path to avoid re-reading the same file)
    _cache = {}

    def _load(path):
        if path not in _cache:
            _cache[path] = parse_puremagic_output(path)
        return _cache[path]

    loaded_groups = [
        (_load(f1), label1, _load(f2), label2) for (f1, label1), (f2, label2) in groups
    ]

    # Read circuit names
    with open(args.benchmarks, "r") as f:
        circuit_names = [line.strip() for line in f if line.strip()]

    # Format helpers
    DASH = "---"

    def _fmt_vol(v):
        return fmt_volume(v) if v is not None else DASH

    def _fmt_pct(v):
        return fmt_pct(v) if v is not None else DASH

    def _fmt_cult(v):
        return fmt_cultivation(v) if v is not None else DASH

    # Build formatted rows.
    # Each row: [circuit_name, (vol1, vol2, pct, cult1, cult2) per group]
    formatted_rows = []
    for name in circuit_names:
        # Determine qubit count from first group's first file
        data_qubits = None
        for d1, _, d2, _ in loaded_groups:
            data_qubits = d1.get(name, {}).get("data_qubits") or d2.get(name, {}).get("data_qubits")
            if data_qubits:
                break

        display_name = latex_escape(pretty_name(name, data_qubits))
        cells = [display_name]

        for d1, label1, d2, label2 in loaded_groups:
            vol1 = d1.get(name, {}).get("volume")
            vol2 = d2.get(name, {}).get("volume")
            cult1 = d1.get(name, {}).get("magic_cultivation_avg")
            cult2 = d2.get(name, {}).get("magic_cultivation_avg")
            pe1 = d1.get(name, {}).get("parallel_efficiency")
            pe2 = d2.get(name, {}).get("parallel_efficiency")

            # % improvement in parallel efficiency: positive means file2 is better
            pct = (
                (pe2 - pe1) / pe1 * 100.0
                if (pe1 is not None and pe2 is not None and pe1 != 0.0)
                else None
            )

            cells += [_fmt_vol(vol1), _fmt_vol(vol2), _fmt_pct(pct)]
            if show_cultivation:
                cells += [_fmt_cult(cult1), _fmt_cult(cult2)]

        formatted_rows.append(tuple(cells))

    # ---------------------------------------------------------------------------
    # Build LaTeX table with multi-column headers
    #
    # Layout per -d group (3 cols): label1 | label2 | % Impr.
    #   -> \multicolumn{3}{c|}{Scheduling Efficiency}
    # If single -d, append 2 Magic Cultivation cols: label1 | label2
    #   -> \multicolumn{2}{c|}{Magic Cultivation Avg.}
    # ---------------------------------------------------------------------------

    num_groups = len(groups)
    # cols: 1 (Circuit) + 3*num_groups (eff triples) + 2*show_cultivation (cult pair)
    col_spec = "|l|" + "r|r|r|" * num_groups + ("r|r|" if show_cultivation else "")

    # Sub-header labels
    sub_headers = ["Circuit"]
    for _, label1, _, label2 in loaded_groups:
        sub_headers += [label1, label2, "\% Impr."]
    if show_cultivation:
        _, label1, _, label2 = loaded_groups[0]
        sub_headers += [label1, label2]

    # Compute column widths for alignment (plain-text widths, ignoring LaTeX macros)
    def _plain(s):
        """Strip LaTeX commands for width estimation."""
        s = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", s)
        s = re.sub(r"\\[a-zA-Z]+", "", s)
        return s

    col_widths = [len(_plain(h)) for h in sub_headers]
    for row in formatted_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(_plain(cell)))

    def fmt_row(cells, widths):
        """Format a data row, left-aligning col 0, right-aligning the rest."""
        parts = [cells[0].ljust(widths[0])]
        for i in range(1, len(cells)):
            parts.append(cells[i].rjust(widths[i]))
        return "    " + " & ".join(parts) + " \\\\"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{2pt}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline")

    # Top header row
    top_parts = ["    Circuit"]
    for _ in range(num_groups):
        top_parts.append(r"\multicolumn{3}{c|}{Volume ($\times 10^3$)}")
    if show_cultivation:
        top_parts.append(r"\multicolumn{2}{c|}{Magic Cultivation Avg.}")
    lines.append(" & ".join(top_parts) + r" \\")

    # Sub-header row
    lines.append(fmt_row(sub_headers, col_widths))
    lines.append(r"    \hline")

    for row in formatted_rows:
        lines.append(fmt_row(row, col_widths))

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(
        r"  \caption{Volume (logical cycles $\times$ total qubits) and magic cultivation average.}"
    )
    lines.append(r"  \label{tab:scheduling}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    print(latex_table)

    if args.pdf:
        try:
            generate_pdf(latex_table, args.pdf)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
