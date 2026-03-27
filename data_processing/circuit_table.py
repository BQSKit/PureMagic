#!/usr/bin/env python3
"""
Generate a LaTeX table of circuit statistics from QASM files.

Usage:
    python circuit_table.py --benchmarks <benchmarks_file> --qasmdir <qasm_directory>
                            [--transpiled <out_file>] [--puremagic <out_file>]
                            [--pdf <output.pdf>]

Arguments:
    --benchmarks   File containing a list of circuit names (one per line)
    --qasmdir      Directory containing <circuit_name>.qasm files
    --transpiled   Optional transpiler 'out' file; adds Compiled Gates, T Gates,
                   and Transpiled Cliffords columns
    --puremagic    Optional PureMagic 'out' file; adds Circuit Depth column
                   (Layers from Circuit statistics)
    --pdf          If given, also render the table to a PDF at this path using
                   pdflatex/xelatex/lualatex (preferred) or pandoc as fallback

Output:
    LaTeX-formatted table with: circuit name, number of qubits, circuit length
    (circuit length = number of unitary gate instruction lines, excluding
     whitespace, comments, QASM header declarations, measurements, resets,
     barriers, and identity gates)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def parse_qasm(filepath):
    """
    Parse a QASM file and return (num_qubits, circuit_length).

    num_qubits     - extracted from the 'qreg' declaration
    circuit_length - number of non-empty, non-comment lines that are gate
                     instructions (i.e. excluding OPENQASM, include, qreg,
                     creg declarations)
    """
    num_qubits = None
    circuit_length = 0

    # Lines that are not unitary gate instructions (excluded from count)
    header_patterns = [
        re.compile(r"^\s*OPENQASM\b"),
        re.compile(r"^\s*include\b"),
        re.compile(r"^\s*qreg\b"),
        re.compile(r"^\s*creg\b"),
        re.compile(r"^\s*measure\b"),
        re.compile(r"^\s*reset\b"),
        re.compile(r"^\s*barrier\b"),
        re.compile(r"^\s*id\b"),
    ]
    qreg_pattern = re.compile(r"^\s*qreg\s+\w+\[(\d+)\]")

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip comment lines
            if stripped.startswith("//"):
                continue

            # Remove inline comments
            code_part = stripped.split("//")[0].strip()
            if not code_part:
                continue

            # Check for qreg to extract qubit count
            m = qreg_pattern.match(code_part)
            if m:
                num_qubits = int(m.group(1))

            # Check if this is a header/declaration line
            is_header = any(p.match(code_part) for p in header_patterns)
            if not is_header:
                circuit_length += 1

    return num_qubits, circuit_length


def parse_transpiler_output_file(filepath):
    """
    Parse a transpiler output file and return a dict mapping circuit name to
    (compiled_depth, transpiled_depth, cliffords_after).

    Relevant lines in the file look like:
      Circuit length:    <before> (before) -> <after> (after transpilation)
        Clifford gates:    <before> (before) -> <after> (after transpilation)
      Wrote transpiled circuit to <name>.trans
    """
    result = {}

    length_re = re.compile(
        r"Circuit length:\s+([\d,]+)\s+\(before\)\s+->\s+([\d,]+)\s+\(after transpilation\)"
    )
    clifford_re = re.compile(
        r"Clifford gates:\s+([\d,]+)\s+\(before\)\s+->\s+([\d,]+)\s+\(after transpilation\)"
    )
    wrote_re = re.compile(r"Wrote transpiled circuit to (.+?)\.trans")

    cur_length_before = None
    cur_length_after = None
    cur_cliffords_after = None

    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()

            m = length_re.search(s)
            if m:
                cur_length_before = int(m.group(1).replace(",", ""))
                cur_length_after = int(m.group(2).replace(",", ""))
                continue

            m = clifford_re.search(s)
            if m:
                cur_cliffords_after = int(m.group(2).replace(",", ""))
                continue

            m = wrote_re.search(s)
            if m:
                name = m.group(1)
                if cur_length_before is not None:
                    result[name] = (cur_length_before, cur_length_after, cur_cliffords_after)
                cur_length_before = None
                cur_length_after = None
                cur_cliffords_after = None

    return result


def parse_puremagic_file(filepath):
    """
    Parse a PureMagic 'out' file and return a dict mapping circuit name to
    circuit_depth (the 'Layers:' value from the 'Circuit statistics:' block).

    Circuit name is taken from:
      Scheduled products written to <name>.schedule
    Layers is taken from the preceding 'Circuit statistics:' block:
      Layers:  <n>
    """
    result = {}
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    cur_layers = None
    in_stats = False

    layers_re = re.compile(r"Layers:\s+(\d+)")
    wrote_re = re.compile(r"Scheduled products written to (.+?)\.schedule")
    stats_re = re.compile(r"Circuit statistics:")

    with open(filepath, "r") as f:
        for line in f:
            s = ansi_escape.sub("", line).strip()

            if stats_re.search(s):
                in_stats = True
                cur_layers = None
                continue

            if in_stats:
                m = layers_re.search(s)
                if m:
                    cur_layers = int(m.group(1))

            m = wrote_re.search(s)
            if m:
                name = m.group(1)
                if cur_layers is not None:
                    result[name] = cur_layers
                in_stats = False
                cur_layers = None

    return result


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


def format_number(n):
    """Format a large integer with comma thousands separators."""
    return f"{n:,}"


# Names that should be fully uppercased after prefix truncation
_UPPERCASE_NAMES = {"dnn", "knn", "qft", "qv"}


def generate_pdf(latex_table: str, pdf_path: str) -> None:
    """
    Wrap *latex_table* in a minimal standalone document and render it to
    *pdf_path*.  Tries pdflatex / xelatex / lualatex first (they compile the
    .tex directly and preserve all LaTeX commands), then falls back to pandoc.
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

    # --- try pdflatex / xelatex / lualatex (compile .tex directly) ---
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

    # --- fall back to pandoc (passes --pdf-engine so it invokes pdflatex) ---
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


def pretty_name(name):
    """
    Apply human-readable substitutions to raw circuit names.

    Rules (applied in order):
      square_heisenberg_N<k>  ->  Heisenberg
      qaoa_barabasi_albert_N<k>_3reps  ->  QAOA
      Truncate at first '_' or ' '
      dnn/knn/qft/qv  ->  uppercase
      everything else ->  title-case first letter
    """
    m = re.fullmatch(r"square_heisenberg_[Nn](\d+)", name)
    if m:
        return "Heisenberg"

    m = re.fullmatch(r"qaoa_barabasi_albert_[Nn](\d+)_3reps", name)
    if m:
        return "QAOA"

    # Truncate at first underscore or space
    prefix = re.split(r"[_ ]", name)[0]

    if prefix.lower() in _UPPERCASE_NAMES:
        return prefix.upper()

    return prefix.capitalize()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table of circuit statistics from QASM files."
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        required=True,
        metavar="FILE",
        help="File containing a list of circuit names (one per line)",
    )
    parser.add_argument(
        "-q",
        "--qasmdir",
        required=True,
        metavar="DIR",
        help="Directory containing <circuit_name>.qasm files",
    )
    parser.add_argument(
        "-t",
        "--transpiled",
        metavar="FILE",
        default=None,
        help="Transpiler 'out' file; adds Compiled Depth, Transpiled Depth, and Cliffords columns",
    )
    parser.add_argument(
        "-m",
        "--puremagic",
        metavar="FILE[:LABEL]",
        action="append",
        default=[],
        help=(
            "PureMagic 'out' file; adds a Circuit Depth column. "
            "Append :LABEL to set the column header (e.g. file.out:w=1). "
            "May be repeated to add multiple columns."
        ),
    )
    parser.add_argument(
        "-p",
        "--pdf",
        metavar="FILE",
        default=None,
        help="If given, also render the table to a PDF at this path (requires pandoc or a LaTeX engine)",
    )
    args = parser.parse_args()

    benchmarks_file = args.benchmarks
    qasm_dir = args.qasmdir

    # Optionally load transpiler output
    transpiled_data = {}
    if args.transpiled:
        transpiled_data = parse_transpiler_output_file(args.transpiled)

    # Parse each --puremagic entry as "filepath[:label]"
    puremagic_entries = []  # list of (label, data_dict)
    for spec in args.puremagic:
        if ":" in spec:
            filepath, label = spec.rsplit(":", 1)
        else:
            filepath = spec
            label = "Circuit Depth"
        puremagic_entries.append((label, parse_puremagic_file(filepath)))

    # Read circuit names
    with open(benchmarks_file, "r") as f:
        circuit_names = [line.strip() for line in f if line.strip()]

    rows = []
    for name in circuit_names:
        qasm_path = os.path.join(qasm_dir, f"{name}.qasm")
        if not os.path.isfile(qasm_path):
            print(f"Warning: {qasm_path} not found, skipping.", file=sys.stderr)
            continue

        num_qubits, circuit_length = parse_qasm(qasm_path)

        trans = transpiled_data.get(name)
        compiled_depth = trans[0] if trans else None
        transpiled_depth = trans[1] if trans else None
        cliffords = trans[2] if trans else None
        pm_depths = [data.get(name) for _, data in puremagic_entries]

        rows.append(
            (
                name,
                num_qubits,
                circuit_length,
                compiled_depth,
                transpiled_depth,
                cliffords,
                pm_depths,
            )
        )

    use_transpiled = bool(args.transpiled)

    # Build header tuple
    header_list = ["Circuit", "Qubits", "Unitary Gates"]
    if use_transpiled:
        header_list += ["Compiled Gates", "Transpiled T Gates", "Transpiled Cliffords"]
    for label, _ in puremagic_entries:
        header_list.append(label)
    header = tuple(header_list)

    formatted_rows = []
    for (
        name,
        num_qubits,
        circuit_length,
        compiled_depth,
        transpiled_depth,
        cliffords,
        pm_depths,
    ) in rows:
        cells = [
            latex_escape(pretty_name(name)),
            str(num_qubits) if num_qubits is not None else "?",
            format_number(circuit_length),
        ]
        if use_transpiled:
            cells.append(format_number(compiled_depth) if compiled_depth is not None else "—")
            t_gates = (
                (transpiled_depth - cliffords - (num_qubits or 0))
                if (transpiled_depth is not None and cliffords is not None)
                else None
            )
            cells.append(format_number(t_gates) if t_gates is not None else "—")
            cells.append(format_number(cliffords) if cliffords is not None else "—")
        for depth in pm_depths:
            cells.append(format_number(depth) if depth is not None else "—")
        formatted_rows.append(tuple(cells))

    # Split each header into (top_word, bottom_word); single-word headers have bottom=""
    header_top = []
    header_bot = []
    for h in header:
        if " " in h:
            top, bot = h.split(" ", 1)
        else:
            top, bot = h, ""
        header_top.append(top)
        header_bot.append(bot)

    # Compute column widths: max of top word, bottom word, and all data cells
    col_widths = [max(len(header_top[i]), len(header_bot[i])) for i in range(len(header))]
    for row in formatted_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells, widths):
        # Left-align col 0, right-align all numeric columns
        parts = [cells[0].ljust(widths[0])]
        for i in range(1, len(cells)):
            parts.append(cells[i].rjust(widths[i]))
        return "    " + " & ".join(parts) + " \\\\"

    # Build tabular column spec: l for name, r for all others
    col_spec = "l" + "r" * (len(header) - 1)

    # Assemble LaTeX table string
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \hline\hline")
    lines.append(fmt_row(header_top, col_widths))
    lines.append(fmt_row(header_bot, col_widths))
    lines.append(r"    \hline\hline")
    for row in formatted_rows:
        lines.append(fmt_row(row, col_widths))
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{Circuit benchmark statistics.}")
    lines.append(r"  \label{tab:benchmarks}")
    lines.append(r"\end{table}")
    latex_table = "\n".join(lines)

    # Print to stdout
    print(latex_table)

    # Optionally render to PDF
    if args.pdf:
        try:
            generate_pdf(latex_table, args.pdf)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
