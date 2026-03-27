#!/usr/bin/env python3
"""
Generate a LaTeX table of circuit statistics from QASM files.

Usage:
    python circuit_table.py --benchmarks <benchmarks_file> --qasmdir <qasm_directory>
                            [--transpiled <out_file>]

Arguments:
    --benchmarks   File containing a list of circuit names (one per line)
    --qasmdir      Directory containing <circuit_name>.qasm files
    --transpiled   Optional 'out' file produced by the transpiler; when given,
                   adds Compiled Depth, Transpiled Depth, and Cliffords columns

Output:
    LaTeX-formatted table with: circuit name, number of qubits, circuit length
    (circuit length = number of gate instruction lines, excluding whitespace,
     comments, and QASM header declarations)
"""

import argparse
import os
import re
import sys


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


def parse_out_file(filepath):
    """
    Parse a transpiler 'out' file and return a dict mapping circuit name to
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
    args = parser.parse_args()

    benchmarks_file = args.benchmarks
    qasm_dir = args.qasmdir

    # Optionally load transpiler output
    transpiled_data = {}
    if args.transpiled:
        transpiled_data = parse_out_file(args.transpiled)

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

        rows.append((name, num_qubits, circuit_length, compiled_depth, transpiled_depth, cliffords))

    use_transpiled = bool(args.transpiled)

    # Build formatted cell values
    if use_transpiled:
        header = ("Circuit", "Qubits", "Gates", "Compiled Depth", "Transpiled Depth", "Cliffords")
    else:
        header = ("Circuit", "Qubits", "Gates")

    formatted_rows = []
    for name, num_qubits, circuit_length, compiled_depth, transpiled_depth, cliffords in rows:
        cells = [
            latex_escape(pretty_name(name)),
            str(num_qubits) if num_qubits is not None else "?",
            format_number(circuit_length),
        ]
        if use_transpiled:
            cells.append(format_number(compiled_depth) if compiled_depth is not None else "—")
            cells.append(format_number(transpiled_depth) if transpiled_depth is not None else "—")
            cells.append(format_number(cliffords) if cliffords is not None else "—")
        formatted_rows.append(tuple(cells))

    # Compute column widths (max of header and all data cells)
    col_widths = [len(h) for h in header]
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

    # Print LaTeX table
    print(r"\begin{table}[ht]")
    print(r"  \centering")
    print(f"  \\begin{{tabular}}{{{col_spec}}}")
    print(r"    \toprule")
    print(fmt_row(header, col_widths))
    print(r"    \midrule")
    for row in formatted_rows:
        print(fmt_row(row, col_widths))
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Circuit benchmark statistics.}")
    print(r"  \label{tab:benchmarks}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
