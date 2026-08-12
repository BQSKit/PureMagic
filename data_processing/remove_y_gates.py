#!/usr/bin/env python3
"""
Rewrite `y` gates out of a Clifford+T circuit.

wisq's GUOQ optimizer backend cannot process `y` gates, even though `y` is
one of the generators compile_cliffordt.py's exact-synthesis path can emit
(see CLIFFORD_GENERATORS there) and is listed as valid Clifford+T
(CLIFFORD_T_BASIS). This is a standalone preprocessing step to run over
.cliffordt.qasm files before handing them to wisq -- not a change to
compile_cliffordt.py's own output, which other consumers (bqskit-ft,
cyclosynth, compile_cliffordt.rs) accept `y` in just fine.

Uses qiskit's standard equivalence library via BasisTranslator (the same
mechanism wisq's own resynth.py uses for gateset translation) to rewrite each
`y` into `z` then `x`, rather than hand-substituting the gates textually --
the equivalence library tracks the accompanying global-phase correction
automatically, so the output's unitary matches the input's exactly (not just
up to global phase).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary as sel
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasisTranslator

from compile_cliffordt import (
    CLIFFORD_T_BASIS,
    load_circuit,
    non_basis_ops,
    operation_counts,
    write_circuit,
)

# compile_cliffordt.py's own Clifford+T basis, minus `y`.
TARGET_BASIS = tuple(g for g in CLIFFORD_T_BASIS if g != "y")


def remove_y_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Rewrite every `y` in `circuit` into `z`+`x`, preserving the exact unitary."""
    return PassManager([BasisTranslator(sel, TARGET_BASIS)]).run(circuit)


def output_path(source: Path, target: Optional[Path], many: bool) -> Path:
    default_name = source.stem + ".noy.qasm"
    if target is None:
        return source.parent / default_name
    if many or target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite `y` gates in a Clifford+T QASM circuit into other "
        "Clifford+T gates, so wisq's GUOQ backend (which cannot handle `y`) can "
        "accept it."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input Clifford+T .qasm file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (single input) or directory (multiple inputs); "
        "default is <input>.noy.qasm next to the input",
    )
    args = parser.parse_args(argv)

    failures = 0
    for source in args.inputs:
        circuit = load_circuit(source)
        y_count = operation_counts(circuit).get("y", 0)
        converted = remove_y_gates(circuit)

        remaining_y = operation_counts(converted).get("y", 0)
        remaining_non_basis = non_basis_ops(converted)
        if remaining_y or remaining_non_basis:
            print(
                f"ERROR {source}: conversion left y={remaining_y}, "
                f"non-Clifford+T ops {remaining_non_basis}",
                file=sys.stderr,
            )
            failures += 1
            continue

        destination = output_path(source, args.output, len(args.inputs) > 1)
        version = write_circuit(converted, destination)
        print(f"{source}: rewrote {y_count} y gate(s) -> {destination} (OpenQASM {version})")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
