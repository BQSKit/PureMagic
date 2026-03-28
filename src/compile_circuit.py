#!/usr/bin/env -S python -u
"""
Compile a QASM quantum circuit to the Clifford+T gate set using bqskit.

Outputs:
  <stem>.cliffordt.qasm  — compiled circuit in QASM format (for use with transpile_circuit.py)
"""

import argparse
import sys
from timeit import default_timer as timer
from pathlib import Path

from bqskit import Circuit
from bqskit.compiler.compile import compile
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ir.gates import IdentityGate, MeasurementPlaceholder, BarrierPlaceholder


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Compile a QASM quantum circuit to the Clifford+T gate set. "
            "Outputs a .cliffordt.qasm file that can be passed directly to transpile_circuit.py."
        )
    )
    parser.add_argument(
        "--input_file",
        "-i",
        required=True,
        help="Input QASM circuit file (must have a .qasm extension).",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help=(
            "Output file stem (without extension). "
            "Defaults to the stem of the input file. "
            "The suffix .cliffordt.qasm is appended automatically."
        ),
        default="",
    )

    args: argparse.Namespace = parser.parse_args()

    input_file: str = args.input_file
    if not input_file.endswith(".qasm"):
        print(f"Error: input file must be a .qasm file, got: {input_file}", file=sys.stderr)
        sys.exit(1)

    output_stem: str = Path(input_file).stem if args.output_file == "" else args.output_file

    # Load the input QASM circuit
    print(f"Loading QASM circuit from {input_file}")
    load_start = timer()
    circuit: Circuit = Circuit.from_file(input_file)
    load_end = timer()
    print(f"Circuit loaded in {(load_end - load_start):.2f} seconds")
    print(f"Input circuit has {len(circuit)} gates on {circuit.num_qudits} qubits")

    # Compile to Clifford+T using bqskit
    print("Compiling to Clifford+T...")
    compile_start: float = timer()
    machine: CliffordTModel = CliffordTModel(circuit.num_qudits)
    circuit = compile(circuit, model=machine)
    compile_end: float = timer()
    print(f"Compilation took {(compile_end - compile_start):.2f} seconds")

    # Flatten any CircuitGate wrappers that bqskit may have left around
    # sub-circuits (e.g. U3Gate wrapped in a CircuitGate).  Without this,
    # transpile_circuit.py crashes because it has no rule for CircuitGate.
    circuit.unfold_all()

    # Remove IdentityGate operations: they are semantic no-ops but bqskit
    # serialises them as a custom "identity1" gate using U(0,0,0).  When the
    # QASM is reloaded, that custom gate definition is parsed back as a
    # CircuitGate(U3Gate), which causes transpile_circuit.py to crash.
    identity = IdentityGate(1)
    if identity in circuit.gate_set:
        circuit.remove_all(identity)

    # Warn about any gates that are not in the Clifford+T gate set
    for g in circuit.gate_set:
        if (
            not isinstance(g, IdentityGate)
            and not isinstance(g, MeasurementPlaceholder)
            and not isinstance(g, BarrierPlaceholder)
        ):
            if g not in clifford_t_gates:
                print(f"Warning: gate {g} is not Clifford+T")
    unique_gate_set: list[str] = sorted({type(g).__name__ for g in circuit.gate_set})
    print(f"Gate set: {', '.join(unique_gate_set)}")
    print(f"Compiled circuit has {len(circuit)} gates")

    # Save compiled circuit as QASM (for use with transpile_circuit.py)
    qasm_path = Path(f"{output_stem}.cliffordt.qasm")
    circuit.save(str(qasm_path))

    # bqskit emits one creg declaration per MeasurementPlaceholder, producing
    # duplicate lines that cause QASM parsers to reject the file.  Deduplicate
    # them while preserving the first occurrence and the original line order.
    qasm_text = qasm_path.read_text()
    seen_lines: set[str] = set()
    deduped_lines: list[str] = []
    for line in qasm_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("creg "):
            if stripped in seen_lines:
                continue
            seen_lines.add(stripped)
        deduped_lines.append(line)
    qasm_path.write_text("".join(deduped_lines))

    print(f"Saved compiled circuit (QASM) to {qasm_path}")


if __name__ == "__main__":
    main()
