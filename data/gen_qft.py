#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from qiskit import qasm2, transpile
from qiskit.circuit.library import QFT

DEFAULT_QUBITS = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]

# Benchpress's own QFT export is fully decomposed to rx/ry/rz/cx (even
# Hadamard expanded to its Euler-angle form) plus a full swap network
# reversing qubit order at the end -- confirmed by matching this generator's
# output gate counts exactly against data/benchpress/qft_N024.qasm
# (rz=828, cx=588, ry=24, rx=24 at n=24). do_swaps=True is required to get
# the swap network (do_swaps=False, this file's previous setting, omits it
# entirely and no longer reaches even a resemblance of Benchpress's own
# style under current Qiskit); optimization_level=0 is required too, since
# any optimization pass cancels/merges rotations and drifts from the exact
# reference counts.
BASIS_GATES = ["rx", "ry", "rz", "cx"]


def generate_benchpress_qft_set(qubit_counts, output_dir="custom_qft_benchmarks"):
    """
    Generates a homogeneous set of QFT QASM files matching
    the native Benchpress testing logic up to any qubit size.
    """
    os.makedirs(output_dir, exist_ok=True)

    for n in qubit_counts:
        # QFT is deprecated as of Qiskit 2.1 in favor of QFTGate /
        # qiskit.synthesis.qft.synth_qft_full -- kept here since it's the
        # form verified (see BASIS_GATES's comment) to reproduce Benchpress's
        # exact gate counts; revisit if this warning becomes an error in a
        # future Qiskit major version.
        qft_circuit = QFT(num_qubits=n, do_swaps=True)

        # A single .decompose() only unrolls one level (to h/cp) under
        # current Qiskit -- transpile to the exact target basis instead, so
        # the output only ever contains gates this generator's own callers
        # can rely on.
        decomposed_circuit = transpile(qft_circuit, basis_gates=BASIS_GATES, optimization_level=0)

        # Export cleanly to an OpenQASM 2.0 text file
        file_path = os.path.join(output_dir, f"qft_N{n:03d}_qiskit.qasm")
        qasm2.dump(decomposed_circuit, Path(file_path))
        print(f"Generated: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate QFT benchmark circuits matching Qiskit Benchpress's own "
            "representation (fully decomposed to rx/ry/rz/cx, with a full swap "
            "network reversing qubit order)."
        )
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=DEFAULT_QUBITS,
        help=f"Qubit count(s) to generate, e.g. --qubits 24 63 160 (default: {DEFAULT_QUBITS})",
    )
    parser.add_argument("-o", "--output-dir", default="custom_qft_benchmarks")
    args = parser.parse_args()

    generate_benchpress_qft_set(args.qubits, output_dir=args.output_dir)


if __name__ == "__main__":
    # Guarded so importing this module (e.g. to reuse
    # generate_benchpress_qft_set elsewhere) doesn't also generate a set as
    # a side effect.
    main()
