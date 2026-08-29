#!/usr/bin/env python3

"""Generate Grover's-algorithm-shaped benchmark circuits, with qubit count
and iteration count chosen independently of each other.

"Textbook" Grover needs ~pi/4 * sqrt(2**n) oracle+diffusion iterations to
actually amplify the marked state -- exponential in n, which is why every
benchmark suite's Grover circuit tops out around 30 qubits (already ~32,000
iterations there; 40 qubits would need ~1e6). For stress-testing a
compiler/scheduler, the correct iteration count doesn't matter -- only the
oracle+diffusion *shape*, repeated some controlled number of times -- so
this generator takes qubit count and iteration count as separate,
independently-scalable parameters instead of deriving one from the other.

The oracle marks a fixed bitstring (default: alternating 1010...) via the
standard X-sandwiched multi-controlled-Z construction, so it exercises both
branches (bits that do and don't need an X) rather than degenerating to a
bare MCZ for an all-ones target.
"""

import argparse

from qiskit import QuantumCircuit, transpile

from gen_common import write_benchmark_qasm

# Matches push_gate's accepted vocabulary in src/cliffordt/qasm.rs -- ccx is
# kept as a literal primitive (the Rust side decomposes it to Clifford+T
# itself) rather than letting Qiskit expand it further, the same way other
# family circuits under data/all_compiled_pipe-testing/ represent Toffolis.
BASIS_GATES = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rz", "u3", "cx", "cz", "ccx", "swap"]


def build_oracle(num_qubits: int, target: str) -> QuantumCircuit:
    """Phase-flip |target> via an X-sandwiched multi-controlled Z."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    zero_bits = [i for i, bit in enumerate(reversed(target)) if bit == "0"]
    if zero_bits:
        qc.x(zero_bits)
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    if zero_bits:
        qc.x(zero_bits)
    return qc


def build_diffusion(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator: inversion about the mean."""
    qc = QuantumCircuit(num_qubits, name="diffusion")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover(num_qubits: int, iterations: int, target: str | None = None) -> QuantumCircuit:
    if target is None:
        target = "".join("1" if i % 2 == 0 else "0" for i in range(num_qubits))
    if len(target) != num_qubits:
        raise ValueError(f"target bitstring length {len(target)} != num_qubits {num_qubits}")

    oracle = build_oracle(num_qubits, target)
    diffusion = build_diffusion(num_qubits)

    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    return qc


def generate(num_qubits: int, iterations: int, output_dir: str, target: str | None = None) -> str:
    qc = build_grover(num_qubits, iterations, target)
    qc.measure_all()
    decomposed = transpile(qc, basis_gates=BASIS_GATES, optimization_level=1)
    return write_benchmark_qasm(decomposed, output_dir, f"grover_n{num_qubits}_i{iterations}.qasm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        required=True,
        help="Search-register qubit count(s) to generate, e.g. --qubits 50 100 160",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Oracle+diffusion repetitions, independent of qubit count (default: 10)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Bitstring to mark (default: alternating 1010...); must match qubit count if given",
    )
    parser.add_argument("-o", "--output-dir", default="grover_benchmarks")
    args = parser.parse_args()

    for n in args.qubits:
        generate(n, args.iterations, args.output_dir, args.target)


if __name__ == "__main__":
    main()
