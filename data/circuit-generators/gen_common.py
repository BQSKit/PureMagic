"""Shared benchmark-generation helpers for gen_qft.py and gen_grover.py."""

import os
from pathlib import Path

from qiskit import qasm2


def write_benchmark_qasm(circuit, output_dir: str, filename: str) -> str:
    """Writes `circuit` to `output_dir/filename` as OpenQASM 2.0 (creating `output_dir`
    if needed), prints a "Generated: ..." confirmation, and returns the written path.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    qasm2.dump(circuit, Path(file_path))
    print(f"Generated: {file_path}")
    return file_path
