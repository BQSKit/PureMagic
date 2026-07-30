#!/usr/bin/env -S python -u
"""
Compile a QASM quantum circuit to the Clifford+T gate set using bqskit.

Outputs:
  <stem>.cliffordt.qasm  — compiled circuit in QASM format (for use with the transpile binary)
"""

import argparse
import cmath
import sys
import warnings
from math import log10
from timeit import default_timer as timer
from pathlib import Path

import numpy as np

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler.compile import compile
from bqskit.compiler.registry import register_workflow
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates
from bqskit.ft.cliffordt.defaultworkflow import (
    build_circuit_workflow,
    clifford_replace,
    rz_decomposition_passes,
    single_qudit_filter,
)
from bqskit.ft.ftpasses.rounding import RoundToDiscreteZPass
from bqskit.ir.gates import IdentityGate, MeasurementPlaceholder, BarrierPlaceholder
from bqskit.ir.gates.constant.sx import SqrtXGate
from bqskit.ir.gates.parameterized.rx import RXGate
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1 import U1Gate
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.util.unfold import UnfoldPass

# try:
#    import torch
#    _CUDA_AVAILABLE = torch.cuda.is_available()
# except ImportError:
#    _CUDA_AVAILABLE = False

# bqskit's own default, used here wherever an optimization_level has to be given
# explicitly (register_workflow has no default of its own).  This script does not
# expose a way to change it: level 1 is fast and works at any circuit size, and
# levels 2-4 add slower passes (level 2 rebases small blocks numerically; 3 and 4
# additionally synthesise the whole circuit from its unitary, which only works up
# to ~12 qubits) that are not needed for the benchmarks this script targets.
OPTIMIZATION_LEVEL = 1


# Tolerance for recognising the ZXZXZ middle angle as Clifford.  It only has to
# separate "t is 0 or +-pi" from every other value the decomposition produces,
# and those two come out exact to floating-point precision.
CLIFFORD_ANGLE_TOL = 1e-10


def _wrap_angle(angle: float) -> float:
    """Move an angle into [-pi, pi), as ZXZXZDecomposition does."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class GaugeFixedZXZXZDecomposition(ZXZXZDecomposition):
    """ZXZXZDecomposition that does not split a diagonal rotation into two.

    bqskit's version (bqskit/passes/rules/zxzxz.py:76-80) writes a single-qubit
    unitary as rz(l) . sx . rz(t) . sx . rz(p).  For a *diagonal* target
    utry[1, 0] is 0, so cmath.phase(0) == 0.0 forces t == pi and leaves
    l == p + pi == a/2: the total z-rotation is split evenly between the two
    outer gates and both halves are generic, so GridSynthPass is called twice
    where once would do.  A gridsynth word is ~3*log2(1/epsilon) T gates (~100
    at the epsilon this script uses), so that doubles the T count of every
    diagonal run -- on an 18-qubit Hubbard benchmark, all 544 runs that need
    synthesis are diagonal.

    The split is a free gauge, not a constraint.  When t == +-pi the middle
    block sx . rz(t) . sx is diagonal and commutes with the outer rz gates, so
    only l + p is determined; when t == 0 that block is X and only p - l is.
    Either way the whole rotation can be moved into one gate, leaving rz(0),
    which costs nothing.  For any other t, l and p are individually determined
    and this class behaves exactly like the base class.

    Everything here is exact -- it changes which gate carries the rotation, not
    the unitary -- so accuracy is still whatever gridsynth delivers at epsilon.
    """

    async def run(self, circuit: Circuit, data) -> None:
        if circuit.num_qudits != 1:
            raise ValueError('Cannot convert multi-qudit circuit into ZXZXZ sequence.')
        if circuit.radixes[0] != 2:
            raise ValueError('Cannot convert non-qubit circuit into ZXZXZ sequence.')

        no_sx = RXGate() in data.gate_set and SqrtXGate() not in data.gate_set
        use_rx = self.always_use_rx or no_sx
        no_rz = U1Gate() in data.gate_set and RZGate() not in data.gate_set
        use_u1 = self.always_use_u1 or no_rz

        utry = circuit.get_unitary()
        utry = np.linalg.det(utry) ** (-0.5) * utry
        i1 = cmath.phase(utry[1, 1])
        i2 = cmath.phase(utry[1, 0])
        t = 2 * np.arctan2(abs(utry[1, 0]), abs(utry[0, 0])) + np.pi
        p = i1 + i2 + np.pi
        l = i1 - i2

        t, p, l = _wrap_angle(t), _wrap_angle(p), _wrap_angle(l)

        # The fix: collapse the gauge when the middle block is Clifford.
        if abs(abs(t) - np.pi) < CLIFFORD_ANGLE_TOL:
            l, p = 0.0, _wrap_angle(l + p)
        elif abs(t) < CLIFFORD_ANGLE_TOL:
            l, p = 0.0, _wrap_angle(p - l)

        z_gate = U1Gate() if use_u1 else RZGate()
        new_circuit = Circuit(1)
        new_circuit.append_gate(z_gate, 0, [l])
        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)
        new_circuit.append_gate(z_gate, 0, [t])
        if use_rx:
            new_circuit.append_gate(RXGate(), 0, [np.pi / 2])
        else:
            new_circuit.append_gate(SqrtXGate(), 0)
        new_circuit.append_gate(z_gate, 0, [p])
        circuit.become(new_circuit)


def gate_counts(circuit: Circuit) -> dict[str, int]:
    """Operation counts keyed by gate class name, e.g. {"TGate": 3020, ...}.

    Counts operations rather than reading circuit.gate_set, so a gate that
    appears in the gate set without any operations cannot inflate the report.
    """
    counts: dict[str, int] = {}
    for op in circuit:
        name = type(op.gate).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def print_gate_counts(circuit: Circuit) -> None:
    """Report the final gate breakdown, T count first among the summaries.

    T count is the number that matters for a fault-tolerant cost estimate and is
    not directly readable off the table, since it is TGate plus TdgGate.
    """
    counts = gate_counts(circuit)
    t_count = counts.get("TGate", 0) + counts.get("TdgGate", 0)
    # A barrier spans the qubits it separates, so it has to be excluded here or it
    # is reported as an entangling gate.
    multi_qubit = sum(
        1
        for op in circuit
        if op.num_qudits > 1
        and not isinstance(op.gate, (MeasurementPlaceholder, BarrierPlaceholder))
    )

    print(f"Compiled circuit has {len(circuit)} gates")
    width = max((len(name) for name in counts), default=0)
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {name:<{width}}  {count:>9d}")
    print(f"T count: {t_count} (TGate + TdgGate)")
    print(f"Multi-qubit gates: {multi_qubit}")


def decompose_rz_gauge_fixed(
    circuit: Circuit,
    synthesis_epsilon: float = 1e-8,
) -> Circuit:
    """Take a {Clifford, RZ} circuit to Clifford+T without the doubled rotations.

    Feed this the output of a workflow built with ``decompose_rz=False``, which
    stops with the rotation angles still exact.  The fix cannot be applied to a
    finished Clifford+T circuit, where the two gridsynth words have already been
    expanded into ~200 gates and recovering the intended angle would mean
    approximating an approximation.

    The pass list mirrors bqskit's own ordering in build_cliffordt_workflow --
    group single-qubit runs, decompose, replace exact Cliffords, round
    near-discrete z-rotations, then gridsynth what is left -- with only the
    decomposition swapped, so a run differs from a stock one purely in which of
    the two outer angles carries the rotation.
    """
    precision = int(log10(1 / synthesis_epsilon)) + 2
    passes = [
        GroupSingleQuditGatePass(),
        ForEachBlockPass(
            [GaugeFixedZXZXZDecomposition()],
            collection_filter=single_qudit_filter,
        ),
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        UnfoldPass(),
        *rz_decomposition_passes(precision),
    ]
    # bqskit runs passes in spawned workers, which unpickle the pass and import
    # its module.  That works for GaugeFixedZXZXZDecomposition whether this file
    # is run as a script (multiprocessing re-imports the parent's __main__) or
    # imported as a module, but it would not work for a class defined inside a
    # function or monkeypatched onto bqskit in this process.
    with Compiler() as compiler:
        return compiler.compile(circuit, passes)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Compile a QASM quantum circuit to the Clifford+T gate set. "
            "Outputs a .cliffordt.qasm file that can be passed directly to the transpile binary."
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
    parser.add_argument(
        "--no-rz-gauge-fix",
        action="store_true",
        help=(
            "Use bqskit's ZXZXZDecomposition unchanged. By default this script "
            "substitutes a gauge-fixed version, which halves the T count of every "
            "diagonal single-qubit rotation; see GaugeFixedZXZXZDecomposition. Pass "
            "this to reproduce stock bqskit output."
        ),
    )
    parser.add_argument(
        "--synthesis-epsilon",
        type=float,
        default=1e-8,
        help=(
            "Distance allowed between target and synthesised unitary, i.e. bqskit's "
            "synthesis_epsilon. Gridsynth precision is int(log10(1/epsilon)) + 2 "
            "digits. (default: 1e-8, bqskit's own default)"
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help=(
            "Seed for bqskit's numerical instantiation. bqskit is nondeterministic "
            "without one: repeated runs of the same input have been seen to differ by "
            "~40%% in T count, which makes gate counts unreproducible and unsafe to "
            "compare. Pass a different integer to sample another compilation; there is "
            "no unseeded mode, since an unrecorded seed is never preferable to a "
            "recorded one. (default: 0)"
        ),
    )
    # parser.add_argument(
    #    "--gpu",
    #    action="store_true",
    #    default=False,
    #    help="Use GPU-accelerated instantiation via QFactor (requires PyTorch with CUDA).",
    # )

    args: argparse.Namespace = parser.parse_args()
    overall_start: float = timer()

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
    # use_gpu: bool = args.gpu
    # if use_gpu and not _CUDA_AVAILABLE:
    #    print(
    #        "Warning: --gpu requested but CUDA is not available; falling back to CPU.",
    #        file=sys.stderr,
    #    )
    #    use_gpu = False

    #instantiate_options: dict = {}
    # if use_gpu:
    #    instantiate_options = {"method": "qfactor", "device": "cuda"}
    #    print("Using GPU-accelerated QFactor instantiation.")
    # else:
    #    print("Using CPU instantiation.")

    print("Compiling to Clifford+T...")

    compile_start: float = timer()
    machine: CliffordTModel = CliffordTModel(circuit.num_qudits)

    # CliffordTModel registers a workflow per optimization level at construction,
    # and compile() returns that registered workflow before it ever looks at its
    # own seed argument -- so seeding has to be done by registering a workflow
    # that already carries the seed.
    seed: int = args.seed

    # Two reasons to register a workflow rather than use the model's default: it is
    # the only way to seed (see above), and with the gauge fix we have to own the rz
    # decomposition, so it is registered with decompose_rz=False and stopped while
    # the rotation angles are still exact -- decompose_rz_gauge_fixed then finishes
    # the job.
    gauge_fix: bool = not args.no_rz_gauge_fix
    with warnings.catch_warnings():
        # CliffordTModel registers a default workflow for every optimization level
        # in its constructor, so registering ours always displaces one and bqskit
        # warns about it.  Displacing it is the whole point, and the warning's
        # advice (about Namespace packages fighting over a default) does not apply,
        # so drop just that message and leave every other warning visible.
        warnings.filterwarnings("ignore", message="Overwritting workflow")
        register_workflow(
            machine,
            build_circuit_workflow(
                OPTIMIZATION_LEVEL,
                synthesis_epsilon=args.synthesis_epsilon,
                decompose_rz=not gauge_fix,
                seed=seed,
            ),
            OPTIMIZATION_LEVEL,
            "circuit",
        )

    #circuit = compile(circuit, model=machine, instantiate_options=instantiate_options or None)
    circuit = compile(circuit, model=machine, optimization_level=OPTIMIZATION_LEVEL, seed=seed)
    if gauge_fix:
        circuit = decompose_rz_gauge_fixed(circuit, args.synthesis_epsilon)
    compile_end: float = timer()
    print(f"Compilation took {(compile_end - compile_start):.2f} seconds")

    # Flatten any CircuitGate wrappers that bqskit may have left around
    # sub-circuits (e.g. U3Gate wrapped in a CircuitGate).  Without this,
    # the transpile binary crashes because it has no rule for CircuitGate.
    circuit.unfold_all()

    # Remove IdentityGate operations: they are semantic no-ops but bqskit
    # serialises them as a custom "identity1" gate using U(0,0,0).  When the
    # QASM is reloaded, that custom gate definition is parsed back as a
    # CircuitGate(U3Gate), which causes the transpile binary to crash.
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
    print_gate_counts(circuit)

    # Save compiled circuit as QASM (for use with the transpile binary)
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
    print(f"Total time: {(timer() - overall_start):.2f} seconds")


if __name__ == "__main__":
    main()
