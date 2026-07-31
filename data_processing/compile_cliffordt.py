#!/usr/bin/env python3
"""
Compile an arbitrary OpenQASM circuit into the Clifford+T gate set, via qiskit or bqskit.

Output gate set: h, s, sdg, x, y, z, t, tdg, cx (plus measure/barrier/reset and
any classical control flow that was present in the input -- the bqskit backend
rejects circuits with control flow, since bqskit's own `Circuit` has no concept
of it). The bqskit backend also natively emits sx/sxdg (sqrt(X) and its
inverse): bqskit's own ZXZXZ decomposition and Clifford+T gate set treat sx as
a Clifford generator in its own right rather than expanding it to h/s, and the
downstream Rust `transpile` binary understands both natively, so this is not
rewritten away. The qiskit backend never produces either.

Pipeline
--------
1. Load the QASM file (OpenQASM 2 first, falling back to OpenQASM 3 -- which
   needs the optional `qiskit-qasm3-import` package).
2. Preopt: transpile to {u, cx} so that every multi-qubit gate (ccx, cswap,
   cry, rzz, ryy, ...) is broken down into cx plus single-qubit rotations.
   Both backends resynthesise from this same {u, cx} circuit -- necessary for
   the qiskit backend (which only knows how to re-synthesise 1-qubit runs, not
   arbitrary multi-qubit gates) and a real, if smaller, improvement for the
   bqskit backend too (bqskit's own partitioner refragments single-qubit runs
   across 2-qubit block boundaries regardless of input quality, so it cannot
   benefit from the preopt step as much as qiskit's own pipeline does).
3. Resynthesise over Clifford+T (--backend bqskit, the default, or qiskit):

   qiskit: merge each maximal run of consecutive single-qubit gates on a wire
   into one 2x2 matrix, then re-synthesise that matrix:
     * Clifford            -> shortest word in {h, s, sdg, x, y, z} (BFS table).
     * exactly representable -> exact sequence.  A gate is exact iff its ZXZ
       Euler angles are all integer multiples of pi/4, since Rz(k*pi/4) is a
       T/S/Z word and Rx(theta) = H Rz(theta) H.
     * otherwise           -> each generic Rz in the ZXZ decomposition is
       approximated to --epsilon via gridsynth (see below), memoised per angle.
     Neither "exact" path is taken on trust: the word it produces is measured
     against the target and rejected if it is off by more than --tol, which
     defaults to --epsilon.  The check matters because the Clifford lookup key
     rounds to 7 decimals -- deliberately, so that gates differing only by
     floating-point noise share a table entry -- which also makes the lookup
     match anything within ~5e-8 of a Clifford.  Unguarded, that discards small
     rotations for free: the pi/2^k tail of a wide QFT, for instance, where
     every rotation below ~1e-7 would cost zero T at an error hundreds of
     times --epsilon.  Rotations that really are within --epsilon of a
     Clifford still cost nothing, but that is now the synthesis backend's
     decision, made against the requested accuracy, and it shows up in the
     reported error.
     Then cleans up: cancels adjacent inverse pairs (t.tdg, h.h, cx.cx, ...)
     and collapses blocks of gates that have a shorter exact form (t.t -> s,
     tdg.tdg.tdg.tdg -> z, any Clifford block -> its shortest word).  The
     block collapses are exact; the whole-run rewrite goes through the same
     guarded exact paths as above, so it can trade up to --tol of accuracy for
     a shorter run, and what it spends is added to the reported error bound.

   bqskit: hands the {u,cx} circuit to bqskit's own compiler
   (bqskit.compiler.compile) using bqskit's CliffordTModel with this script's
   own workflow registered against it (build_bqskit_workflow -- bqskit's own
   build_circuit_workflow minus its multi-qudit retargeting stage, which
   unconditionally and needlessly re-synthesises already-native <=3-qubit
   blocks; see its docstring), then re-synthesises the diagonal single-qubit
   rotations via bqskit's own ZXZXZDecomposition and stock, pygridsynth-based
   GridSynthPass (on by default, --bqskit-inline-decompose-rz uses bqskit's
   own inline decompose_rz=True workflow instead -- see decompose_rz_tracked's
   docstring for what the difference actually is). Produces fewer T gates
   than the qiskit backend on every benchmark measured so far, which is why
   it is the default -- but rejects circuits with classical control flow
   (which bqskit's own Circuit has no concept of), unlike qiskit's own
   backend, which is kept as the fallback for those.

Rotation synthesis: gridsynth
------------------------------
The Ross-Selinger algorithm, near T-optimal at T ~ 3*log2(1/epsilon) per
rotation. The qiskit backend uses qiskit's own Rust implementation
(qiskit.synthesis.gridsynth_rz, qiskit >= 2.5, ~5 ms per distinct rotation),
falling back to pygridsynth -- the pure-Python/mpmath implementation, ~8x
slower -- for the rare angles rsgridsynth 0.2.0 panics on at coarse epsilon.
That fallback already handles gridsynth failing on part of a circuit; there is
no separate "worse but always works" mode. The bqskit backend uses only
pygridsynth (bqskit's own stock GridSynthPass), with no Rust extension
involved at all. Each generic 1q gate needs up to 3 Rz rotations (ZXZ Euler
angles), so the error per gate is up to 3*epsilon; angles that are exact
multiples of pi/4 are synthesised exactly and cost nothing.  Each distinct
rotation is synthesised once and reused, so cost scales with the number of
distinct angles rather than the number of gates.

--epsilon's default depends on --backend (1e-10 for qiskit, 1e-8 for bqskit)
rather than being one shared value, because "epsilon" is not the same
quantity across the two implementations in terms of delivered accuracy: on
the reference slice (data/hubbard_18_slice600.qasm), bqskit at its own
default of 1e-8 already measures fidelity 1.000000000000, statistically
identical (to the 12 decimals this script prints) to what tightening it to
1e-10 gets -- tightening costs 2348 -> 4900 T for no measurable accuracy
gain, because bqskit's own numerical instantiation step (also controlled by
this parameter) turns out to work much harder for a target it was already
comfortably beating. qiskit's own resynthesis has no equivalent instantiation
step, so 1e-10 costs it nothing extra and is worth keeping as its default.

The result is written next to the input as <name>.cliffordt.qasm (OpenQASM 2,
or OpenQASM 3 if the circuit uses control flow that OpenQASM 2 cannot express),
and per-circuit gate/T counts go to stdout and optionally to --stats JSON.

Verification
------------
basis + error bound: always run, no flag needed -- both are cheap (no
    simulation), and a broken basis is worth failing the run over regardless
    of whether numeric verification was asked for.  Every operation in the
    output must be in the Clifford+T basis (plus measure/barrier/reset/control
    flow); a violation fails the run.  For the qiskit backend, the per-rewrite
    errors measured during resynthesis and clean-up are summed into an error
    bound: each rewrite replaces one wire's run by a phase-aligned
    approximation of it, so by subadditivity of the spectral norm that sum is
    a genuine upper bound on ||U_compiled - U_unrolled||_2 -- taking qiskit's
    own unroll and inverse cancellation as exact.  The bqskit backend (by
    default) gets a narrower analogue from bqskit's own machinery: each
    ZXZXZDecomposition/GridSynthPass ForEachBlockPass call runs with
    calculate_error_bound=True, so bqskit measures the exact unitary distance
    of every 1-qubit block before and after and composes them via
    PassData.update_error_mul -- see decompose_rz_tracked's docstring for
    what this covers and doesn't (notably: not RoundToDiscreteZPass's own
    rounding, and not bqskit's own earlier 2-qubit block instantiation). With
    --bqskit-inline-decompose-rz, bqskit has no equivalent bookkeeping at
    all, so no error bound is reported.

--verify: numeric fidelity only, cascading from exact to sampled as the
    circuit grows, so it is always tractable -- unlike an error bound, which
    is only ever a bound, this actually measures how close the compiled
    circuit's action is to the source's.
  unitary fidelity: up to DENSE_VERIFY_MAX_QUBITS (10) qubits and
    DENSE_VERIFY_MAX_GATES (20000) gates.  The full dense 2^n x 2^n
    comparison, exhaustive but limited to ~12 qubits by memory.
  statevector fidelity: up to STATEVECTOR_VERIFY_MAX_QUBITS (24) qubits,
    subject to a work budget of gates * 2^qubits <= STATEVECTOR_VERIFY_MAX_OPS
    (5e9, about a minute).  Evolves one Haar-random state (seeded by
    --verify-seed, so runs are reproducible) through both circuits and
    compares.  A single random state is a strong test: a systematic error
    survives it with negligible probability.
  random-window fidelity: the automatic fallback for circuits too large for
    either check above, or containing classical control flow (which both
    skip).  Samples WINDOW_VERIFY_COUNT (5) random contiguous windows of the
    *source* circuit's instructions (not the compiled output -- resynthesis
    restructures gates, so indices would not correspond).  Each window is
    grown greedily from a random start point, tracking the distinct qubits
    touched, until either WINDOW_VERIFY_MAX_QUBITS (24, the same cap as the
    direct statevector check) or WINDOW_VERIFY_MAX_OPS (that check's own
    budget, divided across the samples so total cost stays bounded regardless
    of the real circuit's size) would be exceeded, or the next instruction has
    classical bits, is a measurement/reset, or is control flow (a hard stop;
    barrier is skipped, not stopped on).  Each window is independently
    compiled through the same backend and this run's other arguments
    (exercising preopt too, not just resynthesis) and fidelity-checked against
    its own source window (dense or statevector, whichever applies -- a
    window is always small enough for one of them).  The worst fidelity found
    across the samples is reported, alongside each window's own qubit/gate
    count and fidelity.  This is a spot check, not a proof: it only catches
    what shows up in the sampled windows, and how much of the circuit a
    window's greedy growth can cover before hitting its qubit cap depends on
    the circuit's actual locality -- which is exactly why several independent
    samples are taken rather than one, and why none of this is exposed as a
    tunable flag: the point is that it always runs, at a bounded cost, no
    matter how large or oddly-shaped the input circuit is.

Examples
--------
    ./compile_cliffordt.py ../data/qasmbench/ising_n26.qasm
    ./compile_cliffordt.py ../data/qasmbench/dnn_n8.qasm -o dnn_n8.ct.qasm \
        --epsilon 1e-6 --verify
    ./compile_cliffordt.py ../data/qasmbench/*.qasm -o out_dir --stats stats.json
    ./compile_cliffordt.py circuit.qasm --backend bqskit --seed 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit.circuit import ControlFlowOp, Gate, Qubit
from qiskit.quantum_info import Operator, random_statevector
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, RemoveBarriers
from qiskit.circuit.library import (
    CXGate,
    HGate,
    SdgGate,
    SGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.compile import compile as bqskit_compile
from bqskit.compiler.registry import register_workflow
# bqskit-ft is a separate distribution installed alongside the editable-cloned
# bqskit/ (see its __init__.py for how bqskit.ft resolves at runtime via
# pkgutil.extend_path -- a dynamic sys.path merge pyright cannot evaluate
# statically, hence the ignores below).
from bqskit.ft.cliffordt.cliffordtgates import clifford_t_gates  # pyright: ignore[reportMissingImports]
from bqskit.ft.cliffordt.cliffordtmodel import CliffordTModel  # pyright: ignore[reportMissingImports]
from bqskit.ft.cliffordt.defaultworkflow import (  # pyright: ignore[reportMissingImports]
    clifford_replace,
    single_qudit_filter,
)
from bqskit.ft.ftpasses.gridsynth import GridSynthPass  # pyright: ignore[reportMissingImports]
from bqskit.ft.ftpasses.rounding import RoundToDiscreteZPass  # pyright: ignore[reportMissingImports]
from bqskit.ft.rules.isolate_rz import IsolateRZGatePass  # pyright: ignore[reportMissingImports]
from bqskit.ir.gates import BarrierPlaceholder, IdentityGate, MeasurementPlaceholder
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.rules.zxzxz import ZXZXZDecomposition
from bqskit.passes.util.log import LogErrorPass
from bqskit.passes.util.random import SetRandomSeedPass
from bqskit.passes.util.unfold import UnfoldPass

try:  # qiskit >= 2.5: Ross-Selinger in Rust, the rotation synthesis both backends use
    from qiskit.synthesis import gridsynth_rz
except ImportError:
    gridsynth_rz = None

try:  # optional fallback for the angles rsgridsynth 0.2.0 panics on
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates
except ImportError:
    mpmath = None
    gridsynth_gates = None

# sx/sxdg (sqrt(X) and its inverse) are included because the bqskit backend
# emits them natively -- bqskit's own Clifford+T gate set treats sx as a
# Clifford generator in its own right, and the downstream Rust `transpile`
# binary understands both directly (src/transpile.rs's Gate1Q::SX/SXdg) -- not
# because either is reachable from the qiskit backend, which never emits them.
CLIFFORD_T_BASIS = ("h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "t", "tdg", "cx")
PI_4 = math.pi / 4

# qiskit's transpile() default is None, which resolves to level 2 -- but level 2
# restructures which single-qubit gates sit adjacent to which cx gates relative
# to level 1, in a way that produces more, smaller single-qubit runs for the
# qiskit backend to re-synthesise.  Measured on an 18-qubit/600-gate slice of a
# Hubbard benchmark: level 1 merges runs down to 18 non-Clifford rotations
# (1858 T); qiskit's own default merges them into 39 (7085 T).  Same cx count
# either way, so this is purely about how well the runs merge for this
# script's purposes, not circuit quality by qiskit's own metrics.  Pinned
# rather than exposed as a CLI option, since a worse choice was never useful
# here.  Shared by both backends' preopt step (unroll_to_u_cx).
UNROLL_OPTIMIZATION_LEVEL = 1

# The optimization_level slot build_bqskit_workflow's own workflow is
# registered and invoked under -- register_workflow and bqskit_compile both
# require one, and the two calls must agree on which slot to use. Since
# build_bqskit_workflow always builds the same pass list regardless of this
# number (unlike bqskit's own levels 2-4, which select genuinely different,
# slower workflows), there is nothing to gain from exposing it as a CLI
# option; it is fixed at 1 purely because some value has to be picked.
BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL = 1

# --epsilon's default, backend-dependent -- see the module docstring's
# "Rotation synthesis: gridsynth" section for why these are not one shared
# value: bqskit's own instantiation step (not just its final rotation
# synthesis) is also controlled by this parameter, and tightening it to
# qiskit's default measurably inflates bqskit's T count with no measured
# accuracy gain, since bqskit already saturates this script's fidelity check
# at its own default of 1e-8.
QISKIT_EPSILON_DEFAULT = 1e-10
BQSKIT_EPSILON_DEFAULT = 1e-8

# Single-qubit Clifford generators used to build the shortest-word table.
CLIFFORD_GENERATORS = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
}

GATE_MATRICES = dict(
    CLIFFORD_GENERATORS,
    t=np.array([[1, 0], [0, np.exp(1j * PI_4)]], dtype=complex),
    tdg=np.array([[1, 0], [0, np.exp(-1j * PI_4)]], dtype=complex),
    id=np.eye(2, dtype=complex),
)

# Diagonal gates, as the multiple of pi/4 they rotate about z by.  A block of
# them commutes and collapses to a single Rz(k * pi/4).
DIAGONAL_PHASES = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4, "id": 0}

CLIFFORD_1Q_NAMES = frozenset(CLIFFORD_GENERATORS) | {"id"}

# pygridsynth emits words over {H, S, T, X, W}; W is the e^{i pi/4} global phase,
# which we drop because the phase is recomputed against the target anyway.
# (qiskit's gridsynth_rz returns a circuit with lowercase gate names instead.)
GRIDSYNTH_NAMES = {"H": "h", "S": "s", "T": "t", "X": "x"}

# Working precision pygridsynth is driven at, matching bqskit's GridSynthPass.
GRIDSYNTH_DPS = 128

# Floor on the exactness tolerance (see CliffordTSynthesizer.tol).  A run product
# is accumulated over hundreds of 2x2 multiplications, so comparing a word
# against it is only meaningful down to about this much floating-point noise.
EXACTNESS_FLOOR = 1e-12

# Non-gate operations that are allowed to survive into the output.
PASSTHROUGH_OPS = frozenset({"measure", "barrier", "reset", "delay"})

# Thresholds for the --verify fidelity cascade (dense unitary -> single random
# statevector -> automatic random-window sampling). Not exposed as CLI flags,
# so the cascade always has somewhere to fall back to instead of dead-ending
# at "no numeric check".
DENSE_VERIFY_MAX_QUBITS = 10
DENSE_VERIFY_MAX_GATES = 20_000
STATEVECTOR_VERIFY_MAX_QUBITS = 24
STATEVECTOR_VERIFY_MAX_OPS = 5e9

# Random-window sampling: the fallback for circuits too large for either check
# above (or containing classical control flow, which both skip). Derived from
# the direct-check constants above rather than invented fresh, so the total
# cost of windowed sampling stays bounded by roughly the same budget as a
# single direct statevector check would have used, regardless of how large the
# real circuit is -- this is what makes it always tractable.
WINDOW_VERIFY_COUNT = 5
WINDOW_VERIFY_MAX_QUBITS = STATEVECTOR_VERIFY_MAX_QUBITS
WINDOW_VERIFY_MAX_OPS = STATEVECTOR_VERIFY_MAX_OPS / WINDOW_VERIFY_COUNT
WINDOW_VERIFY_MAX_ATTEMPTS = WINDOW_VERIFY_COUNT * 5


def canonical_key(matrix: np.ndarray, decimals: int = 7) -> tuple:
    """Hashable key for a 2x2 unitary, insensitive to global phase.

    The rounding is deliberately coarse so that gates that differ only by
    floating-point noise (e.g. an H that came back from the transpiler as a u
    gate) hit the same table entry.  It is far finer than the accuracy of any
    approximate synthesis, so sharing a gridsynth sequence between two
    matrices with the same key is harmless.

    The pivot has to be chosen off the *rounded* magnitudes.  Unitarity forces
    |a| == |d| and |b| == |c|, so the largest element is always tied with another
    one; a raw argmax would let floating-point noise pick which, and the two
    choices normalise to different keys for the same matrix.  Rounding first
    makes the tie exact, so argmax breaks it deterministically by index.
    """
    flat = matrix.reshape(-1)
    pivot = int(np.argmax(np.round(np.abs(flat), decimals)))
    normalised = flat * np.exp(-1j * np.angle(flat[pivot]))
    rounded = np.round(normalised, decimals)
    # -0.0 and 0.0 must hash the same.
    rounded = rounded + 0.0
    return tuple((c.real, c.imag) for c in rounded)


def global_phase_between(target: np.ndarray, built: np.ndarray) -> float:
    """Phase gamma minimising ||target - exp(i*gamma) * built||.

    The minimiser is arg(tr(built^dag @ target)), which averages over all four
    elements.  Taking the ratio of one element instead would let that element's
    approximation error alone set the phase, which matters because `built` is an
    approximation of `target` everywhere except on the exact paths.
    """
    return float(np.angle(np.trace(built.conj().T @ target)))


def spectral_error(target: np.ndarray, built: np.ndarray, phase: float) -> float:
    """||exp(i*phase) * built - target||_2."""
    return float(np.linalg.norm(np.exp(1j * phase) * built - target, ord=2))


def build_clifford_words() -> dict[tuple, tuple[str, ...]]:
    """Map every single-qubit Clifford (up to phase) to a shortest generator word."""
    identity = np.eye(2, dtype=complex)
    words: dict[tuple, tuple[str, ...]] = {canonical_key(identity): ()}
    matrices: dict[tuple, np.ndarray] = {canonical_key(identity): identity}
    frontier = [canonical_key(identity)]
    while frontier:
        next_frontier = []
        for key in frontier:
            for name, gen in CLIFFORD_GENERATORS.items():
                product = gen @ matrices[key]
                new_key = canonical_key(product)
                if new_key in words:
                    continue
                words[new_key] = words[key] + (name,)
                matrices[new_key] = product
                next_frontier.append(new_key)
        frontier = next_frontier
    return words


def word_matrix(word: Iterable[str]) -> np.ndarray:
    """Unitary of a word of named single-qubit gates, in circuit order."""
    matrix = np.eye(2, dtype=complex)
    for name in word:
        matrix = GATE_MATRICES[name] @ matrix
    return matrix


def word_error(target: np.ndarray, word: Iterable[str]) -> tuple[float, float]:
    """(error, global phase) of a named gate word against `target`.

    The phase is the one that minimises the error, i.e. the one _from_word bakes
    into the circuit it builds, so the error returned is the error of the
    circuit that will actually be emitted.
    """
    built = word_matrix(word)
    phase = global_phase_between(target, built)
    return spectral_error(target, built, phase), phase


def rz_pi_4_word(k: int) -> tuple[str, ...]:
    """Shortest Clifford+T word (up to global phase) for Rz(k * pi/4)."""
    return {
        0: (),
        1: ("t",),
        2: ("s",),
        3: ("s", "t"),
        4: ("z",),
        5: ("sdg", "tdg"),
        6: ("sdg",),
        7: ("tdg",),
    }[k % 8]


def collapse_diagonal_blocks(names: list[str]) -> list[str]:
    """Replace each maximal block of diagonal gates by its shortest equivalent."""
    out: list[str] = []
    index = 0
    while index < len(names):
        if names[index] not in DIAGONAL_PHASES:
            out.append(names[index])
            index += 1
            continue
        total = 0
        while index < len(names) and names[index] in DIAGONAL_PHASES:
            total += DIAGONAL_PHASES[names[index]]
            index += 1
        out.extend(rz_pi_4_word(total))
    return out


def collapse_clifford_blocks(
    names: list[str], clifford_words: dict[tuple, tuple[str, ...]]
) -> list[str]:
    """Replace each maximal block of Clifford gates by its shortest word."""
    out: list[str] = []
    index = 0
    while index < len(names):
        if names[index] not in CLIFFORD_1Q_NAMES:
            out.append(names[index])
            index += 1
            continue
        start = index
        while index < len(names) and names[index] in CLIFFORD_1Q_NAMES:
            index += 1
        block = names[start:index]
        shortest = clifford_words.get(canonical_key(word_matrix(block)))
        out.extend(block if shortest is None or len(shortest) > len(block) else shortest)
    return out


class CliffordTSynthesizer:
    """Re-synthesise single-qubit unitaries over {h, s, sdg, x, y, z, t, tdg} via gridsynth."""

    def __init__(
        self,
        epsilon: float = 1e-10,
        tol: Optional[float] = None,
    ) -> None:
        self.epsilon = epsilon
        # How much error an "exact" rewrite is allowed to introduce.  Defaulting
        # it to epsilon keeps the exact paths from being looser than the
        # approximate one: a rotation only comes out free if it really is within
        # the requested accuracy of a Clifford.
        self.tol = max(epsilon, EXACTNESS_FLOOR) if tol is None else tol
        self._decomposer = OneQubitEulerDecomposer(basis="ZXZ")
        self._clifford_words = build_clifford_words()
        self._gridsynth_cache: dict[float, tuple[str, ...]] = {}
        if gridsynth_rz is None and gridsynth_gates is None:
            raise RuntimeError(
                "gridsynth needs qiskit >= 2.5 (which ships "
                "qiskit.synthesis.gridsynth_rz) or the pygridsynth package"
            )
        if mpmath is not None:
            mpmath.mp.dps = max(mpmath.mp.dps, GRIDSYNTH_DPS)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_clifford = 0
        self.n_exact = 0
        self.n_approx = 0
        self.n_merged = 0
        self.max_error = 0.0
        self.error_bound = 0.0

    def _record(self, error: float) -> None:
        """Account for one rewrite's error.

        Every rewrite replaces one wire's run by a phase-aligned approximation,
        so the sum bounds the error of the whole circuit (subadditivity of the
        spectral norm), while the max is the worst single rewrite.
        """
        self.max_error = max(self.max_error, error)
        self.error_bound += error

    def synthesize(self, matrix: np.ndarray, _run=None) -> QuantumCircuit:
        """Return a 1-qubit Clifford+T circuit implementing `matrix`."""
        exact = self._exact(matrix)
        if exact is not None:
            circuit, kind, error = exact
            if kind == "clifford":
                self.n_clifford += 1
            else:
                self.n_exact += 1
            self._record(error)
            return circuit
        self.n_approx += 1
        return self._gridsynth(matrix)

    def shorten_run(self, matrix: np.ndarray, run: list[Gate]) -> Optional[QuantumCircuit]:
        """Shorten an already-compiled run of Clifford+T gates.

        Returns None if nothing can be improved.  Two rewrites are tried: the
        whole run at once (it may be Clifford, or a pi/4 rotation), and failing
        that a local collapse of sub-blocks -- a gridsynth word is a long
        h/t/tdg sequence which is *not* exactly representable as a whole, but its
        diagonal sub-blocks (t.t == s, tdg^4 == z, ...) are, and collapsing them
        removes T gates for free.

        The block collapse is exact.  The whole-run rewrite goes through _exact,
        so it is only as exact as --tol: it can also merge two genuine rotations
        whose product happens to land within --tol of a Clifford.  Both report
        what they cost, so it lands in the error bound either way.
        """
        exact = self._exact(matrix)
        if exact is not None and gate_cost(exact[0]) < gate_cost(run):
            self.n_merged += 1
            self._record(exact[2])
            return exact[0]

        names = [gate.name for gate in run]
        if not all(name in GATE_MATRICES for name in names):
            return None
        collapsed = collapse_clifford_blocks(collapse_diagonal_blocks(names), self._clifford_words)
        if gate_cost(collapsed) >= gate_cost(names):
            return None
        # Exact by construction; this only guards against a bug in the collapses.
        error, phase = word_error(matrix, collapsed)
        if error > self.tol:
            return None
        self.n_merged += 1
        self._record(error)
        return self._from_word(tuple(collapsed), matrix, phase)

    def _exact(self, matrix: np.ndarray) -> Optional[tuple[QuantumCircuit, str, float]]:
        """Exact synthesis as (circuit, kind, error), or None if it needs approximating.

        A candidate word is only accepted once it has been measured against
        `matrix` and found to be within self.tol.  The Clifford lookup needs that
        check because canonical_key rounds to 7 decimals, so the table matches
        anything within ~5e-8 of a Clifford; without it, every rotation smaller
        than that -- the pi/2^k tail of a wide QFT, say -- would be silently
        thrown away for free at an error far above --epsilon.  The pi/4 path
        needs it because _is_pi_4_multiple accepts angles up to --tol off a
        multiple, and three such angles compound.
        """
        word = self._clifford_words.get(canonical_key(matrix))
        if word is not None:
            error, phase = word_error(matrix, word)
            if error <= self.tol:
                return self._from_word(word, matrix, phase), "clifford", error

        euler = self._decomposer(matrix)
        angles = [inst.operation.params[0] for inst in euler.data if inst.operation.params]
        if all(self._is_pi_4_multiple(a) for a in angles):
            word = self._euler_word(euler, self._pi_4_word)
            error, phase = word_error(matrix, word)
            if error <= self.tol:
                return self._from_word(word, matrix, phase), "pi_4", error
        return None

    def _is_pi_4_multiple(self, angle: float) -> bool:
        return abs(angle / PI_4 - round(angle / PI_4)) < self.tol

    def _euler_word(self, euler: QuantumCircuit, rz_word) -> tuple[str, ...]:
        """Assemble a word for an Rz/Rx circuit, sending each angle to `rz_word`."""
        word: list[str] = []
        for inst in euler.data:
            name = inst.operation.name
            inner = rz_word(inst.operation.params[0])
            if name == "rz":
                word.extend(inner)
            elif name == "rx":
                # Rx(theta) = H Rz(theta) H
                if inner:
                    word.extend(("h", *inner, "h"))
            else:  # pragma: no cover - ZXZ only emits rz/rx
                raise RuntimeError(f"unexpected gate {name} in ZXZ decomposition")
        return tuple(word)

    def _pi_4_word(self, angle: float) -> tuple[str, ...]:
        return rz_pi_4_word(round(angle / PI_4))

    def _from_word(
        self,
        word: tuple[str, ...],
        target: np.ndarray,
        phase: Optional[float] = None,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(1)
        for name in word:
            getattr(circuit, name)(0)
        circuit.global_phase = (
            global_phase_between(target, word_matrix(word)) if phase is None else phase
        )
        return circuit

    def _gridsynth(self, matrix: np.ndarray) -> QuantumCircuit:
        """Approximate via ZXZ Euler angles, each Rz handed to gridsynth."""
        word = self._euler_word(self._decomposer(matrix), self._gridsynth_word)
        error, phase = word_error(matrix, word)
        self._record(error)
        return self._from_word(word, matrix, phase)

    def _gridsynth_word(self, angle: float) -> tuple[str, ...]:
        if self._is_pi_4_multiple(angle):
            return self._pi_4_word(angle)
        key = round(angle, 12)
        word = self._gridsynth_cache.get(key)
        if word is None:
            word = self._synthesize_rz(angle)
            self._gridsynth_cache[key] = word
        return word

    def _synthesize_rz(self, angle: float) -> tuple[str, ...]:
        """Ross-Selinger word for Rz(angle), accurate to self.epsilon."""
        if gridsynth_rz is None:
            return self._pygridsynth_word(angle)
        try:
            circuit = gridsynth_rz(angle, self.epsilon)
        except BaseException as error:
            # rsgridsynth 0.2.0 panics on some angles at coarse epsilon ("Invalid
            # coefficients for inverse sqrt2 multiplication"): 26% of 300 random
            # angles at 1e-2, 6% at 1e-4, none at 1e-6 or below.  Which angles fail
            # depends on process state, not just the angle, so retrying is not a
            # fix.  A pyo3 panic is a BaseException, so it would otherwise escape
            # the per-file error handling in main().
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if gridsynth_gates is None:
                raise RuntimeError(
                    f"qiskit's gridsynth failed on angle {angle} at epsilon "
                    f"{self.epsilon:g} ({type(error).__name__}: {error}). Use a "
                    "smaller --epsilon or install pygridsynth as a fallback."
                ) from error
            return self._pygridsynth_word(angle)
        return tuple(inst.operation.name for inst in circuit.data)

    def _pygridsynth_word(self, angle: float) -> tuple[str, ...]:
        """Same rotation via pygridsynth, the implementation bqskit-ft calls."""
        if gridsynth_gates is None or mpmath is None:  # __init__ checks this
            raise RuntimeError("pygridsynth is not installed")
        sequence = gridsynth_gates(mpmath.mpf(angle), mpmath.mpf(self.epsilon))
        # pygridsynth returns the word in matrix order, so it is applied in reverse.
        return tuple(GRIDSYNTH_NAMES[symbol] for symbol in reversed(sequence) if symbol != "W")


def load_circuit(path: Path) -> QuantumCircuit:
    """Load an OpenQASM 2 (preferred) or OpenQASM 3 file."""
    try:
        return qasm2.load(
            path,
            include_path=(str(path.parent), "."),
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        )
    except qasm2.QASM2Error as qasm2_error:
        try:
            return qasm3.load(str(path))
        except Exception as qasm3_error:
            raise RuntimeError(
                f"could not parse {path} as OpenQASM 2 ({qasm2_error}) "
                f"or OpenQASM 3 ({qasm3_error})"
            ) from qasm2_error


def rewrite_single_qubit_runs(circuit: QuantumCircuit, resynthesize) -> QuantumCircuit:
    """Re-synthesise every maximal run of single-qubit gates on every wire.

    Consecutive single-qubit gates on a wire are accumulated into one 2x2 matrix
    and handed to `resynthesize(matrix, run)`, which returns a replacement
    1-qubit circuit, or None to keep the original gates.  Working on runs
    rather than
    individual gates keeps the output much shorter.  Non-unitary ops
    (measure/reset/barrier) and multi-qubit gates flush the wires they touch, so
    the original ordering is preserved.
    """
    # copy_empty_like keeps loose qubits, which control-flow blocks are built from.
    out = circuit.copy_empty_like()
    pending: dict[Qubit, tuple[np.ndarray, list[Gate]]] = {}

    def flush(qubits: Iterable[Qubit]) -> None:
        for qubit in qubits:
            entry = pending.pop(qubit, None)
            if entry is None:
                continue
            matrix, run = entry
            replacement = resynthesize(matrix, run)
            if replacement is None:
                for gate in run:
                    out.append(gate, [qubit], [])
            else:
                out.compose(replacement, qubits=[qubit], inplace=True)

    for inst in circuit.data:
        operation, qubits, clbits = inst.operation, inst.qubits, inst.clbits
        if len(qubits) == 1 and not clbits and isinstance(operation, Gate):
            try:
                matrix = np.asarray(operation.to_matrix(), dtype=complex)
            except Exception:
                matrix = None
            if matrix is not None:
                qubit = qubits[0]
                previous = pending.get(qubit)
                if previous is None:
                    pending[qubit] = (matrix, [operation])
                else:
                    pending[qubit] = (matrix @ previous[0], previous[1] + [operation])
                continue

        flush(qubits)
        if isinstance(operation, ControlFlowOp):
            operation = operation.replace_blocks(
                rewrite_single_qubit_runs(block, resynthesize) for block in operation.blocks
            )
        out.append(operation, qubits, clbits)

    flush(list(pending))
    return out


def cancel_inverses(circuit: QuantumCircuit) -> QuantumCircuit:
    # Self-inverse gates are given bare, genuine inverse pairs as tuples.
    cancellable = [
        HGate(),
        XGate(),
        YGate(),
        ZGate(),
        CXGate(),
        (TGate(), TdgGate()),
        (SGate(), SdgGate()),
    ]
    return PassManager([InverseCancellation(cancellable)]).run(circuit)


def gate_cost(gates) -> tuple[int, int]:
    """(T count, gate count) of a circuit, a list of gates, or a list of names.

    Counts inside control-flow blocks as well.  Without that, shortening a run in
    an if/for body leaves the top-level cost unchanged, and the clean-up loop in
    compile_qiskit would stop after a single round.
    """
    if isinstance(gates, QuantumCircuit):
        return operation_counts_cost(gates)
    names = [gate if isinstance(gate, str) else gate.name for gate in gates]
    return sum(name in ("t", "tdg") for name in names), len(names)


def operation_counts(circuit: QuantumCircuit) -> dict[str, int]:
    """Operation-name counts, recursing into control-flow blocks."""
    counts: dict[str, int] = {}
    for inst in circuit.data:
        operation = inst.operation
        if isinstance(operation, ControlFlowOp):
            for block in operation.blocks:
                for name, n in operation_counts(block).items():
                    counts[name] = counts.get(name, 0) + n
        else:
            counts[operation.name] = counts.get(operation.name, 0) + 1
    return counts


def operation_counts_cost(circuit: QuantumCircuit) -> tuple[int, int]:
    counts = operation_counts(circuit)
    t_count = counts.get("t", 0) + counts.get("tdg", 0)
    return t_count, sum(n for name, n in counts.items() if name != "barrier")


def unroll_to_u_cx(circuit: QuantumCircuit) -> QuantumCircuit:
    """The {u,cx} unroll + optimization step both backends resynthesise from.

    Structurally necessary, not just an optimization: it is what breaks
    multi-qubit gates neither backend otherwise knows how to re-synthesise
    (ccx, cp, rzz, ...) down into {1-qubit unitary, cx}.  UNROLL_OPTIMIZATION_LEVEL
    also matters a great deal for how well single-qubit runs merge before
    resynthesis (see its comment).
    """
    return transpile(circuit, basis_gates=["u", "cx"], optimization_level=UNROLL_OPTIMIZATION_LEVEL)


def compile_qiskit(
    unrolled: QuantumCircuit,
    synth: CliffordTSynthesizer,
    optimize: bool = True,
    max_rounds: int = 5,
) -> QuantumCircuit:
    """Re-synthesise an already-{u,cx}-unrolled circuit over Clifford+T, then clean up."""
    out = rewrite_single_qubit_runs(unrolled, synth.synthesize)
    if not optimize:
        return out
    # Cancelling inverses brings new gates together, which lets the next round of
    # block collapsing find more, so iterate until it stops paying off.  Inverse
    # cancellation and the block collapses are exact; the whole-run rewrite in
    # shorten_run can spend up to --tol per run, and does so at most once per
    # round, which synth.error_bound accounts for.
    for _ in range(max_rounds):
        cost = gate_cost(out)
        out = cancel_inverses(rewrite_single_qubit_runs(out, synth.shorten_run))
        if gate_cost(out) >= cost:
            break
    return out


def circuit_stats(circuit: QuantumCircuit) -> dict:
    """Reported per-circuit statistics.

    The counts come from operation_counts, which recurses into control-flow
    blocks; QuantumCircuit.count_ops and .size do not, and would report an
    if/else body as a single `if_else` op with no T gates in it.  The two depths
    are qiskit's, which do not recurse either -- a control-flow block counts as
    one layer -- since the depth of a circuit whose length depends on a
    measurement outcome is not well defined in the first place.
    """
    counts = {name: int(n) for name, n in operation_counts(circuit).items()}
    is_t = lambda inst: inst.operation.name in ("t", "tdg")  # noqa: E731
    t_count, gates = operation_counts_cost(circuit)
    return {
        "qubits": circuit.num_qubits,
        "gates": gates,
        "depth": int(circuit.depth()),
        "t_count": t_count,
        "t_depth": int(circuit.depth(filter_function=is_t)),
        "cx_count": counts.get("cx", 0),
        "clifford_count": sum(n for name, n in counts.items() if name in CLIFFORD_T_BASIS)
        - t_count,
        "op_counts": counts,
    }


def has_control_flow(circuit: QuantumCircuit) -> bool:
    return any(isinstance(inst.operation, ControlFlowOp) for inst in circuit.data)


def unitary_part(circuit: QuantumCircuit) -> QuantumCircuit:
    """The circuit with final measurements and barriers stripped, for simulation."""
    stripped = circuit.copy()
    stripped.remove_final_measurements(inplace=True)
    return PassManager([RemoveBarriers()]).run(stripped)


def unitary_fidelity(lhs: QuantumCircuit, rhs: QuantumCircuit) -> float:
    """Global-phase-insensitive fidelity of two circuits' unitaries.

    Dense: builds both 2^n x 2^n operators, so it is limited to ~12 qubits by
    memory.  Use statevector_fidelity above that.
    """

    def unitary(circuit: QuantumCircuit) -> np.ndarray:
        return np.asarray(Operator(unitary_part(circuit)).data, dtype=complex)

    left, right = unitary(lhs), unitary(rhs)
    dim = left.shape[0]
    return float(abs(np.trace(left.conj().T @ right)) / dim)


def statevector_fidelity(lhs: QuantumCircuit, rhs: QuantumCircuit, seed: int = 0) -> float:
    """|<lhs psi | rhs psi>| for one Haar-random |psi>.

    Costs 2^n amplitudes rather than 4^n, which is what makes verification
    possible past ~12 qubits.  It samples the unitaries rather than comparing
    them everywhere, but a Haar-random state misses a systematic discrepancy
    with negligible probability: for U^dag V = exp(i*theta) I + E the shortfall
    in fidelity is O(||E||), and only a measure-zero set of states hides it.
    """
    left, right = unitary_part(lhs), unitary_part(rhs)
    state = random_statevector(2**left.num_qubits, seed=seed)
    return float(abs(np.vdot(state.evolve(left).data, state.evolve(right).data)))


def non_basis_ops(circuit: QuantumCircuit) -> dict[str, int]:
    """Operations in the output that are not Clifford+T or a passthrough op.

    Nothing else validates the basis: a single-qubit gate whose to_matrix() fails
    is re-emitted verbatim by rewrite_single_qubit_runs, and a `u` or `unitary`
    left in the output would contribute nothing to the T count while still being
    a non-Clifford gate.
    """
    return {
        name: n
        for name, n in operation_counts(circuit).items()
        if name not in CLIFFORD_T_BASIS and name not in PASSTHROUGH_OPS
    }


def build_bqskit_workflow(
    synthesis_epsilon: float,
    decompose_rz: bool,
    seed: Optional[int],
) -> list[BasePass]:
    """Build this script's own Clifford+T circuit workflow for bqskit.

    Mirrors bqskit's own build_circuit_workflow (bqskit.ft.cliffordt.
    defaultworkflow), minus its multi-qudit retargeting stage
    (build_multi_qudit_retarget_workflow, from core bqskit). That stage is
    gated on NotPredicate(WidthPredicate(2)), which is true for any circuit
    with 2 or more qubits -- not just ones containing gates outside the
    target model's native gate set -- so it unconditionally runs
    AutoRebase2QuditGatePass over every <=3-qubit block, numerically
    re-synthesising it even when the block is already expressed in native
    gates. That discards exact Clifford+T structure (e.g. the H/T/Tdg/CX
    from a Toffoli/CSWAP decomposition) in favour of generic-angle
    rotations that each then need their own gridsynth call: measured 12x
    more T gates on data/qasmbench/knn_n25.qasm, a CSWAP-heavy circuit.
    Skipping it is safe here because unroll_to_u_cx already guarantees the
    circuit handed to bqskit contains only {u, cx} -- no gate outside
    CliffordTModel's native set for retargeting to act on.
    """
    passes: list[BasePass] = [SetRandomSeedPass(seed)] if seed is not None else []
    zxzxz = ForEachBlockPass([ZXZXZDecomposition()], collection_filter=single_qudit_filter)
    passes += [
        GroupSingleQuditGatePass(),
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        QuickPartitioner(2),
        ForEachBlockPass([ScanningGateRemovalPass()]),
        UnfoldPass(),
        GroupSingleQuditGatePass(),
        zxzxz,
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        UnfoldPass(),
    ]
    if decompose_rz:
        # Matches decompose_rz_tracked's own precision formula below (ceil, not
        # bqskit's int(...)+2 padding) so the two workflows' T counts stay
        # comparable regardless of which one a run ends up using.
        precision = math.ceil(math.log10(1 / synthesis_epsilon))
        passes += [
            IsolateRZGatePass(),
            ForEachBlockPass([GridSynthPass(precision=precision)]),
            UnfoldPass(),
        ]
    passes += [LogErrorPass()]
    return passes


def decompose_rz_tracked(
    circuit: Circuit, synthesis_epsilon: float = BQSKIT_EPSILON_DEFAULT
) -> tuple[Circuit, float]:
    """Take a {Clifford, RZ} circuit to Clifford+T, tracking an error bound.

    Feed this the output of a workflow built with ``decompose_rz=False``, which
    stops with the rotation angles still exact -- this cannot be applied to a
    finished Clifford+T circuit, where the gridsynth words have already been
    expanded into gates and recovering the intended angle would mean
    approximating an approximation.

    The pass list mirrors bqskit's own ordering in build_cliffordt_workflow --
    group single-qubit runs, decompose, replace exact Cliffords, round
    near-discrete z-rotations, then gridsynth what is left -- using bqskit's
    own stock ZXZXZDecomposition and GridSynthPass, inlined here (rather than
    calling bqskit's own rz_decomposition_passes() helper, which hardcodes the
    same pass) so calculate_error_bound=True can be set on both below, which
    bqskit's own decompose_rz=True path (used when --bqskit-inline-decompose-rz
    is passed) does not do.

    bqskit's stock ZXZXZDecomposition has a gauge bug: for a diagonal target,
    the middle rotation is Clifford, so how the total rotation splits between
    the two outer RZ/U1 gates is a free gauge choice, but the stock
    implementation always splits it evenly, generating two generic rotations
    where one would do -- doubling the gridsynth cost of every diagonal
    single-qubit rotation. The fix lives in bqskit itself (this repo's
    bqskit/ clone, branch ZXZXZ-fix, submitted upstream as a PR to
    BQSKit/bqskit), not patched locally here -- this function uses whichever
    ZXZXZDecomposition is active. ``pip install -e ./bqskit`` activates the
    fix ahead of an upstream release; without it (stock bqskit from PyPI),
    diagonal-heavy circuits cost ~25% more T gates via this backend.

    Returns (circuit, error_bound). error_bound is bqskit's own
    ``calculate_error_bound`` mechanism (bqskit/compiler/basepass.py's
    ``_sub_do_work``, which every ``ForEachBlockPass`` below with that flag set
    invokes per block: it measures the *exact* distance between each 1-qubit
    block's unitary before and after its sub-workflow runs -- cheap, since
    these are single-qubit blocks, not the whole circuit -- and composes the
    per-block sums multiplicatively across passes via
    ``PassData.update_error_mul``, the same fidelity-complement composition
    qiskit's own subadditive sum approximates). This is analogous to, but
    narrower in scope than, the qiskit backend's ``error_bound``: it covers
    ZXZXZDecomposition (contributes ~0, since the gauge-collapse rewrite is
    exact by construction) and GridSynthPass (the real source of error here),
    but not RoundToDiscreteZPass's own rounding (which isn't run inside a
    ForEachBlockPass, so isn't measured by this mechanism, and could in
    principle spend up to synthesis_epsilon per rounded rotation without being
    counted) or anything from bqskit's own earlier 2-qubit block instantiation
    in compile_bqskit's first bqskit_compile() call. Only available when
    use_custom_rz_decomposition=True -- bqskit's own decompose_rz=True path
    (used when --bqskit-inline-decompose-rz is passed) doesn't wrap its
    ForEachBlockPass calls with calculate_error_bound, so there is no
    equivalent bound to read there.
    """
    # bqskit's own build_cliffordt_workflow computes this as
    # int(log10(1/synthesis_epsilon)) + 2, padding to 100x tighter than
    # synthesis_epsilon. ceil (not int, which truncates) here instead only
    # rounds up enough to guarantee the digit-count still meets
    # synthesis_epsilon for non-power-of-10 values, without that overshoot.
    precision = math.ceil(math.log10(1 / synthesis_epsilon))
    passes = [
        GroupSingleQuditGatePass(),
        ForEachBlockPass(
            [ZXZXZDecomposition()],
            collection_filter=single_qudit_filter,
            calculate_error_bound=True,
        ),
        clifford_replace(),
        UnfoldPass(),
        RoundToDiscreteZPass(synthesis_epsilon),
        UnfoldPass(),
        IsolateRZGatePass(),
        ForEachBlockPass([GridSynthPass(precision=precision)], calculate_error_bound=True),
        UnfoldPass(),
    ]
    with Compiler() as compiler:
        out, data = compiler.compile(circuit, passes, request_data=True)
    return out, data.error


def _dedupe_creg_lines(path: Path) -> None:
    """bqskit emits one creg declaration per MeasurementPlaceholder, producing
    duplicate lines that cause QASM parsers to reject the file.  Deduplicate
    them while preserving the first occurrence and the original line order.
    """
    text = path.read_text()
    seen: set[str] = set()
    deduped: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("creg "):
            if stripped in seen:
                continue
            seen.add(stripped)
        deduped.append(line)
    path.write_text("".join(deduped))


def compile_bqskit(
    unrolled: QuantumCircuit,
    epsilon: float,
    seed: int,
    use_custom_rz_decomposition: bool,
) -> tuple[QuantumCircuit, Optional[float]]:
    """Compile an already-{u,cx}-unrolled circuit via bqskit, returning a qiskit
    QuantumCircuit (round-tripped through qiskit's own loader, so it can share
    verification/reporting/writing with the qiskit backend) and an error bound.

    The error bound is bqskit's own ``calculate_error_bound`` mechanism, read
    from ``decompose_rz_tracked`` -- see its docstring for exactly what it
    covers. Only available when ``use_custom_rz_decomposition`` is True;
    ``None`` otherwise, since bqskit's own ``decompose_rz=True`` path (used
    when it is False) has no equivalent tracking.
    """
    error_bound: Optional[float] = None
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.qasm"
        qasm2.dump(unrolled, in_path)
        bq_circuit = Circuit.from_file(str(in_path))

        machine = CliffordTModel(bq_circuit.num_qudits)
        # CliffordTModel registers a default workflow for every optimization
        # level in its constructor, so registering ours always displaces one
        # and bqskit warns about it.  Displacing it is the whole point -- it
        # is also the only way to seed, since compile() returns the registered
        # workflow before it ever looks at its own seed argument -- and the
        # warning's advice about namespace packages does not apply, so drop
        # just that message and leave every other warning visible.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Overwritting workflow")
            register_workflow(
                machine,
                build_bqskit_workflow(
                    synthesis_epsilon=epsilon,
                    decompose_rz=not use_custom_rz_decomposition,
                    seed=seed,
                ),
                BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL,
                "circuit",
            )
        bq_circuit = bqskit_compile(
            bq_circuit,
            model=machine,
            optimization_level=BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL,
            seed=seed,
        )
        if use_custom_rz_decomposition:
            bq_circuit, error_bound = decompose_rz_tracked(bq_circuit, epsilon)

        # Flatten any CircuitGate wrappers bqskit may have left around
        # sub-circuits (e.g. U3Gate wrapped in a CircuitGate).  Without this,
        # the transpile binary crashes because it has no rule for CircuitGate.
        bq_circuit.unfold_all()

        # Remove IdentityGate operations: they are semantic no-ops but bqskit
        # serialises them as a custom "identity1" gate using U(0,0,0).  When the
        # QASM is reloaded, that custom gate definition is parsed back as a
        # CircuitGate(U3Gate), which causes the transpile binary to crash.
        identity = IdentityGate(1)
        if identity in bq_circuit.gate_set:
            bq_circuit.remove_all(identity)

        for g in bq_circuit.gate_set:
            if (
                not isinstance(g, IdentityGate)
                and not isinstance(g, MeasurementPlaceholder)
                and not isinstance(g, BarrierPlaceholder)
                and g not in clifford_t_gates
            ):
                print(f"Warning: gate {g} is not Clifford+T", file=sys.stderr)

        out_path = Path(tmp) / "out.qasm"
        bq_circuit.save(str(out_path))
        _dedupe_creg_lines(out_path)
        return load_circuit(out_path), error_bound  # qiskit's own loader -- now a QuantumCircuit


def compile_dispatch(
    unrolled: QuantumCircuit,
    args: argparse.Namespace,
    synth: Optional[CliffordTSynthesizer] = None,
) -> tuple[QuantumCircuit, list[str], Optional[float], Optional[int]]:
    """Compile an already-unrolled circuit via whichever backend args.backend selects.

    Returns (compiled, extra report lines, error_bound, n_rewrites). `synth`, if
    given, is reused (and its counters read afterward) rather than creating a
    fresh one -- this lets main()'s per-file loop keep sharing one
    CliffordTSynthesizer, and its gridsynth cache, across a batch of input
    files. Random-window sampling (windowed_fidelity) instead passes no synth,
    since a window is small enough that losing cache reuse across windows
    doesn't matter.
    """
    if args.backend == "qiskit":
        synth = synth or CliffordTSynthesizer(epsilon=args.epsilon, tol=args.tol)
        compiled = compile_qiskit(unrolled, synth, optimize=not args.no_optimize)
        extra = [
            f"1q runs: {synth.n_clifford} Clifford, {synth.n_exact} exact,"
            f" {synth.n_approx} approximated, {synth.n_merged} shortened"
            f" (worst rewrite {synth.max_error:.2e}, total {synth.error_bound:.2e})"
        ]
        return compiled, extra, synth.error_bound, synth.n_approx + synth.n_merged
    compiled, error_bound = compile_bqskit(
        unrolled,
        epsilon=args.epsilon,
        seed=args.seed,
        use_custom_rz_decomposition=not args.bqskit_inline_decompose_rz,
    )
    return compiled, [], error_bound, None


def _random_window(
    source: QuantumCircuit, max_qubits: int, max_ops: float, rng: random.Random
) -> Optional[QuantumCircuit]:
    """Greedily grow a random contiguous window of source's instructions.

    Stops (not including the next instruction) as soon as either the
    touched-qubit count or the projected gates * 2^qubits cost would exceed the
    given caps, or the next instruction has classical bits, is a
    measure/reset, or is control flow -- a hard stop, since a window must be a
    plain unitary circuit to be independently recompiled and fidelity-checked.
    `barrier` is skipped over rather than stopped on: it's a no-op here and
    would otherwise fragment windows for no reason.

    Returns None if the window would be empty (e.g. a random start landing
    exactly on a hard-stop instruction).
    """
    n = len(source.data)
    if n == 0:
        return None
    start = rng.randrange(n)
    touched: dict[Qubit, int] = {}
    end = start
    window_gates = 0
    for i in range(start, n):
        inst = source.data[i]
        if inst.operation.name == "barrier":
            continue
        if inst.clbits or isinstance(inst.operation, ControlFlowOp) or inst.operation.name in (
            "measure",
            "reset",
            "delay",
        ):
            break
        new_qubits = [q for q in inst.qubits if q not in touched]
        prospective_qubits = len(touched) + len(new_qubits)
        prospective_gates = window_gates + 1
        if prospective_qubits > max_qubits or prospective_gates * 2**prospective_qubits > max_ops:
            break
        for q in new_qubits:
            touched[q] = len(touched)
        window_gates = prospective_gates
        end = i + 1
    if end == start or not touched:
        return None
    window = QuantumCircuit(len(touched))
    for i in range(start, end):
        inst = source.data[i]
        if inst.operation.name == "barrier":
            continue
        window.append(inst.operation, [touched[q] for q in inst.qubits], [])
    return window


def windowed_fidelity(
    source: QuantumCircuit, args: argparse.Namespace
) -> tuple[Optional[float], list[str]]:
    """Automatic fallback verification for circuits too large (or containing
    classical control flow) to verify directly.

    Samples WINDOW_VERIFY_COUNT random contiguous windows of source's
    instructions (see _random_window), each independently compiled through the
    same backend and args as the real run (unroll_to_u_cx then
    compile_dispatch, exactly mirroring the real pipeline so this validates
    preopt too, not just resynthesis) and fidelity-checked against its own
    source window. Bounded, constant cost regardless of the real circuit's
    size -- unlike the direct checks above, this is always tractable.

    _random_window's own budget check only bounds the *source* window's gate
    count, which isn't the whole story: Clifford+T compilation can blow gate
    count up by two orders of magnitude (a 59-gate window was measured
    compiling to 6944 gates), and the fidelity check's real cost depends on
    the *compiled* circuit's size, not the source's. So the ops budget is
    checked again here, after compiling, using both circuits' real gate counts
    -- exactly the same check the direct statevector path above already makes.
    A window that turns out too expensive only after compiling is skipped (a
    cheap loss -- compiling is fast; the fidelity check is what's expensive)
    rather than run anyway.

    A spot check, not a proof: it only catches what shows up in the sampled
    windows, and how much of the circuit a window can cover depends on the
    circuit's actual locality -- a Trotterized local-Hamiltonian circuit will
    yield large, useful windows; something built from wide/all-to-all
    entanglers (e.g. QFT) may only yield small ones. That is inherent to the
    technique, not a bug, and is exactly why several independent samples are
    taken rather than one.
    """
    rng = random.Random(args.verify_seed)
    fidelities: list[float] = []
    notes: list[str] = []
    for _ in range(WINDOW_VERIFY_MAX_ATTEMPTS):
        if len(fidelities) >= WINDOW_VERIFY_COUNT:
            break
        window = _random_window(source, WINDOW_VERIFY_MAX_QUBITS, WINDOW_VERIFY_MAX_OPS, rng)
        if window is None:
            continue
        compiled_window, _, _, _ = compile_dispatch(unroll_to_u_cx(window), args)
        qubits = compiled_window.num_qubits
        cost = (operation_counts_cost(compiled_window)[1] + operation_counts_cost(window)[1]) * (
            2**qubits
        )
        if qubits > DENSE_VERIFY_MAX_QUBITS and cost > WINDOW_VERIFY_MAX_OPS:
            continue  # compiled blowup made this window too expensive after all; try another
        fid = (
            unitary_fidelity(window, compiled_window)
            if qubits <= DENSE_VERIFY_MAX_QUBITS
            else statevector_fidelity(window, compiled_window, args.verify_seed)
        )
        fidelities.append(fid)
        notes.append(
            f"  window {len(fidelities)}: {window.num_qubits} qubits,"
            f" {len(window.data)} gates, fidelity {fid:.12f}"
        )
    if not fidelities:
        return None, ["no numeric check: could not find any sample window small enough to verify"]
    worst = min(fidelities)
    header = (
        f"random-window fidelity (worst of {len(fidelities)} samples,"
        f" up to {WINDOW_VERIFY_MAX_QUBITS} qubits each): {worst:.12f}"
    )
    return worst, [header] + notes


def verify_fidelity(
    source: QuantumCircuit,
    compiled: QuantumCircuit,
    args: argparse.Namespace,
) -> tuple[dict, list[str]]:
    """Numeric fidelity check, as thoroughly as the circuit's size allows.

    The basis check and error bound are handled unconditionally by main(), not
    here -- they're cheap enough to always run regardless of --verify, which
    gates only this, the potentially-expensive numeric comparison.

    Cascades: dense unitary comparison (exact, tiny circuits) -> a single
    random statevector (medium circuits) -> automatic random-window sampling
    (any size, or classical control flow, which the first two skip -- see
    windowed_fidelity). Unlike the first two, windowed sampling is always
    tractable, so this never dead-ends the way the old four-flag version could.
    """
    entry: dict = {"fidelity": None, "state_fidelity": None, "fidelity_method": None}
    notes: list[str] = []
    control_flow = has_control_flow(compiled) or has_control_flow(source)
    qubits = compiled.num_qubits
    gates = operation_counts_cost(compiled)[1] + operation_counts_cost(source)[1]
    try:
        if not control_flow and qubits <= DENSE_VERIFY_MAX_QUBITS and gates <= DENSE_VERIFY_MAX_GATES:
            entry["fidelity"] = unitary_fidelity(source, compiled)
            entry["fidelity_method"] = "dense"
            notes.append(f"unitary fidelity vs input: {entry['fidelity']:.12f}")
        elif (
            not control_flow
            and qubits <= STATEVECTOR_VERIFY_MAX_QUBITS
            and gates * 2**qubits <= STATEVECTOR_VERIFY_MAX_OPS
        ):
            entry["state_fidelity"] = statevector_fidelity(source, compiled, args.verify_seed)
            entry["fidelity_method"] = "statevector"
            notes.append(
                f"statevector fidelity vs input: {entry['state_fidelity']:.12f}"
                f" (1 random state, seed {args.verify_seed})"
            )
        else:
            if control_flow:
                notes.append(
                    "circuit has classical control flow: falling back to random-window sampling"
                )
            worst, window_notes = windowed_fidelity(source, args)
            entry["state_fidelity"] = worst
            entry["fidelity_method"] = "windowed" if worst is not None else None
            notes.extend(window_notes)
    except Exception as error:  # mid-circuit measurement, unsupported op, memory
        notes.append(f"numeric check failed to run ({type(error).__name__}: {error})")
    return entry, notes


def write_circuit(circuit: QuantumCircuit, path: Path) -> str:
    """Write as OpenQASM 2 if possible, else OpenQASM 3.  Returns the version."""
    try:
        text, version = qasm2.dumps(circuit), "2.0"
    except qasm2.QASM2ExportError:
        text, version = qasm3.dumps(circuit), "3.0"
    path.write_text(text + "\n")
    return version


def output_path(source: Path, target: Optional[Path], many: bool) -> Path:
    default_name = source.stem + ".cliffordt.qasm"
    if target is None:
        return source.parent / default_name
    if many or target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def print_report(log, before: dict, after: dict, extra: list[str]) -> None:
    """Print a report line identical in shape for both backends.

    `extra` carries the qiskit-only "1q runs: N Clifford, ..." diagnostic
    (from CliffordTSynthesizer's counters); it reflects bookkeeping internal to
    qiskit's own resynthesis algorithm that bqskit's compiler passes don't
    expose, so it is empty for the bqskit backend.
    """
    log(
        f"  {before['qubits']} qubits, {before['gates']} gates -> {after['gates']} gates "
        f"(T={after['t_count']}, cx={after['cx_count']}, depth={after['depth']}, "
        f"T-depth={after['t_depth']})"
    )
    for line in extra:
        log(f"  {line}")


@contextlib.contextmanager
def timed(label: str, timings: dict) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        timings[label] = time.time() - start


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a QASM circuit to the Clifford+T gate set, via qiskit or bqskit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="input .qasm file(s)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output file (single input) or directory (multiple inputs); "
        "default is <input>.cliffordt.qasm next to the input",
    )
    parser.add_argument(
        "--backend",
        choices=("qiskit", "bqskit"),
        default="bqskit",
        help="compilation backend: bqskit (fewer T gates on every benchmark "
        "measured so far, since the multi-qudit retargeting and "
        "ZXZXZDecomposition gauge-collapse fixes -- see build_bqskit_workflow's "
        "and decompose_rz_tracked's docstrings -- but rejects circuits with "
        "classical control flow) or qiskit (kept for comparison and as an "
        "independent implementation; the only option for circuits bqskit "
        "rejects)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=None,
        help="gridsynth target error per rotation; default depends on --backend "
        f"({QISKIT_EPSILON_DEFAULT:g} for qiskit, {BQSKIT_EPSILON_DEFAULT:g} for "
        "bqskit -- see module docstring for why these differ rather than being "
        "one shared value)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="qiskit backend only: skip the post-synthesis clean-up (inverse "
        "cancellation and exact re-synthesis of collapsible runs)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        help="qiskit backend only: how much error an exact rewrite may introduce: "
        "the tolerance for treating an angle as a multiple of pi/4 or a unitary "
        "as a Clifford. Defaults to --epsilon, so that no rotation comes out "
        "free unless it is within the requested accuracy of a Clifford",
    )
    parser.add_argument(
        "--bqskit-inline-decompose-rz",
        action="store_true",
        help="bqskit backend only: use bqskit's own inline decompose_rz=True "
        "workflow instead of this script's decompose_rz_tracked "
        "post-processing pass (see its docstring). The latter is on by "
        "default because it additionally tracks an error bound that the "
        "inline workflow does not.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="bqskit backend only: seed for bqskit's numerical instantiation. "
        "bqskit is nondeterministic without one: repeated runs of the same "
        "input have been seen to differ by ~40%% in T count, which makes gate "
        "counts unreproducible and unsafe to compare. Pass a different integer "
        "to sample another compilation; there is no unseeded mode, since an "
        "unrecorded seed is never preferable to a recorded one.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the compiled circuit's fidelity against the input: an exact "
        "comparison if small enough, else a single random statevector, else "
        "automatic random-window sampling -- always tractable, at a bounded "
        "cost, regardless of circuit size. The basis check and error bound "
        "are always reported, regardless of this flag",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=0,
        help="seed for the random state used by the statevector comparison, and "
        "for window selection when falling back to random-window sampling",
    )
    parser.add_argument("--stats", type=Path, help="write per-circuit statistics as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")

    args = parser.parse_args(argv)

    if args.epsilon is None:
        args.epsilon = (
            QISKIT_EPSILON_DEFAULT if args.backend == "qiskit" else BQSKIT_EPSILON_DEFAULT
        )

    # Warn (not error) if a backend-inappropriate flag was explicitly set to a
    # non-default value, so the two backends can share one parser without
    # silently ignoring a flag the user thought they were setting.
    backend_only_flags = {
        "qiskit": {"bqskit_inline_decompose_rz": "--bqskit-inline-decompose-rz", "seed": "--seed"},
        "bqskit": {"no_optimize": "--no-optimize", "tol": "--tol"},
    }
    for dest, flag in backend_only_flags[args.backend].items():
        if getattr(args, dest) != parser.get_default(dest):
            print(
                f"warning: {flag} has no effect with --backend {args.backend}",
                file=sys.stderr,
            )

    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = (lambda *a: None) if args.quiet else lambda *a: print(*a, flush=True)

    synth: Optional[CliffordTSynthesizer] = None
    if args.backend == "qiskit":
        synth = CliffordTSynthesizer(epsilon=args.epsilon, tol=args.tol)
        log(f"backend: qiskit (epsilon={args.epsilon:g})")
    else:
        log(
            f"backend: bqskit (epsilon={args.epsilon:g}, seed={args.seed}, "
            f"rz decomposition: {'bqskit inline' if args.bqskit_inline_decompose_rz else 'tracked'})"
        )

    all_stats = []
    failures = 0
    for source in args.inputs:
        log(f"=== {source}")
        timings: dict[str, float] = {}
        fidelity_result: Optional[dict] = None
        if synth is not None:
            synth.reset_counters()  # counters are per circuit
        try:
            with timed("total", timings):
                with timed("load", timings):
                    circuit = load_circuit(source)
                before = circuit_stats(circuit)

                with timed("preopt", timings):
                    unrolled = unroll_to_u_cx(circuit)

                with timed("compile", timings):
                    if args.backend == "bqskit" and has_control_flow(circuit):
                        raise RuntimeError(
                            "the bqskit backend does not support classical control flow"
                        )
                    compiled, extra, error_bound, n_rewrites = compile_dispatch(
                        unrolled, args, synth
                    )

                after = circuit_stats(compiled)
                print_report(log, before, after, extra)

                # Basis check + error bound: always, no flag needed -- both are
                # cheap (no simulation), and a broken basis is worth failing the
                # run over regardless of whether numeric verification was asked for.
                non_basis = non_basis_ops(compiled)
                if error_bound is not None:
                    scope = f" over {n_rewrites} rewrites" if n_rewrites is not None else ""
                    log(f"  error bound{scope}: {error_bound:.2e}")
                if non_basis:
                    log(f"  FAILED basis check, output is not Clifford+T: {non_basis}")
                else:
                    log(
                        f"  basis check passed ({', '.join(CLIFFORD_T_BASIS)}"
                        " + measure/barrier/reset)"
                    )

                timings["verify"] = 0.0
                if args.verify:
                    with timed("verify", timings):
                        fidelity_result, notes = verify_fidelity(circuit, compiled, args)
                    for note in notes:
                        log(f"  {note}")

                with timed("write", timings):
                    destination = output_path(source, args.output, len(args.inputs) > 1)
                    version = write_circuit(compiled, destination)
        except Exception as error:  # keep going over a batch of files
            print(f"ERROR {source}: {type(error).__name__}: {error}", file=sys.stderr)
            failures += 1
            continue

        for label in ("load", "preopt", "compile", "verify", "write"):
            log(f"  {label}: {timings[label]:.2f}s")
        log(f"  total: {timings['total']:.2f}s")

        entry = {
            "input": str(source),
            "output": str(destination),
            "qasm_version": version,
            "backend": args.backend,
            "epsilon": args.epsilon,
            "before": before,
            "after": after,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "runs_clifford": synth.n_clifford if synth else None,
            "runs_exact": synth.n_exact if synth else None,
            "runs_approximated": synth.n_approx if synth else None,
            "runs_shortened": synth.n_merged if synth else None,
            "tol": synth.tol if synth else None,
            "max_rewrite_error": synth.max_error if synth else None,
            "error_bound": error_bound,
            "non_basis_ops": non_basis,
        }
        if fidelity_result is not None:
            entry.update(fidelity_result)
        if non_basis:
            print(
                f"ERROR {source}: output is not Clifford+T: {non_basis}",
                file=sys.stderr,
            )
            failures += 1

        log(f"  wrote {destination} (OpenQASM {version})")
        all_stats.append(entry)

    if args.stats:
        args.stats.write_text(json.dumps(all_stats, indent=2) + "\n")
        log(f"wrote statistics to {args.stats}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
