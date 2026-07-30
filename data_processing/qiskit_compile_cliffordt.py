#!/usr/bin/env python3
"""
Compile an arbitrary OpenQASM circuit into the Clifford+T gate set using qiskit.

Output gate set: h, s, sdg, x, y, z, t, tdg, cx (plus measure/barrier/reset and
any classical control flow that was present in the input).

Pipeline
--------
1. Load the QASM file (OpenQASM 2 first, falling back to OpenQASM 3 -- which
   needs the optional `qiskit-qasm3-import` package).
2. Transpile to {u, cx} so that every multi-qubit gate (ccx, cswap, cry, rzz,
   ryy, ...) is broken down into cx plus single-qubit rotations.
3. Merge each maximal run of consecutive single-qubit gates on a wire into one
   2x2 matrix, then re-synthesise that matrix over Clifford+T:
     * Clifford            -> shortest word in {h, s, sdg, x, y, z} (BFS table).
     * exactly representable -> exact sequence.  A gate is exact iff its ZXZ
       Euler angles are all integer multiples of pi/4, since Rz(k*pi/4) is a
       T/S/Z word and Rx(theta) = H Rz(theta) H.
     * otherwise           -> each generic Rz in the ZXZ decomposition is
       approximated to --epsilon, memoised per angle.
   Neither "exact" path is taken on trust: the word it produces is measured
   against the target and rejected if it is off by more than --tol, which
   defaults to --epsilon.  The check matters because the Clifford lookup key
   rounds to 7 decimals -- deliberately, so that gates differing only by
   floating-point noise share a table entry -- which also makes the lookup
   match anything within ~5e-8 of a Clifford.  Unguarded, that discards small
   rotations for free: the pi/2^k tail of a wide QFT, for instance, where every
   rotation below ~1e-7 would cost zero T at an error hundreds of times
   --epsilon.  Rotations that really are within --epsilon of a Clifford still
   cost nothing, but that is now the synthesis backend's decision, made against
   the requested accuracy, and it shows up in the reported error.
4. Clean up: cancel adjacent inverse pairs (t.tdg, h.h, cx.cx, ...) and collapse
   blocks of gates that have a shorter exact form (t.t -> s, tdg.tdg.tdg.tdg ->
   z, any Clifford block -> its shortest word).  It takes ~30% off
   Solovay-Kitaev output and a few percent off gridsynth output, which is
   already close to optimal.  The block collapses are exact; the whole-run
   rewrite goes through the same guarded exact paths as step 3, so it can trade
   up to --tol of accuracy for a shorter run, and what it spends is added to
   the reported error bound.

Rotation synthesis backends (--synthesis)
-----------------------------------------
gridsynth (default): the Ross-Selinger algorithm, near T-optimal at
    T ~ 3*log2(1/epsilon) per rotation.  Uses qiskit's own Rust implementation
    (qiskit.synthesis.gridsynth_rz, qiskit >= 2.5, ~5 ms per distinct rotation),
    falling back to pygridsynth -- the implementation bqskit-ft calls, ~10x
    slower -- for the angles rsgridsynth 0.2.0 panics on at coarse epsilon.
    The default --epsilon 1e-10 is both qiskit's own default and a match for
    bqskit_compile_cliffordt.py, where bqskit's default synthesis_epsilon=1e-8
    becomes a gridsynth precision of int(log10(1/1e-8)) + 2 = 10 digits.  So
    this script and the bqskit one approximate to the same accuracy.
    Note that each generic 1q gate needs up to 3 Rz rotations (ZXZ Euler
    angles), so the error per gate is up to 3*epsilon; angles that are exact
    multiples of pi/4 are synthesised exactly and cost nothing.
sk: qiskit's Solovay-Kitaev pass.  Needs no extra dependency but is far worse on
    both axes -- at --recursion-degree 3 a rotation costs ~370 T for an error of
    only ~1e-2, and reaching 1e-10 is not practical (the sequence length grows
    ~5x per level).  Only useful as a fallback if pygridsynth is unavailable.

Both backends synthesise each distinct rotation once and reuse it, so cost scales
with the number of distinct angles rather than the number of gates.

The result is written next to the input as <name>.cliffordt.qasm (OpenQASM 2,
or OpenQASM 3 if the circuit uses control flow that OpenQASM 2 cannot express),
and per-circuit gate/T counts go to stdout and optionally to --stats JSON.

Verification (--verify)
-----------------------
Three checks, applied from cheapest to most expensive; every circuit gets as
many of them as its size allows, and the log says which ran.

basis + error bound: always, at any qubit count.  Every operation in the output
    must be in the Clifford+T basis (plus measure/barrier/reset/control flow),
    and the per-rewrite errors measured in steps 3 and 4 are summed into
    `error_bound`.  Each rewrite replaces one wire's run by a phase-aligned
    approximation of it, so by subadditivity of the spectral norm that sum is a
    genuine upper bound on ||U_compiled - U_unrolled||_2 -- taking qiskit's own
    unroll and inverse cancellation as exact.  This is the only check available
    for the large circuits, where nothing can be simulated.
statevector fidelity: up to --verify-statevector-qubits (default 24), subject to
    a work budget of gates * 2^qubits <= --verify-statevector-ops.  Evolves one
    Haar-random state (seeded by --verify-seed, so runs are reproducible)
    through both circuits and compares.  A single random state is a strong test:
    a systematic error survives it with negligible probability.  Cost is
    inherently gates * 2^qubits -- qiskit's Statevector manages roughly 1e8 of
    those per second, falling to ~5e7 by 22 qubits as the state stops fitting in
    cache -- so the default budget of 5e9 is about a minute of work, and a
    100k-gate circuit at 22 qubits would need hours.  Raise the budget if you
    want to spend them; the skip message estimates the wall time.  Memory is
    ~3 * 16 * 2^qubits bytes (~800 MB at 24 qubits), doubling per qubit.
unitary fidelity: up to --verify-max-qubits (default 10) and --verify-max-gates.
    The full dense 2^n x 2^n comparison, exhaustive but limited to ~12 qubits by
    memory.

Examples
--------
    ./qiskit_compile_cliffordt.py ../data/qasmbench/ising_n26.qasm
    ./qiskit_compile_cliffordt.py ../data/qasmbench/dnn_n8.qasm -o dnn_n8.ct.qasm \
        --epsilon 1e-6 --verify
    ./qiskit_compile_cliffordt.py ../data/qasmbench/*.qasm -o out_dir --stats stats.json
    ./qiskit_compile_cliffordt.py circuit.qasm --synthesis sk --recursion-degree 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit.circuit import ControlFlowOp, Gate, Qubit
from qiskit.quantum_info import Operator, random_statevector
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, RemoveBarriers, SolovayKitaev
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

try:  # qiskit >= 2.5: Ross-Selinger in Rust, the default rotation backend here
    from qiskit.synthesis import gridsynth_rz
except ImportError:
    gridsynth_rz = None

try:  # optional fallback for the angles rsgridsynth 0.2.0 panics on
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates
except ImportError:
    mpmath = None
    gridsynth_gates = None

CLIFFORD_T_BASIS = ("h", "s", "sdg", "x", "y", "z", "t", "tdg", "cx")
PI_4 = math.pi / 4

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


def canonical_key(matrix: np.ndarray, decimals: int = 7) -> tuple:
    """Hashable key for a 2x2 unitary, insensitive to global phase.

    The rounding is deliberately coarse so that gates that differ only by
    floating-point noise (e.g. an H that came back from the transpiler as a u
    gate) hit the same table entry.  It is far finer than the accuracy of any
    approximate synthesis, so sharing a Solovay-Kitaev sequence between two
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
    """Re-synthesise single-qubit unitaries over {h, s, sdg, x, y, z, t, tdg}."""

    def __init__(
        self,
        synthesis: str = "gridsynth",
        epsilon: float = 1e-10,
        recursion_degree: int = 3,
        depth: int = 12,
        approximations: Optional[str] = None,
        tol: Optional[float] = None,
    ) -> None:
        if synthesis not in ("gridsynth", "sk"):
            raise ValueError(f"unknown synthesis backend {synthesis!r}")
        self.synthesis = synthesis
        self.epsilon = epsilon
        # How much error an "exact" rewrite is allowed to introduce.  Defaulting
        # it to epsilon keeps the exact paths from being looser than the
        # approximate one: a rotation only comes out free if it really is within
        # the requested accuracy of a Clifford.
        self.tol = max(epsilon, EXACTNESS_FLOOR) if tol is None else tol
        self._decomposer = OneQubitEulerDecomposer(basis="ZXZ")
        self._clifford_words = build_clifford_words()
        self._gridsynth_cache: dict[float, tuple[str, ...]] = {}
        self._sk_cache: dict[tuple, tuple[QuantumCircuit, np.ndarray]] = {}
        self._sk_pass = None
        if synthesis == "gridsynth":
            if gridsynth_rz is None and gridsynth_gates is None:
                raise RuntimeError(
                    "--synthesis gridsynth needs qiskit >= 2.5 (which ships "
                    "qiskit.synthesis.gridsynth_rz) or the pygridsynth package; "
                    "otherwise use --synthesis sk"
                )
            if mpmath is not None:
                mpmath.mp.dps = max(mpmath.mp.dps, GRIDSYNTH_DPS)
        else:
            self._sk_pass = SolovayKitaev(
                recursion_degree=recursion_degree,
                basic_approximations=approximations,
                depth=depth,
            )
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
        return self._approximate(matrix, canonical_key(matrix))

    def shorten_run(self, matrix: np.ndarray, run: list[Gate]) -> Optional[QuantumCircuit]:
        """Shorten an already-compiled run of Clifford+T gates.

        Returns None if nothing can be improved.  Two rewrites are tried: the
        whole run at once (it may be Clifford, or a pi/4 rotation), and failing
        that a local collapse of sub-blocks -- a Solovay-Kitaev word is a long
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

    def _approximate(self, matrix: np.ndarray, key: tuple) -> QuantumCircuit:
        if self.synthesis == "gridsynth":
            return self._gridsynth(matrix)
        return self._solovay_kitaev(matrix, key)

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
                    "smaller --epsilon, install pygridsynth as a fallback, or "
                    "pass --synthesis sk."
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

    def _solovay_kitaev(self, matrix: np.ndarray, key: tuple) -> QuantumCircuit:
        if self._sk_pass is None:  # __init__ builds it whenever synthesis == "sk"
            raise RuntimeError("the Solovay-Kitaev pass is not configured")
        entry = self._sk_cache.get(key)
        if entry is None:
            source = QuantumCircuit(1)
            source.unitary(matrix, [0])
            approximation = self._sk_pass(transpile(source, basis_gates=["u"]))
            entry = (approximation, np.asarray(Operator(approximation).data, dtype=complex))
            self._sk_cache[key] = entry
        approximation, unitary = entry
        # The key ignores global phase, so re-align on every hit rather than
        # baking the first target's phase into the cached sequence.
        phase = global_phase_between(matrix, unitary)
        self._record(spectral_error(matrix, unitary, phase))
        aligned = approximation.copy()
        aligned.global_phase += phase
        return aligned


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
    compile_to_clifford_t would stop after a single round.
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


def compile_to_clifford_t(
    circuit: QuantumCircuit,
    synth: CliffordTSynthesizer,
    opt_level: int = 1,
    optimize: bool = True,
    max_rounds: int = 5,
) -> QuantumCircuit:
    """Full unroll -> re-synthesise -> clean-up pipeline."""
    unrolled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=opt_level)
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


def verify_compilation(
    source: QuantumCircuit,
    compiled: QuantumCircuit,
    synth: CliffordTSynthesizer,
    args: argparse.Namespace,
) -> tuple[dict, list[str]]:
    """Check the compiled circuit as thoroughly as its size allows.

    Returns the statistics to record and the lines to log.  The basis check and
    the error bound apply at any qubit count; the numeric checks fall back from
    dense unitaries to a single random statevector to nothing as the circuit
    grows.
    """
    entry: dict = {
        "non_basis_ops": non_basis_ops(compiled),
        "error_bound": synth.error_bound,
        "fidelity": None,
        "state_fidelity": None,
    }
    notes = [f"error bound over {synth.n_approx + synth.n_merged} rewrites: {synth.error_bound:.2e}"]
    if entry["non_basis_ops"]:
        notes.append(f"FAILED basis check, output is not Clifford+T: {entry['non_basis_ops']}")
    else:
        notes.append(f"basis check passed ({', '.join(CLIFFORD_T_BASIS)} + measure/barrier/reset)")

    qubits = compiled.num_qubits
    gates = operation_counts_cost(compiled)[1] + operation_counts_cost(source)[1]
    if has_control_flow(compiled) or has_control_flow(source):
        notes.append("no numeric check: circuit has classical control flow")
        return entry, notes
    try:
        if qubits <= args.verify_max_qubits and gates <= args.verify_max_gates:
            entry["fidelity"] = unitary_fidelity(source, compiled)
            notes.append(f"unitary fidelity vs input: {entry['fidelity']:.12f}")
        elif (
            qubits <= args.verify_statevector_qubits
            and gates * 2**qubits <= args.verify_statevector_ops
        ):
            entry["state_fidelity"] = statevector_fidelity(source, compiled, args.verify_seed)
            notes.append(
                f"statevector fidelity vs input: {entry['state_fidelity']:.12f}"
                f" (1 random state, seed {args.verify_seed})"
            )
        else:
            # Both limits can bind at once, and naming only one of them sends the
            # reader to raise a knob that will not actually let the check run.
            over = []
            if qubits > args.verify_statevector_qubits:
                over.append(
                    f"--verify-statevector-qubits {args.verify_statevector_qubits}"
                    f" (needs ~{3 * 16 * 2**qubits / 2**30:.0f} GB)"
                )
            if gates * 2**qubits > args.verify_statevector_ops:
                over.append(
                    f"--verify-statevector-ops {args.verify_statevector_ops:.1e}"
                    f" (needs {gates * 2**qubits:.1e},"
                    f" ~{gates * 2**qubits / 1e8 / 3600:.1f} h at ~1e8/s)"
                )
            notes.append(
                f"no numeric check: {qubits} qubits x {gates} gates is over "
                + " and ".join(over)
            )
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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a QASM circuit to the Clifford+T gate set with qiskit.",
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
        "--synthesis",
        choices=("gridsynth", "sk"),
        default="gridsynth",
        help="rotation synthesis backend: gridsynth (Ross-Selinger, near T-optimal, "
        "same algorithm and accuracy as bqskit_compile_cliffordt.py) or sk (qiskit's "
        "Solovay-Kitaev, no extra dependency but much longer and less accurate)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=1e-10,
        help="gridsynth target error per rotation; the default matches bqskit's "
        "default synthesis_epsilon=1e-8 as used by bqskit_compile_cliffordt.py",
    )
    parser.add_argument(
        "-r",
        "--recursion-degree",
        type=int,
        default=3,
        help="Solovay-Kitaev recursion degree (--synthesis sk only); higher is more "
        "accurate and much longer",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="depth of the Solovay-Kitaev basic-approximation search " "(--synthesis sk only)",
    )
    parser.add_argument(
        "--approximations",
        help="file of pre-generated Solovay-Kitaev basic approximations (.npy) "
        "(--synthesis sk only)",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=1,
        choices=(0, 1, 2, 3),
        help="qiskit optimization level for the initial unroll to {u, cx}",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="skip the post-synthesis clean-up (inverse cancellation and exact "
        "re-synthesis of collapsible runs)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        help="how much error an exact rewrite may introduce: the tolerance for "
        "treating an angle as a multiple of pi/4 or a unitary as a Clifford. "
        "Defaults to --epsilon, so that no rotation comes out free unless it is "
        "within the requested accuracy of a Clifford",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the compiled circuit against the input: basis and error bound "
        "at any size, plus a dense or statevector fidelity if it is small enough",
    )
    parser.add_argument(
        "--verify-max-qubits",
        type=int,
        default=10,
        help="use the dense 2^n x 2^n unitary comparison up to this many qubits",
    )
    parser.add_argument(
        "--verify-max-gates",
        type=int,
        default=20000,
        help="use the dense unitary comparison up to this many gates",
    )
    parser.add_argument(
        "--verify-statevector-qubits",
        type=int,
        default=24,
        help="use the random-statevector comparison up to this many qubits, above "
        "which only the basis check and error bound are reported; memory is "
        "~800 MB at 24 and doubles per qubit",
    )
    parser.add_argument(
        "--verify-statevector-ops",
        type=float,
        default=5e9,
        help="work budget for the random-statevector comparison, as gates * 2^qubits; "
        "roughly 1e8 per second, so the default is about a minute",
    )
    parser.add_argument(
        "--verify-seed",
        type=int,
        default=0,
        help="seed for the random state used by the statevector comparison",
    )
    parser.add_argument("--stats", type=Path, help="write per-circuit statistics as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    log = (lambda *a: None) if args.quiet else lambda *a: print(*a, flush=True)

    synth = CliffordTSynthesizer(
        synthesis=args.synthesis,
        epsilon=args.epsilon,
        recursion_degree=args.recursion_degree,
        depth=args.depth,
        approximations=args.approximations,
        tol=args.tol,
    )
    log(
        f"rotation synthesis: {args.synthesis}"
        + (
            f" (epsilon={args.epsilon:g})"
            if args.synthesis == "gridsynth"
            else f" (recursion degree {args.recursion_degree})"
        )
    )

    all_stats = []
    failures = 0
    for source in args.inputs:
        log(f"=== {source}")
        start = time.time()
        synth.reset_counters()  # counters are per circuit; the SK cache is shared
        try:
            circuit = load_circuit(source)
            before = circuit_stats(circuit)
            compiled = compile_to_clifford_t(
                circuit, synth, opt_level=args.opt_level, optimize=not args.no_optimize
            )
            after = circuit_stats(compiled)
            destination = output_path(source, args.output, len(args.inputs) > 1)
            version = write_circuit(compiled, destination)
        except Exception as error:  # keep going over a batch of files
            print(f"ERROR {source}: {type(error).__name__}: {error}", file=sys.stderr)
            failures += 1
            continue

        elapsed = time.time() - start
        entry = {
            "input": str(source),
            "output": str(destination),
            "qasm_version": version,
            "synthesis": args.synthesis,
            "epsilon": args.epsilon if args.synthesis == "gridsynth" else None,
            "recursion_degree": (args.recursion_degree if args.synthesis == "sk" else None),
            "seconds": round(elapsed, 3),
            "before": before,
            "after": after,
            "runs_clifford": synth.n_clifford,
            "runs_exact": synth.n_exact,
            "runs_approximated": synth.n_approx,
            "runs_shortened": synth.n_merged,
            "tol": synth.tol,
            "max_rewrite_error": synth.max_error,
            "error_bound": synth.error_bound,
        }

        log(
            f"  {before['qubits']} qubits, {before['gates']} gates"
            f" -> {after['gates']} gates"
            f" (T={after['t_count']}, cx={after['cx_count']},"
            f" depth={after['depth']}, T-depth={after['t_depth']})"
        )
        log(
            f"  1q runs: {synth.n_clifford} Clifford, {synth.n_exact} exact,"
            f" {synth.n_approx} approximated, {synth.n_merged} shortened"
            f" (worst rewrite {synth.max_error:.2e}, total {synth.error_bound:.2e})"
        )

        if args.verify:
            verification, notes = verify_compilation(circuit, compiled, synth, args)
            entry.update(verification)
            for note in notes:
                log(f"  {note}")
            if verification["non_basis_ops"]:
                print(
                    f"ERROR {source}: output is not Clifford+T: "
                    f"{verification['non_basis_ops']}",
                    file=sys.stderr,
                )
                failures += 1

        log(f"  wrote {destination} (OpenQASM {version}) in {elapsed:.1f}s")
        all_stats.append(entry)

    if args.stats:
        args.stats.write_text(json.dumps(all_stats, indent=2) + "\n")
        log(f"wrote statistics to {args.stats}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
