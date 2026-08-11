#!/usr/bin/env python3
"""
Compile an arbitrary OpenQASM circuit into the Clifford+T gate set, via qiskit
or bqskit (--backend). Deliberately naive: both backends are the simplest
reasonable implementation, not a competitive compiler -- no exact-rewrite
preprocessing, no post-synthesis clean-up, no hand-tuned pass lists. This
script exists to be an unoptimized baseline that the Rust `compile_cliffordt`
implementation (src/compile_cliffordt.rs) can be measured against, not to
produce the smallest possible T-count itself.

Pipeline: load the QASM file, unroll to {u, cx} (structurally necessary --
see unroll_to_u_cx -- not an optimization), then resynthesise each backend's
own simplest way: qiskit re-synthesises each single-qubit run once via
gridsynth (see CliffordTSynthesizer); bqskit hands the circuit to its own
stock `compile()` + `CliffordTModel`, the same call `src/compile_circuit.py`
makes (see compile_bqskit). Both use the Ross-Selinger algorithm (gridsynth)
for generic rotations, accurate to --epsilon. The result is written next to
the input as <name>.cliffordt.qasm; a basis check and (qiskit only -- bqskit's
stock workflow doesn't expose one) error bound always run, and --verify adds
a numeric fidelity check (exact, sampled, or windowed, whichever the circuit's
size allows). See the individual functions' docstrings for the details behind
each of these.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Does not actually suppress rsgridsynth's panic backtrace (confirmed: it
# bypasses RUST_BACKTRACE) -- kept in case some other dependency honors it,
# and setdefault won't override a value the user set deliberately.
os.environ.setdefault("RUST_BACKTRACE", "0")

import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, transpile
from qiskit.circuit import ControlFlowOp, Gate, Qubit
from qiskit.quantum_info import Operator, random_statevector
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from bqskit import Circuit
from bqskit.compiler.compile import compile as bqskit_compile

# bqskit.ft resolves at runtime via pkgutil.extend_path (see bqskit-ft's
# __init__.py) -- a dynamic sys.path merge pyright cannot evaluate statically,
# hence the ignores below.
from bqskit.ft.cliffordt.cliffordtgates import (  # pyright: ignore[reportMissingImports]
    clifford_t_gates,
)
from bqskit.ft.cliffordt.cliffordtmodel import (  # pyright: ignore[reportMissingImports]
    CliffordTModel,
)
from bqskit.ir.gates import BarrierPlaceholder, IdentityGate, MeasurementPlaceholder

try:  # qiskit >= 2.5 ships Ross-Selinger in Rust
    from qiskit.synthesis import gridsynth_rz
except ImportError:
    gridsynth_rz = None

try:  # optional fallback for the angles rsgridsynth 0.2.0 panics on
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates
except ImportError:
    mpmath = None
    gridsynth_gates = None

# sx/sxdg included because the bqskit backend emits them natively (its own
# Clifford+T gate set treats sx as a generator) and src/transpile.rs's
# Gate1Q::SX/SXdg understands them -- the qiskit backend never emits either.
CLIFFORD_T_BASIS = ("h", "s", "sdg", "sx", "sxdg", "x", "y", "z", "t", "tdg", "cx")
PI_4 = math.pi / 4

# qiskit's transpile() default (level 2) produces more, smaller single-qubit
# runs than level 1 for this script's resynthesis purposes -- pinned rather
# than left at qiskit's default. Shared by all three backends' preopt step
# (unroll_to_u_cx).
UNROLL_OPTIMIZATION_LEVEL = 1

# --epsilon's default, shared by all three backends -- see the module
# docstring's "Rotation synthesis" section for the measurements behind this
# number.
EPSILON_DEFAULT = 1e-8

# Minimum time between resynthesis progress updates (see _with_progress) --
# a UI refresh rate, not a behavioral knob, so not exposed as a CLI option.
PROGRESS_INTERVAL_SECONDS = 1.0

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

# The one rsgridsynth panic message known safe to swallow (falls back to
# pygridsynth) -- see _capture_stderr_fd's use in _synthesize_rz. Matched
# against str(exception), not raw stderr: pyo3 surfaces the panic payload as
# the exception's message.
KNOWN_GRIDSYNTH_PANIC = "Invalid coefficients for inverse sqrt2 multiplication"

# Non-gate operations that are allowed to survive into the output.
PASSTHROUGH_OPS = frozenset({"measure", "barrier", "reset", "delay"})

# Thresholds for the --verify fidelity cascade (dense unitary -> single random
# statevector -> automatic random-window sampling). Not exposed as CLI flags,
# so the cascade always has somewhere to fall back to instead of dead-ending.
DENSE_VERIFY_MAX_QUBITS = 10
DENSE_VERIFY_MAX_GATES = 20_000
STATEVECTOR_VERIFY_MAX_QUBITS = 24
STATEVECTOR_VERIFY_MAX_OPS = 5e9

# Seed for --verify's random state(s): the statevector check's single
# Haar-random state, and random-window sampling's window selection. Fixed
# rather than exposed as a CLI flag -- reproducibility is what matters here
# (a systematic error survives any seed with negligible probability, see
# statevector_fidelity), not exploring different random draws.
VERIFY_SEED = 0

# Random-window sampling: fallback for circuits too large for either check
# above (or with classical control flow). Derived from the direct-check
# constants above so its total cost stays bounded by roughly the same budget
# as a single direct statevector check, regardless of circuit size.
WINDOW_VERIFY_COUNT = 5
WINDOW_VERIFY_MAX_QUBITS = STATEVECTOR_VERIFY_MAX_QUBITS
WINDOW_VERIFY_MAX_OPS = STATEVECTOR_VERIFY_MAX_OPS / WINDOW_VERIFY_COUNT
WINDOW_VERIFY_MAX_ATTEMPTS = WINDOW_VERIFY_COUNT * 5


def canonical_key(matrix: np.ndarray, decimals: int = 7) -> tuple:
    """Hashable key for a single-qubit unitary, insensitive to global phase.

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

    The phase is the one that minimises the error, i.e. the one
    circuit_from_word bakes into the circuit it builds, so the error returned
    is the error of the circuit that will actually be emitted.
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


def circuit_from_word(
    word: tuple[str, ...],
    target: np.ndarray,
    phase: Optional[float] = None,
) -> QuantumCircuit:
    """Single-qubit circuit applying `word` in order, phase-aligned against `target`."""
    circuit = QuantumCircuit(1)
    for name in word:
        getattr(circuit, name)(0)
    circuit.global_phase = (
        global_phase_between(target, word_matrix(word)) if phase is None else phase
    )
    return circuit


@contextlib.contextmanager
def _capture_stderr_fd():
    """Redirect the OS-level stderr file descriptor to a temp file for
    the duration of the block, then always restore it -- including on an
    unexpected exception, via `finally`, so a bug here can't leave stderr
    silently broken for the rest of the process.

    Needed because Rust panics (see CliffordTSynthesizer._synthesize_rz)
    write their message and backtrace directly to the underlying OS file
    descriptor, bypassing Python's sys.stderr object entirely --
    contextlib.redirect_stderr only redirects the latter, so it can't
    intercept them.

    Yields the temp file; `with ... as capture:` binds it before the guarded
    call runs, so it stays readable in the caller's scope even if that call
    raises and the exception is caught outside this block. The caller is
    responsible for seeking/reading (and closing) it afterward.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    capture = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(capture.fileno(), stderr_fd)
        yield capture
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


class CliffordTSynthesizer:
    """Re-synthesise single-qubit unitaries over {h, s, sdg, x, y, z, t, tdg} via gridsynth."""

    def __init__(
        self,
        epsilon: float = EPSILON_DEFAULT,
    ) -> None:
        self.epsilon = epsilon
        # Always epsilon (floored at EXACTNESS_FLOOR) so the exact paths
        # aren't looser than the approximate one: a rotation only comes out
        # free if it really is within the requested accuracy of a Clifford.
        # Not independently tunable -- see _exact's docstring for what this
        # tolerance gates.
        self.tol = max(epsilon, EXACTNESS_FLOOR)
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
        """Return a single-qubit Clifford+T circuit implementing `matrix`."""
        circuit, kind, error = self._synthesize_uncounted(matrix)
        if kind == "clifford":
            self.n_clifford += 1
        elif kind == "approx":
            self.n_approx += 1
        else:
            self.n_exact += 1
        self._record(error)
        return circuit

    def _synthesize_uncounted(self, matrix: np.ndarray) -> tuple[QuantumCircuit, str, float]:
        """(circuit, kind, error) for `matrix`, without touching counters or
        error_bound -- kind is "clifford"/"pi_4" (exact path) or "approx"
        (gridsynth path). Split out of synthesize() so a caller that wants to
        account the result into its own counters (rather than this
        instance's) can reuse the computation directly.
        """
        exact = self._exact(matrix)
        if exact is not None:
            return exact
        word = self._euler_word(self._decomposer(matrix), self._gridsynth_word)
        error, phase = word_error(matrix, word)
        return circuit_from_word(word, matrix, phase), "approx", error

    def _exact(self, matrix: np.ndarray) -> Optional[tuple[QuantumCircuit, str, float]]:
        """Exact synthesis as (circuit, kind, error), or None if it needs approximating.

        A candidate word is only accepted once it has been measured against
        `matrix` and found to be within self.tol (== epsilon, floored at
        EXACTNESS_FLOOR).  The Clifford lookup needs that check because
        canonical_key rounds to a fixed decimal precision, so the table
        matches anything extremely close to a Clifford; without it, every
        rotation smaller than that -- the pi/2^k tail of a wide QFT, say --
        would be silently thrown away for free at an error far above
        --epsilon.  The pi/4 path needs it because _is_pi_4_multiple accepts
        angles up to self.tol off a multiple, and three such angles compound.
        """
        word = self._clifford_words.get(canonical_key(matrix))
        if word is not None:
            error, phase = word_error(matrix, word)
            if error <= self.tol:
                return circuit_from_word(word, matrix, phase), "clifford", error

        euler = self._decomposer(matrix)
        angles = [inst.operation.params[0] for inst in euler.data if inst.operation.params]
        if all(self._is_pi_4_multiple(a) for a in angles):
            word = self._euler_word(euler, self._pi_4_word)
            error, phase = word_error(matrix, word)
            if error <= self.tol:
                return circuit_from_word(word, matrix, phase), "pi_4", error
        return None

    def estimate_synthesis_calls(self, circuit: QuantumCircuit) -> int:
        """How many distinct Rz(angle) gridsynth searches resynthesizing
        `circuit` will actually perform, i.e. cache misses in
        _gridsynth_word -- cheap to compute up front (Euler decomposition
        only, no lattice search) but gives an accurate denominator for
        weighting progress by real work rather than block count, since a
        handful of unique angles can dominate the wall-clock cost of
        thousands of repeated ones (see _with_progress).

        `self._gridsynth_cache` may already hold entries from an earlier
        circuit (compile_dispatch's callers can reuse one synth across a
        batch of files) -- those angles are excluded rather than counted, so
        `total` reflects only the *new* misses this circuit will cause, not
        the accumulated cache size.
        """
        already_cached = set(self._gridsynth_cache)
        new_keys: set[float] = set()

        def counter(matrix, run):
            if self._exact(matrix) is not None:
                return None
            for inst in self._decomposer(matrix).data:
                if not inst.operation.params:
                    continue
                angle = inst.operation.params[0]
                if not self._is_pi_4_multiple(angle):
                    key = round(angle, 12)
                    if key not in already_cached:
                        new_keys.add(key)
            return None

        rewrite_single_qubit_runs(circuit, counter)
        return len(new_keys)

    def _expensive_cache(self) -> dict:
        """The cache whose growth marks a genuine gridsynth search, as
        opposed to a cache hit or an exact/Clifford block -- watched by
        _with_progress to weight progress by real work, not block count."""
        return self._gridsynth_cache

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
        capture = None
        try:
            with _capture_stderr_fd() as capture:
                circuit = gridsynth_rz(angle, self.epsilon)
        except BaseException as error:
            # rsgridsynth panics on some angles (KNOWN_GRIDSYNTH_PANIC) --
            # unpredictably: which angles fail depends on process state, not
            # just the angle, so retrying is not a fix. Caught as BaseException
            # because a pyo3 panic escapes as one, bypassing the per-file
            # Exception handling in main().
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if gridsynth_gates is None:
                raise RuntimeError(
                    f"qiskit's gridsynth failed on angle {angle} at epsilon "
                    f"{self.epsilon:g} ({type(error).__name__}: {error}). Use a "
                    "smaller --epsilon or install pygridsynth as a fallback."
                ) from error
            # pyo3/rsgridsynth prints its own message and backtrace directly
            # to fd 2, bypassing sys.stderr -- _capture_stderr_fd caught it so
            # it can be judged instead of shown unconditionally. The
            # known-safe panic is swallowed entirely; anything else is
            # forwarded verbatim, since it hasn't been verified safe to hide.
            if KNOWN_GRIDSYNTH_PANIC not in str(error):
                if capture is not None:
                    capture.seek(0)
                    raw = capture.read()
                    if raw:
                        print(raw.decode(errors="replace"), file=sys.stderr, end="")
                print(
                    f"note: qiskit's gridsynth (Rust) raised an unexpected "
                    f"error on one rotation ({type(error).__name__}: {error}) "
                    "-- used the pygridsynth fallback instead. This is NOT "
                    "the known rsgridsynth panic _synthesize_rz was written "
                    "for (raw Rust output above, if any) -- if this recurs, "
                    "it may need its own handling.",
                    file=sys.stderr,
                )
            return self._pygridsynth_word(angle)
        else:
            # Forward any stderr rsgridsynth wrote without panicking --
            # unexpected but never silently discarded.
            if capture is not None:
                capture.seek(0)
                raw = capture.read()
                if raw:
                    print(raw.decode(errors="replace"), file=sys.stderr, end="")
            return tuple(inst.operation.name for inst in circuit.data)
        finally:
            if capture is not None:
                capture.close()

    def _pygridsynth_word(self, angle: float) -> tuple[str, ...]:
        """Same rotation via pygridsynth, the implementation bqskit-ft calls."""
        if gridsynth_gates is None or mpmath is None:  # __init__ checks this
            raise RuntimeError("pygridsynth is not installed")
        sequence = gridsynth_gates(mpmath.mpf(angle), mpmath.mpf(self.epsilon))
        # pygridsynth returns the word in matrix order, so it is applied in reverse.
        return tuple(GRIDSYNTH_NAMES[symbol] for symbol in reversed(sequence) if symbol != "W")


# The synthesizer interface compile_via_resynthesis's pipeline expects
# (epsilon, tol, the n_*/max_error/error_bound counters, reset_counters(),
# synthesize()) -- currently just CliffordTSynthesizer, kept as its own name
# since compile_via_resynthesis/compile_dispatch/main all document their
# `synth` parameter against this interface rather than the concrete class.
ResynthesisSynthesizer = CliffordTSynthesizer


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

    Consecutive single-qubit gates on a wire are accumulated into one matrix
    and handed to `resynthesize(matrix, run)`, which returns a replacement
    single-qubit circuit, or None to keep the original gates.  Working on runs
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
    """The {u,cx} unroll both backends resynthesise from.

    Structurally necessary, not an optimization: it is what breaks multi-qubit
    gates neither backend otherwise knows how to re-synthesise (ccx, cp, rzz,
    ...) down into {single-qubit unitary, cx}. UNROLL_OPTIMIZATION_LEVEL also
    matters for how well single-qubit runs merge before resynthesis (see its
    comment). Deliberately just this one transpile and nothing else -- this
    script is an intentionally naive baseline, not a competitive compiler; no
    exact-rewrite preprocessing (e.g. merging redundant phase-polynomial
    rotations) runs ahead of it.
    """
    return transpile(circuit, basis_gates=["u", "cx"], optimization_level=UNROLL_OPTIMIZATION_LEVEL)


def _no_log(*_a, **_k) -> None:
    """The -q/--quiet log() implementation, named so _progress_line can
    recognize it by identity and skip its own output (including the
    /dev/tty path, which bypasses log() and so wouldn't otherwise see -q)."""


@functools.cache
def _progress_tty():
    """Lazily open /dev/tty for direct terminal writes, memoized for the
    life of the process. Returns None if there is no controlling terminal at
    all (headless: CI, a detached cron job, `nohup` with no pty).
    """
    try:
        return open("/dev/tty", "w")
    except OSError:
        return None


def _progress_line(log, label: str, pct: int, *, final: bool = False) -> None:
    """Emit one progress update, overwriting a single line in place (a
    trailing \\r) exactly as before.

    Writes straight to /dev/tty rather than through log()/stdout, whenever a
    controlling terminal exists -- checking sys.stdout.isatty() instead is
    not enough: piped through `tee` (or any pipe), stdout is never a tty even
    though a real terminal is watching on the other end, and tee duplicates
    whatever reaches stdout byte-for-byte into its file, so a \\r-overwrite
    written there shows up as ^M-separated junk in the file no matter what.
    Writing to /dev/tty directly is invisible to tee/redirection entirely:
    the terminal still gets the live single-line ticker, and anything
    capturing stdout gets none of these bytes at all.

    Falls back to one plain line per update via log() only when there's no
    controlling terminal whatsoever (fully headless), so a persistent log
    still shows some progress instead of nothing. `log` being the no-op
    (quiet mode, see _no_log) suppresses this the same as everywhere else.
    """
    if log is _no_log:
        return
    tty = _progress_tty()
    if tty is not None:
        print(f"\r  {label}: {pct}%   ", end="\n" if final else "", file=tty, flush=True)
    else:
        log(f"  {label}: {pct}%")


def _with_progress(resynthesize, total: int, log, label: str, cache: dict):
    """Wrap a resynthesize callback to report percentage progress via
    _progress_line (overwriting a single line in place on a terminal, one
    line per update otherwise), throttled to at most one update per
    PROGRESS_INTERVAL_SECONDS. Does not itself print a final completion line
    -- compile_via_resynthesis does that unconditionally once the real pass
    returns, rather than relying on this wrapper to recognize its own last
    call (see `count`'s clamp below for why that can't be done reliably by
    watching `cache` alone).

    Time-throttled rather than just percentage-throttled: a circuit with only
    a few dozen blocks would otherwise update close to once per block -- each
    essentially instant if the blocks are cheap (Clifford/exact), far faster
    than the line is readable.  A fast compile (finishing inside one
    interval) shows no progress line until the unconditional completion
    line; a slow one visibly ticks upward for as long as it actually runs.

    `cache` is the synthesizer's cache dict (CliffordTSynthesizer's
    `_expensive_cache()`); `count` advances by however many entries a call
    actually *adds* to it (not just whether it grew), i.e. genuine gridsynth
    searches, not merged runs. A single CliffordTSynthesizer block can need
    gridsynth for more than one of its Euler angles (Rz and Rx) in one call,
    growing the cache by more than one entry at once -- counting only "grew:
    yes/no" as a single unit systematically undercounts (confirmed via
    data/qasmbench/dnn_n8.qasm: the old scheme stalled well short of
    complete). CliffordTSynthesizer caches by angle key, so on circuits with
    a lot of repeated rotations (QFT-family circuits, say) the vast majority
    of calls are instant cache hits -- with plain block counting, progress
    used to race through the first stretch (the actual searches) then jump
    straight to complete on the cache hits, rather than tracking real elapsed
    time. `total` (from estimate_synthesis_calls) is only an ESTIMATE of how
    many new cache entries will appear, so `count` reaching `total` exactly
    is not guaranteed; it is clamped at `total` (never shown past complete)
    but completion is never inferred from reaching it.
    """
    if total == 0:
        return resynthesize
    count = 0
    last_report = 0.0

    def wrapped(matrix, run):
        nonlocal count, last_report
        before = len(cache)
        result = resynthesize(matrix, run)
        count = min(count + (len(cache) - before), total)
        now = time.monotonic()
        if now - last_report >= PROGRESS_INTERVAL_SECONDS:
            # Percentage only, not "N/M": M's meaning differs enough between
            # backends that raw counts would invite an inapt cross-backend
            # comparison.
            _progress_line(log, label, count * 100 // total)
            last_report = now
        return result

    return wrapped


def compile_via_resynthesis(
    unrolled: QuantumCircuit,
    synth: ResynthesisSynthesizer,
    log=lambda *a, **k: None,
) -> QuantumCircuit:
    """Re-synthesise an already-{u,cx}-unrolled circuit over Clifford+T.

    Used by the qiskit backend: one unconditional pass over every
    single-qubit run via rewrite_single_qubit_runs, dispatching each matrix to
    `synth` (a CliffordTSynthesizer). `log` reports percentage progress
    through the pass. Deliberately just this one pass and nothing else --
    this script is an intentionally naive baseline, not a competitive
    compiler; there is no post-synthesis clean-up (inverse cancellation,
    exact re-synthesis of collapsible runs).
    """
    total = synth.estimate_synthesis_calls(unrolled)
    label = "resynthesizing"
    out = rewrite_single_qubit_runs(
        unrolled,
        _with_progress(synth.synthesize, total, log, label, synth._expensive_cache()),
    )
    if total > 0:
        # Printed unconditionally: `total` is only an estimate of cache
        # growth, so the tracked count reaching it exactly isn't guaranteed.
        _progress_line(log, label, 100, final=True)
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

    Dense: builds both full-dimensional operators, so it is limited to a
    modest qubit count by memory.  Use statevector_fidelity above that.
    """

    def unitary(circuit: QuantumCircuit) -> np.ndarray:
        return np.asarray(Operator(unitary_part(circuit)).data, dtype=complex)

    left, right = unitary(lhs), unitary(rhs)
    dim = left.shape[0]
    return float(abs(np.trace(left.conj().T @ right)) / dim)


def statevector_fidelity(lhs: QuantumCircuit, rhs: QuantumCircuit, seed: int = 0) -> float:
    """|<lhs psi | rhs psi>| for one Haar-random |psi>.

    Costs amplitudes scaling with the state space rather than its square,
    which is what makes verification possible well past where the dense
    check above runs out.  It samples the unitaries rather than comparing
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
) -> tuple[QuantumCircuit, Optional[float]]:
    """Compile an already-{u,cx}-unrolled circuit via bqskit's own stock
    Clifford+T workflow, returning a qiskit QuantumCircuit (round-tripped
    through qiskit's own loader, so it can share verification/reporting/
    writing with the qiskit backend) and an error bound (always `None` --
    see below).

    Deliberately the simplest possible bqskit usage -- the exact same call
    src/compile_circuit.py makes (`CliffordTModel` + bqskit's own top-level
    `compile()`), not a hand-tuned pass list: this script is an intentionally
    naive baseline, not a competitive compiler.

    Always returns `None` for the error bound, matching src/compile_circuit.py
    (which reports none either): bqskit's stock workflow
    (bqskit.ft.cliffordt.defaultworkflow.build_cliffordt_workflow) never sets
    `calculate_error_bound=True` on any of its passes, so there is nothing
    real to read back even via `Compiler().compile(..., request_data=True)`
    -- confirmed by reading its source, not assumed. Wrapping the stock passes
    with error tracking of our own would mean hand-tuning the workflow again,
    which defeats the point of using bqskit's own default as-is.
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.qasm"
        qasm2.dump(unrolled, in_path)
        bq_circuit = Circuit.from_file(str(in_path))

        machine = CliffordTModel(bq_circuit.num_qudits)
        bq_circuit = bqskit_compile(bq_circuit, model=machine, synthesis_epsilon=epsilon, seed=seed)

        # Flatten any CircuitGate wrappers left around sub-circuits (e.g.
        # U3Gate) -- without this, the transpile binary crashes on CircuitGate.
        bq_circuit.unfold_all()

        # Remove IdentityGate: bqskit serialises it as a custom "identity1"
        # gate that, once reloaded, parses back as CircuitGate(U3Gate) and
        # crashes the transpile binary.
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
        return load_circuit(out_path), None  # qiskit's own loader -- now a QuantumCircuit


def compile_dispatch(
    unrolled: QuantumCircuit,
    args: argparse.Namespace,
    synth: Optional[ResynthesisSynthesizer] = None,
    log=lambda *a, **k: None,
) -> tuple[QuantumCircuit, list[str], Optional[float], Optional[int]]:
    """Compile an already-unrolled circuit via whichever backend args.backend selects.

    Returns (compiled, extra report lines, error_bound, n_rewrites). `synth`, if
    given, is reused (and its counters read afterward) rather than creating a
    fresh one -- this lets main()'s per-file loop keep sharing one synthesizer,
    and its resynthesis cache, across a batch of input files. Random-window
    sampling (windowed_fidelity) instead passes no synth, since a window is
    small enough that losing cache reuse across windows doesn't matter. `log`
    reports resynthesis progress for the qiskit backend (see
    compile_via_resynthesis); windowed_fidelity doesn't pass one, since a
    window is small enough that per-window progress would just be noise.
    """
    if args.backend == "qiskit":
        if synth is None:
            synth = CliffordTSynthesizer(epsilon=args.epsilon)
        compiled = compile_via_resynthesis(unrolled, synth, log=log)
        extra = [
            f"1q runs: {synth.n_clifford} Clifford, {synth.n_exact} exact,"
            f" {synth.n_approx} approximated"
            f" (worst rewrite {synth.max_error:.2e}, total {synth.error_bound:.2e})"
        ]
        return compiled, extra, synth.error_bound, synth.n_approx
    compiled, error_bound = compile_bqskit(unrolled, epsilon=args.epsilon, seed=args.seed)
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
        if (
            inst.clbits
            or isinstance(inst.operation, ControlFlowOp)
            or inst.operation.name
            in (
                "measure",
                "reset",
                "delay",
            )
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


def windowed_fidelity(source: QuantumCircuit, args: argparse.Namespace) -> list[str]:
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
    count up by orders of magnitude (a small window was measured compiling
    to a much larger one), and the fidelity check's real cost depends on
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
    rng = random.Random(VERIFY_SEED)
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
            else statevector_fidelity(window, compiled_window, VERIFY_SEED)
        )
        fidelities.append(fid)
        notes.append(
            f"  window {len(fidelities)}: {window.num_qubits} qubits,"
            f" {len(window.data)} gates, fidelity {fid:.12f}"
        )
    if not fidelities:
        return ["no numeric check: could not find any sample window small enough to verify"]
    worst = min(fidelities)
    header = (
        f"random-window fidelity (worst of {len(fidelities)} samples,"
        f" up to {WINDOW_VERIFY_MAX_QUBITS} qubits each): {worst:.12f}"
    )
    return [header] + notes


def verify_fidelity(
    source: QuantumCircuit,
    compiled: QuantumCircuit,
    args: argparse.Namespace,
) -> list[str]:
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
    notes: list[str] = []
    control_flow = has_control_flow(compiled) or has_control_flow(source)
    qubits = compiled.num_qubits
    gates = operation_counts_cost(compiled)[1] + operation_counts_cost(source)[1]
    try:
        if (
            not control_flow
            and qubits <= DENSE_VERIFY_MAX_QUBITS
            and gates <= DENSE_VERIFY_MAX_GATES
        ):
            fidelity = unitary_fidelity(source, compiled)
            notes.append(f"unitary fidelity vs input: {fidelity:.12f}")
        elif (
            not control_flow
            and qubits <= STATEVECTOR_VERIFY_MAX_QUBITS
            and gates * 2**qubits <= STATEVECTOR_VERIFY_MAX_OPS
        ):
            state_fidelity = statevector_fidelity(source, compiled, VERIFY_SEED)
            notes.append(
                f"statevector fidelity vs input: {state_fidelity:.12f}"
                f" (1 random state, seed {VERIFY_SEED})"
            )
        else:
            if control_flow:
                notes.append(
                    "circuit has classical control flow: falling back to random-window sampling"
                )
            notes.extend(windowed_fidelity(source, args))
    except Exception as error:  # mid-circuit measurement, unsupported op, memory
        notes.append(f"numeric check failed to run ({type(error).__name__}: {error})")
    return notes


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

    `extra` carries the "1q runs: N Clifford, ..." diagnostic (from
    CliffordTSynthesizer's counters); it reflects bookkeeping internal to
    compile_via_resynthesis's own pipeline that bqskit's compiler passes
    don't expose, so it is empty for the bqskit backend.
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
        description="Compile a QASM circuit to the Clifford+T gate set, via qiskit "
        "or bqskit.",
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
        default="qiskit",
        help="compilation backend: qiskit or bqskit (rejects circuits with "
        "classical control flow). Both are deliberately naive, unoptimized "
        "baselines -- see the module docstring -- for comparing against the "
        "Rust compile_cliffordt implementation",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="gridsynth target error per rotation, shared by both backends -- "
        "see module docstring's \"Rotation synthesis: gridsynth\" section "
        "for the measurements behind this number",
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
    parser.add_argument("-q", "--quiet", action="store_true", help="only report errors")

    args = parser.parse_args(argv)

    # Warn (not error) if a backend-inappropriate flag was explicitly set to a
    # non-default value, so both backends can share one parser without
    # silently ignoring a flag the user thought they were setting.
    backend_only_flags = {
        "qiskit": {
            "seed": "--seed",
        },
    }
    for dest, flag in backend_only_flags.get(args.backend, {}).items():
        if getattr(args, dest) != parser.get_default(dest):
            print(
                f"warning: {flag} has no effect with --backend {args.backend}",
                file=sys.stderr,
            )

    return args


def _git_branch_and_commit() -> tuple[str, str]:
    """(branch, short commit sha) for the repo this script lives in, or
    ("unknown", "unknown") if git isn't available or this isn't a checkout.

    Mirrors puremagic.rs's vergen-baked "Git branch: ... | Commit: ..."
    banner (env!("VERGEN_GIT_BRANCH")/env!("VERGEN_GIT_SHA")), but read at
    run time: a script has no separate compile step to bake it in at.
    """
    repo_dir = Path(__file__).resolve().parent
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return branch, commit[:8]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    branch, commit = _git_branch_and_commit()
    run_time = datetime.now(timezone.utc).isoformat()
    print(f"compile_cliffordt.py - Git branch: {branch} | Commit: {commit} | Run: {run_time}")
    # **k forwards e.g. end="" to print, which _with_progress uses to overwrite
    # a single progress line via \r rather than printing one line per update.
    log = _no_log if args.quiet else lambda *a, **k: print(*a, flush=True, **k)

    synth: Optional[ResynthesisSynthesizer] = None
    if args.backend == "qiskit":
        synth = CliffordTSynthesizer(epsilon=args.epsilon)
        log(f"backend: qiskit (epsilon={args.epsilon:g})")
    else:
        log(f"backend: bqskit (epsilon={args.epsilon:g}, seed={args.seed})")

    failures = 0
    for source in args.inputs:
        log(f"=== {source}")
        timings: dict[str, float] = {}
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
                        unrolled, args, synth, log=log
                    )

                after = circuit_stats(compiled)
                print_report(log, before, after, extra)

                # Basis check + error bound always run, no flag needed: cheap,
                # and worth failing the run over regardless of --verify.
                non_basis = non_basis_ops(compiled)
                if error_bound is not None:
                    scope = f" over {n_rewrites} rewrites" if n_rewrites is not None else ""
                    log(f"  error bound{scope}: {error_bound:.2e}")
                else:
                    log("  error bound: n/a (bqskit's stock workflow doesn't report one)")
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
                        notes = verify_fidelity(circuit, compiled, args)
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

        if non_basis:
            print(
                f"ERROR {source}: output is not Clifford+T: {non_basis}",
                file=sys.stderr,
            )
            failures += 1

        log(f"  wrote {destination} (OpenQASM {version})")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
