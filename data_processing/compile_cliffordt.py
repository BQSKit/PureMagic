#!/usr/bin/env python3
"""
Compile an arbitrary OpenQASM circuit into the Clifford+T gate set, via qiskit,
bqskit, or cyclosynth.

Output gate set: h, s, sdg, x, y, z, t, tdg, cx (plus measure/barrier/reset and
any classical control flow that was present in the input -- the bqskit backend
rejects circuits with control flow, since bqskit's own `Circuit` has no concept
of it; the qiskit and cyclosynth backends both handle it, since they share the
same rewrite_single_qubit_runs pipeline, which recurses into control-flow
blocks). The bqskit backend also natively emits sx/sxdg (sqrt(X) and its
inverse): bqskit's own ZXZXZ decomposition and Clifford+T gate set treat sx as
a Clifford generator in its own right rather than expanding it to h/s, and the
downstream Rust `transpile` binary understands both natively, so this is not
rewritten away. The qiskit and cyclosynth backends never produce either.

Pipeline
--------
1. Load the QASM file (OpenQASM 2 first, falling back to OpenQASM 3 -- which
   needs the optional `qiskit-qasm3-import` package).
2. Preopt: first merge_phase_polynomial exactly cancels/merges redundant
   diagonal (rz-family) rotations via their phase-polynomial "parity" (see
   its docstring) -- e.g. a QFT's CX-ladder-decomposed controlled-phase gates
   are full of rotations that are exactly redundant with each other in this
   sense. That merge is not always a net win (moving a rotation to satisfy a
   global parity match can incidentally cost more than it saves -- see
   unroll_to_u_cx's docstring), so both the merged and unmerged circuits get
   transpiled to {u, cx} (so that every multi-qubit gate -- ccx, cswap, cry,
   rzz, ryy, ... -- is broken down into cx plus single-qubit rotations) and
   CliffordTSynthesizer.count_real_rotations, a cheap gridsynth-free proxy for
   actual T-count, picks whichever needs fewer real rotations. All three
   backends resynthesise from this same {u, cx} circuit -- necessary for the
   qiskit backend (which only knows how to re-synthesise 1-qubit runs, not
   arbitrary multi-qubit gates) and a real, if smaller, improvement for the
   bqskit backend too (bqskit's own partitioner refragments single-qubit runs
   across 2-qubit block boundaries regardless of input quality, so it cannot
   benefit from the preopt step as much as qiskit's own pipeline does).
3. Resynthesise over Clifford+T (--backend qiskit, the default, bqskit, or
   cyclosynth):

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
   docstring for what the difference actually is). At the same --epsilon,
   produces more T gates than the qiskit backend on every non-QFT benchmark
   measured so far -- kept for comparison and as an independently implemented
   Clifford+T compiler, not because it is competitive.

   QFT-family circuits are the one measured exception: there, bqskit can
   still produce fewer T gates even after merge_phase_polynomial's exact
   upstream reduction (e.g. qft_N032: 27341 vs qiskit's 42207, ~1.5x fewer --
   before that preopt step existed, the same comparison was 76466 vs 113142,
   so the ratio hasn't really changed even though both numbers dropped by
   more than half). This isn't a synthesis-quality difference -- gridsynth
   costs the same ~82-86 T per rotation at epsilon=1e-8 regardless of whether
   qiskit's Rust gridsynth_rz or bqskit's pygridsynth does the synthesis
   (verified directly, same angles). It comes from build_bqskit_workflow's
   QuickPartitioner(2) + ScanningGateRemovalPass step, which runs on raw
   {u,cx} 2-qubit blocks before any Euler decomposition and numerically tests
   whether each gate can simply be dropped and still keep the block's unitary
   within --epsilon -- a fundamentally different, approximate mechanism from
   merge_phase_polynomial's exact parity matching, and one that still finds
   real reductions merge_phase_polynomial cannot: the latter only merges
   rotations that are *exactly* redundant (same parity), while
   ScanningGateRemovalPass's numerical search also catches approximate
   cancellations between rotations that don't share a parity at all.

   --bqskit-trbo: an optional, off-by-default extra stage (needs the
   optional trbo package -- see requirements.txt) that runs TRbO
   (arxiv.org/abs/2603.25101) right after RoundToDiscreteZPass, before
   isolating/synthesising whatever Rz gates remain. Where RoundToDiscreteZPass
   only checks each Rz independently against --epsilon, TRbO numerically
   re-optimises a partitioned block's Rz angles *jointly*, rounding as many as
   possible to Clifford/T-exact while letting the rest absorb the compensating
   error. Measured: 0% change on qft_N008 and on the pinned
   hubbard_18_slice600.qasm reference (both have angles that are analytically
   or physically fixed -- already exactly deduplicated by
   merge_phase_polynomial where applicable -- so there is no leftover gauge
   freedom for a joint optimiser to exploit, matching the TRbO paper's own
   null result on plain QFT); 2-3% fewer T gates on qv_N008_12345 (27084 ->
   26132-26452 across repeated runs -- see the reproducibility note below), a
   Haar-random circuit whose ZXZXZDecomposition blocks do have real gauge
   slack, and which is exactly the circuit class merge_phase_polynomial
   cannot help (and previously regressed, see unroll_to_u_cx). Off by default
   because the win is real but circuit-dependent and not free: the numerical
   optimisation itself cost about 61s for ~250 partition blocks on an
   8-qubit circuit in measurement, unlike every other pass in this workflow.
   Also unlike every other part of the bqskit backend, its T-count is not
   reproducible run to run at a fixed --seed: TRbO's own multi-start search
   dispatches retries through bqskit's runtime independently of this script's
   seeding (confirmed directly: two --seed 0 runs of qv_N008_12345 gave 26452
   and 26239 T), so a --bqskit-trbo run is not safe to treat as a single
   reproducible data point the way every other number in this file is.
   rewrite_single_qubit_runs (the qiskit/cyclosynth backends' shared
   pipeline) has no equivalent of its own: it only merges consecutive
   single-qubit gates on one wire between CX boundaries, and never tests
   whether an entire gate can be dropped from a multi-gate block. Also rejects
   circuits with classical control flow (bqskit's own Circuit has no concept
   of it) -- the qiskit and cyclosynth backends are the only options for
   those.

   cyclosynth: shares the qiskit backend's rewrite_single_qubit_runs pipeline
   entirely (merge each maximal single-qubit run into one 2x2 matrix, clean up
   the same way afterward), differing only in how a non-Clifford block gets
   synthesised: instead of a ZXZ decomposition into up to 3 independently
   gridsynth'd Rz rotations, this backend takes the block's ZYZ Euler angles
   (qiskit's own U3 convention) and hands all three to cyclosynth's
   synthesize_u3 in one call, which returns a single, near-T-optimal
   Clifford+T word for the whole block via a diamond-distance lattice search
   (see "Rotation synthesis" below) rather than gridsynth's Ross-Selinger
   algorithm. Produces fewer T gates than the qiskit backend on every
   benchmark measured so far, at real but not prohibitive compile-time cost
   (see CyclosynthSynthesizer's docstring for measured numbers) -- needs the
   cyclosynth extension built separately (it is a git submodule, not a PyPI
   package; see cyclosynth/README.md and this repo's requirements.txt).
   Blocks whose target lands within CYCLOSYNTH_NEAR_CLIFFORD_MARGIN of a
   Clifford element are instead routed to the qiskit backend's gridsynth
   path: cyclosynth's search can hang or fail to terminate on such targets
   (e.g. a QFT's deep phase gates) -- see cyclosynth-bug-report.md.

The qiskit and cyclosynth backends report percentage progress through both
the main resynthesis pass and each cleanup round after it (silenced by -q,
like all other logging) -- compile_via_resynthesis's shared pipeline makes
this cheap to add once for both. The main pass is weighted by actual
gridsynth/cyclosynth calls (cache misses), not by block count: both
synthesizers cache by angle/matrix key, so on circuits with a lot of
repeated rotations (QFT-family circuits, say) the vast majority of blocks
are instant cache hits, and counting them equally would make progress look
stuck near 0% until the last moment, then jump to 100% -- see
estimate_synthesis_calls/_with_progress. Cleanup rounds instead weight by
plain block count (count_resynthesis_blocks/_with_block_progress):
shorten_run never calls gridsynth/cyclosynth, so there is no cache-hit/
cache-miss split to correct for there, but on a large enough circuit those
rounds are not the "expected to be fast" afterthought they once looked like
-- measured on a 36-qubit, ~1.2M-gate QV circuit, each of 5 cleanup rounds
took about as long as the main resynthesis pass itself, entirely silent
before this was added. bqskit reports none: it exposes no public per-block progress callback,
and the only usable signal (DEBUG-level runtime log lines, one per block)
would need splitting decompose_rz_tracked's currently-atomic compile() call
in two -- with the two halves' error bounds recombined by hand to avoid
changing the already-verified error_bound numbers -- and risks adding real
overhead of its own inside bqskit's runtime-server/worker pipeline. Not
worth it for a backend that is "kept for comparison, not because it is
competitive" (see the bqskit paragraph above).

Rotation synthesis: gridsynth and cyclosynth
---------------------------------------------
The qiskit and bqskit backends both use the Ross-Selinger algorithm
(gridsynth), near T-optimal at T ~ 3*log2(1/epsilon) per rotation. The qiskit
backend uses qiskit's own Rust implementation (qiskit.synthesis.gridsynth_rz,
qiskit >= 2.5, ~5 ms per distinct rotation), falling back to pygridsynth --
the pure-Python/mpmath implementation, ~8x slower -- for the rare angles
rsgridsynth 0.2.0 panics on at coarse epsilon. That fallback already handles
gridsynth failing on part of a circuit; there is no separate "worse but
always works" mode. The bqskit backend uses only pygridsynth (bqskit's own
stock GridSynthPass), with no Rust extension involved at all. Each generic 1q
gate needs up to 3 Rz rotations (ZXZ Euler angles), so the error per gate is
up to 3*epsilon; angles that are exact multiples of pi/4 are synthesised
exactly and cost nothing. Each distinct rotation is synthesised once and
reused, so cost scales with the number of distinct angles rather than the
number of gates.

The cyclosynth backend instead uses cyclosynth's own lattice-search algorithm
(see CyclosynthSynthesizer's docstring), synthesising a whole block's 3 Euler
angles in one call rather than 3 independent Rz rotations -- so its epsilon
is a diamond-distance bound on the whole block, not a per-elementary-rotation
bound like gridsynth's up-to-3*epsilon. Despite that difference in what
epsilon formally bounds, comparing both at the same nominal --epsilon in
practice delivers comparable real accuracy (see below) -- this was verified,
not assumed.

--epsilon defaults to EPSILON_DEFAULT (1e-8) for all three backends. Measured
via exact dense-unitary fidelity (not this script's own coarser --verify
checks) on two small benchmarks (data/qasmbench/dnn_n8.qasm, data/qasmbench/
ising_n10.qasm), real infidelity plateaus by around 1e-8 for every backend's
resynthesis -- tightening further to 1e-10 or 1e-12 costs substantially more
T gates (e.g. dnn_n8 via qiskit: 23592 -> 35144 T from 1e-8 to 1e-12) for a
change in delivered accuracy indistinguishable from float64 rounding noise.
Looser than 1e-8 does cost real accuracy (e.g. 1e-6 measures ~1e-11
infidelity on the same benchmarks, still fine for most purposes but a
genuine, if small, step up from 1e-8's ~1e-12). The cyclosynth backend
plateaus at essentially the same ~2e-12 infidelity at the same 1e-8, on the
same two benchmarks (see CyclosynthSynthesizer's docstring) -- confirming
the shared default is a meaningful apples-to-apples comparison point despite
the differing epsilon semantics above.

The error bound this script reports (all three backends -- see "Verification"
below) is a real upper bound, not an estimate of actual fidelity loss: it
sums per-rewrite worst-case errors, which above assumes every rewrite's error
constructively interferes with every other's. In practice they mostly do
not, so the bound is typically several orders of magnitude looser than the
true error measured above (e.g. dnn_n8 at 1e-8: bound 8.5e-7, actual
infidelity 3.2e-12) -- useful as a ceiling, not as a proxy for how accurate
the output really is.

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
    own unroll and inverse cancellation as exact.  The cyclosynth backend
    computes its error bound identically (CyclosynthSynthesizer._resynthesize
    always re-measures the built word's spectral-norm error via word_error,
    never trusting cyclosynth's own SynthResult.distance -- a diamond-distance
    bound, not the same quantity -- for accounting), so the two backends'
    error bounds are directly comparable.  The bqskit backend (by
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
    ./compile_cliffordt.py circuit.qasm --backend cyclosynth --verify
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

# rsgridsynth's occasional panic (see CliffordTSynthesizer._synthesize_rz) is
# caught and falls back to pygridsynth, but something in the pyo3/rsgridsynth
# panic-to-exception path still prints its own message plus a full backtrace
# to stderr before the catch runs -- confirmed NOT to be the standard Rust
# panic hook honoring RUST_BACKTRACE (tested with it explicitly set to "0" at
# the OS level: the backtrace still printed), so this does not suppress it.
# Kept anyway, defensively: harmless, and setdefault won't override a value
# the user has deliberately set for their own debugging, in case some other
# panic in the dependency chain does honor it.
os.environ.setdefault("RUST_BACKTRACE", "0")

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

try:  # qiskit >= 2.5: Ross-Selinger in Rust, the qiskit backend's own rotation synthesis
    from qiskit.synthesis import gridsynth_rz
except ImportError:
    gridsynth_rz = None

try:  # optional fallback for the angles rsgridsynth 0.2.0 panics on
    import mpmath
    from pygridsynth.gridsynth import gridsynth_gates
except ImportError:
    mpmath = None
    gridsynth_gates = None

try:  # optional: only needed for --backend cyclosynth (see cyclosynth/README.md)
    import cyclosynth
except ImportError:
    cyclosynth = None

try:  # optional: only needed for --bqskit-trbo (see requirements.txt)
    import trbo.trbo  # pyright: ignore[reportMissingImports]
    import trbo.utils  # pyright: ignore[reportMissingImports]
    import trbo.clift  # pyright: ignore[reportMissingImports]
except ImportError:
    trbo = None

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
# Hubbard benchmark at EPSILON_DEFAULT: level 1 merges runs down to 18
# non-Clifford rotations (1476 T); qiskit's own default merges them into 39
# (5703 T).  Same cx count either way, so this is purely about how well the
# runs merge for this script's purposes, not circuit quality by qiskit's own
# metrics.  Pinned rather than exposed as a CLI option, since a worse choice
# was never useful here.  Shared by all three backends' preopt step
# (unroll_to_u_cx).
UNROLL_OPTIMIZATION_LEVEL = 1

# The optimization_level slot build_bqskit_workflow's own workflow is
# registered and invoked under -- register_workflow and bqskit_compile both
# require one, and the two calls must agree on which slot to use. Since
# build_bqskit_workflow always builds the same pass list regardless of this
# number (unlike bqskit's own levels 2-4, which select genuinely different,
# slower workflows), there is nothing to gain from exposing it as a CLI
# option; it is fixed at 1 purely because some value has to be picked.
BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL = 1

# Partition size for --bqskit-trbo's QuickPartitioner stage -- matches
# trbo.workflows' own default() partition_size. Not exposed as a CLI option in
# v1: revisit only if real usage shows a need to trade TRbO's own runtime
# (dominated by per-block multistart numerical optimisation, roughly linear in
# block count) against how much joint gauge freedom a larger block exposes.
TRBO_PARTITION_SIZE = 4

# --epsilon's default, shared by all three backends -- see the module
# docstring's "Rotation synthesis" section for the exact-fidelity
# measurements behind this number: real infidelity plateaus by around this
# value for every backend, so tightening further costs T gates for no
# measurable gain.
EPSILON_DEFAULT = 1e-8

# Minimum time between resynthesis progress updates (see _with_progress).
# Not exposed as a CLI option: this is a UI refresh rate, not a behavioral
# knob -- there is no reason a user would want a different value.
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

# Diagonal gates, as the multiple of pi/4 they rotate about z by.  A block of
# them commutes and collapses to a single Rz(k * pi/4).
DIAGONAL_PHASES = {"t": 1, "tdg": -1, "s": 2, "sdg": -2, "z": 4, "id": 0}

CLIFFORD_1Q_NAMES = frozenset(CLIFFORD_GENERATORS) | {"id"}

# pygridsynth emits words over {H, S, T, X, W}; W is the e^{i pi/4} global phase,
# which we drop because the phase is recomputed against the target anyway.
# (qiskit's gridsynth_rz returns a circuit with lowercase gate names instead.)
GRIDSYNTH_NAMES = {"H": "h", "S": "s", "T": "t", "X": "x"}

# cyclosynth's alphabet is {H,S,T,X,Y,Z}; lowercase = dagger (only S/T have one).
CYCLOSYNTH_GATE_NAMES = {
    "H": "h", "S": "s", "s": "sdg", "T": "t", "t": "tdg", "X": "x", "Y": "y", "Z": "z",
}

# Working precision pygridsynth is driven at, matching bqskit's GridSynthPass.
GRIDSYNTH_DPS = 128

# Floor on the exactness tolerance (see CliffordTSynthesizer.tol).  A run product
# is accumulated over hundreds of 2x2 multiplications, so comparing a word
# against it is only meaningful down to about this much floating-point noise.
EXACTNESS_FLOOR = 1e-12

# cyclosynth's Clifford+T search can become pathologically slow, or fail to
# terminate, for targets whose distance to the nearest Clifford element is
# much smaller than epsilon but still requires genuine (non-exact) synthesis
# -- see cyclosynth-bug-report.md. Measured at EPSILON_DEFAULT (1e-8): solves
# (slowly, ~8s) at ~5e-6 distance, doesn't return in 15s+ at ~2e-6 and below
# -- a continuum, not a sharp cutoff. This margin (1e4 * epsilon = 1e-4 at
# EPSILON_DEFAULT) sits well clear of the observed danger zone with
# comfortable safety margin. Not exposed via --cli: it's a safety heuristic,
# not something a user should need to tune, and setting it too small could
# reintroduce the hang.
CYCLOSYNTH_NEAR_CLIFFORD_MARGIN = 1e4

# The one specific rsgridsynth panic message CliffordTSynthesizer._synthesize_rz
# knows is safe to swallow (falls back to pygridsynth, already measured and
# accounted for in error_bound) -- see _capture_stderr_fd's use there. Matched
# against str(the caught exception), not the raw captured stderr text: pyo3
# already surfaces the panic payload as the exception's message.
KNOWN_GRIDSYNTH_PANIC = "Invalid coefficients for inverse sqrt2 multiplication"

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

    The phase is the one that minimises the error, i.e. the one
    circuit_from_word bakes into the circuit it builds, so the error returned
    is the error of the circuit that will actually be emitted.
    """
    built = word_matrix(word)
    phase = global_phase_between(target, built)
    return spectral_error(target, built, phase), phase


def nearest_clifford_distance(matrix: np.ndarray, clifford_words: dict) -> float:
    """Spectral-norm distance from `matrix` to the closest of the 24
    single-qubit Clifford elements in `clifford_words` -- cheap (24 small
    matrix ops). Used to detect "near-Clifford but not close enough to treat
    as exact" targets, which cyclosynth's search can hang or fail on (see
    cyclosynth-bug-report.md and CYCLOSYNTH_NEAR_CLIFFORD_MARGIN).
    """
    return min(word_error(matrix, word)[0] for word in clifford_words.values())


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


def circuit_from_word(
    word: tuple[str, ...],
    target: np.ndarray,
    phase: Optional[float] = None,
) -> QuantumCircuit:
    """1-qubit circuit applying `word` in order, phase-aligned against `target`."""
    circuit = QuantumCircuit(1)
    for name in word:
        getattr(circuit, name)(0)
    circuit.global_phase = (
        global_phase_between(target, word_matrix(word)) if phase is None else phase
    )
    return circuit


@contextlib.contextmanager
def _capture_stderr_fd():
    """Redirect the OS-level stderr file descriptor (fd 2) to a temp file for
    the duration of the block, then always restore it -- including on an
    unexpected exception, via `finally`, so a bug here can't leave stderr
    silently broken for the rest of the process.

    Needed because Rust panics (see CliffordTSynthesizer._synthesize_rz)
    write their message and backtrace directly to fd 2, bypassing Python's
    sys.stderr object entirely -- contextlib.redirect_stderr only redirects
    the latter, so it can't intercept them.

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
        (gridsynth path). Split out of synthesize() so
        CyclosynthSynthesizer's gridsynth fallback (see its docstring) can
        reuse this computation while accounting the result into its OWN
        counters, rather than this instance's.
        """
        exact = self._exact(matrix)
        if exact is not None:
            return exact
        word = self._euler_word(self._decomposer(matrix), self._gridsynth_word)
        error, phase = word_error(matrix, word)
        return circuit_from_word(word, matrix, phase), "approx", error

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
        return circuit_from_word(tuple(collapsed), matrix, phase)

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

    def count_real_rotations(self, circuit: QuantumCircuit) -> int:
        """Total count of non-Clifford, non-pi/4-multiple Euler angles across
        every 1-qubit run in `circuit`, counted per OCCURRENCE -- unlike
        estimate_synthesis_calls, this does NOT deduplicate by angle. Each
        occurrence costs roughly the same ~constant T regardless of whether
        its angle turns out to be a cache hit or a fresh gridsynth search
        (gridsynth's cost is ~3*log2(1/epsilon) per rotation, largely
        independent of the angle), so occurrence count -- not distinct-angle
        count -- is the right cheap proxy for a circuit's actual T-count
        without running gridsynth on it at all.

        Used by unroll_to_u_cx to choose between merge_phase_polynomial's
        output and the unmerged circuit: that merge is an exact rewrite, but
        not always a net T-count win (see its docstring) -- this gives a
        gridsynth-free way to check which candidate is actually cheaper
        before committing to either.
        """
        total = 0

        def counter(matrix, run):
            nonlocal total
            if self._exact(matrix) is not None:
                return None
            for inst in self._decomposer(matrix).data:
                if not inst.operation.params:
                    continue
                if not self._is_pi_4_multiple(inst.operation.params[0]):
                    total += 1
            return None

        rewrite_single_qubit_runs(circuit, counter)
        return total

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
            # rsgridsynth 0.2.0 panics on some angles (KNOWN_GRIDSYNTH_PANIC):
            # 26% of 300 random angles at 1e-2, 6% at 1e-4 in earlier testing --
            # rare but not confined to coarse epsilon, as first believed: it has
            # since been observed at the default 1e-8 too, on a large enough
            # circuit (qv_N036_12345.qasm, 36 qubits). Which angles fail
            # depends on process state, not just the angle, so retrying is not
            # a fix. A pyo3 panic is a BaseException, so it would otherwise
            # escape the per-file error handling in main().
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if gridsynth_gates is None:
                raise RuntimeError(
                    f"qiskit's gridsynth failed on angle {angle} at epsilon "
                    f"{self.epsilon:g} ({type(error).__name__}: {error}). Use a "
                    "smaller --epsilon or install pygridsynth as a fallback."
                ) from error
            # Something in the pyo3/rsgridsynth panic path prints its own
            # message plus a full backtrace directly to fd 2, bypassing
            # sys.stderr and ignoring RUST_BACKTRACE -- _capture_stderr_fd
            # caught it so it can be judged rather than shown unconditionally.
            # The one specific, already-measured-safe panic is swallowed
            # entirely (already accounted for in error_bound, nothing the
            # user needs to see); anything else is forwarded verbatim plus a
            # note, since it hasn't been verified safe to hide.
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
            # Success: forward any stderr output rsgridsynth wrote without
            # panicking. Unexpected (nothing has ever been observed here),
            # but nothing should be silently lost -- the whole point of
            # capturing rather than discarding.
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


class CyclosynthSynthesizer:
    """Re-synthesise single-qubit unitaries over {h,s,sdg,x,y,z,t,tdg} via cyclosynth.

    Unlike CliffordTSynthesizer's ZXZ-decompose-then-gridsynth-each-Rz
    approach, cyclosynth's synthesize_u3 takes a whole block's ZYZ Euler
    angles (qiskit's own U3 convention: Rz(phi)*Ry(theta)*Rz(lam)) and
    returns one jointly near-T-optimal word for the entire block in a single
    call, via a diamond-distance lattice search (arXiv:2510.05816) rather
    than gridsynth's Ross-Selinger algorithm.

    Measured at EPSILON_DEFAULT and --cyclosynth-threads 1 (for reproducible
    numbers -- see the cost/determinism paragraph below) against the qiskit
    backend (both real circuits, both fidelity 1.000000000000 to the
    precision --verify prints): fewer T gates every time
    (data/hubbard_18_slice600.qasm: 1411 vs 1476; data/qasmbench/knn_n25.qasm:
    1977 vs 2054; data/qasmbench/dnn_n8.qasm: 13316 vs 23592;
    data/qasmbench/ising_n10.qasm: 11349 vs 15333), and (via exact
    dense-unitary fidelity on the two small enough for it) essentially the
    same real infidelity (~2e-12, the same float64 noise floor both backends
    plateau at -- see module docstring's "Rotation synthesis" section),
    confirming the comparison is apples-to-apples despite the two backends'
    epsilon meaning slightly different things (diamond distance vs. closer to
    a spectral-norm bound).

    The cost: real per-call compile time, and results that (unlike qiskit's
    exact algorithm, or bqskit's --seed) are only reproducible if pinned to a
    single thread. cyclosynth's own lattice search is parallelised via rayon,
    with no per-call seed in its public API; whichever thread's candidate
    happens to finish first can vary run to run, so both the exact word and
    the overall T-count are only reproducible at --cyclosynth-threads 1 (see
    that flag's help text). Measured on 10 random angles at EPSILON_DEFAULT
    (1e-8, this 20-core machine): ~0.16s/call at rayon's own default thread
    count vs ~2.4s/call pinned to 1 thread -- about 15x, and the gap widens
    at tighter epsilon (~12x at 1e-10, where a single call can take tens of
    seconds pinned to 1 thread). --cyclosynth-threads therefore defaults to
    rayon's own default (fast, not reproducible) rather than 1 (reproducible,
    much slower) -- pin it to 1 when comparing exact T-counts run to run
    matters more than speed.
    """

    def __init__(
        self,
        epsilon: float = EPSILON_DEFAULT,
        tol: Optional[float] = None,
        threads: Optional[int] = None,
    ) -> None:
        if cyclosynth is None:
            raise RuntimeError(
                "the cyclosynth backend needs the cyclosynth extension module, "
                "which is not installed. From cyclosynth/: pip install maturin, "
                "then maturin build --release and pip install the wheel it "
                "produces (needs a Rust toolchain -- see "
                "cyclosynth/rust-toolchain.toml -- and system gmp/mpfr; see "
                "cyclosynth/README.md)."
            )
        if threads is not None:
            # Must happen before cyclosynth's first search call: rayon builds
            # its global thread pool lazily on first use and reads this env
            # var at that point, not at import time -- setting it here (even
            # though cyclosynth was already imported at module load) still
            # works, confirmed empirically. Only the first value set in a
            # given process actually takes effect (rayon's pool, once built,
            # is fixed for the process's lifetime); this script only ever
            # constructs one CyclosynthSynthesizer per run, so that's moot
            # here, but a second instance with a different `threads` value
            # in the same process would silently keep the first one's count.
            os.environ["RAYON_NUM_THREADS"] = str(threads)
        self.epsilon = epsilon
        self.tol = max(epsilon, EXACTNESS_FLOOR) if tol is None else tol
        self._decomposer = OneQubitEulerDecomposer(basis="ZYZ")
        self._clifford_words = build_clifford_words()
        self._synth = cyclosynth.Synthesizer(epsilon=epsilon, sqrt_t=False)
        self._cache: dict[tuple, tuple[str, ...]] = {}
        # Fallback synthesizer for blocks too close to a Clifford element for
        # cyclosynth's search to handle safely (see _resynthesize and
        # CYCLOSYNTH_NEAR_CLIFFORD_MARGIN). Its own counters are never read --
        # _synthesize_uncounted doesn't populate them -- only its (circuit,
        # kind, error) return and its gridsynth cache (for repeat near-Clifford
        # angles) are used.
        self._fallback = CliffordTSynthesizer(epsilon=epsilon, tol=self.tol)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_clifford = 0
        self.n_exact = 0
        self.n_approx = 0
        self.n_merged = 0
        self.n_gridsynth_fallback = 0
        self.max_error = 0.0
        self.error_bound = 0.0

    def _record(self, error: float) -> None:
        self.max_error = max(self.max_error, error)
        self.error_bound += error

    def synthesize(self, matrix: np.ndarray, _run=None) -> QuantumCircuit:
        circuit, kind, error = self._resynthesize(matrix)
        if kind == "clifford":
            self.n_clifford += 1
        elif kind == "exact":
            self.n_exact += 1
        elif kind == "gridsynth_fallback":
            self.n_gridsynth_fallback += 1
        else:
            self.n_approx += 1
        self._record(error)
        return circuit

    def shorten_run(self, matrix: np.ndarray, run: list[Gate]) -> Optional[QuantumCircuit]:
        """Re-synthesise an already-compiled run; cyclosynth already returns a
        jointly-minimal word per block, so (unlike CliffordTSynthesizer, whose
        gridsynth path leaves diagonal-subblock slack to collect) there is
        nothing further to collapse within one block -- this only helps when
        cancel_inverses has newly merged two previously CX-separated runs."""
        circuit, _, error = self._resynthesize(matrix)
        if gate_cost(circuit) >= gate_cost(run):
            return None
        self.n_merged += 1
        self._record(error)
        return circuit

    def _resynthesize(self, matrix: np.ndarray) -> tuple[QuantumCircuit, str, float]:
        """(circuit, kind, error). Never trusts cyclosynth's own result.distance
        (a diamond-distance bound) for accounting -- always re-measures the
        built word's spectral-norm error via word_error, exactly like
        CliffordTSynthesizer's own paths, so error_bound/max_error stay one
        homogeneous, backend-comparable metric (see module docstring).

        Blocks whose target is very close to (but not within tol of) a
        Clifford element are routed to CliffordTSynthesizer's gridsynth path
        (self._fallback) instead of cyclosynth: cyclosynth's lattice search
        can become pathologically slow or fail to terminate for such targets
        (see cyclosynth-bug-report.md and CYCLOSYNTH_NEAR_CLIFFORD_MARGIN).
        The catch-None branch below is a defense-in-depth backstop for the
        same failure mode slipping past that check -- it protects against a
        clean empty result, not an actual hang.
        """
        key = canonical_key(matrix)
        word = self._clifford_words.get(key)
        if word is not None:
            error, phase = word_error(matrix, word)
            if error <= self.tol:
                return circuit_from_word(word, matrix, phase), "clifford", error

        if nearest_clifford_distance(matrix, self._clifford_words) < CYCLOSYNTH_NEAR_CLIFFORD_MARGIN * self.epsilon:
            circuit, _, error = self._fallback._synthesize_uncounted(matrix)
            return circuit, "gridsynth_fallback", error

        word = self._cache.get(key)
        if word is None:
            theta, phi, lam = self._zyz_angles(matrix)
            result = self._synth.synthesize_u3(theta, phi, lam)
            if result is None or result.gates is None:
                circuit, _, error = self._fallback._synthesize_uncounted(matrix)
                return circuit, "gridsynth_fallback", error
            # cyclosynth's gate string is in matrix order (leftmost = leftmost
            # matrix factor, confirmed empirically against word_matrix/
            # spectral_error), so it is applied in reverse -- same convention
            # as pygridsynth's own output, handled the same way above.
            word = tuple(
                CYCLOSYNTH_GATE_NAMES[ch] for ch in reversed(result.gates)
            )
            self._cache[key] = word

        error, phase = word_error(matrix, word)
        kind = "exact" if error <= EXACTNESS_FLOOR else "approx"
        return circuit_from_word(word, matrix, phase), kind, error

    def estimate_synthesis_calls(self, circuit: QuantumCircuit) -> int:
        """How many distinct blocks resynthesizing `circuit` will actually
        send to cyclosynth's lattice search, i.e. cache misses in
        _resynthesize -- cheap to compute up front (a Clifford-table/
        canonical-key check, no search) but gives an accurate denominator
        for weighting progress by real work rather than block count (see
        CliffordTSynthesizer.estimate_synthesis_calls).

        Deliberately does not call _resynthesize itself: on a genuine miss
        that would trigger the real, expensive synthesize_u3 call. Instead
        it inlines just the Clifford-check and near-Clifford-fallback gating
        from the start of _resynthesize, which must stay in sync with that
        method -- blocks routed to the gridsynth fallback never touch
        self._cache, so they must not be counted as future cyclosynth misses
        either.

        `self._cache` may already hold entries from an earlier circuit
        (compile_dispatch's callers can reuse one synth across a batch of
        files) -- those keys are excluded rather than counted, so `total`
        reflects only the *new* misses this circuit will cause, not the
        accumulated cache size.
        """
        already_cached = set(self._cache)
        new_keys: set[tuple] = set()

        def counter(matrix, run):
            key = canonical_key(matrix)
            word = self._clifford_words.get(key)
            if word is not None:
                error, _ = word_error(matrix, word)
                if error <= self.tol:
                    return None
            if nearest_clifford_distance(matrix, self._clifford_words) < CYCLOSYNTH_NEAR_CLIFFORD_MARGIN * self.epsilon:
                return None
            if key not in already_cached:
                new_keys.add(key)
            return None

        rewrite_single_qubit_runs(circuit, counter)
        return len(new_keys)

    def _expensive_cache(self) -> dict:
        """The cache whose growth marks a genuine cyclosynth search, as
        opposed to a cache hit or an exact/Clifford block -- watched by
        _with_progress to weight progress by real work, not block count."""
        return self._cache

    def _zyz_angles(self, matrix: np.ndarray) -> tuple[float, float, float]:
        """(theta, phi, lam) matching cyclosynth's synthesize_u3 = qiskit's
        U3 = Rz(phi)*Ry(theta)*Rz(lam). OneQubitEulerDecomposer drops trivial
        instructions (e.g. a lone T decomposes to just one rz), so angles are
        assigned by name/position, not a fixed index."""
        theta = phi = lam = 0.0
        seen_ry = False
        for inst in self._decomposer(matrix).data:
            name, angle = inst.operation.name, inst.operation.params[0]
            if name == "ry":
                theta, seen_ry = angle, True
            elif name == "rz":
                if seen_ry:
                    phi = angle
                else:
                    lam = angle
            else:  # pragma: no cover - ZYZ only emits ry/rz
                raise RuntimeError(f"unexpected gate {name} in ZYZ decomposition")
        return theta, phi, lam


# The synthesizer interface compile_via_resynthesis's pipeline expects
# (epsilon, tol, the n_*/max_error/error_bound counters, reset_counters(),
# synthesize(), shorten_run()) -- shared by the qiskit and cyclosynth backends.
ResynthesisSynthesizer = Union[CliffordTSynthesizer, CyclosynthSynthesizer]


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
    compile_via_resynthesis would stop after a single round.
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


PHASE_MERGE_BASIS = ["cx", "rz", "ry", "rx", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"]

# angle contributed by one occurrence of each diagonal single-qubit gate --
# used by merge_phase_polynomial to fold every occurrence into one rz per
# distinct "parity" (see its docstring).  Deliberately a small, exact,
# gate-name-keyed table rather than a numeric matrix check: this pass runs
# before any 1-qubit fusion, precisely so every occurrence still has its own
# recognisable name.
DIAGONAL_1Q_ANGLE = {
    "rz": lambda params: float(params[0]),
    "z": lambda params: math.pi,
    "s": lambda params: math.pi / 2,
    "sdg": lambda params: -math.pi / 2,
    "t": lambda params: math.pi / 4,
    "tdg": lambda params: -math.pi / 4,
}


def merge_phase_polynomial(circuit: QuantumCircuit) -> QuantumCircuit:
    """Exactly merge/cancel redundant diagonal (rz-family) rotations via their
    phase-polynomial "parity" (the t-par technique of Amy, Maslov, Mosca).

    Two rz-family gates anywhere in a {cx, diagonal-gate} region of the
    circuit commute and add exactly whenever they act on the same XOR-parity
    of the original input qubits at the time each is applied -- regardless of
    which physical qubit holds that parity or what runs in between, since cx
    and every diagonal gate commute freely with each other. This matters a
    lot for CX-ladder-decomposed controlled-phase gates (cx, rz(-a), cx,
    rz(a)), which is exactly how QFT-family circuits' controlled-phase gates
    show up after decomposition: measured on qft_N032, this drops the number
    of rotations that actually need gridsynth from 1350 to 522. Verified
    correct (not just counted) by comparing a merged circuit's Operator
    against the original on qft_N008: matched to 2.5e-15 after correcting for
    global phase.

    Tracks each qubit's parity as a Python int bitmask (bit i = "depends on
    original qubit i"), updated by cx as parity[target] ^= parity[control].
    Any gate this doesn't specifically recognise as diagonal -- h/x/y/rx/ry/
    sx/sxdg, any 2+-qubit gate other than cx, measurement, reset, control
    flow -- resets every qubit it touches to a fresh symbol never reused
    elsewhere, rather than trying to track what it does. That's conservative
    by construction: an unrecognised gate can only cause a missed merge
    opportunity, never an incorrect one, since two occurrences can only share
    a parity key if every operation on every contributing qubit in between
    was one of the diagonal gates or cx gates this function explicitly
    understands to commute freely.

    Must run before unroll_to_u_cx's own {u,cx} transpile: that call's
    UNROLL_OPTIMIZATION_LEVEL fuses maximal single-qubit runs into one u3
    gate, which would bake a genuinely mergeable rz together with a
    neighbouring non-diagonal rotation (e.g. an H immediately before or
    after it, as in QFT's own per-qubit "rz; h; rz" runs) into one opaque,
    unaddressable non-diagonal matrix. Runs its own translate-only
    (optimization_level=0, so no 1-qubit fusion) decompose to PHASE_MERGE_BASIS
    first, both to keep every original gate's name intact for classification
    and to break down whatever compound gates the input used (ccx, cp, crz,
    rzz, cswap, ...) into cx plus this vocabulary.

    No CLI flag: this is an exact, tolerance-free rewrite (no epsilon spent),
    the same category as unroll_to_u_cx itself, not a tunable knob -- runs
    unconditionally for all three backends. Does not recurse into
    control-flow bodies (resets their qubits' parity instead, like any other
    unrecognised construct) and does not attempt to also reduce cx count
    (qiskit's own synth_cnot_phase_aam GraySynth implementation could do
    that from the same merged {parity: angle} table, but rebuilding the cx
    ladder from scratch is a separate, riskier change for no extra T-count
    benefit -- left as possible future work).
    """
    decomposed = transpile(circuit, basis_gates=PHASE_MERGE_BASIS, optimization_level=0)
    n = decomposed.num_qubits
    parity = [1 << i for i in range(n)]
    next_symbol = n

    group_key: list[Optional[int]] = []
    group_total: dict[int, float] = {}
    group_last_index: dict[int, int] = {}

    for i, instr in enumerate(decomposed.data):
        operation = instr.operation
        name = operation.name
        qubits = [decomposed.find_bit(q).index for q in instr.qubits]
        if name == "cx":
            c, t = qubits
            parity[t] ^= parity[c]
            group_key.append(None)
        elif name in DIAGONAL_1Q_ANGLE:
            key = parity[qubits[0]]
            group_key.append(key)
            group_total[key] = group_total.get(key, 0.0) + DIAGONAL_1Q_ANGLE[name](operation.params)
            group_last_index[key] = i
        elif name == "barrier":
            group_key.append(None)
        else:
            for q in qubits:
                parity[q] = 1 << next_symbol
                next_symbol += 1
            group_key.append(None)

    out = decomposed.copy_empty_like()
    for i, instr in enumerate(decomposed.data):
        key = group_key[i]
        if key is None:
            out.append(instr)
            continue
        if i != group_last_index[key]:
            continue  # an earlier occurrence of this parity already covers it
        total = group_total[key] % (2 * math.pi)
        if total > EXACTNESS_FLOOR:
            out.rz(total, instr.qubits[0])
    return out


def unroll_to_u_cx(circuit: QuantumCircuit, epsilon: float = EPSILON_DEFAULT) -> QuantumCircuit:
    """The {u,cx} unroll + optimization step all three backends resynthesise from.

    Structurally necessary, not just an optimization: it is what breaks
    multi-qubit gates none of the backends otherwise know how to re-synthesise
    (ccx, cp, rzz, ...) down into {1-qubit unitary, cx}.  UNROLL_OPTIMIZATION_LEVEL
    also matters a great deal for how well single-qubit runs merge before
    resynthesis (see its comment).

    merge_phase_polynomial runs first on a separate candidate (see its own
    docstring for why it can't run after this call's 1-qubit fusion), exactly
    cancelling/merging redundant diagonal rotations before anything here has a
    chance to obscure them -- but that merge is not always a net win: moving a
    rotation to satisfy a global parity match can incidentally break a
    neighbouring gate's *local* gauge-cancellation (see merge_phase_
    polynomial's docstring), costing more real rotations than it saves on
    circuits without much genuine redundancy to find (measured 10% more T on
    a random Quantum Volume benchmark, vs. ~2.7x fewer on a QFT one). Rather
    than guess which applies, both the merged and unmerged candidates are
    unrolled and CliffordTSynthesizer.count_real_rotations -- cheap, no
    gridsynth involved -- picks whichever needs fewer real rotations. `epsilon`
    only affects that comparison's Clifford/exact tolerance, not either
    candidate's actual gate content.
    """
    merged_unrolled = transpile(
        merge_phase_polynomial(circuit), basis_gates=["u", "cx"], optimization_level=UNROLL_OPTIMIZATION_LEVEL
    )
    unmerged_unrolled = transpile(circuit, basis_gates=["u", "cx"], optimization_level=UNROLL_OPTIMIZATION_LEVEL)
    probe = CliffordTSynthesizer(epsilon=epsilon)
    if probe.count_real_rotations(merged_unrolled) <= probe.count_real_rotations(unmerged_unrolled):
        return merged_unrolled
    return unmerged_unrolled


def _progress_line(log, label: str, pct: int, *, final: bool = False) -> None:
    """Emit one progress update: overwrites a single line in place (a
    trailing \\r) on a terminal, same as before. Redirected to a file, \\r has
    no such meaning -- it's just a literal byte -- so every throttled update
    would otherwise pile up on one line as ^M-separated junk; there, print a
    normal, independent line per update instead (with no padding, since
    there's no previous line's leftover characters to clear).
    """
    if sys.stdout.isatty():
        log(f"\r  {label}: {pct}%   ", end="\n" if final else "")
    else:
        log(f"  {label}: {pct}%")


def _with_progress(resynthesize, total: int, log, label: str, cache: dict):
    """Wrap a resynthesize callback to report percentage progress via
    _progress_line (overwriting a single line in place on a terminal, one
    line per update otherwise), throttled to at most one update per
    PROGRESS_INTERVAL_SECONDS. Does not itself print a final 100% line --
    compile_via_resynthesis does that unconditionally once the real pass
    returns, rather than relying on this wrapper to recognize its own last
    call (see `count`'s clamp below for why that can't be done reliably by
    watching `cache` alone).

    Time-throttled rather than just percentage-throttled: a circuit with only
    a few dozen blocks would otherwise update close to once per block -- each
    essentially instant if the blocks are cheap (Clifford/exact), far faster
    than the line is readable.  A fast compile (finishing inside one
    interval) shows no progress line until the unconditional 100%; a slow
    one visibly ticks upward for as long as it actually runs.

    `cache` is the synthesizer's cache dict (CliffordTSynthesizer/
    CyclosynthSynthesizer's `_expensive_cache()`); `count` advances by
    however many entries a call actually *adds* to it (not just whether it
    grew), i.e. genuine gridsynth/cyclosynth searches, not merged runs. A
    single CliffordTSynthesizer block can need gridsynth for more than one
    of its Euler angles (Rz and Rx) in one call, growing the cache by 2 or 3
    at once -- counting only "grew: yes/no" as +1 systematically undercounts
    (confirmed via data/qasmbench/dnn_n8.qasm: cache grows by 36 total
    across only 17 growing calls, stalling the old scheme at 47%). Both
    synthesizers cache by angle/matrix key, so on circuits with a lot of
    repeated rotations (QFT-family circuits, say) the vast majority of calls
    are instant cache hits -- with plain block counting, progress used to
    race through the first few percent (the actual searches) then jump
    straight to 100% on the cache hits, rather than tracking real elapsed
    time.  `total` (from estimate_synthesis_calls) is only an ESTIMATE of how
    many new cache entries will appear -- e.g. CyclosynthSynthesizer's
    catch-None gridsynth fallback can also legitimately not grow `cache` at
    all for a block the estimate counted as a future cyclosynth call -- so
    `count` reaching `total` exactly is not guaranteed; it is clamped at
    `total` (never shown over 100%) but completion is never inferred from
    reaching it.
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
            # Percentage only, not "N/M": M's meaning (distinct new cache
            # misses -- see estimate_synthesis_calls) differs enough between
            # backends that showing the raw counts invited comparing them
            # directly across backends, which isn't meaningful. Trailing
            # spaces clear any leftover characters from a longer previous
            # update; the string only grows as the percentage gains digits,
            # so this is defensive padding, not required alignment.
            _progress_line(log, label, count * 100 // total)
            last_report = now
        return result

    return wrapped


def count_resynthesis_blocks(circuit: QuantumCircuit) -> int:
    """How many times rewrite_single_qubit_runs will call its callback on
    `circuit`, without doing anything else -- only 2x2 matrix merges, no
    gridsynth/cyclosynth/shorten_run calls, so this is cheap even for large
    circuits. Used as the denominator for _with_block_progress, unlike
    estimate_synthesis_calls (which counts cache misses specifically for the
    main resynthesis pass) -- shorten_run has no such cache-hit/cache-miss
    split to weight by (see _with_block_progress), so a plain block count is
    the right denominator for it.
    """
    count = 0

    def counter(matrix, run):
        nonlocal count
        count += 1
        return None

    rewrite_single_qubit_runs(circuit, counter)
    return count


def _with_block_progress(
    callback, total: int, log, label: str, round_num: int, max_rounds: int
):
    """Wrap a rewrite_single_qubit_runs callback to report percentage
    progress via _progress_line, the same time-throttle scheme as
    _with_progress -- but counting every call as one unit of work, unlike
    _with_progress's cache-growth weighting.

    Built for shorten_run (compile_via_resynthesis's cleanup rounds): unlike
    synthesize(), shorten_run never calls gridsynth/cyclosynth -- every call
    does similarly cheap "exact" work (a Clifford-table lookup plus a block
    collapse over the run's own gates), roughly proportional to the run's
    length, not split into rare-expensive-search vs. common-instant-cache-hit
    the way synthesize() is. So plain per-call counting doesn't have the
    "races through cache hits" distortion _with_progress was built to avoid
    -- there's no cache to watch here in the first place.

    `round_num`/`max_rounds` blend this round's own 0-100% into a single
    running "cleanup" percentage spanning all cleanup rounds, rather than
    resetting to 0% (and printing a new line) at the start of each round --
    this round contributes one `1/max_rounds` slice of the overall range,
    offset by the rounds already completed. Since a round can converge (and
    the cleanup loop break) before max_rounds is reached, this can plateau
    below 100% -- compile_via_resynthesis prints the final 100% itself, the
    same way and for the same reason _with_progress does for the main pass.
    """
    if total == 0:
        return callback
    count = 0
    last_report = 0.0

    def wrapped(matrix, run):
        nonlocal count, last_report
        result = callback(matrix, run)
        count = min(count + 1, total)
        now = time.monotonic()
        if now - last_report >= PROGRESS_INTERVAL_SECONDS:
            round_pct = count * 100 // total
            overall_pct = ((round_num - 1) * 100 + round_pct) // max_rounds
            _progress_line(log, label, overall_pct)
            last_report = now
        return result

    return wrapped


def compile_via_resynthesis(
    unrolled: QuantumCircuit,
    synth: ResynthesisSynthesizer,
    optimize: bool = True,
    max_rounds: int = 5,
    log=lambda *a, **k: None,
) -> QuantumCircuit:
    """Re-synthesise an already-{u,cx}-unrolled circuit over Clifford+T, then clean up.

    Shared by the qiskit and cyclosynth backends -- both re-synthesise single-
    qubit runs via the same rewrite_single_qubit_runs/cancel_inverses/
    shorten_run cleanup loop, differing only in what `synth` does with a
    matrix (see CliffordTSynthesizer vs CyclosynthSynthesizer). `log` reports
    percentage progress through both the main resynthesis pass and each
    cleanup round below: shorten_run's own per-call cost is cheap (no
    gridsynth/cyclosynth search, just exact rewrites -- see
    _with_block_progress), but on a large enough circuit the sheer number of
    calls across up to max_rounds full passes dominates total compile time
    just as much as the main pass does (measured on a 36-qubit, ~1.2M-gate
    QV circuit: ~28s resynthesizing, then ~16s per cleanup round for 5
    rounds -- silent before this was added).
    """
    total = synth.estimate_synthesis_calls(unrolled)
    label = "resynthesizing"
    out = rewrite_single_qubit_runs(
        unrolled,
        _with_progress(synth.synthesize, total, log, label, synth._expensive_cache()),
    )
    if total > 0:
        # Printed unconditionally, not by _with_progress detecting its own
        # last call: `total` is only an estimate of cache growth (see its
        # docstring), so the tracked count reaching it exactly isn't
        # guaranteed -- this is what actually completes the line.
        _progress_line(log, label, 100, final=True)
    if not optimize:
        return out
    # Cancelling inverses brings new gates together, which lets the next round of
    # block collapsing find more, so iterate until it stops paying off.  Inverse
    # cancellation and the block collapses are exact; the whole-run rewrite in
    # shorten_run can spend up to --tol per run, and does so at most once per
    # round, which synth.error_bound accounts for.
    cleanup_label = "cleanup"
    any_cleanup_progress = False
    for round_num in range(1, max_rounds + 1):
        cost = gate_cost(out)
        round_total = count_resynthesis_blocks(out)
        any_cleanup_progress = any_cleanup_progress or round_total > 0
        shortened = rewrite_single_qubit_runs(
            out,
            _with_block_progress(
                synth.shorten_run, round_total, log, cleanup_label, round_num, max_rounds
            ),
        )
        out = cancel_inverses(shortened)
        if gate_cost(out) >= cost:
            break
    if any_cleanup_progress:
        # Printed unconditionally for the same reason the main pass's 100% is
        # (see above): the loop can break before max_rounds is reached, in
        # which case _with_block_progress's blended percentage plateaus
        # below 100% on its own.
        _progress_line(log, cleanup_label, 100, final=True)
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
    trbo_flag: bool = False,
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

    trbo_flag (--bqskit-trbo) inserts an extra stage right after the second
    RoundToDiscreteZPass/UnfoldPass pair, where the circuit is genuinely
    Clifford+Rz -- see the module docstring's bqskit section for what it does
    and the measurements behind making it opt-in. Applies uniformly whether
    decompose_rz ends up True or False below, so it combines correctly with
    --bqskit-inline-decompose-rz either way. Its own optimisation error
    (bounded per block by success_threshold=synthesis_epsilon) is not
    captured by this script's error_bound reporting, for the same reason
    RoundToDiscreteZPass's rounding isn't (see decompose_rz_tracked's
    docstring): it doesn't run inside a calculate_error_bound=True
    ForEachBlockPass reachable from a request_data=True compile() call.
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
    if trbo_flag:
        if trbo is None:
            raise RuntimeError(
                "--bqskit-trbo needs the trbo package, which is not "
                "installed. pip install git+https://github.com/WolfLink/trbo "
                "(see requirements.txt)."
            )
        passes += [
            QuickPartitioner(TRBO_PARTITION_SIZE),
            ForEachBlockPass(
                [
                    # TRbOPass rejects any gate outside Clifford+T+Rz, including
                    # IdentityGate -- which RoundToDiscreteZPass/clifford_replace
                    # can leave behind (see compile_bqskit's own identical cleanup
                    # for the same gate, done later for a different reason: bqskit
                    # serialises it as a custom gate the transpile binary chokes
                    # on). Strip it per-block first rather than once for the whole
                    # circuit, since QuickPartitioner already needs to run first.
                    trbo.utils.RemoveGatePass(IdentityGate(1)),
                    trbo.utils.AppendGatePass(trbo.clift.GlobalPhaseGate()),
                    trbo.trbo.TRbOPass(success_threshold=synthesis_epsilon),
                    trbo.utils.RemoveGatePass(trbo.clift.GlobalPhaseGate()),
                ]
            ),
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
    circuit: Circuit, synthesis_epsilon: float = EPSILON_DEFAULT
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
    trbo_flag: bool = False,
) -> tuple[QuantumCircuit, Optional[float]]:
    """Compile an already-{u,cx}-unrolled circuit via bqskit, returning a qiskit
    QuantumCircuit (round-tripped through qiskit's own loader, so it can share
    verification/reporting/writing with the qiskit backend) and an error bound.

    The error bound is bqskit's own ``calculate_error_bound`` mechanism, read
    from ``decompose_rz_tracked`` -- see its docstring for exactly what it
    covers. Only available when ``use_custom_rz_decomposition`` is True;
    ``None`` otherwise, since bqskit's own ``decompose_rz=True`` path (used
    when it is False) has no equivalent tracking. ``trbo_flag`` (--bqskit-trbo)
    is passed straight through to build_bqskit_workflow -- see its docstring.
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
                    trbo_flag=trbo_flag,
                ),
                BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL,
                "circuit",
            )
        # TRbO's own MatrixDistanceCost.get_grad (trbo/tcount.py) computes
        # (1 - frac**degree)**(1/degree - 1), a genuine 0**negative
        # singularity whenever a candidate's fidelity to the target rounds to
        # exactly 1.0 in floating point -- a symptom of the optimiser having
        # already converged, not an error. That produces an inf (line 56:
        # "divide by zero encountered in power"), which the very next line's
        # p1 * p2 * p3 (line 59: "invalid value encountered in multiply") can
        # turn into inf * 0 = nan whenever some parameter's own gradient
        # contribution (p3) happens to vanish for that entry -- same root
        # cause, one line downstream. numpy warns rather than raises for
        # either by default, and both surface from inside bqskit's own worker
        # processes, where a plain warnings.filterwarnings() call here has no
        # effect (confirmed empirically: those workers do not inherit this
        # process's warnings filters). Only PYTHONWARNINGS, read by each
        # worker at its own interpreter startup, reliably reaches them.
        old_pythonwarnings = os.environ.get("PYTHONWARNINGS")
        if trbo_flag:
            suppress_trbo_warning = ",".join(
                [
                    "ignore:divide by zero encountered in power:RuntimeWarning:trbo.tcount",
                    "ignore:invalid value encountered in multiply:RuntimeWarning:trbo.tcount",
                ]
            )
            os.environ["PYTHONWARNINGS"] = (
                f"{old_pythonwarnings},{suppress_trbo_warning}"
                if old_pythonwarnings
                else suppress_trbo_warning
            )
        try:
            bq_circuit = bqskit_compile(
                bq_circuit,
                model=machine,
                optimization_level=BQSKIT_WORKFLOW_OPTIMIZATION_LEVEL,
                seed=seed,
            )
        finally:
            if trbo_flag:
                if old_pythonwarnings is None:
                    os.environ.pop("PYTHONWARNINGS", None)
                else:
                    os.environ["PYTHONWARNINGS"] = old_pythonwarnings
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
    reports resynthesis progress for the qiskit/cyclosynth backends (see
    compile_via_resynthesis); windowed_fidelity doesn't pass one, since a
    window is small enough that per-window progress would just be noise.
    """
    if args.backend in ("qiskit", "cyclosynth"):
        if synth is None:
            synth = (
                CliffordTSynthesizer(epsilon=args.epsilon, tol=args.tol)
                if args.backend == "qiskit"
                else CyclosynthSynthesizer(
                    epsilon=args.epsilon, tol=args.tol, threads=args.cyclosynth_threads
                )
            )
        compiled = compile_via_resynthesis(
            unrolled, synth, optimize=not args.no_optimize, log=log
        )
        # CliffordTSynthesizer has no gridsynth-fallback concept of its own
        # (gridsynth *is* its synthesis) -- getattr rather than a shared
        # counter so this stays 0/absent for that backend without needing a
        # dead always-zero attribute on it.
        n_fallback = getattr(synth, "n_gridsynth_fallback", 0)
        fallback_note = f", {n_fallback} gridsynth-fallback" if n_fallback else ""
        extra = [
            f"1q runs: {synth.n_clifford} Clifford, {synth.n_exact} exact,"
            f" {synth.n_approx} approximated, {synth.n_merged} shortened{fallback_note}"
            f" (worst rewrite {synth.max_error:.2e}, total {synth.error_bound:.2e})"
        ]
        return compiled, extra, synth.error_bound, synth.n_approx + synth.n_merged
    compiled, error_bound = compile_bqskit(
        unrolled,
        epsilon=args.epsilon,
        seed=args.seed,
        use_custom_rz_decomposition=not args.bqskit_inline_decompose_rz,
        trbo_flag=args.bqskit_trbo,
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
        compiled_window, _, _, _ = compile_dispatch(unroll_to_u_cx(window, args.epsilon), args)
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
    """Print a report line identical in shape for all three backends.

    `extra` carries the "1q runs: N Clifford, ..." diagnostic (from
    CliffordTSynthesizer's or CyclosynthSynthesizer's counters); it reflects
    bookkeeping internal to compile_via_resynthesis's own pipeline that
    bqskit's compiler passes don't expose, so it is empty for the bqskit
    backend.
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
        description="Compile a QASM circuit to the Clifford+T gate set, via qiskit, "
        "bqskit, or cyclosynth.",
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
        choices=("qiskit", "bqskit", "cyclosynth"),
        default="qiskit",
        help="compilation backend: qiskit (fewer T gates at a given --epsilon "
        "on every benchmark measured so far) or bqskit (kept for comparison "
        "and as an independent implementation; rejects circuits with "
        "classical control flow) or cyclosynth (near-T-optimal per single-qubit "
        "block via a diamond-distance lattice search, at some compile-time "
        "cost; needs cyclosynth/ built separately, see cyclosynth/README.md)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help="gridsynth/cyclosynth target error per rotation, shared by all "
        "three backends -- see module docstring's \"Rotation synthesis: "
        "gridsynth\" section for the measurements behind this number",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="qiskit and cyclosynth backends only: skip the post-synthesis "
        "clean-up (inverse cancellation and exact re-synthesis of collapsible "
        "runs)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        help="qiskit and cyclosynth backends only: how much error an exact "
        "rewrite may introduce: the tolerance for treating an angle as a "
        "multiple of pi/4 or a unitary as a Clifford. Defaults to --epsilon, "
        "so that no rotation comes out free unless it is within the requested "
        "accuracy of a Clifford",
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
        "--bqskit-trbo",
        action="store_true",
        help="bqskit backend only: after RoundToDiscreteZPass, run TRbO "
        "(numerical joint optimization of remaining Rz angles -- see "
        "https://arxiv.org/abs/2603.25101) before isolating/synthesizing "
        "what's left. Off by default: it is a real but circuit-dependent "
        "T-count win (measured 0%% on QFT-family/Hubbard circuits, ~5%% on "
        "Haar-random circuits like QV) bought with real wall-clock cost "
        "(tens of seconds per few hundred partition blocks at default "
        "settings), unlike this script's other bqskit-workflow passes, "
        "which are unconditional. Requires the optional trbo package (see "
        "requirements.txt). Unlike every other part of this script's bqskit "
        "backend, T-counts are not reproducible run to run at a fixed "
        "--seed: TRbO's own multi-start search dispatches its retries "
        "through bqskit's runtime independently of this script's seeding, "
        "so repeated runs of the same input have been observed to differ "
        "by roughly 1%% in T count.",
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
        "--cyclosynth-threads",
        type=int,
        default=None,
        help="cyclosynth backend only: threads for cyclosynth's own lattice "
        "search (sets RAYON_NUM_THREADS). Defaults to rayon's own default "
        "(all cores) for speed; cyclosynth has no per-call seed, so its "
        "results (both the exact word and the overall T-count) are only "
        "reproducible run to run at --cyclosynth-threads 1, which was "
        "measured at this repo's EPSILON_DEFAULT to cost about 15x the "
        "compile time of the multi-threaded default (see "
        "CyclosynthSynthesizer's docstring) -- pin it to 1 when comparing "
        "exact T-counts matters more than speed.",
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

    # Warn (not error) if a backend-inappropriate flag was explicitly set to a
    # non-default value, so all three backends can share one parser without
    # silently ignoring a flag the user thought they were setting.
    backend_only_flags = {
        "qiskit": {
            "bqskit_inline_decompose_rz": "--bqskit-inline-decompose-rz",
            "bqskit_trbo": "--bqskit-trbo",
            "seed": "--seed",
            "cyclosynth_threads": "--cyclosynth-threads",
        },
        "cyclosynth": {
            "bqskit_inline_decompose_rz": "--bqskit-inline-decompose-rz",
            "bqskit_trbo": "--bqskit-trbo",
            "seed": "--seed",
        },
        "bqskit": {
            "no_optimize": "--no-optimize",
            "tol": "--tol",
            "cyclosynth_threads": "--cyclosynth-threads",
        },
    }
    for dest, flag in backend_only_flags[args.backend].items():
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
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
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
    log = (lambda *a, **k: None) if args.quiet else lambda *a, **k: print(*a, flush=True, **k)

    synth: Optional[ResynthesisSynthesizer] = None
    if args.backend == "qiskit":
        synth = CliffordTSynthesizer(epsilon=args.epsilon, tol=args.tol)
        log(f"backend: qiskit (epsilon={args.epsilon:g})")
    elif args.backend == "cyclosynth":
        synth = CyclosynthSynthesizer(
            epsilon=args.epsilon, tol=args.tol, threads=args.cyclosynth_threads
        )
        threads_desc = args.cyclosynth_threads if args.cyclosynth_threads else "rayon default"
        log(f"backend: cyclosynth (epsilon={args.epsilon:g}, threads={threads_desc})")
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
                    unrolled = unroll_to_u_cx(circuit, args.epsilon)

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
            "runs_gridsynth_fallback": getattr(synth, "n_gridsynth_fallback", None) if synth else None,
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
