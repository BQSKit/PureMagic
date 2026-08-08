# Circuits

Eleven circuits spanning quantum chemistry/physics simulation, optimization, machine learning,
reversible arithmetic, and a structured transform -- chosen to stress different parts of the
Clifford+T compilation pipeline (`compile_cliffordt`) and the lattice-surgery scheduler
(`puremagic`) in different ways. The property that matters most for the compiler is how many
*distinct* non-Clifford rotation angles a circuit has relative to its gate count: few distinct
angles reused many times gives the phase-polynomial merge and gauge-collapse stages real work to
do; many (or all) distinct angles means those stages have nothing to exploit and the circuit
instead stresses raw per-block synthesis throughput. `cdkm_ripple_carry_adder` is the outlier: it
has no rotation gates at all, and instead stresses exact Toffoli decomposition and a long serial
dependency chain.

| Circuit | Qubits | Gates | Pattern | Distinct rotation angles |
|---|---|---|---|---|
| `qft_n160.qasm` | 160 | ~63,900 | Quantum Fourier Transform | 95 dyadic `u1(pi/2^k)` angles reused ~64,000 times |
| `fermi_hubbard_1d_128q.qasm` | 128 | ~542,900 | 1D Fermi-Hubbard model, Trotterized | 11 distinct `rz` angles reused ~371,700 times |
| `square_heisenberg_N225.qasm` | 225 | ~17,900 | 2D Heisenberg model, Trotterized | 92% of `rz` gates are the identical `rz(1.0)` |
| `qaoa_barabasi_albert_N149_3reps.qasm` | 149 | ~3,400 | QAOA MaxCut, 3 layers | only 3 distinct `rz` angles, one per layer |
| `ising_n98.qasm` | 98 | ~1,200 | Disordered Ising model, Trotterized | 195 distinct angles (per-edge random couplings) |
| `grover_n100_i10.qasm` | 100 | ~478,500 | Grover search, 10 iterations | ~90% of gates already Clifford+T native; the rest cluster into near-duplicate floats |
| `qv_N064_12345.qasm` | 64 | ~88,100 | Quantum Volume random circuit, seed 12345 | 13,150 distinct angles -- essentially none repeat |
| `cdkm_ripple_carry_adder_indep_opt0_200.qasm` | 200 | ~600 | Ripple-carry adder (reversible arithmetic) | none -- pure Clifford (`cx`/`ccx`), no rotations at all |
| `knn_n129.qasm` | 129 | ~200 | Quantum k-NN classifier (swap test) | 128 distinct `ry` angles -- continuous data encoding |
| `dnn_n51.qasm` | 51 | ~440 | Quantum neural network | ~all-distinct trained angles |
| `qugan_n111.qasm` | 111 | ~1,200 | Quantum GAN | 329 distinct trained angles |

## `qft_n160.qasm` -- Quantum Fourier Transform

The textbook `h` + controlled-phase ladder on 160 qubits. Every phase angle is a dyadic fraction of
pi determined only by the separation between the two qubits it acts on, so the same ~95 angles
(`pi/16` down to `pi/140737488355328`) repeat across the whole circuit. This is the best possible
case for the pipeline's phase-polynomial merge and gauge-collapse stages: disabling them lets a huge
number of otherwise-identical small-angle `Rz` rotations go unmerged, and T-count blows up. Taken
from `data/qasmbench/qft_n160.qasm`, part of [QASMBench](https://github.com/pnnl/QASMBench).

## `fermi_hubbard_1d_128q.qasm` -- 1D Fermi-Hubbard model simulation

Trotterized time evolution of the Fermi-Hubbard model on a real 1D lattice of 128 qubits: `h`-flanked
`Rz` rotations implementing the hopping and interaction terms of the Jordan-Wigner-mapped fermionic
Hamiltonian. Extreme angle reuse -- only **11 distinct `Rz` angles across all ~371,700 `Rz` gates**
in the circuit, since every hopping/interaction term of the same type at the same Trotter step
shares an identical coupling angle. Even more favorable to phase-merge than `square_heisenberg`, and
with no dead (`rz(0.0)`) gates at all, unlike the ad hoc 18-qubit Hubbard circuit this one replaced.
Taken from [MQT Bench](https://www.cda.cit.tum.de/mqtbench/).

## `square_heisenberg_N225.qasm` -- 2D Heisenberg model simulation

Trotterized time evolution of the Heisenberg spin model on a 15x15 square lattice (225 qubits).
2,520 of its 2,745 `Rz` gates (92%) are the identical `rz(1.0)` -- one shared coupling/Trotter-step
angle applied across nearly every nearest-neighbor interaction term. Similar in spirit to QFT (huge
merge/gauge-collapse payoff) but arising from lattice-physics structure rather than a phase-
polynomial ladder. Also this repo's primary physics workhorse on the *scheduler* side -- used
throughout `data_processing/paper_figs.sh` and `run.sh` for scheduling-weight sweeps. Taken from
`data/benchpress/square_heisenberg_N225.qasm`, one instance of a full `N4`..`N225` size family from
[Benchpress](https://github.com/Qiskit/benchpress).

## `qaoa_barabasi_albert_N149_3reps.qasm` -- QAOA MaxCut

QAOA (3 repetitions) for MaxCut on a 149-node Barabasi-Albert scale-free random graph. Despite the
random graph structure, the circuit has only **3 distinct `Rz` angles in the entire circuit** -- one
cost-Hamiltonian angle per QAOA layer, shared identically across every edge since the graph is
unweighted. About as favorable a case for phase-polynomial merging as exists: nearly every diagonal
rotation in a layer is an exact duplicate. Taken from
`data/benchpress/qaoa_barabasi_albert_N149_3reps.qasm`, one of a `N9`..`N149` family from
[Benchpress](https://github.com/Qiskit/benchpress).

## `ising_n98.qasm` -- Disordered Ising model simulation

Trotterized transverse-field Ising model on 98 qubits: an initial layer of `h` on every qubit,
followed by two-qubit `ZZ` interaction terms with disordered (effectively random) couplings -- 195
distinct angles across the circuit. Angles look unstructured at a glance, but enough of them repeat
(shared coupling values recurring across the lattice) that disabling phase-merge still measurably
regresses T-count. Taken from `data/qasmbench/ising_n98.qasm`, part of
[QASMBench](https://github.com/pnnl/QASMBench).

## `grover_n100_i10.qasm` -- Grover search

Grover's algorithm on a 100-qubit search register, 10 oracle+diffusion iterations, generated by this
repo's own [`data/gen_grover.py`](../gen_grover.py) (`--qubits 100 --iterations 10`; iteration count
is chosen independently of qubit count rather than derived from the textbook ~pi/4*sqrt(2^n)
formula, since only the oracle/diffusion *shape* matters for stress-testing, not a realistic
amplification count). The oracle's multi-controlled Z is transpiled down into a long Toffoli-style
chain; unlike every other circuit here, roughly 90% of the resulting ~478,500 gates (`cx`/`t`/`tdg`/
`x`/`h`) already sit natively in the Clifford+T target set, needing no resynthesis at all. The
remaining ~46,000 `u3` rotations are also unusual: many come in near-duplicate pairs that agree to
6+ significant digits before diverging (independent floating-point evaluations of what's
mathematically the same angle at different circuit locations) rather than either exact repeats or
genuinely unrelated random values -- a case phase-merge can't dedupe by exact equality alone.

## `qv_N064_12345.qasm` -- Quantum Volume random circuit

Qiskit's standard Quantum Volume construction on 64 qubits (seed `12345`): random SU(4) blocks on
randomly paired qubits per layer, decomposed here into `Rz`/`Rx(pi/2)` Euler triples. The purest
"no structure" case in this set -- 13,150 distinct `Rz` angles among ~49,000 total, i.e. essentially
every rotation is a unique random real number with zero repetition. Deliberately included as the
negative control for phase-polynomial merging: only local windowed resynthesis can help here, never
merge/dedup. Same naming and seed convention (`qv_N<size>_12345`) as the `N008`/`N036`/`N100`
instances taken from `data/benchpress/` (this specific `N064` size was generated separately for this
repo's own ablation studies using the same [Benchpress](https://github.com/Qiskit/benchpress)
Quantum Volume generator and seed).

## `cdkm_ripple_carry_adder_indep_opt0_200.qasm` -- Reversible ripple-carry adder

A 200-qubit (99+99 data bits plus carry-in/carry-out) ripple-carry integer adder, built from the
Cuccaro/CDKM `MAJ`/`UMA` gate macros (each a fixed 3-gate `cx`/`cx`/`ccx` or `ccx`/`cx`/`cx`
sequence). The only circuit in this set with **no rotation gates of any kind** -- it's exactly
Clifford+Toffoli already, so the numerical instantiation/rounding stages never engage at all; what
it stresses instead is exact Toffoli-to-Clifford+T decomposition and the scheduler's handling of a
long, inherently serial carry-propagation dependency chain (each bit position's `MAJ` depends on the
previous one's output) rather than the highly parallel structure of the other circuits here. Taken
from `data/mqt/cdkm_ripple_carry_adder_indep_opt0_200.qasm`, generated by
[MQT Bench](https://www.cda.cit.tum.de/mqtbench/) (per its embedded header: MQT Bench 2.1.0, Qiskit
2.1.1, OpenQASM 3 output -- loaded directly in that format by this pipeline's QASM loader).

## `knn_n129.qasm` -- Quantum k-nearest-neighbor classifier

A swap-test circuit: `ry(theta)` amplitude-encoding rotations (one per data point, 128 distinct
angles for continuous-valued classical data) feeding a cascade of `cswap` gates into a single
ancilla, measured at the end. No two rotation angles are meaningfully related, so there's no
merge/dedup opportunity at all -- this one stresses raw single-qubit rotation synthesis volume
instead. Taken from `data/qasmbench/knn_n129.qasm`, part of
[QASMBench](https://github.com/pnnl/QASMBench).

## `dnn_n51.qasm` -- Quantum neural network

A parameterized quantum neural network: layers of custom `ryy(theta)` two-qubit gates, each
hard-coding its own distinct trained angle (98 distinct values for a 51-qubit circuit). Like `knn`,
a fixed-parameter model with essentially no repeated-angle structure to exploit -- stresses
synthesis rather than merging. Taken from `data/qasmbench/dnn_n51.qasm`, part of
[QASMBench](https://github.com/pnnl/QASMBench).

## `qugan_n111.qasm` -- Quantum GAN

A quantum generative adversarial network circuit built from custom `cry`/`ryy` two-qubit gate
macros, each again hard-coding a distinct trained angle (329 distinct values). Same profile as
`dnn_n51` and `knn_n129`: a trained model with no shared-angle structure, exercising per-block
synthesis rather than phase merging. Taken from `data/qasmbench/qugan_n111.qasm`, part of
[QASMBench](https://github.com/pnnl/QASMBench).
