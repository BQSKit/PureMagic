# Correcting failed T injections

A $T$ gate in the Pauli-based circuit is a $\pi/8$ rotation about a Pauli product $P$,

$$
R_P(\pi/8) = e^{-i\pi P/8}.
$$

It is injected by measuring $P \otimes Z_m$ jointly with a magic-state patch $m$. Half the time
the rotation comes out as $R_P(-\pi/8)$, and the circuit then owes the Clifford correction
$R_P(\pi/4)$. This document covers how PureMagic used to account for that correction, why that
was wrong, and the two models that replace it.

| Flag | Values | Default |
|---|---|---|
| `--t-correction` | `litinski`, `frame` | `litinski` |
| `--reaction-latency` | $L$: lcycles from a $T$ injection until its outcome is decoded and acted on | $0$ |
| `--y-meas-lcycles` | extra lcycles for an in-place $Y$ measurement (Litinski fallback, below) | $1$ |

`-F` (no $T$ failures) skips all of this: nothing is ever owed.

## 1. The old model and why it was wrong

PureMagic recorded a failed $T$ as a per-qubit counter: for each qubit $q$ of $P$,
`correction_power[q] += 1 (mod 4)` and `correction_basis[q]` = the qubit's axis. Later $T$ gates
were relabelled qubit by qubit ($X \leftrightarrow Y$ about $Z$, and so on), power $2$ was
treated as a free Pauli, and a pending odd power was paid as a one-qubit $S$ gate before the next
Clifford on that qubit. The paper (arXiv 2512.06484, §III-E) describes the same approach. It is
wrong in two independent ways.

1. **Multi-qubit $P$.** The correction is entangling,

   $$
   R_{Z \otimes Z}(\pi/4) = e^{i\pi/4}\,\mathrm{CZ}\,(S \otimes S),
   $$

   so no product of one-qubit corrections equals it. Deferred past a later operation $Q$ that
   anticommutes with $P$, it turns $Q$ into $\propto QP$, which can add qubits: $P = Z_0 Z_1$ and
   $Q = X_0$ give $Y_0 Z_1$, a weight-2 operation. The per-qubit relabelling gives $Y_0$, on the
   wrong qubits.
2. **Two failures about different axes on one qubit**, even for weight-1 $T$ gates. The counter
   adds powers and overwrites the axis. $S_Z$ followed by $S_{-Y}$ is recorded as "power 2" and
   dropped as a Pauli, but it is a non-Pauli Clifford that cycles the axes
   ($X \mapsto -Y$, $Z \mapsto X$). A (power, axis) record can represent only 10 of the 24
   one-qubit Cliffords.

Smallest example, one qubit, both $T$ gates failing: `+Z<T> +X<T> +Z<S> +X<M>`. Exactly, the
operations run about $Z$, $-Y$, $X$, and the measurement is of $-Y$. The old model gets the first
two right up to sign, but then runs the $S$ about $Z$ and measures $X$. The two failed corrections
combine into a non-Pauli Clifford, and no correction gate is emitted for it.

The old model also charged no time for decoding an injection outcome before its correction is
decided.

## 2. Auto-corrected $\pi/8$ rotation (`--t-correction litinski`, default)

Litinski, *A Game of Surface Codes* (Quantum 3, 128, 2019), auto-corrected $\pi/8$ rotations.
With a $|0\rangle$ helper patch $h$ next to the magic patch $m$:

1. measure $P \otimes Z_m$ (the injection) and $Z_m \otimes Y_h$ in the same lcycle, then measure
   $m$ in $X$. The magic patch is free again from the next lcycle;
2. once the outcome is decoded, measure $h$ in $X$ to apply the correction $R_P(\pi/4)$, or in
   $Z$ not to.

The correction is a choice of measurement basis on an ancilla. Every outcome only adds Paulis to
the frame, so **no later product is ever changed and no $S$ gate is emitted.** Only the helper
waits.

What the scheduler does:

- **Helper.** A free neighbour of the magic patch, borrowed in the injection lcycle. A bus
  patch comes first, then the cultivating magic patch with the least progress (its cultivation
  restarts). Never a data patch or a ready magic state. With magic routing (`-u`) there are no
  bus patches, so the helper is always a cultivating magic patch. No patch is dedicated to
  helpers, so injection stays single-patch.
- **No free neighbour yet.** The injection waits up to 2 lcycles for one, holding the magic
  patch. After that, or if no neighbour could ever be a helper, the magic patch is measured in
  place instead: Gidney, *Inplace Access to the Surface Code Y Basis* (Quantum 8, 1310, 2024).
  The patch waits for the decision and is measured in $X$ (no correction) or $Y$ (applies it).
  The $Y$ measurement takes $\lfloor d/2 \rfloor + 2$ rounds, modelled as `--y-meas-lcycles`
  extra lcycles. Waiting longer for a helper could deadlock two magic patches that wait on each
  other.
- **Hold.** The helper stays occupied until the correction is decided (the decision lcycle below).
  Data-side products are never delayed: the helper's measurement acts on a different patch, so
  it commutes with everything that follows on the data.

### When is the correction decided?

A $T$'s own outcome is known $L$ lcycles after injection, where $L$ is `--reaction-latency`.
Whether a correction is owed also depends on the frame: if an earlier $T$'s Pauli byproduct
anticommutes with this $P$, it flips this rotation's sign. That byproduct comes from the earlier
$T$'s final ancilla measurement, made at that $T$'s decision and decoded $L$ later. So decisions
chain. For the $k$-th $T$ gate $T_k$,

$$
\mathrm{dec}(T_k) = \max\Big( \mathrm{inj}(T_k) + L,\; \max_{j \in A_k} \big( \mathrm{dec}(T_j) + L \big) \Big),
$$

where $\mathrm{inj}$ is the injection lcycle and $A_k$ is the set of earlier $T$ gates that share
a qubit with $T_k$ and whose Paulis anticommute with $P_k$. When $T_j$'s patch was measured in
$Y$, the $Y$ measurement time is added to $\mathrm{dec}(T_j)$.

With $L = 0$ everything is decided at once. A run of anticommuting $T$ gates, such as a
synthesized one-qubit rotation, therefore needs about $L$ lcycles per $T$. This is Litinski's
reaction limit; the old model did not charge it.

## 3. Exact Clifford frame (`--t-correction frame`)

Never apply the correction at all. $R_P(\pi/4)$ is a Clifford, so it can be folded into a frame
$W$ (`src/frame.rs`): the hardware state is $W$ applied to the ideal state, and every later $T$ or
measurement about $Q$ runs about $W Q W^\dagger$. The frame is updated by

$$
\text{failed } T \text{ about } P:\quad W \leftarrow W\, R_P(-\pi/4),
\qquad\qquad
\text{physical Clifford } G:\quad W \leftarrow G\, W\, G^\dagger .
$$

Physical $S$, $SX$ and $CX$ lines in the `.trans` run unchanged. The frame merges into
measurements for free (they stay Pauli measurements), but not into $T$ gates:

- **Supports change.** $W Q W^\dagger$ can act on more qubits than $Q$, or on fewer: after a
  failed $T$ about $Z_0 Z_1$, $X_0$ runs as $Y_0 Z_1$. A rewritten product gets new terminals, and
  a multi-qubit measurement leaves its precomputed tree.
- **Physical Cliffords spread the frame.** Each $CX$ conjugates $W$ onto more qubits, so on
  circuits with many physical Cliffords the rewritten products grow (measured below).
- **Data-side waits.** Whether a later $Q$ is rewritten depends on whether the earlier $T$ failed,
  and that is known only $L$ lcycles after injection. So a $T$ or measurement that anticommutes
  with a $T$ injected in the last $L$ lcycles waits. (With Litinski only the helper waits.)
- **No helper and no hold.** At $L = 0$ nothing waits at all.

The frame keeps signs out (only supports and letters affect scheduling). It is a
$\mathrm{GF}(2)$ tableau on 64-qubit words, so rewriting a product of weight $w$ on $n$ qubits
costs $O(w \cdot n/64)$ word operations.

## 4. Choosing `--reaction-latency`

$L$ is the time to decode an outcome and act on it, in logical cycles, rounded up. A logical
cycle is about $d$ syndrome-extraction rounds, so

$$
L = \left\lceil \frac{t_{\text{reaction}}}{d \cdot t_{\text{round}}} \right\rceil .
$$

| Setting | $t_{\text{round}}$ | $t_{\text{reaction}}$ | $L$ at $d = 25$ | Source |
|---|---|---|---|---|
| Superconducting, fault-tolerance assumption | $1\,\mu\text{s}$ | $10\,\mu\text{s}$ | $1$ ($0.4$, rounded up) | Gidney & Ekerå 2019 (RSA-2048 estimate); reused in later resource estimates |
| Superconducting, measured today | $1.1\,\mu\text{s}$ | $63\,\mu\text{s}$ mean decoder latency ($d = 5$, real time) | $3$ ($2.3$); $12$ at $d = 5$ | Google, "Quantum error correction below the surface code threshold" (Nature 2025) |
| Trapped ions, neutral atoms | $\sim 100\,\mu\text{s}$ to $\sim 1\,\text{ms}$ (mostly projections) | small next to a logical cycle | $0$–$1$ | e.g. trapped-ion lattice-surgery and neutral-atom architecture estimates |

Litinski's *Game of Surface Codes* assumes the $1\,\mu\text{s}$ code cycle; the $10\,\mu\text{s}$
reaction time is Gidney and Ekerå's. Google's $63\,\mu\text{s}$ is a mean measured at $d = 5$; a
larger-distance decoder would be slower, but the logical cycle grows with $d$ too. Use
$L = 1$ as the main setting (the standard fault-tolerance assumption) and $L = 2$–$5$ as a
sensitivity range for today's decoders.

## 5. Approximations

- **Correction needed = injection failed.** The true condition is
  $(\text{injection outcome}) \oplus (\text{frame parity})$, which is also a fair coin. The
  scheduler uses the failure coin. This matters only for the in-place fallback, where it decides
  which patches pay the $Y$ time.
- **$Z_m \otimes Y_h$ takes one lcycle**, in parallel with $P \otimes Z_m$. The helper's $X$/$Z$
  measurement and the magic patch's $X$ measurement are instantaneous (transversal).
- **No decoder backlog**: every outcome takes exactly $L$ lcycles.

## 6. Results

Five PureMagic benchmarks (cdkm_ripple_carry_adder_indep_opt0_200, dnn_n51, ising_n98,
knn_n129, qugan_n111), each transpiled at `-m 0` and `-m 1`, scheduled with `-u` at seeds 0 and
1. Lcycles are relative to the old model (`main` before this change) at the same seed. Volume
changes exactly as lcycles do (the patch count is fixed).

| circuit | litinski $L=0$ | litinski $L=1$ | frame $L=0$ | frame $L=1$ |
|---|---|---|---|---|
| cdkm_ripple_carry_adder `-m 0` | +5.9% | +7.4% | +10.4% | +10.8% |
| cdkm_ripple_carry_adder `-m 1` | +4.8% | +6.3% | +4.2% | +4.2% |
| dnn_n51 `-m 0` | +3.4% | +6.3% | +8.4% | +58.9% |
| dnn_n51 `-m 1` | +1.7% | +3.7% | +32.4% | +73.6% |
| ising_n98 `-m 0` | +1.5% | +4.8% | +0.3% | +0.0% |
| ising_n98 `-m 1` | +2.1% | +5.0% | +2.9% | +5.5% |
| knn_n129 `-m 0` | +8.5% | +11.4% | +2.4% | +2.6% |
| knn_n129 `-m 1` | −0.4% | +4.2% | +1.2% | +1.2% |
| qugan_n111 `-m 0` | +2.3% | +3.7% | +0.8% | +97.4% |
| qugan_n111 `-m 1` | +0.3% | +1.6% | +3.7% | +76.5% |
| **geomean** | **+3.0%** | **+5.4%** | **+6.3%** | **+28.4%** |

- **The helper itself is cheap.** At $L = 0$ Litinski costs +3%: it borrows a cultivating
  neighbour for one lcycle, which restarts that patch's cultivation. Its cost grows slowly with
  $L$ because only the helper waits; the data path never does.
- **Physical Cliffords spread the frame.** Every $S$/$SX$/$CX$ line conjugates $W$, so circuits
  with many of them see the frame's products grow. On cdkm_ripple_carry_adder `-m 1`, $T$ gates
  and measurements written with weight $1.00$ run with mean weight $17.9$ (max $143$).
- **Without physical Cliffords the frame can win at low latency.** There is no helper, and folding
  $P$ into an overlapping $Q$ can lower the weight (knn_n129 `-m 0`: +2.4% against Litinski's
  +8.5%).
- **Data-side waits dominate from $L = 1$.** Long runs of anticommuting $T$ gates stall the data
  in frame mode: dnn_n51 and qugan_n111 cost +59% to +97% at $L = 1$, against +2% to +6% with
  Litinski.
- **At larger $L$ the reaction limit dominates both models.** Runs of anticommuting $T$ gates
  need about $L$ lcycles per $T$, which the old model never charged. On a wider set of 24
  circuits (including dcsynth benchmarks; measured with the same models before they were ported
  to this branch) Litinski costs +4.1%, +6.7%, +25% and +74% over the old model at
  $L = 0, 1, 2, 4$, and frame +9.6%, +23%, +40% and +80%.

`litinski` is the default: it is the cheaper model at every latency from $L = 1$ up and close to
the best at $L = 0$. `frame` is the exact reference, and the better choice at $L = 0$ on circuits
with few physical Cliffords.

## 7. Verification

- Unit tests (`src/scheduler.rs`):
  - `litinski_leaves_every_later_product_unchanged`: no relabelling and no added products, over 8
    seeds;
  - `litinski_helpers_are_free_neighbours_and_holds_are_respected`: each helper is adjacent and not
    a data patch; held patches are never used as a magic state; the hold totals add up;
  - `litinski_decisions_chain_only_through_anticommuting_t_gates`;
  - `litinski_without_failures_borrows_no_helpers`;
  - `frame_mode_runs_every_product_once_and_rewrites_after_failures`;
  - `frame_mode_executes_the_conjugated_pauli`: replays each run with a fresh frame built from the
    logged failures and physical Cliffords, and checks every $T$ and measurement ran about
    $W Q W^\dagger$;
  - `frame_mode_rewrites_a_measurement_after_a_failed_t`,
    `frame_mode_waits_for_an_anticommuting_outcome`, `frame_mode_without_failures_matches_litinski`.
- `src/frame.rs` checks the tableau against a slow event-list reference on random histories (up to
  130 qubits) and its gate rules against the transpiler's signed tableau (`src/tableau.rs`).
- Frame mode against exact unitaries: 480 runs (80 random circuits of 2–4 qubits, $T$ gates and
  measurements of weight 1–3, physical $S$, $S^\dagger$, $SX$, $SX^\dagger$ and $CX$; seeds 1–3;
  $L \in \{0, 2\}$), replayed from the debug-build scheduler trace in execution order. Every
  executed $T$ and measurement matches the exact unitary calculation, and flipping the outcome of
  any $T$ not yet decoded never changes a product that already ran.
- A two-qubit circuit that shows both problems, and the one-qubit circuit above, were run with
  both models and compared with an exact Clifford-tableau calculation: every operation is executed
  on the right Pauli (the old model executed 4 and 2 operations on the wrong Pauli).
