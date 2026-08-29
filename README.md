# PureMagic

A dynamic lactice surgery scheduler for quantum surface code topologies. PureMagic schedules transpiled quantum circuits onto an abstract physical hardware layer, minimizing execution time by exploiting parallelism through Steiner tree packing. It dynamically simulates the execution of the circuit, including magic state cultivation and T gate injection failures.

The original Lattice Surgery Scheduling problem is described in the paper [Game of Surface Codes](https://arxiv.org/abs/1808.02892).

The implementation here is inspired by the approach described in [Multi-qubit lattice surgery scheduling](https://arxiv.org/pdf/2405.17688v2).

The code in this repository is used for the paper [Scheduling Lattice Surgery with Magic State
Cultivation](https://arxiv.org/pdf/2512.06484).


## Building

Requires Rust (stable). Build with:

```bash
cargo build --release
```

The benchmark circuits under `data/` are tracked with [Git LFS](https://git-lfs.com/);
install it once per machine (`git lfs install`) before cloning, or run `git lfs pull`
afterwards, or those files check out as small pointer stubs instead of real QASM/`.trans`
content. Building and testing do not otherwise require it — `tests/fixtures/` is plain
git content, not LFS.

This produces five binaries:

| Binary | Path | Description |
|--------|------|-------------|
| `puremagic` | `target/release/puremagic` | Lattice surgery scheduler |
| `transpile` | `target/release/transpile` | Clifford+T QASM → `.trans` transpiler |
| `circuit_stats` | `target/release/circuit_stats` | Estimate circuit statistics and layer/volume bounds without full scheduling |
| `gen_circuit` | `target/release/gen_circuit` | Generate random T-gate circuits for benchmarking |
| `compile_cliffordt` | `target/release/compile_cliffordt` | Clifford+T compiler (Step 1 below) |

Building `compile_cliffordt` links against the `cyclosynth` git submodule unconditionally (it's a
regular Cargo path dependency, and the only rotation-synthesis backend the binary has -- used on
every run, `--cyclosynth` or not), so `cargo build --release` needs
`git submodule update --init --recursive` to have been run first.

## Usage

```
puremagic [OPTIONS] --circuit <FILE>
```

Run with `-h` to see all options.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-c, --circuit <FILE>` | *(required)* | Input circuit file (`.trans` format) |
| `-t, --topo <FILE>` | *(auto-generated)* | Topology file; generated from qubit count if omitted |
| `-m, --magic-state-lambda <F>` | `0.0387396` | λ parameter for exponential cultivation time distribution |
| `-r, --rseed <N>` | `29` | Random seed for reproducible results |
| `-R, --randomize-data-qubits` | off | Randomize data qubit numbering |
| `-u, --use-magic-routing` | off | Use magic qubits for routing in addition to bus qubits |
| `-S, --sides_only` | off | Use only side edges of data patches (not top/bottom) |
| `-F, --no-t-failures` | off | Disable T gate failures (every T gate succeeds on first attempt) |
| `-C, --record-cultivation-dist` | off | Record normalized cultivation-time distribution to `<name>.cultivation_dist` |
| `-a, --ancilla-rows <N>` | `1` | Number of ancilla rows between data patches (magic routing only) |
| `-l, --log-scheduler <LEVEL>` | `none` | Scheduler trace log level: `none`, `info`, or `debug`; only populated in debug builds |
| `-I, --show-product-ids` | off | Show product IDs instead of Pauli terms in circuit plots |
| `-p, --plot <LIST>` | *(none)* | Comma-separated plot options: `topo`, `circuit`, `coupling`, `cstats`, `paths` |

### Basic Examples

```bash
# Schedule a transpiled circuit from file
./target/release/puremagic --circuit qft_n63.trans

# Use magic routing with a specific topology file
./target/release/puremagic --circuit circuit.trans --use-magic-routing --topo my.topo.txt

# Generate plots of topology, circuit layers, and scheduling paths
./target/release/puremagic --circuit circuit.trans --plot topo,circuit,cstats,paths
```

## Circuit Format

Input circuits use a transpiled Pauli product format. For example, here is a 4-qubit circuit:

```
+_Z__<T>         # T gate with Z on qubit 1
-_X_Y<T>         # T gate with X on qubit 1, Y on qubit 3
-XZ__<CX>        # CX Clifford gate on qubits 0 and 1
+_X_Z<M>         # Measurement on qubits 1 and 3
```

Each line encodes a Pauli product with a sign (`+`/`-`), per-qubit operators (`_` for identity, `X`, `Y`, `Z`), and a gate type tag. Currently supported gates are:

```
<T>     T gate (pi/8 rotation)
<CX>    CX Clifford gate
<S>     S/Sdg Clifford gate
<SX>    SX/SXdg Clifford gate
<M>     Measurement
<Z>     Z Pauli gate
<X>     X Pauli gate
```

Files in this format are produced by the `transpile` binary. The full pipeline from a raw QASM circuit to a scheduled output is:

### Step 1 — Compile to Clifford+T (Rust, optional)

If your circuit is not already in the Clifford+T gate set, compile it first using the
`compile_cliffordt` binary:

```bash
./target/release/compile_cliffordt circuit.qasm
```

This produces `circuit.cliffordt.qasm`. Its pipeline: exact phase-polynomial merge, a
gauge-collapse cycle (blocking + exact-Clifford recognition + angle rounding), windowed
multi-qubit resynthesis, gauge collapse again, final per-block synthesis via cyclosynth --
its joint ZYZ lattice search by default, or its independent per-axis Rz synthesis with
`--skip-cyclosynth` -- then a Clifford-run simplification pass that canonicalizes the Clifford
gates synthesis leaves around each T gate (exact, no fidelity cost).

Notable flags (`--help` for the full list): `--epsilon` (default `1e-8`);
`--verify` (exact unitary fidelity check against the original circuit,
circuits of ≤10 qubits only); `--skip-cyclosynth` to fall back to independent per-axis Rz
synthesis for Stage 3; and `--skip-gauge-collapse`/`--skip-windowed-resynthesis`/
`--skip-phase-merge`/`--skip-clifford-simplify` to isolate each stage's contribution, e.g. for
ablation studies. Accepts multiple input files at once, writing `<name>.cliffordt.qasm` next to
each input by default (`-o` to override, or to name an output directory when compiling more than
one file).

Its QASM loader is deliberately narrow (this pipeline's own Clifford+`rz`/`u3` vocabulary, plus the
OpenQASM 3 subset needed for MQT Bench exports) and fails loudly on anything it doesn't recognize,
rather than silently dropping a gate. Output works as input to Step 2. A basis check and error
bound are always reported.

**Acknowledgements**: cyclosynth's joint search runs [mtweiden/cyclosynth](https://github.com/mtweiden/cyclosynth),
implementing the lattice-search algorithm of [Morisaki et al.](https://arxiv.org/abs/2510.05816).

### Step 2 — Transpile to `.trans` format (Rust)

Convert the Clifford+T QASM file to the Pauli product `.trans` format using the `transpile` binary:

```bash
./target/release/transpile -i circuit.cliffordt.qasm
```

This produces `circuit.trans`, which is the correct input format for `puremagic`.

Notable flags (`--help` for the full list): `--max_width` (default `0`, no limit) to cap Pauli
product weight -- lower weights emit more, cheaper-to-route products; `--defer_trailing` to hold
back each qubit's trailing single-qubit Clifford run at a flush instead of always emitting it,
reducing logical cycles at any weight; and `--auto`, which ignores `--max_width` and instead
transpiles at each candidate weight (always with `--defer_trailing`) and keeps the one with the
lowest predicted circuit depth.

## Output Files

After scheduling, the following files are produced. Throughout the output, **lcycle** refers to one unit of parallel scheduling time, which is a single logical cycle in which all non-conflicting Pauli products that can be routed simultaneously are executed together.


| File | Contents |
|------|----------|
| `<name>.circuit.txt` | Circuit layer and dependency information. Debug builds only. |
| `<name>.sched_trace` | Detailed scheduling trace (requires `--log-scheduler info` or `debug`). Only populated in debug builds; a release build still creates the file but leaves it empty. |
| `<name>.schedule` | Final schedule (lcycle → operations). |
| `<name>.cultivation_dist` | Normalized cultivation-time distribution (requires `--record-cultivation-dist`). |
| `<name>.topo.png` | Topology visualization (requires `--plot topo`). |
| `<name>.topo.txt` | Topology grid dump (requires `--plot topo`; not gated by build profile). |
| `<name>.circuit/` | Circuit layer plots as PNGs in a subdirectory (requires `--plot circuit`). |
| `<name>.layer_stats.svg` | Circuit layer statistics (requires `--plot cstats`). |
| `<name>.qubit_coupling.svg` | Qubit coupling matrix heatmap (requires `--plot coupling`). |
| `<name>.paths/` | Per-lcycle path visualizations, first 100 lcycles (requires `--plot paths`). |

## Topology File Format

Topologies can be provided as a text file with node labels, grid positions, and types, with m for magic, b for bus, and d for data. The data qubits are double, and marked with X and Z. For example, here is an 8-data qubit topology:

```
b  m  m  m  m  m  m  m  m  m  b
m  b  b  b  b  b  b  b  b  b  m
m  b  dX b  dX b  dX b  dX b  m
m  b  dZ b  dZ b  dZ b  dZ b  m
m  b  b  b  b  b  b  b  b  b  m
b  m  m  m  m  m  m  m  m  m  b
```

If no topology file is provided, one is auto-generated based on the circuit's qubit count and the `ancilla_rows` option.
