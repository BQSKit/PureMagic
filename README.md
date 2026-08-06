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

This produces four binaries:

| Binary | Path | Description |
|--------|------|-------------|
| `puremagic` | `target/release/puremagic` | Lattice surgery scheduler |
| `transpile` | `target/release/transpile` | Clifford+T QASM → `.trans` transpiler |
| `circuit_stats` | `target/release/circuit_stats` | Estimate circuit statistics and layer/volume bounds without full scheduling |
| `gen_circuit` | `target/release/gen_circuit` | Generate random T-gate circuits for benchmarking |

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
| `-S, --sides-only` | off | Use only side edges of data patches (not top/bottom) |
| `-F, --no-t-failures` | off | Disable T gate failures (every T gate succeeds on first attempt) |
| `-a, --ancilla-rows <N>` | `1` | Number of ancilla rows between data patches (magic routing only) |
| `-l, --log-scheduler <LEVEL>` | `none` | Scheduler trace log level: `none`, `info`, or `debug` |
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

### Step 1 — Compile to Clifford+T (Python, optional)

If your circuit is not already in the Clifford+T gate set, compile it first using
[`data_processing/compile_cliffordt.py`](data_processing/compile_cliffordt.py):

```bash
# Install Python dependencies once
pip install -r requirements.txt

python data_processing/compile_cliffordt.py circuit.qasm
```

Outside the dev container, also set `PYTHONPATH` to this repo's `bqskit/` directory before
running the script (every backend imports `bqskit` unconditionally, not just `--backend bqskit`):

```bash
export PYTHONPATH="$(pwd)/bqskit"
```

Without it, `from bqskit import Circuit` fails with `ImportError: cannot import name 'Circuit'
from 'bqskit' (unknown location)`: `bqskit-ft`'s own install lands in `site-packages/bqskit/ft/`,
which -- without this repo's own `bqskit/` on `sys.path` ahead of it -- Python resolves as a bare
namespace package instead of the editable clone that actually defines `Circuit`. The dev container
sets this automatically (see `.devcontainer/devcontainer.json`'s `containerEnv`); outside it, it's
on you.

This produces `circuit.cliffordt.qasm`. Three backends are available via `--backend`:

| Backend | Description |
|---|---|
| `qiskit` (default) | Resynthesises each maximal single-qubit run via a ZXZ decomposition, gridsynthing each Rz rotation with qiskit's own Rust Ross-Selinger implementation (falling back to [`pygridsynth`](https://github.com/inmzhang/pygridsynth) for the rare angle it panics on). Fastest option, with low T gate counts compared to default `bqskit`. |
| `bqskit` | Compiles via [BQSKit](https://bqskit.readthedocs.io/) with a custom workflow built around its `ScanningGateRemovalPass`, which gives this backend an edge over the other two specifically on QFT-family circuits. Rejects circuits with classical control flow (BQSKit's own `Circuit` has no concept of it) -- `qiskit` and `cyclosynth` are the only options for those. Has two optional, off-by-default extra stages (see below). |
| `cyclosynth` | Shares the `qiskit` backend's resynthesis pipeline, but re-synthesises each single-qubit block by handing its full ZYZ Euler-angle triple to [cyclosynth](https://github.com/mtweiden/cyclosynth) in one call, rather than gridsynth-ing up to three rotations independently. Usually fewer T gates than either of the other two, at real but not prohibitive compile-time cost. Needs the `cyclosynth` git submodule built locally (`git submodule update --init` then see `cyclosynth/README.md`) -- not required for the other two backends. |

The `bqskit` backend's two extra stages compose with its base workflow and with each other, rather
than being alternatives to pick between:

- `--bqskit-trbo` runs [TRbO](https://github.com/WolfLink/trbo) (T Reduction by Optimization) right
  before final synthesis, numerically re-optimising a partitioned block's Rz angles jointly so more
  of them round to Clifford/T for free. A real T-count win on circuits with genuine gauge freedom
  (e.g. Haar-random-like circuits); no help on circuits whose angles are already exactly
  deduplicated (e.g. QFT). Needs the optional `trbo` package (see `requirements.txt`).
- `--bqskit-cyclosynth` replaces `bqskit`'s own per-Rz gridsynth final-synthesis stage with
  cyclosynth's joint per-block search (see the `cyclosynth` backend above), falling back to
  gridsynth for the handful of blocks cyclosynth's search can't safely handle. Needs the same
  `cyclosynth` submodule as the `cyclosynth` backend.

TRbO reduces how many rotations need synthesis at all; cyclosynth only changes how whatever
residual rotations are left get synthesised -- so `--backend bqskit --bqskit-trbo
--bqskit-cyclosynth` combines all three backends' individual strengths (QFT structure, gauge-freedom
rounding, and joint-block synthesis) into a single compile, rather than requiring a different
backend per circuit family.

Any backend's output works as input to Step 2. A basis check and error bound are always
reported; run with `--help` to see all options, including `--verify` (fidelity checks against the
input -- exact, single-statevector, or automatic random-window sampling, whichever the circuit's
size allows) and `--stats` (per-circuit JSON statistics, including per-stage timings).

**Acknowledgements**: the `--bqskit-trbo` stage runs [WolfLink/trbo](https://github.com/WolfLink/trbo),
implementing the technique described in Marc Davis's ["T Count as a Numerically Solvable Optimization
Problem"](https://arxiv.org/abs/2603.25101). The `cyclosynth` backend and `--bqskit-cyclosynth` stage
run [mtweiden/cyclosynth](https://github.com/mtweiden/cyclosynth), implementing the lattice-search
algorithm of [Morisaki et al.](https://arxiv.org/abs/2510.05816).

### Step 2 — Transpile to `.trans` format (Rust)

Convert the Clifford+T QASM file to the Pauli product `.trans` format using the `transpile` binary:

```bash
./target/release/transpile -i circuit.cliffordt.qasm
```

This produces `circuit.trans`, which is the correct input format for `puremagic`.

Run `./target/release/transpile --help` to see all options, including `--max_width` to limit Pauli product weight.

## Output Files

After scheduling, the following files are produced. Throughout the output, **lcycle** refers to one unit of parallel scheduling time, which is a single logical cycle in which all non-conflicting Pauli products that can be routed simultaneously are executed together.


| File | Contents |
|------|----------|
| `<name>.circuit.txt` | Circuit layer and dependency information. Debug builds only. |
| `<name>.sched_trace` | Detailed scheduling trace (requires `--log-scheduler info` or `debug`). |
| `<name>.schedule` | Final schedule (lcycle → operations). |
| `<name>.topo.png` | Topology visualization (requires `--plot topo`). |
| `<name>.topo.txt` | Topology grid dump. Debug builds only. |
| `<name>.circuit/` | Circuit layer plots as PNGs in a subdirectory (requires `--plot circuit`). |
| `<name>.layer_stats.svg` | Circuit layer statistics (requires `--plot cstats`). |
| `<name>.qubit_coupling.svg` | Qubit coupling matrix heatmap (requires `--plot coupling`). |
| `<name>.paths/` | Per-lcycle path visualizations (requires `--plot paths`). |

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

## Project Structure

```
src/
├── puremagic.rs        # CLI entry point and argument parsing (puremagic binary)
├── transpile.rs        # CLI entry point for transpiler (transpile binary)
├── circuit_stats.rs    # CLI entry point for circuit statistics estimator (circuit_stats binary)
├── gen_circuit.rs      # CLI entry point for random circuit generator (gen_circuit binary)
├── tableau.rs          # Clifford tableau simulation used by transpile
├── scheduler.rs        # Core EAF scheduling algorithm
├── cultivation.rs      # Magic state cultivation pool management
├── astar.rs            # A* pathfinding (single-qubit T gate routing)
├── steinertree.rs      # Steiner tree computation (greedy multi-source BFS)
├── treegraph.rs        # Steiner tree subgraph node representation
├── circuit.rs          # Circuit DAG: products, layers, dependencies
├── pauliproduct.rs     # Pauli product operations and gate types
├── node.rs             # Node type definitions (Magic, Bus, Data)
├── topograph.rs        # Topology graph: lattice layout and qubit placement
├── topograph_plotter.rs # SVG/PNG topology and path visualizations
└── utils.rs            # Timing utilities and logging macros

data_processing/        # Python/shell tooling; not built by cargo
├── compile_cliffordt.py     # Compile QASM → Clifford+T QASM (--backend qiskit/bqskit/cyclosynth,
│                                #   plus --bqskit-trbo/--bqskit-cyclosynth for the bqskit backend)
├── mcmc_cultivation.py      # MCMC model of the cultivation time distribution,
│                                #   fitted to Figure 15 of arXiv:2409.17595
├── flasq_lower_bound.py     # FLASQ lower bound (Algorithm 1 of Beverland et al.)
├── dascot_qubit_count.py    # Logical qubit counts for DASCOT square-sparse/compact
├── analyze_wisq.py          # Summarise WISQ JSON: steps, total and mean IDs per step
├── generate_rnd_layout.py   # Generate random Pauli product sequences
├── convert_lss_to_trans.py  # Lattice Surgery Simulator output → .trans
├── convert_lss_to_qasm.py   # Lattice Surgery Simulator output → QASM
├── circuit_table.py         # LaTeX table of circuit statistics from QASM files
├── scheduling_table.py      # LaTeX table comparing two scheduling runs
├── plot_puremagic.py        # Unified results plotter
├── plot_cultivation_dist.py # Plot *.cultivation_dist files as distributions
├── plot_all.sh              # Driver: regenerate the paper figures via plot_puremagic.py
└── .gitignore               # Ignores the *.png this directory generates

data/                   # Benchmark circuits, selected from Benchpress; see data/README.md
├── all_compiled/       # Source QASM plus its .cliffordt.qasm, and count-gates.sh
└── transpiled/         # .trans inputs for puremagic, under max-weight-0/ and max-weight-1/

tests/
├── integration_test.rs # End-to-end tests over the fixtures below
└── fixtures/           # tiny.cliffordt.qasm, tiny.trans, small_4q.trans
```

Build and CI configuration lives in `Cargo.toml`, `Cargo.lock`, `build.rs`,
`.cargo/config.toml`, `.rustfmt.toml` and `.github/workflows/ci.yml`; Python
dependencies are in `requirements.txt` and the dev container in
`.devcontainer/devcontainer.json`. Git metadata is `.gitignore` and
`.gitattributes`, which declares Git LFS filters for `data/**/*.qasm` and
`data/**/*.trans`.
