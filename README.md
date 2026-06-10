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

If your circuit is not already in the Clifford+T gate set, compile it first using [`src/compile_circuit.py`](src/compile_circuit.py):

```bash
# Install Python dependencies once
pip install -r requirements.txt

python src/compile_circuit.py -i circuit.qasm
```

This produces `circuit.cliffordt.qasm` using [BQSKit](https://bqskit.readthedocs.io/).

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
├── compile_circuit.py  # Python script: compile QASM → Clifford+T QASM (uses BQSKit)
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
```