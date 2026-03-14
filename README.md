# PureMagic

A **Lattice Surgery Scheduling Problem (LSSP) solver** for quantum surface code topologies. PureMagic schedules transpiled quantum circuits onto an abstract physical hardware layer, minimizing execution time by exploiting parallelism through Steiner tree packing.

The original Lattice Surgery Scheduling problem is described in the paper [Game of Surface Codes](https://arxiv.org/abs/1808.02892).

The implementation here follows the approach described in [Multi-qubit lattice surgery scheduling](https://arxiv.org/pdf/2405.17688v2).

The code in this repository is used for the paper [Scheduling Lattice Surgery with Magic State
Cultivation](https://arxiv.org/pdf/2512.06484).


## Building

Requires Rust (stable). Build with:

```bash
cargo build --release
```

The binary is placed at `target/release/puremagic`.

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
| `-g, --use-greedy` | off | Use faster but suboptimal greedy path algorithm instead of A* |
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

# Use the greedy path algorithm and debug logging
./target/release/puremagic --circuit circuit.trans --use-greedy --log-scheduler debug

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

Files in this format can be obtained by using Tableau in the `tableau` submodule directory, for example running:

```
python transpile_circuit.py -i circuit.qasm
```

will produce a file `circuit.trans` which is in the correct input format for `puremagic`.

Consult the `README.md` in the `tableau` directory for usage details.

After cloning the PureMagic repository, make sure to checkout the tableau submodule with

```
git submodule update --init
```

## Output Files

After scheduling, the following files are produced:

| File | Contents |
|------|----------|
| `<name>.circuit.txt` | Circuit layer and dependency information. Debug builds only. |
| `<name>.sched_trace` | Detailed scheduling trace (requires `--log-scheduler info` or `debug`). |
| `<name>.schedule` | Final schedule (timestep → operations). |
| `<name>.topo.png` | Topology visualization (requires `--plot topo`). |
| `<name>.topo.txt` | Topology grid dump. Debug builds only. |
| `<name>.layer_stats.png` | Circuit layer statistics (requires `--plot cstats`). |
| `<name>.paths/` | Per-timestep path visualizations (requires `--plot paths`). |

## Topology File Format

Topologies can be provided as a text file with node labels, grid positions, and types, with m for magic, b for bus, and d for data. The data qubits are double, and marked with X and Z. For example, here is an 8-data qubit topology:

```
m  m  m  m  m  m  m  m  m
m  dX m  dX m  dX m  dX m
m  dZ m  dZ m  dZ m  dZ m
m  m  m  m  m  m  m  m  m
```

If no topology file is provided, one is auto-generated based on the circuit's qubit count and the `ancilla_rows` option.

## Project Structure

```
src/
├── puremagic.rs       # CLI entry point and argument parsing
├── scheduler.rs       # Core EAF scheduling algorithm
├── astar.rs           # A* pathfinding (single-qubit T gate routing)
├── greedypath.rs      # Greedy path algorithm (faster alternative to A*)
├── steinertree.rs     # Steiner tree computation (greedy multi-source BFS)
├── treegraph.rs       # Tree graph structure for scheduled operation paths
├── circuit.rs         # Circuit DAG: products, layers, dependencies
├── pauliproduct.rs    # Pauli product operations and gate types
├── topograph.rs       # Topology graph: nodes, grid layout, qubit types
├── node.rs            # Node type definitions (Magic, Bus, Data)
└── utils.rs           # Timing utilities and logging macros
```
