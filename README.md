# PureMagic

A **Lattice Surgery Scheduling Problem (LSSP) solver** for quantum surface code topologies. PureMagic schedules transpiled quantum circuits onto an abstract physical hardware layer, minimizing execution time by exploiting parallelism through Steiner tree packing.

The original Lattice Surgery Scheduling problem is described in the paper [Game of Surface Codes](https://arxiv.org/abs/1808.02892).

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

Run with `-h` to see the options.

### Basic Examples

```bash
# Schedule a transpiled circuit from file
./target/release/puremagic --circuit qft_n63.trans

# Use best-fit scheduling with Pure Magic routing and debug logging
./target/release/puremagic --circuit circuit.trans --best-fit --use-magic-routing --log-scheduler debug

# Generate plots of topology, circuit layers, and scheduling paths
./target/release/puremagic --circuit circuit.trans --plot topo,circuit,cstats,paths
```

## Circuit Format

Input circuits use a transpiled Pauli product format. For example, here is a 4-qubit circuit:

```
+_Z__<pi/8>      # T gate with Z on qubit 1
-_X_Y<pi/8>      # T gate with X on qubit 1, Y on qubit 3
-XZ__<CX>        # CX Clifford gate on qubits 0 and 1
+_X_Z<M>         # Measurement on qubits 1 and 3
```

Each line encodes a Pauli product with a sign (`+`/`-`), per-qubit operators (`_` for identity, `X`, `Y`, `Z`), and a gate type tag. Currently supported gates are:

```
<pi/8>  T gate
<CX>    CX Clifford gate
<S>     S/Sdg Clifford gate
<SX>    SX/SXdg Clifford gate
<M>     Measurement
```

## Output Files

After scheduling, the following files are produced:

| File | Contents |
|------|----------|
| `<name>.circuit.txt` | Circuit layer and dependency information. Debug builds. |
| `<name>.sched_trace` | Detailed scheduling trace (requires `--log-scheduler`). Debug builds. |
| `<name>.schedule` | Final schedule (timestep → operations) |
| `<name>.topo.png` | Topology visualization (requires `--plot topo`) |
| `<name>.topo.txt` | Topology file. Debug builds. |
| `<name>.layer_stats.png` | Circuit layer statistics (requires `--plot cstats`) |
| `<name>.paths/` | Per-timestep path visualizations (requires `--plot paths`) |

## Topology File Format

Topologies can be provided as a text file with node labels, grid positions, and types:

```
LABEL    X  Y  TYPE
d0       0  0  Data
d1       2  0  Data
m0       1  1  Magic
b0       1  0  Bus
```

For example, here is an 8-data qubit topology:

```
m  m  m  m  m  m  m  m  m
m  dX m  dX m  dX m  dX m
m  dZ m  dZ m  dZ m  dZ m
m  m  m  m  m  m  m  m  m
```

If no topology file is provided, one is auto-generated based on the circuit's qubit count.

## Project Structure

```
src/
├── puremagic.rs       # CLI entry point and argument parsing
├── scheduler.rs       # Core EAF scheduling algorithm
├── circuit.rs         # Circuit DAG: products, layers, dependencies
├── pauliproduct.rs    # Pauli product operations and gate types
├── topograph.rs       # Topology graph: nodes, grid layout, qubit types
├── steinertree.rs     # Steiner tree computation (greedy multi-source BFS)
├── treegraph.rs       # Tree graph structure for scheduled operation paths
├── node.rs            # Node type definitions (Magic, Bus, Data)
└── utils.rs           # Timing utilities and logging macros
```

