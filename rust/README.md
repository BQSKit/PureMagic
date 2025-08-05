# LSSP (Local Stabilizer Simulation Protocol)

A quantum circuit transformation tool that performs Clifford operator commutations on Pauli products.

## Description

This tool reads a quantum circuit represented as a directed acyclic graph (DAG) of Pauli products and performs commutations of Clifford operators to transform the circuit into a standardized form.

## Features

- Pauli product parsing and manipulation
- DAG-based circuit representation
- Topological ordering maintenance
- Clifford operator commutation
- Progress tracking and timing
- Detailed logging

## Prerequisites

- Rust 1.70.0 or later
- Cargo (Rust's package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lssp.git
cd lssp
```

2. Build the project:
```bash
cargo build --release
```

The executable will be created at `target/release/transpile`

## Usage

### Basic Usage

```bash
cargo run --release -- <input_file>
```

or using the compiled binary directly:

```bash
./target/release/transpile <input_file>
```

### Input File Format

The input file should be a tab-separated text file with the following format:

```
id	product	children	parents
0	PauliX(+1)0<Angle(pi/2)>	[1, 2]	[]
1	PauliY(-1)1<Angle(pi)>	[3]	[0]
...
```

Each line contains:
1. Node ID (integer)
2. Pauli product description
3. List of child node IDs
4. List of parent node IDs

### Output

The program generates two files:
- `<input_file>-loaded.txt`: The circuit after initial loading
- `<input_file>-transpiled.txt`: The transformed circuit after Clifford commutations

## Development

### Running Tests

```bash
cargo test
```

### Running with Debug Output

```bash
RUST_LOG=debug cargo run -- <input_file>
```

## Performance

The tool includes performance timing for:
- Loading the circuit
- Commuting Clifford operators
- Updating topological ordering

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
