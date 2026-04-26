#!/usr/bin/env -S cargo run --bin circuit_stats --
//! Estimate circuit statistics and layer/volume bounds from a `.trans` circuit file.
//!
//! Loads the circuit, prints statistics (products, Cliffords, layers), then
//! computes the estimated minimum number of layers and volume using the same
//! methodology as the main `puremagic` scheduler — but without running the
//! full scheduler.  The number of magic-state qubits is derived from a
//! generated topology (either bus-routing or PureMagic layout).

// SeedableRng must be in scope for StdRng::seed_from_u64 even though it
// appears unused to the compiler's name-resolution pass.
#![allow(unused_imports)]

use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[allow(dead_code)]
mod circuit;
#[allow(dead_code)]
mod node;
#[allow(dead_code)]
mod pauliproduct;
#[allow(dead_code)]
mod topograph;
#[allow(dead_code)]
mod topograph_plotter;
#[allow(dead_code)]
mod treegraph;
#[macro_use]
#[allow(dead_code)]
mod utils;

use circuit::Circuit;
use topograph::TopoGraph;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Estimate circuit statistics and volume from a .trans circuit file"
)]
struct Args {
    /// Input circuit file in .trans format (required).
    #[arg(short, long = "circuit")]
    circuit_fname: String,

    /// Lambda parameter for the exponential distribution of magic-state
    /// cultivation cycles (magic states produced per magic qubit per lcycle).
    #[arg(short, long, default_value = "0.0387396")]
    magic_state_lambda: f64,

    /// Disable T-gate failures (every T gate succeeds on the first attempt).
    /// When set, no S-correction layers are added to the layer estimate.
    #[arg(short = 'F', long)]
    no_t_failures: bool,

    /// Use bus routing instead of PureMagic routing when computing the number
    /// of magic-state qubits.  By default PureMagic (all-magic) routing is
    /// used.
    #[arg(short = 'b', long)]
    bus_routing: bool,

    /// Number of ancilla rows between data patches (PureMagic routing only).
    #[arg(short, long, default_value = "1")]
    ancilla_rows: usize,

    /// Random seed used for the stochastic layer estimate.
    #[arg(short, long, default_value = "29")]
    rseed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // ── 1. Load circuit ──────────────────────────────────────────────────────
    let mut circuit = Circuit::new(&args.circuit_fname);
    circuit.load_circuit()?;
    let n_data_qubits = circuit.n_qubits;

    // ── 2. Print circuit statistics (also returns n_layers) ──────────────────
    let n_layers = circuit.print_statistics();

    // ── 3. Build topology to get magic-qubit count ───────────────────────────
    //
    // We only need the qubit counts, not a full routing graph, so we generate
    // the default topology (no topo file, no randomisation).
    let use_magic_routing = !args.bus_routing;
    let routing_label = if use_magic_routing { "PureMagic" } else { "Bus" };

    let mut topo = TopoGraph::new();
    // rseed = 0 → no randomisation of data-qubit numbering
    let rseed: u32 = 0;
    topo.set_topo(
        n_data_qubits,
        &args.circuit_fname,
        &String::new(), // no topo file
        &rseed,
        use_magic_routing,
        args.ancilla_rows,
        false, // sides_only = false (default)
    );

    let n_qubits = topo.n_qubits;
    let n_magic_qubits = topo.n_magic_qubits;

    // ── 4. Estimate minimum layers ───────────────────────────────────────────
    //
    // estimate_num_layers expands the circuit (CX→2, S→3, T→1 + optional S
    // correction) and counts layers via topological sort.
    let mut rng = StdRng::seed_from_u64(args.rseed);
    let min_layers = circuit.estimate_num_layers(&mut rng, args.no_t_failures);

    // T-gate statistics
    let (n_t_gates, n_t_layers) = circuit.count_t_stats();
    // Clifford layers = estimated total − pure-T layers
    let n_clifford_layers = min_layers.saturating_sub(n_t_layers);

    // Magic-state throughput constraint:
    //   each magic qubit produces λ magic states per lcycle on average, so
    //   the minimum number of lcycles just to supply all T gates is:
    let max_t_parallelism = n_magic_qubits as f64 * args.magic_state_lambda;
    let magic_min_layers = if max_t_parallelism > 0.0 {
        (n_t_gates as f64 / max_t_parallelism).ceil() as usize
    } else {
        0
    };

    // Overall lower bound: whichever is larger — the circuit-depth bound or
    // the magic-state throughput bound.
    let lmin = std::cmp::max(min_layers, n_clifford_layers + magic_min_layers);
    let vmin = lmin * n_qubits;

    // ── 5. Print results ─────────────────────────────────────────────────────
    println!("Routing mode: {}", routing_label);
    println!("Layer estimates:");
    println!("  Circuit layers (DAG depth):    {}", n_layers);
    println!("  T gates:                       {}", n_t_gates);
    println!("  T layers:                      {}", n_t_layers);
    println!("  Clifford layers (est.):        {}", n_clifford_layers);
    println!(
        "  Estimated min layers (circuit depth, T-failures={}): {}",
        !args.no_t_failures, min_layers
    );
    println!(
        "  Magic-state throughput min layers (λ={:.7}, {} magic qubits): {}",
        args.magic_state_lambda, n_magic_qubits, magic_min_layers
    );
    println!("  Combined min layers (lmin):    {}", lmin);
    println!("Volume estimate:");
    println!("  lmin × total_qubits = {} × {} = {}", lmin, n_qubits, vmin);

    Ok(())
}
