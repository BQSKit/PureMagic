#!/usr/bin/env -S cargo run --bin circuit_stats --
//! Estimate circuit statistics and layer/volume bounds from a `.trans` circuit file,
//! using the same methodology as the main `puremagic` scheduler but without running it.

// SeedableRng must stay in scope for StdRng::seed_from_u64, though it looks unused.
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
use utils::CommonArgs;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Estimate circuit statistics and volume from a .trans circuit file"
)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Use bus routing instead of PureMagic routing when computing the number of
    /// magic-state qubits (default: PureMagic / all-magic routing).
    #[arg(short = 'b', long)]
    bus_routing: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    utils::print_banner("circuit_stats");

    let mut circuit = Circuit::new(&args.common.circuit_fname);
    circuit.load_circuit()?;
    let n_data_qubits = circuit.n_qubits;

    let n_layers = circuit.print_statistics();

    // Only qubit counts are needed here, not a full routing graph, so this uses the
    // default topology (no topo file, no randomisation).
    let use_magic_routing = !args.bus_routing;
    let routing_label = if use_magic_routing { "PureMagic" } else { "Bus" };

    let mut topo = TopoGraph::new();
    // rseed = 0 → no randomisation of data-qubit numbering
    let rseed: u32 = 0;
    topo.set_topo(
        n_data_qubits,
        &args.common.circuit_fname,
        &String::new(), // no topo file
        &rseed,
        use_magic_routing,
        args.common.ancilla_rows,
        false, // sides_only = false (default)
    );

    let n_qubits = topo.n_qubits;
    let n_magic_qubits = topo.n_magic_qubits;

    let mut rng = StdRng::seed_from_u64(args.common.rseed);
    let est = circuit.estimate_layer_volume(
        n_magic_qubits,
        n_qubits,
        args.common.magic_state_lambda,
        args.common.no_t_failures,
        &mut rng,
    );

    println!("Routing mode: {}", routing_label);
    println!("Layer estimates:");
    println!("  Circuit layers (DAG depth):    {}", n_layers);
    println!("  T gates:                       {}", est.n_t_gates);
    println!("  T layers:                      {}", est.n_t_layers);
    println!("  Clifford layers (est.):        {}", est.n_clifford_layers);
    println!(
        "  Estimated min layers (circuit depth, T-failures={}): {}",
        !args.common.no_t_failures, est.min_layers
    );
    println!(
        "  Magic-state throughput min layers (λ={:.7}, {} magic qubits): {}",
        args.common.magic_state_lambda, n_magic_qubits, est.magic_min_layers
    );
    println!("  Combined min layers (lmin):    {}", est.lmin);
    println!("Volume estimate:");
    println!("  lmin × total_qubits = {} × {} = {}", est.lmin, n_qubits, est.vmin);

    Ok(())
}
