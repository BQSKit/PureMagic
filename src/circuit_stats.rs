#!/usr/bin/env -S cargo run --bin circuit_stats --
//! Estimate circuit statistics and layer/volume bounds from a `.trans` circuit file,
//! using the same methodology as the main `puremagic` scheduler but without running it.

use clap::Parser;

#[allow(dead_code)]
mod circuit;
#[allow(dead_code)]
mod frame;
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
use frame::{predicted_depth_with_failures, Kind};
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

    /// Also predict the depth with T-injection failures, replaying the scheduler's
    /// correction frame over this many seeds (0: skip). Printed as the mean.
    #[arg(long, default_value_t = 0)]
    failure_seeds: u64,

    /// The weight limit corrections may widen products to (0: none). Default: the
    /// circuit's header, else N from a `.transN` file name, else none.
    #[arg(long)]
    max_weight: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    utils::print_banner("circuit_stats");
    // Same form as the scheduler's Args dump, so the same parsers pick it up.
    println!("circuit_fname: {:?}", args.common.circuit_fname);
    println!("magic_state_lambda: {}", args.common.magic_state_lambda);

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

    let est = circuit.estimate_layer_volume(n_magic_qubits, n_qubits, args.common.magic_state_lambda);

    println!("Routing mode: {}", routing_label);
    println!("Layer estimates:");
    println!("  Circuit layers (DAG depth):    {}", n_layers);
    println!("  T gates:                       {}", est.n_t_gates);
    println!("  T layers:                      {}", est.n_t_layers);
    println!("  Clifford layers (est.):        {}", est.n_clifford_layers);
    println!("  Estimated min layers (circuit depth, no T failures): {}", est.min_layers);
    println!(
        "  Magic-state throughput min layers (λ={:.7}, {} magic qubits): {}",
        args.common.magic_state_lambda, n_magic_qubits, est.magic_min_layers
    );
    println!("  Combined min layers (lmin):    {}", est.lmin);
    println!("Volume estimate: {}", est.vmin);
    println!("  lmin × total_qubits = {} × {} = {}", est.lmin, n_qubits, est.vmin);

    if args.failure_seeds > 0 {
        let (w, source) = circuit.resolve_max_weight(args.max_weight);
        let steps: Vec<(Kind, Vec<pauliproduct::Operator>)> = circuit
            .pps
            .iter()
            .map(|pp| {
                let g = pp.gate_type;
                let q0 = pp.operators[0].qubit;
                let one = |b| vec![pauliproduct::Operator { qubit: q0, basis: b }];
                if g.is_t() {
                    (Kind::T, pp.operators.clone())
                } else if g.is_m() {
                    (Kind::M, pp.operators.clone())
                } else if g.is_s() {
                    (Kind::S, one('Z'))
                } else if g.is_sx() {
                    (Kind::SX, one('X'))
                } else {
                    (Kind::CX, pp.operators.clone())
                }
            })
            .collect();
        let run = |p: f64, seed: u64| {
            predicted_depth_with_failures(
                steps.iter().map(|(k, o)| (*k, &o[..])),
                n_data_qubits,
                w,
                p,
                seed,
            )
        };
        let base = run(0.0, 0);
        let mean = (0..args.failure_seeds).map(|s| run(0.5, s)).sum::<usize>() as f64
            / args.failure_seeds as f64;
        println!("Max weight: {} (from {})", w, source);
        println!("Predicted depth: {} without failures, {:.1} with failures", base, mean);
    }

    Ok(())
}
