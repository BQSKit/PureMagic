use clap::Parser;
use std::time::Instant;

mod circuit;
mod pauliproduct;
mod topograph;
mod utils;

use circuit::Circuit;
use topograph::TopoGraph;
use utils::Timer;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Random seed
    #[arg(short = 'r', long, default_value = "29")]
    rseed: u64,
    /// Circuit file name
    #[arg(short, long = "circuit")]
    circuit_fname: String,
    /// Topology file name (topology will be generated if this is not set)
    #[arg(short, long = "topo", default_value = "")]
    topo_fname: String,
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
    /// Lambda parameter for exponential distribution of timesteps
    #[arg(short = 'm', long, default_value = "0.0387396")]
    magic_state_lambda: f64,
    /// Show product IDs when plotting the circuit
    #[arg(long)]
    show_product_ids: bool,
    /// Log scheduler actions to .sched file
    #[arg(short = 'l', long)]
    log_scheduler: bool,
    /// Plotting options
    #[arg(short, long, value_delimiter = ' ', default_value = "none")]
    plot: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");

    let args = Args::parse();

    println!("Arguments:");
    for (key, value) in [
        ("rseed", args.rseed.to_string()),
        ("circuit", args.circuit_fname.clone()),
        ("topo", args.topo_fname.clone()),
        ("verbose", args.verbose.to_string()),
        ("magic_state_lambda", args.magic_state_lambda.to_string()),
        ("show_product_ids", args.show_product_ids.to_string()),
        ("log_scheduler", args.log_scheduler.to_string()),
        ("plot", format!("{:?}", args.plot)),
    ] {
        println!("  {}={}", key, value);
    }

    let start = Instant::now();
    // Initialize circuit
    let mut circuit = Circuit::new(&args.circuit_fname)?;
    circuit.split_ys();
    let num_layers = circuit.get_statistics();
    circuit.print()?;

    // Plot circuit if requested
    if args.plot.contains(&"circuit".to_string()) {
        circuit.plot(args.show_product_ids)?;
    }
    // Initialize topology
    let mut topo_graph = TopoGraph::new();
    topo_graph.set_topo(circuit.num_qubits, &args.circuit_fname, &args.topo_fname);
    topo_graph.print()?;

    if args.plot.contains(&"topo".to_string()) {
        topo_graph.plot()?;
    }

    println!("Total time: {:?}", start.elapsed());
    Ok(())
}
