use clap::Parser;

mod circuit;
mod pauliproduct;
mod scheduler;
mod topograph;
mod utils;

use circuit::Circuit;
use scheduler::Scheduler;
use topograph::TopoGraph;
use utils::Timer;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Random seed for reproducible results.
    #[arg(short, long, default_value = "29")]
    rseed: u32,
    /// Randomize data qubit numbering.
    #[arg(short = 'R', long)]
    randomize_data_qubits: bool,
    /// Name of file containing input circuit .qasm format (required).
    #[arg(short, long = "circuit", required = true)]
    circuit_fname: String,
    /// Name of file containing topology. If this is not set, it will be generated.
    #[arg(short, long = "topo")]
    topo_fname: String,
    /// Verbose output.
    #[arg(short, long)]
    verbose: bool,
    /// Lambda parameter for exponential distribution of magic state cultivation timesteps.
    #[arg(short, long, default_value = "0.0387396")]
    magic_state_lambda: f64,
    /// Show product IDs instead of Pauli terms when plotting the circuit.
    #[arg(long)]
    show_product_ids: bool,
    /// Log scheduler actions to <CIRCUIT_FNAME>.sched file.
    #[arg(short = 'l', long)]
    log_scheduler: bool,
    /// Use first fit to choose the next product to schedule.
    #[arg(short = 'f', long)]
    first_fit: bool,
    /// Use magic qubits for routing in addition to bus qubits
    #[arg(short = 'u', long)]
    use_magic_routing: bool,
    #[arg(
        short,
        long,
        value_delimiter = ',',
        value_parser = |s: &str| {
            match s.to_lowercase().as_str() {
                "topo" | "circuit" | "cstats" | "paths" | "" => Ok(s.to_string()),
                _ => Err(format!(
                    "invalid plot option '{}'; must be one of: topo, circuit, cstats, paths",
                    s
                ))
            }
        },
        default_value = "",
        help = format!("Plot options (one or more):\n{}{}{}{}",
        "  topo:     plot topology in <CIRCUIT_FNAME>.topo.png\n",
        "  circuit:  plot full circuit in files in subdirectory <CIRCUIT_FNAME>.circuit\n",
        "  cstats:   plot circuit statistics over time in <CIRCUIT_FNAME>.layer_stats.png\n",
        "  paths:    plot paths for first 100 timesteps in subdirectory <CIRCUIT_FNAME>.paths")
    )]
    plot: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");

    let args = Args::parse();
    println!("{:#?}", args);

    // Initialize circuit
    let mut circuit = Circuit::new(&args.circuit_fname)?;
    circuit.split_ys();
    let num_layers = circuit.print_statistics();
    circuit.print()?;

    // Plot circuit if requested
    if args.plot.contains(&"circuit".to_string()) {
        circuit.plot(args.show_product_ids)?;
    }
    if args.plot.contains(&"cstats".to_string()) {
        circuit.plot_layer_stats()?;
        //return Ok(());
    }
    // Initialize topology
    let mut topo_graph = TopoGraph::new();
    let rseed = if args.randomize_data_qubits { args.rseed } else { 0 };
    topo_graph.set_topo(circuit.num_qubits,
                        &args.circuit_fname,
                        &args.topo_fname,
                        &rseed,
                        args.use_magic_routing);
    topo_graph.print()?;

    if args.plot.contains(&"topo".to_string()) {
        topo_graph.plot(".topo", &[], "")?;
    }

    let num_qubits = topo_graph.num_qubits;

    let mut scheduler = Scheduler::new(circuit,
                                       topo_graph,
                                       args.magic_state_lambda,
                                       args.log_scheduler,
                                       args.plot.join(" "),
                                       args.rseed);

    let (tot_num_steps, num_scheduled, space_utilization) =
        scheduler.schedule_circuit(args.first_fit)?;

    // Calculate and print statistics
    let speedup = num_scheduled as f64 / tot_num_steps as f64;
    let volume = num_qubits * tot_num_steps;
    println!("Scheduled {} in {} time steps ({:.3} speedup) volume {}",
             num_scheduled, tot_num_steps, speedup, volume);

    let optimal_speedup = num_scheduled as f64 / num_layers as f64;
    let optimal_volume = num_qubits * num_layers;
    println!("Optimal time steps {} ({:.3} speedup) volume {}",
             num_layers, optimal_speedup, optimal_volume);

    println!("Scheduling time efficiency {:.3}", speedup as f64 / optimal_speedup as f64);
    println!("Scheduling space efficiency {:.3}", space_utilization);
    println!("Scheduling overall efficiency {:.3}", optimal_volume as f64 / volume as f64);

    Ok(())
}
