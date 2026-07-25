use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;

mod astar;
mod circuit;
mod cultivation;
mod node;
mod pauliproduct;
mod scheduler;
mod steinertree;
mod topograph;
mod topograph_plotter;
mod treegraph;
#[macro_use]
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
    /// Name of file containing input circuit (required).
    #[arg(short, long = "circuit")]
    circuit_fname: String,
    /// Name of file containing topology. If this is not set, it will be generated.
    #[arg(short, long = "topo", default_value = "")]
    topo_fname: String,
    /// Lambda parameter for exponential distribution of magic state cultivation lcycles.
    #[arg(short, long, default_value = "0.0387396")]
    magic_state_lambda: f64,
    /// Show product IDs instead of Pauli terms when plotting the circuit.
    #[arg(short = 'I', long)]
    show_product_ids: bool,
    /// Log scheduler actions to <CIRCUIT_FNAME>.sched file.
    #[arg(
        short = 'l',
        long = "log-scheduler",
        default_value = "none",
        value_parser = |s: &str| {
            match s.to_lowercase().as_str() {
                "none" | "info" | "debug" => Ok(s.to_string()),
                _ => Err(format!(
                    "invalid log level '{}'; must be one of: none, info, debug",
                    s
                ))
            }
        },
        help = "Log level for scheduler (none, info, or debug)"
    )]
    log_scheduler: String,
    /// Use magic qubits for routing in addition to bus qubits
    #[arg(short = 'u', long)]
    use_magic_routing: bool,
    /// Use only the sides of data qubits for edges, not the top and bottom
    #[arg(short = 'S', long = "sides_only")]
    sides_only: bool,
    /// Disable T gate failures (every T gate succeeds on first attempt)
    #[arg(short = 'F', long)]
    no_t_failures: bool,
    /// Number of ancilla between each data patch (all magic routing only)
    #[arg(short, long, default_value = "1")]
    ancilla_rows: usize,
    #[arg(
        short,
        long,
        value_delimiter = ',',
        value_parser = |s: &str| {
            match s.to_lowercase().as_str() {
                "topo" | "circuit" | "coupling" | "cstats" | "paths" | "" => Ok(s.to_string()),
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
        "  paths:    plot paths for first 100 lcycles in subdirectory <CIRCUIT_FNAME>.paths")
    )]
    plot: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");
    let args = Args::parse();
    let mut hdr = format!(
        "PureMagic - Git branch: {} | Commit: {} | Built: {}",
        env!("VERGEN_GIT_BRANCH"),
        &(env!("VERGEN_GIT_SHA"))[0..8],
        env!("VERGEN_BUILD_TIMESTAMP")
    );
    println!("{}\n{:#?}", hdr, args);
    hdr = format!("# {}\n# {:?}", &hdr, args);
    let circuit_fname = args.circuit_fname;
    let mut circuit = Circuit::new(&circuit_fname);
    circuit.load_circuit()?;
    let n_products = circuit.n_products();
    let _n_layers = circuit.print_statistics();
    #[cfg(debug_assertions)]
    circuit.print()?;
    if args.plot.contains(&"circuit".to_string()) {
        circuit.plot(args.show_product_ids)?;
    }
    if args.plot.contains(&"coupling".to_string()) {
        circuit.plot_qubit_coupling()?;
    }
    if args.plot.contains(&"cstats".to_string()) {
        circuit.plot_layer_stats()?;
    }
    let mut topo_graph = TopoGraph::new();
    let rseed = if args.randomize_data_qubits { args.rseed } else { 0 };
    let n_data_qubits = circuit.n_qubits;
    topo_graph.set_topo(
        n_data_qubits,
        &circuit_fname.to_string(),
        &args.topo_fname,
        &rseed,
        args.use_magic_routing,
        args.ancilla_rows,
        args.sides_only,
    );
    if args.plot.contains(&"topo".to_string()) {
        topo_graph.plot(".topo", &[], "")?;
        topo_graph.print()?;
    }
    let n_qubits = topo_graph.n_qubits;
    let n_magic_qubits = topo_graph.n_magic_qubits;
    let mut sched = Scheduler::new(
        circuit,
        topo_graph,
        args.magic_state_lambda,
        &args.log_scheduler,
        args.plot.join(" "),
        args.rseed,
        args.no_t_failures,
    );

    let (tot_lcycles, n_scheduled) = sched.sched_circuit()?;
    assert!(n_scheduled >= n_products);
    let volume = n_qubits * tot_lcycles;
    println!("Scheduled {} in {} logical cycles, volume {}", n_scheduled, tot_lcycles, volume);
    println!("Parallelism: {:.3}x", n_scheduled as f64 / tot_lcycles as f64);

    //let min_layers = if args.no_t_failures { n_layers } else { (n_layers * 3) / 2 };
    let mut rng = StdRng::seed_from_u64(args.rseed as u64);
    // this estimates layers including additional clifford cycles and T correction cycles
    let min_layers = sched.input.circuit.estimate_num_layers(&mut rng, args.no_t_failures);
    // this is just the T layers, so it doesn't include T corrections
    let (n_t_gates, n_t_layers) = sched.input.circuit.count_t_stats();
    // this means the clifford layers will also include T corrections
    let n_clifford_layers = min_layers - n_t_layers;
    // how many layers do we need to execute just T gates, given the magic state production rate?
    let max_t_parallelism = n_magic_qubits as f64 * args.magic_state_lambda;
    let magic_min_layers = (n_t_gates as f64 / max_t_parallelism) as usize;
    let lmin = std::cmp::max::<usize>(min_layers, n_clifford_layers as usize + magic_min_layers);
    let vmin = lmin * n_qubits;
    let max_parallelism_estimate = (n_products + n_t_gates / 2) as f64 / lmin as f64;
    println!("Max parallelism estimate: {:.3}", max_parallelism_estimate);
    println!("Volume estimate: {}", vmin);
    println!("Normalized scheduling efficiency: {:.3}", (vmin as f64 / volume as f64).min(1.0));

    sched.print_schedule(&hdr)?;
    Ok(())
}
