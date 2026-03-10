use clap::Parser;

mod astar;
mod circuit;
mod node;
mod pauliproduct;
mod scheduler;
mod steinertree;
mod topograph;
mod treegraph;
#[macro_use]
mod utils;
mod greedypath;

use circuit::Circuit;
use scheduler::Scheduler;
use topograph::TopoGraph;
use utils::Timer;

/// Command-line arguments for PureMagic.
/// Controls circuit input, topology, scheduling strategy, and output options.
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
    #[arg(short, long = "circuit")]
    circuit_fname: Option<String>,
    /// Generate a random circuit instead of loading from file
    #[arg(long, help = "Generate a random circuit instead of loading from file")]
    generate_random: bool,
    /// Number of qubits for random circuit generation
    #[arg(long, default_value = "64", help = "Number of qubits for random circuit generation")]
    random_qubits: usize,
    /// Number of products for random circuit generation
    #[arg(long, default_value = "1000", help = "Number of products for random circuit generation")]
    random_products: usize,
    /// Spread probability for random circuit generation (probability of adding operators to
    /// adjacent qubits)
    #[arg(long,
          default_value = "0.4",
          help = "Spread probability for random circuit generation (0.0-1.0)")]
    spread_probability: f64,
    /// Decay factor for random circuit generation (how much probability decreases with distance)
    #[arg(long,
          default_value = "0.7",
          help = "Decay factor for random circuit generation (0.0-1.0)")]
    decay_factor: f64,

    /// Name of file containing topology. If this is not set, it will be generated.
    #[arg(short, long = "topo", default_value = "")]
    topo_fname: String,
    /// Verbose output.
    #[arg(short, long)]
    verbose: bool,
    /// Lambda parameter for exponential distribution of magic state cultivation timesteps.
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
    /// Use the faster, suboptimal greedy path algorithm
    #[arg(short = 'g', long = "use_greedy")]
    greedy_path: bool,
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
        "  paths:    plot paths for first 100 timesteps in subdirectory <CIRCUIT_FNAME>.paths")
    )]
    plot: Vec<String>,
}

/// Entry point: parses arguments, loads or generates the circuit and topology, runs the
/// scheduler, then prints scheduling efficiency and parallelism statistics.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");
    let args = Args::parse();
    let mut hdr = format!("PureMagic - Git branch: {} | Commit: {} | Built: {}",
                          env!("VERGEN_GIT_BRANCH"),
                          &(env!("VERGEN_GIT_SHA"))[0..8],
                          env!("VERGEN_BUILD_TIMESTAMP"));
    println!("{}\n{:#?}", hdr, args);
    hdr = format!("# {}\n# {:?}", &hdr, args);
    // Validate arguments
    if !args.generate_random && args.circuit_fname.is_none() {
        eprintln!("Error: Either --circuit <filename> or --generate-random must be specified");
        std::process::exit(1);
    }
    if args.generate_random && (args.spread_probability < 0.0 || args.spread_probability > 1.0) {
        eprintln!("Error: spread_probability must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    if args.generate_random && (args.decay_factor < 0.0 || args.decay_factor > 1.0) {
        eprintln!("Error: decay_factor must be between 0.0 and 1.0");
        std::process::exit(1);
    }
    // Initialize circuit
    let (circuit, circuit_fname) = if args.generate_random {
        println!("Generating random circuit with {} products on {} qubits",
                 args.random_products, args.random_qubits);
        println!("  spread_probability: {}", args.spread_probability);
        println!("  decay_factor: {}", args.decay_factor);
        let spread_str = args.spread_probability.to_string().replace(".", "_");
        let decay_str = args.decay_factor.to_string().replace(".", "_");
        let fname = format!("random_circuit-{}-{}_n{}", spread_str, decay_str, args.random_qubits);
        let mut circuit = Circuit::new(&fname);
        circuit.generate_random(args.random_products,
                                args.random_qubits,
                                args.spread_probability,
                                args.decay_factor);
        let save_fname = format!("{}.generated.txt", fname);
        circuit.save_circuit_to_file(save_fname)?;
        (circuit, fname)
    } else {
        let fname = args.circuit_fname.unwrap();
        let mut circuit = Circuit::new(&fname.to_string());
        circuit.load_circuit()?;
        (circuit, fname)
    };
    let num_products = circuit.num_products();
    let num_layers = circuit.print_statistics();
    #[cfg(debug_assertions)]
    circuit.print()?;
    // Plot circuit if requested
    if args.plot.contains(&"circuit".to_string()) {
        circuit.plot(args.show_product_ids)?;
    }
    if args.plot.contains(&"coupling".to_string()) {
        circuit.plot_qubit_coupling()?;
    }
    if args.plot.contains(&"cstats".to_string()) {
        circuit.plot_layer_stats()?;
    }
    // Initialize topology
    let mut topo_graph = TopoGraph::new();
    let rseed = if args.randomize_data_qubits { args.rseed } else { 0 };
    let num_data_qubits = circuit.num_qubits;
    topo_graph.set_topo(num_data_qubits,
                        &circuit_fname.to_string(),
                        &args.topo_fname,
                        &rseed,
                        args.use_magic_routing,
                        args.ancilla_rows,
                        args.sides_only);
    #[cfg(debug_assertions)]
    topo_graph.print()?;
    if args.plot.contains(&"topo".to_string()) {
        topo_graph.plot(".topo", &[], "")?;
    }
    let mut num_qubits = topo_graph.num_qubits;

    let mut scheduler = Scheduler::new(circuit,
                                       topo_graph,
                                       args.magic_state_lambda,
                                       &args.log_scheduler,
                                       args.plot.join(" "),
                                       args.rseed,
                                       args.greedy_path);

    let (tot_num_steps, num_scheduled) = scheduler.schedule_circuit()?;
    assert_eq!(num_scheduled, num_products);
    // Calculate and print statistics
    let volume = num_qubits * tot_num_steps;
    println!("Scheduled {} in {} timesteps, volume {}", num_scheduled, tot_num_steps, volume);
    scheduler.print_schedule(&hdr)?;
    print!("Generating Pure Magic layout for comparison:\n  ");
    let mut best_magic_topo_graph = TopoGraph::new();
    best_magic_topo_graph.gen_pure_magic_topo(num_data_qubits, 1, false);
    best_magic_topo_graph.update_statistics();
    num_qubits = best_magic_topo_graph.num_qubits;
    let optimal_speedup = num_scheduled as f64 / num_layers as f64;
    let optimal_volume = num_qubits * num_layers;
    println!("Optimal timesteps {} ({:.3} speedup) volume {}",
             num_layers, optimal_speedup, optimal_volume);
    println!("Scheduling efficiency: {:.3}", optimal_volume as f64 / volume as f64);
    println!("Parallelism: {:.3}x", num_products as f64 / tot_num_steps as f64);
    Ok(())
}
