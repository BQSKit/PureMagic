use clap::Parser;

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
    let n_layers = circuit.print_statistics();
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
    // The definition of efficiency used in the PureMagic paper is 1/V. We normalize here
    let (n_t_gates, n_t_layers) = sched.input.circuit.count_t_stats();
    /*
    let mut optimal_n_layers = n_layers;
    if !args.no_t_failures {
        optimal_n_layers += n_t_layers / 2;
    };
    This complicated calculation assumes every T correction takes adds a cycle, once the full
    parallelism is met
    // Magic states accumulate at rate n_magic_qubits * lambda per lcycle (including Clifford
    // and correction lcycles). They are only consumed during T-gate lcycles. 50% of T gates
    // need a Pauli correction that does not consume a magic state.
    // max_parallelism = total magic states available / number of T-gate lcycles,
    // capped by the average T gates per T-gate layer (circuit's inherent parallelism).
    let expected_total_lcycles = optimal_n_layers as f64;
    let avg_t_per_layer = n_t_gates as f64 / n_t_layers as f64;
    let max_parallelism = if n_t_layers > 0 {
        let supply_limited =
            n_magic_qubits as f64 * args.magic_state_lambda * expected_total_lcycles
                / n_t_layers as f64;
        supply_limited.min(avg_t_per_layer)
    } else {
        0.0
    };
    println!("Max parallelism estimate: {:.3}", max_parallelism);
    // Min volume estimate: schedule T gates at max_parallelism rate, Cliffords unchanged.
    let n_clifford_layers = (n_layers - n_t_layers) as f64;
    let min_t_lcycles =
        if max_parallelism > 0.0 { n_t_gates as f64 / max_parallelism } else { 0.0 };
    let min_correction_lcycles = if args.no_t_failures { 0.0 } else { min_t_lcycles / 2.0 };
    let min_lcycles = (n_clifford_layers + min_t_lcycles + min_correction_lcycles).ceil() as usize;
    let min_volume = n_qubits * min_lcycles;
    println!("Min volume estimate: {}", min_volume);
     */

    // the big question is what to do with the T failure corrections. They don't need magic state,
    // but they do reduce efficiency. For heavyweight PBC, they reduce it by 50% but for
    // lightweight, they do get absorbed into the schedule to some extent. But there are some
    // circuits where they all get absormed, so if we want a lower bound, we have to assume they
    // all get absorbed
    // They definitely can't be absorbed if the layer following the T contains another product on
    // the same qubit, because they'll push back that product and add a cycle, guaranteed
    let max_t_parallelism = n_magic_qubits as f64 * args.magic_state_lambda;
    let magic_min_layers = (n_t_gates as f64 / max_t_parallelism) as usize;
    // Each CX takes 2 and S takes 3 cycles, but tt is also possible that these could all be
    // absorbed, so we can't have a scaling factor here.
    // This could be refined by generating the circuit layers fter adding the extra Cliffords.
    // But in general it shouldn't really matter because these are a small fraction of the total
    // Well, we see some circuits with 8% Cliffords, so that could blow it up a bit.
    let n_clifford_layers = n_layers - n_t_layers;
    let lmin = std::cmp::max::<usize>(n_layers, n_clifford_layers as usize + magic_min_layers);
    let vmin = lmin * n_qubits;
    let max_parallelism_estimate = (n_products + n_t_gates / 2) as f64 / lmin as f64;
    println!("Max parallelism estimate: {:.3}", max_parallelism_estimate);
    println!("Min volume estimate: {}", vmin);
    println!("Normalized scheduling efficiency: {:.3}", (vmin as f64 / volume as f64).min(1.0));

    sched.print_schedule(&hdr)?;
    Ok(())
}
