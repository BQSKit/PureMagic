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
use utils::{CommonArgs, Timer};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,
    /// Randomize data qubit numbering.
    #[arg(short = 'R', long)]
    randomize_data_qubits: bool,
    /// Name of file containing topology. If this is not set, it will be generated.
    #[arg(short, long = "topo", default_value = "")]
    topo_fname: String,
    /// Show product IDs instead of Pauli terms when plotting the circuit.
    #[arg(short = 'I', long)]
    show_product_ids: bool,
    /// Log scheduler actions to <CIRCUIT_FNAME>.sched_trace. Only populated in debug
    /// builds: the debug_sched!/info_sched! call sites are compiled out entirely in
    /// release builds, so a release build still creates the file but leaves it empty.
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
        help = "Log level for scheduler (none, info, or debug); debug builds only"
    )]
    log_scheduler: String,
    /// Use magic qubits for routing in addition to bus qubits
    #[arg(short = 'u', long)]
    use_magic_routing: bool,
    /// Use only the sides of data qubits for edges, not the top and bottom
    #[arg(short = 'S', long = "sides_only")]
    sides_only: bool,
    /// Record normalized cultivation-time distribution to <CIRCUIT_FNAME>.cultivation_dist
    #[arg(short = 'C', long)]
    record_cultivation_dist: bool,
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
        "  cstats:   plot circuit statistics over time in <CIRCUIT_FNAME>.layer_stats.svg\n",
        "  paths:    plot paths for first 100 lcycles in subdirectory <CIRCUIT_FNAME>.paths")
    )]
    plot: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _timer = Timer::new("main");
    let args = Args::parse();
    let mut hdr = utils::print_banner("PureMagic");
    println!("{:#?}", args);
    hdr = format!("# {}\n# {:?}", &hdr, args);
    let circuit_fname = args.common.circuit_fname;
    let mut circuit = Circuit::new(&circuit_fname);
    circuit.load_circuit()?;
    let n_products = circuit.n_products();
    let n_sx_cliffords =
        circuit.pps.iter().filter(|pp| pp.gate_type.is_s() || pp.gate_type.is_sx()).count();
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
    let rseed: u32 = if args.randomize_data_qubits { args.common.rseed as u32 } else { 0 };
    let n_data_qubits = circuit.n_qubits;
    topo_graph.set_topo(
        n_data_qubits,
        &circuit_fname.to_string(),
        &args.topo_fname,
        &rseed,
        args.use_magic_routing,
        args.common.ancilla_rows,
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
        args.common.magic_state_lambda,
        &args.log_scheduler,
        args.plot.join(" "),
        args.common.rseed as u32,
        args.common.no_t_failures,
        args.record_cultivation_dist,
    );

    let (tot_lcycles, n_scheduled) = sched.sched_circuit()?;
    assert!(n_scheduled >= n_products);
    let volume = n_qubits * tot_lcycles;
    println!("Scheduled {} in {} logical cycles, volume {}", n_scheduled, tot_lcycles, volume);
    if !args.common.no_t_failures && n_sx_cliffords > 0 {
        let corrections = sched.correction_gates_emitted;
        println!(
            "S correction gates: {} / {} S/SX Cliffords ({:.1}%)",
            corrections,
            n_sx_cliffords,
            corrections as f64 * 100.0 / n_sx_cliffords as f64
        );
    }
    println!("Parallelism: {:.3}x", n_scheduled as f64 / tot_lcycles as f64);

    let mut rng = StdRng::seed_from_u64(args.common.rseed);
    let est = sched.input.circuit.estimate_layer_volume(
        n_magic_qubits,
        n_qubits,
        args.common.magic_state_lambda,
        args.common.no_t_failures,
        &mut rng,
    );
    let max_parallelism_estimate = (n_products + est.n_t_gates / 2) as f64 / est.lmin as f64;
    println!("Max parallelism estimate: {:.3}", max_parallelism_estimate);
    println!("Volume estimate: {}", est.vmin);
    println!("Normalized scheduling efficiency: {:.3}", (est.vmin as f64 / volume as f64).min(1.0));

    sched.print_schedule(&hdr)?;
    Ok(())
}
