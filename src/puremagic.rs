use clap::Parser;

mod astar;
mod circuit;
mod cultivation;
#[allow(dead_code)]
mod frame;
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
    /// Restrict `--plot circuit` to layers `START:END`, as one file. Without
    /// this the whole circuit is plotted in 1000-layer chunks, which is far
    /// too wide to use as a figure.
    #[arg(long, value_name = "START:END", value_parser = parse_layer_range)]
    plot_layers: Option<(usize, usize)>,
    /// Restrict `--plot circuit` to qubits `START:END`. A window of layers
    /// usually touches a small part of the register -- the published excerpt
    /// of DNN(51) uses 34 of its 51 qubits -- and plotting the idle rows
    /// anyway makes the figure mostly blank, so it has to be scaled down to
    /// fit a column and stops being legible.
    #[arg(long, value_name = "START:END", value_parser = parse_layer_range)]
    plot_qubits: Option<(usize, usize)>,
    /// Write <CIRCUIT_STEM>.trace: the layout and every product's patches per
    /// lcycle, for independent checking by data_processing/check_schedule.py.
    #[arg(long)]
    trace: bool,
    /// Write <CIRCUIT_STEM>.cycle_times: the wall-clock time of each logical
    /// cycle's scheduling in ns, one per line, for worst-cycle analysis.
    #[arg(long)]
    cycle_times: bool,
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
    /// The weight limit the circuit was transpiled with (0: none). A T gate or
    /// measurement absorbs T-failure corrections, widening onto other qubits, up to
    /// this weight. Default: the circuit's `# max-weight N` header, else N from a
    /// `.transN` file name, else none
    #[arg(long)]
    max_weight: Option<usize>,
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

/// Parses the `START:END` argument of `--plot-layers` and `--plot-qubits`.
fn parse_layer_range(s: &str) -> Result<(usize, usize), String> {
    let (a, b) = s.split_once(':').ok_or_else(|| format!("expected START:END, got '{s}'"))?;
    let start: usize = a.trim().parse().map_err(|_| format!("bad start layer '{a}'"))?;
    let end: usize = b.trim().parse().map_err(|_| format!("bad end layer '{b}'"))?;
    if end <= start {
        return Err(format!("end {end} must exceed start {start}"));
    }
    Ok((start, end))
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
        circuit.plot(args.show_product_ids, args.plot_layers, args.plot_qubits)?;
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
    // Taken from the input circuit, before scheduling adds correction gates to it,
    // so it is the same bound circuit_stats reports and doesn't vary with the seed.
    let est = circuit.estimate_layer_volume(n_magic_qubits, n_qubits, args.common.magic_state_lambda);
    let mut sched = Scheduler::new(
        circuit,
        topo_graph,
        args.common.magic_state_lambda,
        args.common.cultivation_floor,
        &args.log_scheduler,
        args.plot.join(" "),
        args.common.rseed as u32,
        args.common.no_t_failures,
        args.record_cultivation_dist,
    );

    let (max_weight, source) = sched.input.circuit.resolve_max_weight(args.max_weight);
    sched.max_weight = max_weight;
    println!("Max weight: {} (from {})", max_weight, source);

    if args.trace {
        sched.enable_trace(&format!(
            "{}.trace",
            std::path::Path::new(&circuit_fname).file_stem().unwrap().to_string_lossy()
        ))?;
    }
    if args.cycle_times {
        sched.enable_cycle_times();
    }
    let (tot_lcycles, n_scheduled) = sched.sched_circuit()?;
    if args.cycle_times {
        let stem =
            std::path::Path::new(&circuit_fname).file_stem().unwrap().to_string_lossy().to_string();
        sched.write_cycle_times(&format!("{}.cycle_times", stem))?;
        #[cfg(not(feature = "diagnostics"))]
        sched.print_cycle_time_summary();
    }
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

    let max_parallelism_estimate = (n_products + est.n_t_gates / 2) as f64 / est.lmin as f64;
    println!("Max parallelism estimate: {:.3}", max_parallelism_estimate);
    println!("Volume estimate: {}", est.vmin);
    println!("Normalized scheduling efficiency: {:.3}", (est.vmin as f64 / volume as f64).min(1.0));

    sched.print_schedule(&hdr)?;
    Ok(())
}
