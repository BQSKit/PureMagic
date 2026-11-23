use crate::accum_start;
use crate::astar::{AStar, PathResult};
use crate::circuit::Circuit;
use crate::cultivation::CultivationManager;
use crate::debug_sched;
use crate::fn_timer;
use crate::frame::{Added, CorrectionFrame, Pass, sorted};
use crate::info_sched;
use crate::node::NodeType;
use crate::pauliproduct::{GateType, Operator, PauliProduct};
use crate::steinertree::SteinerTree;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::AccumTimers;
use colored::{Color, Colorize};

use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

/// What the correction frame lets a T gate or measurement do this lcycle.
#[derive(Debug, PartialEq)]
enum Rotation {
    /// Run, about its operators conjugated by the owed corrections.
    Run,
    /// Emit these corrections first: absorbing them would pass the weight limit.
    Emit(Vec<usize>),
    /// It widened onto a qubit something else holds this lcycle.
    Wait,
}

/// How the S corrections that failed T gates owe are resolved. A failed T gate
/// about P owes S_P on all of P's qubits, carried in the correction frame
/// (`frame.rs`) until an operation on those qubits has to deal with it.
#[derive(Debug, Default, Clone)]
pub(crate) struct CorrectionStats {
    /// T gates scheduled with their basis conjugated through a pending correction.
    pub(crate) t_conjugated: usize,
    /// Corrections folded into a same-axis S/SX with even net power: the gate vanishes.
    pub(crate) merged_free: usize,
    /// Corrections folded into a same-axis S/SX with odd net power: still one gate.
    pub(crate) merged_gate: usize,
    /// A failure about the same Pauli as the newest owed correction: the two make
    /// a Pauli, tracked classically.
    pub(crate) classical: usize,
    /// Standalone single-qubit correction gates emitted, by axis X, Y, Z.
    pub(crate) emitted: [usize; 3],
    /// Standalone multi-qubit correction gates emitted (a failed multi-qubit T gate).
    pub(crate) emitted_multi: usize,
    /// Of all emitted corrections, those emitted because absorbing them into a T
    /// gate or measurement would have taken it past the weight limit.
    pub(crate) over_limit: usize,
    /// T gates and measurements that absorbed corrections onto qubits beyond
    /// their own, running wider than the circuit wrote them.
    pub(crate) widened: usize,
    /// Measurements scheduled about operators conjugated through owed
    /// corrections, as T gates are.
    pub(crate) m_conjugated: usize,
    /// Of `emitted`, those placed ahead of a CX that does not commute with them.
    pub(crate) before_multi: usize,
}

/// Per-lcycle and cumulative qubit utilisation counters.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) struct ScheduleStats {
    data_qubits: usize,
    bus_qubits: usize,
    magic_qubits: usize,
    /// Cumulative totals across all lcycles (reset only at construction).
    tot_data_used: usize,
    tot_bus_used: usize,
    tot_magic_used: usize,
    /// Per-lcycle counters; reset to 0 at the end of each `update()` call.
    bus_used: usize,
    data_used: usize,
    magic_used: usize,
    /// T products that consumed a magic node this lcycle (first attempt only).
    t_scheduled: usize,
    /// Ready magic nodes used in any path this lcycle (routing or T-terminal).
    magic_ready_used: usize,
    tot_magic_unused: usize,
    /// Non-data patch-lcycles reserved by any path (routing, injection, held Clifford).
    tot_ancilla_reserved: usize,
    /// Lcycles that magic states consumed by T gates spent ready before consumption.
    /// States routed over or never used are not counted: they carry no error forward.
    tot_consumed_wait: usize,
    plot_info: String,
}

impl ScheduleStats {
    pub(crate) fn new(data_qubits: usize, bus_qubits: usize, magic_qubits: usize) -> Self {
        ScheduleStats {
            data_qubits,
            bus_qubits,
            magic_qubits,
            tot_data_used: 0,
            tot_bus_used: 0,
            tot_magic_used: 0,
            bus_used: 0,
            data_used: 0,
            magic_used: 0,
            t_scheduled: 0,
            magic_ready_used: 0,
            tot_magic_unused: 0,
            tot_ancilla_reserved: 0,
            tot_consumed_wait: 0,
            plot_info: String::new(),
        }
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables, dead_code))]
    pub(crate) fn summarize(&self, n_lcycles: usize) {
        let data_frac = self.tot_data_used as f64 / (self.data_qubits * n_lcycles) as f64;
        let bus_frac = self.tot_bus_used as f64 / (self.bus_qubits * n_lcycles) as f64;
        let magic_frac = self.tot_magic_used as f64 / (self.magic_qubits * n_lcycles) as f64;
        let magic_unused_frac =
            self.tot_magic_unused as f64 / (self.magic_qubits * n_lcycles) as f64;
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("Magic unused {:.3}", magic_unused_frac);
        // Patch-lcycles in which an ancilla carries logical information: reserved by a
        // path, or holding a magic state a T gate later consumes. The remainder are idle
        // or cultivating, where an error is discarded with the state.
        println!(
            "Ancilla patch-lcycles: reserved {} consumed-wait {} held-ready {} of {}",
            self.tot_ancilla_reserved,
            self.tot_consumed_wait,
            self.tot_magic_unused,
            (self.bus_qubits + self.magic_qubits) * n_lcycles
        );
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables, dead_code))]
    pub(crate) fn update(
        &mut self, lcycle_i: usize, pp_paths_len: usize, tot_available: usize, magic_ready: usize,
        magic_unused: usize, plotting: bool,
    ) {
        #[cfg(feature = "diagnostics")]
        {
            self.tot_data_used += self.data_used;
            self.tot_bus_used += self.bus_used;
            self.tot_magic_used += self.magic_used;
            self.tot_magic_unused += magic_unused;

            let tot_qubits = self.data_qubits + self.bus_qubits + self.magic_qubits;
            let tot_qubits_used = self.data_used + self.bus_used + self.magic_used;

            info_sched!("Scheduling results:");
            let frac_paths =
                if tot_available == 0 { 1.0 } else { pp_paths_len as f64 / tot_available as f64 };
            let frac_qubits =
                if tot_qubits == 0 { 0.0 } else { tot_qubits_used as f64 / tot_qubits as f64 };
            // magic_denom excludes nodes consumed purely as routing intermediaries (magic-routing mode).
            let magic_ready_routing = self.magic_ready_used.saturating_sub(self.t_scheduled);
            let magic_denom = magic_ready.saturating_sub(magic_ready_routing);
            let frac_magic =
                if magic_denom == 0 { 0.0 } else { self.t_scheduled as f64 / magic_denom as f64 };
            info_sched!("  products:    {}/{} ({:.2})", pp_paths_len, tot_available, frac_paths);
            info_sched!("  qubits:      {}/{} ({:.2})", tot_qubits_used, tot_qubits, frac_qubits);
            info_sched!("  magic:       {}/{} ({:.2})", self.t_scheduled, magic_denom, frac_magic);
            if plotting {
                self.plot_info = format!(
                    "lcycle {}: products scheduled {}/{} ({:.2}), qubits {}/{} ({:.2}), magic {}/{} ({:.2})",
                    lcycle_i,
                    pp_paths_len,
                    tot_available,
                    frac_paths,
                    tot_qubits_used,
                    tot_qubits,
                    frac_qubits,
                    self.t_scheduled,
                    magic_denom,
                    frac_magic,
                );
            }

            self.data_used = 0;
            self.bus_used = 0;
            self.magic_used = 0;
            self.t_scheduled = 0;
            self.magic_ready_used = 0;
        }
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables, dead_code))]
    pub(crate) fn inc(&mut self, node_type: NodeType) {
        #[cfg(feature = "diagnostics")]
        {
            match node_type {
                NodeType::Bus => self.bus_used += 1,
                NodeType::Magic => self.magic_used += 1,
                NodeType::Data => self.data_used += 1,
            }
        }
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables, dead_code))]
    pub(crate) fn inc_with_cultivation(&mut self, node_type: NodeType, cultivation_time: i32) {
        #[cfg(feature = "diagnostics")]
        {
            self.inc(node_type);
            if node_type == NodeType::Magic && cultivation_time == 0 {
                self.magic_ready_used += 1;
            }
        }
    }

    pub(crate) fn inc_t(&mut self) {
        #[cfg(feature = "diagnostics")]
        {
            self.t_scheduled += 1;
        }
    }

    pub(crate) fn plot_info_str(&self) -> &str {
        &self.plot_info
    }
}

/// Read-only inputs grouped so methods can borrow `input` and mutable state
/// fields simultaneously without conflicting borrows.
pub(crate) struct SchedulerInput {
    pub circuit: Circuit,
    pub topo: TopoGraph,
}

impl SchedulerInput {
    pub(crate) fn circuit_stem(&self) -> &str {
        use std::path::Path;
        Path::new(&self.circuit.circuit_fname)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("circuit")
    }
}

/// Assigns Pauli products to lcycles and routes them through the topology.
pub(crate) struct Scheduler {
    pub(crate) input: SchedulerInput,
    rng_uniform: StdRng,
    magic_state_lambda: f64,
    cultivation_floor: f64,
    plot_option: String,
    pub(crate) cultivation: CultivationManager,
    pub(crate) stats: ScheduleStats,
    /// (lcycle_index, product_ids) for every lcycle that scheduled at least one product.
    pub(crate) lcycle_scheduled: Vec<(usize, Vec<i32>)>,
    pub(crate) scheduled_products: IndexSet<i32>,
    /// Per-node occupancy bitmap for the current lcycle; reset to all-false each lcycle.
    used: Vec<bool>,
    /// Per-node lcycle at which its current magic state became ready; `usize::MAX`
    /// while the node is cultivating or is not a magic node.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    ready_since: Vec<usize>,
    /// Magic nodes a T gate took its state from this lcycle, for the yield
    /// accounting in `CultivationManager::update_cultivators`.
    consumed: Vec<bool>,
    /// Schedule trace for independent validation (`--trace`), see `enable_trace`.
    trace: Option<BufWriter<File>>,
    /// Footprint and magic node of this lcycle's T gates routed by A* without a
    /// tree, which `pp_paths` cannot carry; only filled while tracing.
    trace_untreed: HashMap<i32, (Vec<u16>, u16)>,
    /// Wall-clock time of each `schedule_lcycle` call in ns (`--cycle-times`).
    cycle_times: Option<Vec<u64>>,
    /// Footprint of this lcycle's other products scheduled without a tree
    /// (precomputed, carried forward, S/SX, M); only filled while tracing.
    trace_fp: HashMap<i32, Vec<u16>>,
    /// Patches of S/SX products scheduled without a tree, by product id, so
    /// that `advance_clifford_state` can hold them for the gate's later lcycles.
    s_sx_nodes: HashMap<i32, Vec<u16>>,
    /// In-progress multi-lcycle Clifford gates: maps product_id →
    /// (remaining_lcycles, product, node_ids, opt_tree).
    /// CX occupies 2 lcycles; S/SX occupy 3.
    clifford_paths: IndexMap<i32, (usize, PauliProduct, Vec<u16>, Option<Rc<TreeGraph>>)>,
    pub(crate) t_gate_failures: usize,
    /// How T-injection corrections were resolved; see `CorrectionStats`.
    pub(crate) corr: CorrectionStats,
    pub(crate) correction_gates_emitted: usize,
    pub(crate) stree_computation: SteinerTree,
    pub(crate) astar: AStar,
    no_t_failures: bool,
    record_cultivation_dist: bool,
    /// Reusable buffers to avoid per-lcycle allocations.
    terminals_buf: Vec<u16>,
    scheduled_ids_buf: Vec<i32>,
    children_buf: Vec<i32>,
    precomputed_clifford_trees: HashMap<i32, Rc<TreeGraph>>,
    remaining_ids_buf: Vec<i32>,
    /// Indexed by product_id; built once before scheduling begins.
    precomputed_terminals: Vec<Vec<u16>>,
    /// (is_y_paired, preferred_roots, side_roots) per terminal per product.
    precomputed_root_info: Vec<Vec<(bool, Vec<u16>, Vec<u16>)>>,
    timers: AccumTimers,
    loop_timer: usize,
    other_timer: usize,
    pps_pending: Vec<PauliProduct>,
    /// Remaining unresolved parent count per product; decremented as parents complete.
    remaining_parents: Vec<usize>,
    pub(crate) pp_paths: Vec<(i32, Option<Rc<TreeGraph>>)>,
    current_lcycle: usize,
    /// The S_P corrections failed T gates owe, each over all of its gate's qubits
    /// (see `frame.rs`).
    frame: CorrectionFrame,
    /// The weight limit the circuit was transpiled with (0: none). A T gate or
    /// measurement absorbs owed corrections, widening onto other qubits, as long
    /// as it stays within this; a correction that would take it past is emitted.
    pub(crate) max_weight: usize,
    /// Qubits beyond their own that products widened onto this lcycle: no other
    /// product may touch them in it.
    widened_qubits: Vec<u16>,
    /// Multi-qubit measurements the frame conjugates this lcycle. Their
    /// precomputed patches no longer fit, so they are routed like T gates.
    dynamic_m: HashSet<i32>,
    /// For each qubit, the lcycle at which an injected correction gate on it ends
    /// (exclusive). A correction holds only the patch edges it routes through, and
    /// has no place in the dependency graph, so this keeps every other product off
    /// its qubits until it completes.
    qubit_locked_until: Vec<usize>,
    /// The lcycle each correction gate still holding a lock ends at. Empty lets
    /// the lock check be skipped.
    lock_ends: Vec<usize>,
    /// Next ID to assign when creating a dynamic S correction product.
    next_correction_id: i32,
    /// Products in the circuit file; ids from here on are injected corrections.
    n_circuit_products: usize,
    /// The operators T gates and measurements scheduled this lcycle ran about, where the
    /// frame changed them. Populated in sched_remaining; consumed in process_t_gate_outcomes.
    t_conjugated_bases: HashMap<i32, Vec<Operator>>,
    /// T gate and measurement IDs whose precomputed_terminals/root_info currently reflect
    /// conjugated operators; `conjugate_rotation` restores them once nothing is owed.
    conjugated_t_products: HashSet<i32>,
    /// S product IDs fully discharged for free by `try_merge_s_correction`: a pending
    /// correction merged with this product's own S to a net power of 0 or 2 (identity or
    /// classical Z), so zero physical gates were scheduled. Consulted by `unlock_children`/
    /// `advance_clifford_state`/`check_clifford_repetitions` to treat these as complete,
    /// single-lcycle, zero-node events instead of ordinary 3-lcycle S/SX Cliffords.
    free_merged_products: HashSet<i32>,
}

impl Scheduler {
    pub(crate) fn new(
        circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, cultivation_floor: f64,
        log_level: &str, plot_option: String, rseed: u32, no_t_failures: bool,
        record_cultivation_dist: bool,
    ) -> Self {
        let n_qubits = circuit.n_qubits;
        let n_initial_products = circuit.n_products() as i32;
        if log_level != "none" {
            // Installed regardless of build profile, but debug_sched!/info_sched! (see
            // utils.rs) are compiled out entirely in release, so a release build still
            // creates trace_fname, it just never gets anything written to it.
            let trace_fname = format!("{}.sched_trace", circuit_stem(&circuit.circuit_fname));
            let level_filter = match log_level.to_lowercase().as_str() {
                "debug" => log::LevelFilter::Debug,
                "info" => log::LevelFilter::Info,
                _ => log::LevelFilter::Off,
            };
            simple_logging::log_to_file(&trace_fname, level_filter)
                .expect("Failed to initialize logging");
        }
        let n_data_qubits = topo.n_data_qubits;
        let n_bus_qubits = topo.n_bus_qubits;
        let n_magic_qubits = topo.n_magic_qubits;
        let n_nodes = topo.n_nodes;
        let mut timers = AccumTimers::new();
        let loop_timer = timers.add_or_get("schedule loop");
        let other_timer = timers.add_or_get("other ");
        Scheduler {
            input: SchedulerInput { circuit, topo },
            rng_uniform: StdRng::seed_from_u64(rseed as u64),
            magic_state_lambda,
            cultivation_floor,
            plot_option,
            cultivation: CultivationManager::new(rseed),
            stats: ScheduleStats::new(n_data_qubits, n_bus_qubits, n_magic_qubits),
            lcycle_scheduled: Vec::new(),
            scheduled_products: IndexSet::new(),
            used: vec![false; n_nodes],
            ready_since: vec![usize::MAX; n_nodes],
            consumed: vec![false; n_nodes],
            trace: None,
            trace_untreed: HashMap::new(),
            trace_fp: HashMap::new(),
            cycle_times: None,
            s_sx_nodes: HashMap::new(),
            clifford_paths: IndexMap::new(),
            t_gate_failures: 0,
            corr: CorrectionStats::default(),
            correction_gates_emitted: 0,
            stree_computation: SteinerTree::new(n_nodes),
            astar: AStar::new(n_nodes),
            no_t_failures,
            record_cultivation_dist,
            terminals_buf: Vec::new(),
            scheduled_ids_buf: Vec::new(),
            children_buf: Vec::new(),
            precomputed_clifford_trees: HashMap::new(),
            remaining_ids_buf: Vec::new(),
            precomputed_terminals: Vec::new(),
            precomputed_root_info: Vec::new(),
            timers: timers,
            loop_timer: loop_timer,
            other_timer: other_timer,
            pps_pending: Vec::new(),
            remaining_parents: Vec::new(),
            pp_paths: Vec::new(),
            current_lcycle: 0,
            frame: CorrectionFrame::new(n_qubits),
            max_weight: 0,
            widened_qubits: Vec::new(),
            dynamic_m: HashSet::new(),
            qubit_locked_until: vec![0; n_qubits],
            lock_ends: Vec::new(),
            next_correction_id: n_initial_products,
            n_circuit_products: n_initial_products as usize,
            t_conjugated_bases: HashMap::new(),
            conjugated_t_products: HashSet::new(),
            free_merged_products: HashSet::new(),
        }
    }

    pub(crate) fn count_t_products(&self) -> usize {
        (0..self.input.circuit.n_products())
            .filter(|&id| self.input.circuit.product(id as i32).gate_type.is_t())
            .count()
    }

    /// Greedily assigns products to lcycles. Returns (total lcycles, total scheduled products).
    pub(crate) fn sched_circuit(&mut self) -> io::Result<(usize, usize)> {
        let _timer = fn_timer!();
        self.cultivation
            .set_lambda(self.magic_state_lambda, self.cultivation_floor)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        self.init_magic_nodes();
        let n_t_products = self.count_t_products();
        self.cultivation.t_products_remaining = n_t_products;
        // Pre-generate a large pool: ~120 samples per T gate plus one per topology node.
        self.cultivation.fill_pool(120 * n_t_products.max(1) + self.input.topo.n_nodes);
        self.precompute_terminals_and_roots();
        self.precompute_multi_term_clifford_trees();
        self.pps_pending = self.input.circuit.initial_products().cloned().collect();
        self.remaining_parents = (0..self.input.circuit.n_products())
            .map(|id| self.input.circuit.product(id as i32).parents.len())
            .collect();
        debug_sched!("Initial pps_to_sched len {}", self.pps_pending.len());
        let mut plot_lcycles = 0usize;
        let mut path_dir: Option<String> = None;
        if self.plot_option.contains("paths") {
            let dir_name = format!("{}.paths", circuit_stem(&self.input.circuit.circuit_fname));
            std::fs::create_dir_all(&dir_name)?;
            path_dir = Some(dir_name);
            plot_lcycles = 30;
        }
        let plotting = path_dir.is_some();
        let tot_pps_to_sched = self.input.circuit.n_products();
        let mut prev_pct_complete = 0usize;
        self.current_lcycle = 0;
        self.pp_paths = Vec::new();
        while !self.pps_pending.is_empty() || !self.clifford_paths.is_empty() {
            self.timers.start(self.loop_timer);
            self.current_lcycle += 1;
            info_sched!(
                "{}",
                format!(
                    "lcycle {}: {:?}",
                    self.current_lcycle,
                    self.pps_pending
                        .iter()
                        .map(|pp| format!("{}:{}", pp.id, pp.to_operator_str()))
                        .collect::<Vec<_>>(),
                )
                .cyan()
            );
            if self.schedule_lcycle(plotting) {
                self.complete_lcycle()?;
                let n_scheduled = self.scheduled_products.len();
                // Show a progress bar during the non-plotting middle section of the run.
                // The first `plot_lcycles` and last `plot_lcycles` are plotted instead.
                if self.current_lcycle >= plot_lcycles
                    && n_scheduled + plot_lcycles <= tot_pps_to_sched
                {
                    if self.current_lcycle == plot_lcycles {
                        print!("Scheduling {} products:    ", tot_pps_to_sched);
                    }
                    let pct_complete = ((n_scheduled * 100) / tot_pps_to_sched).min(100);
                    if pct_complete > prev_pct_complete {
                        // \x08 is backspace; overwrite the previous "XX%" in-place.
                        print!("\x08\x08\x08{:02}%", pct_complete);
                        std::io::stdout().flush()?;
                        prev_pct_complete = pct_complete;
                    }
                    if n_scheduled + plot_lcycles == tot_pps_to_sched {
                        print!("\n");
                    }
                } else if plotting {
                    let plot_info_str = self.stats.plot_info_str();
                    assert!(!plot_info_str.is_empty());
                    let fname_added = format!(".{}", self.current_lcycle);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    let pp_paths_plot: Vec<(PauliProduct, Rc<TreeGraph>, u32)> = self
                        .pp_paths
                        .iter()
                        .filter_map(|(pp_id, opt_tree)| {
                            opt_tree.as_ref().map(|t| {
                                let pp = self.input.circuit.product(*pp_id);
                                // Determine which cycle of a multi-lcycle product this is;
                                // clifford_paths already reflects post-advance_clifford_state.
                                let cycle: u32 = if pp.gate_type.is_clifford() {
                                    match self.clifford_paths.get(pp_id) {
                                        Some((c, _, _, _)) => {
                                            let initial =
                                                if pp.gate_type.is_cx() { 1u32 } else { 2u32 };
                                            initial - *c as u32 + 1
                                        }
                                        None => {
                                            if pp.gate_type.is_cx() {
                                                2
                                            } else {
                                                3
                                            }
                                        }
                                    }
                                } else if pp.gate_type.is_t() {
                                    1
                                } else {
                                    1
                                };
                                (pp.clone(), Rc::clone(t), cycle)
                            })
                        })
                        .collect();
                    self.input
                        .topo
                        .plot(&fname_added, &pp_paths_plot, &plot_info_str)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                }
            } else {
                debug_sched!("Could not schedule anything on lcycle {}", self.current_lcycle);
                // Only error if T gates are pending: pending Cliffords may just be waiting on a
                // temporarily-blocked correction S gate and don't need magic to make progress.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                if has_pending_t
                    && !(0..self.input.topo.n_nodes)
                        .any(|node_i| self.input.topo.is_cultivating(node_i as u16))
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "Cannot schedule on current layout".red().to_string(),
                    ));
                }
            }
            self.timers.stop(self.loop_timer);
        }
        self.print_sched_stats(self.current_lcycle);
        #[cfg(debug_assertions)]
        self.check_clifford_repetitions()?;
        #[cfg(debug_assertions)]
        self.check_schedule()?;
        Ok((self.current_lcycle, self.scheduled_products.len()))
    }

    fn init_magic_nodes(&mut self) {
        self.cultivation.init_magic_nodes(&mut self.input.topo);
    }

    /// Multi-terminal non-T products have topology-independent routing trees that
    /// can be computed once on an empty grid and reused every lcycle.
    fn should_precompute(pp: &PauliProduct) -> bool {
        !pp.gate_type.is_t() && pp.operators.len() > 1
    }

    fn precompute_steiner_tree(&mut self, pp: &PauliProduct) -> Option<TreeGraph> {
        if !self.terminal_nodes(pp.id) {
            return None;
        }
        let root_ids = self.root_nodes(pp.id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            return None;
        }
        self.stree_computation.compute(
            &self.input.topo,
            &self.used,
            &root_ids,
            &self.terminals_buf,
            pp.gate_type,
        )
    }

    /// Fills `terminals_buf` from precomputed IDs. Returns false early if any terminal
    /// is occupied or all its root candidates are occupied (cheap pre-check before `root_nodes`).
    #[inline]
    fn terminal_nodes(&mut self, pp_id: i32) -> bool {
        let pp_id = pp_id as usize;
        self.terminals_buf.clear();
        let terminals = &self.precomputed_terminals[pp_id];
        let root_info = &self.precomputed_root_info[pp_id];
        for (i, &node_id) in terminals.iter().enumerate() {
            if self.used[node_id as usize] {
                info_sched!("  Node {} is already used", node_id);
                return false;
            }
            let (_, preferred, side) = &root_info[i];
            // Only reject when candidates exist and all are occupied; empty preferred+side
            // means the gate uses sched_s_sx (no root needed), so don't reject vacuously.
            if (!preferred.is_empty() || !side.is_empty())
                && preferred.iter().all(|&rid| self.used[rid as usize])
                && side.iter().all(|&rid| self.used[rid as usize])
            {
                info_sched!("  No unused root candidates for node {}", node_id);
                return false;
            }
            self.terminals_buf.push(node_id);
        }
        true
    }

    /// Returns routing nodes adjacent to each terminal.
    /// For Y-basis operators the paired X/Z data nodes can share a single routing node
    /// (the "preferred" root), decrementing `unmatched_count` by 2; if none is free,
    /// each terminal falls back to a side nb.
    fn root_nodes(&self, pp_id: usize, terminals: &[u16]) -> Vec<u16> {
        let root_info = &self.precomputed_root_info[pp_id];
        let mut root_ids: Vec<u16> = Vec::new();
        let mut unmatched_count: usize = terminals.len();
        for (i, _node_id) in terminals.iter().enumerate() {
            let (is_paired, preferred, side) = &root_info[i];
            let mut pair_found = false;
            if *is_paired {
                for &nb_id in preferred {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    // One root serves both X and Z terminals of the Y pair.
                    unmatched_count = unmatched_count.saturating_sub(2);
                    pair_found = true;
                    break;
                }
            }
            if !pair_found {
                let fallback = if *is_paired { side.as_slice() } else { preferred.as_slice() };
                for &nb_id in fallback {
                    if self.used[nb_id as usize] {
                        continue;
                    }
                    if !root_ids.contains(&nb_id) {
                        root_ids.push(nb_id);
                    }
                    unmatched_count = unmatched_count.saturating_sub(1);
                    break;
                }
            }
        }
        if unmatched_count > 0 {
            debug_sched!(
                "    could not find root nodes for {} unmatched terminals",
                unmatched_count
            );
            return Vec::new();
        }
        root_ids
    }

    fn precompute_multi_term_clifford_trees(&mut self) {
        let _timer = fn_timer!("precompute_clifford_trees");
        self.used.fill(false);
        let n_products = self.input.circuit.n_products();
        let mut n_precomputed = 0;
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            if Self::should_precompute(&pp) {
                if let Some(tree) = self.precompute_steiner_tree(&pp) {
                    self.precomputed_clifford_trees.insert(pp.id, Rc::new(tree));
                    n_precomputed += 1;
                } else {
                    eprintln!(
                        "{}",
                        format!("Warning: failed to precompute tree for {}", pp).yellow()
                    );
                }
            }
        }
        println!("Precomputed {} multi-term Clifford trees", n_precomputed);
    }

    /// Computes root candidate info for a set of already-computed terminal node IDs.
    /// Extracted from `precompute_terminals_and_roots` so it can be called for dynamically
    /// constructed operator lists (e.g. after S^k conjugation of T gate operators).
    fn compute_root_info_for_terminals(
        topo: &TopoGraph, terminals: &[u16],
    ) -> Vec<(bool, Vec<u16>, Vec<u16>)> {
        let mut root_info = Vec::with_capacity(terminals.len());
        for &term_id in terminals {
            let node = topo.node(term_id);
            let is_paired =
                node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
            let mut preferred = Vec::new();
            let mut side = Vec::new();
            if is_paired {
                let is_x = topo.label(term_id).contains('X');
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    if !nb.is_routing() {
                        continue;
                    }
                    if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                        if Self::joins_routing(topo, nb_id) {
                            preferred.push(nb_id);
                        }
                    } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        side.push(nb_id);
                    }
                }
            } else {
                for &nb_id in node.nbs_slice() {
                    let nb = topo.node(nb_id);
                    if nb.is_routing() && nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        preferred.push(nb_id);
                    }
                }
            }
            root_info.push((is_paired, preferred, side));
        }
        root_info
    }

    /// Whether an injected correction gate still runs on any of these qubits.
    fn locked(&self, ops: &[Operator]) -> bool {
        !self.lock_ends.is_empty()
            && ops.iter().any(|o| self.qubit_locked_until[o.qubit as usize] > self.current_lcycle)
    }

    /// Points a product's precomputed terminals and roots at `ops`.
    fn set_terminals(&mut self, pp_id: i32, ops: &[Operator]) {
        let terminals = operators_to_node_ids(&self.input.topo, ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals[pp_id as usize] = terminals;
        self.precomputed_root_info[pp_id as usize] = root_info;
    }

    /// The operators a product scheduled this lcycle runs about.
    fn executed_ops(&self, pp_id: i32) -> &[Operator] {
        match self.t_conjugated_bases.get(&pp_id) {
            Some(v) => v,
            None => &self.input.circuit.product(pp_id).operators,
        }
    }

    /// Whether a product already running this lcycle, placed earlier in it or
    /// carried from an earlier one, touches any of these qubits. Circuit products
    /// cannot share a qubit while both are ready, but a widened product or an
    /// injected correction has no place in the dependency graph.
    fn running_on(&self, ops: &[Operator]) -> bool {
        self.pp_paths.iter().any(|(id, _)| {
            self.executed_ops(*id).iter().any(|a| ops.iter().any(|b| a.qubit == b.qubit))
        })
    }

    /// Passes a T gate or measurement through the correction frame. If it can
    /// run, its terminals are pointed at the operators it runs about (recorded in
    /// `t_conjugated_bases` when they differ).
    fn conjugate_rotation(&mut self, pp_id: i32) -> Rotation {
        let orig = &self.input.circuit.product(pp_id).operators;
        let was_conjugated = self.conjugated_t_products.contains(&pp_id);
        // Common case: nothing owed on these qubits, and nothing to restore.
        if self.frame.clear_on(orig) && !was_conjugated {
            return Rotation::Run;
        }
        let orig = sorted(orig);
        // The transpiler's limit does not bound measurements, so a product may be
        // written wider than it. It may always run as wide as written.
        let limit = if self.max_weight == 0 { 0 } else { self.max_weight.max(orig.len()) };
        let ops = match self.frame.pass_rotation(&orig, limit) {
            Pass::Blocked(order) => return Rotation::Emit(order),
            Pass::Run(ops) => ops,
        };
        // Qubits it widened onto must be free this lcycle.
        let extra: Vec<Operator> = ops
            .iter()
            .filter(|o| orig.binary_search_by_key(&o.qubit, |x| x.qubit).is_err())
            .cloned()
            .collect();
        if !extra.is_empty()
            && (self.locked(&extra)
                || self.running_on(&extra)
                || extra.iter().any(|o| self.widened_qubits.contains(&o.qubit)))
        {
            return Rotation::Wait;
        }
        if ops == orig {
            if was_conjugated {
                self.set_terminals(pp_id, &orig);
                self.conjugated_t_products.remove(&pp_id);
            }
        } else {
            info_sched!("  {} conjugated by the correction frame to {:?}", pp_id, ops);
            self.set_terminals(pp_id, &ops);
            self.conjugated_t_products.insert(pp_id);
            self.t_conjugated_bases.insert(pp_id, ops);
        }
        Rotation::Run
    }

    /// Whether a candidate shared root links to at least one other routing node.
    ///
    /// A shared root that does not is a dead end: in the compact bus layout, the
    /// node above a top-row patch's Z pair is a border cell between two
    /// cultivators. A product rooted there could reach only those two, so with
    /// neither ready the scheduler found no path and reported a deadlock, though
    /// the side bus columns route it. Such a root is left out, and the pair falls
    /// back to side roots.
    fn joins_routing(topo: &TopoGraph, root_id: u16) -> bool {
        topo.node(root_id).nbs_slice().iter().any(|&id| topo.node(id).is_routing())
    }

    /// Precomputes terminal node IDs and root candidates for every product.
    ///
    /// For each terminal data node the root candidates are split into:
    /// - `preferred`: the routing node between the X and Z patches of a Y-pair
    ///   (X looks downward toward Z; Z looks upward toward X), or the same-row
    ///   side nb for unpaired terminals.
    /// - `side`: same-row side nbs used as fallback for Y-paired terminals.
    fn precompute_terminals_and_roots(&mut self) {
        let _timer = fn_timer!("precompute_terminals_and_roots");
        let n_products = self.input.circuit.n_products();
        self.precomputed_terminals = vec![Vec::new(); n_products];
        self.precomputed_root_info = vec![Vec::new(); n_products];
        for pp_id in 0..n_products {
            let pp = self.input.circuit.product(pp_id as i32).clone();
            let terminals = operators_to_node_ids(&self.input.topo, &pp.operators);
            let mut root_info: Vec<(bool, Vec<u16>, Vec<u16>)> =
                Vec::with_capacity(terminals.len());
            for &term_id in &terminals {
                let node = self.input.topo.node(term_id);
                // is_paired: the operator is Y-basis, so both its X and Z data nodes are terminals.
                let is_paired =
                    node.paired_data_id.map(|pid| terminals.contains(&pid)).unwrap_or(false);
                let mut preferred: Vec<u16> = Vec::new();
                let mut side: Vec<u16> = Vec::new();
                if is_paired {
                    // X nodes look downward (toward paired Z), Z nodes look upward.
                    let is_x = self.input.topo.label(term_id).contains('X');
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if !nb.is_routing() {
                            continue;
                        }
                        if (is_x && nb.pos.1 < node.pos.1) || (!is_x && nb.pos.1 > node.pos.1) {
                            if Self::joins_routing(&self.input.topo, nb_id) {
                                preferred.push(nb_id);
                            }
                        } else if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                            side.push(nb_id);
                        }
                    }
                } else {
                    // Unpaired terminal: only same-row side nbs are valid roots.
                    for &nb_id in node.nbs_slice() {
                        let nb = self.input.topo.node(nb_id);
                        if nb.is_routing() && nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                            preferred.push(nb_id);
                        }
                    }
                }
                root_info.push((is_paired, preferred, side));
            }
            self.precomputed_terminals[pp_id] = terminals;
            self.precomputed_root_info[pp_id] = root_info;
        }
        println!("Precomputed terminals and root candidates for {} products", n_products);
    }

    fn mark_nodes_used(&mut self, node_ids: &[u16]) {
        for &node_id in node_ids {
            self.used[node_id as usize] = true;
            let node = self.input.topo.node(node_id);
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
        }
    }

    /// Reserves the nodes of an in-progress multi-lcycle product for this lcycle.
    fn carry_forward_path(
        &mut self, pp_id: i32, node_ids: &[u16], opt_tree: Option<Rc<TreeGraph>>,
    ) {
        self.trace_footprint(pp_id, node_ids);
        self.mark_nodes_used(node_ids);
        self.pp_paths.push((pp_id, opt_tree));
    }

    /// Schedules as many products as possible in one lcycle; returns false if nothing scheduled.
    fn schedule_lcycle(&mut self, plotting: bool) -> bool {
        let _timer = accum_start!(self.timers);
        let start = self.cycle_times.is_some().then(std::time::Instant::now);
        let scheduled = self.schedule_lcycle_body(plotting);
        if let (Some(t0), Some(times)) = (start, self.cycle_times.as_mut()) {
            times.push(t0.elapsed().as_nanos() as u64);
        }
        scheduled
    }

    fn schedule_lcycle_body(&mut self, plotting: bool) -> bool {
        self.timers.start(self.other_timer);
        if !self.lock_ends.is_empty() {
            let now = self.current_lcycle;
            self.lock_ends.retain(|&end| end > now);
        }
        let mut n_avail_magic = self.update_cultivators();
        #[cfg(feature = "diagnostics")]
        self.update_ready_since();
        self.t_conjugated_bases.clear();
        self.widened_qubits.clear();
        self.dynamic_m.clear();
        let initial_magic = n_avail_magic;
        self.pp_paths.clear();
        self.used.fill(false);
        // Collect carry-forward data before the loop to release borrows on the map.
        let clifford_carry: Vec<(i32, Vec<u16>, Option<Rc<TreeGraph>>)> = self
            .clifford_paths
            .values()
            .map(|(_, pp, node_ids, opt_tree)| {
                (pp.id, node_ids.clone(), opt_tree.as_ref().map(Rc::clone))
            })
            .collect();
        for (pp_id, node_ids, opt_tree) in clifford_carry {
            self.carry_forward_path(pp_id, &node_ids, opt_tree);
        }
        let carry_forward_count = self.clifford_paths.len();
        let tot_available = carry_forward_count + self.pps_pending.len();
        info_sched!("  Remaining to schedule: {}", self.pps_pending.len());
        self.sched_precomputed(plotting);
        self.timers.stop(self.other_timer);
        self.sched_remaining(&mut n_avail_magic, plotting);
        #[cfg(feature = "diagnostics")]
        {
            let topo = &self.input.topo;
            self.stats.tot_ancilla_reserved += (0..topo.n_nodes)
                .filter(|&i| self.used[i] && topo.node(i as u16).node_type != NodeType::Data)
                .count();
        }
        self.stats.update(
            self.current_lcycle,
            self.pp_paths.len(),
            tot_available,
            initial_magic,
            n_avail_magic,
            plotting,
        );
        if self.pp_paths.is_empty() {
            if n_avail_magic > 0 {
                // Only panic when T gates couldn't consume available magic; pending Cliffords
                // whose correction S gate emission failed will simply retry next lcycle.
                let has_pending_t = self.pps_pending.iter().any(|pp| pp.gate_type.is_t());
                if has_pending_t {
                    panic!(
                        "{}",
                        format!(
                            "lcycle {}: Cannot schedule products [{}] on current layout ({} magic)",
                            self.current_lcycle,
                            self.pps_pending
                                .iter()
                                .map(|pp| pp.to_operator_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            n_avail_magic,
                        )
                        .red()
                    );
                }
            }
            false
        } else {
            true
        }
    }

    /// Stamps each magic node with the lcycle its state became ready, and clears the
    /// stamp of every node that is cultivating again (consumed, routed over or reset).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn update_ready_since(&mut self) {
        let topo = &self.input.topo;
        for i in 0..topo.n_nodes {
            let ready =
                topo.node(i as u16).node_type == NodeType::Magic && topo.cultivation_times[i] == 0;
            if !ready {
                self.ready_since[i] = usize::MAX;
            } else if self.ready_since[i] == usize::MAX {
                self.ready_since[i] = self.current_lcycle;
            }
        }
    }

    /// Charges a consumed magic state the lcycles it waited ready, excluding the
    /// consuming lcycle, which the path reservation already counts.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn record_consumed_wait(&mut self, magic_id: u16) {
        let since = self.ready_since[magic_id as usize];
        if since != usize::MAX {
            self.stats.tot_consumed_wait += self.current_lcycle - since;
        }
    }

    pub(crate) fn update_cultivators(&mut self) -> usize {
        let _timer = accum_start!(self.timers);
        let n_avail_magic =
            self.cultivation.update_cultivators(&mut self.input.topo, &self.used, &self.consumed);
        #[cfg(feature = "diagnostics")]
        self.consumed.fill(false);
        info_sched!("  Available magic {}", n_avail_magic);
        n_avail_magic
    }

    /// First pass: schedule products with precomputed Clifford trees.
    /// Products whose tree nodes are all free are scheduled immediately.
    /// Products whose data nodes are blocked mark those nodes as used to prevent
    /// other products from routing through them this lcycle.
    fn sched_precomputed(&mut self, plotting: bool) {
        let _timer = accum_start!(self.timers);
        self.remaining_ids_buf.clear();
        self.remaining_ids_buf.extend(self.pps_pending.iter().map(|pp| pp.id));
        let mut to_remove: Vec<i32> = Vec::new();
        // Clone into a local Vec to release the borrow on `remaining_ids_buf`
        // before the loop body calls `&mut self` methods.
        let ids_to_process: Vec<i32> = self.remaining_ids_buf.clone();
        for &pp_id in &ids_to_process {
            // Clone the Rc to end the immutable borrow on `precomputed_clifford_trees`.
            let Some(tree) = self.precomputed_clifford_trees.get(&pp_id).map(Rc::clone) else {
                continue;
            };
            // An owed correction this Clifford or measurement does not commute with
            // must be applied first: a product on fixed precomputed patches cannot be
            // conjugated the way a T gate's basis can. Emit it and hold the product.
            if self.locked(&self.input.circuit.product(pp_id).operators) {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                continue;
            }
            let pp = self.input.circuit.product(pp_id);
            if pp.gate_type.is_m()
                && (!self.frame.clear_on(&pp.operators)
                    || self.conjugated_t_products.contains(&pp_id))
            {
                // Owed corrections conjugate it, possibly widening it, so its
                // precomputed patches no longer fit: it is routed like a T gate.
                self.dynamic_m.insert(pp_id);
                continue;
            }
            if let Some(order) = self.clifford_blockers(pp_id) {
                self.corr.before_multi += self.emit_corrections(order, plotting);
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                continue;
            }
            let all_free = tree.iter_nodes().all(|nid| !self.used[nid as usize]);
            if all_free {
                to_remove.push(pp_id);
                debug_assert!(self.clifford_blockers(pp_id).is_none());
                let node_ids: Vec<u16> = tree.iter_nodes().collect();
                let (_tree_n_nodes, _tree_n_edges) = (tree.n_nodes, tree.n_edges);
                let opt_tree: Option<Rc<TreeGraph>> =
                    if plotting { Some(Rc::clone(&tree)) } else { None };
                // Drop `tree` before the `&mut self` call to `mark_nodes_used`.
                drop(tree);
                self.trace_footprint(pp_id, &node_ids);
                self.mark_nodes_used(&node_ids);
                info_sched!(
                    "  Scheduled product {} (precomputed) with {} nodes and {} edges",
                    self.input.circuit.product(pp_id),
                    _tree_n_nodes,
                    _tree_n_edges
                );
                self.pp_paths.push((pp_id, opt_tree));
            } else {
                let pp = self.input.circuit.product(pp_id);
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
            }
        }
        self.pps_pending.retain(|pp| !to_remove.contains(&pp.id));
    }

    /// Marks the data nodes of a blocked product as used so that other products
    /// cannot route through them this lcycle (takes separate params to avoid
    /// borrow conflicts in caller loops).
    fn mark_blocked_product_as_used(used: &mut Vec<bool>, topo: &TopoGraph, pp: &PauliProduct) {
        for node_id in operators_to_node_ids(topo, &pp.operators) {
            used[node_id as usize] = true;
        }
    }

    /// Emits the owed corrections in `order` as S gates, stopping at the first that
    /// cannot be placed this lcycle (an emission later in the order may need it
    /// done first). Returns how many were placed.
    fn emit_corrections(&mut self, order: Vec<usize>, plotting: bool) -> usize {
        let mut n = 0;
        for slot in order {
            if !self.try_emit_correction(slot, plotting) {
                break;
            }
            n += 1;
        }
        n
    }

    /// Places an owed correction S_P as a gate. About X or Z on one qubit it is
    /// an S on the data patch and a neighbouring ancilla. About Y, or on several
    /// qubits, it is routed as a tree over P's patch edges. Either way it holds its
    /// patches for three lcycles, like every S. The correction leaves the frame
    /// only once placed.
    fn try_emit_correction(&mut self, slot: usize, plotting: bool) -> bool {
        let ops = self.frame.get(slot).ops.clone();
        if self.locked(&ops) {
            return false;
        }
        // Nor may it share a qubit with a product already running this lcycle:
        // that product may hold another edge of the qubit, which the patch check
        // would not catch.
        if self.running_on(&ops) || ops.iter().any(|o| self.widened_qubits.contains(&o.qubit)) {
            return false;
        }
        info_sched!("  Emitting correction {:?}", ops);
        let s_id = self.next_correction_id;
        let s_pp = PauliProduct {
            id: s_id,
            gate_type: GateType::S,
            operators: ops.clone(),
            parents: vec![],
            children: vec![],
            max_qubit: ops.iter().map(|o| o.qubit).max().unwrap_or(0),
        };
        self.input.circuit.pps.push(s_pp);
        let terminals = operators_to_node_ids(&self.input.topo, &ops);
        let root_info = Self::compute_root_info_for_terminals(&self.input.topo, &terminals);
        self.precomputed_terminals.push(terminals);
        self.precomputed_root_info.push(root_info);
        let placed = if self.terminal_nodes(s_id) {
            if ops.len() == 1 && ops[0].basis != 'Y' {
                self.sched_s_sx(s_id, plotting)
            } else {
                self.sched_t_or_multi(s_id, plotting)
            }
        } else {
            PathResult::NoPath
        };
        match placed {
            PathResult::PathFound(opt_graph) => {
                self.next_correction_id += 1;
                self.correction_gates_emitted += 1;
                if ops.len() == 1 {
                    self.corr.emitted[match ops[0].basis {
                        'X' => 0,
                        'Y' => 1,
                        _ => 2,
                    }] += 1;
                } else {
                    self.corr.emitted_multi += 1;
                }
                if let Some(ref g) = opt_graph {
                    let node_ids: Vec<u16> = g.iter_nodes().collect();
                    self.mark_nodes_used(&node_ids);
                }
                self.pp_paths.push((s_id, opt_graph.map(Rc::new)));
                self.frame.emitted(slot);
                for o in &ops {
                    self.qubit_locked_until[o.qubit as usize] = self.current_lcycle + 3;
                }
                self.lock_ends.push(self.current_lcycle + 3);
                true
            }
            PathResult::NoPath => {
                info_sched!("  Correction {:?} could not be placed, retry next lcycle", ops);
                self.input.circuit.pps.pop();
                self.precomputed_terminals.pop();
                self.precomputed_root_info.pop();
                false
            }
        }
    }

    /// Merges the owed correction in `slot`, a single-qubit S about the same axis,
    /// into the blocked S or SX product `pp_id`: the two share a generator, so the
    /// powers add. An even net power leaves the identity or a Pauli and no gate is
    /// scheduled. An odd one is still one gate, `pp_id` itself. Returns `true` if
    /// `pp_id` was handled this lcycle, `false` if its gate could not be routed
    /// (the correction stays owed and the caller retries next lcycle).
    fn try_merge_clifford_correction(
        &mut self, pp_id: i32, slot: usize, net: u8, plotting: bool,
    ) -> bool {
        if net % 2 == 0 {
            self.corr.merged_free += 1;
            self.frame.emitted(slot);
            self.free_merged_products.insert(pp_id);
            self.pp_paths.push((pp_id, None));
            return true;
        }
        if !self.terminal_nodes(pp_id) {
            return false;
        }
        match self.sched_s_sx(pp_id, plotting) {
            PathResult::PathFound(opt_graph) => {
                self.corr.merged_gate += 1;
                if let Some(ref g) = opt_graph {
                    let node_ids: Vec<u16> = g.iter_nodes().collect();
                    self.mark_nodes_used(&node_ids);
                }
                self.pp_paths.push((pp_id, opt_graph.map(Rc::new)));
                self.frame.emitted(slot);
                true
            }
            PathResult::NoPath => false,
        }
    }

    /// Second pass: greedily schedule T gates, measurements, and S/SX gates.
    /// T gates are skipped when no magic state is available.
    fn sched_remaining(&mut self, n_avail_magic: &mut usize, plotting: bool) {
        let _timer = accum_start!(self.timers);
        for i in 0..self.pps_pending.len() {
            let pp_id = self.pps_pending[i].id;
            let pp = self.input.circuit.product(pp_id);
            if Self::should_precompute(pp) && !self.dynamic_m.contains(&pp_id) {
                continue;
            }
            let (pp_id, gate_type) = (pp.id, pp.gate_type);
            if self.locked(&pp.operators)
                || (!self.widened_qubits.is_empty()
                    && pp.operators.iter().any(|o| self.widened_qubits.contains(&o.qubit)))
            {
                Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                continue;
            }
            // An S or SX meets owed corrections on its qubit. A lone single-qubit
            // one about the same axis shares its generator and merges. One it does
            // not commute with must be emitted first. The rest commute and stay owed.
            if gate_type.is_s() || gate_type.is_sx() {
                let qubit = self.input.circuit.product(pp_id).operators[0].qubit;
                let axis = if gate_type.is_s() { 'Z' } else { 'X' };
                let g = [Operator { qubit, basis: axis }];
                if !self.frame.clear_on(&g) {
                    if let Some((slot, net)) = self.frame.merge_candidate(qubit, axis) {
                        if !self.try_merge_clifford_correction(pp_id, slot, net, plotting) {
                            let pp = self.input.circuit.product(pp_id);
                            Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                        }
                        continue;
                    }
                    if let Some(order) = self.frame.pass_clifford(&g) {
                        info_sched!("  Clifford {:?} {} waits for corrections", gate_type, pp_id);
                        // Block the qubit only if a correction was placed. Otherwise leave
                        // it free so T gates can still route through it (avoids deadlock).
                        if self.emit_corrections(order, plotting) > 0 {
                            let pp = self.input.circuit.product(pp_id);
                            Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                        }
                        continue;
                    }
                }
            }
            // A T gate or measurement runs about its operators conjugated by the owed
            // corrections, unless that would take it past the weight limit, when the
            // correction is emitted first, or it widened onto a busy qubit.
            if gate_type.is_t() || gate_type.is_m() {
                match self.conjugate_rotation(pp_id) {
                    Rotation::Run => {}
                    r => {
                        if let Rotation::Emit(order) = r {
                            self.corr.over_limit += self.emit_corrections(order, plotting);
                        }
                        let pp = self.input.circuit.product(pp_id);
                        Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
                        continue;
                    }
                }
            }
            if *n_avail_magic > 0 || !gate_type.is_t() {
                info_sched!("  Trying to schedule product {}", self.input.circuit.product(pp_id));
                let result = if !self.terminal_nodes(pp_id) {
                    info_sched!(
                        "    Cannot schedule {}: no data nodes found in working graph",
                        pp_id
                    );
                    PathResult::NoPath
                } else if self.terminals_buf.len() == 1 && gate_type.is_m() {
                    self.sched_measurement(pp_id, plotting)
                } else if gate_type.is_s() || gate_type.is_sx() {
                    self.sched_s_sx(pp_id, plotting)
                } else {
                    self.sched_t_or_multi(pp_id, plotting)
                };
                if let PathResult::PathFound(opt_graph) = result {
                    info_sched!("  Scheduled product {}", self.input.circuit.product(pp_id));
                    if gate_type.is_t() && self.t_conjugated_bases.contains_key(&pp_id) {
                        self.corr.t_conjugated += 1;
                    }
                    if gate_type.is_m() && self.t_conjugated_bases.contains_key(&pp_id) {
                        self.corr.m_conjugated += 1;
                    }
                    if let Some(ops) = self.t_conjugated_bases.get(&pp_id) {
                        let orig = &self.input.circuit.product(pp_id).operators;
                        let mut wide = false;
                        for o in ops {
                            if !orig.iter().any(|x| x.qubit == o.qubit) {
                                self.widened_qubits.push(o.qubit);
                                wide = true;
                            }
                        }
                        self.corr.widened += wide as usize;
                    }
                    if gate_type.is_t() {
                        // A Steiner tree names its cultivator as the root; a single-qubit
                        // T found by A* in non-plotting mode returns no tree.
                        let magic_id = match opt_graph {
                            Some(ref g) => g.root_node_id,
                            None => Some(self.astar.last_magic_id),
                        };
                        if let Some(id) = magic_id {
                            #[cfg(feature = "diagnostics")]
                            {
                                self.record_consumed_wait(id);
                                self.consumed[id as usize] = true;
                            }
                            if self.trace.is_some() && opt_graph.is_none() {
                                self.trace_untreed
                                    .insert(pp_id, (self.astar.last_path.clone(), id));
                            }
                        }
                    }
                    if let Some(ref pp_graph) = opt_graph {
                        let node_ids: Vec<u16> = pp_graph.iter_nodes().collect();
                        self.mark_nodes_used(&node_ids);
                    }
                    self.pp_paths.push((pp_id, opt_graph.map(Rc::new)));
                    if gate_type.is_t() {
                        *n_avail_magic -= 1;
                        self.stats.inc_t();
                    }
                    continue;
                }
            }
            info_sched!("  Could not schedule {} on graph", pp_id);
            let pp = self.input.circuit.product(pp_id);
            Self::mark_blocked_product_as_used(&mut self.used, &self.input.topo, pp);
        }
    }

    fn sched_measurement(&mut self, _pp_id: i32, plotting: bool) -> PathResult {
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            info_sched!(
                "    Cannot schedule {}: node for M {} is used",
                _pp_id,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        if !plotting {
            self.used[node_id as usize] = true;
            self.stats.inc_with_cultivation(
                node.node_type,
                self.input.topo.cultivation_times[node_id as usize],
            );
            self.trace_footprint(_pp_id, &[node_id]);
            return PathResult::PathFound(None);
        }
        let mut g = TreeGraph::new(self.input.topo.n_nodes);
        g.add_node(node, self.input.topo.label(node_id));
        PathResult::PathFound(Some(g))
    }

    /// Schedules an S or SX gate: data node plus one same-row ancilla nb.
    /// S/SX require two adjacent nodes in the same row (the data qubit and its
    /// horizontal routing nb).
    fn sched_s_sx(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        let node_id = self.terminals_buf[0];
        let node = self.input.topo.node(node_id);
        if self.used[node.id as usize] {
            let _gate_type = self.input.circuit.product(pp_id).gate_type;
            info_sched!(
                "    Cannot schedule {}: node for {:?} {} is used",
                pp_id,
                _gate_type,
                self.input.topo.label(node_id)
            );
            return PathResult::NoPath;
        }
        for &nb_id in node.nbs_slice() {
            let nb = self.input.topo.node(nb_id);
            if nb.pos.1 == node.pos.1 {
                info_sched!(
                    "    product {} on node {} has available ancilla {}",
                    self.input.circuit.product(pp_id),
                    self.input.topo.label(node_id),
                    self.input.topo.label(nb_id)
                );
                if !self.used[nb_id as usize] {
                    if !plotting {
                        self.used[node_id as usize] = true;
                        self.used[nb_id as usize] = true;
                        self.stats.inc_with_cultivation(
                            node.node_type,
                            self.input.topo.cultivation_times[node_id as usize],
                        );
                        self.stats.inc_with_cultivation(
                            nb.node_type,
                            self.input.topo.cultivation_times[nb_id as usize],
                        );
                        self.trace_footprint(pp_id, &[node_id, nb_id]);
                        self.s_sx_nodes.insert(pp_id, vec![node_id, nb_id]);
                        return PathResult::PathFound(None);
                    }
                    let mut g = TreeGraph::new(self.input.topo.n_nodes);
                    g.add_node(node, self.input.topo.label(node_id));
                    g.add_node(nb, self.input.topo.label(nb_id));
                    g.add_edge(node_id, nb_id);
                    return PathResult::PathFound(Some(g));
                }
            }
        }
        info_sched!("    Cannot schedule S/SX {}: no available ancilla", pp_id);
        PathResult::NoPath
    }

    fn sched_t_or_multi(&mut self, pp_id: i32, plotting: bool) -> PathResult {
        debug_assert!(!self.terminals_buf.iter().any(|node_id| self.used[*node_id as usize]));
        let root_ids = self.root_nodes(pp_id as usize, &self.terminals_buf[..]);
        if root_ids.is_empty() {
            info_sched!("    Cannot schedule {}: no roots available", pp_id);
            return PathResult::NoPath;
        }
        let pp = self.input.circuit.product(pp_id);
        let gate_type = pp.gate_type;
        // The frame may have widened or narrowed it.
        let n_operators = self.executed_ops(pp_id).len();
        let result = if gate_type.is_t() && n_operators == 1 {
            self.astar.compute(
                &self.terminals_buf[..],
                &root_ids[..],
                &self.input.topo,
                &mut self.used,
                &self.cultivation.ready_magic_positions,
                plotting,
            )
        } else {
            // Injected multi-qubit corrections are routed here, not from fixed trees.
            debug_assert!(
                pp_id as usize >= self.n_circuit_products
                    || self.dynamic_m.contains(&pp_id)
                    || !Self::should_precompute(self.input.circuit.product(pp_id)),
                "should_precompute product {:?} reached Steiner path",
                pp_id
            );
            match self.stree_computation.compute(
                &self.input.topo,
                &self.used,
                &root_ids,
                &self.terminals_buf,
                gate_type,
            ) {
                Some(tree) => PathResult::PathFound(Some(tree)),
                None => PathResult::NoPath,
            }
        };
        if let PathResult::PathFound(opt_g) = result {
            return PathResult::PathFound(opt_g);
        }
        info_sched!("    Cannot schedule {}: no path found", pp_id);
        PathResult::NoPath
    }

    /// Post-lcycle bookkeeping: remove scheduled products, unlock children,
    /// advance Clifford state, and process T-gate coin-flip outcomes.
    fn complete_lcycle(&mut self) -> io::Result<()> {
        let _timer = accum_start!(self.timers);
        // Before T outcomes are processed, while each product's terminals still
        // reflect the basis it was routed with.
        self.write_trace_lcycle()?;
        self.scheduled_ids_buf.clear();
        self.scheduled_ids_buf.extend(self.pp_paths.iter().map(|(id, _)| *id));
        self.pps_pending.retain(|pp| !self.scheduled_ids_buf.contains(&pp.id));
        debug_sched!("After purge, pps_to_sched len {}", self.pps_pending.len());
        let t_newly_scheduled = self
            .pp_paths
            .iter()
            .filter(|(id, _)| self.input.circuit.product(*id).gate_type.is_t())
            .count();
        self.cultivation.t_products_remaining =
            self.cultivation.t_products_remaining.saturating_sub(t_newly_scheduled);
        self.process_t_gate_outcomes();
        self.unlock_children();
        self.advance_clifford_state();
        debug_sched!(
            "After inserting previous lcycle cliffords, pps_to_sched len {}",
            self.pps_pending.len()
        );
        self.pps_pending
            .extend(self.children_buf.iter().map(|&id| self.input.circuit.product(id).clone()));
        debug_sched!(
            "After adding {} children, pps_to_sched len {}",
            self.children_buf.len(),
            self.pps_pending.len()
        );
        let lcycle_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        self.lcycle_scheduled.push((self.current_lcycle, lcycle_ids));
        #[cfg(debug_assertions)]
        self.check_lcycle()?;
        self.scheduled_products.extend(self.pp_paths.iter().map(|(id, _)| *id));
        Ok(())
    }

    /// Coin-flip T gate outcomes and update per-qubit correction tracking.
    ///
    /// T gates complete in one lcycle regardless of outcome. A failed T gate
    /// owes S about the operators it ran about, which the correction frame carries;
    /// no recovery lcycle is needed.
    fn process_t_gate_outcomes(&mut self) {
        let pp_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        for pp_id in pp_ids {
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_t() {
                continue;
            }
            if self.no_t_failures || self.rng_uniform.gen_bool(0.5) {
                info_sched!("  T gate {} succeeded", pp_id);
            } else {
                self.t_gate_failures += 1;
                info_sched!("  T gate {} failed (50% probability), S correction owed", pp_id);
                // The correction is S about the operators the gate actually ran about.
                let ops = match self.t_conjugated_bases.get(&pp_id) {
                    Some(ops) => ops.clone(),
                    None => sorted(&self.input.circuit.product(pp_id).operators),
                };
                if let Some(w) = self.trace.as_mut() {
                    let s: Vec<String> =
                        ops.iter().map(|o| format!("{}:{}", o.qubit, o.basis)).collect();
                    writeln!(w, "f {} {} {}", self.current_lcycle, pp_id, s.join(","))
                        .expect("writing the schedule trace");
                }
                if self.frame.add_failure(&ops) == Added::Cancelled {
                    self.corr.classical += 1;
                }
            }
        }
    }

    /// Decrements `remaining_parents` for each completed product and collects
    /// newly-ready children into `children_buf`.
    fn unlock_children(&mut self) {
        self.children_buf.clear();
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            let pp = self.input.circuit.product(pp_id);
            let gate_type = pp.gate_type;
            if gate_type.is_clifford() && !self.free_merged_products.contains(&pp_id) {
                match self.clifford_paths.get(&pp_id) {
                    // S/SX: count==2 means this is the second of three lcycles;
                    // children are not unlocked until the third (final) lcycle.
                    Some((count, _, _, _)) if *count == 2 => {
                        debug_assert!(gate_type.is_s() || gate_type.is_sx());
                        continue;
                    }
                    // count==1 (CX) or count==1 (S/SX final): fall through to unlock.
                    // None means this is the first lcycle: children not yet unlocked.
                    None => continue,
                    _ => {}
                }
            }
            // Free-merged S products (see try_merge_s_correction) are single-lcycle,
            // zero-node events - fall straight through to unlocking their children.
            let children: Vec<i32> = self.input.circuit.product(pp_id).children.clone();
            for child_id in children {
                self.remaining_parents[child_id as usize] -= 1;
                if self.remaining_parents[child_id as usize] == 0
                    && !self.children_buf.contains(&child_id)
                {
                    self.children_buf.push(child_id);
                }
            }
        }
    }

    /// Advances multi-lcycle Clifford state: decrements remaining-lcycle counters
    /// and inserts new entries for first-seen Clifford products.
    ///
    /// CX requires 2 consecutive lcycles (initial count = 1, removed when it hits 0).
    /// S/SX require 3 consecutive lcycles (initial count = 2).
    fn advance_clifford_state(&mut self) {
        let pp_paths_snapshot: Vec<(i32, Option<Rc<TreeGraph>>)> =
            self.pp_paths.iter().map(|(id, opt)| (*id, opt.as_ref().map(Rc::clone))).collect();
        for (pp_id, opt_pp_path) in &pp_paths_snapshot {
            let pp_id = *pp_id;
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_clifford() || self.free_merged_products.contains(&pp_id) {
                // Free-merged S products (see try_merge_s_correction) are already complete
                // after their single lcycle - never enter the multi-lcycle carry-forward state.
                continue;
            }
            let gate_type = pp.gate_type;
            if let Some(clifford_path) = self.clifford_paths.get_mut(&pp_id) {
                clifford_path.0 -= 1;
                if clifford_path.0 == 0 {
                    self.clifford_paths.swap_remove(&pp_id);
                }
            } else {
                // First lcycle for this Clifford: insert with remaining count.
                let count = if gate_type.is_cx() { 1 } else { 2 };
                // Without a tree the patches come from the precomputed tree or,
                // for an S/SX placed by `sched_s_sx`, from what it recorded.
                // Falling through to no patches would leave the gate's later
                // lcycles unreserved.
                let node_ids: Vec<u16> = if let Some(tree) = opt_pp_path {
                    tree.iter_nodes().collect()
                } else if let Some(t) = self.precomputed_clifford_trees.get(&pp_id) {
                    t.iter_nodes().collect()
                } else {
                    self.s_sx_nodes.remove(&pp_id).unwrap_or_default()
                };
                debug_assert!(
                    !node_ids.is_empty(),
                    "multi-lcycle product {} holds no patches",
                    pp_id
                );
                let pp_owned = self.input.circuit.product(pp_id).clone();
                self.clifford_paths.insert(
                    pp_id,
                    (count, pp_owned, node_ids, opt_pp_path.as_ref().map(Rc::clone)),
                );
            }
        }
    }

    /// Writes `<stem>.trace`: the layout, then one line per product per lcycle
    /// with the patches it occupied, so that `data_processing/check_schedule.py`
    /// can check the schedule against the circuit without trusting the
    /// scheduler. Line formats:
    ///   node <id> <D|B|M> <label>
    ///   adj <id> <neighbour ids, comma-separated>
    ///   s <lcycle> <product id> <gate> <magic id|-> <attempt length|-> <terminal ids> <footprint ids>
    /// A product id at or past the circuit's product count is an injected S
    /// correction, whose operators follow as `c <id> <qubit>:<basis>,...`. A failed
    /// T gate adds `f <lcycle> <product id> <qubit>:<basis>,...`, the operators it
    /// ran about, so the checker can replay the correction frame.
    /// The attempt length is the cultivation attempt that produced the
    /// consumed state, so the checker can confirm the patch was free for it.
    /// Records the wall-clock time of every logical cycle's scheduling, the span
    /// the "schedule_lcycle" timer accumulates, for `write_cycle_times`.
    pub(crate) fn enable_cycle_times(&mut self) {
        self.cycle_times = Some(Vec::new());
    }

    /// Writes the recorded per-cycle times, one per line in nanoseconds, in
    /// cycle order. The average of these is the "schedule_lcycle" avg.
    pub(crate) fn write_cycle_times(&self, fname: &str) -> io::Result<()> {
        let Some(times) = &self.cycle_times else { return Ok(()) };
        let mut w = BufWriter::new(File::create(fname)?);
        for t in times {
            writeln!(w, "{}", t)?;
        }
        w.flush()
    }

    /// Prints the per-cycle times in the format of the accumulated
    /// "schedule_lcycle" timer, which a build without diagnostics does not
    /// keep, so the analysis scripts read either build's logs the same way.
    #[cfg(not(feature = "diagnostics"))]
    pub(crate) fn print_cycle_time_summary(&self) {
        let Some(times) = &self.cycle_times else { return };
        if times.is_empty() {
            return;
        }
        let fmt = |ns: f64| -> String {
            if ns >= 1e9 {
                format!("{:.2} s", ns / 1e9)
            } else if ns >= 1e6 {
                format!("{:.1} ms", ns / 1e6)
            } else {
                format!("{:.2} μs", ns / 1e3)
            }
        };
        let total: u64 = times.iter().sum();
        let max = *times.iter().max().unwrap();
        println!(
            "  {:<25} total: {:>10}  avg: {:>10}  max: {:>10}  calls: {}",
            "schedule_lcycle",
            fmt(total as f64),
            fmt(total as f64 / times.len() as f64),
            fmt(max as f64),
            times.len()
        );
    }

    pub(crate) fn enable_trace(&mut self, fname: &str) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(fname)?);
        let topo = &self.input.topo;
        writeln!(w, "circuit {}", self.input.circuit.circuit_fname)?;
        writeln!(w, "products {}", self.n_circuit_products)?;
        writeln!(w, "max-weight {}", self.max_weight)?;
        for id in 0..topo.n_nodes as u16 {
            let t = match topo.node(id).node_type {
                NodeType::Data => 'D',
                NodeType::Bus => 'B',
                NodeType::Magic => 'M',
            };
            writeln!(w, "node {} {} {}", id, t, topo.label(id))?;
        }
        for id in 0..topo.n_nodes as u16 {
            let nbs: Vec<String> =
                topo.node(id).nbs_slice().iter().map(|n| n.to_string()).collect();
            writeln!(w, "adj {} {}", id, nbs.join(","))?;
        }
        self.trace = Some(w);
        self.astar.record_path = true;
        Ok(())
    }

    /// The owed corrections a CX must wait for, in emission order: those it does
    /// not commute with. A multi-qubit measurement never gets here with any owed
    /// on its qubits: it is conjugated instead.
    fn clifford_blockers(&self, pp_id: i32) -> Option<Vec<usize>> {
        let pp = self.input.circuit.product(pp_id);
        if self.frame.clear_on(&pp.operators) {
            return None;
        }
        self.frame.pass_clifford(&pp.operators)
    }

    #[inline]
    fn trace_footprint(&mut self, pp_id: i32, nodes: &[u16]) {
        if self.trace.is_some() {
            self.trace_fp.insert(pp_id, nodes.to_vec());
        }
    }

    fn write_trace_lcycle(&mut self) -> io::Result<()> {
        let Some(w) = self.trace.as_mut() else { return Ok(()) };
        let n_circuit = self.n_circuit_products;
        let join = |v: &[u16]| v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",");
        for (pp_id, opt_tree) in &self.pp_paths {
            let pp = self.input.circuit.product(*pp_id);
            let (mut nodes, magic): (Vec<u16>, Option<u16>) = match opt_tree {
                Some(t) => (t.iter_nodes().collect(), t.root_node_id),
                None => match self.trace_untreed.get(pp_id) {
                    Some((n, m)) => (n.clone(), Some(*m)),
                    None => (self.trace_fp.get(pp_id).cloned().unwrap_or_default(), None),
                },
            };
            nodes.sort_unstable();
            let (magic_s, len_s) = match (pp.gate_type.is_t(), magic) {
                (true, Some(m)) => (m.to_string(), self.cultivation.done_len(m).to_string()),
                _ => ("-".to_string(), "-".to_string()),
            };
            let terms = &self.precomputed_terminals[*pp_id as usize];
            writeln!(
                w,
                "s {} {} {:?} {} {} {} {}",
                self.current_lcycle,
                pp_id,
                pp.gate_type,
                magic_s,
                len_s,
                if terms.is_empty() { "-".to_string() } else { join(terms) },
                if nodes.is_empty() { "-".to_string() } else { join(&nodes) },
            )?;
            if *pp_id as usize >= n_circuit {
                let ops: Vec<String> =
                    pp.operators.iter().map(|o| format!("{}:{}", o.qubit, o.basis)).collect();
                writeln!(w, "c {} {}", pp_id, ops.join(","))?;
            }
        }
        self.trace_untreed.clear();
        self.trace_fp.clear();
        Ok(())
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    fn print_sched_stats(&mut self, n_lcycles: usize) {
        #[cfg(feature = "diagnostics")]
        self.stats.summarize(n_lcycles);
        let tot_t = self.count_t_products();
        let fail_pct =
            if tot_t > 0 { 100.0 * self.t_gate_failures as f64 / tot_t as f64 } else { 0.0 };
        println!("Magic state cultivation time:");
        let mean = self.cultivation.cultivation_times_log.iter().sum::<i32>() as f64
            / self.cultivation.cultivation_times_log.len() as f64;
        let min = self.cultivation.cultivation_times_log.iter().min().copied().unwrap_or(0);
        let max = self.cultivation.cultivation_times_log.iter().max().copied().unwrap_or(0);
        println!("  number:  {}", self.cultivation.cultivation_times_log.len());
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);
        println!("T gate failures: {}/{} ({:.1}%)", self.t_gate_failures, tot_t, fail_pct);
        let c = &self.corr;
        println!(
            "Corrections: t-conjugated {} merged-free {} merged-gate {} classical {} \
             emitted-x {} emitted-y {} emitted-z {} m-conjugated {} before-multi {} open-at-end {} \
             emitted-multi {} over-limit {} widened {}",
            c.t_conjugated,
            c.merged_free,
            c.merged_gate,
            c.classical,
            c.emitted[0],
            c.emitted[1],
            c.emitted[2],
            c.m_conjugated,
            c.before_multi,
            self.frame.open(),
            c.emitted_multi,
            c.over_limit,
            c.widened
        );
        if let Some(w) = self.trace.as_mut() {
            w.flush().expect("flushing the schedule trace");
        }
        #[cfg(feature = "diagnostics")]
        self.print_yield();
        println!("Steiner tree computation called {} times", self.stree_computation.n_calls);
        println!("A* computation called {} times", self.astar.n_calls);

        if self.record_cultivation_dist {
            let dist_fname = format!("{}.cultivation_dist", self.input.circuit_stem());
            match self.write_cultivation_dist(&dist_fname, min, max) {
                Ok(()) => println!("Cultivation time distribution written to {}", dist_fname),
                Err(e) => eprintln!("Warning: could not write cultivation dist: {}", e),
            }
        }
    }

    /// Prints the cultivation yield (see `YieldStats`), after charging the
    /// final lcycle's uses, which no later update sees.
    #[cfg(feature = "diagnostics")]
    fn print_yield(&mut self) {
        self.cultivation.settle(&self.input.topo, &self.used, &self.consumed);
        let unfinished = self.cultivation.unfinished_cycles(&self.input.topo, &self.used);
        let y = &self.cultivation.yield_stats;
        println!(
            "Cultivation yield: delivered {} ({} lcycles) interrupted {} ({} lcycles) \
             lost {} ({} lcycles) unfinished {} lcycles, {} cultivators",
            y.n_delivered,
            y.cycles_delivered,
            y.n_interrupted,
            y.cycles_interrupted,
            y.n_lost,
            y.cycles_lost,
            unfinished,
            self.cultivation.magic_node_ids.len()
        );
    }

    /// Writes the normalized distribution of cultivation times to `fname`.
    /// Each line contains: `<cultivation_time> <normalized_count>` where
    /// `normalized_count = count(t) / total_events` for t in [min, max].
    fn write_cultivation_dist(&self, fname: &str, min: i32, max: i32) -> io::Result<()> {
        let times = &self.cultivation.cultivation_times_log;
        let total = times.len();
        if total == 0 || min > max {
            return Ok(());
        }
        let f = File::create(fname)?;
        let mut buf_f = BufWriter::new(f);
        for t in min..=max {
            let count = times.iter().filter(|&&v| v == t).count();
            let normalized = count as f64 / total as f64;
            writeln!(buf_f, "{} {:.6}", t, normalized)?;
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_lcycle(&self) -> io::Result<()> {
        let mut lcycle_used = vec![false; self.input.topo.n_nodes];
        for &(pp_id, ref opt_tree) in &self.pp_paths {
            let Some(tree) = opt_tree else { continue };
            let tree = tree.as_ref();
            let pp = self.input.circuit.product(pp_id);
            if self.scheduled_products.contains(&pp_id) && !pp.gate_type.is_clifford() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("product {} scheduled twice", pp_id),
                ));
            }
            for &parent_id in &pp.parents {
                if !self.scheduled_products.contains(&parent_id) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("product {} scheduled before parent {}", pp_id, parent_id),
                    ));
                }
            }
            for &nid in &self.precomputed_terminals[pp_id as usize] {
                if !tree.contains_node(nid) {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} terminal node {} missing from tree",
                            pp_id,
                            self.input.topo.label(nid)
                        ),
                    ));
                }
            }
            if pp.gate_type.is_t() {
                match tree.root_node_id {
                    None => {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            format!("product {}: T gate has no magic root node", pp_id),
                        ));
                    }
                    Some(magic_id) => {
                        if self.input.topo.node(magic_id).node_type != NodeType::Magic {
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                format!(
                                    "product {}: root node {} is not a Magic node",
                                    pp_id, magic_id
                                ),
                            ));
                        }
                    }
                }
            }
            for node_id in tree.iter_nodes() {
                if lcycle_used[node_id as usize] {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!(
                            "product {} shares node '{}' with another \
                                                       product in the same lcycle",
                            pp_id,
                            self.input.topo.label(node_id)
                        ),
                    ));
                }
                lcycle_used[node_id as usize] = true;
            }
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_clifford_repetitions(&self) -> io::Result<()> {
        let mut cx_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        let mut s_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            for &pp_id in lcycle_ids {
                let pp = self.input.circuit.product(pp_id);
                if pp.gate_type.is_cx() {
                    let lcycles = cx_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                    let lcycles = s_counts.entry(pp_id).or_insert(Vec::new());
                    lcycles.push(*lcycle_i);
                }
            }
        }
        let mut errors = Vec::new();
        for (pp_id, lcycles) in &cx_counts {
            let pp = self.input.circuit.product(*pp_id);
            if pp.gate_type.is_cx() {
                if lcycles.len() != 2 || lcycles[0] != lcycles[1] - 1 {
                    errors.push(format!("  product {} not scheduled 2x {:?}", pp, lcycles));
                }
            }
        }
        for (pp_id, lcycles) in &s_counts {
            let pp = self.input.circuit.product(*pp_id);
            if self.free_merged_products.contains(pp_id) {
                if lcycles.len() != 1 {
                    errors.push(format!(
                        "  merge-completed product {} expected exactly 1 lcycle, got {:?}",
                        pp, lcycles
                    ));
                }
            } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                if lcycles.len() != 3
                    || lcycles[0] != lcycles[1] - 1
                    || lcycles[1] != lcycles[2] - 1
                {
                    errors.push(format!("  product {} not scheduled 3x {:?}", pp, lcycles));
                }
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Clifford repetition errors:\n{}", errors.join("\n")),
            ));
        }
        println!(
            "Clifford repetition check passed ({} CX, {} S/SX products)",
            cx_counts.len(),
            s_counts.len()
        );
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn check_schedule(&self) -> io::Result<()> {
        let n_products = self.input.circuit.n_products();
        let mut errors: Vec<String> = Vec::new();
        for pp_id in 0..n_products as i32 {
            if !self.scheduled_products.contains(&pp_id) {
                errors.push(format!("  product {} was never scheduled", pp_id));
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Completeness errors:\n{}", errors.join("\n")),
            ));
        }
        println!("Schedule check passed: all {} products scheduled", n_products);
        Ok(())
    }

    pub(crate) fn print_schedule(&self, hdr: &str) -> io::Result<()> {
        let _timer = fn_timer!();
        debug_sched!("Printing schedule");
        let output_fname = format!("{}.schedule", self.input.circuit_stem());

        let f = File::create(&output_fname)?;
        let mut buf_f = BufWriter::new(f);

        let max_lcycle: usize =
            self.lcycle_scheduled.last().map(|(lcycle_i, _)| *lcycle_i).unwrap_or(0);
        let max_width = max_lcycle.to_string().len();
        let tot_products = self.lcycle_scheduled.iter().map(|(_, v)| v.len()).sum::<usize>();
        writeln!(buf_f, "{}", hdr)?;
        writeln!(buf_f, "# Total active logical cycles: {}", self.lcycle_scheduled.len())?;
        writeln!(buf_f, "# Total logical cycles: {}", max_lcycle)?;
        writeln!(buf_f, "# Total products: {}", tot_products)?;
        writeln!(buf_f, "# Parallelism: {:.2}", tot_products as f64 / max_lcycle as f64)?;

        let colors = [
            Color::Green,
            Color::Red,
            Color::Yellow,
            Color::Blue,
            Color::Magenta,
            Color::Cyan,
            Color::White,
            Color::BrightGreen,
            Color::BrightRed,
            Color::BrightYellow,
            Color::BrightBlue,
            Color::BrightMagenta,
            Color::BrightCyan,
            Color::BrightWhite,
        ];

        // prev_cx tracks CX products seen in the previous lcycle so we can
        // suppress the second operator on the correct lcycle in the output.
        let mut prev_cx: IndexSet<i32> = IndexSet::new();
        for (lcycle_i, lcycle_ids) in &self.lcycle_scheduled {
            let mut sorted_ids = lcycle_ids.clone();
            sorted_ids.sort_by_key(|&id| {
                self.input
                    .circuit
                    .product(id)
                    .operators
                    .iter()
                    .map(|op| op.qubit)
                    .min()
                    .unwrap_or(u16::MAX)
            });
            let mut combined_chars = vec!['_'; self.input.circuit.n_qubits];
            let mut combined_colors: Vec<Option<Color>> = vec![None; self.input.circuit.n_qubits];
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                for op in &pp.operators {
                    if op.qubit < self.input.circuit.n_qubits as u16 {
                        combined_chars[op.qubit as usize] = op.basis;
                        combined_colors[op.qubit as usize] = Some(color);
                    }
                }
                if pp.gate_type.is_cx() {
                    // CX spans 2 lcycles. On the first lcycle show only operator[0]
                    // (the control); on the second show only operator[1] (the target).
                    if !prev_cx.swap_remove(&pp_id) {
                        debug_sched!("  first lcycle of CX {} {}", pp_id, pp);
                        prev_cx.insert(pp_id);
                        let qubit = pp.operators[1].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    } else {
                        debug_sched!("  second lcycle of CX {} {}", pp_id, pp);
                        let qubit = pp.operators[0].qubit;
                        combined_colors[qubit as usize] = None;
                        combined_chars[qubit as usize] = '_';
                    }
                }
            }
            write!(buf_f, "{:width$}: ", lcycle_i, width = max_width)?;
            for i in 0..self.input.circuit.n_qubits {
                let ch = combined_chars[i].to_string();
                let colored_ch = match combined_colors[i] {
                    Some(c) => ch.color(c).to_string(),
                    None => ch,
                };
                write!(buf_f, "{}", colored_ch)?;
            }
            let mut id_string = String::new();
            for (idx, &pp_id) in sorted_ids.iter().enumerate() {
                let pp = self.input.circuit.product(pp_id);
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(
                    " {}",
                    format!("{}<{:?}>", pp_id, pp.gate_type).color(color)
                ));
            }
            writeln!(buf_f, "{}", id_string)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }
}

fn circuit_stem(fname: &str) -> &str {
    Path::new(fname).file_stem().and_then(|s| s.to_str()).unwrap_or("circuit")
}

/// Expands operators into data node IDs; Y-basis operators expand to two nodes
/// (X patch + Z patch) since a Y measurement needs both stabiliser data qubits.
fn operators_to_node_ids(topo: &TopoGraph, operators: &[Operator]) -> Vec<u16> {
    let mut node_ids = Vec::with_capacity(operators.len());
    for op in operators {
        if op.basis == 'Y' {
            node_ids.push(topo.get_data_node_id(op.qubit, 'X'));
            node_ids.push(topo.get_data_node_id(op.qubit, 'Z'));
        } else {
            node_ids.push(topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase()));
        }
    }
    node_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::node::Node;
    use crate::topograph::TopoGraph;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn run_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let mut sched = Scheduler::new(
            circuit,
            topo,
            0.0387396,
            0.0,
            "none",
            String::new(),
            rseed,
            false,
            false,
        );
        sched.sched_circuit().expect("sched_circuit failed");
        sched
    }

    #[test]
    fn t_gate_failures_bounded_by_total_t_gates() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 0);
        let tot_t = 4usize;
        assert!(
            sched.t_gate_failures <= tot_t,
            "t_gate_failures {} exceeds total T gates {}",
            sched.t_gate_failures,
            tot_t
        );
    }

    #[test]
    fn t_gate_failures_deterministic_with_fixed_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched1 = run_scheduler(lines, 42);
        let sched2 = run_scheduler(lines, 42);
        assert_eq!(
            sched1.t_gate_failures, sched2.t_gate_failures,
            "t_gate_failures differs between runs with the same seed"
        );
    }

    #[test]
    fn t_gate_failures_varies_with_seed() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let counts: Vec<usize> =
            (0u32..20).map(|s| run_scheduler(lines, s).t_gate_failures).collect();
        let distinct = counts.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 1, "t_gate_failures never varied across 20 seeds: {:?}", counts);
    }

    #[test]
    fn all_products_appear_exactly_once_in_lcycle_scheduled() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let mut id_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for (_, ids) in &sched.lcycle_scheduled {
            for &id in ids {
                *id_counts.entry(id).or_insert(0) += 1;
            }
        }
        let n_products = 4;
        for pp_id in 0..n_products as i32 {
            let count = id_counts.get(&pp_id).copied().unwrap_or(0);
            assert_eq!(
                count, 1,
                "product {} appears {} times in lcycle_scheduled (expected 1)",
                pp_id, count
            );
        }
    }

    #[test]
    fn lcycle_scheduled_tot_entries_equals_n_products() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        let n_products = 4usize;
        assert_eq!(
            tot_entries, n_products,
            "total lcycle_scheduled entries {} != n_products {}",
            tot_entries, n_products
        );
    }

    #[test]
    fn lcycle_count_not_inflated_by_t_gate_failures() {
        // T gate failures are tracked in the correction frame and do not add
        // extra scheduling lcycles.
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let n_t = 4usize;
        let active_lcycles = sched.lcycle_scheduled.len();
        assert!(
            active_lcycles <= n_t,
            "active lcycles {} > n_t {} (failures should not add extra lcycles)",
            active_lcycles,
            n_t,
        );
    }

    fn build_scheduler(lines: &[&str], rseed: u32) -> Scheduler {
        Node::set_magic_routing(true);
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        Scheduler::new(circuit, topo, 0.0387396, 0.0, "none", String::new(), rseed, false, false)
    }

    #[test]
    fn compact_bus_routes_a_z_pair_on_one_patch() {
        // Z on both qubits of the patch d2/3. The node above the pair's Z faces is a
        // border bus cell between two cultivators with no other routing link. Rooted
        // there, the product could never be routed, and the scheduler panicked with
        // "Cannot schedule products". It must fall back to the side bus columns.
        Node::set_magic_routing(false);
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "+__ZZ<T>").unwrap();
        let fname = f.path().to_string_lossy().to_string();
        let mut circuit = Circuit::new(&fname);
        circuit.load_circuit().expect("circuit load failed");
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, false, 0, false);
        let mut sched =
            Scheduler::new(circuit, topo, 0.0387396, 0.0, "none", String::new(), 1, false, false);
        sched.precompute_terminals_and_roots();
        for (_, preferred, _) in &sched.precomputed_root_info[0] {
            for &root in preferred {
                assert!(Scheduler::joins_routing(&sched.input.topo, root));
            }
        }
        let result = sched.sched_circuit();
        Node::set_magic_routing(true);
        result.expect("sched_circuit failed");
        assert!(sched.scheduled_products.contains(&0));
    }

    fn op(qubit: u16, basis: char) -> Operator {
        Operator { qubit, basis }
    }

    #[test]
    fn t_conjugation_uses_the_pending_correction_own_axis_not_always_z() {
        // S about X (left by a failed T gate on X) turns a later Z rotation into Y.
        let mut sched = build_scheduler(&["+Z___<T>"], 0);
        sched.precompute_terminals_and_roots();
        sched.frame.add_failure(&[op(0, 'X')]);
        assert_eq!(sched.conjugate_rotation(0), Rotation::Run);
        assert_eq!(sched.t_conjugated_bases[&0], vec![op(0, 'Y')]);
    }

    #[test]
    fn a_commuting_multi_qubit_correction_leaves_a_product_alone() {
        // S about ZZ commutes with XX. The per-qubit model turned XX into YY.
        let mut sched = build_scheduler(&["+XX__<T>"], 0);
        sched.frame.add_failure(&[op(0, 'Z'), op(1, 'Z')]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.corr.t_conjugated, 0);
        assert!(sched.frame.open() >= 1, "ZZ still owed");
    }

    #[test]
    fn a_multi_qubit_correction_inside_the_support_is_absorbed() {
        // S about ZZ anticommutes with ZX, and ZZ lies within ZX's qubits: ZX runs
        // as (ZZ)(ZX) = IY, on qubit 1 alone.
        let mut sched = build_scheduler(&["+ZX__<T>"], 0);
        sched.frame.add_failure(&[op(0, 'Z'), op(1, 'Z')]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.corr.t_conjugated, 1);
        let y = vec![
            sched.input.topo.get_data_node_id(1, 'X'),
            sched.input.topo.get_data_node_id(1, 'Z'),
        ];
        let mut t = sched.precomputed_terminals[0].clone();
        t.sort_unstable();
        let mut want = y;
        want.sort_unstable();
        assert_eq!(t, want);
    }

    #[test]
    fn a_correction_that_would_pass_the_weight_limit_is_emitted_first() {
        // S about ZZ anticommutes with X on qubit 0, and absorbing it would widen
        // X0 onto qubit 1, past a weight limit of 1. It is emitted first, as a
        // two-qubit S held three lcycles.
        // The second product puts qubit 1 in the circuit. It commutes with ZZ.
        let mut sched = build_scheduler(&["+X___<T>", "+_Z__<M>"], 0);
        sched.max_weight = 1;
        sched.frame.add_failure(&[op(0, 'Z'), op(1, 'Z')]);
        let trace = NamedTempFile::new().unwrap();
        sched.enable_trace(&trace.path().to_string_lossy()).unwrap();
        sched.sched_circuit().expect("sched_circuit failed");
        sched.trace.as_mut().unwrap().flush().unwrap();
        assert_eq!(sched.correction_gates_emitted, 1);
        assert_eq!(sched.corr.emitted_multi, 1);
        assert_eq!(sched.corr.over_limit, 1);
        let text = std::fs::read_to_string(trace.path()).unwrap();
        let c = sched.n_circuit_products as i32;
        let fps = traced_footprints(&text, c);
        assert_eq!(fps.len(), 3, "the correction takes three lcycles");
        assert!(fps.iter().all(|fp| fp == &fps[0]), "same patches: {:?}", fps);
        let t_cycle = sched.lcycle_scheduled.iter().find(|(_, ids)| ids.contains(&0)).unwrap().0;
        let c_last = sched.lcycle_scheduled.iter().filter(|(_, ids)| ids.contains(&c)).last().unwrap().0;
        assert!(t_cycle > c_last, "T at {} after correction ending {}", t_cycle, c_last);
        assert!(text.contains(&format!("c {} 0:Z,1:Z", c)), "trace names both qubits");
    }

    #[test]
    fn a_product_widens_onto_a_qubit_nothing_else_uses_that_lcycle() {
        // With no weight limit, X0 absorbs the owed ZZ and runs as YZ over qubits
        // 0 and 1. The measurement on qubit 1 commutes with ZZ and is ready at
        // once, but may not share qubit 1 with the widened T gate.
        let mut sched = build_scheduler(&["+X___<T>", "+_Z__<M>"], 0);
        sched.frame.add_failure(&[op(0, 'Z'), op(1, 'Z')]);
        let trace = NamedTempFile::new().unwrap();
        sched.enable_trace(&trace.path().to_string_lossy()).unwrap();
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.corr.widened, 1);
        let cycle_of = |id: i32| {
            sched.lcycle_scheduled.iter().find(|(_, ids)| ids.contains(&id)).unwrap().0
        };
        assert_ne!(cycle_of(0), cycle_of(1), "T and M share qubit 1");
        let mut want = vec![
            sched.input.topo.get_data_node_id(0, 'X'),
            sched.input.topo.get_data_node_id(0, 'Z'),
            sched.input.topo.get_data_node_id(1, 'Z'),
        ];
        want.sort_unstable();
        let mut t = sched.precomputed_terminals[0].clone();
        t.sort_unstable();
        assert_eq!(t, want, "the T gate runs about YZ");
    }

    #[test]
    fn mixed_axis_corrections_are_not_dropped_as_a_pauli() {
        // S_X then S_Y on one qubit is not a Pauli. The per-qubit model counted it
        // as power 2 and dropped it. The frame keeps it as S_Z then S_X: the S
        // waits for S_X to be emitted, then absorbs S_Z, its own axis.
        let mut sched = build_scheduler(&["+Z___<S>"], 0);
        sched.frame.add_failure(&[op(0, 'X')]);
        sched.frame.add_failure(&[op(0, 'Y')]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1);
        assert_eq!(sched.corr.merged_free, 1);
        assert_eq!(sched.frame.open(), 0);
        let c = sched.n_circuit_products as i32;
        assert_eq!(sched.input.circuit.product(c).operators, vec![op(0, 'X')]);
    }

    #[test]
    fn s_correction_merges_free_when_axis_matches_and_net_power_is_even() {
        // Native S products are always Z-basis, so a pending Z-axis correction shares S's
        // own generator. Power 1 (owes one S) plus the blocked S's own power gives net 2
        // (Z, classical) - fully absorbed with zero physical gates.
        let lines = &["+Z___<S>"];
        let mut sched = build_scheduler(lines, 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'Z' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.frame.open(), 0);
        assert!(sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 1, "fully-absorbed correction costs zero extra lcycles");
    }

    #[test]
    fn s_correction_falls_back_to_materializing_when_axis_does_not_match() {
        // A pending correction about X (e.g. left over from a failed T gate that targeted
        // X) is a different generator from S's fixed Z axis and can't be folded in -
        // it must still be materialized as its own standalone gate first.
        let lines = &["+Z___<S>"];
        let mut sched = build_scheduler(lines, 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'X' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1, "standalone correction gate injected");
        assert_eq!(sched.frame.open(), 0);
        assert!(sched.free_merged_products.is_empty());
        assert!(sched.scheduled_products.contains(&0), "original S product scheduled");
        assert!(sched.scheduled_products.contains(&1), "injected correction (id=1) scheduled");
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 6, "3 lcycles for the injected correction + 3 for the S gate");
    }

    /// Footprint per lcycle of product `pp_id` in a schedule trace.
    fn traced_footprints(trace: &str, pp_id: i32) -> Vec<Vec<u16>> {
        trace
            .lines()
            .filter_map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f.first() == Some(&"s") && f[2] == pp_id.to_string()).then(|| {
                    if f[7] == "-" {
                        Vec::new()
                    } else {
                        f[7].split(',').map(|x| x.parse().unwrap()).collect()
                    }
                })
            })
            .collect()
    }

    #[test]
    fn noncommuting_correction_is_applied_before_a_cx() {
        // A correction about Y on the CX control, whose operator there is Z, does not
        // commute with the CX. It must be emitted first and complete before the CX.
        let mut sched = build_scheduler(&["+ZX__<CX>"], 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'Y' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1);
        assert_eq!(sched.corr.before_multi, 1);
        let cycles = |id: i32| -> Vec<usize> {
            sched
                .lcycle_scheduled
                .iter()
                .filter(|(_, ids)| ids.contains(&id))
                .map(|(l, _)| *l)
                .collect()
        };
        let corr = cycles(1);
        let cx = cycles(0);
        assert_eq!(corr.len(), 3, "correction takes three lcycles");
        assert!(cx[0] > *corr.last().unwrap(), "CX {:?} after correction {:?}", cx, corr);
    }

    #[test]
    fn commuting_correction_passes_through_a_cx() {
        // A correction about Z on the control commutes with the CX's Z there.
        let mut sched = build_scheduler(&["+ZX__<CX>"], 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'Z' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
    }

    #[test]
    fn measurement_basis_is_conjugated_by_a_pending_correction() {
        // S about X turns a Z measurement into a Y measurement, on both edges, with no
        // correction gate.
        let mut sched = build_scheduler(&["+Z___<M>"], 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'X' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.corr.m_conjugated, 1);
        let x = sched.input.topo.get_data_node_id(0, 'X');
        let z = sched.input.topo.get_data_node_id(0, 'Z');
        let mut t = sched.precomputed_terminals[0].clone();
        t.sort_unstable();
        let mut want = vec![x, z];
        want.sort_unstable();
        assert_eq!(t, want);
    }

    #[test]
    fn s_gate_holds_its_patches_for_all_three_lcycles() {
        // An S placed without a tree used to reserve its data and ancilla patches
        // only in its first lcycle, leaving the next two free for other paths.
        let mut sched = build_scheduler(&["+Z___<S>"], 0);
        let trace = NamedTempFile::new().unwrap();
        sched.enable_trace(&trace.path().to_string_lossy()).unwrap();
        sched.sched_circuit().expect("sched_circuit failed");
        sched.trace.as_mut().unwrap().flush().unwrap();
        let fps = traced_footprints(&std::fs::read_to_string(trace.path()).unwrap(), 0);
        assert_eq!(fps.len(), 3, "S takes three lcycles");
        assert_eq!(fps[0].len(), 2, "data edge and one ancilla");
        assert!(fps.iter().all(|fp| fp == &fps[0]), "same patches every lcycle: {:?}", fps);
    }

    #[test]
    fn y_axis_correction_uses_both_edges_of_the_data_patch() {
        // A correction about Y (left by a failed T gate on Y) is an S about Y,
        // which needs both the X and the Z edge, like any Y operator. It used to
        // be placed on the X edge alone.
        let mut sched = build_scheduler(&["+Z___<S>"], 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'Y' }]);
        let trace = NamedTempFile::new().unwrap();
        sched.enable_trace(&trace.path().to_string_lossy()).unwrap();
        sched.sched_circuit().expect("sched_circuit failed");
        sched.trace.as_mut().unwrap().flush().unwrap();
        assert_eq!(sched.correction_gates_emitted, 1);
        let x = sched.input.topo.get_data_node_id(0, 'X');
        let z = sched.input.topo.get_data_node_id(0, 'Z');
        let fps = traced_footprints(&std::fs::read_to_string(trace.path()).unwrap(), 1);
        assert_eq!(fps.len(), 3, "the correction takes three lcycles");
        for fp in &fps {
            assert!(fp.contains(&x) && fp.contains(&z), "both edges held: {:?}", fp);
        }
    }

    #[test]
    fn sx_correction_merges_free_when_axis_matches_and_net_power_is_even() {
        // Native SX products are always X-basis, so a pending X-axis correction shares
        // SX's own generator, exactly like S/Z. Power 1 + SX's own power gives net 2
        // (classical) - fully absorbed with zero physical gates.
        let lines = &["+X___<SX>"];
        let mut sched = build_scheduler(lines, 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'X' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 0);
        assert_eq!(sched.frame.open(), 0);
        assert!(sched.free_merged_products.contains(&0));
        assert!(sched.scheduled_products.contains(&0));
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 1, "fully-absorbed correction costs zero extra lcycles");
    }

    #[test]
    fn sx_correction_falls_back_to_materializing_when_axis_does_not_match() {
        // A pending correction about Z is a different generator from SX's fixed X axis
        // and can't be folded in - it must still be materialized as its own gate first.
        let lines = &["+X___<SX>"];
        let mut sched = build_scheduler(lines, 0);
        sched.frame.add_failure(&[Operator { qubit: 0, basis: 'Z' }]);
        sched.sched_circuit().expect("sched_circuit failed");
        assert_eq!(sched.correction_gates_emitted, 1, "standalone correction gate injected");
        assert_eq!(sched.frame.open(), 0);
        assert!(sched.free_merged_products.is_empty());
        assert!(sched.scheduled_products.contains(&0), "original SX product scheduled");
        assert!(sched.scheduled_products.contains(&1), "injected correction (id=1) scheduled");
        let tot_entries: usize = sched.lcycle_scheduled.iter().map(|(_, ids)| ids.len()).sum();
        assert_eq!(tot_entries, 6, "3 lcycles for the injected correction + 3 for the SX gate");
    }
}
