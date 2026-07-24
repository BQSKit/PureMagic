use crate::accum_start;
use crate::astar::{AStar, PathResult};
use crate::circuit::Circuit;
use crate::cultivation::CultivationManager;
use crate::debug_sched;
use crate::fn_timer;
use crate::info_sched;
use crate::node::NodeType;
use crate::pauliproduct::{Operator, PauliProduct};
use crate::steinertree::SteinerTree;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::AccumTimers;
use colored::{Color, Colorize};

use indexmap::{IndexMap, IndexSet};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::rc::Rc;

/// Per-lcycle and cumulative qubit utilisation counters.
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
            plot_info: String::new(),
        }
    }

    pub(crate) fn summarize(&self, n_lcycles: usize) {
        let data_frac = self.tot_data_used as f64 / (self.data_qubits * n_lcycles) as f64;
        let bus_frac = self.tot_bus_used as f64 / (self.bus_qubits * n_lcycles) as f64;
        let magic_frac = self.tot_magic_used as f64 / (self.magic_qubits * n_lcycles) as f64;
        let magic_unused_frac =
            self.tot_magic_unused as f64 / (self.magic_qubits * n_lcycles) as f64;
        let total_available = (self.data_qubits + self.bus_qubits + self.magic_qubits) * n_lcycles;
        let total_used = self.tot_data_used + self.tot_bus_used + self.tot_magic_used;
        let total_idle = total_available.saturating_sub(total_used);
        let idle_frac = if total_available > 0 { total_idle as f64 / total_available as f64 } else { 0.0 };
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("Magic unused {:.3}", magic_unused_frac);
        println!("Idle qubit-cycles: {} ({:.4})", total_idle, idle_frac);
    }

    pub(crate) fn update(
        &mut self, lcycle_i: usize, pp_paths_len: usize, tot_available: usize, magic_ready: usize,
        magic_unused: usize, plotting: bool,
    ) {
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
        // magic_denom = ready nodes that were available for T gates (excludes those
        // consumed purely as routing intermediaries in magic-routing mode).
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

    pub(crate) fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_used += 1,
            NodeType::Magic => self.magic_used += 1,
            NodeType::Data => self.data_used += 1,
        }
    }

    pub(crate) fn inc_with_cultivation(&mut self, node_type: NodeType, cultivation_time: i32) {
        self.inc(node_type);
        if node_type == NodeType::Magic && cultivation_time == 0 {
            self.magic_ready_used += 1;
        }
    }

    pub(crate) fn inc_t(&mut self) {
        self.t_scheduled += 1;
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
    plot_option: String,
    pub(crate) cultivation: CultivationManager,
    pub(crate) stats: ScheduleStats,
    /// (lcycle_index, product_ids) for every lcycle that scheduled at least one product.
    pub(crate) lcycle_scheduled: Vec<(usize, Vec<i32>)>,
    pub(crate) scheduled_products: IndexSet<i32>,
    /// Per-node occupancy bitmap for the current lcycle; reset to all-false each lcycle.
    used: Vec<bool>,
    /// In-progress multi-lcycle Clifford gates: maps product_id →
    /// (remaining_lcycles, product, node_ids, opt_tree).
    /// CX occupies 2 lcycles; S/SX occupy 3.
    clifford_paths: IndexMap<i32, (usize, PauliProduct, Vec<u16>, Option<Rc<TreeGraph>>)>,
    /// T gates that failed the 50% coin flip; held for one recovery lcycle.
    failed_t_paths: IndexMap<i32, (PauliProduct, Vec<u16>, Option<Rc<TreeGraph>>)>,
    pub(crate) t_gate_failures: usize,
    pub(crate) stree_computation: SteinerTree,
    pub(crate) astar: AStar,
    no_t_failures: bool,
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
    /// T gate IDs that completed a recovery lcycle this cycle (populated by complete_lcycle).
    recovery_t_ids: Vec<i32>,
}

impl Scheduler {
    pub(crate) fn new(
        circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
        plot_option: String, rseed: u32, no_t_failures: bool,
    ) -> Self {
        if log_level != "none" {
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
            plot_option,
            cultivation: CultivationManager::new(rseed),
            stats: ScheduleStats::new(n_data_qubits, n_bus_qubits, n_magic_qubits),
            lcycle_scheduled: Vec::new(),
            scheduled_products: IndexSet::new(),
            used: vec![false; n_nodes],
            clifford_paths: IndexMap::new(),
            failed_t_paths: IndexMap::new(),
            t_gate_failures: 0,
            stree_computation: SteinerTree::new(n_nodes),
            astar: AStar::new(n_nodes),
            no_t_failures,
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
            recovery_t_ids: Vec::new(),
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
            .set_lambda(self.magic_state_lambda)
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
        while !self.pps_pending.is_empty()
            || !self.clifford_paths.is_empty()
            || !self.failed_t_paths.is_empty()
        {
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
                    && (tot_pps_to_sched - n_scheduled >= plot_lcycles)
                {
                    if self.current_lcycle == plot_lcycles {
                        print!("Scheduling {} products:    ", tot_pps_to_sched);
                    }
                    let pct_complete = (n_scheduled * 100) / tot_pps_to_sched;
                    if pct_complete > prev_pct_complete {
                        // \x08 is backspace; overwrite the previous "XX%" in-place.
                        print!("\x08\x08\x08{:02}%", pct_complete);
                        std::io::stdout().flush()?;
                        prev_pct_complete = pct_complete;
                    }
                    if tot_pps_to_sched - n_scheduled == plot_lcycles {
                        print!("\n");
                    }
                } else {
                    let plot_info_str = self.stats.plot_info_str();
                    assert!(!plot_info_str.is_empty());
                    let fname_added = format!(".{}", self.current_lcycle);
                    // fname_added is used as a suffix for the plot file name
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    let pp_paths_plot: Vec<(PauliProduct, Rc<TreeGraph>, u32)> = self
                        .pp_paths
                        .iter()
                        .filter_map(|(pp_id, opt_tree)| {
                            opt_tree.as_ref().map(|t| {
                                let pp = self.input.circuit.product(*pp_id);
                                // Determine which cycle of a multi-lcycle product this is.
                                // advance_clifford_state has already run, so clifford_paths
                                // reflects the post-advance state for this lcycle.
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
                                    if self.recovery_t_ids.contains(pp_id) { 2 } else { 1 }
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
                // If nothing is cultivating, no magic state will ever become ready,
                // so the layout is fundamentally unable to make progress.
                if !(0..self.input.topo.n_nodes)
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
        // Return total lcycles and total "attempts" (successes + failures), so the
        // caller can compute the true T-gate count including failed first attempts.
        Ok((self.current_lcycle, self.scheduled_products.len() + self.t_gate_failures))
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

    /// Fills `terminals_buf` from precomputed IDs.
    /// Returns false early if any terminal data node is already occupied, or if
    /// all root candidates for a terminal are occupied (fast pre-check before
    /// the more expensive `root_nodes` call).
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
            if preferred.iter().all(|&rid| self.used[rid as usize])
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
    ///
    /// For Y-basis operators the X and Z data nodes are paired; a single routing
    /// node between them (in the "preferred" direction) can serve both, so
    /// `unmatched_count` is decremented by 2 when a paired root is found.
    /// If no paired root is free, each terminal falls back to a side nb.
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
                // is_paired: this terminal's paired data node is also a terminal
                // (i.e. the operator is Y-basis, producing both X and Z data nodes).
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
                            preferred.push(nb_id);
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
    /// Called for both Clifford carry-forwards and failed-T recovery paths.
    fn carry_forward_path(
        &mut self, pp_id: i32, node_ids: &[u16], opt_tree: Option<Rc<TreeGraph>>,
    ) {
        self.mark_nodes_used(node_ids);
        self.pp_paths.push((pp_id, opt_tree));
    }

    /// Schedules as many products as possible in one lcycle; returns false if nothing scheduled.
    fn schedule_lcycle(&mut self, plotting: bool) -> bool {
        let _timer = accum_start!(self.timers);
        self.timers.start(self.other_timer);
        let mut n_avail_magic = self.update_cultivators();
        let initial_magic = n_avail_magic;
        self.pp_paths.clear();
        self.used.fill(false);
        // Collect carry-forward data before the loop to release borrows on the maps.
        let clifford_carry: Vec<(i32, Vec<u16>, Option<Rc<TreeGraph>>)> = self
            .clifford_paths
            .values()
            .map(|(_, pp, node_ids, opt_tree)| {
                (pp.id, node_ids.clone(), opt_tree.as_ref().map(Rc::clone))
            })
            .collect();
        let failed_t_carry: Vec<(i32, Vec<u16>, Option<Rc<TreeGraph>>)> = self
            .failed_t_paths
            .values()
            .map(|(pp, node_ids, opt_tree)| {
                (pp.id, node_ids.clone(), opt_tree.as_ref().map(Rc::clone))
            })
            .collect();
        for (pp_id, node_ids, opt_tree) in clifford_carry {
            self.carry_forward_path(pp_id, &node_ids, opt_tree);
        }
        for (pp_id, node_ids, opt_tree) in failed_t_carry {
            self.carry_forward_path(pp_id, &node_ids, opt_tree);
        }
        let carry_forward_count = self.clifford_paths.len() + self.failed_t_paths.len();
        let tot_available = carry_forward_count + self.pps_pending.len();
        info_sched!("  Remaining to schedule: {}", self.pps_pending.len());
        self.sched_precomputed(plotting);
        self.timers.stop(self.other_timer);
        self.sched_remaining(&mut n_avail_magic, plotting);
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
            false
        } else {
            true
        }
    }

    pub(crate) fn update_cultivators(&mut self) -> usize {
        let _timer = accum_start!(self.timers);
        let n_avail_magic = self.cultivation.update_cultivators(&mut self.input.topo, &self.used);
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
            let all_free = tree.iter_nodes().all(|nid| !self.used[nid as usize]);
            if all_free {
                to_remove.push(pp_id);
                let node_ids: Vec<u16> = tree.iter_nodes().collect();
                let (_tree_n_nodes, _tree_n_edges) = (tree.n_nodes, tree.n_edges);
                let opt_tree: Option<Rc<TreeGraph>> =
                    if plotting { Some(Rc::clone(&tree)) } else { None };
                // Drop `tree` before the `&mut self` call to `mark_nodes_used`.
                drop(tree);
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

    /// Second pass: greedily schedule T gates, measurements, and S/SX gates.
    /// T gates are skipped when no magic state is available.
    fn sched_remaining(&mut self, n_avail_magic: &mut usize, plotting: bool) {
        let _timer = accum_start!(self.timers);
        for i in 0..self.pps_pending.len() {
            let pp_id = self.pps_pending[i].id;
            let pp = self.input.circuit.product(pp_id);
            if Self::should_precompute(pp) {
                continue;
            }
            let (pp_id, gate_type) = (pp.id, pp.gate_type);
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
        let n_operators = pp.operators.len();
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
            debug_assert!(
                !Self::should_precompute(self.input.circuit.product(pp_id)),
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
        self.scheduled_ids_buf.clear();
        self.scheduled_ids_buf.extend(self.pp_paths.iter().map(|(id, _)| *id));
        self.pps_pending.retain(|pp| !self.scheduled_ids_buf.contains(&pp.id));
        debug_sched!("After purge, pps_to_sched len {}", self.pps_pending.len());
        // Only count T gates that are newly scheduled (not recovery lcycles) to
        // keep the pool-size estimate accurate.
        let t_newly_scheduled = self
            .pp_paths
            .iter()
            .filter(|(id, _)| {
                self.input.circuit.product(*id).gate_type.is_t()
                    && !self.failed_t_paths.contains_key(id)
            })
            .count();
        self.cultivation.t_products_remaining =
            self.cultivation.t_products_remaining.saturating_sub(t_newly_scheduled);
        let (t_failed_ids, t_recovery_ids) = self.process_t_gate_outcomes();
        self.recovery_t_ids = t_recovery_ids;
        self.unlock_children(&t_failed_ids);
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
        let lcycle_ids: Vec<i32> = self
            .pp_paths
            .iter()
            .filter(|(id, _)| !t_failed_ids.contains(id))
            .map(|(id, _)| *id)
            .collect();
        self.lcycle_scheduled.push((self.current_lcycle, lcycle_ids));
        #[cfg(debug_assertions)]
        self.check_lcycle(&t_failed_ids, &self.recovery_t_ids)?;
        self.scheduled_products.extend(
            self.pp_paths.iter().filter(|(id, _)| !t_failed_ids.contains(id)).map(|(id, _)| *id),
        );
        Ok(())
    }

    /// Coin-flip T gate outcomes; updates `failed_t_paths`; returns (failed_ids, recovery_ids).
    ///
    /// First-attempt T gates succeed with 50% probability (or always if `no_t_failures`).
    /// Recovery-lcycle T gates (already in `failed_t_paths`) always succeed.
    /// Failed gates are stored in `failed_t_paths` with the magic root trimmed off
    /// so the routing subtree can be reused in the recovery lcycle.
    fn process_t_gate_outcomes(&mut self) -> (Vec<i32>, Vec<i32>) {
        let mut t_failed_ids: Vec<i32> = Vec::new();
        let mut t_recovery_ids: Vec<i32> = Vec::new();
        let pp_ids: Vec<i32> = self.pp_paths.iter().map(|(id, _)| *id).collect();
        for pp_id in &pp_ids {
            let pp_id = *pp_id;
            let pp = self.input.circuit.product(pp_id);
            if pp.gate_type.is_t() {
                if self.failed_t_paths.contains_key(&pp_id) {
                    t_recovery_ids.push(pp_id);
                    info_sched!("  T gate {} recovery lcycle succeeded", pp_id);
                } else if self.no_t_failures || self.rng_uniform.gen_bool(0.5) {
                    info_sched!("  T gate {} succeeded on first attempt", pp_id);
                } else {
                    t_failed_ids.push(pp_id);
                    self.t_gate_failures += 1;
                    info_sched!(
                        "  T gate {} failed (50% probability), recovery lcycle next",
                        pp_id
                    );
                }
            }
        }
        let pp_paths_snapshot: Vec<(i32, Option<Rc<TreeGraph>>)> =
            self.pp_paths.iter().map(|(id, opt)| (*id, opt.as_ref().map(Rc::clone))).collect();
        for (pp_id, opt_pp_path) in &pp_paths_snapshot {
            let pp_id = *pp_id;
            let pp = self.input.circuit.product(pp_id);
            if !pp.gate_type.is_t() {
                continue;
            }
            if t_failed_ids.contains(&pp_id) {
                // Store the routing subtree without the magic root so the recovery
                // lcycle can reuse the same data/routing nodes while finding a new
                // magic state.  When not plotting, fall back to just the terminal IDs.
                let trimmed_opt_tree: Option<Rc<TreeGraph>> = opt_pp_path.as_ref().map(|tree| {
                    let mut t = (**tree).clone();
                    t.trim_magic_root();
                    Rc::new(t)
                });
                let node_ids: Vec<u16> = if let Some(ref trimmed) = trimmed_opt_tree {
                    trimmed.iter_nodes().collect()
                } else {
                    self.precomputed_terminals[pp_id as usize].clone()
                };
                self.failed_t_paths.insert(pp_id, (pp.clone(), node_ids, trimmed_opt_tree));
            } else {
                self.failed_t_paths.swap_remove(&pp_id);
            }
        }
        (t_failed_ids, t_recovery_ids)
    }

    /// Decrements `remaining_parents` for each completed product and collects
    /// newly-ready children into `children_buf`.
    fn unlock_children(&mut self, t_failed_ids: &[i32]) {
        self.children_buf.clear();
        for i in 0..self.pp_paths.len() {
            let pp_id = self.pp_paths[i].0;
            let pp = self.input.circuit.product(pp_id);
            let gate_type = pp.gate_type;
            if gate_type.is_clifford() {
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
            if gate_type.is_t() && t_failed_ids.contains(&pp_id) {
                continue;
            }
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
            if !pp.gate_type.is_clifford() {
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
                let node_ids: Vec<u16> = if let Some(tree) = opt_pp_path {
                    tree.iter_nodes().collect()
                } else {
                    self.precomputed_clifford_trees
                        .get(&pp_id)
                        .map(|t| t.iter_nodes().collect())
                        .unwrap_or_default()
                };
                let pp_owned = self.input.circuit.product(pp_id).clone();
                self.clifford_paths.insert(
                    pp_id,
                    (count, pp_owned, node_ids, opt_pp_path.as_ref().map(Rc::clone)),
                );
            }
        }
    }

    fn print_sched_stats(&mut self, n_lcycles: usize) {
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
        let magic_volume = self.current_lcycle * self.stats.magic_qubits;
        let aborted_frac = if magic_volume > 0 {
            self.cultivation.aborted_cultivation_cycles as f64 / magic_volume as f64
        } else {
            0.0
        };
        println!(
            "Aborted cultivation cycles: {} ({:.4})",
            self.cultivation.aborted_cultivation_cycles, aborted_frac
        );
        println!("T gate failures: {}/{} ({:.1}%)", self.t_gate_failures, tot_t, fail_pct);
        println!("Steiner tree computation called {} times", self.stree_computation.n_calls);
        println!("A* computation called {} times", self.astar.n_calls);

        // Write normalized cultivation-time distribution to file.
        let dist_fname = format!("{}.cultivation_dist", self.input.circuit_stem());
        match self.write_cultivation_dist(&dist_fname, min, max) {
            Ok(()) => println!("Cultivation time distribution written to {}", dist_fname),
            Err(e) => eprintln!("Warning: could not write cultivation dist: {}", e),
        }
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
    fn check_lcycle(&self, _t_failed_ids: &[i32], t_recovery_ids: &[i32]) -> io::Result<()> {
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
            for nid in operators_to_node_ids(&self.input.topo, &pp.operators) {
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
            if pp.gate_type.is_t() && !t_recovery_ids.contains(&pp_id) {
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
            if pp.gate_type.is_s() || pp.gate_type.is_sx() {
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

/// Expands operators into data node IDs.
/// Y-basis operators expand to two nodes (X patch + Z patch) because a Y
/// measurement requires both the X and Z stabiliser data qubits.
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
        let mut sched =
            Scheduler::new(circuit, topo, 0.0387396, "none", String::new(), rseed, false);
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
    fn failed_t_paths_empty_after_schedule_completes() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 0);
        assert!(
            sched.failed_t_paths.is_empty(),
            "failed_t_paths not empty after schedule_circuit: {:?}",
            sched.failed_t_paths.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn lcycle_count_bounded_by_t_gate_failure_overhead() {
        let lines = &["+X___<T>", "-_X__<T>", "+__X_<T>", "-___X<T>"];
        let sched = run_scheduler(lines, 5);
        let n_t = 4usize;
        let active_lcycles = sched.lcycle_scheduled.len();
        assert!(
            active_lcycles <= n_t + sched.t_gate_failures,
            "active lcycles {} > n_t {} + failures {}",
            active_lcycles,
            n_t,
            sched.t_gate_failures
        );
    }
}
