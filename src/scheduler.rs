use crate::astar::AStarComputation;
use crate::circuit::Circuit;
use crate::debug_sched;
use crate::fn_timer;
use crate::info_sched;
use crate::node::NodeType;
use crate::pauliproduct::PauliProduct;
use crate::steinertree::SteinerTreeComputation;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::IntermittentTimer;
use crate::utils::{
    _BLUE, _CYAN, _GREEN, _LBLUE, _LCYAN, _LGREEN, _LMAGENTA, _LRED, _LWHITE, _LYELLOW, _MAGENTA,
    _RED, _RESET, _WHITE, _YELLOW,
};

use indexmap::{IndexMap, IndexSet};
use rand_simple::Exponential;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

/// Accumulates statistics about scheduled Pauli products (node types and counts).
struct ScheduleStats {
    data_qubits: usize,
    bus_qubits: usize,
    magic_qubits: usize,
    sum_data_scheduled: usize,
    sum_bus_scheduled: usize,
    sum_magic_scheduled: usize,
    bus_scheduled: usize,
    data_scheduled: usize,
    magic_scheduled: usize,
    sum_magic_unused: usize,
    plot_info_str: String,
}

impl ScheduleStats {
    /// Creates a new statistics tracker with qubit counts.
    pub fn new(data_qubits: usize, bus_qubits: usize, magic_qubits: usize) -> Self {
        ScheduleStats { data_qubits,
                        bus_qubits,
                        magic_qubits,
                        sum_data_scheduled: 0,
                        sum_bus_scheduled: 0,
                        sum_magic_scheduled: 0,
                        bus_scheduled: 0,
                        data_scheduled: 0,
                        magic_scheduled: 0,
                        sum_magic_unused: 0,
                        plot_info_str: String::new() }
    }

    pub fn summarize(&self, num_steps: usize) {
        // Calculate statistics
        let data_frac = self.sum_data_scheduled as f64 / (self.data_qubits * num_steps) as f64;
        let bus_frac = self.sum_bus_scheduled as f64 / (self.bus_qubits * num_steps) as f64;
        let magic_frac = self.sum_magic_scheduled as f64 / (self.magic_qubits * num_steps) as f64;
        let magic_unused_frac =
            self.sum_magic_unused as f64 / (self.magic_qubits * num_steps) as f64;
        // Print final statistics
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("Magic unused {:.3}", magic_unused_frac);
    }

    pub fn update(&mut self, step_i: usize, pp_paths_len: usize, to_schedule_len: usize,
                  magic_unused: usize, plotting: bool) {
        self.sum_data_scheduled += self.data_scheduled;
        self.sum_bus_scheduled += self.bus_scheduled;
        self.sum_magic_scheduled += self.magic_scheduled;
        self.sum_magic_unused += magic_unused;

        info_sched!("Scheduling results:");
        let frac_paths = pp_paths_len as f64 / to_schedule_len as f64;
        let frac_data = self.data_scheduled as f64 / self.data_qubits as f64;
        let frac_bus = self.bus_scheduled as f64 / self.bus_qubits as f64;
        let frac_magic = self.magic_scheduled as f64 / self.magic_qubits as f64;
        let tot_qubits = self.magic_scheduled + self.bus_scheduled + self.data_scheduled;
        info_sched!("  products:    {}/{} ({:.2})", pp_paths_len, to_schedule_len, frac_paths);
        info_sched!("  data:        {}/{} ({:.2})",
                    self.data_scheduled,
                    self.data_qubits,
                    frac_data);
        info_sched!("  bus:         {}/{} ({:.2})", self.bus_scheduled, self.bus_qubits, frac_bus);
        info_sched!("  magic:       {}/{} ({:.2})",
                    self.magic_scheduled,
                    self.magic_qubits,
                    frac_magic);
        // Only build the format string when path plotting is active (called rarely).
        if plotting {
            self.plot_info_str =
                format!("Step {} Products scheduled: {:.2}; qubits: data {:.2}, \
                            bus {:.2}, magic {:.2}, total qubits {}",
                        step_i, frac_paths, frac_data, frac_bus, frac_magic, tot_qubits);
        }

        self.data_scheduled = 0;
        self.bus_scheduled = 0;
        self.magic_scheduled = 0;
    }

    /// Increments count for a scheduled node of the given type.
    pub fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_scheduled += 1,
            NodeType::Magic => self.magic_scheduled += 1,
            NodeType::Data => self.data_scheduled += 1,
        }
    }

    /// Returns the accumulated plot info string.
    pub fn get_plot_info_str(&self) -> &str {
        &self.plot_info_str
    }
}

/// Tracks timing of major scheduling operations.
struct SchedulerTimers {
    schedule_product: IntermittentTimer,
    steiner_tree: IntermittentTimer,
    astar: IntermittentTimer,
    timestep: IntermittentTimer,
}

impl SchedulerTimers {
    /// Creates new timers for all scheduling operations.
    pub fn new() -> Self {
        SchedulerTimers { schedule_product: IntermittentTimer::new("schedule_pauli_product", ""),
                          steiner_tree: IntermittentTimer::new("compute_steiner_tree", ""),
                          astar: IntermittentTimer::new("compute_astar", ""),
                          timestep: IntermittentTimer::new("schedule_timestep", "") }
    }
}

/// Main scheduler that assigns Pauli products to timesteps and routes them through the topology.
/// Manages magic state cultivation, dependency tracking, and Clifford repetition logic.
pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng_exp: Exponential,
    magic_state_lambda: f64,
    plot_option: String,
    cultivation_times: Vec<i32>,
    stats: ScheduleStats,
    timestep_scheduled: Vec<(usize, Vec<PauliProduct>)>,
    scheduled_products: IndexSet<i32>,
    used: Vec<bool>,
    clifford_paths: IndexMap<i32, (usize, PauliProduct, Arc<TreeGraph>)>,
    stree_computation: SteinerTreeComputation,
    timers: SchedulerTimers,
    ready_magic_positions: Vec<(f32, f32)>,
    astar: AStarComputation,
    cannot_schedule: Vec<i32>,
    terminals_scratch: Vec<usize>,
    scheduled_ids_scratch: Vec<i32>,
    children_scratch: Vec<i32>,
    new_cultivation_times: Vec<i32>,
    precomputed_clifford_trees: HashMap<i32, Arc<TreeGraph>>,
    // Reusable scratch buffers for schedule_timestep (avoids per-timestep allocations).
    products_with_dist_scratch: Vec<(i32, f32)>,
    remaining_ids_scratch: Vec<i32>,
}

impl Scheduler {
    /// Creates a new scheduler for a circuit on a topology.
    /// `magic_state_lambda` controls magic state cultivation timing (exponential distribution parameter).
    pub fn new(circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
               plot_option: String, rseed: u32, stree_termination_threshold: usize)
               -> Self {
        if log_level != "none" {
            let circuit_stem = Path::new(&circuit.circuit_fname).file_stem()
                                                                .and_then(|s| s.to_str())
                                                                .unwrap_or("circuit");
            let sched_fname = format!("{}.sched_trace", circuit_stem);
            let level_filter = match log_level.to_lowercase().as_str() {
                "debug" => log::LevelFilter::Debug,
                "info" => log::LevelFilter::Info,
                _ => log::LevelFilter::Off,
            };
            simple_logging::log_to_file(&sched_fname, level_filter)
                .expect("Failed to initialize logging");
        }
        let num_data_qubits = topo.num_data_qubits;
        let num_bus_qubits = topo.num_bus_qubits;
        let num_magic_qubits = topo.num_magic_qubits;
        let num_nodes = topo.num_nodes;

        Scheduler { circuit,
                    topo,
                    rng_exp: Exponential::new(rseed),
                    magic_state_lambda,
                    plot_option,
                    cultivation_times: Vec::new(),
                    stats: ScheduleStats::new(num_data_qubits, num_bus_qubits, num_magic_qubits),
                    timestep_scheduled: Vec::new(),
                    scheduled_products: IndexSet::new(),
                    used: vec![false; num_nodes],
                    clifford_paths: IndexMap::new(),
                    stree_computation: SteinerTreeComputation::new(num_nodes,
                                                                   stree_termination_threshold),
                    timers: SchedulerTimers::new(),
                    ready_magic_positions: Vec::new(),
                    astar: AStarComputation::new(num_nodes),
                    cannot_schedule: Vec::new(),
                    terminals_scratch: Vec::new(),
                    scheduled_ids_scratch: Vec::new(),
                    children_scratch: Vec::new(),
                    new_cultivation_times: Vec::new(),
                    precomputed_clifford_trees: HashMap::new(),
                    products_with_dist_scratch: Vec::new(),
                    remaining_ids_scratch: Vec::new() }
    }

    /// Main scheduling algorithm: greedily assigns products to timesteps.
    /// Returns (total timesteps, total scheduled products).
    pub fn schedule_circuit(&mut self) -> io::Result<(usize, usize)> {
        let _timer = fn_timer!();
        self.rng_exp
            .try_set_params(1.0 / self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        self.init_magic_nodes();
        self.precompute_multi_term_clifford_trees();
        // Initialize scheduling
        let mut to_schedule: Vec<_> = self.circuit.initial_products().cloned().collect();
        debug_sched!("Initial to_schedule len {}", to_schedule.len());
        // Track parent relationships
        let mut remaining_parents: Vec<_> = (0..self.circuit.num_products()).map(|id| {
                                                let pp = self.circuit.get_product(id as i32);
                                                pp.parents.len()
                                            })
                                            .collect();
        let mut num_steps = 0;
        // Setup path plotting
        let mut plot_steps = 0;
        let mut path_dir = None;
        if self.plot_option.contains("paths") {
            let circuit_stem = Path::new(&self.circuit.circuit_fname).file_stem()
                                                                     .and_then(|s| s.to_str())
                                                                     .unwrap_or("circuit");
            let dir_name = format!("{}.paths", circuit_stem);
            std::fs::create_dir_all(&dir_name)?;
            path_dir = Some(dir_name);
            plot_steps = 100;
        }
        // Progress tracking
        let total_to_schedule = self.circuit.num_products();
        let mut prev_perc_complete = 0;
        if plot_steps == 0 {
            print!("Scheduling {} products:    ", total_to_schedule);
        }
        let plotting = path_dir.is_some();
        // Reuse pp_paths Vec across timesteps so the backing buffer is never freed and reallocated.
        let mut pp_paths: Vec<(PauliProduct, Arc<TreeGraph>)> = Vec::new();
        // Main scheduling loop
        while !to_schedule.is_empty() || !self.clifford_paths.is_empty() {
            self.timers.timestep.start();
            num_steps += 1;
            info_sched!("{}Step {}: {:?}{}",
                        _CYAN,
                        num_steps,
                        to_schedule.iter()
                                   .map(|pp| format!("{}:{}", pp.id, pp.to_operator_str()))
                                   .collect::<Vec<_>>(),
                        _RESET);
            if self.schedule_timestep(num_steps, &to_schedule, &mut pp_paths, plotting) {
                debug_sched!("Scheduled timestep {}", num_steps);
                debug_sched!("After timestep, to_schedule len {}", to_schedule.len());
                // Collect scheduled product ids for fast lookup
                self.scheduled_ids_scratch.clear();
                self.scheduled_ids_scratch.extend(pp_paths.iter().map(|(pp, _)| pp.id));
                // Remove scheduled products from to_schedule
                to_schedule.retain(|pp| !self.scheduled_ids_scratch.contains(&pp.id));
                debug_sched!("After purge, to_schedule len {}", to_schedule.len());
                // Add children from scheduled products
                self.children_scratch.clear();
                for (pp, _) in pp_paths.iter() {
                    if pp.gate_type.is_clifford() {
                        if let Some((count, _, _)) = self.clifford_paths.get(&pp.id) {
                            if *count == 2 {
                                debug_assert!(pp.gate_type.is_s() || pp.gate_type.is_sx());
                                // second round of S/SX clifford, don't add children
                                continue;
                            }
                        } else {
                            // first round of clifford - don't add children
                            continue;
                        }
                    }
                    // Add children to next round if all parents scheduled
                    for &child_id in &pp.children {
                        remaining_parents[child_id as usize] -= 1;
                        if remaining_parents[child_id as usize] == 0 {
                            if !self.children_scratch.contains(&child_id) {
                                self.children_scratch.push(child_id);
                            }
                        }
                    }
                }
                // First add back all cliffords that were scheduled for the first or second rounds
                for (pp, pp_path) in pp_paths.iter() {
                    if pp.gate_type.is_clifford() {
                        if let Some(clifford_path) = self.clifford_paths.get_mut(&pp.id) {
                            clifford_path.0 -= 1;
                            if clifford_path.0 == 0 {
                                self.clifford_paths.swap_remove(&pp.id);
                            }
                        } else {
                            let count = if pp.gate_type.is_cx() { 1 } else { 2 };
                            self.clifford_paths
                                .insert(pp.id, (count, (*pp).clone(), Arc::clone(pp_path)));
                        }
                    }
                }
                debug_sched!("After inserting previous round cliffords, to_schedule len {}",
                             to_schedule.len());
                // Extend next_to_schedule with children from scratch buffer
                to_schedule.extend(self.children_scratch.iter().map(|&id| {
                                                                   self.circuit
                                                                       .get_product(id)
                                                                       .clone()
                                                               }));
                debug_sched!("After adding {} children, to_schedule len {}",
                             self.children_scratch.len(),
                             to_schedule.len());
                // add products to the current step list
                let products_in_step: Vec<PauliProduct> =
                    pp_paths.iter().map(|(pp, _)| pp.clone()).collect();
                self.timestep_scheduled.push((num_steps, products_in_step));
                #[cfg(debug_assertions)]
                self.check_timestep(&pp_paths)?;
                self.scheduled_products.extend(pp_paths.iter().map(|(pp, _)| pp.id));
                let num_scheduled = self.scheduled_products.len();
                if num_steps >= plot_steps && (total_to_schedule - num_scheduled >= plot_steps) {
                    // Update progress counter if not plotting at the start or end of the loop
                    if num_steps == plot_steps {
                        print!("Scheduling {} products:    ", total_to_schedule);
                    }
                    let perc_complete = (num_scheduled * 100) / total_to_schedule;
                    if perc_complete > prev_perc_complete {
                        print!("\x08\x08\x08{:02}%", perc_complete);
                        std::io::stdout().flush()?;
                        prev_perc_complete = perc_complete;
                    }
                    if total_to_schedule - num_scheduled == plot_steps {
                        print!("\n");
                    }
                } else {
                    // Else plot if requested - plot_steps was set at the beginning of the loop
                    let plot_info_str = self.stats.get_plot_info_str();
                    assert!(!plot_info_str.is_empty());
                    let fname_added = format!(".{}", num_steps);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    self.topo
                        .plot(&fname_added, &pp_paths, &plot_info_str)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                }
            } else {
                debug_sched!("Could not schedule anything on timestep {}", num_steps);
                // if all magic nodes are available but nothing could be scheduled, this means
                // we must terminate with an error, since we should be able to schedule something
                if !self.topo.iter_nodes().any(|node| node.is_cultivating()) {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              format!("{}Cannot schedule on current layout{}",
                                                      _RED, _RESET)));
                }
                // otherwise try again
            }
            self.timers.timestep.stop();
        }
        self.stats.summarize(num_steps);
        println!("Magic state cultivation time:");
        let mean =
            self.cultivation_times.iter().sum::<i32>() as f64 / self.cultivation_times.len() as f64;
        let min = self.cultivation_times.iter().min().copied().unwrap_or(0);
        let max = self.cultivation_times.iter().max().copied().unwrap_or(0);
        println!("  number:  {}", self.cultivation_times.len());
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);

        let (num_calls, early_terminations) = self.stree_computation.get_call_counts();
        println!("Steiner tree computation called {} times, with {} ({:.2}%) early terminations",
                 num_calls,
                 early_terminations,
                 100.0 * early_terminations as f64 / num_calls as f64);
        self.timers.steiner_tree.done();
        self.timers.astar.done();
        self.timers.schedule_product.done();
        self.timers.timestep.done();

        #[cfg(debug_assertions)]
        self.check_clifford_repetitions()?;
        #[cfg(debug_assertions)]
        self.check_schedule()?;
        Ok((num_steps, self.scheduled_products.len()))
    }

    /// Initializes all magic nodes with random cultivation times (exponential distribution).
    fn init_magic_nodes(&mut self) {
        // Initialize magic nodes with busy counts
        // Collect magic node labels first to avoid borrow conflicts
        let magic_ids: Vec<usize> = self.topo
                                        .iter_nodes()
                                        .filter(|node| node.node_type == NodeType::Magic)
                                        .map(|node| node.id)
                                        .collect();
        for id in magic_ids {
            self.topo.get_node_mut(id).cultivation_time = self.gen_cultivation_time();
            self.topo.get_node_mut(id).busy_count = 0;
        }
    }

    /// Returns true if a product should have its Steiner tree precomputed.
    /// Multi-term non-T gates (e.g. CX) always route the same way regardless of magic state,
    /// so their trees can be fixed once on an empty topology.
    fn should_precompute(pp: &PauliProduct) -> bool {
        !pp.gate_type.is_t() && pp.operators.len() > 1
    }

    /// Builds a Steiner tree for a multi-term Clifford product on an empty topology.
    /// Only used during precomputation (self.used must be all-false on entry).
    fn precompute_steiner_tree(&mut self, pp: &PauliProduct) -> Option<TreeGraph> {
        if !self.get_terminal_nodes(pp) {
            return None;
        }
        let root_ids = self.get_root_nodes(&self.terminals_scratch[..]);
        if root_ids.is_empty() {
            return None;
        }
        self.stree_computation.compute(&self.topo,
                                       &self.used,
                                       &root_ids,
                                       &self.terminals_scratch,
                                       pp.gate_type,
                                       0)
    }

    /// Precomputes Steiner trees for all multi-term non-T products in the circuit,
    /// using an empty topology (no nodes marked used). The result is stored in
    /// `self.precomputed_clifford_trees` and used in `schedule_timestep` to skip
    /// runtime Steiner tree search whenever the precomputed route is free.
    fn precompute_multi_term_clifford_trees(&mut self) {
        let _timer = fn_timer!("precompute_clifford_trees");
        self.used.fill(false);
        let num_products = self.circuit.num_products();
        let mut num_precomputed = 0;
        for pp_id in 0..num_products {
            let pp = self.circuit.get_product(pp_id as i32).clone();
            if Self::should_precompute(&pp) {
                if let Some(tree) = self.precompute_steiner_tree(&pp) {
                    self.precomputed_clifford_trees.insert(pp.id, Arc::new(tree));
                    num_precomputed += 1;
                } else {
                    // Precomputation failed — this indicates a topology/circuit issue.
                    // The product will be skipped at runtime (never scheduled via Steiner).
                    eprintln!("{}Warning: failed to precompute tree for {}{}", _YELLOW, pp, _RESET);
                }
            }
        }
        println!("Precomputed {} multi-term Clifford trees", num_precomputed);
    }

    /// Schedules as many products as possible in a single timestep.
    /// Fills `pp_paths` with (product, routing tree) pairs; returns false if nothing scheduled.
    /// `pp_paths` is cleared on entry so the caller's buffer is reused across timesteps.
    fn schedule_timestep(&mut self, step_i: usize, to_schedule: &[PauliProduct],
                         pp_paths: &mut Vec<(PauliProduct, Arc<TreeGraph>)>,
                         plotting: bool)
                         -> bool {
        let mut num_avail_magic = self.update_cultivators();
        pp_paths.clear();
        // clear out used from previous timestep
        self.used.fill(false);
        // mark all previous first round cliffords as used
        for (_, (_, pp, pp_path)) in &self.clifford_paths {
            // mark clifford nodes as used so they can't be double scheduled
            for node_id in pp_path.iter_nodes() {
                self.used[node_id] = true;
            }
            // add the previous trees to the paths collection
            pp_paths.push(((*pp).clone(), Arc::clone(pp_path)));
        }
        // Presort products from smallest to largest tree size estimates.
        // Distances are precomputed once to avoid redundant calls during sorting.
        // Reuse the scratch buffer to avoid per-timestep Vec allocation.
        self.products_with_dist_scratch.clear();
        for pp in to_schedule.iter() {
            // Sequential borrows: tree_size_estimate borrow ends before push borrow begins.
            let est = self.tree_size_estimate(pp);
            self.products_with_dist_scratch.push((pp.id, est));
        }
        self.products_with_dist_scratch.sort_by(|(_, da), (_, db)| {
                                            da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal)
                                        });
        // Build remaining_to_schedule from to_schedule refs (lifetime tied to the parameter,
        // not to self), so later &mut self calls don't conflict. Use a local lookup map
        // to index into to_schedule by id without an O(n²) scan.
        let mut remaining_to_schedule: IndexMap<i32, &PauliProduct> = {
            let to_sched_lookup: HashMap<i32, &PauliProduct> =
                to_schedule.iter().map(|pp| (pp.id, pp)).collect();
            self.products_with_dist_scratch
                .iter()
                .filter_map(|(id, _)| to_sched_lookup.get(id).map(|&pp| (*id, pp)))
                .collect()
        }; // to_sched_lookup drops here; remaining_to_schedule holds to_schedule-lifetime refs
        info_sched!("  Remaining to schedule: {}", remaining_to_schedule.len());
        // First pass: schedule multi-term Cliffords using precomputed trees.
        // Iterate remaining products and look up each in the precomputed tree map.
        // Only schedules if every node in the precomputed tree is currently free.
        // Products whose tree is blocked are dropped from remaining and skipped this timestep.
        // Reuse scratch buffer for the key snapshot to avoid allocation.
        self.remaining_ids_scratch.clear();
        self.remaining_ids_scratch.extend(remaining_to_schedule.keys().copied());
        for &pp_id in &self.remaining_ids_scratch {
            // Clone the Arc immediately to end the borrow on precomputed_clifford_trees.
            let Some(tree) = self.precomputed_clifford_trees.get(&pp_id).map(Arc::clone) else {
                continue; // No precomputed tree; leave in remaining for find_next_product.
            };
            let all_free = tree.iter_nodes().all(|nid| !self.used[nid]);
            // Remove from remaining: precomputed products are not handed to find_next_product.
            remaining_to_schedule.swap_remove(&pp_id);
            let pp = self.circuit.get_product(pp_id).clone();
            if all_free {
                for node_id in tree.iter_nodes() {
                    let node_type = self.topo.get_node(node_id).node_type;
                    self.stats.inc(node_type);
                    self.used[node_id] = true;
                }
                info_sched!("  Scheduled product {} (precomputed) with {} nodes and {} edges",
                            pp,
                            tree.num_nodes,
                            tree.num_edges);
                pp_paths.push((pp, tree));
            } else {
                // Tree is blocked; mark this product's data qubits as used so nothing else
                // occupies them this timestep, mirroring find_next_product's behaviour.
                for op in &pp.operators {
                    if op.basis == 'Y' {
                        self.used[self.topo.get_data_node_id(op.qubit, 'X')] = true;
                        self.used[self.topo.get_data_node_id(op.qubit, 'Z')] = true;
                    } else {
                        self.used
                            [self.topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase())] =
                            true;
                    }
                }
            }
        }
        // Second pass: schedule remaining products (T gates, M, S/SX) via find_next_product.
        while !remaining_to_schedule.is_empty() {
            let best_pp =
                self.find_next_product(&remaining_to_schedule, pp_paths.len(), num_avail_magic);
            if let Some((best_pp_idx, best_graph)) = best_pp {
                let pp: &PauliProduct = remaining_to_schedule.get(&best_pp_idx).unwrap();
                info_sched!("  Scheduled product {} with {} nodes and {} edges",
                            pp,
                            best_graph.num_nodes,
                            best_graph.num_edges);
                // Update node statistics and mark as used
                for node_id in best_graph.iter_nodes() {
                    let node = self.topo.get_node(node_id);
                    self.stats.inc(node.node_type);
                    self.used[node.id] = true;
                }
                pp_paths.push(((*pp).clone(), Arc::new(best_graph)));
                // don't schedule again
                remaining_to_schedule.swap_remove(&best_pp_idx);
                if pp.gate_type.is_t() {
                    num_avail_magic -= 1;
                }
            }
            // remove all those we cannot schedule this timestep
            // cannot_schedule is now self.cannot_schedule, but we can't borrow self mutably
            // while calling swap_remove on remaining_to_schedule (which is also borrowed from self? No - it's a local).
            // Safe to use self.cannot_schedule here since find_next_product has returned.
            for i in 0..self.cannot_schedule.len() {
                remaining_to_schedule.swap_remove(&self.cannot_schedule[i]);
            }
        }
        self.stats.update(step_i, pp_paths.len(), to_schedule.len(), num_avail_magic as usize,
                          plotting);
        if pp_paths.is_empty() {
            if num_avail_magic > 0 {
                // if any magic node is available and we scheduled nothing, then we must terminate
                // with an error, since we should be able to schedule something
                panic!("{}Step {}: Cannot schedule products [{}] on current layout ({} magic){}",
                       _RED,
                       step_i,
                       to_schedule.iter()
                                  .map(|pp| pp.to_operator_str())
                                  .collect::<Vec<_>>()
                                  .join(", "),
                       num_avail_magic,
                       _RESET);
            }
            false
        } else {
            true
        }
    }

    /// Updates magic node cultivation state: increments busy counts and generates new cultivation
    /// times for nodes just scheduled. Returns count of ready (cultivation_time=0) magic nodes.
    fn update_cultivators(&mut self) -> usize {
        // Collect magic nodes that need new busy counts
        let num_used_magic_nodes =
            self.topo
                .iter_nodes()
                .filter(|node| self.used[node.id] && node.node_type == NodeType::Magic)
                .count();
        // Generate new cultivation times for magic nodes
        self.new_cultivation_times.clear();
        for _ in 0..num_used_magic_nodes {
            let t = self.gen_cultivation_time();
            self.new_cultivation_times.push(t);
        }
        // Update busy counts and reset used flags
        let mut num_avail_magic = 0;
        let mut cultivation_time_index = 0;
        for node in self.topo.iter_nodes_mut() {
            if self.used[node.id] && node.node_type == NodeType::Magic {
                node.cultivation_time = self.new_cultivation_times[cultivation_time_index];
                node.busy_count = 0;
                cultivation_time_index += 1;
            } else if !self.used[node.id] && node.is_cultivating() {
                node.busy_count += 1;
                if node.busy_count == node.cultivation_time {
                    self.cultivation_times.push(node.cultivation_time);
                    node.cultivation_time = 0;
                    node.busy_count = 0;
                }
            }
            if node.node_type == NodeType::Magic && node.cultivation_time == 0 {
                num_avail_magic += 1;
            }
        }
        // Rebuild cache of ready magic positions used by tree_size_estimate.
        self.ready_magic_positions =
            self.topo
                .iter_nodes()
                .filter(|n| n.node_type == NodeType::Magic && n.cultivation_time == 0)
                .map(|n| n.pos)
                .collect();
        info_sched!("  Available magic {}", num_avail_magic);
        num_avail_magic
    }

    /// Searches remaining products for the netxt schedulable one
    /// Returns product ID and tree, or None if no schedulable product found.
    /// Products that cannot be scheduled this timestep are stored in self.cannot_schedule.
    fn find_next_product(&mut self, remaining_to_schedule: &IndexMap<i32, &PauliProduct>,
                         num_scheduled: usize, num_avail_magic: usize)
                         -> Option<(i32, TreeGraph)> {
        self.cannot_schedule.clear();

        for (&pp_i, &pp) in remaining_to_schedule {
            // Should-precompute products are only scheduled via the precomputed-tree pass.
            // If one reaches here (e.g. precomputation failed), skip it this timestep.
            if Self::should_precompute(pp) {
                self.cannot_schedule.push(pp_i);
                continue;
            }
            let pp_graph = if num_avail_magic == 0 && pp.gate_type.is_t() {
                None
            } else {
                self.timers.schedule_product.start();
                let pp_graph = self.schedule_pauli_product(pp, num_scheduled);
                self.timers.schedule_product.stop();
                pp_graph
            };
            if let Some(pp_graph) = pp_graph {
                info_sched!("  Schedule {} on graph", pp.id);
                return Some((pp_i, pp_graph));
            } else {
                info_sched!("  Could not schedule {} on graph", pp.id);
                // Mark dependent nodes as used
                for op in &pp.operators {
                    if op.basis == 'Y' {
                        self.used[self.topo.get_data_node_id(op.qubit, 'X')] = true;
                        self.used[self.topo.get_data_node_id(op.qubit, 'Z')] = true;
                    } else {
                        self.used
                            [self.topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase())] =
                            true;
                    }
                }
                self.cannot_schedule.push(pp_i);
            }
        }
        None
    }

    /// Attempts to route a single Pauli product through the topology.
    /// Uses A* for single-qubit T gates, Steiner tree for others.
    /// Returns a routing tree or None if no valid routing exists.
    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct, num_scheduled: usize)
                              -> Option<TreeGraph> {
        info_sched!("  Trying to schedule product {}", pauli_product);
        // Terminal nodes contain only the data qubits
        if !self.get_terminal_nodes(pauli_product) {
            info_sched!("    Cannot schedule {}: no data nodes found in working graph",
                        pauli_product.id);
            return None;
        }
        // Handle single data node case
        if self.terminals_scratch.len() == 1 && pauli_product.gate_type.is_m() {
            let node_id = self.terminals_scratch[0];
            let node = self.topo.get_node(node_id);
            if self.used[node.id] {
                info_sched!("    Cannot schedule {}: node for M {} is used",
                            pauli_product.id,
                            node.label);
                return None;
            }
            let mut g = TreeGraph::new(self.topo.num_nodes);
            g.add_node(node);
            return Some(g);
        } else if pauli_product.gate_type.is_s() || pauli_product.gate_type.is_sx() {
            let node_id = self.terminals_scratch[0];
            let node = self.topo.get_node(node_id);
            if self.used[node.id] {
                info_sched!("    Cannot schedule {}: node for {:?} {} is used",
                            pauli_product.id,
                            pauli_product.gate_type,
                            node.label);
                return None;
            }
            for nb_id in &node.nbors {
                let nb = self.topo.get_node(*nb_id);
                if nb.pos.1 == node.pos.1 {
                    info_sched!("    product {} on node {} has available ancilla {}",
                                pauli_product,
                                node.label,
                                nb.label);
                    if !self.used[*nb_id] {
                        let mut g = TreeGraph::new(self.topo.num_nodes);
                        g.add_node(node);
                        g.add_node(nb);
                        g.add_edge(node_id, *nb_id);
                        return Some(g);
                    }
                }
            }
            info_sched!("    Cannot schedule S/SX {}: no available ancilla", pauli_product.id);
            return None;
        } else {
            // first check that all terminals are accessible
            if self.terminals_scratch.iter().any(|node_id| self.used[*node_id]) {
                info_sched!("    Cannot schedule {}: used terminals", pauli_product.id);
                return None;
            }
            // Get root nodes next to terminals
            let root_ids = self.get_root_nodes(&self.terminals_scratch[..]);
            if root_ids.is_empty() {
                info_sched!("    Cannot schedule {}: no roots available", pauli_product.id);
                return None;
            }
            let g = if pauli_product.gate_type.is_t() && pauli_product.operators.len() == 1 {
                // Single-qubit T gate (X, Z, or Y): use multi-source A*.
                // For X/Z: one root, one terminal. For Y: two roots, two terminals.
                self.timers.astar.start();
                let g = self.astar.compute(&self.terminals_scratch[..],
                                           &root_ids[..],
                                           &self.topo,
                                           &self.used,
                                           &self.ready_magic_positions);
                self.timers.astar.stop();
                g
            } else {
                debug_assert!(!Self::should_precompute(pauli_product),
                              "should_precompute product {:?} reached Steiner path",
                              pauli_product.id);
                self.timers.steiner_tree.start();
                let g = self.stree_computation.compute(&self.topo,
                                                       &self.used,
                                                       &root_ids,
                                                       &self.terminals_scratch,
                                                       pauli_product.gate_type,
                                                       num_scheduled);
                self.timers.steiner_tree.stop();
                g
            };
            if let Some(g) = g {
                return Some(g);
            }
            info_sched!("    Cannot schedule {}: no steiner tree found", pauli_product.id);
            None
        }
    }

    /// Extracts the data qubit nodes that a product operates on (terminals for tree routing).
    /// For Y operators, both X and Z bases are included. Returns false if any terminal is unavailable.
    /// Results are stored in self.terminals_scratch.
    fn get_terminal_nodes(&mut self, pauli_product: &PauliProduct) -> bool {
        // Initially terminal nodes contain only the data qubits
        self.terminals_scratch.clear();
        for op in &pauli_product.operators {
            if op.basis == 'Y' {
                for basis in ['X', 'Z'] {
                    let node_id = self.topo.get_data_node_id(op.qubit, basis);
                    let node = self.topo.get_node(node_id);
                    // Check if node is already used
                    if self.used[node.id] {
                        info_sched!("  Node {} is already used", node.label);
                        return false;
                    }
                    // check for at least one unused magic or bus nb
                    if !node.nbors.iter().any(|nb_id| {
                                             let nb = self.topo.get_node(*nb_id);
                                             !self.used[nb.id]
                                         })
                    {
                        info_sched!("  No unused neighbors for node {}", node.id);
                        return false;
                    }
                    self.terminals_scratch.push(node.id);
                }
            } else {
                let node_id = self.topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase());
                let node = self.topo.get_node(node_id);
                // Check if node is already used
                if self.used[node.id] {
                    info_sched!("  Node {} is already used", node.label);
                    return false;
                }
                // check for at least one unused magic or bus nb
                if !node.nbors.iter().any(|nb_id| {
                                         let nb = self.topo.get_node(*nb_id);
                                         !self.used[nb.id]
                                     })
                {
                    info_sched!("  No unused neighbors for node {}", node.id);
                    return false;
                }
                self.terminals_scratch.push(node.id);
            }
        }
        true
    }

    /// Finds routing nodes adjacent to each terminal (roots for tree construction).
    /// Prefers vertical (top/bottom) roots for Y/paired operations; falls back to side roots.
    fn get_root_nodes(&self, terminals: &[usize]) -> Vec<usize> {
        let mut root_ids: Vec<usize> = Vec::new();
        // need to get a root node for every terminal
        let mut unmatched_count: usize = terminals.len();
        for node_id in terminals.iter() {
            let node = self.topo.get_node(*node_id);
            let paired_node = self.topo.get_node(node.paired_data_id.unwrap());
            let mut pair_found = false;
            // first look for paired nodes (top/bottom)
            if terminals.contains(&paired_node.id) {
                // we have already converted Ys into XZ in get_terminal_nodes
                let pair = if node.label.contains("X") { Some("XX") } else { Some("ZZ") };
                debug_sched!("    Found {} pair {},{} in terminals",
                             pair.unwrap(),
                             node.label,
                             paired_node.label);
                for nb_id in node.nbors.iter() {
                    let nb = self.topo.get_node(*nb_id);
                    if self.used[nb.id] || !nb.is_routing() {
                        continue;
                    }
                    // If we are using top/bottom
                    if (pair == Some("XX") && nb.pos.1 < node.pos.1)
                       || (pair == Some("ZZ") && nb.pos.1 > node.pos.1)
                    {
                        if !root_ids.contains(nb_id) {
                            root_ids.push(*nb_id);
                        }
                        // saturating_sub guards against the second iteration over a matched pair
                        unmatched_count = unmatched_count.saturating_sub(2);
                        pair_found = true;
                        break;
                    }
                }
            };
            if !pair_found {
                for nb_id in node.nbors.iter() {
                    let nb = self.topo.get_node(*nb_id);
                    if self.used[nb.id] || !nb.is_routing() {
                        continue;
                    }
                    // Only include neighbors on the side (same row, different column)
                    if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        if !root_ids.contains(nb_id) {
                            root_ids.push(*nb_id);
                        }
                        unmatched_count = unmatched_count.saturating_sub(1);
                        break;
                    }
                }
            }
        }
        if unmatched_count > 0 {
            debug_sched!("    could not find root nodes for {} unmatched terminals",
                         unmatched_count);
            return Vec::new();
        }
        root_ids
    }

    /// Scheduling sort key for a product (used for greedy prioritization).
    /// - T gates:        Manhattan distance from terminal centroid to nearest ready magic node.
    ///                   Returns f32::MAX when no magic node is ready.
    /// - Clifford gates: sum of pairwise Manhattan distances between terminals (a proxy for
    ///                   Steiner tree size; 0 for a single terminal).
    fn tree_size_estimate(&self, pp: &PauliProduct) -> f32 {
        // Collect terminal positions.
        let mut positions: Vec<(f32, f32)> = Vec::new();
        for op in &pp.operators {
            if op.basis == 'Y' {
                for basis in ['X', 'Z'] {
                    let node = self.topo.get_node(self.topo.get_data_node_id(op.qubit, basis));
                    positions.push(node.pos);
                }
            } else {
                let node =
                    self.topo.get_node(self.topo.get_data_node_id(op.qubit,
                                                                  op.basis.to_ascii_uppercase()));
                positions.push(node.pos);
            }
        }
        if positions.is_empty() {
            return f32::MAX;
        }
        if pp.gate_type.is_t() {
            // Distance from centroid to nearest ready magic node.
            let n = positions.len() as f32;
            let cx = positions.iter().map(|p| p.0).sum::<f32>() / n;
            let cy = positions.iter().map(|p| p.1).sum::<f32>() / n;
            self.ready_magic_positions
                .iter()
                .map(|p| (p.0 - cx).abs() + (p.1 - cy).abs())
                .fold(f32::MAX, f32::min)
        } else {
            // Sum of pairwise Manhattan distances between terminals.
            let mut total = 0.0f32;
            for i in 0..positions.len() {
                for j in (i + 1)..positions.len() {
                    total += (positions[i].0 - positions[j].0).abs()
                             + (positions[i].1 - positions[j].1).abs();
                }
            }
            total
        }
    }

    /// Generates a random magic state cultivation time from exponential distribution.
    fn gen_cultivation_time(&mut self) -> i32 {
        let cultivation_time = self.rng_exp.sample().round() as i32; // + 1;
        cultivation_time
    }

    /// End-of-run completeness check (debug builds only):
    /// verifies that every product in the circuit was scheduled at least once.
    /// Per-timestep checks (dependency order, tree validity, overlap) are done in check_timestep.
    #[cfg(debug_assertions)]
    fn check_schedule(&self) -> io::Result<()> {
        let num_products = self.circuit.num_products();
        let mut errors: Vec<String> = Vec::new();
        for pp_id in 0..num_products as i32 {
            if !self.scheduled_products.contains(&pp_id) {
                errors.push(format!("  product {} was never scheduled", pp_id));
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(io::ErrorKind::Other,
                                      format!("Completeness errors:\n{}", errors.join("\n"))));
        }
        println!("Schedule check passed: all {} products scheduled", num_products);
        Ok(())
    }

    /// Per-timestep validation (debug builds only), called immediately after each timestep
    /// while pp_paths is still in scope. Checks:
    /// 1. No non-Clifford product is scheduled twice.
    /// 2. All parents of each product were scheduled in a prior timestep.
    /// 3. Every product's routing tree contains all required terminal data nodes.
    /// 4. T-gate trees have a magic root node.
    /// 5. No two products in this timestep share a topology node.
    #[cfg(debug_assertions)]
    fn check_timestep(&self, pp_paths: &[(PauliProduct, Arc<TreeGraph>)]) -> io::Result<()> {
        let mut step_used = vec![false; self.topo.num_nodes];
        for (pp, tree) in pp_paths {
            // 1. Already scheduled?
            if self.scheduled_products.contains(&pp.id) && !pp.gate_type.is_clifford() {
                return Err(io::Error::new(io::ErrorKind::Other,
                                          format!("product {} scheduled twice", pp.id)));
            }
            // 2. All parents scheduled in a prior timestep?
            for &parent_id in &pp.parents {
                if !self.scheduled_products.contains(&parent_id) {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              format!("product {} scheduled before parent {}",
                                                      pp.id, parent_id)));
                }
            }
            // 3. Terminal data nodes present in tree?
            for op in &pp.operators {
                if op.basis == 'Y' {
                    for basis in ['X', 'Z'] {
                        let nid = self.topo.get_data_node_id(op.qubit, basis);
                        if !tree.contains_node(nid) {
                            return Err(io::Error::new(io::ErrorKind::Other,
                                                      format!("product {} (step 3): terminal \
                                                               qubit {} basis {} missing from tree",
                                                              pp.id, op.qubit, basis)));
                        }
                    }
                } else {
                    let nid = self.topo.get_data_node_id(op.qubit, op.basis.to_ascii_uppercase());
                    if !tree.contains_node(nid) {
                        return Err(io::Error::new(io::ErrorKind::Other,
                                                  format!("product {} terminal qubit {} basis \
                                                           {} missing from tree",
                                                          pp.id, op.qubit, op.basis)));
                    }
                }
            }
            // 4. Magic root node present for T gates?
            if pp.gate_type.is_t() {
                match tree.root_node_id {
                    None => {
                        return Err(io::Error::new(io::ErrorKind::Other,
                                                  format!("product {}: T gate has no magic root \
                                                           node",
                                                          pp.id)));
                    }
                    Some(magic_id) => {
                        if self.topo.get_node(magic_id).node_type != NodeType::Magic {
                            return Err(io::Error::new(io::ErrorKind::Other,
                                                      format!("product {}: root node {} is not \
                                                               a Magic node",
                                                              pp.id, magic_id)));
                        }
                    }
                }
            }
            // 5. No overlap with other products in this timestep?
            for node_id in tree.iter_nodes() {
                if step_used[node_id] {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              format!("product {} shares node '{}' with another \
                                                       product in the same timestep",
                                                      pp.id,
                                                      self.topo.get_node(node_id).label)));
                }
                step_used[node_id] = true;
            }
        }
        Ok(())
    }

    /// Writes the schedule to a file with products colored and listed per timestep.
    pub fn print_schedule(&self, hdr: &str) -> io::Result<()> {
        let _timer = fn_timer!();
        debug_sched!("Printing schedule");
        let circuit_stem = Path::new(&self.circuit.circuit_fname).file_stem()
                                                                 .and_then(|s| s.to_str())
                                                                 .unwrap_or("circuit");
        let output_fname = format!("{}.schedule", circuit_stem);

        let file = File::create(&output_fname)?;
        let mut buf_file = BufWriter::new(file);

        let max_step: usize =
            self.timestep_scheduled.last().map(|(step_i, _)| *step_i).unwrap_or(0);
        let max_width = max_step.to_string().len();
        let tot_products = self.timestep_scheduled.iter().map(|(_, v)| v.len()).sum::<usize>();
        writeln!(buf_file, "{}", hdr)?;
        writeln!(buf_file, "# Total active steps: {}", self.timestep_scheduled.len())?;
        writeln!(buf_file, "# Total steps: {}", max_step)?;
        writeln!(buf_file, "# Total products: {}", tot_products)?;
        writeln!(buf_file, "# Parallelism: {:.2}", tot_products as f64 / max_step as f64)?;

        let colors = [_GREEN, _RED, _YELLOW, _BLUE, _MAGENTA, _CYAN, _WHITE, _LGREEN, _LRED,
                      _LYELLOW, _LBLUE, _LMAGENTA, _LCYAN, _LWHITE];

        // FIXME: check that each CX is repeated 2x exactly, and each S/SX is repeated 3x
        let mut prev_cx: IndexSet<i32> = IndexSet::new();
        for (step_i, step_products) in &self.timestep_scheduled {
            let mut sorted_products = step_products.clone();
            sorted_products.sort_by_key(|pp| {
                               pp.operators.iter().map(|op| op.qubit).min().unwrap_or(usize::MAX)
                           });
            let mut combined_chars = vec!['_'; self.circuit.num_qubits];
            let mut combined_colors = vec![_RESET; self.circuit.num_qubits];
            for (idx, pp) in sorted_products.iter().enumerate() {
                let color = colors[idx % colors.len()];
                for op in &pp.operators {
                    if op.qubit < self.circuit.num_qubits {
                        combined_chars[op.qubit] = op.basis;
                        combined_colors[op.qubit] = color;
                    }
                }
                if pp.gate_type.is_cx() {
                    if !prev_cx.swap_remove(&pp.id) {
                        debug_sched!("  first round of CX {} {}", pp.id, pp);
                        prev_cx.insert(pp.id);
                        // First round is qubit 0, clear qubit 1
                        let qubit = pp.operators[1].qubit;
                        combined_colors[qubit] = _RESET;
                        combined_chars[qubit] = '_';
                    } else {
                        debug_sched!("  second round of CX {} {}", pp.id, pp);
                        // Second round is qubit 1, clear qubit 0
                        let qubit = pp.operators[0].qubit;
                        combined_colors[qubit] = _RESET;
                        combined_chars[qubit] = '_';
                    }
                }
            }
            write!(buf_file, "{:width$}: ", step_i, width = max_width)?;
            for i in 0..self.circuit.num_qubits {
                write!(buf_file, "{}{}", combined_colors[i], combined_chars[i])?;
            }
            let mut id_string = String::new();
            for (idx, pp) in sorted_products.iter().enumerate() {
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(" {}{}<{:?}>", color, pp.id, pp.gate_type));
            }
            writeln!(buf_file, "{}{}", id_string, _RESET)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }

    /// Validates that CX gates are scheduled exactly 2 consecutive times and S/SX 3 consecutive
    /// times (debug only).
    #[cfg(debug_assertions)]
    fn check_clifford_repetitions(&self) -> io::Result<()> {
        // map the product id to a vector containing the timesteps on which the product was found
        let mut cx_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        let mut s_counts: IndexMap<i32, Vec<usize>> = IndexMap::new();
        for (step_i, step_products) in &self.timestep_scheduled {
            for pp in step_products {
                if pp.gate_type.is_cx() {
                    let steps = cx_counts.entry(pp.id).or_insert(Vec::new());
                    steps.push(*step_i);
                } else if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                    let steps = s_counts.entry(pp.id).or_insert(Vec::new());
                    steps.push(*step_i);
                }
            }
        }
        let mut errors = Vec::new();
        for (pp_id, steps) in &cx_counts {
            let pp = self.circuit.get_product(*pp_id);
            if pp.gate_type.is_cx() {
                if steps.len() != 2 || steps[0] != steps[1] - 1 {
                    errors.push(format!("  product {} not scheduled 2x {:?}", pp, steps));
                }
            }
        }
        for (pp_id, steps) in &s_counts {
            let pp = self.circuit.get_product(*pp_id);
            if pp.gate_type.is_s() || pp.gate_type.is_sx() {
                if steps.len() != 3 || steps[0] != steps[1] - 1 || steps[1] != steps[2] - 1 {
                    errors.push(format!("  product {} not scheduled 3x {:?}", pp, steps));
                }
            }
        }
        if !errors.is_empty() {
            return Err(io::Error::new(io::ErrorKind::Other,
                                      format!("Clifford repetition errors:\n{}",
                                              errors.join("\n"))));
        }
        println!("Clifford repetition check passed ({} CX, {} S/SX products)",
                 cx_counts.len(),
                 s_counts.len());
        Ok(())
    }
}
