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

use indexmap::IndexSet;
use rand_simple::Exponential;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

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
}

impl ScheduleStats {
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
                        sum_magic_unused: 0 }
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
                  magic_unused: usize)
                  -> String {
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
        let title =
            format!("Step {} Products scheduled: {:.2}; qubits: data {:.2}, \
                        bus {:.2}, magic {:.2}, total qubits {}",
                    step_i, frac_paths, frac_data, frac_bus, frac_magic, tot_qubits);

        self.data_scheduled = 0;
        self.bus_scheduled = 0;
        self.magic_scheduled = 0;
        title
    }

    pub fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_scheduled += 1,
            NodeType::Magic => self.magic_scheduled += 1,
            NodeType::Data => self.data_scheduled += 1,
        }
    }
}

struct SchedulerTimers {
    schedule_product: IntermittentTimer,
    steiner_tree: IntermittentTimer,
    timestep: IntermittentTimer,
}

impl SchedulerTimers {
    pub fn new() -> Self {
        SchedulerTimers { schedule_product: IntermittentTimer::new("schedule_pauli_product", ""),
                          steiner_tree: IntermittentTimer::new("get_steiner_tree", ""),
                          timestep: IntermittentTimer::new("schedule_timestep", "") }
    }
}

pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng_exp: Exponential,
    magic_state_lambda: f64,
    plot_option: String,
    cultivation_times: Vec<i32>,
    stats: ScheduleStats,
    scheduled_products: Vec<Vec<PauliProduct>>,
    used: Vec<bool>,
    stree_computation: SteinerTreeComputation,
    timers: SchedulerTimers,
}

impl Scheduler {
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
                    scheduled_products: Vec::new(),
                    used: vec![false; num_nodes],
                    stree_computation: SteinerTreeComputation::new(num_nodes,
                                                                   stree_termination_threshold),
                    timers: SchedulerTimers::new() }
    }

    pub fn schedule_circuit(&mut self, best_fit: bool) -> io::Result<(usize, usize)> {
        let _timer = fn_timer!();
        /*
        #[cfg(debug_assertions)]
        for node in self.topo.iter_nodes() {
            debug_sched!("Node id {} is {}", node.id, node.label);
        } */
        self.rng_exp
            .try_set_params(1.0 / self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        // Initialize magic nodes with busy counts
        // Collect magic node labels first to avoid borrow conflicts
        let magic_ids: Vec<usize> = self.topo
                                        .iter_nodes()
                                        .filter(|node| node.node_type == NodeType::Magic)
                                        .map(|node| node.id.clone())
                                        .collect();
        for id in magic_ids {
            self.topo.get_node_mut(id).cultivation_time = self.gen_cultivation_time();
            self.topo.get_node_mut(id).busy_count = 0;
        }
        // Initialize scheduling
        let mut to_schedule: Vec<_> = self.circuit.initial_products().cloned().collect();
        // Track parent relationships
        let mut remaining_parents: Vec<_> = (0..self.circuit.num_products()).map(|id| {
                                                let pp = self.circuit.get_product(id as i32);
                                                pp.parents.len()
                                            })
                                            .collect();

        let mut scheduled = IndexSet::new();
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
        // Main scheduling loop
        while !to_schedule.is_empty() {
            self.timers.timestep.start();
            num_steps += 1;
            info_sched!("{}Step {}: {:?}{}",
                        _CYAN,
                        num_steps,
                        to_schedule.iter()
                                   .map(|pp| format!("{}:{}", pp.id, pp.get_product_str()))
                                   .collect::<Vec<_>>(),
                        _RESET);

            let (title_str, pp_paths, mut next_to_schedule) =
                self.schedule_timestep(num_steps, &to_schedule, best_fit);
            for pp in next_to_schedule.clone() {
                if scheduled.contains(&pp.id) {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              format!("Next to schedule: {} is already scheduled",
                                                      pp.id)));
                }
            }

            if pp_paths.is_none() {
                // if all magic nodes are available but nothing could be scheduled, this means
                // we must terminate with an error, since we should be able to schedule something
                if !self.topo.iter_nodes().any(|node| node.is_cultivating()) {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              format!("{}Cannot schedule on current layout{}",
                                                      _RED, _RESET)));
                }
                to_schedule = next_to_schedule;
                continue;
            }
            // Process scheduled products
            if let Some(ref pp_paths) = pp_paths {
                let mut products_in_step = Vec::new();
                let mut children_to_schedule = IndexSet::new();
                for (pp, _) in pp_paths {
                    products_in_step.push(pp.clone());
                    // Add children to next round if all parents scheduled
                    for &child_id in &pp.children {
                        remaining_parents[child_id as usize] -= 1;
                        if remaining_parents[child_id as usize] == 0 {
                            children_to_schedule.insert(child_id);
                        }
                    }
                }
                self.scheduled_products.push(products_in_step);
                // Extend next_to_schedule with children from IndexSet
                next_to_schedule.extend(children_to_schedule.iter().map(|&id| {
                                                                       self.circuit
                                                                           .get_product(id)
                                                                           .clone()
                                                                   }));
                for (pp, _) in pp_paths {
                    self.check_dependencies(pp, &scheduled)?;
                    scheduled.insert(pp.id);
                }
            }
            // Update progress
            if num_steps >= plot_steps && (total_to_schedule - scheduled.len() >= plot_steps) {
                if num_steps == plot_steps {
                    print!("Scheduling {} products:    ", total_to_schedule);
                }
                let perc_complete = (scheduled.len() * 100) / total_to_schedule;
                if perc_complete > prev_perc_complete {
                    print!("\x08\x08\x08{:02}%", perc_complete);
                    std::io::stdout().flush()?;
                    prev_perc_complete = perc_complete;
                }
                if total_to_schedule - scheduled.len() == plot_steps {
                    print!("\n");
                }
            } else {
                // Plot if requested
                if title_str.is_some() {
                    let fname_added = format!(".{}", num_steps);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir.as_ref().unwrap())?;
                    self.topo
                        .plot(&fname_added, pp_paths.as_ref().unwrap(), &title_str.unwrap())
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                }
            }
            to_schedule = next_to_schedule;
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
        self.timers.schedule_product.done();
        self.timers.timestep.done();

        self.print_schedule()?;
        Ok((num_steps, scheduled.len()))
    }

    fn schedule_timestep(
        &mut self, step_i: usize, to_schedule: &[PauliProduct], best_fit: bool)
        -> (Option<String>, Option<Vec<(PauliProduct, TreeGraph)>>, Vec<PauliProduct>) {
        let mut num_avail_magic = self.setup_timestep();
        let mut pp_paths = Vec::with_capacity(to_schedule.len().min(10));
        let mut next_to_schedule = Vec::with_capacity(to_schedule.len().min(10));
        let mut _num_dependent_nodes = 0;

        let mut remaining_to_schedule: IndexSet<usize> = (0..to_schedule.len()).collect();
        // Presort products from most to least resource-intensive
        remaining_to_schedule.sort_by_key(|&idx| {
                                 std::cmp::Reverse(to_schedule[idx].count_weighted_terms())
                             });
        info_sched!("  Remaining to schedule: {}", remaining_to_schedule.len());
        while !remaining_to_schedule.is_empty() {
            let mut to_remove = Vec::new();
            let mut best_pp: Option<(usize, TreeGraph)> = None;
            let mut best_pp_graph_size = usize::MAX;
            let mut best_pp_term_weight = 0;

            for &pp_i in &remaining_to_schedule {
                let pp = &to_schedule[pp_i];
                let pp_term_weight = pp.count_weighted_terms();
                if pp_term_weight < best_pp_term_weight {
                    info_sched!("  Skip lower weight product {}", pp);
                    continue;
                }
                let pp_graph = if num_avail_magic == 0 && pp.gate_type.is_t() {
                    None
                } else {
                    self.timers.schedule_product.start();
                    let pp_graph = self.schedule_pauli_product(pp, pp_paths.len());
                    self.timers.schedule_product.stop();
                    pp_graph
                };
                if pp_graph.is_none() {
                    info_sched!("  Could not schedule {} on graph", pp.id);
                    next_to_schedule.push(pp.clone());
                    // Mark dependent nodes as used
                    for op in &pp.operators {
                        if op.basis == 'Y' {
                            let node_label_x = format!("d{}{}", op.qubit, 'X');
                            self.used[self.topo.get_node_id_from_label(&node_label_x)] = true;
                            let node_label_z = format!("d{}{}", op.qubit, 'Z');
                            self.used[self.topo.get_node_id_from_label(&node_label_z)] = true;
                            _num_dependent_nodes += 2;
                        } else {
                            let node_label =
                                format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                            self.used[self.topo.get_node_id_from_label(&node_label)] = true;
                            _num_dependent_nodes += 1;
                        }
                    }
                    to_remove.push(pp_i);
                } else {
                    let pp_graph = pp_graph.unwrap();
                    if pp_term_weight >= best_pp_term_weight {
                        // regard the best graph as the one with the most terms and the smallest
                        // tree with those number of terms
                        let pp_graph_size = pp_graph.num_nodes;
                        if pp_graph_size < best_pp_graph_size {
                            best_pp_term_weight = pp_term_weight;
                            best_pp_graph_size = pp_graph_size;
                            best_pp = Some((pp_i, pp_graph));
                            info_sched!("  New best graph for pp {}, term weight {}, size {}",
                                        pp.get_product_str(),
                                        pp_term_weight,
                                        best_pp_graph_size);
                            if !best_fit {
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((best_pp_idx, best_graph)) = best_pp {
                let pp = &to_schedule[best_pp_idx];
                let _node_list = best_graph.node_list();
                info_sched!("  * Scheduled product {} with {} nodes and {} edges",
                            pp,
                            best_graph.num_nodes,
                            best_graph.num_edges);
                //_node_list);
                // Update node statistics and mark as used
                for node_id in best_graph.iter_nodes() {
                    let node = self.topo.get_node(node_id);
                    self.stats.inc(node.node_type);
                    self.used[node.id] = true;
                }
                pp_paths.push((pp.clone(), best_graph));
                to_remove.push(best_pp_idx);
                if pp.gate_type.is_t() {
                    num_avail_magic -= 1;
                }
            }

            for pp_i in to_remove {
                remaining_to_schedule.shift_remove(&pp_i);
            }
        }
        let mut title =
            self.stats.update(step_i, pp_paths.len(), to_schedule.len(), num_avail_magic as usize);
        if next_to_schedule.len() > 0 {
            title += "\nUnscheduled: ";
        }
        for pp in &next_to_schedule {
            title += &format!("{} ", pp.get_product_str());
        }
        info_sched!("  Removed {} dependent nodes", _num_dependent_nodes);
        if !pp_paths.is_empty() {
            (Some(title), Some(pp_paths), next_to_schedule)
        } else {
            if num_avail_magic > 0 {
                // if any magic node is available and we scheduled nothing, then we must terminate
                // with an error, since we should be able to schedule something
                panic!("{}Step {}: Cannot schedule products [{}] on current layout ({} magic){}",
                       _RED,
                       step_i,
                       to_schedule.iter()
                                  .map(|pp| pp.get_product_str())
                                  .collect::<Vec<_>>()
                                  .join(", "),
                       num_avail_magic,
                       _RESET);
            }
            (None, None, next_to_schedule)
        }
    }

    fn setup_timestep(&mut self) -> usize {
        // Collect magic nodes that need new busy counts
        let num_used_magic_nodes =
            self.topo
                .iter_nodes()
                .filter(|node| self.used[node.id] && node.node_type == NodeType::Magic)
                .count();
        // Generate new cultivation times for magic nodes
        let new_cultivation_times: Vec<i32> =
            (0..num_used_magic_nodes).map(|_| self.gen_cultivation_time()).collect();
        // Update busy counts and reset used flags
        let mut num_avail_magic = 0;
        let mut cultivation_time_index = 0;
        for node in self.topo.iter_nodes_mut() {
            if !self.used[node.id] {
                if node.is_cultivating() {
                    node.busy_count += 1;
                    if node.busy_count == node.cultivation_time {
                        self.cultivation_times.push(node.cultivation_time);
                        node.cultivation_time = 0;
                        node.busy_count = 0;
                    }
                }
            } else {
                if node.node_type == NodeType::Magic {
                    node.cultivation_time = new_cultivation_times[cultivation_time_index];
                    node.busy_count = 0;
                    cultivation_time_index += 1;
                }
            }
            if node.node_type == NodeType::Magic && node.cultivation_time == 0 {
                num_avail_magic += 1;
            }
        }
        self.used.fill(false);
        info_sched!("  Available magic {}", num_avail_magic);
        num_avail_magic
    }

    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct, num_scheduled: usize)
                              -> Option<TreeGraph> {
        info_sched!("  Trying to schedule product {}", pauli_product);
        // Terminal nodes contain only the data qubits
        let terminals = self.get_terminal_nodes(pauli_product);
        if terminals.is_none() {
            info_sched!("  No data nodes found in working graph");
            return None;
        }
        let terminals = terminals.unwrap();
        // Handle single data node case
        if terminals.len() == 1 && !pauli_product.gate_type.is_t() {
            let node_id = terminals[0];
            let node = self.topo.get_node(node_id);
            if self.used[node.id] {
                info_sched!("  Single node {} is used", node.label);
                return None;
            }
            let mut g = TreeGraph::new(self.topo.num_nodes);
            g.add_node(node);
            info_sched!("  Can schedule product {} on {} nodes", pauli_product, g.num_nodes);
            return Some(g);
        }
        // first check that all terminals are accessible
        if terminals.iter().any(|node_id| self.used[*node_id]) {
            return None;
        }
        // Get root nodes next to terminals
        let root_ids = self.get_root_nodes(&terminals);
        if root_ids.is_empty() {
            return None;
        }
        self.timers.steiner_tree.start();
        let g = self.stree_computation.get_steiner_tree(&self.topo,
                                                        &self.used,
                                                        &root_ids,
                                                        &terminals,
                                                        pauli_product.gate_type,
                                                        num_scheduled);
        //let g = self.get_steiner_tree(&root_ids, &terminals, pauli_product.is_tgate);
        self.timers.steiner_tree.stop();
        if let Some(g) = g {
            info_sched!("  Can schedule product {} on {} nodes", pauli_product, g.num_nodes);
            return Some(g);
        }
        None
    }

    fn get_terminal_nodes(&self, pauli_product: &PauliProduct) -> Option<Vec<usize>> {
        // Initially terminal nodes contain only the data qubits
        let mut terminals = Vec::new();
        for op in &pauli_product.operators {
            if op.basis == 'Y' {
                for term in ['X', 'Z'] {
                    let node_label = format!("d{}{}", op.qubit, term);
                    let node = self.topo.get_node_from_label(&node_label);
                    // Check if node is already used
                    if self.used[node.id] {
                        info_sched!("  Node {} is already used", node_label);
                        return None;
                    }
                    // check for at least one unused magic or bus nb
                    if !node.nbors.iter().any(|nb_id| {
                                             let nb = self.topo.get_node(*nb_id);
                                             !self.used[nb.id]
                                         })
                    {
                        info_sched!("  No unused neighbors for node {}", node.id);
                        return None;
                    }
                    terminals.push(node.id);
                }
            } else {
                let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                let node = self.topo.get_node_from_label(&node_label);
                // Check if node is already used
                if self.used[node.id] {
                    info_sched!("  Node {} is already used", node_label);
                    return None;
                }
                // check for at least one unused magic or bus nb
                if !node.nbors.iter().any(|nb_id| {
                                         let nb = self.topo.get_node(*nb_id);
                                         !self.used[nb.id]
                                     })
                {
                    info_sched!("  No unused neighbors for node {}", node.id);
                    return None;
                }
                terminals.push(node.id);
            }
        }
        Some(terminals)
    }

    fn get_root_nodes(&self, terminals: &[usize]) -> Vec<usize> {
        let mut root_ids = IndexSet::new();
        // need to get a root node for every terminal
        let mut terminals_matched: IndexSet<usize> = terminals.iter().copied().collect();
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
                        root_ids.insert(nb_id.clone());
                        terminals_matched.swap_remove(node_id);
                        terminals_matched.swap_remove(&paired_node.id);
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
                        root_ids.insert(nb_id.clone());
                        terminals_matched.swap_remove(node_id);
                        break;
                    }
                }
            }
        }
        if !terminals_matched.is_empty() {
            debug_sched!("    could not find root nodes for terminals: {:?}",
                         terminals_matched.iter()
                                          .map(|id| &self.topo.get_node(*id).label)
                                          .collect::<Vec<_>>(),);
            return Vec::new();
        }
        root_ids.into_iter().collect()
    }

    fn gen_cultivation_time(&mut self) -> i32 {
        let cultivation_time = self.rng_exp.sample().round() as i32; // + 1;
        cultivation_time
    }

    fn check_dependencies(&self, pp: &PauliProduct, scheduled: &IndexSet<i32>) -> io::Result<()> {
        if scheduled.contains(&pp.id) {
            return Err(io::Error::new(io::ErrorKind::Other,
                                      format!("pp {} already scheduled", pp.id)));
        }

        for &parent_id in &pp.parents {
            if !scheduled.contains(&parent_id) {
                return Err(io::Error::new(io::ErrorKind::Other,
                                          format!("pp {} scheduled before parent {}",
                                                  pp.id, parent_id)));
            }
        }

        Ok(())
    }

    fn print_schedule(&self) -> io::Result<()> {
        let _timer = fn_timer!();
        let circuit_stem = Path::new(&self.circuit.circuit_fname).file_stem()
                                                                 .and_then(|s| s.to_str())
                                                                 .unwrap_or("circuit");
        let output_fname = format!("{}.schedule", circuit_stem);

        let file = File::create(&output_fname)?;
        let mut buf_file = BufWriter::new(file);

        writeln!(buf_file, "# Scheduled Products by Timestep")?;
        writeln!(buf_file, "# Circuit: {}", self.circuit.circuit_fname)?;
        writeln!(buf_file, "# Total steps: {}", self.scheduled_products.len())?;
        writeln!(buf_file,
                 "# Total products: {}",
                 self.scheduled_products.iter().map(|v| v.len()).sum::<usize>())?;
        writeln!(buf_file)?;

        let colors = [_GREEN, _RED, _YELLOW, _BLUE, _MAGENTA, _CYAN, _WHITE, _LGREEN, _LRED,
                      _LYELLOW, _LBLUE, _LMAGENTA, _LCYAN, _LWHITE];

        for step_products in &self.scheduled_products {
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
            }
            for i in 0..self.circuit.num_qubits {
                write!(buf_file, "{}{}", combined_colors[i], combined_chars[i])?;
            }
            let mut id_string = String::new();
            for (idx, pp) in sorted_products.iter().enumerate() {
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(" {}{}", color, pp.id));
            }
            writeln!(buf_file, "{}{}", id_string, _RESET)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }
}
