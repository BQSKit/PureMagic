use crate::circuit::Circuit;
use crate::node::NodeType;
use crate::pauliproduct::PauliProduct;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use crate::utils::{
    BLUE, CYAN, GREEN, LBLUE, LCYAN, LGREEN, LMAGENTA, LRED, LWHITE, LYELLOW, MAGENTA, RED, RESET,
    WHITE, YELLOW,
};
use crate::utils::{IntermittentTimer, Timer};

use indexmap::{IndexMap, IndexSet};
#[cfg(debug_assertions)]
use log::{debug, info};
use rand_simple::Exponential;
use simple_logging;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::Path;

macro_rules! debug_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        debug!($($arg)*);
    };
}

macro_rules! info_sched {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        info!($($arg)*);
    };
}

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
                        magic_scheduled: 0 }
    }

    pub fn summarize(&self, num_steps: usize) {
        // Calculate statistics
        let data_frac = self.sum_data_scheduled as f64 / (self.data_qubits * num_steps) as f64;
        let bus_frac = self.sum_bus_scheduled as f64 / (self.bus_qubits * num_steps) as f64;
        let magic_frac = self.sum_magic_scheduled as f64 / (self.magic_qubits * num_steps) as f64;
        // Print final statistics
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
    }

    pub fn update(&mut self, step_i: usize, pp_paths_len: usize, to_schedule_len: usize) -> String {
        self.sum_data_scheduled += self.data_scheduled;
        self.sum_bus_scheduled += self.bus_scheduled;
        self.sum_magic_scheduled += self.magic_scheduled;

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

pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng_exp: Exponential,
    magic_state_lambda: f64,
    plot_option: String,
    cultivation_times: Vec<i32>,
    schedule_product_timer: IntermittentTimer,
    steiner_tree_timer: IntermittentTimer,
    timestep_timer: IntermittentTimer,
    stats: ScheduleStats,
    scheduled_products: Vec<Vec<PauliProduct>>,
    used: Vec<bool>,
}

impl Scheduler {
    pub fn new(circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
               plot_option: String, rseed: u32)
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
                    schedule_product_timer: IntermittentTimer::new("schedule_pauli_product", ""),
                    steiner_tree_timer: IntermittentTimer::new("get_steiner_tree", ""),
                    timestep_timer: IntermittentTimer::new("schedule_timestep", ""),
                    stats: ScheduleStats::new(num_data_qubits, num_bus_qubits, num_magic_qubits),
                    scheduled_products: Vec::new(),
                    used: vec![false; num_nodes] }
    }

    pub fn schedule_circuit(&mut self, best_fit: bool) -> io::Result<(usize, usize)> {
        let _timer = Timer::new("schedule_circuit");
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
            self.timestep_timer.start();
            num_steps += 1;
            info_sched!("{}Step {}: {:?}{}",
                        CYAN,
                        num_steps,
                        to_schedule.iter()
                                   .map(|pp| format!("{}:{}", pp.id, pp.get_product_str()))
                                   .collect::<Vec<_>>(),
                        RESET);

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
                                                      RED, RESET)));
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
            self.timestep_timer.stop();
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

        self.steiner_tree_timer.done();
        self.schedule_product_timer.done();
        self.timestep_timer.done();

        self.print_schedule()?;
        Ok((num_steps, scheduled.len()))
    }

    fn setup_timestep(&mut self) -> i32 {
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
                let pp_graph = if num_avail_magic == 0 && pp.is_tgate {
                    None
                } else {
                    self.schedule_product_timer.start();
                    let pp_graph = self.schedule_pauli_product(pp);
                    self.schedule_product_timer.stop();
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
                info_sched!("  * Scheduled product {} with {} nodes and {} edges: {:?}",
                            pp,
                            best_graph.num_nodes,
                            best_graph.num_edges,
                            _node_list);
                // Update node statistics and mark as used
                for node in best_graph.iter_nodes() {
                    self.stats.inc(node.node_type);
                    self.used[node.id] = true;
                }
                pp_paths.push((pp.clone(), best_graph));
                to_remove.push(best_pp_idx);
                num_avail_magic -= 1;
            }

            for pp_i in to_remove {
                remaining_to_schedule.shift_remove(&pp_i);
            }
        }
        let mut title = self.stats.update(step_i, pp_paths.len(), to_schedule.len());
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
                       RED,
                       step_i,
                       to_schedule.iter()
                                  .map(|pp| pp.get_product_str())
                                  .collect::<Vec<_>>()
                                  .join(", "),
                       num_avail_magic,
                       RESET);
            }
            (None, None, next_to_schedule)
        }
    }

    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct) -> Option<TreeGraph> {
        info_sched!("  Trying to schedule product {}", pauli_product);
        // Terminal nodes contain only the data qubits
        let terminals = self.get_terminal_nodes(pauli_product);
        if terminals.is_none() {
            info_sched!("  No data nodes found in working graph");
            return None;
        }
        let terminals = terminals.unwrap();
        // Handle single data node case
        if terminals.len() == 1 && !pauli_product.is_tgate {
            let node_id = terminals[0];
            let node = self.topo.get_node(node_id);
            if self.used[node.id] {
                info_sched!("  Single node {} is used", node.label);
                return None;
            }
            let mut g = TreeGraph::new();
            g.add_node(node.clone());
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
        self.steiner_tree_timer.start();
        let g = self.get_steiner_tree(&root_ids, &terminals, pauli_product.is_tgate);
        self.steiner_tree_timer.stop();
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
        for node_id in terminals.iter() {
            let node = self.topo.get_node(*node_id);
            let paired_node = self.topo.get_node(node.paired_data_id.unwrap());
            let mut pair_found = false;
            // first look for paired nodes (top/bottom)
            if terminals.contains(&paired_node.id) {
                let pair = if node.label.contains("X") { Some("XX") } else { Some("ZZ") };
                debug_sched!("    Found {} pair {}{} in terminals",
                             pair.unwrap(),
                             node.id,
                             paired_node.id);
                for nb_id in node.nbors.iter() {
                    let nb = self.topo.get_node(*nb_id);
                    if self.used[nb.id] || !nb.is_routing() {
                        continue;
                    }
                    // If we are using top/bottom
                    if (pair == Some("XX") && nb.pos.1 < node.pos.1)
                       || (pair == Some("ZZ") && nb.pos.1 > node.pos.1)
                    {
                        //debug_sched!("    {}Found XX pair {} -> {}{}", BLUE, node_label, nb_label, RESET);
                        root_ids.insert(nb_id.clone());
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
                        break;
                    }
                }
            }
        }
        root_ids.into_iter().collect()
    }

    // this can be viewed as a greedy multi-source shortest path algorithm
    fn get_steiner_tree(&mut self, root_ids: &Vec<usize>, terminal_nodes: &Vec<usize>,
                        is_tgate: bool)
                        -> Option<TreeGraph> {
        debug_sched!("    BFS from nodes {:?} to nodes {:?}", root_ids, terminal_nodes);
        let mut visited: IndexMap<usize, usize> = IndexMap::with_capacity(self.topo.num_nodes);
        let mut paths: IndexMap<usize, IndexSet<usize>> = IndexMap::with_capacity(root_ids.len());
        let mut queue = VecDeque::with_capacity(self.topo.num_nodes);
        let mut tree = TreeGraph::with_capacity(terminal_nodes.len() * 5);
        let mut cultivator = None;
        let mut total_paths = 0;
        debug_sched!("    Number of root labels {}", root_ids.len());
        // every root must have a path to every other root
        let reqd_paths = root_ids.len() * (root_ids.len() - 1);
        debug_sched!("    Require {} paths", reqd_paths);

        for root_id in root_ids {
            debug_sched!("      {}root node {}{}", GREEN, root_id, RESET);
            paths.insert(root_id.clone(), IndexSet::new());
            visited.insert(root_id.clone(), root_id.clone());
            queue.push_back(root_id);
            let root = self.topo.get_node(*root_id);
            tree.add_node(root.clone());
            if cultivator.is_none()
               && root.node_type == NodeType::Magic
               && root.cultivation_time == 0
            {
                cultivator = Some(root_id);
                debug_sched!("      {}found root cultivator {}{}",
                             GREEN,
                             cultivator.unwrap(),
                             RESET);
            }
            // add terminals
            let root_node = self.topo.get_node(*root_id);
            for nb_id in root_node.nbors.iter() {
                let nb = self.topo.get_node(*nb_id);
                if terminal_nodes.contains(&nb_id) {
                    tree.add_node(nb.clone());
                    tree.add_edge(*root_id, *nb_id);
                    debug_sched!("      {}add node {}{}", GREEN, nb_id, RESET);
                    debug_sched!("      {}add edge {}->{}{}", GREEN, root_id, nb_id, RESET);
                }
            }
        }
        while let Some(node_id) = queue.pop_front() {
            let node = self.topo.get_node(*node_id);
            let curr_root_id = visited.get(node_id).unwrap().clone();
            for nb_id in node.nbors.iter() {
                let nb = self.topo.get_node(*nb_id);
                if self.used[nb.id] {
                    continue;
                }
                if nb.node_type == NodeType::Data {
                    // all data nodes are already linked in
                    continue;
                }
                // check for path links between roots via routing nodes
                if nb.is_routing() && node.is_routing() && visited.contains_key(nb_id) {
                    let nb_root_id = visited.get(nb_id).unwrap().clone();
                    if curr_root_id == nb_root_id {
                        continue;
                    }
                    let curr_root_paths = paths.get(&curr_root_id).unwrap();
                    if !curr_root_paths.contains(&nb_root_id) {
                        // update the nb root IndexSet to contain paths to all the roots in
                        // the curr_root IndexSet
                        let nb_root_paths = paths.get(&nb_root_id).unwrap().clone();
                        // Create merged set containing all roots from both groups
                        let mut merged_set = curr_root_paths.clone();
                        merged_set.insert(nb_root_id.clone());
                        merged_set.extend(nb_root_paths.iter().cloned());
                        merged_set.insert(curr_root_id.clone());
                        // Update all roots in the merged set to have the complete merged set
                        for root_id in merged_set.iter() {
                            let mut full_set = merged_set.clone();
                            full_set.swap_remove(root_id); // Don't include self
                            paths.insert(root_id.clone(), full_set);
                        }
                        // Recalculate total_paths
                        total_paths = paths.values().map(|set| set.len()).sum::<usize>();
                        debug_sched!("      {}path from {} to {} (total paths {}/{}){}",
                                     GREEN,
                                     curr_root_id,
                                     nb_root_id,
                                     total_paths,
                                     reqd_paths,
                                     RESET);
                        debug_sched!("      {}paths:{:?}{}", GREEN, paths, RESET);
                        tree.add_edge(*node_id, *nb_id);
                        debug_sched!("      {}add edge {}->{}{}", GREEN, node_id, nb_id, RESET);
                        if total_paths == reqd_paths {
                            if is_tgate && cultivator.is_none() {
                                continue;
                            }
                            // we break here because we previously found a cultivator, and now have
                            // found all the paths
                            break;
                        }
                    }
                    continue;
                }
                let nb_is_cultivator = is_tgate
                                       && cultivator == None
                                       && nb.node_type == NodeType::Magic
                                       && nb.cultivation_time == 0;
                // add routing node/cultivator
                if nb.is_routing() || nb_is_cultivator {
                    tree.add_node(nb.clone());
                    tree.add_edge(*node_id, *nb_id);
                    debug_sched!("      {}add node {}{}", GREEN, nb_id, RESET);
                    debug_sched!("      {}add edge {}->{}{}", GREEN, node_id, nb_id, RESET);
                    queue.push_back(nb_id);
                    if cultivator.is_none() && nb_is_cultivator {
                        cultivator = Some(nb_id);
                        debug_sched!("      {}found clutivator {}{}",
                                     GREEN,
                                     cultivator.unwrap(),
                                     RESET);
                        if total_paths == reqd_paths {
                            // we break here because we previously found all the paths, and now have
                            // found a cultivator
                            break;
                        }
                    }
                }
                visited.insert(nb_id.clone(), curr_root_id.clone());
            }
            if total_paths == reqd_paths {
                if is_tgate && cultivator.is_none() {
                    continue;
                }
                // we have all the paths and terms and a cultivator (if needed), so we can now
                // return the tree (bfs_graph)
                tree.root_node = if is_tgate {
                    debug_sched!("      {}tree complete, cultivator {}{}",
                                 GREEN,
                                 cultivator.unwrap(),
                                 RESET);
                    Some(*cultivator.unwrap())
                } else {
                    debug_sched!("      {}tree complete{}", GREEN, RESET);
                    Some(root_ids[0])
                };
                let _num_trimmed = tree.trim_dangling_nodes();
                debug_sched!("    Trimmed {} dangling nodes", _num_trimmed);
                // FIXME: for XX and ZZ, replace side edges with top/bottom, if that
                // makes the path shorter
                return Some(tree);
            }
        }
        None
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
        let circuit_stem = Path::new(&self.circuit.circuit_fname).file_stem()
                                                                 .and_then(|s| s.to_str())
                                                                 .unwrap_or("circuit");
        let output_fname = format!("{}.schedule", circuit_stem);

        let mut file = std::fs::File::create(&output_fname)?;

        writeln!(file, "# Scheduled Products by Timestep")?;
        writeln!(file, "# Circuit: {}", self.circuit.circuit_fname)?;
        writeln!(file, "# Total steps: {}", self.scheduled_products.len())?;
        writeln!(file,
                 "# Total products: {}",
                 self.scheduled_products.iter().map(|v| v.len()).sum::<usize>())?;
        writeln!(file)?;

        let colors = [GREEN, RED, YELLOW, BLUE, MAGENTA, CYAN, WHITE, LGREEN, LRED, LYELLOW,
                      LBLUE, LMAGENTA, LCYAN, LWHITE];

        for step_products in &self.scheduled_products {
            let mut sorted_products = step_products.clone();
            sorted_products.sort_by_key(|pp| {
                               pp.operators.iter().map(|op| op.qubit).min().unwrap_or(usize::MAX)
                           });
            let mut combined_chars = vec!['_'; self.circuit.num_qubits];
            let mut combined_colors = vec![RESET; self.circuit.num_qubits];
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
                write!(file, "{}{}", combined_colors[i], combined_chars[i])?;
            }
            let mut id_string = String::new();
            for (idx, pp) in sorted_products.iter().enumerate() {
                let color = colors[idx % colors.len()];
                id_string.push_str(&format!(" {}{}", color, pp.id));
            }
            writeln!(file, "{}{}", id_string, RESET)?;
        }
        println!("Scheduled products written to {}", output_fname);
        Ok(())
    }
}
