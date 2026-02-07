use crate::circuit::Circuit;
use crate::pauliproduct::PauliProduct;
use crate::topograph::{NodeType, TopoGraph};
use crate::utils::{CYAN, GREEN, IntermittentTimer, RED, RESET, Timer};

use indexmap::{IndexMap, IndexSet};
use log::{debug, info};
use rand_simple::Exponential;
use simple_logging;
use std::collections::VecDeque;
use std::io::{self, Write};
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

        info!("Scheduling results:");
        let frac_paths = pp_paths_len as f64 / to_schedule_len as f64;
        let frac_data = self.data_scheduled as f64 / self.data_qubits as f64;
        let frac_bus = self.bus_scheduled as f64 / self.bus_qubits as f64;
        let frac_magic = self.magic_scheduled as f64 / self.magic_qubits as f64;
        let tot_qubits = self.magic_scheduled + self.bus_scheduled + self.data_scheduled;
        info!("  products:    {}/{} ({:.2})", pp_paths_len, to_schedule_len, frac_paths);
        info!("  data:        {}/{} ({:.2})", self.data_scheduled, self.data_qubits, frac_data);
        info!("  bus:         {}/{} ({:.2})", self.bus_scheduled, self.bus_qubits, frac_bus);
        info!("  magic:       {}/{} ({:.2})", self.magic_scheduled, self.magic_qubits, frac_magic);
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
    stats: ScheduleStats,
}

impl Scheduler {
    pub fn new(circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_level: &str,
               plot_option: String, rseed: u32)
               -> Self {
        if log_level != "none" {
            let circuit_stem = Path::new(&circuit.circuit_fname).file_stem()
                                                                .and_then(|s| s.to_str())
                                                                .unwrap_or("circuit");
            let sched_fname = format!("{}.sched", circuit_stem);
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

        Scheduler { circuit,
                    topo,
                    rng_exp: Exponential::new(rseed),
                    magic_state_lambda,
                    plot_option,
                    cultivation_times: Vec::new(),
                    schedule_product_timer: IntermittentTimer::new("sched product", ""),
                    stats: ScheduleStats::new(num_data_qubits, num_bus_qubits, num_magic_qubits) }
    }

    pub fn schedule_circuit(&mut self, best_fit: bool) -> io::Result<(usize, usize)> {
        let _timer = Timer::new("schedule_circuit");
        self.rng_exp
            .try_set_params(1.0 / self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        // Initialize magic nodes with busy counts
        // Collect magic node labels first to avoid borrow conflicts
        let magic_labels: Vec<String> = self.topo
                                            .iter_nodes()
                                            .filter(|node| node.node_type == NodeType::Magic)
                                            .map(|node| node.label.clone())
                                            .collect();
        for label in magic_labels {
            self.topo.get_node_mut(&label).cultivation_time = self.gen_cultivation_time();
            self.topo.get_node_mut(&label).busy_count = 0;
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
        let mut sum_iteration_time = 0;
        let mut max_iteration_time = 0;
        // Main scheduling loop
        while !to_schedule.is_empty() {
            let iteration_start_time = std::time::Instant::now();
            num_steps += 1;
            info!("{}Step {}: {:?}{}",
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
                let mut children_to_schedule = IndexSet::new();
                for (pp, _) in pp_paths {
                    // Add children to next round if all parents scheduled
                    for &child_id in &pp.children {
                        remaining_parents[child_id as usize] -= 1;
                        if remaining_parents[child_id as usize] == 0 {
                            children_to_schedule.insert(child_id);
                        }
                    }
                }
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
            let iteration_time = iteration_start_time.elapsed().as_micros();
            sum_iteration_time += iteration_time;
            if iteration_time > max_iteration_time {
                max_iteration_time = iteration_time;
            }
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
        println!("Scheduling iteration time per step {:} μs avg, {:} μs max",
                 sum_iteration_time / num_steps as u128,
                 max_iteration_time);
        self.schedule_product_timer.done();

        Ok((num_steps, scheduled.len()))
    }

    fn schedule_timestep(
        &mut self, step_i: usize, to_schedule: &[PauliProduct], best_fit: bool)
        -> (Option<String>, Option<Vec<(PauliProduct, TopoGraph)>>, Vec<PauliProduct>) {
        // Update busy counts and reset used flags
        // First, collect magic nodes that need new busy counts
        let num_used_magic_nodes =
            self.topo
                .iter_nodes()
                .filter(|node| node.used && node.node_type == NodeType::Magic)
                .count();
        // Generate new cultivation times for magic nodes
        let new_cultivation_times: Vec<i32> =
            (0..num_used_magic_nodes).map(|_| self.gen_cultivation_time()).collect();
        // Now update all nodes
        let mut num_avail_magic = 0;
        let mut cultivation_time_index = 0;
        for node in self.topo.iter_nodes_mut() {
            if !node.used && node.is_cultivating() {
                node.busy_count += 1;
                if node.busy_count == node.cultivation_time {
                    self.cultivation_times.push(node.cultivation_time);
                    node.cultivation_time = 0;
                    node.busy_count = 0;
                }
            }
            if node.used && node.node_type == NodeType::Magic {
                node.cultivation_time = new_cultivation_times[cultivation_time_index];
                node.busy_count = 0;
                cultivation_time_index += 1;
            }
            node.used = false;
            if node.node_type == NodeType::Magic && node.cultivation_time == 0 {
                num_avail_magic += 1;
            }
        }
        info!("  Available magic {}", num_avail_magic);

        let mut pp_paths = Vec::new();
        let mut next_to_schedule = Vec::new();
        let mut num_dependent_nodes = 0;

        let mut remaining_to_schedule: IndexSet<usize> = (0..to_schedule.len()).collect();
        // Presort products from most to least resource-intensive
        remaining_to_schedule.sort_by_key(|&idx| {
                                 std::cmp::Reverse(to_schedule[idx].count_weighted_terms())
                             });
        info!("  Remaining to schedule: {}", remaining_to_schedule.len());
        while !remaining_to_schedule.is_empty() {
            let mut to_remove = Vec::new();
            let mut best_pp: Option<(usize, TopoGraph)> = None;
            let mut best_pp_graph_size = usize::MAX;
            let mut best_pp_term_weight = 0;

            for &pp_i in &remaining_to_schedule {
                let pp = &to_schedule[pp_i];
                let pp_term_weight = pp.count_weighted_terms();
                if pp_term_weight < best_pp_term_weight {
                    info!("  Skip lower weight product {}", pp);
                    continue;
                }
                self.schedule_product_timer.start();
                let pp_graph = if num_avail_magic == 0 && pp.is_tgate {
                    None
                } else {
                    self.schedule_pauli_product(pp)
                };
                self.schedule_product_timer.stop();
                if pp_graph.is_none() {
                    info!("  Could not schedule {} on graph", pp.id);
                    next_to_schedule.push(pp.clone());
                    // Mark dependent nodes as used
                    for op in &pp.operators {
                        if op.basis == 'Y' {
                            let node_label_x = format!("d{}{}", op.qubit, 'X');
                            self.topo.get_node_mut(&node_label_x).used = true;
                            let node_label_z = format!("d{}{}", op.qubit, 'Z');
                            self.topo.get_node_mut(&node_label_z).used = true;
                            num_dependent_nodes += 2;
                        } else {
                            let node_label =
                                format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                            self.topo.get_node_mut(&node_label).used = true;
                            num_dependent_nodes += 1;
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
                            info!("  New best graph for pp {}, term weight {}, size {}",
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
                info!("  * Scheduled product {} with {} nodes and {} edges: {:?}",
                      pp,
                      best_graph.num_nodes,
                      best_graph.num_edges,
                      best_graph.node_list());
                // Update node statistics and mark as used
                for node in best_graph.iter_nodes() {
                    self.stats.inc(node.node_type);
                    self.topo.get_node_mut(&node.label).used = true;
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
        info!("  Removed {} dependent nodes", num_dependent_nodes);

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

    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct) -> Option<TopoGraph> {
        info!("  Trying to schedule product {}", pauli_product);
        // Terminal nodes contain only the data qubits
        let terminals = self.get_terminal_nodes(pauli_product);
        if terminals.is_none() {
            info!("  No data nodes found in working graph");
            return None;
        }
        let terminals = terminals.unwrap();
        // Handle single data node case
        if terminals.len() == 1 && !pauli_product.is_tgate {
            let node_label = &terminals[0];
            let node = self.topo.get_node(node_label);
            if node.used {
                info!("  Single node {} is used", node_label);
                return None;
            }
            let mut g = TopoGraph::new();
            g.add_node(node.clone());
            info!("  Can schedule product {} on {} nodes", pauli_product, g.num_nodes);
            return Some(g);
        }
        // first check that all terminals are accessible
        for node_label in terminals.iter() {
            let node = self.topo.get_node(node_label);
            if node.used {
                return None;
            }
        }
        // Get root nodes next to terminals
        let root_labels = self.get_root_nodes(&terminals);
        if root_labels.is_empty() {
            return None;
        }
        let g = self.get_steiner_tree(&root_labels, &terminals, pauli_product.is_tgate);
        if let Some(g) = g {
            info!("  Can schedule product {} on {} nodes", pauli_product, g.num_nodes);
            return Some(g);
        }
        None
    }

    fn get_terminal_nodes(&self, pauli_product: &PauliProduct) -> Option<Vec<String>> {
        // Initially terminal nodes contain only the data qubits
        let mut terminals = Vec::new();
        for op in &pauli_product.operators {
            if op.basis == 'Y' {
                for term in ['X', 'Z'] {
                    let node_label = format!("d{}{}", op.qubit, term);
                    let node = self.topo.get_node(&node_label);
                    // Check if node is already used
                    if node.used {
                        info!("  Node {} is already used", node_label);
                        return None;
                    }
                    // check for at least one unused magic or bus nb
                    if !node.edges.iter().any(|nb_label| {
                                             let nb = self.topo.get_node(nb_label);
                                             !nb.used
                                         })
                    {
                        info!("  No unused neighbors for node {}", node.label);
                        return None;
                    }
                    terminals.push(node_label);
                }
            } else {
                let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                let node = self.topo.get_node(&node_label);
                // Check if node is already used
                if node.used {
                    info!("  Node {} is already used", node_label);
                    return None;
                }
                // check for at least one unused magic or bus nb
                if !node.edges.iter().any(|nb_label| {
                                         let nb = self.topo.get_node(nb_label);
                                         !nb.used
                                     })
                {
                    info!("  No unused neighbors for node {}", node.label);
                    return None;
                }
                terminals.push(node_label);
            }
        }
        Some(terminals)
    }

    fn get_root_nodes(&self, terminals: &[String]) -> Vec<String> {
        let mut root_labels = IndexSet::new();
        for node_label in terminals.iter() {
            let node = self.topo.get_node(node_label);
            let paired_node = self.topo.get_paired_data_node(node);
            let mut pair_found = false;
            // first look for paired nodes (top/bottom)
            if terminals.contains(&paired_node.label) {
                let pair = if node.label.contains("X") { Some("XX") } else { Some("ZZ") };
                debug!("    Found {} pair {}{} in terminals",
                       pair.unwrap(),
                       node.label,
                       paired_node.label);
                for nb_label in node.edges.iter() {
                    let nb = self.topo.get_node(nb_label);
                    if nb.used || !self.topo.is_routing_node(nb) {
                        continue;
                    }
                    // If we are using top/bottom
                    if (pair == Some("XX") && nb.pos.1 < node.pos.1)
                       || (pair == Some("ZZ") && nb.pos.1 > node.pos.1)
                    {
                        //debug!("    {}Found XX pair {} -> {}{}", BLUE, node_label, nb_label, RESET);
                        root_labels.insert(nb_label.clone());
                        pair_found = true;
                        break;
                    }
                }
            };
            if !pair_found {
                for nb_label in node.edges.iter() {
                    let nb = self.topo.get_node(nb_label);
                    if nb.used || !self.topo.is_routing_node(nb) {
                        continue;
                    }
                    // Only include neighbors on the side (same row, different column)
                    if nb.pos.0 != node.pos.0 && nb.pos.1 == node.pos.1 {
                        root_labels.insert(nb_label.clone());
                        break;
                    }
                }
            }
        }
        root_labels.into_iter().collect()
    }

    // this can be viewed as a greedy multi-source shortest path algorithm
    fn get_steiner_tree(&self, root_labels: &Vec<String>, terminal_nodes: &Vec<String>,
                        is_tgate: bool)
                        -> Option<TopoGraph> {
        debug!("    BFS from nodes {:?} to nodes {:?}", root_labels, terminal_nodes);
        let mut visited: IndexMap<String, String> = IndexMap::with_capacity(self.topo.num_nodes);
        let mut paths: IndexMap<String, IndexSet<String>> =
            IndexMap::with_capacity(root_labels.len());
        let mut queue = VecDeque::with_capacity(self.topo.num_nodes);
        let mut tree = TopoGraph::new();
        let mut cultivator = None;
        let mut total_paths = 0;
        debug!("    Number of root labels {}", root_labels.len());
        // every root must have a path to every other root
        let reqd_paths = root_labels.len() * (root_labels.len() - 1);
        debug!("    Require {} paths", reqd_paths);

        for root_label in root_labels {
            debug!("      {}root node {}{}", GREEN, root_label, RESET);
            paths.insert(root_label.clone(), IndexSet::new());
            visited.insert(root_label.clone(), root_label.clone());
            queue.push_back(root_label);
            let root = self.topo.get_node(root_label);
            tree.add_node(root.clone());
            if cultivator.is_none()
               && root.node_type == NodeType::Magic
               && root.cultivation_time == 0
            {
                cultivator = Some(root_label);
                debug!("      {}found root cultivator {}{}", GREEN, cultivator.unwrap(), RESET);
            }
            // add terminals
            let root_node = self.topo.get_node(&root_label);
            for nb_label in root_node.edges.iter() {
                let nb = self.topo.get_node(&nb_label);
                if terminal_nodes.contains(&nb_label) {
                    tree.add_node(nb.clone());
                    tree.add_edge(&root_label, &nb_label);
                    debug!("      {}add node {}{}", GREEN, nb_label, RESET);
                    debug!("      {}add edge {}->{}{}", GREEN, root_label, nb_label, RESET);
                }
            }
        }
        while let Some(node_label) = queue.pop_front() {
            let node = self.topo.get_node(&node_label);
            let curr_root_label = visited.get(node_label).unwrap().clone();
            for nb_label in node.edges.iter() {
                let nb = self.topo.get_node(&nb_label);
                if nb.used {
                    continue;
                }
                if nb.node_type == NodeType::Data {
                    // all data nodes are already linked in
                    continue;
                }
                // check for path links between roots via routing nodes
                let routing_edge = self.topo.is_routing_node(nb) && self.topo.is_routing_node(node);
                if routing_edge && visited.contains_key(nb_label) {
                    let nb_root_label = visited.get(nb_label).unwrap().clone();
                    if curr_root_label == nb_root_label {
                        continue;
                    }
                    let curr_root_paths = paths.get(&curr_root_label).unwrap().clone();
                    if !curr_root_paths.contains(&nb_root_label) {
                        // update the nb root IndexSet to contain paths to all the roots in
                        // the curr_root IndexSet
                        let nb_root_paths = paths.get(&nb_root_label).unwrap().clone();
                        // Create merged set containing all roots from both groups
                        let mut merged_set = curr_root_paths;
                        merged_set.insert(nb_root_label.clone());
                        merged_set.extend(nb_root_paths.iter().cloned());
                        merged_set.insert(curr_root_label.clone());
                        // Update all roots in the merged set to have the complete merged set
                        for root_label in merged_set.iter() {
                            let mut full_set = merged_set.clone();
                            full_set.swap_remove(root_label); // Don't include self
                            paths.insert(root_label.clone(), full_set);
                        }
                        // Recalculate total_paths
                        total_paths = paths.values().map(|set| set.len()).sum::<usize>();
                        debug!("      {}path from {} to {} (total paths {}/{}){}",
                               GREEN,
                               curr_root_label,
                               nb_root_label,
                               total_paths,
                               reqd_paths,
                               RESET);
                        debug!("      {}paths:{:?}{}", GREEN, paths, RESET);
                        tree.add_edge(&node_label, &nb_label);
                        debug!("      {}add edge {}->{}{}", GREEN, node_label, nb_label, RESET);
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
                if self.topo.is_routing_node(nb) || nb_is_cultivator {
                    tree.add_node(nb.clone());
                    tree.add_edge(&node_label, &nb_label);
                    debug!("      {}add node {}{}", GREEN, nb_label, RESET);
                    debug!("      {}add edge {}->{}{}", GREEN, node_label, nb_label, RESET);
                    queue.push_back(nb_label);
                    if cultivator.is_none() && nb_is_cultivator {
                        cultivator = Some(nb_label);
                        debug!("      {}found clutivator {}{}", GREEN, cultivator.unwrap(), RESET);
                        if total_paths == reqd_paths {
                            // we break here because we previously found all the paths, and now have
                            // found a cultivator
                            break;
                        }
                    }
                }
                visited.insert(nb_label.clone(), curr_root_label.clone());
            }
            if total_paths == reqd_paths {
                if is_tgate && cultivator.is_none() {
                    continue;
                }
                // we have all the paths and terms and a cultivator (if needed), so we can now
                // return the tree (bfs_graph)
                tree.root_node = if is_tgate {
                    Some(cultivator.unwrap().to_string())
                } else {
                    Some(root_labels[0].to_string())
                };
                debug!("      {}tree complete, cultivator {}{}",
                       GREEN,
                       cultivator.map_or("none", |v| v),
                       RESET);
                let root = tree.root_node.as_ref().unwrap().clone();
                let num_trimmed = tree.trim_dangling_nodes(&root);
                debug!("    Trimmed {} dangling nodes", num_trimmed);
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
}

/*
/// Information about a node's nearest terminal in the Voronoi diagram
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// The nearest terminal node label
    pub nearest_terminal: String,
    /// Distance (number of hops) to the nearest terminal
    pub distance: usize,
    /// Parent node in the shortest path to the nearest terminal
    /// None if this is a terminal node itself
    pub parent: Option<String>,
}

impl Scheduler {
    /// Build a Voronoi diagram using multi-source BFS from all terminals
    /// Returns a map from each node label to its nearest terminal and distance
    pub fn build_voronoi_diagram(&self, terminals: &[String]) -> IndexMap<String, VoronoiCell> {
        let mut voronoi: IndexMap<String, VoronoiCell> = IndexMap::new();
        let mut queue: VecDeque<(String, String, usize)> = VecDeque::new();
        let mut visited: IndexSet<String> = IndexSet::new();

        info!("Building Voronoi diagram for {} terminals", terminals.len());

        // Initialize BFS from all terminals simultaneously
        for terminal in terminals {
            let node = self.topo.get_node(terminal);
            assert!(!node.used);
            // Each terminal starts with distance 0 to itself and no parent
            queue.push_back((terminal.clone(), terminal.clone(), 0));
            visited.insert(terminal.clone());
            voronoi.insert(terminal.clone(), VoronoiCell {
                nearest_terminal: terminal.clone(),
                distance: 0,
                parent: None, // Terminals have no parent
            });
        }
        info!("  Starting multi-source BFS from {} valid terminals", voronoi.len());

        // Multi-source BFS to compute Voronoi regions
        let mut total_nodes_visited = 0;
        while let Some((node_label, terminal_label, dist)) = queue.pop_front() {
            let node = self.topo.get_node(&node_label);

            for neighbor_label in &node.edges {
                let neighbor = self.topo.get_node(neighbor_label);

                // Skip used nodes
                if neighbor.used {
                    continue;
                }

                // Only process unvisited nodes
                if !visited.contains(neighbor_label) {
                    visited.insert(neighbor_label.clone());
                    total_nodes_visited += 1;

                    // Assign this node to the same terminal with distance + 1
                    // and record the parent for path reconstruction
                    voronoi.insert(neighbor_label.clone(), VoronoiCell {
                        nearest_terminal: terminal_label.clone(),
                        distance: dist + 1,
                        parent: Some(node_label.clone()), // Parent is the node we came from
                    });

                    // Continue BFS from this neighbor
                    queue.push_back((neighbor_label.clone(), terminal_label.clone(), dist + 1));
                }
            }
        }

        info!("  Voronoi diagram complete: {} nodes assigned to {} regions",
                   total_nodes_visited,
                   terminals.len());

        // Log statistics about the Voronoi regions
        let mut region_sizes: IndexMap<String, usize> = IndexMap::new();
        let mut max_distance = 0;

        for cell in voronoi.values() {
            *region_sizes.entry(cell.nearest_terminal.clone()).or_insert(0) += 1;
            max_distance = max_distance.max(cell.distance);
        }

        info!("  Region statistics:");
        for (terminal, size) in &region_sizes {
            info!("    Terminal {}: {} nodes", terminal, size);
        }
        info!("  Max distance to any terminal: {}", max_distance);

        voronoi
    }

    /// Reconstruct the path from a node to its nearest terminal
    pub fn reconstruct_path_to_terminal(&self, voronoi: &IndexMap<String, VoronoiCell>,
                                        node_label: &str)
                                        -> Vec<String> {
        let mut path = Vec::new();
        let mut current = node_label.to_string();

        while let Some(cell) = voronoi.get(&current) {
            path.push(current.clone());

            if let Some(parent) = &cell.parent {
                current = parent.clone();
            } else {
                // Reached a terminal (no parent)
                break;
            }
        }

        path.reverse(); // Reverse to get path from terminal to node
        path
    }

    /// Find the path between two nodes using their Voronoi cells
    /// Returns the path if both nodes are in the same region, or through boundary if different regions
    pub fn find_path_through_voronoi(&self, voronoi: &IndexMap<String, VoronoiCell>, from: &str,
                                     to: &str)
                                     -> Option<Vec<String>> {
        let from_cell = voronoi.get(from)?;
        let to_cell = voronoi.get(to)?;

        if from_cell.nearest_terminal == to_cell.nearest_terminal {
            // Same region: find common ancestor
            let path_from = self.reconstruct_path_to_terminal(voronoi, from);
            let path_to = self.reconstruct_path_to_terminal(voronoi, to);

            // Find lowest common ancestor
            let mut common_prefix_len = 0;
            for i in 0..path_from.len().min(path_to.len()) {
                if path_from[i] == path_to[i] {
                    common_prefix_len = i + 1;
                } else {
                    break;
                }
            }

            if common_prefix_len > 0 {
                // Build path: from -> LCA -> to
                let mut path = path_from[common_prefix_len..].to_vec();
                path.reverse();
                path.extend_from_slice(&path_from[..common_prefix_len]);
                path.extend_from_slice(&path_to[common_prefix_len..]);
                return Some(path);
            }
        }

        // Different regions: would need boundary crossing
        // This is a more complex case that could be handled separately
        None
    }
}
 */
