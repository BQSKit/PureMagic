use crate::circuit::Circuit;
use crate::pauliproduct::PauliProduct;
use crate::topograph::{NodeType, TopoGraph};
use crate::utils::{IntermittentTimer, Timer};

use indexmap::IndexSet;
use log;
use rand_simple::Exponential;
use simple_logging;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::Path;

struct ScheduleStats {
    qubits: usize,
    data_qubits: usize,
    bus_qubits: usize,
    magic_qubits: usize,
    estabilizer_qubits: usize,
    sum_data_scheduled: usize,
    sum_bus_scheduled: usize,
    sum_magic_scheduled: usize,
    sum_ancilla_scheduled: usize,
    sum_estabilizer_scheduled: usize,
    bus_scheduled: usize,
    data_scheduled: usize,
    magic_scheduled: usize,
    ancilla_scheduled: usize,
    estabilizers_scheduled: usize,
}

impl ScheduleStats {
    pub fn new(qubits: usize, data_qubits: usize, bus_qubits: usize, magic_qubits: usize,
               estabilizer_qubits: usize)
               -> Self {
        ScheduleStats { qubits,
                        data_qubits,
                        bus_qubits,
                        magic_qubits,
                        estabilizer_qubits,
                        sum_data_scheduled: 0,
                        sum_bus_scheduled: 0,
                        sum_magic_scheduled: 0,
                        sum_ancilla_scheduled: 0,
                        sum_estabilizer_scheduled: 0,
                        bus_scheduled: 0,
                        data_scheduled: 0,
                        magic_scheduled: 0,
                        ancilla_scheduled: 0,
                        estabilizers_scheduled: 0 }
    }

    pub fn summarize(&self, num_steps: usize) -> f64 {
        // Calculate statistics
        let data_frac = self.sum_data_scheduled as f64 / (self.data_qubits * num_steps) as f64;
        let bus_frac = self.sum_bus_scheduled as f64 / (self.bus_qubits * num_steps) as f64;
        let magic_frac = self.sum_magic_scheduled as f64 / (self.magic_qubits * num_steps) as f64;
        let estabilizer_frac =
            self.sum_estabilizer_scheduled as f64 / (self.estabilizer_qubits * num_steps) as f64;

        let overall_frac = (self.data_qubits * num_steps
                            + self.sum_bus_scheduled
                            + self.sum_magic_scheduled
                            + self.sum_ancilla_scheduled
                            + self.sum_estabilizer_scheduled) as f64
                           / (num_steps * self.qubits) as f64;

        // Print final statistics
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("  estabilizer: {:.3}", estabilizer_frac);

        overall_frac
    }

    pub fn update(&mut self, step_i: usize, pp_paths_len: usize, to_schedule_len: usize) -> String {
        self.sum_data_scheduled += self.data_scheduled;
        self.sum_bus_scheduled += self.bus_scheduled;
        self.sum_magic_scheduled += self.magic_scheduled;
        self.sum_ancilla_scheduled += self.ancilla_scheduled;
        self.sum_estabilizer_scheduled += self.estabilizers_scheduled;

        log::info!("Scheduling results:");
        let frac_paths = pp_paths_len as f64 / to_schedule_len as f64;
        let frac_data = self.data_scheduled as f64 / self.data_qubits as f64;
        let frac_bus = self.bus_scheduled as f64 / self.bus_qubits as f64;
        let frac_magic = self.magic_scheduled as f64 / self.magic_qubits as f64;
        let frac_estabilizers = self.estabilizers_scheduled as f64 / self.estabilizer_qubits as f64;
        log::info!("  products:    {}/{} ({:.2})", pp_paths_len, to_schedule_len, frac_paths);
        log::info!("  data:        {}/{} ({:.2})",
                   self.data_scheduled,
                   self.data_qubits,
                   frac_data);
        log::info!("  bus:         {}/{} ({:.2})", self.bus_scheduled, self.bus_qubits, frac_bus);
        log::info!("  magic:       {}/{} ({:.2})",
                   self.magic_scheduled,
                   self.magic_qubits,
                   frac_magic);
        log::info!("  estabilizer: {}/{} ({:.2})",
                   self.estabilizers_scheduled,
                   self.estabilizer_qubits,
                   frac_estabilizers);

        let title =
            format!("Step {} Products scheduled: {:.2}; qubits: data {:.2}, \
                        bus {:.2}, magic {:.2}, estabilizer {:.2}",
                    step_i, frac_paths, frac_data, frac_bus, frac_magic, frac_estabilizers);

        self.data_scheduled = 0;
        self.bus_scheduled = 0;
        self.magic_scheduled = 0;
        self.ancilla_scheduled = 0;
        self.estabilizers_scheduled = 0;
        title
    }

    pub fn inc(&mut self, node_type: NodeType) {
        match node_type {
            NodeType::Bus => self.bus_scheduled += 1,
            NodeType::Magic => self.magic_scheduled += 1,
            NodeType::Data => self.data_scheduled += 1,
            NodeType::Estabilizer => self.estabilizers_scheduled += 1,
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
    schedule_non_clifford_timer: IntermittentTimer,
    schedule_clifford_timer: IntermittentTimer,
    stats: ScheduleStats,
}

impl Scheduler {
    pub fn new(circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_scheduler: bool,
               plot_option: String, rseed: u32)
               -> Self {
        if log_scheduler {
            let circuit_stem = Path::new(&circuit.circuit_fname).file_stem()
                                                                .and_then(|s| s.to_str())
                                                                .unwrap_or("circuit");
            let sched_fname = format!("{}.sched", circuit_stem);
            simple_logging::log_to_file(&sched_fname, log::LevelFilter::Info)
                .expect("Failed to initialize logging");
        }
        let num_qubits = topo.num_qubits;
        let num_data_qubits = topo.num_data_qubits;
        let num_bus_qubits = topo.num_bus_qubits;
        let num_magic_qubits = topo.num_magic_qubits;
        let num_estabilizer_qubits = topo.num_estabilizer_qubits;

        Scheduler { circuit,
                    topo,
                    rng_exp: Exponential::new(rseed),
                    magic_state_lambda,
                    plot_option,
                    cultivation_times: Vec::new(),
                    schedule_non_clifford_timer: IntermittentTimer::new("sched non-clifford", ""),
                    schedule_clifford_timer: IntermittentTimer::new("sched clifford", ""),
                    stats: ScheduleStats::new(num_qubits,
                                              num_data_qubits,
                                              num_bus_qubits,
                                              num_magic_qubits,
                                              num_estabilizer_qubits) }
    }

    pub fn schedule_circuit(&mut self, best_fit: bool) -> io::Result<(usize, usize, f64)> {
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
        // Main scheduling loop
        while !to_schedule.is_empty() {
            num_steps += 1;
            log::info!("Step {}: {:?}",
                       num_steps,
                       to_schedule.iter()
                                  .map(|pp| format!("{}:{}", pp.id, pp.get_product_str()))
                                  .collect::<Vec<_>>());

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
                                              "Cannot schedule on current layout"));
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
        }

        let overall_frac = self.stats.summarize(num_steps);
        println!("Magic state cultivation time:");
        let mean =
            self.cultivation_times.iter().sum::<i32>() as f64 / self.cultivation_times.len() as f64;
        let min = self.cultivation_times.iter().min().copied().unwrap_or(0);
        let max = self.cultivation_times.iter().max().copied().unwrap_or(0);
        println!("  number:  {}", self.cultivation_times.len());
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);

        self.schedule_clifford_timer.done();
        self.schedule_non_clifford_timer.done();

        Ok((num_steps, scheduled.len(), overall_frac))
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
        }

        let mut pp_paths = Vec::new();
        let mut next_to_schedule = Vec::new();
        let mut num_dependent_nodes = 0;

        let mut remaining_to_schedule: IndexSet<usize> = (0..to_schedule.len()).collect();
        // if we are only selecting the first product, then presort products from those needing the
        // most resources to those needing the least - this seems to work the best
        if !best_fit {
            let mut remaining_vec: Vec<usize> = remaining_to_schedule.into_iter().collect();
            remaining_vec.sort_by_key(|&idx| {
                             let pp = &to_schedule[idx];
                             self.circuit.num_qubits - pp.operators.len() + (pp.num_ys + 1) % 2
                         });
            remaining_to_schedule = remaining_vec.into_iter().collect();
        }
        while !remaining_to_schedule.is_empty() {
            let mut to_remove = Vec::new();
            let mut best_pp: Option<(usize, TopoGraph)> = None;
            let mut best_pp_size = usize::MAX;
            for &pp_i in &remaining_to_schedule {
                let pp = &to_schedule[pp_i];
                match self.schedule_pauli_product(pp) {
                    None => {
                        log::info!("  * Could not schedule {} on graph", pp.id);
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
                    }
                    Some(pp_graph) => {
                        let pp_size = pp_graph.num_nodes;
                        if best_pp_size >= pp_size {
                            best_pp_size = pp_size;
                            best_pp = Some((pp_i, pp_graph));
                            log::info!("  New best graph for pp {}, size {}",
                                       pp.get_product_str(),
                                       best_pp_size);
                            if !best_fit {
                                break;
                            }
                        }
                    }
                }
            }
            if let Some((best_pp_idx, best_graph)) = best_pp {
                let pp = &to_schedule[best_pp_idx];
                log::info!("* Scheduled product {} with {} nodes and {} edges: {:?}",
                           pp.id,
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
        log::info!("Removed {} dependent nodes", num_dependent_nodes);

        if !pp_paths.is_empty() {
            (Some(title), Some(pp_paths), next_to_schedule)
        } else {
            (None, None, next_to_schedule)
        }
    }

    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct) -> Option<TopoGraph> {
        log::info!("Trying to schedule product {}", pauli_product);
        // Initially terminal nodes contain only the data qubits
        let mut data_nodes = Vec::new();
        for op in &pauli_product.operators {
            if op.basis == 'Y' {
                for term in ['X', 'Z'] {
                    let node_label = format!("d{}{}", op.qubit, term);
                    let node = self.topo.get_node(&node_label);
                    // Check if node is already used
                    if node.used {
                        log::info!("  Node {} is already used", node_label);
                        return None;
                    }
                    // check for at least one unused magic or bus nb
                    if !node.edges.iter().any(|nb_label| {
                                             let nb = self.topo.get_node(nb_label);
                                             !nb.used
                                         })
                    {
                        log::info!("  No unused neighbors for node {}", node.label);
                        return None;
                    }
                    data_nodes.push(node_label);
                }
            } else {
                let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                let node = self.topo.get_node(&node_label);
                // Check if node is already used
                if node.used {
                    log::info!("  Node {} is already used", node_label);
                    return None;
                }
                // check for at least one unused magic or bus nb
                if !node.edges.iter().any(|nb_label| {
                                         let nb = self.topo.get_node(nb_label);
                                         !nb.used
                                     })
                {
                    log::info!("  No unused neighbors for node {}", node.label);
                    return None;
                }
                data_nodes.push(node_label);
            }
        }
        if data_nodes.is_empty() {
            log::info!("  No data nodes found in working graph");
            return None;
        }
        if !pauli_product.is_clifford {
            self.schedule_non_clifford_timer.start();
            let g = self.schedule_non_clifford(&mut data_nodes, pauli_product);
            self.schedule_non_clifford_timer.stop();
            g
        } else {
            self.schedule_clifford_timer.start();
            let g = self.schedule_clifford(&mut data_nodes, pauli_product);
            self.schedule_clifford_timer.stop();
            g
        }
    }

    fn schedule_non_clifford(&self, data_nodes: &mut Vec<String>, pauli_product: &PauliProduct)
                             -> Option<TopoGraph> {
        // Find available magic nodes
        let mut magic_nodes = Vec::new();
        for node in self.topo.iter_nodes() {
            if node.node_type == NodeType::Magic && !node.is_cultivating() && !node.used {
                let mut unused_nb = false;
                for nb_label in &node.edges {
                    let nb = self.topo.get_node(nb_label);
                    if self.topo.is_routing_node(nb) && !nb.used {
                        unused_nb = true;
                        break;
                    }
                }
                if unused_nb {
                    magic_nodes.push(node.label.clone());
                }
            }
        }
        if magic_nodes.is_empty() {
            log::info!("  No available magic nodes");
            return None;
        }
        log::info!("  Found {} available magic nodes {:?}", magic_nodes.len(), magic_nodes);
        if pauli_product.need_estabilizer {
            self.find_estabilizer_tree(&magic_nodes, data_nodes, pauli_product)
        } else {
            let magic_nodes_sorted = self.get_nodes_by_dist(&magic_nodes, pauli_product)
                                         .into_iter()
                                         .map(|(node, _)| node)
                                         .collect::<Vec<_>>();
            log::info!("  Magic nodes by distance to {}: {:?}",
                       pauli_product.id,
                       magic_nodes_sorted);
            self.find_tree(&magic_nodes_sorted, data_nodes, pauli_product, false)
        }
    }

    fn find_estabilizer_tree(&self, magic_nodes: &[String], data_nodes: &mut Vec<String>,
                             pauli_product: &PauliProduct)
                             -> Option<TopoGraph> {
        // Find available estabilizer nodes
        let mut estabilizer_nodes = Vec::new();
        for node in self.topo.iter_nodes() {
            if node.node_type == NodeType::Estabilizer {
                let mut num_unused_nbs = 0;
                for nb_label in &node.edges {
                    let nb = self.topo.get_node(nb_label);
                    if self.topo.is_routing_node(nb) && !nb.used {
                        num_unused_nbs += 1;
                        if num_unused_nbs == 2 {
                            break;
                        }
                    }
                }
                if num_unused_nbs == 2 {
                    estabilizer_nodes.push(node.label.clone());
                }
            }
        }
        if estabilizer_nodes.is_empty() {
            log::info!("  No available estabilizer nodes");
            return None;
        }
        // Get distances from estabilizer nodes to data nodes
        let estabilizer_distances = self.get_nodes_by_dist(&estabilizer_nodes, pauli_product);
        // Calculate distances from magic nodes through estabilizer nodes
        let mut magic_path_dists = Vec::new();
        for magic_node in magic_nodes {
            for (estabilizer_node, estabilizer_d) in &estabilizer_distances {
                let d = self.get_node_dist(magic_node, estabilizer_node) + estabilizer_d;
                magic_path_dists.push((magic_node.clone(), estabilizer_node.clone(), d));
            }
        }
        magic_path_dists.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        // Try each magic-estabilizer path
        for (magic_node, estabilizer_node, d) in magic_path_dists {
            // Find path from magic to estabilizer
            let mut estabilizer_vec = vec![estabilizer_node.clone()];
            let magic_path_g = self.get_bfs_graph(&magic_node, &mut estabilizer_vec, false, None);
            if magic_path_g.is_none() {
                //log::info!("  No path from {} to {}", magic_node, estabilizer_node);
                continue;
            }
            let magic_path_g = magic_path_g.unwrap();
            log::info!("  Found graph from {} to {} of size {}",
                       magic_node,
                       estabilizer_node,
                       magic_path_g.num_edges);
            // Find path from estabilizer to data nodes
            let estabilizer_g =
                self.get_bfs_graph(&estabilizer_node, data_nodes, false, Some(&magic_path_g));
            if estabilizer_g.is_none() {
                log::info!("  No path from {} to {}", estabilizer_node, pauli_product);
                continue;
            }
            let mut estabilizer_g = estabilizer_g.unwrap();
            log::info!("  Found graph from {} ({}) of size {}",
                       estabilizer_node,
                       magic_node,
                       estabilizer_g.num_edges);
            if pauli_product.need_ancilla {
                if let Some(ancilla) = self.find_ancilla(&mut estabilizer_g) {
                    estabilizer_g.ancilla_node = Some(ancilla);
                } else {
                    log::info!("    Couldn't find ancilla for tree");
                    return None;
                }
            }
            // Merge the two graphs
            for node in magic_path_g.iter_nodes() {
                if node.node_type != NodeType::Estabilizer {
                    estabilizer_g.add_node(node.clone());
                }
            }
            for (from, to) in magic_path_g.iter_edges() {
                estabilizer_g.add_edge(from, to);
            }
            estabilizer_g.root_node = magic_path_g.root_node.clone();
            log::info!("  Final graph has {} edges (estimated distance {:.0})",
                       estabilizer_g.num_edges,
                       d);
            return Some(estabilizer_g);
        }
        log::info!("  No path from estabilizer nodes {:?} to {:?}", estabilizer_nodes, data_nodes);
        None
    }

    fn get_nodes_by_dist(&self, node_labels: &[String], pauli_product: &PauliProduct)
                         -> Vec<(String, f64)> {
        // Sort nodes by distance to pp data nodes
        let mut node_distances = Vec::new();

        for node_label in node_labels {
            let mut min_d = f64::MAX;
            for op in &pauli_product.operators {
                if op.basis == 'Y' {
                    let data_node_label_x = format!("d{}{}", op.qubit, 'X');
                    let d_x = self.get_node_dist(node_label, &data_node_label_x);
                    if d_x < min_d {
                        min_d = d_x;
                    }
                    let data_node_label_z = format!("d{}{}", op.qubit, 'Z');
                    let d_z = self.get_node_dist(node_label, &data_node_label_z);
                    if d_z < min_d {
                        min_d = d_z;
                    }
                } else {
                    let data_node_label = format!("d{}", op.to_string().to_ascii_uppercase());
                    let d = self.get_node_dist(node_label, &data_node_label);
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            node_distances.push((node_label.clone(), min_d));
        }
        node_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        node_distances
    }

    fn get_node_dist(&self, node1: &str, node2: &str) -> f64 {
        let pos1 = self.topo.get_node(node1).pos;
        let pos2 = self.topo.get_node(node2).pos;
        let dx = pos1.0 as f64 - pos2.0 as f64;
        let dy = pos1.1 as f64 - pos2.1 as f64;
        (dx * dx + dy * dy).sqrt()
    }

    fn schedule_clifford(&self, data_nodes: &mut Vec<String>, pauli_product: &PauliProduct)
                         -> Option<TopoGraph> {
        // Handle single data node case
        if data_nodes.len() == 1 && !pauli_product.need_estabilizer {
            let node_label = &data_nodes[0];
            let node = self.topo.get_node(node_label);
            if node.used {
                log::info!("  Single node {} is used", node_label);
                return None;
            }

            let mut g = TopoGraph::new();
            g.add_node(node.clone());

            if pauli_product.need_ancilla {
                // Try to find an available bus/magic neighbor
                for nb_label in node.edges.iter() {
                    let nb = self.topo.get_node(nb_label);
                    if self.topo.is_routing_node(nb) && !nb.used {
                        g.add_node(nb.clone());
                        g.add_edge(node_label, nb_label);
                        g.ancilla_node = Some(nb_label.clone());
                        break;
                    }
                }
                if g.num_nodes == 0 {
                    log::info!("  Could not find ancilla for node {}", node_label);
                    return None;
                }
            }
            log::info!("Scheduled clifford on {:?} nodes",
                       g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>());
            return Some(g);
        }
        // root node needs to be a bus/magic node next to one of the data nodes
        let mut root_nodes = IndexSet::new();
        for node_label in data_nodes.iter() {
            let node = self.topo.get_node(node_label);
            if node.used {
                return None;
            }
            for nb_label in node.edges.iter() {
                let nb = self.topo.get_node(&nb_label);
                if !nb.used && self.topo.is_routing_node(nb) {
                    root_nodes.insert(nb_label.clone());
                }
            }
        }
        if root_nodes.is_empty() {
            return None;
        }
        // Try to find a tree using each root node
        let root_nodes_vec: Vec<String> = root_nodes.iter().cloned().collect();
        // find_tree also finds the ancilla if needed
        let g = self.find_tree(&root_nodes_vec,
                               data_nodes,
                               pauli_product,
                               pauli_product.need_estabilizer);
        if let Some(ref g) = g {
            log::info!("Scheduled clifford in {:?} nodes",
                       g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>());
        }
        g
    }

    fn find_tree(&self, root_nodes: &[String], data_nodes: &mut Vec<String>,
                 pauli_product: &PauliProduct, estabilizer: bool)
                 -> Option<TopoGraph> {
        log::info!("  Find tree for {}:", pauli_product.id);
        for root_node_label in root_nodes {
            let g = self.get_bfs_graph(root_node_label, data_nodes, estabilizer, None);
            match g {
                None => {
                    log::info!("    No tree from root node {} to {:?}, {}",
                               root_node_label,
                               data_nodes,
                               pauli_product.need_ancilla);
                    continue;
                }
                Some(mut graph) => {
                    log::info!("    Found tree from {} to {:?} of size {}",
                               root_node_label,
                               data_nodes,
                               graph.num_edges,);
                    if pauli_product.need_ancilla {
                        if let Some(ancilla) = self.find_ancilla(&mut graph) {
                            graph.ancilla_node = Some(ancilla);
                        } else {
                            log::info!("    Couldn't find ancilla for tree");
                            return None;
                        }
                    }
                    return Some(graph);
                }
            }
        }
        None
    }

    fn get_bfs_graph(&self, root_node: &str, terminal_nodes: &mut Vec<String>,
                     with_estabilizer: bool, exclude: Option<&TopoGraph>)
                     -> Option<TopoGraph> {
        log::info!("  BFS from node {} to nodes {:?}", root_node, terminal_nodes);
        let mut visited = IndexSet::with_capacity(self.topo.num_nodes);
        let mut queue = VecDeque::with_capacity(self.topo.num_nodes);
        let mut bfs_graph = TopoGraph::new();

        visited.insert(root_node);
        queue.push_back(root_node);
        let mut num_terminals_reqd = terminal_nodes.len();
        let mut num_found_terminals = 0;
        if with_estabilizer {
            num_terminals_reqd += 1;
        }
        let mut ez_label = String::new();

        bfs_graph.add_node(self.topo.get_node(root_node).clone());
        while let Some(node_label) = queue.pop_front() {
            let node = self.topo.get_node(&node_label);
            for nb_label in node.edges.iter() {
                let nb = self.topo.get_node(&nb_label);
                if nb.used {
                    continue;
                }
                if visited.contains(nb_label.as_str()) {
                    continue;
                }
                // Skip excluded nodes unless they are terminal nodes
                if let Some(ex) = exclude {
                    if ex.contains_node(&nb_label) && !terminal_nodes.contains(&nb_label) {
                        continue;
                    }
                }
                if with_estabilizer
                   && nb.node_type == NodeType::Estabilizer
                   && ez_label == ""
                   && !terminal_nodes.contains(&nb_label)
                {
                    ez_label = nb.label.clone();
                    terminal_nodes.push(ez_label.clone());
                }
                if self.topo.is_routing_node(nb) {
                    bfs_graph.add_node(nb.clone());
                    bfs_graph.add_edge(&node_label, &nb_label);
                    queue.push_back(nb_label);
                } else if terminal_nodes.contains(&nb_label) {
                    if nb.node_type == NodeType::Data {
                        let paired_nb = self.topo.get_paired_data_node(nb);
                        if node.edges.contains(&paired_nb.label) {
                            // this is a top or bottom connection
                            log::info!("    Node {} has a top/bottom connection to {}/{}",
                                       node.label,
                                       nb.label,
                                       paired_nb.label);
                            // only use the top/bottom if both nodes are in the terminals
                            if !terminal_nodes.contains(&nb.label)
                               || !terminal_nodes.contains(&paired_nb.label)
                            {
                                continue;
                            }
                            if visited.contains(paired_nb.label.as_str()) {
                                // paired node is already in the graph, remove the previous edge
                                let prev_paired_node = bfs_graph.get_node_mut(&paired_nb.label);
                                // there should be only one edge to remove
                                assert!(prev_paired_node.edges.len() == 1);
                                bfs_graph.remove_all_edges(&paired_nb.label);
                            }
                            log::info!("    Using top/bottom connection");
                            // if we haven't used both data nodes yet, then make this
                            // top/bottom the one used
                            visited.insert(paired_nb.label.as_str());
                            bfs_graph.add_node(paired_nb.clone());
                            bfs_graph.add_edge(&node_label, &paired_nb.label);
                            num_found_terminals += 1;
                        }
                    }
                    bfs_graph.add_node(nb.clone());
                    bfs_graph.add_edge(&node_label, &nb_label);
                    num_found_terminals += 1;
                    if num_found_terminals == num_terminals_reqd {
                        log::info!("    Found tree of {} nodes", bfs_graph.node_list().len());
                        bfs_graph.trim_dangling_nodes(root_node);
                        bfs_graph.root_node = Some(root_node.to_string());
                        return Some(bfs_graph);
                    }
                }
                visited.insert(nb_label);
            }
        }
        None
    }

    fn find_ancilla(&self, graph: &mut TopoGraph) -> Option<String> {
        // Collect bus nodes first to avoid borrowing issues
        let bus_nodes: Vec<_> = graph.iter_nodes()
                                     .filter(|node| self.topo.is_routing_node(node))
                                     .map(|node| node.label.clone())
                                     .collect();
        for node_label in bus_nodes {
            // Check neighbors in the topology
            for nb_label in &self.topo.get_node(&node_label).edges {
                let nb = self.topo.get_node(&nb_label);
                // Check if neighbor is an unused bus/magic node not in graph
                if nb.used || graph.contains_node(nb_label) {
                    continue;
                }
                if self.topo.is_routing_node(nb) {
                    log::info!("    Selected {} as ancilla", nb_label);
                    // Add the node and edge to the graph
                    graph.add_node(nb.clone());
                    graph.add_edge(&node_label, nb_label);
                    return Some(nb_label.clone());
                }
            }
        }
        None
    }

    fn gen_cultivation_time(&mut self) -> i32 {
        let cultivation_time = self.rng_exp.sample().round() as i32 + 1;
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
