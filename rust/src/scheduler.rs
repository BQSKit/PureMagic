use crate::circuit::Circuit;
use crate::pauliproduct::PauliProduct;
use crate::topograph::{NodeType, TopoGraph};
use crate::utils::{IntermittentTimer, Timer};

use log;
use rand_simple::Exponential;
use simple_logging;
use std::collections::{HashSet, VecDeque};
use std::io::{self, Write};
use std::path::Path;

pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng_exp: Exponential,
    magic_state_lambda: f64,
    plot_option: String,
    sum_data_qubits: usize,
    sum_bus_qubits: usize,
    sum_magic_qubits: usize,
    sum_ancilla_qubits: usize,
    sum_estabilizer_qubits: usize,
    busy_count_list: Vec<i32>,
    schedule_non_clifford_timer: IntermittentTimer,
    schedule_clifford_timer: IntermittentTimer,
}

impl Scheduler {
    pub fn new(circuit: Circuit, topo: TopoGraph, magic_state_lambda: f64, log_scheduler: bool,
               plot_option: String)
               -> Self {
        if log_scheduler {
            let circuit_stem = Path::new(&circuit.circuit_fname).file_stem()
                                                                .and_then(|s| s.to_str())
                                                                .unwrap_or("circuit");
            let sched_fname = format!("{}.sched", circuit_stem);
            simple_logging::log_to_file(&sched_fname, log::LevelFilter::Info)
                .expect("Failed to initialize logging");
        }
        Scheduler { circuit,
                    topo,
                    rng_exp: Exponential::new(29),
                    magic_state_lambda,
                    plot_option,
                    sum_data_qubits: 0,
                    sum_bus_qubits: 0,
                    sum_magic_qubits: 0,
                    sum_ancilla_qubits: 0,
                    sum_estabilizer_qubits: 0,
                    busy_count_list: Vec::new(),
                    schedule_non_clifford_timer: IntermittentTimer::new("schedule non-clifford",
                                                                        ""),
                    schedule_clifford_timer: IntermittentTimer::new("schedule clifford", "") }
    }

    pub fn schedule_circuit(&mut self) -> io::Result<(usize, usize, f64)> {
        let _timer = Timer::new("schedule_circuit");
        self.rng_exp
            .try_set_params(self.magic_state_lambda)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        // Initialize magic nodes with busy counts
        // Collect magic node labels first to avoid borrow conflicts
        let magic_labels: Vec<String> = self.topo
                                            .iter_nodes()
                                            .filter(|node| node.node_type == NodeType::Magic)
                                            .map(|node| node.label.clone())
                                            .collect();
        for label in magic_labels {
            let busy_count = self.gen_busy_count();
            //log::info!("Set node {} busy count to {}", &label, busy_count);
            self.topo.get_node_mut(&label).busy_count = Some(busy_count);
        }
        // Initialize scheduling
        let mut to_schedule: Vec<_> =
            self.circuit.products.iter().filter(|pp| pp.parents.is_empty()).cloned().collect();
        let mut circuit_products = self.circuit.products.to_vec();
        let mut scheduled = HashSet::new();
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
        let total_to_schedule = circuit_products.len();
        let mut prev_perc_complete = 0;
        print!("Scheduling {} products:    ", total_to_schedule);
        if plot_steps > 0 {
            println!();
        }
        // Main scheduling loop
        while !to_schedule.is_empty() {
            num_steps += 1;
            // Update progress
            if plot_steps == 0 {
                let perc_complete = (scheduled.len() * 100) / total_to_schedule;
                if perc_complete > prev_perc_complete {
                    print!("\x08\x08\x08{:02}%", perc_complete);
                    std::io::stdout().flush()?;
                    prev_perc_complete = perc_complete;
                }
            }
            log::info!("Step {}: {:?}",
                       num_steps,
                       to_schedule.iter()
                                  .map(|pp| format!("{}:{}", pp.id, pp.get_product_str()))
                                  .collect::<Vec<_>>());

            let (title_str, pp_paths, next_to_schedule) =
                self.schedule_timestep(num_steps, &to_schedule);

            if pp_paths.is_none() {
                // carry on if there are no available magic nodes
                let mut has_busy_magic = false;
                for node in self.topo.iter_nodes() {
                    if node.node_type == NodeType::Magic && node.busy_count.unwrap_or(0) > 0 {
                        has_busy_magic = true;
                        break;
                    }
                }
                if !has_busy_magic {
                    return Err(io::Error::new(io::ErrorKind::Other,
                                              "Cannot schedule on current layout"));
                }
                to_schedule = next_to_schedule;
                continue;
            }
            // Process scheduled products
            if let Some(ref pp_paths) = pp_paths {
                for (pp, _) in pp_paths {
                    // Add children to next round if all parents scheduled
                    for &child_id in &pp.children {
                        let child = &mut circuit_products[child_id as usize];
                        child.parents.retain(|&x| x != pp.id);
                        if child.parents.is_empty() {
                            to_schedule.push(child.clone());
                        }
                    }
                    self.check_dependencies(pp, &scheduled)?;
                    scheduled.insert(pp.id);
                }
            }
            // Plot if requested
            if let Some(ref path_dir) = path_dir {
                if title_str.is_some() && num_steps > 0 && plot_steps > 0 {
                    let fname_added = format!(".{}", num_steps);
                    let curr_dir = std::env::current_dir()?;
                    std::env::set_current_dir(path_dir)?;
                    self.topo
                        .plot(&fname_added, pp_paths.as_ref().unwrap(), &title_str.unwrap())
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
                    std::env::set_current_dir(curr_dir)?;
                    plot_steps -= 1;
                }
            }

            to_schedule = next_to_schedule;
        }
        print!("\x08\x08\x08{:02}%\n", 100.0);

        // Calculate statistics
        let data_frac =
            self.sum_data_qubits as f64 / (self.topo.num_data_qubits * num_steps) as f64;
        let bus_frac = self.sum_bus_qubits as f64 / (self.topo.num_bus_qubits * num_steps) as f64;
        let magic_frac =
            self.sum_magic_qubits as f64 / (self.topo.num_magic_qubits * num_steps) as f64;
        let estabilizer_frac = self.sum_estabilizer_qubits as f64
                               / (self.topo.num_estabilizer_qubits * num_steps) as f64;

        let overall_frac = (self.topo.num_data_qubits * num_steps
                            + self.sum_bus_qubits
                            + self.sum_magic_qubits
                            + self.sum_ancilla_qubits
                            + self.sum_estabilizer_qubits) as f64
                           / (num_steps * self.topo.num_qubits) as f64;

        self.schedule_clifford_timer.done();
        self.schedule_non_clifford_timer.done();
        // Print final statistics
        println!("Qubit fractions used:");
        println!("  data:        {:.3}", data_frac);
        println!("  bus:         {:.3}", bus_frac);
        println!("  magic:       {:.3}", magic_frac);
        println!("  estabilizer: {:.3}", estabilizer_frac);

        println!("Magic state cultivation time:");
        let mean =
            self.busy_count_list.iter().sum::<i32>() as f64 / self.busy_count_list.len() as f64;
        let min = self.busy_count_list.iter().min().copied().unwrap_or(0);
        let max = self.busy_count_list.iter().max().copied().unwrap_or(0);
        println!("  average: {:.2}", mean);
        println!("  min:     {}", min);
        println!("  max:     {}", max);

        Ok((num_steps, scheduled.len(), overall_frac))
    }

    fn schedule_timestep(
        &mut self, step_i: usize, to_schedule: &[PauliProduct])
        -> (Option<String>, Option<Vec<(PauliProduct, TopoGraph)>>, Vec<PauliProduct>) {
        // Update busy counts and reset used flags
        let mut num_busy = 0;
        for node in self.topo.iter_nodes_mut() {
            if node.node_type == NodeType::Magic && node.busy_count.unwrap_or(0) > 0 {
                node.busy_count = Some(node.busy_count.unwrap() - 1);
                if node.busy_count.unwrap() > 0 {
                    num_busy += 1;
                }
            }
            node.used = false;
        }

        // Sort products by number of operators (reverse)
        let mut to_schedule = to_schedule.to_vec();
        to_schedule.sort_by_key(|pp| std::cmp::Reverse(pp.operators.len()));

        let mut pp_paths = Vec::new();
        let mut num_scheduled = 0;
        let mut num_bus_scheduled = 0;
        let mut num_data_scheduled = 0;
        let mut num_magic_scheduled = num_busy;
        let mut num_ancilla_scheduled = 0;
        let mut num_estabilizers_scheduled = 0;
        let mut num_dependent_nodes = 0;
        let mut next_to_schedule = Vec::new();

        for pp in &to_schedule {
            let pp_graph = self.schedule_pauli_product(pp);
            match pp_graph {
                None => {
                    log::info!("  * Could not schedule on graph");
                    next_to_schedule.push(pp.clone());
                    // Mark dependent nodes as used
                    for op in &pp.operators {
                        let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
                        self.topo.get_node_mut(&node_label).used = true;
                        num_dependent_nodes += 1;
                    }
                }
                Some(graph) => {
                    log::info!("  * Scheduled with {} nodes and {} edges: {:?}",
                               graph.num_nodes,
                               graph.num_edges,
                               graph.node_list());
                    // Update node statistics and mark as used
                    for node in graph.iter_nodes() {
                        match node.node_type {
                            NodeType::Bus => num_bus_scheduled += 1,
                            NodeType::Magic => {
                                let busy_count = self.gen_busy_count();
                                self.topo.get_node_mut(&node.label).busy_count = Some(busy_count);
                                //log::info!("Set node {} busy count to {}", node.label, busy_count);
                                num_magic_scheduled += 1;
                            }
                            NodeType::Data => num_data_scheduled += 1,
                            NodeType::Ancilla => num_ancilla_scheduled += 1,
                            NodeType::Estabilizer => num_estabilizers_scheduled += 1,
                        }
                        self.topo.get_node_mut(&node.label).used = true;
                    }
                    pp_paths.push((pp.clone(), graph));
                    num_scheduled += pp.operators.len();
                }
            }
        }

        // Print statistics
        log::info!("Scheduling results:");
        let frac_paths = pp_paths.len() as f64 / to_schedule.len() as f64;
        let frac_data = num_data_scheduled as f64 / self.topo.num_data_qubits as f64;
        let frac_bus = num_bus_scheduled as f64 / self.topo.num_bus_qubits as f64;
        let frac_magic = num_magic_scheduled as f64 / self.topo.num_magic_qubits as f64;
        let frac_estabilizers =
            num_estabilizers_scheduled as f64 / self.topo.num_estabilizer_qubits as f64;
        log::info!("  products:    {}/{} ({:.2})", pp_paths.len(), to_schedule.len(), frac_paths);
        log::info!("  data:        {}/{} ({:.2})",
                   num_data_scheduled,
                   self.topo.num_data_qubits,
                   frac_data);
        log::info!("  bus:         {}/{} ({:.2})",
                   num_bus_scheduled,
                   self.topo.num_bus_qubits,
                   frac_bus);
        log::info!("  magic:       {}/{} ({:.2})",
                   num_magic_scheduled,
                   self.topo.num_magic_qubits,
                   frac_magic);
        log::info!("  estabilizer: {}/{} ({:.2})",
                   num_estabilizers_scheduled,
                   self.topo.num_estabilizer_qubits,
                   frac_estabilizers);
        log::info!("Removed {} dependent nodes", num_dependent_nodes);

        // Update statistics
        self.sum_data_qubits += num_scheduled;
        self.sum_bus_qubits += num_bus_scheduled;
        self.sum_magic_qubits += num_magic_scheduled;
        self.sum_ancilla_qubits += num_ancilla_scheduled;
        self.sum_estabilizer_qubits += num_estabilizers_scheduled;

        if !pp_paths.is_empty() {
            let frac_paths = pp_paths.len() as f64 / to_schedule.len() as f64;
            let frac_data = num_data_scheduled as f64 / self.topo.num_data_qubits as f64;
            let frac_bus = num_bus_scheduled as f64 / self.topo.num_bus_qubits as f64;

            let title = format!("Step {} Products scheduled {:.2}, data {:.2}, bus {:.2}",
                                step_i, frac_paths, frac_data, frac_bus);
            (Some(title), Some(pp_paths), next_to_schedule)
        } else {
            (None, None, next_to_schedule)
        }
    }

    fn schedule_pauli_product(&mut self, pauli_product: &PauliProduct) -> Option<TopoGraph> {
        log::info!("Trying to schedule {}", pauli_product);
        // Initially terminal nodes contain only the data qubits
        let mut data_nodes = Vec::new();

        for op in &pauli_product.operators {
            let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
            // Check if node is already used
            if self.topo.get_node(&node_label).used {
                log::info!("  Node {} is already used", node_label);
                return None;
            }
            data_nodes.push(node_label);
        }
        if data_nodes.is_empty() {
            log::info!("  No data nodes found in working graph");
            return None;
        }
        if !pauli_product.is_clifford {
            self.schedule_non_clifford_timer.start();
            let g = self.schedule_non_clifford(&data_nodes, pauli_product);
            self.schedule_non_clifford_timer.stop();
            g
        } else {
            self.schedule_clifford_timer.start();
            let g = self.schedule_clifford(&data_nodes, pauli_product);
            self.schedule_clifford_timer.stop();
            g
        }
    }

    fn schedule_non_clifford(&self, data_nodes: &[String], pauli_product: &PauliProduct)
                             -> Option<TopoGraph> {
        // Find available magic nodes (busy_count == 0)
        let magic_nodes: Vec<String> =
            self.topo
                .iter_nodes()
                .filter(|node| {
                    node.node_type == NodeType::Magic && node.busy_count.unwrap_or(1) == 0
                })
                .map(|node| node.label.clone())
                .collect();
        if magic_nodes.is_empty() {
            log::info!("  No available magic nodes");
            return None;
        }
        log::info!("  Found {} available magic nodes", magic_nodes.len());
        if pauli_product.need_estabilizer {
            self.find_estabilizer_tree(&magic_nodes, data_nodes, pauli_product)
        } else {
            let magic_nodes_sorted = self.get_nodes_by_dist(&magic_nodes, pauli_product)
                                         .into_iter()
                                         .map(|(node, _)| node)
                                         .collect::<Vec<_>>();
            log::info!("  Magic nodes by distance to {}: {:?}", pauli_product, magic_nodes_sorted);
            self.find_tree(&magic_nodes_sorted, data_nodes, pauli_product)
        }
    }

    fn find_estabilizer_tree(&self, magic_nodes: &[String], data_nodes: &[String],
                             pauli_product: &PauliProduct)
                             -> Option<TopoGraph> {
        let which_ancilla = if pauli_product.need_ancilla {
            pauli_product.operators
                         .first()
                         .map(|op| op.basis.to_ascii_uppercase().to_string())
                         .unwrap_or_default()
        } else {
            String::new()
        };
        // Find available estabilizer nodes
        let estabilizer_nodes: Vec<String> =
            self.topo
                .iter_nodes()
                .filter(|node| node.node_type == NodeType::Estabilizer && !node.used)
                .map(|node| node.label.clone())
                .collect();
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
            let magic_path_g =
                self.get_bfs_graph(&magic_node, &[estabilizer_node.clone()], "", None);
            if magic_path_g.is_none() {
                log::info!("  No path from {} to {}", magic_node, estabilizer_node);
                continue;
            }
            let magic_path_g = magic_path_g.unwrap();
            log::info!("  Found graph from {} to {} of size {}",
                       magic_node,
                       estabilizer_node,
                       magic_path_g.num_edges);
            // Find path from estabilizer to data nodes
            let estabilizer_g = self.get_bfs_graph(&estabilizer_node,
                                                   data_nodes,
                                                   &which_ancilla,
                                                   Some(&magic_path_g));
            if estabilizer_g.is_none() {
                log::info!("  No path from {} to {}", estabilizer_node, pauli_product);
                continue;
            }
            let mut estabilizer_g = estabilizer_g.unwrap();
            log::info!("  Found graph from {} ({}) of size {}",
                       estabilizer_node,
                       magic_node,
                       estabilizer_g.num_edges);
            // Merge the two graphs
            for node in magic_path_g.iter_nodes() {
                if node.node_type != NodeType::Estabilizer {
                    estabilizer_g.add_node(node.clone());
                }
            }
            for (from, to) in magic_path_g.iter_edges() {
                estabilizer_g.add_edge(from, to);
            }
            log::info!("  Final graph has {} edges (estimated distance {:.0})",
                       estabilizer_g.num_edges,
                       d);
            return Some(estabilizer_g);
        }
        log::info!("  No path from estabilizer nodes {:?} to {:?}", estabilizer_nodes, data_nodes);
        None
    }

    fn get_nodes_by_dist(&self, nodes: &[String], pauli_product: &PauliProduct)
                         -> Vec<(String, f64)> {
        // Sort nodes by distance to pp data nodes
        let mut node_distances = Vec::new();

        for node in nodes {
            let mut min_d = f64::MAX;
            for op in &pauli_product.operators {
                let data_node = format!("d{}", op.to_string().to_ascii_uppercase());
                let d = self.get_node_dist(node, &data_node);
                if d < min_d {
                    min_d = d;
                }
            }
            node_distances.push((node.clone(), min_d));
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

    fn schedule_clifford(&self, data_nodes: &[String], pauli_product: &PauliProduct)
                         -> Option<TopoGraph> {
        // Handle single data node case
        if data_nodes.len() == 1 {
            let node_label = &data_nodes[0];
            let node = self.topo.get_node(node_label);
            if node.used {
                return None;
            }

            let mut g = TopoGraph::new();
            g.add_node(node.clone());

            if pauli_product.need_ancilla {
                // Try to find an available bus neighbor
                for nb_label in node.edges.iter() {
                    let nb = self.topo.get_node(nb_label);
                    if nb.node_type == NodeType::Bus && !nb.used {
                        g.add_node(nb.clone());
                        g.add_edge(node_label, nb_label);
                        break;
                    }
                }
                return None;
            }
            log::info!("Scheduled clifford on {:?} nodes",
                       g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>());
            return Some(g);
        }
        // root node needs to be a bus node next to one of the data nodes
        let mut root_nodes = HashSet::new();
        for node_label in data_nodes {
            let node = self.topo.get_node(node_label);
            if node.used {
                return None;
            }
            for nb_label in node.edges.iter() {
                let nb = self.topo.get_node(&nb_label);
                if !nb.used && nb.node_type == NodeType::Bus {
                    root_nodes.insert(nb_label.clone());
                }
            }
        }
        if root_nodes.is_empty() {
            return None;
        }
        // Try to find a tree using each root node
        let root_nodes_vec: Vec<String> = root_nodes.iter().cloned().collect();
        let g = self.find_tree(&root_nodes_vec, data_nodes, pauli_product);
        if let Some(ref g) = g {
            log::info!("Scheduled clifford in {:?} nodes",
                       g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>());
        }
        g
    }

    fn find_tree(&self, root_nodes: &[String], data_nodes: &[String],
                 pauli_product: &PauliProduct)
                 -> Option<TopoGraph> {
        log::info!("  Find tree for {}:", pauli_product);
        // Determine which_ancilla based on first operator's basis
        let which_ancilla = if pauli_product.need_ancilla {
            pauli_product.operators
                         .first()
                         .map(|op| op.basis.to_ascii_uppercase().to_string())
                         .unwrap_or_default()
        } else {
            String::new()
        };
        for root_node_label in root_nodes {
            let g = self.get_bfs_graph(root_node_label, data_nodes, &which_ancilla, None);
            match g {
                None => {
                    log::info!("    No tree from root node {} to {:?}, {}",
                               root_node_label,
                               data_nodes,
                               if which_ancilla.is_empty() { "" } else { &which_ancilla });
                    continue;
                }
                Some(graph) => {
                    log::info!("    Found tree from {} to {:?} of size {}",
                               root_node_label,
                               data_nodes,
                               graph.num_edges,);
                    return Some(graph);
                }
            }
        }
        None
    }

    fn get_bfs_graph(&self, root_node: &str, terminal_nodes: &[String], which_ancilla: &str,
                     exclude: Option<&TopoGraph>)
                     -> Option<TopoGraph> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut bfs_graph = TopoGraph::new();

        visited.insert(root_node.to_string());
        queue.push_back(root_node.to_string());
        let num_terminals_reqd = terminal_nodes.len();
        let mut num_found_terminals = 0;

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
                // Only add bus nodes or terminal nodes
                if nb.node_type != NodeType::Bus && !terminal_nodes.contains(&nb_label) {
                    continue;
                }
                visited.insert(nb_label.clone());
                bfs_graph.add_node(nb.clone());
                bfs_graph.add_edge(&node_label, &nb_label);
                if nb.node_type == NodeType::Bus {
                    queue.push_back(nb_label.to_string());
                } else {
                    if nb.node_type != NodeType::Ancilla {
                        num_found_terminals += 1;
                    }
                    if num_found_terminals == num_terminals_reqd {
                        bfs_graph.trim_dangling_bus_nodes();
                        if !which_ancilla.is_empty() {
                            if !self.find_ancilla(&mut bfs_graph, which_ancilla) {
                                return None;
                            }
                        }
                        return Some(bfs_graph);
                    }
                }
            }
        }
        None
    }

    fn find_ancilla(&self, graph: &mut TopoGraph, which_ancilla: &str) -> bool {
        // Collect bus nodes first to avoid borrowing issues
        let bus_nodes: Vec<_> = graph.iter_nodes()
                                     .filter(|node| node.node_type == NodeType::Bus)
                                     .map(|node| node.label.clone())
                                     .collect();

        for node_label in bus_nodes {
            // Check neighbors in the topology
            for nb_label in &self.topo.get_node(&node_label).edges {
                let nb = self.topo.get_node(nb_label);
                // Check if neighbor is an unused bus node not in graph
                if nb.node_type == NodeType::Bus && !nb.used && !graph.contains_node(nb_label) {
                    log::info!("    Selected {} as {} ancilla", nb_label, which_ancilla);
                    // Add the node and edge to the graph
                    graph.add_node(nb.clone());
                    graph.add_edge(&node_label, nb_label);
                    return true;
                }
            }
        }
        false
    }

    fn gen_busy_count(&mut self) -> i32 {
        let count = self.rng_exp.sample().round() as i32 + 1;
        self.busy_count_list.push(count);
        count
    }

    fn check_dependencies(&self, pp: &PauliProduct, scheduled: &HashSet<i32>) -> io::Result<()> {
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
