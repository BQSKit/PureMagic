use crate::circuit::Circuit;
use crate::pauliproduct::PauliProduct;
use crate::topograph::{NodeType, TopoGraph};
use crate::utils::Timer;

use log;
use rand_simple::Exponential;
use simple_logging;
use std::collections::{HashSet, VecDeque};
use std::path::Path;

pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng_exp: Exponential,
    magic_state_lambda: f64,
    plot_option: String,
    used_nodes: HashSet<String>,
    sum_data_qubits: usize,
    sum_bus_qubits: usize,
    sum_magic_qubits: usize,
    sum_ancilla_qubits: usize,
    sum_estabilizer_qubits: usize,
    busy_count_list: Vec<i32>,
}

impl Scheduler {
    pub fn new(
        circuit: Circuit,
        topo: TopoGraph,
        magic_state_lambda: f64,
        log_scheduler: bool,
        plot_option: String,
    ) -> Self {
        if log_scheduler {
            let circuit_stem = Path::new(&circuit.circuit_fname)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("circuit");
            let sched_fname = format!("{}.sched", circuit_stem);
            simple_logging::log_to_file(&sched_fname, log::LevelFilter::Info)
                .expect("Failed to initialize logging");
        }
        Scheduler {
            circuit,
            topo,
            rng_exp: Exponential::new(29),
            magic_state_lambda,
            plot_option,
            used_nodes: HashSet::new(),
            sum_data_qubits: 0,
            sum_bus_qubits: 0,
            sum_magic_qubits: 0,
            sum_ancilla_qubits: 0,
            sum_estabilizer_qubits: 0,
            busy_count_list: Vec::new(),
        }
    }

    fn schedule_timestep(
        &mut self,
        step_i: usize,
        to_schedule: &[PauliProduct],
    ) -> (Option<String>, Option<Vec<(PauliProduct, TopoGraph)>>, Vec<PauliProduct>) {
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
                    log::info!(
                        "  * Scheduled with {} nodes and {} edges",
                        graph.num_nodes,
                        graph.num_edges
                    );
                    // Update node statistics and mark as used
                    for node in graph.iter_nodes() {
                        match node.node_type {
                            NodeType::Bus => num_bus_scheduled += 1,
                            NodeType::Magic => {
                                self.topo.get_node_mut(&node.label).busy_count =
                                    Some(self.gen_busy_count());
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
        log::info!(
            "  data:        {}/{} ({:.2})",
            num_data_scheduled,
            self.topo.num_data_qubits,
            frac_data
        );
        log::info!(
            "  bus:         {}/{} ({:.2})",
            num_bus_scheduled,
            self.topo.num_bus_qubits,
            frac_bus
        );
        log::info!(
            "  magic:       {}/{} ({:.2})",
            num_magic_scheduled,
            self.topo.num_magic_qubits,
            frac_magic
        );
        log::info!(
            "  estabilizer: {}/{} ({:.2})",
            num_estabilizers_scheduled,
            self.topo.num_estabilizer_qubits,
            frac_estabilizers
        );
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

            let title = format!(
                "Step {} Products scheduled {:.2}, data {:.2}, bus {:.2}",
                step_i, frac_paths, frac_data, frac_bus
            );
            (Some(title), Some(pp_paths), next_to_schedule)
        } else {
            (None, None, next_to_schedule)
        }
    }

    fn schedule_pauli_product(&self, pauli_product: &PauliProduct) -> Option<TopoGraph> {
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
            //
            None
        } else {
            self.schedule_clifford(&data_nodes, pauli_product)
        }
    }

    fn schedule_clifford(
        &self,
        data_nodes: &[String],
        pauli_product: &PauliProduct,
    ) -> Option<TopoGraph> {
        // Handle single data node case
        if data_nodes.len() == 1 {
            let node_label = &data_nodes[0];
            let node = self.topo.get_node(node_label);
            if node.used {
                return None;
            }

            let mut g = TopoGraph::new();
            g.add_node_copied(node.clone());

            if pauli_product.need_ancilla {
                // Try to find an available bus neighbor
                for nb_label in node.edges.iter() {
                    let nb = self.topo.get_node(nb_label);
                    if nb.node_type == NodeType::Bus && !nb.used {
                        g.add_node_copied(nb.clone());
                        g.add_edge(node_label, nb_label);
                        break;
                    }
                }
                return None;
            }
            log::info!(
                "Scheduled clifford on {:?} nodes",
                g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>()
            );
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
        let g = self.find_tree(&root_nodes, data_nodes, pauli_product);
        if let Some(ref g) = g {
            log::info!(
                "Scheduled clifford in {:?} nodes",
                g.iter_nodes().map(|n| &n.label).collect::<Vec<_>>()
            );
        }
        g
    }

    fn find_tree(
        &self,
        root_nodes: &HashSet<String>,
        data_nodes: &[String],
        pauli_product: &PauliProduct,
    ) -> Option<TopoGraph> {
        log::info!("  Find tree for {}:", pauli_product);
        // Determine which_ancilla based on first operator's basis
        let which_ancilla = if pauli_product.need_ancilla {
            pauli_product
                .operators
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
                    log::info!(
                        "    No tree from root node {} to {:?}, {}",
                        root_node_label,
                        data_nodes,
                        if which_ancilla.is_empty() { "" } else { &which_ancilla }
                    );
                    continue;
                }
                Some(graph) => {
                    log::info!(
                        "    Tree from {} to {:?} has size {}",
                        root_node_label,
                        data_nodes,
                        graph.num_edges,
                    );
                    return Some(graph);
                }
            }
        }
        None
    }

    fn get_bfs_graph(
        &self,
        root_node: &str,
        terminal_nodes: &[String],
        which_ancilla: &str,
        exclude: Option<&TopoGraph>,
    ) -> Option<TopoGraph> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut bfs_graph = TopoGraph::new();

        visited.insert(root_node.to_string());
        queue.push_back(root_node.to_string());
        let num_terminals_reqd = terminal_nodes.len();
        let mut num_found_terminals = 0;

        bfs_graph.add_node_copied(self.topo.get_node(root_node).clone());

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
                bfs_graph.add_node_copied(nb.clone());
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
        let bus_nodes: Vec<_> = graph
            .iter_nodes()
            .filter(|node| node.node_type == NodeType::Bus)
            .map(|node| node.label.clone())
            .collect();

        for node_label in bus_nodes {
            // Check neighbors in the topology
            for nb_label in &self.topo.get_node(&node_label).edges {
                let nb = self.topo.get_node(nb_label);
                // Check if neighbor is an unused bus node not in graph
                if nb.node_type == NodeType::Bus && !nb.used && !graph.contains_node(nb_label) {
                    // FIXME: do we need left and right for X and Y, given we can create the
                    // ancilla on the fly?
                    log::info!("    Selecting {} as {}", nb_label, which_ancilla);
                    // Add the node and edge to the graph
                    graph.add_node_copied(nb.clone());
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
}
