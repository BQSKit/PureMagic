use crate::circuit::Circuit;
use crate::pauliproduct::PauliProduct;
use crate::topograph::{NodeType, TopoGraph};
use crate::utils::Timer;
use petgraph::graphmap::NodeTrait;
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};

pub struct Scheduler {
    circuit: Circuit,
    topo: TopoGraph,
    rng: StdRng,
    magic_state_lambda: f64,
    log_scheduler: bool,
    plot_option: String,
    used_nodes: HashSet<String>,
}

impl Scheduler {
    pub fn new(
        circuit: Circuit,
        topo: TopoGraph,
        rng: StdRng,
        magic_state_lambda: f64,
        log_scheduler: bool,
        plot_option: String,
    ) -> Self {
        Scheduler {
            circuit,
            topo,
            rng,
            magic_state_lambda,
            log_scheduler,
            plot_option,
            used_nodes: HashSet::new(),
        }
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
                    if self.log_scheduler {
                        println!("    Selecting {} as {}", nb_label, which_ancilla);
                    }
                    // Add the node and edge to the graph
                    graph.add_node_copied(nb.clone());
                    graph.add_edge(&node_label, nb_label);
                    return true;
                }
            }
        }
        false
    }

    fn schedule_pauli_product(&self, pauli_product: &PauliProduct) -> Option<TopoGraph> {
        if self.log_scheduler {
            println!("Trying to schedule {}", pauli_product);
        }
        // Initially terminal nodes contain only the data qubits
        let mut data_nodes = Vec::new();

        for op in &pauli_product.operators {
            let node_label = format!("d{}{}", op.qubit, op.basis.to_ascii_uppercase());
            // Check if node is already used
            if self.topo.get_node(&node_label).used {
                if self.log_scheduler {
                    println!("  Node {} is already used", node_label);
                }
                return None;
            }
            data_nodes.push(node_label);
        }

        if data_nodes.is_empty() {
            if self.log_scheduler {
                println!("  No data nodes found in working graph");
            }
            return None;
        }
        None
        /*
        if !pauli_product.is_clifford {
            self.schedule_non_clifford(&data_nodes, pauli_product)
        } else {
            self.schedule_clifford(&data_nodes, pauli_product)
        }
         */
    }
}
