use crate::debug_sched;
use crate::node::{Node, NodeType};
#[allow(unused_imports)]
use crate::utils::{_BLUE, _RESET};

/// Internal node representation for a tree subgraph.
/// Stores neighbor adjacencies, node type classification, and position for layout queries.
#[derive(Debug, Clone)]
struct TreeNode {
    pub nbors: Vec<u16>,
    pub is_routing: bool,
    pub is_data: bool,
    // position is needed so that we can determine which links are top or bottom
    pub pos: (f32, f32),
    #[cfg(debug_assertions)]
    pub label: String,
}

impl TreeNode {
    /// Creates a tree node from a topology node.
    pub fn new(node: &Node) -> Self {
        TreeNode { nbors: Vec::new(),
                   is_routing: node.is_routing(),
                   is_data: node.node_type == NodeType::Data,
                   pos: node.pos,
                   #[cfg(debug_assertions)]
                   label: node.label.clone() }
    }

    /// Removes an edge to a neighbor node.
    pub fn remove_edge(&mut self, nb_id: u16) {
        // FIXME: this could be inefficient
        let pos = self.nbors.iter().position(|&id| id == nb_id).unwrap();
        self.nbors.swap_remove(pos);
    }
}

/// A sparse tree subgraph of the topology containing scheduled Pauli product routing.
/// Nodes are sparse (only included nodes present); root marks the magic cultivator.
#[derive(Debug, Clone)]
pub struct TreeGraph {
    // if a node is included in the graph, then it is a vector of its neighbors
    nodes: Vec<Option<TreeNode>>,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub root_node_id: Option<u16>,
}

impl TreeGraph {
    /// Creates a new empty tree graph with capacity for `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Self {
        TreeGraph { nodes: vec![None; num_nodes],
                    num_edges: 0,
                    num_nodes: num_nodes,
                    root_node_id: None }
    }

    /// Returns an iterator over node IDs in the tree.
    pub fn iter_nodes(&self) -> impl Iterator<Item = u16> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i as u16))
    }

    /// Checks if a node exists in the tree.
    pub fn contains_node(&self, id: u16) -> bool {
        self.nodes[id as usize].is_some()
    }

    /// Checks if an undirected edge exists between two nodes.
    pub fn contains_edge(&self, node_id1: u16, node_id2: u16) -> bool {
        if let Some(node) = &self.nodes[node_id1 as usize] {
            node.nbors.contains(&node_id2)
        } else {
            false
        }
    }

    /// Returns the degree of a node (debug builds only).
    #[cfg(debug_assertions)]
    pub fn get_num_node_edges(&self, node_id: u16) -> usize {
        self.nodes[node_id as usize].as_ref().map(|node| node.nbors.len()).unwrap_or(0)
    }

    /// Adds a node to the tree from topology node data.
    pub fn add_node(&mut self, node: &Node) {
        assert!(self.nodes[node.id as usize].is_none());
        self.nodes[node.id as usize] = Some(TreeNode::new(node));
        self.num_nodes += 1;
        debug_sched!("      {}add node {}{}", _BLUE, node.label, _RESET);
    }

    /// Adds an undirected edge between two existing nodes.
    pub fn add_edge(&mut self, node_id1: u16, node_id2: u16) {
        // make sure the edge doesn't already exist
        #[cfg(debug_assertions)]
        {
            let node1 = self.nodes[node_id1 as usize].as_ref().unwrap();
            let node2 = self.nodes[node_id2 as usize].as_ref().unwrap();
            debug_assert!(!node1.nbors.contains(&node_id2));
            debug_assert!(!node2.nbors.contains(&node_id1));
            debug_sched!("      {}add edge {}->{}{}", _BLUE, node1.label, node2.label, _RESET);
        }
        self.nodes[node_id1 as usize].as_mut().unwrap().nbors.push(node_id2);
        self.nodes[node_id2 as usize].as_mut().unwrap().nbors.push(node_id1);
        self.num_edges += 1;
    }

    /// Removes routing nodes with degree ≤ 1 (except root) until none remain.
    /// Returns the number of nodes trimmed.
    pub fn trim_dangling_nodes(&mut self) -> usize {
        let mut num_trimmed = 0;
        let root_id = self.root_node_id.unwrap();
        loop {
            // Find dangling bus nodes
            let mut dangling_ids: Vec<u16> = Vec::new();
            for (node_id, node_opt) in self.nodes.iter().enumerate() {
                if let Some(node) = node_opt {
                    if node.is_routing && node.nbors.len() <= 1 && root_id != node_id as u16 {
                        dangling_ids.push(node_id as u16);
                    }
                }
            }
            // Remove dangling nodes if any found
            if dangling_ids.is_empty() {
                break;
            } else {
                for id in dangling_ids {
                    self.remove_node(id);
                    num_trimmed += 1;
                }
            }
        }
        num_trimmed
    }

    /// Removes a node and all its edges from the tree.
    fn remove_node(&mut self, node_id: u16) {
        let node = self.nodes[node_id as usize].as_ref().unwrap();
        let nb_ids: Vec<u16> = node.nbors.iter().copied().collect();
        #[cfg(debug_assertions)]
        {
            debug_sched!("      {}remove node {}{}", _BLUE, node.label, _RESET);
            for nb_id in &nb_ids {
                debug_sched!("      {}remove edge {}->{}{}",
                             _BLUE,
                             self.nodes[*nb_id as usize].as_ref().unwrap().label,
                             node.label,
                             _RESET);
            }
        }
        // Remove edges from neighbor nodes
        for nb_id in nb_ids {
            self.nodes[nb_id as usize].as_mut().unwrap().remove_edge(node_id);
            self.num_edges -= 1;
        }
        // Remove the node itself
        self.nodes[node_id as usize] = None;
        self.num_nodes -= 1;
    }

    /// For data nodes with two edges (one side, one vertical), removes the weaker edge.
    /// Prefers keeping vertical edges if they are unique connections.
    pub fn remove_double_edges(&mut self) {
        let mut edges_to_remove: Vec<(u16, u16)> = Vec::new();
        for (node_id, node_opt) in self.nodes.iter().enumerate() {
            if let Some(node) = node_opt {
                if node.is_data && node.nbors.len() == 2 {
                    let (side_nb_id, vert_nb_id) = {
                        if node.pos.1 == self.nodes[node.nbors[0] as usize].as_ref().unwrap().pos.1
                        {
                            (node.nbors[0], node.nbors[1])
                        } else {
                            (node.nbors[1], node.nbors[0])
                        }
                    };
                    debug_sched!("      {}Found node {} with two edges{}",
                                 _BLUE,
                                 node.label,
                                 _RESET);
                    let vert_nb = self.nodes[vert_nb_id as usize].as_ref().unwrap();
                    if node.pos.1 < vert_nb.pos.1 && self.get_below_edge_count(vert_nb) == 1 {
                        debug_sched!("      {}removing single below edge {}->{}{}",
                                     _BLUE,
                                     node.label,
                                     vert_nb.label,
                                     _RESET);
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else if node.pos.1 > vert_nb.pos.1 && self.get_above_edge_count(vert_nb) == 1
                    {
                        debug_sched!("      {}removing single above edge {}->{}{}",
                                     _BLUE,
                                     node.label,
                                     vert_nb.label,
                                     _RESET);
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else {
                        debug_sched!("      {}removing extra side edge {}->{}{}",
                                     _BLUE,
                                     node.label,
                                     self.nodes[side_nb_id as usize].as_ref().unwrap().label,
                                     _RESET);
                        edges_to_remove.push((node_id as u16, side_nb_id));
                    }
                }
            }
        }
        for (node_id1, node_id2) in edges_to_remove {
            self.nodes[node_id1 as usize].as_mut().unwrap().remove_edge(node_id2);
            self.nodes[node_id2 as usize].as_mut().unwrap().remove_edge(node_id1);
            self.num_edges -= 1;
        }
    }

    /// Counts edges pointing upward (higher y position) from a node.
    fn get_above_edge_count(&self, node: &TreeNode) -> usize {
        node.nbors
            .iter()
            .filter(|nb_id| {
                let nb = self.nodes[**nb_id as usize].as_ref().unwrap();
                node.pos.1 < nb.pos.1
            })
            .count()
    }

    /// Counts edges pointing downward (lower y position) from a node.
    fn get_below_edge_count(&self, node: &TreeNode) -> usize {
        node.nbors
            .iter()
            .filter(|nb_id| {
                let nb = self.nodes[**nb_id as usize].as_ref().unwrap();
                node.pos.1 > nb.pos.1
            })
            .count()
    }
}
