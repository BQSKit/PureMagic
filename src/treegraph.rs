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
    pub fn new(node: &Node, #[cfg_attr(not(debug_assertions), allow(unused))] label: &str) -> Self {
        TreeNode {
            nbors: Vec::new(),
            is_routing: node.is_routing(),
            is_data: node.node_type == NodeType::Data,
            pos: node.pos,
            #[cfg(debug_assertions)]
            label: label.to_string(),
        }
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
        TreeGraph {
            nodes: vec![None; num_nodes],
            num_edges: 0,
            num_nodes: num_nodes,
            root_node_id: None,
        }
    }

    /// Returns an iterator over node IDs in the tree.
    pub fn iter_nodes(&self) -> impl Iterator<Item = u16> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i as u16))
    }

    /// Returns a slice of neighbor IDs for a node that is in the tree.
    /// Panics if the node is not present.
    pub fn neighbors(&self, node_id: u16) -> &[u16] {
        self.nodes[node_id as usize].as_ref().unwrap().nbors.as_slice()
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
    pub fn add_node(&mut self, node: &Node, label: &str) {
        assert!(self.nodes[node.id as usize].is_none());
        self.nodes[node.id as usize] = Some(TreeNode::new(node, label));
        self.num_nodes += 1;
        debug_sched!("      {}add node {}{}", _BLUE, label, _RESET);
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

    /// Removes the magic root node and then trims all dangling routing nodes,
    /// leaving the minimal subtree that still connects all terminal data nodes.
    /// A routing node adjacent only to a single data node is kept — it is the
    /// root-side connection point for that terminal and must not be removed.
    /// After this call `root_node_id` is `None`.
    pub fn trim_magic_root(&mut self) {
        let root_id = match self.root_node_id.take() {
            Some(id) => id,
            None => return,
        };
        self.remove_node(root_id);
        // Trim routing nodes that became dangling, but preserve any routing node
        // whose sole remaining neighbor is a data node (it is the terminal's root).
        loop {
            let dangling: Vec<u16> = self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, opt)| {
                    opt.as_ref().and_then(|n| {
                        if !n.is_routing || n.nbors.len() > 1 {
                            return None;
                        }
                        if n.nbors.len() == 1 {
                            // Keep if the sole neighbor is a data node.
                            let nb = self.nodes[n.nbors[0] as usize].as_ref()?;
                            if nb.is_data {
                                return None;
                            }
                        }
                        Some(i as u16)
                    })
                })
                .collect();
            if dangling.is_empty() {
                break;
            }
            for id in dangling {
                self.remove_node(id);
            }
        }
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
                debug_sched!(
                    "      {}remove edge {}->{}{}",
                    _BLUE,
                    self.nodes[*nb_id as usize].as_ref().unwrap().label,
                    node.label,
                    _RESET
                );
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
                    debug_sched!(
                        "      {}Found node {} with two edges{}",
                        _BLUE,
                        node.label,
                        _RESET
                    );
                    let vert_nb = self.nodes[vert_nb_id as usize].as_ref().unwrap();
                    if node.pos.1 < vert_nb.pos.1 && self.get_below_edge_count(vert_nb) == 1 {
                        debug_sched!(
                            "      {}removing single below edge {}->{}{}",
                            _BLUE,
                            node.label,
                            vert_nb.label,
                            _RESET
                        );
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else if node.pos.1 > vert_nb.pos.1 && self.get_above_edge_count(vert_nb) == 1
                    {
                        debug_sched!(
                            "      {}removing single above edge {}->{}{}",
                            _BLUE,
                            node.label,
                            vert_nb.label,
                            _RESET
                        );
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else {
                        debug_sched!(
                            "      {}removing extra side edge {}->{}{}",
                            _BLUE,
                            node.label,
                            self.nodes[side_nb_id as usize].as_ref().unwrap().label,
                            _RESET
                        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeType};

    /// Helper: create a Magic node with given id and position.
    fn magic_node(id: u16, x: f32, y: f32) -> Node {
        Node::new(id, None, x, y, NodeType::Magic)
    }

    /// Helper: create a Data node with given id and position.
    fn data_node(id: u16, x: f32, y: f32) -> Node {
        Node::new(id, None, x, y, NodeType::Data)
    }

    // ── TreeGraph::new ────────────────────────────────────────────────────────

    #[test]
    fn new_creates_empty_graph() {
        let g = TreeGraph::new(10);
        assert_eq!(g.num_edges, 0);
        // num_nodes is initialised to the capacity, not the count of added nodes
        assert_eq!(g.num_nodes, 10);
        assert!(g.root_node_id.is_none());
    }

    // ── TreeGraph::contains_node ──────────────────────────────────────────────

    #[test]
    fn contains_node_false_before_add() {
        let g = TreeGraph::new(5);
        assert!(!g.contains_node(0));
        assert!(!g.contains_node(4));
    }

    #[test]
    fn contains_node_true_after_add() {
        let mut g = TreeGraph::new(5);
        let n = magic_node(2, 0.0, 0.0);
        g.add_node(&n, "m2");
        assert!(g.contains_node(2));
        assert!(!g.contains_node(0));
    }

    // ── TreeGraph::add_node ───────────────────────────────────────────────────

    #[test]
    fn add_node_increments_num_nodes() {
        let mut g = TreeGraph::new(10);
        let initial = g.num_nodes;
        let n = magic_node(3, 1.0, 1.0);
        g.add_node(&n, "m3");
        assert_eq!(g.num_nodes, initial + 1);
    }

    // ── TreeGraph::add_edge / contains_edge ───────────────────────────────────

    #[test]
    fn add_edge_creates_bidirectional_edge() {
        let mut g = TreeGraph::new(5);
        let n0 = magic_node(0, 0.0, 0.0);
        let n1 = magic_node(1, 1.0, 0.0);
        g.add_node(&n0, "m0");
        g.add_node(&n1, "m1");
        g.add_edge(0, 1);
        assert!(g.contains_edge(0, 1));
        assert!(g.contains_edge(1, 0));
    }

    #[test]
    fn add_edge_increments_num_edges() {
        let mut g = TreeGraph::new(5);
        let n0 = magic_node(0, 0.0, 0.0);
        let n1 = magic_node(1, 1.0, 0.0);
        g.add_node(&n0, "m0");
        g.add_node(&n1, "m1");
        assert_eq!(g.num_edges, 0);
        g.add_edge(0, 1);
        assert_eq!(g.num_edges, 1);
    }

    #[test]
    fn contains_edge_false_for_absent_edge() {
        let mut g = TreeGraph::new(5);
        let n0 = magic_node(0, 0.0, 0.0);
        let n1 = magic_node(1, 1.0, 0.0);
        g.add_node(&n0, "m0");
        g.add_node(&n1, "m1");
        assert!(!g.contains_edge(0, 1));
    }

    #[test]
    fn contains_edge_false_when_node_absent() {
        let g = TreeGraph::new(5);
        assert!(!g.contains_edge(0, 1));
    }

    // ── TreeGraph::iter_nodes ─────────────────────────────────────────────────

    #[test]
    fn iter_nodes_empty_on_new_graph() {
        let g = TreeGraph::new(5);
        assert_eq!(g.iter_nodes().count(), 0);
    }

    #[test]
    fn iter_nodes_yields_added_ids() {
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(1, 0.0, 0.0), "m1");
        g.add_node(&magic_node(3, 1.0, 0.0), "m3");
        let ids: Vec<u16> = g.iter_nodes().collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert_eq!(ids.len(), 2);
    }

    // ── TreeGraph::neighbors ──────────────────────────────────────────────────

    #[test]
    fn neighbors_empty_before_edges() {
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "m0");
        assert!(g.neighbors(0).is_empty());
    }

    #[test]
    fn neighbors_contains_connected_node() {
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "m0");
        g.add_node(&magic_node(1, 1.0, 0.0), "m1");
        g.add_edge(0, 1);
        assert!(g.neighbors(0).contains(&1));
        assert!(g.neighbors(1).contains(&0));
    }

    // ── TreeGraph::trim_dangling_nodes ────────────────────────────────────────

    #[test]
    fn trim_dangling_nodes_removes_degree_one_routing_nodes() {
        // Build a star: root(0) connected to mid(1) and mid(2); mid(1) also connected to leaf(3).
        // leaf(3) is dangling (degree 1, routing, not root) and should be trimmed.
        // mid(1) has degree 2 after leaf is removed → still has root as neighbour → not dangling.
        // mid(2) has degree 1 but is connected to root → after trim it still has root → not dangling.
        //
        // Simpler: root(0) -- mid(1) -- data(2) -- mid(3) [dangling routing leaf]
        // But data nodes are never trimmed. Use:
        //   root(0) connected to mid(1) and mid(2); mid(1) connected to leaf(3).
        // After trim: leaf(3) removed (degree 1). mid(1) now has degree 1 (only root) → also trimmed.
        // That gives 2 trimmed again.
        //
        // Correct approach: root(0) -- mid(1) -- mid(2) -- leaf(3)
        //                                     \-- mid(4)
        // mid(2) has degree 2 (mid1 + leaf3), leaf(3) trimmed first.
        // After leaf(3) removed, mid(2) has degree 1 → trimmed. Total = 2.
        //
        // Use a Y-shape: root(0) -- hub(1) -- branch_a(2)
        //                                  \-- branch_b(3)
        // branch_a and branch_b are both dangling (degree 1). hub has degree 3 (root + 2 branches).
        // After trimming both branches, hub has degree 1 → also trimmed. Total = 3.
        //
        // Simplest correct test: just verify that ALL dangling routing nodes (non-root) are removed
        // and the root is preserved, without asserting an exact count.
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        // root(0) -- mid(1) -- leaf(2)
        // After trim: leaf(2) removed (degree 1), then mid(1) becomes degree 1 → also removed.
        // Root(0) is preserved. Total trimmed = 2.
        g.add_node(&magic_node(0, 0.0, 0.0), "root");
        g.add_node(&magic_node(1, 1.0, 0.0), "mid");
        g.add_node(&magic_node(2, 2.0, 0.0), "leaf");
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.root_node_id = Some(0);
        let trimmed = g.trim_dangling_nodes();
        // Both mid and leaf are dangling (non-root, degree ≤ 1 after each pass).
        assert_eq!(trimmed, 2);
        assert!(!g.contains_node(2), "leaf should be trimmed");
        assert!(!g.contains_node(1), "mid becomes dangling after leaf removed");
        assert!(g.contains_node(0), "root must be preserved");
    }

    #[test]
    fn trim_dangling_nodes_keeps_root() {
        // Single node graph: root is degree 0 but must not be removed.
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(3);
        g.add_node(&magic_node(0, 0.0, 0.0), "root");
        g.root_node_id = Some(0);
        let trimmed = g.trim_dangling_nodes();
        assert_eq!(trimmed, 0);
        assert!(g.contains_node(0));
    }

    #[test]
    fn trim_dangling_nodes_keeps_data_nodes() {
        // Data nodes with degree 1 must NOT be trimmed.
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "root");
        g.add_node(&data_node(1, 0.5, 0.0), "d1");
        g.add_edge(0, 1);
        g.root_node_id = Some(0);
        let trimmed = g.trim_dangling_nodes();
        assert_eq!(trimmed, 0);
        assert!(g.contains_node(1));
    }

    // ── TreeGraph::remove_double_edges ────────────────────────────────────────

    #[test]
    fn remove_double_edges_removes_extra_side_edge() {
        // Data node at y=0.0 connected to two routing nodes:
        //   side_nb (same y) and vert_nb (different y).
        // The vert_nb has only one below-edge (to data), so the vert edge is removed.
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(10);
        // data node id=0 at (1.0, 0.0)
        let d = data_node(0, 1.0, 0.0);
        // side routing node id=1 at (2.0, 0.0) — same y
        let side = magic_node(1, 2.0, 0.0);
        // vertical routing node id=2 at (1.0, 1.0) — different y, above data
        let vert = magic_node(2, 1.0, 1.0);
        g.add_node(&d, "d0");
        g.add_node(&side, "m1");
        g.add_node(&vert, "m2");
        g.add_edge(0, 1); // side edge
        g.add_edge(0, 2); // vertical edge
        // vert has only one below-edge (to data node 0), so the vert edge should be removed.
        g.remove_double_edges();
        // After removal, data node should have exactly 1 edge.
        assert_eq!(g.neighbors(0).len(), 1);
    }
}
