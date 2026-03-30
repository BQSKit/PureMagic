use crate::debug_sched;
use crate::node::{Node, NodeType};
#[allow(unused_imports)]
use colored::Colorize;

/// Internal node representation for a tree subgraph.
#[derive(Debug, Clone)]
struct TreeNode {
    pub nbs: Vec<u16>,
    pub is_routing: bool,
    pub is_data: bool,
    pub pos: (f32, f32),
    #[cfg(debug_assertions)]
    pub label: String,
}

impl TreeNode {
    pub(crate) fn new(
        node: &Node, #[cfg_attr(not(debug_assertions), allow(unused))] label: &str,
    ) -> Self {
        TreeNode {
            nbs: Vec::new(),
            is_routing: node.is_routing(),
            is_data: node.node_type == NodeType::Data,
            pos: node.pos,
            #[cfg(debug_assertions)]
            label: label.to_string(),
        }
    }

    pub(crate) fn remove_edge(&mut self, nb_id: u16) {
        let pos = self.nbs.iter().position(|&id| id == nb_id).unwrap();
        self.nbs.swap_remove(pos);
    }
}

/// A sparse tree subgraph of the topology containing scheduled Pauli product routing.
/// Only nodes that are part of the routing path are present (`Some`); all others are `None`.
/// `root_node_id` is the magic cultivator node for T gates, or the first root for Cliffords.
#[derive(Debug, Clone)]
pub(crate) struct TreeGraph {
    nodes: Vec<Option<TreeNode>>,
    pub n_edges: usize,
    /// Capacity (total topology nodes), not the count of nodes actually in the tree.
    pub n_nodes: usize,
    pub root_node_id: Option<u16>,
}

impl TreeGraph {
    pub(crate) fn new(n_nodes: usize) -> Self {
        TreeGraph {
            nodes: vec![None; n_nodes],
            n_edges: 0,
            n_nodes: n_nodes,
            root_node_id: None,
        }
    }

    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = u16> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i as u16))
    }

    pub(crate) fn nbs(&self, node_id: u16) -> &[u16] {
        self.nodes[node_id as usize].as_ref().unwrap().nbs.as_slice()
    }

    pub(crate) fn contains_node(&self, id: u16) -> bool {
        self.nodes[id as usize].is_some()
    }

    pub(crate) fn contains_edge(&self, node_id1: u16, node_id2: u16) -> bool {
        if let Some(node) = &self.nodes[node_id1 as usize] {
            node.nbs.contains(&node_id2)
        } else {
            false
        }
    }

    #[cfg(debug_assertions)]
    pub(crate) fn n_node_edges(&self, node_id: u16) -> usize {
        self.nodes[node_id as usize].as_ref().map(|node| node.nbs.len()).unwrap_or(0)
    }

    pub(crate) fn add_node(&mut self, node: &Node, label: &str) {
        assert!(self.nodes[node.id as usize].is_none(), "node {} already in tree", node.id);
        self.nodes[node.id as usize] = Some(TreeNode::new(node, label));
        self.n_nodes += 1;
        debug_sched!("      {}", format!("add node {}", label).blue());
    }

    pub(crate) fn add_edge(&mut self, node_id1: u16, node_id2: u16) {
        #[cfg(debug_assertions)]
        {
            let node1 = self.nodes[node_id1 as usize].as_ref().unwrap();
            let node2 = self.nodes[node_id2 as usize].as_ref().unwrap();
            debug_assert!(!node1.nbs.contains(&node_id2));
            debug_assert!(!node2.nbs.contains(&node_id1));
            debug_sched!("      {}", format!("add edge {}->{}", node1.label, node2.label).blue());
        }
        self.nodes[node_id1 as usize].as_mut().unwrap().nbs.push(node_id2);
        self.nodes[node_id2 as usize].as_mut().unwrap().nbs.push(node_id1);
        self.n_edges += 1;
    }

    /// Removes the magic root and trims dangling routing nodes.
    /// Routing nodes whose sole remaining nb is a data node are preserved.
    pub(crate) fn trim_magic_root(&mut self) {
        let root_id = match self.root_node_id.take() {
            Some(id) => id,
            None => return,
        };
        self.remove_node(root_id);
        loop {
            let dangling: Vec<u16> = self
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(i, opt)| {
                    opt.as_ref().and_then(|n| {
                        if !n.is_routing || n.nbs.len() > 1 {
                            return None;
                        }
                        if n.nbs.len() == 1 {
                            let nb = self.nodes[n.nbs[0] as usize].as_ref()?;
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
    pub(crate) fn trim_dangling_nodes(&mut self) -> usize {
        let mut n_trimmed = 0;
        let root_id = self.root_node_id.unwrap();
        loop {
            let mut dangling_ids: Vec<u16> = Vec::new();
            for (node_id, node_opt) in self.nodes.iter().enumerate() {
                if let Some(node) = node_opt {
                    if node.is_routing && node.nbs.len() <= 1 && root_id != node_id as u16 {
                        dangling_ids.push(node_id as u16);
                    }
                }
            }
            if dangling_ids.is_empty() {
                break;
            } else {
                for id in dangling_ids {
                    self.remove_node(id);
                    n_trimmed += 1;
                }
            }
        }
        n_trimmed
    }

    fn remove_node(&mut self, node_id: u16) {
        let node = self.nodes[node_id as usize].as_ref().unwrap();
        let nb_ids: Vec<u16> = node.nbs.iter().copied().collect();
        #[cfg(debug_assertions)]
        {
            debug_sched!("      {}", format!("remove node {}", node.label).blue());
            for nb_id in &nb_ids {
                debug_sched!(
                    "      {}",
                    format!(
                        "remove edge {}->{}",
                        self.nodes[*nb_id as usize].as_ref().unwrap().label,
                        node.label
                    )
                    .blue()
                );
            }
        }
        for nb_id in nb_ids {
            self.nodes[nb_id as usize].as_mut().unwrap().remove_edge(node_id);
            self.n_edges -= 1;
        }
        self.nodes[node_id as usize] = None;
        self.n_nodes -= 1;
    }

    /// Resolves data nodes that ended up with two edges (one side, one vertical).
    ///
    /// This can happen during `init_bfs_from_roots` when two adjacent roots both
    /// connect to the same terminal data node. The weaker edge is removed:
    /// - If the vertical nb has only one below/above edge (to this data node),
    ///   the vertical edge is removed (the routing node is not needed vertically).
    /// - Otherwise the side edge is removed.
    pub(crate) fn remove_double_edges(&mut self) {
        let mut edges_to_remove: Vec<(u16, u16)> = Vec::new();
        for (node_id, node_opt) in self.nodes.iter().enumerate() {
            if let Some(node) = node_opt {
                if node.is_data && node.nbs.len() == 2 {
                    let (side_nb_id, vert_nb_id) = {
                        if node.pos.1 == self.nodes[node.nbs[0] as usize].as_ref().unwrap().pos.1
                        {
                            (node.nbs[0], node.nbs[1])
                        } else {
                            (node.nbs[1], node.nbs[0])
                        }
                    };
                    debug_sched!(
                        "      {}",
                        format!("Found node {} with two edges", node.label).blue()
                    );
                    let vert_nb = self.nodes[vert_nb_id as usize].as_ref().unwrap();
                    if node.pos.1 < vert_nb.pos.1
                        && self.get_horizontal_edge_count(vert_nb, false) == 1
                    {
                        debug_sched!(
                            "      {}",
                            format!("removing single below edge {}->{}", node.label, vert_nb.label)
                                .blue()
                        );
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else if node.pos.1 > vert_nb.pos.1
                        && self.get_horizontal_edge_count(vert_nb, true) == 1
                    {
                        debug_sched!(
                            "      {}",
                            format!("removing single above edge {}->{}", node.label, vert_nb.label)
                                .blue()
                        );
                        edges_to_remove.push((node_id as u16, vert_nb_id));
                    } else {
                        debug_sched!(
                            "      {}",
                            format!(
                                "removing extra side edge {}->{}",
                                node.label,
                                self.nodes[side_nb_id as usize].as_ref().unwrap().label
                            )
                            .blue()
                        );
                        edges_to_remove.push((node_id as u16, side_nb_id));
                    }
                }
            }
        }
        for (node_id1, node_id2) in edges_to_remove {
            self.nodes[node_id1 as usize].as_mut().unwrap().remove_edge(node_id2);
            self.nodes[node_id2 as usize].as_mut().unwrap().remove_edge(node_id1);
            self.n_edges -= 1;
        }
    }

    /// Counts vertical nbs of `node` in one direction.
    /// `upward=true` counts nbs with higher y (above); `false` counts lower y (below).
    fn get_horizontal_edge_count(&self, node: &TreeNode, upward: bool) -> usize {
        node.nbs
            .iter()
            .filter(|nb_id| {
                let nb = self.nodes[**nb_id as usize].as_ref().unwrap();
                if upward { node.pos.1 < nb.pos.1 } else { node.pos.1 > nb.pos.1 }
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeType};

    fn magic_node(id: u16, x: f32, y: f32) -> Node {
        Node::new(id, None, x, y, NodeType::Magic)
    }

    fn data_node(id: u16, x: f32, y: f32) -> Node {
        Node::new(id, None, x, y, NodeType::Data)
    }

    #[test]
    fn new_creates_empty_graph() {
        let g = TreeGraph::new(10);
        assert_eq!(g.n_edges, 0);
        assert_eq!(g.n_nodes, 10);
        assert!(g.root_node_id.is_none());
    }

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

    #[test]
    fn add_node_increments_n_nodes() {
        let mut g = TreeGraph::new(10);
        let initial = g.n_nodes;
        let n = magic_node(3, 1.0, 1.0);
        g.add_node(&n, "m3");
        assert_eq!(g.n_nodes, initial + 1);
    }

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
    fn add_edge_increments_n_edges() {
        let mut g = TreeGraph::new(5);
        let n0 = magic_node(0, 0.0, 0.0);
        let n1 = magic_node(1, 1.0, 0.0);
        g.add_node(&n0, "m0");
        g.add_node(&n1, "m1");
        assert_eq!(g.n_edges, 0);
        g.add_edge(0, 1);
        assert_eq!(g.n_edges, 1);
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

    #[test]
    fn nbs_empty_before_edges() {
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "m0");
        assert!(g.nbs(0).is_empty());
    }

    #[test]
    fn nbs_contains_connected_node() {
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "m0");
        g.add_node(&magic_node(1, 1.0, 0.0), "m1");
        g.add_edge(0, 1);
        assert!(g.nbs(0).contains(&1));
        assert!(g.nbs(1).contains(&0));
    }

    #[test]
    fn trim_dangling_nodes_removes_degree_one_routing_nodes() {
        // root(0) -- mid(1) -- leaf(2): leaf then mid become dangling, total trimmed = 2.
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        g.add_node(&magic_node(0, 0.0, 0.0), "root");
        g.add_node(&magic_node(1, 1.0, 0.0), "mid");
        g.add_node(&magic_node(2, 2.0, 0.0), "leaf");
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.root_node_id = Some(0);
        let trimmed = g.trim_dangling_nodes();
        assert_eq!(trimmed, 2);
        assert!(!g.contains_node(2));
        assert!(!g.contains_node(1));
        assert!(g.contains_node(0));
    }

    #[test]
    fn trim_dangling_nodes_keeps_root() {
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

    #[test]
    fn remove_double_edges_removes_extra_side_edge() {
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(10);
        let d = data_node(0, 1.0, 0.0);
        let side = magic_node(1, 2.0, 0.0);
        let vert = magic_node(2, 1.0, 1.0);
        g.add_node(&d, "d0");
        g.add_node(&side, "m1");
        g.add_node(&vert, "m2");
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.remove_double_edges();
        assert_eq!(g.nbs(0).len(), 1);
    }

    #[test]
    fn trim_magic_root_removes_root_when_set() {
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        let m0 = magic_node(0, 0.0, 0.0);
        let m1 = magic_node(1, 1.0, 0.0);
        let d2 = data_node(2, 2.0, 0.0);
        g.add_node(&m0, "m0");
        g.add_node(&m1, "m1");
        g.add_node(&d2, "d2");
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.root_node_id = Some(0);
        g.trim_magic_root();
        assert!(!g.contains_node(0));
        assert!(g.contains_node(1));
        assert!(g.contains_node(2));
        assert!(g.root_node_id.is_none());
    }

    #[test]
    fn trim_magic_root_noop_when_no_root() {
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        let m0 = magic_node(0, 0.0, 0.0);
        g.add_node(&m0, "m0");
        g.root_node_id = None;
        let before = g.n_nodes;
        g.trim_magic_root();
        assert_eq!(g.n_nodes, before);
    }

    #[test]
    fn n_edges_increments_on_add_edge() {
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(5);
        let m0 = magic_node(0, 0.0, 0.0);
        let m1 = magic_node(1, 1.0, 0.0);
        let m2 = magic_node(2, 2.0, 0.0);
        g.add_node(&m0, "m0");
        g.add_node(&m1, "m1");
        g.add_node(&m2, "m2");
        assert_eq!(g.n_edges, 0);
        g.add_edge(0, 1);
        assert_eq!(g.n_edges, 1);
        g.add_edge(1, 2);
        assert_eq!(g.n_edges, 2);
    }

    #[test]
    fn trim_dangling_nodes_chain_removes_all_routing() {
        Node::set_magic_routing(true);
        let mut g = TreeGraph::new(10);
        let d = data_node(0, 0.0, 0.0);
        let r1 = magic_node(1, 1.0, 0.0);
        let r2 = magic_node(2, 2.0, 0.0);
        let r3 = magic_node(3, 3.0, 0.0);
        g.add_node(&d, "d0");
        g.add_node(&r1, "r1");
        g.add_node(&r2, "r2");
        g.add_node(&r3, "r3");
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.root_node_id = Some(0);
        g.trim_dangling_nodes();
        assert!(g.contains_node(0));
        assert!(!g.contains_node(1));
        assert!(!g.contains_node(2));
        assert!(!g.contains_node(3));
    }
}
