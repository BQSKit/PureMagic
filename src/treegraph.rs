use crate::debug_sched;
use crate::utils::{_BLUE, _RESET};

#[derive(Clone)]
struct TreeNode {
    pub nbors: Vec<usize>,
    pub is_routing: bool,
    // position is needed so that we can determine which links are top or bottom
    pub pos: (f32, f32),
}

impl TreeNode {
    pub fn new(is_routing: bool, pos: (f32, f32)) -> Self {
        TreeNode { nbors: Vec::new(), is_routing: is_routing, pos: pos }
    }

    pub fn remove_edge(&mut self, nb_id: usize) {
        // FIXME: this could be inefficient
        let pos = self.nbors.iter().position(|&id| id == nb_id).unwrap();
        self.nbors.swap_remove(pos);
    }
}

pub struct TreeGraph {
    // if a node is included in the graph, then it is a vector of its neighbors
    nodes: Vec<Option<TreeNode>>,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub root_node_id: Option<usize>,
}

impl TreeGraph {
    pub fn new(num_nodes: usize) -> Self {
        TreeGraph { nodes: vec![None; num_nodes],
                    num_edges: 0,
                    num_nodes: num_nodes,
                    root_node_id: None }
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = usize> {
        self.nodes.iter().enumerate().filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i))
    }

    pub fn contains_node(&self, id: usize) -> bool {
        self.nodes[id].is_some()
    }

    pub fn contains_edge(&self, node_id1: usize, node_id2: usize) -> bool {
        if let Some(node) = &self.nodes[node_id1] { node.nbors.contains(&node_id2) } else { false }
    }

    pub fn get_num_node_edges(&self, node_id: usize) -> usize {
        self.nodes[node_id].as_ref().map(|node| node.nbors.len()).unwrap_or(0)
    }

    pub fn add_node(&mut self, id: usize, is_routing: bool, pos: (f32, f32)) {
        assert!(self.nodes[id].is_none());
        self.nodes[id] = Some(TreeNode::new(is_routing, pos));
        self.num_nodes += 1;
        debug_sched!("      {}add node {}{}", _BLUE, id, _RESET);
    }

    pub fn add_edge(&mut self, node_id1: usize, node_id2: usize) {
        // make sure the edge doesn't already exist
        assert!(!self.nodes[node_id1].as_ref().unwrap().nbors.contains(&node_id2));
        assert!(!self.nodes[node_id2].as_ref().unwrap().nbors.contains(&node_id1));
        self.nodes[node_id1].as_mut().unwrap().nbors.push(node_id2);
        self.nodes[node_id2].as_mut().unwrap().nbors.push(node_id1);
        self.num_edges += 1;
        debug_sched!("      {}add edge {}->{}{}", _BLUE, node_id1, node_id2, _RESET);
    }

    pub fn trim_dangling_nodes(&mut self) -> usize {
        let mut num_trimmed = 0;
        let root_id = self.root_node_id.unwrap();
        loop {
            // Find dangling bus nodes
            let mut dangling_ids: Vec<usize> = Vec::new();
            for (node_id, node_opt) in self.nodes.iter().enumerate() {
                if let Some(node) = node_opt {
                    if node.is_routing && node.nbors.len() <= 1 && node_id != root_id {
                        dangling_ids.push(node_id);
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

    fn remove_node(&mut self, node_id: usize) {
        debug_sched!("      {}remove node {}{}", _BLUE, node_id, _RESET);
        let nb_ids: Vec<usize> =
            self.nodes[node_id].as_ref().unwrap().nbors.iter().copied().collect();
        // Remove edges from neighbor nodes
        for nb_id in nb_ids {
            self.nodes[nb_id].as_mut().unwrap().remove_edge(node_id);
            self.num_edges -= 1;
            debug_sched!("      {}remove edge {}->{}{}", _BLUE, nb_id, node_id, _RESET);
        }
        // Remove the node itself
        self.nodes[node_id] = None;
        self.num_nodes -= 1;
    }

    pub fn remove_double_edges(&mut self) {
        let mut edges_to_remove: Vec<(usize, usize)> = Vec::new();
        for (node_id, node_opt) in self.nodes.iter().enumerate() {
            if let Some(node) = node_opt {
                if !node.is_routing && node.nbors.len() == 2 {
                    // remove side edge
                    for nb_id in &node.nbors {
                        let nb = self.nodes[*nb_id].as_ref().unwrap();
                        // check y position
                        if nb.pos.1 == node.pos.1 {
                            edges_to_remove.push((node_id, *nb_id));
                        }
                    }
                }
            }
        }
        for (node_id1, node_id2) in edges_to_remove {
            debug_sched!("      {}Removing additional edge {}->{}{}",
                         _BLUE,
                         node_id1,
                         node_id2,
                         _RESET);
            self.nodes[node_id1].as_mut().unwrap().remove_edge(node_id2);
            self.nodes[node_id2].as_mut().unwrap().remove_edge(node_id1);
            self.num_edges -= 1;
        }
    }

    pub fn node_list(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i))
            .collect()
    }

    #[cfg(debug_assertions)]
    pub fn check_vertical_data_edges(&self, node_id: usize) {
        let node = self.nodes[node_id].as_ref().unwrap();
        let mut above_count = 0;
        let mut below_count = 0;
        for nb_id in &node.nbors {
            let nb = self.nodes[*nb_id].as_ref().unwrap();
            if !nb.is_routing {
                if node.pos.1 < nb.pos.1 {
                    above_count += 1;
                } else if node.pos.1 > nb.pos.1 {
                    below_count += 1;
                }
            }
        }
        if above_count > 0 {
            assert_eq!(above_count, 2, "Routing node {} has {} nbors above", node_id, above_count);
        }
        if below_count > 0 {
            assert_eq!(below_count, 2, "Routing node {} has {} nbors below", node_id, below_count);
        }
    }
}
