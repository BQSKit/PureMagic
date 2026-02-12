pub struct TreeGraph {
    nodes: Vec<Option<Vec<usize>>>,
    routing_nodes: Vec<Option<bool>>,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub root_node_id: Option<usize>,
}

impl TreeGraph {
    pub fn new(num_nodes: usize) -> Self {
        TreeGraph { nodes: vec![None; num_nodes],
                    routing_nodes: vec![None; num_nodes],
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
        if let Some(nbs) = &self.nodes[node_id1] { nbs.contains(&node_id2) } else { false }
    }

    pub fn add_node(&mut self, id: usize, is_routing: bool) {
        self.nodes[id] = Some(Vec::new());
        self.routing_nodes[id] = Some(is_routing);
        self.num_nodes += 1;
    }

    pub fn add_edge(&mut self, node_id1: usize, node_id2: usize) {
        self.nodes[node_id1].as_mut().unwrap().push(node_id2);
        self.nodes[node_id2].as_mut().unwrap().push(node_id1);
        self.num_edges += 1;
    }

    pub fn trim_dangling_nodes(&mut self) -> usize {
        let mut num_trimmed = 0;
        let root_id = self.root_node_id.unwrap();
        loop {
            // Find dangling bus nodes
            let mut dangling_ids: Vec<usize> = Vec::new();
            for (node_id, node_opt) in self.nodes.iter().enumerate() {
                if let Some(nbs) = node_opt {
                    let is_routing = self.routing_nodes[node_id].unwrap();
                    if is_routing && nbs.len() <= 1 && node_id != root_id {
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
        let nb_ids: Vec<usize> = self.nodes[node_id].as_ref().unwrap().iter().copied().collect();
        // Remove edges from neighbor nodes
        for nb_id in nb_ids {
            // FIXME: this could be inefficient
            let pos =
                self.nodes[nb_id].as_ref().unwrap().iter().position(|&id| id == node_id).unwrap();
            self.nodes[nb_id].as_mut().unwrap().swap_remove(pos);
            self.num_edges -= 1;
        }
        // Remove the node itself
        self.nodes[node_id] = None;
        self.routing_nodes[node_id] = None;
        self.num_nodes -= 1;
    }

    pub fn node_list(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node_opt)| node_opt.as_ref().map(|_| i))
            .collect()
    }
}
