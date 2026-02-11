use crate::node::Node;
use indexmap::IndexMap;

pub struct TreeGraph {
    nodes: IndexMap<usize, Node>,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub root_node: Option<usize>,
}

impl TreeGraph {
    pub fn new() -> Self {
        TreeGraph { nodes: IndexMap::new(), num_edges: 0, num_nodes: 0, root_node: None }
    }

    pub fn trim_dangling_nodes(&mut self) -> usize {
        let mut num_trimmed = 0;
        let root_node = self.root_node.unwrap();
        loop {
            // Find dangling bus nodes
            /*
            let mut dangling_ids: Vec<usize> = Vec::new();
            for (id, node) in self.nodes.iter() {
                // there is at most one path going into the bus/magic node
                if node.is_routing() && node.edges.len() <= 1 && node.id != root_node {
                    dangling_ids.push(id.clone());
                }
            } */
            let dangling_ids: Vec<usize> =
                self.nodes
                    .iter()
                    .filter(|(_, node)| {
                        node.is_routing() && node.edges.len() <= 1 && node.id != root_node
                    })
                    .map(|(id, _)| *id)
                    .collect();
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

    pub fn get_node(&self, id: usize) -> &Node {
        self.nodes.get(&id).expect(&format!("Node {} not found", id))
    }

    pub fn get_node_mut(&mut self, id: usize) -> &mut Node {
        self.nodes.get_mut(&id).expect(&format!("Node {} not found", id))
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn _iter_edges(&self) -> impl Iterator<Item = (&usize, &usize)> + '_ {
        self.nodes
            .iter()
            .flat_map(|(node_id, node)| node.edges.iter().map(move |edge_id| (node_id, edge_id)))
    }

    pub fn contains_node(&self, node_id: &usize) -> bool {
        self.nodes.contains_key(node_id)
    }

    pub fn contains_edge(&self, node_id1: &usize, node_id2: &usize) -> bool {
        if let Some(node) = self.nodes.get(node_id1) {
            node.edges.contains(node_id2)
        } else {
            false
        }
    }

    pub fn add_node(&mut self, node: Node) {
        let new_node = Node::new(node.id,
                                 node.paired_data_id,
                                 node.label.to_string(),
                                 node.pos.0,
                                 node.pos.1,
                                 node.node_type,
                                 node.busy_count,
                                 node.cultivation_time);
        self.nodes.insert(new_node.id, new_node);
        self.num_nodes += 1;
    }

    pub fn remove_node(&mut self, node_id: usize) {
        // Get edges to remove from neighbors
        let node = self.get_node(node_id);
        let edges_to_remove: Vec<(usize, usize)> =
            node.edges.iter().map(|neighbor| (neighbor.clone(), node_id)).collect();
        // Remove edges from neighbor nodes
        for (nb_id, edge_to_remove) in edges_to_remove {
            if let Some(nb) = self.nodes.get_mut(&nb_id) {
                nb.edges.swap_remove(&edge_to_remove);
                self.num_edges -= 1;
            }
        }
        // Remove the node itself
        if self.nodes.swap_remove(&node_id).is_some() {
            self.num_nodes -= 1;
        }
    }

    pub fn add_edge(&mut self, node_id1: usize, node_id2: usize) {
        self.get_node_mut(node_id1).add_edge(node_id2);
        self.get_node_mut(node_id2).add_edge(node_id1);
        self.num_edges += 1;
    }

    pub fn node_list(&self) -> Vec<usize> {
        self.nodes.keys().cloned().collect()
    }
}
