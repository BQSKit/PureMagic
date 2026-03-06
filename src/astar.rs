use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct AStarComputation {
    parent: Vec<Option<usize>>,
    g_cost: Vec<u32>,
    closed: Vec<bool>,
}

impl AStarComputation {
    pub fn new(num_nodes: usize) -> Self {
        AStarComputation { parent: vec![None; num_nodes],
                           g_cost: vec![u32::MAX; num_nodes],
                           closed: vec![false; num_nodes] }
    }

    /// A* shortest path from `root_id` to the nearest ready, unused magic node.
    /// Handles single-terminal T (one terminal) and single-Y T (two paired terminals sharing one
    /// root).  Returns a TreeGraph with all terminals attached to root → path → magic,
    /// with `root_node_id` set to the magic node, or None if no path exists.
    pub fn compute(&mut self, terminal_ids: &[usize], root_id: usize, topo: &TopoGraph,
                   used: &[bool], ready_magic_positions: &[(f32, f32)])
                   -> Option<TreeGraph> {
        if used[root_id] {
            return None;
        }
        self.parent.fill(None);
        self.g_cost.fill(u32::MAX);
        self.closed.fill(false);

        self.g_cost[root_id] = 0;
        let h0 = Self::heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        let mut heap: BinaryHeap<(Reverse<u32>, usize)> = BinaryHeap::new();
        heap.push((Reverse(h0), root_id));

        while let Some((_, node_id)) = heap.pop() {
            if self.closed[node_id] {
                continue;
            }
            self.closed[node_id] = true;

            // Copy out what we need from topo before touching the astar fields.
            let (node_type, cultivation_time, nbors) = {
                let node = topo.get_node(node_id);
                (node.node_type, node.cultivation_time, node.nbors.clone())
            };

            // Goal: an unused, ready magic node.
            if node_type == NodeType::Magic && cultivation_time == 0 && !used[node_id] {
                let mut tree = TreeGraph::new(topo.num_nodes);
                tree.root_node_id = Some(node_id);
                // Attach all terminals (one for X/Z single-qubit T, two for single-Y T).
                tree.add_node(topo.get_node(root_id));
                for &tid in terminal_ids {
                    tree.add_node(topo.get_node(tid));
                    tree.add_edge(root_id, tid);
                }
                // Walk parent chain from magic back to root, adding nodes and edges.
                let mut curr = node_id;
                if !tree.contains_node(curr) {
                    tree.add_node(topo.get_node(curr));
                }
                while let Some(prev_id) = self.parent[curr] {
                    if !tree.contains_node(prev_id) {
                        tree.add_node(topo.get_node(prev_id));
                    }
                    tree.add_edge(prev_id, curr);
                    curr = prev_id;
                }
                return Some(tree);
            }

            let g = self.g_cost[node_id];
            for nb_id in nbors {
                if used[nb_id] || self.closed[nb_id] {
                    continue;
                }
                // Copy nb data before mutating the astar fields.
                let (nb_is_data, nb_pos) = {
                    let nb = topo.get_node(nb_id);
                    (nb.node_type == NodeType::Data, nb.pos)
                };
                if nb_is_data {
                    continue;
                }
                let new_g = g + 1;
                if new_g < self.g_cost[nb_id] {
                    self.g_cost[nb_id] = new_g;
                    self.parent[nb_id] = Some(node_id);
                    let h = Self::heuristic(nb_pos, ready_magic_positions);
                    heap.push((Reverse(new_g + h), nb_id));
                }
            }
        }
        None
    }

    /// Lower-bound heuristic: Manhattan distance from `pos` to the nearest ready magic node,
    /// floored to a u32 so it is always admissible for unit-weight edges.
    fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> u32 {
        ready_magic_positions.iter()
                             .map(|mp| (mp.0 - pos.0).abs() + (mp.1 - pos.1).abs())
                             .fold(f32::MAX, f32::min)
                             .floor() as u32
    }
}
