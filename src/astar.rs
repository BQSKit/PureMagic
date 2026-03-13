use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// State container for A* pathfinding computations.
/// Maintains parent pointers, g-costs, and closed set for multi-source searches.
pub struct AStarComputation {
    parent: Vec<Option<u16>>,
    g_cost: Vec<u32>,
    closed: Vec<bool>,
    heap: BinaryHeap<(Reverse<u32>, u16)>,
    pub num_calls: usize,
}

impl AStarComputation {
    /// Creates a new A* computation state for a graph with `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Self {
        AStarComputation {
            parent: vec![None; num_nodes],
            g_cost: vec![u32::MAX; num_nodes],
            closed: vec![false; num_nodes],
            heap: BinaryHeap::new(),
            num_calls: 0,
        }
    }

    /// A* from first root to the nearest ready, unused magic node.
    /// Each `terminal_ids[i]` is attached to `root_ids[i]` in the returned tree.
    /// For single-X/Z T gates: one root, one terminal.
    /// For single-Y T gates: two roots (one above X-data, one below Z-data), two terminals.
    /// After building the main path (magic → root), any remaining roots that are not
    /// on the path are stitched in by finding an adjacent node already in the tree.
    /// When `plotting` is false, marks `used[]` directly and returns `Some(None)` (no tree built).
    /// When `plotting` is true, builds and returns `Some(Some(tree))`.
    /// Returns outer `None` if no path exists.
    pub fn compute(
        &mut self, terminal_ids: &[u16], root_ids: &[u16], topo: &TopoGraph, used: &mut Vec<bool>,
        ready_magic_positions: &[(f32, f32)], plotting: bool,
    ) -> Option<Option<TreeGraph>> {
        self.num_calls += 1;
        self.parent.fill(None);
        self.g_cost.fill(u32::MAX);
        self.closed.fill(false);

        self.heap.clear();

        let root_id = root_ids[0];
        debug_assert!(!used[root_id as usize]);
        self.g_cost[root_id as usize] = 0;
        let (h, ready_idx) = Self::heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        self.heap.push((Reverse(h), root_id));
        // choose this magic node as the target
        let ready_pos = ready_magic_positions[ready_idx];

        while let Some((_, node_id)) = self.heap.pop() {
            if self.closed[node_id as usize] {
                continue;
            }
            self.closed[node_id as usize] = true;

            let (node_type, cultivation_time, num_nbors) = {
                let node = topo.get_node(node_id);
                (node.node_type, topo.cultivation_times[node_id as usize], node.nbors.len())
            };

            if node_type == NodeType::Magic && cultivation_time == 0 && !used[node_id as usize] {
                if !plotting {
                    // Mark path nodes used directly; skip TreeGraph allocation.
                    used[node_id as usize] = true;
                    let mut curr = node_id;
                    while let Some(prev_id) = self.parent[curr as usize] {
                        used[prev_id as usize] = true;
                        curr = prev_id;
                    }
                    for &root_id in root_ids {
                        used[root_id as usize] = true;
                    }
                    for &tid in terminal_ids {
                        used[tid as usize] = true;
                    }
                    return Some(None);
                }
                let mut tree = TreeGraph::new(topo.num_nodes);
                tree.root_node_id = Some(node_id);
                let mut curr = node_id;
                if !tree.contains_node(curr) {
                    tree.add_node(topo.get_node(curr), topo.get_label(curr));
                }
                while let Some(prev_id) = self.parent[curr as usize] {
                    if !tree.contains_node(prev_id) {
                        tree.add_node(topo.get_node(prev_id), topo.get_label(prev_id));
                    }
                    tree.add_edge(prev_id, curr);
                    curr = prev_id;
                }
                for (i, &root_id) in root_ids.iter().enumerate() {
                    if !tree.contains_node(root_id) {
                        let conn = topo
                            .get_node(root_id)
                            .nbors
                            .iter()
                            .copied()
                            .find(|&nb_id| tree.contains_node(nb_id));
                        if let Some(conn_id) = conn {
                            tree.add_node(topo.get_node(root_id), topo.get_label(root_id));
                            tree.add_edge(conn_id, root_id);
                        } else {
                            return None;
                        }
                    }
                    if i < terminal_ids.len() {
                        let tid = terminal_ids[i];
                        tree.add_node(topo.get_node(tid), topo.get_label(tid));
                        tree.add_edge(root_id, tid);
                    }
                }
                return Some(Some(tree));
            }

            let g = self.g_cost[node_id as usize];
            for i in 0..num_nbors {
                let nb_id = topo.get_node(node_id).nbors[i];
                if used[nb_id as usize] || self.closed[nb_id as usize] {
                    continue;
                }
                let (nb_is_data, nb_pos) = {
                    let nb = topo.get_node(nb_id);
                    (nb.node_type == NodeType::Data, nb.pos)
                };
                if nb_is_data {
                    continue;
                }
                let new_g = g + 1;
                if new_g < self.g_cost[nb_id as usize] {
                    self.g_cost[nb_id as usize] = new_g;
                    self.parent[nb_id as usize] = Some(node_id);
                    // always headed to the same target magic node
                    let h = Self::manhattan_dist(nb_pos, ready_pos);
                    self.heap.push((Reverse(new_g + h), nb_id));
                }
            }
        }
        None
    }

    /// Lower-bound heuristic: Manhattan distance from `pos` to the nearest ready magic node,
    /// Used to guide A* search towards available magic state sources.
    fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> (u32, usize) {
        ready_magic_positions
            .iter()
            .enumerate()
            .map(|(idx, &mp)| (Self::manhattan_dist(mp, pos), idx))
            .min_by(|(da, _), (db, _)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    }

    fn manhattan_dist(p1: (f32, f32), p2: (f32, f32)) -> u32 {
        ((p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()) as u32
    }
}
