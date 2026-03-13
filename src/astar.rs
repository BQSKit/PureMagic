use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// State container for A* pathfinding computations.
/// Maintains parent pointers, g-costs, and closed set for multi-source searches.
///
/// Uses a generation-counter scheme to avoid resetting the arrays on every call:
/// - `epoch` is incremented once per `compute` call.
/// - A node slot is considered "initialised" (has a valid g_cost / parent) iff
///   `node_epoch[id] == epoch`.
/// - A node slot is considered "closed" iff `closed_epoch[id] == epoch`.
/// This replaces the three O(N) `fill` calls that previously dominated the hot path.
pub struct AStarComputation {
    /// Parent pointer; valid only when `node_epoch[id] == epoch`.
    /// Stores `u16::MAX` when there is no parent (i.e. the node is the search root).
    parent: Vec<u16>,
    /// Tentative g-cost; valid only when `node_epoch[id] == epoch`.
    g_cost: Vec<u32>,
    /// Per-node epoch stamp: `node_epoch[id] == epoch` means the node has been opened.
    node_epoch: Vec<u32>,
    /// Per-node closed stamp: `closed_epoch[id] == epoch` means the node has been closed.
    closed_epoch: Vec<u32>,
    /// Current epoch counter; bumped once per `compute` call instead of filling arrays.
    epoch: u32,
    heap: BinaryHeap<(Reverse<u32>, u16)>,
    pub num_calls: usize,
}

impl AStarComputation {
    /// Creates a new A* computation state for a graph with `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Self {
        AStarComputation {
            parent: vec![u16::MAX; num_nodes],
            g_cost: vec![u32::MAX; num_nodes],
            node_epoch: vec![0; num_nodes],
            closed_epoch: vec![0; num_nodes],
            epoch: 0,
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
        // Advance epoch to invalidate all previous per-node state in O(1).
        // Wrapping add: if epoch wraps to 0 we do a full reset to keep the invariant.
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            self.node_epoch.fill(0);
            self.closed_epoch.fill(0);
        }
        let epoch = self.epoch;

        self.heap.clear();

        let root_id = root_ids[0];
        debug_assert!(!used[root_id as usize]);
        // Open the root node.
        self.g_cost[root_id as usize] = 0;
        self.parent[root_id as usize] = u16::MAX; // sentinel: no parent
        self.node_epoch[root_id as usize] = epoch;
        let (h, ready_idx) = Self::heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        self.heap.push((Reverse(h), root_id));
        // choose this magic node as the target
        let ready_pos = ready_magic_positions[ready_idx];

        while let Some((_, node_id)) = self.heap.pop() {
            if self.closed_epoch[node_id as usize] == epoch {
                continue;
            }
            self.closed_epoch[node_id as usize] = epoch;

            let (node_type, cultivation_time, num_nbors) = {
                let node = topo.get_node(node_id);
                (node.node_type, topo.cultivation_times[node_id as usize], node.nbors.len())
            };

            if node_type == NodeType::Magic && cultivation_time == 0 && !used[node_id as usize] {
                if !plotting {
                    // Mark path nodes used directly; skip TreeGraph allocation.
                    used[node_id as usize] = true;
                    let mut curr = node_id;
                    loop {
                        let prev_id = self.parent[curr as usize];
                        if prev_id == u16::MAX {
                            break;
                        }
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
                loop {
                    let prev_id = self.parent[curr as usize];
                    if prev_id == u16::MAX {
                        break;
                    }
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
                if used[nb_id as usize] || self.closed_epoch[nb_id as usize] == epoch {
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
                // A node is "unvisited this epoch" when node_epoch[nb] != epoch,
                // in which case its g_cost slot holds a stale value — treat as u32::MAX.
                let nb_g = if self.node_epoch[nb_id as usize] == epoch {
                    self.g_cost[nb_id as usize]
                } else {
                    u32::MAX
                };
                if new_g < nb_g {
                    self.g_cost[nb_id as usize] = new_g;
                    self.parent[nb_id as usize] = node_id;
                    self.node_epoch[nb_id as usize] = epoch;
                    // always headed to the same target magic node
                    let h = Self::manhattan_dist(nb_pos, ready_pos);
                    self.heap.push((Reverse(new_g + h), nb_id));
                }
            }
        }
        None
    }

    /// Lower-bound heuristic: Manhattan distance from `pos` to the nearest ready magic node.
    /// `ready_magic_positions` must be **sorted by x-coordinate** (ascending).
    /// Uses a binary-search anchor + bidirectional sweep with x-gap pruning to find the
    /// nearest entry in O(√N) average time instead of O(N).
    fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> (u32, usize) {
        // Binary-search for the insertion point of pos.x in the sorted x-values.
        let anchor = ready_magic_positions.partition_point(|&(mx, _)| mx < pos.0);

        let mut best_dist = u32::MAX;
        let mut best_idx = 0usize;

        // Sweep right from anchor.
        let mut r = anchor;
        while r < ready_magic_positions.len() {
            let dx = (ready_magic_positions[r].0 - pos.0).abs() as u32;
            if dx >= best_dist {
                break; // all further entries have dx ≥ best_dist, so Manhattan ≥ best_dist
            }
            let d = Self::manhattan_dist(ready_magic_positions[r], pos);
            if d < best_dist {
                best_dist = d;
                best_idx = r;
            }
            r += 1;
        }

        // Sweep left from anchor - 1.
        let mut l = anchor;
        while l > 0 {
            l -= 1;
            let dx = (ready_magic_positions[l].0 - pos.0).abs() as u32;
            if dx >= best_dist {
                break;
            }
            let d = Self::manhattan_dist(ready_magic_positions[l], pos);
            if d < best_dist {
                best_dist = d;
                best_idx = l;
            }
        }

        (best_dist, best_idx)
    }

    fn manhattan_dist(p1: (f32, f32), p2: (f32, f32)) -> u32 {
        ((p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()) as u32
    }
}
