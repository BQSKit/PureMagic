use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;

/// Result of a pathfinding computation.
/// `NoPath` means no valid route exists.
/// `PathFound(None)` means a route was found and `used[]` marked, but no tree was built (non-plotting mode).
/// `PathFound(Some(tree))` means a route was found and a full routing tree was built (plotting mode).
#[derive(Debug)]
pub enum PathResult {
    NoPath,
    PathFound(Option<TreeGraph>),
}

/// Number of buckets in the bucket-queue (Dial's algorithm).
/// f_cost = g + h ≤ 2 × grid_diameter ≈ 112, so 256 buckets gives ample headroom.
const BUCKET_COUNT: usize = 256;

/// State container for A* pathfinding computations.
/// Maintains parent pointers, g-costs, and closed set for multi-source searches.
///
/// Uses a generation-counter to avoid O(N) array resets between calls: `epoch` is
/// bumped once per `compute` call; a node slot is valid iff its stored epoch matches.
/// The priority queue is a bucket queue (Dial's algorithm) over integer f-costs,
/// giving O(1) push and amortised O(1) pop without comparison overhead.
pub struct AStarComputation {
    /// Parent pointer; valid only when `node_epoch[id] == epoch`.
    /// `u16::MAX` is the sentinel meaning "no parent" (search root).
    parent: Vec<u16>,
    /// Tentative g-cost; valid only when `node_epoch[id] == epoch`.
    g_cost: Vec<u32>,
    /// Epoch when this node was last opened; `node_epoch[id] == epoch` means open.
    node_epoch: Vec<u32>,
    /// Epoch when this node was last closed; `closed_epoch[id] == epoch` means closed.
    closed_epoch: Vec<u32>,
    /// Bumped once per `compute` call to invalidate stale per-node state in O(1).
    epoch: u32,
    /// Bucket queue: `buckets[f % BUCKET_COUNT]` holds node IDs with f-cost `f`.
    buckets: Vec<Vec<u16>>,
    /// Lowest bucket index that may be non-empty.
    bucket_min: usize,
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
            buckets: vec![Vec::new(); BUCKET_COUNT],
            bucket_min: 0,
            num_calls: 0,
        }
    }

    #[inline(always)]
    fn bucket_push(&mut self, f_cost: u32, node_id: u16) {
        let idx = (f_cost as usize) % BUCKET_COUNT;
        self.buckets[idx].push(node_id);
        if idx < self.bucket_min {
            self.bucket_min = idx;
        }
    }

    /// Returns the next open node, skipping already-closed (stale) entries.
    /// Returns `None` when all buckets are empty.
    #[inline(always)]
    fn bucket_pop(&mut self, epoch: u32) -> Option<u16> {
        loop {
            if self.bucket_min >= BUCKET_COUNT {
                return None;
            }
            if let Some(node_id) = self.buckets[self.bucket_min].pop() {
                if self.closed_epoch[node_id as usize] != epoch {
                    return Some(node_id);
                }
                // stale duplicate — keep draining this bucket
            } else {
                self.bucket_min += 1;
            }
        }
    }

    #[inline(always)]
    fn bucket_clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.bucket_min = 0;
    }

    /// A* from first root to the nearest ready, unused magic node.
    /// Each `terminal_ids[i]` is attached to `root_ids[i]` in the returned tree.
    /// For single-X/Z T gates: one root, one terminal.
    /// For single-Y T gates: two roots (one above X-data, one below Z-data), two terminals.
    /// After building the main path (magic → root), any remaining roots that are not
    /// on the path are stitched in by finding an adjacent node already in the tree.
    /// When `plotting` is false, marks `used[]` directly and returns `PathFound(None)`.
    /// When `plotting` is true, builds and returns `PathFound(Some(tree))`.
    /// Returns `NoPath` if no path exists.
    pub fn compute(
        &mut self, terminal_ids: &[u16], root_ids: &[u16], topo: &TopoGraph, used: &mut Vec<bool>,
        ready_magic_positions: &[(f32, f32)], plotting: bool,
    ) -> PathResult {
        self.num_calls += 1;
        // Bump epoch to invalidate stale per-node state without filling arrays.
        // On the rare u32 wrap-around, reset epoch arrays to restore the invariant.
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            self.node_epoch.fill(0);
            self.closed_epoch.fill(0);
        }
        let epoch = self.epoch;

        self.bucket_clear();

        let root_id = root_ids[0];
        debug_assert!(!used[root_id as usize]);
        self.g_cost[root_id as usize] = 0;
        self.parent[root_id as usize] = u16::MAX;
        self.node_epoch[root_id as usize] = epoch;
        let (h, ready_idx) = Self::heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        self.bucket_push(h, root_id);
        let ready_pos = ready_magic_positions[ready_idx];

        while let Some(node_id) = self.bucket_pop(epoch) {
            self.closed_epoch[node_id as usize] = epoch;

            let (node_type, cultivation_time, num_nbors) = {
                let node = topo.get_node(node_id);
                (node.node_type, topo.cultivation_times[node_id as usize], node.num_nbors as usize)
            };

            if node_type == NodeType::Magic && cultivation_time == 0 && !used[node_id as usize] {
                if !plotting {
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
                    return PathResult::PathFound(None);
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
                            .nbors_slice()
                            .iter()
                            .copied()
                            .find(|&nb_id| tree.contains_node(nb_id));
                        if let Some(conn_id) = conn {
                            tree.add_node(topo.get_node(root_id), topo.get_label(root_id));
                            tree.add_edge(conn_id, root_id);
                        } else {
                            return PathResult::NoPath;
                        }
                    }
                    if i < terminal_ids.len() {
                        let tid = terminal_ids[i];
                        tree.add_node(topo.get_node(tid), topo.get_label(tid));
                        tree.add_edge(root_id, tid);
                    }
                }
                return PathResult::PathFound(Some(tree));
            }

            // If this is a magic node that is not the goal (not ready/unused), skip expanding
            // its neighbors — magic nodes must not be used as routing intermediaries unless
            // use_magic_routing is enabled.
            if !topo.use_magic_routing && node_type == NodeType::Magic {
                continue;
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
                // Stale slot (different epoch) is treated as g = ∞.
                let nb_g = if self.node_epoch[nb_id as usize] == epoch {
                    self.g_cost[nb_id as usize]
                } else {
                    u32::MAX
                };
                if new_g < nb_g {
                    self.g_cost[nb_id as usize] = new_g;
                    self.parent[nb_id as usize] = node_id;
                    self.node_epoch[nb_id as usize] = epoch;
                    let h = Self::manhattan_dist(nb_pos, ready_pos);
                    self.bucket_push(new_g + h, nb_id);
                }
            }
        }
        PathResult::NoPath
    }

    /// Returns (distance, index) of the nearest ready magic node to `pos`.
    /// `ready_magic_positions` must be sorted by x-coordinate (ascending).
    /// Binary-searches for the x-anchor then sweeps outward, pruning once the
    /// x-gap alone exceeds the current best distance.
    fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> (u32, usize) {
        let anchor = ready_magic_positions.partition_point(|&(mx, _)| mx < pos.0);

        let mut best_dist = u32::MAX;
        let mut best_idx = 0usize;

        let mut r = anchor;
        while r < ready_magic_positions.len() {
            let dx = (ready_magic_positions[r].0 - pos.0).abs() as u32;
            if dx >= best_dist {
                break;
            }
            let d = Self::manhattan_dist(ready_magic_positions[r], pos);
            if d < best_dist {
                best_dist = d;
                best_idx = r;
            }
            r += 1;
        }

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
