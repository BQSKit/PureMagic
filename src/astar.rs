use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;

/// Number of buckets in the bucket-queue (Dial's algorithm).
/// f_cost = g + h where g ≤ grid_diameter (~56) and h ≤ grid_diameter (~56),
/// so f ≤ 112. 256 buckets gives ample headroom and keeps the modulo a cheap
/// bitwise AND.
const BUCKET_COUNT: usize = 256;

/// State container for A* pathfinding computations.
/// Maintains parent pointers, g-costs, and closed set for multi-source searches.
///
/// Uses two optimisations over the naive implementation:
///
/// 1. **Generation counter** (hotspot 1): `epoch` is bumped once per `compute`
///    call instead of filling three O(N) arrays.  A node slot is "open" iff
///    `node_epoch[id] == epoch`; "closed" iff `closed_epoch[id] == epoch`.
///
/// 2. **Bucket queue / Dial's algorithm** (hotspot 3): replaces `BinaryHeap`
///    with a fixed array of `BUCKET_COUNT` `Vec<u16>` buckets indexed by
///    `f_cost % BUCKET_COUNT`.  Push is O(1) (append to a Vec); pop is
///    amortised O(1) (advance a cursor over at most BUCKET_COUNT empty slots).
///    Eliminates all `sift_up` / `sift_down` / `PartialOrd` overhead.
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
    /// Bucket queue: `buckets[f % BUCKET_COUNT]` holds node IDs with that f_cost.
    buckets: Vec<Vec<u16>>,
    /// Cursor: the lowest bucket index that may be non-empty.
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

    /// Push `node_id` into the bucket for `f_cost`.
    #[inline(always)]
    fn bucket_push(&mut self, f_cost: u32, node_id: u16) {
        let idx = (f_cost as usize) % BUCKET_COUNT;
        self.buckets[idx].push(node_id);
        if idx < self.bucket_min {
            self.bucket_min = idx;
        }
    }

    /// Pop the node with the smallest f_cost from the bucket queue.
    /// Returns `None` when all buckets are empty.
    /// Nodes that are already closed (stale entries) are skipped automatically
    /// by the caller's closed-epoch check.
    #[inline(always)]
    fn bucket_pop(&mut self, epoch: u32) -> Option<u16> {
        loop {
            if self.bucket_min >= BUCKET_COUNT {
                return None;
            }
            if let Some(node_id) = self.buckets[self.bucket_min].pop() {
                // Skip stale entries (already closed this epoch).
                if self.closed_epoch[node_id as usize] != epoch {
                    return Some(node_id);
                }
                // stale — keep scanning this bucket
            } else {
                self.bucket_min += 1;
            }
        }
    }

    /// Clear all buckets and reset the cursor. Called once per `compute` invocation.
    #[inline(always)]
    fn bucket_clear(&mut self) {
        // Only clear buckets that were actually used (tracked by bucket_min).
        // In practice almost all buckets are empty so this is fast.
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

        self.bucket_clear();

        let root_id = root_ids[0];
        debug_assert!(!used[root_id as usize]);
        // Open the root node.
        self.g_cost[root_id as usize] = 0;
        self.parent[root_id as usize] = u16::MAX; // sentinel: no parent
        self.node_epoch[root_id as usize] = epoch;
        let (h, ready_idx) = Self::heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        self.bucket_push(h, root_id);
        // choose this magic node as the target
        let ready_pos = ready_magic_positions[ready_idx];

        while let Some(node_id) = self.bucket_pop(epoch) {
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
                    self.bucket_push(new_g + h, nb_id);
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
