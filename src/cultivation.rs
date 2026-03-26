use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::utils::{_RED, _RESET};
use rand_simple::Exponential;

// ── CultivationManager ────────────────────────────────────────────────────────

/// Owns the magic-state cultivation pool, timing state, and magic node tracking
/// for a [`crate::scheduler::Scheduler`].
///
/// Holds the exponential RNG, the pre-generated pool of cultivation times, the
/// log of times consumed, and the magic node ID/position lists used each lcycle.
/// Create with [`CultivationManager::new`].
pub(crate) struct CultivationManager {
    rng_exp: Exponential,
    /// Pre-generated pool of cultivation times (exponential distribution).
    cultivation_time_pool: Vec<i32>,
    pool_index: usize,
    /// T-gate products not yet scheduled; used to size pool refills.
    pub(crate) t_products_remaining: usize,
    /// Scratch buffer used by `update_cultivators` to avoid repeated allocations.
    new_cultivation_times: Vec<i32>,
    /// Log of cultivation times that have been drawn and consumed.
    pub(crate) cultivation_times_log: Vec<i32>,
    /// Magic node IDs, built once at init.
    pub(crate) magic_node_ids: Vec<u16>,
    /// Positions of magic nodes, parallel to `magic_node_ids`.
    pub(crate) magic_node_positions: Vec<(f32, f32)>,
    /// Ready (cultivation_time=0) magic node positions, updated each lcycle.
    /// Sorted by x for binary-search pruning in A* heuristic.
    pub(crate) ready_magic_positions: Vec<(f32, f32)>,
}

impl CultivationManager {
    /// Creates a new manager seeded with `rseed`.
    pub(crate) fn new(rseed: u32) -> Self {
        CultivationManager {
            rng_exp: Exponential::new(rseed),
            cultivation_time_pool: Vec::new(),
            pool_index: 0,
            t_products_remaining: 0,
            new_cultivation_times: Vec::new(),
            cultivation_times_log: Vec::new(),
            magic_node_ids: Vec::new(),
            magic_node_positions: Vec::new(),
            ready_magic_positions: Vec::new(),
        }
    }

    /// Sets the exponential distribution parameter (1/lambda).
    pub(crate) fn set_lambda(&mut self, magic_state_lambda: f64) -> Result<f64, &'static str> {
        self.rng_exp.try_set_params(1.0 / magic_state_lambda)
    }

    /// Fills the cultivation time pool with `n` exponentially-distributed samples.
    pub(crate) fn fill_pool(&mut self, n: usize) {
        self.cultivation_time_pool.clear();
        self.cultivation_time_pool.reserve(n);
        for _ in 0..n {
            self.cultivation_time_pool.push(self.rng_exp.sample().round() as i32);
        }
        self.pool_index = 0;
    }

    /// Returns the next cultivation time from the pool, refilling if exhausted.
    #[inline]
    pub(crate) fn draw(&mut self, num_topo_nodes: usize) -> i32 {
        if self.pool_index >= self.cultivation_time_pool.len() {
            if self.pool_index > 0 {
                eprintln!(
                    "{}Warning: refilling cultivation pool for {} remaining T products{}",
                    _RED, self.t_products_remaining, _RESET
                );
            }
            self.fill_pool(10 * self.t_products_remaining + num_topo_nodes);
        }
        let t = self.cultivation_time_pool[self.pool_index];
        self.pool_index += 1;
        t
    }

    /// Initializes magic node IDs/positions from `topo` and assigns initial cultivation times.
    pub(crate) fn init_magic_nodes(&mut self, topo: &mut TopoGraph) {
        self.magic_node_ids = topo
            .iter_nodes()
            .filter(|node| node.node_type == NodeType::Magic)
            .map(|node| node.id)
            .collect();
        self.magic_node_positions =
            self.magic_node_ids.iter().map(|&id| topo.get_node(id).pos).collect();
        let num_topo_nodes = topo.num_nodes;
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            topo.cultivation_times[id as usize] = self.draw(num_topo_nodes);
            topo.busy_counts[id as usize] = 0;
        }
    }

    /// Advances cultivation state for all magic nodes after an lcycle.
    ///
    /// Resets used nodes with new cultivation times, increments busy counts for
    /// cultivating nodes, and returns the number of available (ready) magic nodes.
    /// Also rebuilds `ready_magic_positions` sorted by x for A* heuristic pruning.
    pub(crate) fn update_cultivators(&mut self, topo: &mut TopoGraph, used: &[bool]) -> usize {
        let num_topo_nodes = topo.num_nodes;
        self.new_cultivation_times.clear();
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            if used[id as usize] {
                let t = self.draw(num_topo_nodes);
                self.new_cultivation_times.push(t);
            }
        }
        let mut num_avail_magic = 0;
        let mut cultivation_time_index = 0;
        self.ready_magic_positions.clear();
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            if used[id as usize] {
                topo.cultivation_times[id as usize] =
                    self.new_cultivation_times[cultivation_time_index];
                topo.busy_counts[id as usize] = 0;
                cultivation_time_index += 1;
            } else if topo.is_cultivating(id) {
                topo.busy_counts[id as usize] += 1;
                if topo.busy_counts[id as usize] == topo.cultivation_times[id as usize] {
                    self.cultivation_times_log.push(topo.cultivation_times[id as usize]);
                    topo.cultivation_times[id as usize] = 0;
                    topo.busy_counts[id as usize] = 0;
                }
            }
            if topo.cultivation_times[id as usize] == 0 {
                num_avail_magic += 1;
                self.ready_magic_positions.push(self.magic_node_positions[i]);
            }
        }
        // Sort by x so AStarComputation::heuristic can prune with binary search.
        self.ready_magic_positions
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        num_avail_magic
    }
}
