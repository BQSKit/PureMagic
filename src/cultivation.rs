use crate::node::NodeType;
use crate::topograph::TopoGraph;
use colored::Colorize;
use rand_simple::Exponential;

/// Owns the magic-state cultivation pool, timing state, and magic node tracking
/// for a [`crate::scheduler::Scheduler`].
///
/// Magic state cultivation is modelled as a stochastic process: each magic node
/// is assigned a random cultivation time drawn from an exponential distribution
/// with parameter `lambda`, optionally shifted by a fixed floor (see
/// [`CultivationManager::set_lambda`]). The node becomes "ready" (cultivation_time = 0) after
/// `busy_count` reaches `cultivation_time`. Once consumed by a T gate, a new
/// cultivation time is drawn and the cycle restarts.
pub(crate) struct CultivationManager {
    rng_exp: Exponential,
    /// Fixed cost added to every cultivation draw; see `set_lambda`.
    floor: f64,
    cultivation_time_pool: Vec<i32>,
    pool_index: usize,
    /// T-gate products not yet scheduled; used to size pool refills.
    pub(crate) t_products_remaining: usize,
    new_cultivation_times: Vec<i32>,
    pub(crate) cultivation_times_log: Vec<i32>,
    pub(crate) magic_node_ids: Vec<u16>,
    pub(crate) magic_node_positions: Vec<(f32, f32)>,
    /// Sorted by x for binary-search pruning in A* heuristic.
    pub(crate) ready_magic_positions: Vec<(f32, f32)>,
}

impl CultivationManager {
    pub(crate) fn new(rseed: u32) -> Self {
        CultivationManager {
            rng_exp: Exponential::new(rseed),
            floor: 0.0,
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

    /// Sets the cultivation time distribution to `floor` plus an exponential,
    /// with the exponential's mean reduced by `floor` so the total mean stays
    /// 1/`magic_state_lambda`. A floor sweep is therefore a pure
    /// shape change at fixed mean: `floor = 0` is the memoryless model, and
    /// `floor -> 1/lambda` approaches a deterministic cultivation time.
    /// Errors if `floor` would leave the exponential component non-positive.
    pub(crate) fn set_lambda(
        &mut self, magic_state_lambda: f64, floor: f64,
    ) -> Result<f64, &'static str> {
        let total_mean = 1.0 / magic_state_lambda;
        if floor < 0.0 {
            return Err("cultivation floor must be non-negative");
        }
        if floor >= total_mean {
            return Err("cultivation floor must be below the total mean 1/lambda");
        }
        self.floor = floor;
        self.rng_exp.try_set_params(total_mean - floor)
    }

    pub(crate) fn fill_pool(&mut self, n: usize) {
        self.cultivation_time_pool.clear();
        self.cultivation_time_pool.reserve(n);
        for _ in 0..n {
            self.cultivation_time_pool.push((self.floor + self.rng_exp.sample()).round() as i32);
        }
        self.pool_index = 0;
    }

    /// The pool is pre-generated to amortise RNG cost; refilling mid-run is a
    /// rare fallback that should only trigger if the initial pool was undersized.
    #[inline]
    pub(crate) fn draw(&mut self, n_topo_nodes: usize) -> i32 {
        if self.pool_index >= self.cultivation_time_pool.len() {
            if self.pool_index > 0 {
                eprintln!(
                    "{}",
                    format!(
                        "Warning: refilling cultivation pool for {} remaining T products",
                        self.t_products_remaining
                    )
                    .red()
                );
            }
            self.fill_pool(10 * self.t_products_remaining + n_topo_nodes);
        }
        let t = self.cultivation_time_pool[self.pool_index];
        self.pool_index += 1;
        t
    }

    pub(crate) fn init_magic_nodes(&mut self, topo: &mut TopoGraph) {
        self.magic_node_ids = topo
            .iter_nodes()
            .filter(|node| node.node_type == NodeType::Magic)
            .map(|node| node.id)
            .collect();
        self.magic_node_positions =
            self.magic_node_ids.iter().map(|&id| topo.node(id).pos).collect();
        let n_topo_nodes = topo.n_nodes;
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            topo.cultivation_times[id as usize] = self.draw(n_topo_nodes);
            topo.busy_counts[id as usize] = 0;
        }
    }

    /// Returns the number of available (ready) magic nodes.
    /// New cultivation times are drawn before the update loop to avoid
    /// interleaving RNG calls with the state mutation.
    pub(crate) fn update_cultivators(&mut self, topo: &mut TopoGraph, used: &[bool]) -> usize {
        let n_topo_nodes = topo.n_nodes;
        self.new_cultivation_times.clear();
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            if used[id as usize] {
                let t = self.draw(n_topo_nodes);
                self.new_cultivation_times.push(t);
            }
        }
        let mut n_ready = 0;
        let mut cultivation_time_index = 0;
        self.ready_magic_positions.clear();
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            if used[id as usize] {
                // Node was consumed this lcycle: assign a fresh cultivation time.
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
                n_ready += 1;
                self.ready_magic_positions.push(self.magic_node_positions[i]);
            }
        }
        self.ready_magic_positions
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        n_ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;
    use crate::topograph::TopoGraph;

    #[test]
    fn new_creates_empty_manager() {
        let mgr = CultivationManager::new(42);
        assert_eq!(mgr.t_products_remaining, 0);
        assert!(mgr.cultivation_times_log.is_empty());
        assert!(mgr.magic_node_ids.is_empty());
        assert!(mgr.magic_node_positions.is_empty());
        assert!(mgr.ready_magic_positions.is_empty());
    }

    #[test]
    fn set_lambda_valid_returns_ok() {
        let mut mgr = CultivationManager::new(1);
        let result = mgr.set_lambda(1.0, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn set_lambda_large_value_returns_ok() {
        let mut mgr = CultivationManager::new(1);
        let result = mgr.set_lambda(100.0, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn set_lambda_rejects_floor_at_or_above_the_mean() {
        let mut mgr = CultivationManager::new(1);
        // 1/lambda = 25, so a floor of 25 leaves the exponential no mass.
        assert!(mgr.set_lambda(0.04, 25.0).is_err());
        assert!(mgr.set_lambda(0.04, 30.0).is_err());
        assert!(mgr.set_lambda(0.04, 24.0).is_ok());
    }

    #[test]
    fn set_lambda_rejects_negative_floor() {
        let mut mgr = CultivationManager::new(1);
        assert!(mgr.set_lambda(0.04, -1.0).is_err());
    }

    #[test]
    fn floor_shifts_the_distribution_but_preserves_its_mean() {
        // A floor is a pure shape change: every draw is at least the floor,
        // while the mean stays 1/lambda, because the exponential component's
        // mean is reduced by exactly the floor.
        let mean = |floor: f64| {
            let mut mgr = CultivationManager::new(11);
            mgr.set_lambda(0.04, floor).unwrap();
            mgr.fill_pool(20_000);
            let pool = &mgr.cultivation_time_pool;
            assert!(
                pool.iter().all(|&t| t as f64 >= floor.floor()),
                "a draw fell below the floor {floor}"
            );
            pool.iter().map(|&t| t as f64).sum::<f64>() / pool.len() as f64
        };
        let (m0, m10, m20) = (mean(0.0), mean(10.0), mean(20.0));
        for (floor, m) in [(0.0, m0), (10.0, m10), (20.0, m20)] {
            assert!((m - 25.0).abs() < 1.0, "floor {floor} gave mean {m}, expected ~25 (1/lambda)");
        }
    }

    #[test]
    fn zero_floor_reproduces_the_unshifted_pool() {
        // The default must be bit-identical to the memoryless model, so that
        // results predating the floor option remain reproducible.
        let pool = |floor: f64| {
            let mut mgr = CultivationManager::new(29);
            mgr.set_lambda(0.0387396, floor).unwrap();
            mgr.fill_pool(500);
            mgr.cultivation_time_pool.clone()
        };
        assert_eq!(pool(0.0), pool(0.0));
        assert_ne!(pool(0.0), pool(5.0));
    }

    #[test]
    fn fill_pool_produces_correct_count() {
        let mut mgr = CultivationManager::new(7);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.fill_pool(10);
        assert_eq!(mgr.cultivation_time_pool.len(), 10);
    }

    #[test]
    fn fill_pool_resets_pool_index() {
        let mut mgr = CultivationManager::new(7);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.fill_pool(5);
        mgr.t_products_remaining = 10;
        let _ = mgr.draw(20);
        let _ = mgr.draw(20);
        mgr.fill_pool(5);
        assert_eq!(mgr.pool_index, 0);
    }

    #[test]
    fn fill_pool_zero_produces_empty_pool() {
        let mut mgr = CultivationManager::new(7);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.fill_pool(0);
        assert!(mgr.cultivation_time_pool.is_empty());
    }

    #[test]
    fn draw_returns_value_from_pool() {
        let mut mgr = CultivationManager::new(99);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.fill_pool(5);
        let first = mgr.cultivation_time_pool[0];
        let drawn = mgr.draw(10);
        assert_eq!(drawn, first);
        assert_eq!(mgr.pool_index, 1);
    }

    #[test]
    fn draw_increments_pool_index() {
        let mut mgr = CultivationManager::new(3);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.fill_pool(3);
        assert_eq!(mgr.pool_index, 0);
        let _ = mgr.draw(10);
        assert_eq!(mgr.pool_index, 1);
        let _ = mgr.draw(10);
        assert_eq!(mgr.pool_index, 2);
    }

    #[test]
    fn draw_refills_when_pool_exhausted() {
        let mut mgr = CultivationManager::new(5);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 2;
        mgr.fill_pool(2);
        let _ = mgr.draw(10);
        let _ = mgr.draw(10);
        let _ = mgr.draw(10);
        assert_eq!(mgr.pool_index, 1);
    }

    #[test]
    fn init_magic_nodes_populates_ids_and_positions() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(11);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 10;
        mgr.fill_pool(100);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        let n_magic_before = topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).count();

        mgr.init_magic_nodes(&mut topo);

        assert_eq!(mgr.magic_node_ids.len(), n_magic_before);
        assert_eq!(mgr.magic_node_positions.len(), n_magic_before);
    }

    #[test]
    fn init_magic_nodes_assigns_cultivation_times() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(13);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 10;
        mgr.fill_pool(100);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        for &id in &mgr.magic_node_ids {
            assert!(topo.cultivation_times[id as usize] >= 0);
        }
    }

    #[test]
    fn update_cultivators_returns_count_of_ready_magic_nodes() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(17);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        for &id in &mgr.magic_node_ids {
            topo.cultivation_times[id as usize] = 0;
        }

        let used = vec![false; topo.n_nodes];
        let n_ready = mgr.update_cultivators(&mut topo, &used);
        assert_eq!(n_ready, mgr.magic_node_ids.len());
    }

    #[test]
    fn update_cultivators_used_nodes_get_new_cultivation_time() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(19);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        if mgr.magic_node_ids.is_empty() {
            return;
        }

        let first_id = mgr.magic_node_ids[0];
        let mut used = vec![false; topo.n_nodes];
        used[first_id as usize] = true;

        topo.cultivation_times[first_id as usize] = 0;

        mgr.update_cultivators(&mut topo, &used);

        assert_eq!(topo.busy_counts[first_id as usize], 0);
    }

    #[test]
    fn update_cultivators_increments_busy_count_for_cultivating_nodes() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(23);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        if mgr.magic_node_ids.is_empty() {
            return;
        }

        let first_id = mgr.magic_node_ids[0];
        topo.cultivation_times[first_id as usize] = 5;
        topo.busy_counts[first_id as usize] = 0;

        let used = vec![false; topo.n_nodes];
        mgr.update_cultivators(&mut topo, &used);

        assert_eq!(topo.busy_counts[first_id as usize], 1);
    }

    #[test]
    fn update_cultivators_logs_completed_cultivation() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(29);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);

        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        if mgr.magic_node_ids.is_empty() {
            return;
        }

        let first_id = mgr.magic_node_ids[0];
        topo.cultivation_times[first_id as usize] = 3;
        topo.busy_counts[first_id as usize] = 2;

        let used = vec![false; topo.n_nodes];
        mgr.update_cultivators(&mut topo, &used);

        assert!(mgr.cultivation_times_log.contains(&3));
        assert_eq!(topo.cultivation_times[first_id as usize], 0);
        assert_eq!(topo.busy_counts[first_id as usize], 0);
    }

    #[test]
    fn update_cultivators_ready_positions_sorted_by_x() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(31);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);

        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        mgr.init_magic_nodes(&mut topo);

        for &id in &mgr.magic_node_ids {
            topo.cultivation_times[id as usize] = 0;
        }

        let used = vec![false; topo.n_nodes];
        mgr.update_cultivators(&mut topo, &used);

        let xs: Vec<f32> = mgr.ready_magic_positions.iter().map(|p| p.0).collect();
        for w in xs.windows(2) {
            assert!(w[0] <= w[1], "ready_magic_positions not sorted by x: {:?}", xs);
        }
    }
}
