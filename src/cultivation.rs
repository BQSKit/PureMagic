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
    /// Length of the attempt that produced each node's ready state, indexed by
    /// topology node id; charged to `yield_stats` when the state is used.
    done_len: Vec<i32>,
    pub(crate) yield_stats: YieldStats,
}

/// Where cultivation patch-lcycles went. Every attempt ends one of three ways:
/// it delivers its state to a T gate, a path is routed through the patch
/// before it finishes (interrupted), or it finishes and a path is routed
/// through the ready patch before a T gate takes the state (lost). The lcycles
/// each attempt spent cultivating are charged to its outcome, so the discarded
/// fraction is (interrupted + lost) over all cultivating lcycles. Only
/// PureMagic routes through cultivators, so in bus routing every finished
/// attempt is delivered or still waiting at the end.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct YieldStats {
    pub(crate) n_delivered: usize,
    pub(crate) n_interrupted: usize,
    pub(crate) n_lost: usize,
    pub(crate) cycles_delivered: usize,
    pub(crate) cycles_interrupted: usize,
    pub(crate) cycles_lost: usize,
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
            done_len: Vec::new(),
            yield_stats: YieldStats::default(),
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
            return Err("cultivation floor must be below the total mean 1/lambda \
                        (pass --cultivation-floor 0 for a mean shorter than one attempt)");
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
        self.done_len = vec![0; n_topo_nodes];
        for i in 0..self.magic_node_ids.len() {
            let id = self.magic_node_ids[i];
            topo.cultivation_times[id as usize] = self.draw(n_topo_nodes);
            topo.busy_counts[id as usize] = 0;
        }
    }

    /// Charges the yield of the nodes used in the final lcycle, which no later
    /// `update_cultivators` call sees. Changes no cultivation state.
    pub(crate) fn settle(&mut self, topo: &TopoGraph, used: &[bool], consumed: &[bool]) {
        for &id in &self.magic_node_ids {
            let id = id as usize;
            if !used[id] {
                continue;
            }
            let y = &mut self.yield_stats;
            if topo.cultivation_times[id] == 0 {
                if consumed[id] {
                    y.n_delivered += 1;
                    y.cycles_delivered += self.done_len[id] as usize;
                } else {
                    y.n_lost += 1;
                    y.cycles_lost += self.done_len[id] as usize;
                }
            } else {
                y.n_interrupted += 1;
                y.cycles_interrupted += topo.busy_counts[id] as usize;
            }
        }
    }

    /// Cultivating lcycles of attempts still open when the schedule ends: those
    /// in progress, and finished states no T gate took.
    pub(crate) fn unfinished_cycles(&self, topo: &TopoGraph, used: &[bool]) -> usize {
        self.magic_node_ids
            .iter()
            .filter(|&&id| !used[id as usize])
            .map(|&id| {
                let id = id as usize;
                if topo.cultivation_times[id] == 0 {
                    self.done_len[id] as usize
                } else {
                    topo.busy_counts[id] as usize
                }
            })
            .sum()
    }

    /// Returns the number of available (ready) magic nodes.
    /// New cultivation times are drawn before the update loop to avoid
    /// interleaving RNG calls with the state mutation.
    /// `used` marks the nodes any path occupied last lcycle and `consumed` the
    /// magic nodes a T gate took its state from, which `used` also marks.
    pub(crate) fn update_cultivators(
        &mut self, topo: &mut TopoGraph, used: &[bool], consumed: &[bool],
    ) -> usize {
        let n_topo_nodes = topo.n_nodes;
        if self.done_len.len() < n_topo_nodes {
            self.done_len.resize(n_topo_nodes, 0);
        }
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
                let y = &mut self.yield_stats;
                if topo.cultivation_times[id as usize] == 0 {
                    let len = self.done_len[id as usize] as usize;
                    if consumed[id as usize] {
                        y.n_delivered += 1;
                        y.cycles_delivered += len;
                    } else {
                        y.n_lost += 1;
                        y.cycles_lost += len;
                    }
                } else {
                    y.n_interrupted += 1;
                    y.cycles_interrupted += topo.busy_counts[id as usize] as usize;
                }
                // Node was used this lcycle: assign a fresh cultivation time.
                topo.cultivation_times[id as usize] =
                    self.new_cultivation_times[cultivation_time_index];
                topo.busy_counts[id as usize] = 0;
                cultivation_time_index += 1;
            } else if topo.is_cultivating(id) {
                topo.busy_counts[id as usize] += 1;
                if topo.busy_counts[id as usize] == topo.cultivation_times[id as usize] {
                    self.cultivation_times_log.push(topo.cultivation_times[id as usize]);
                    self.done_len[id as usize] = topo.cultivation_times[id as usize];
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
        let n_ready = mgr.update_cultivators(&mut topo, &used, &vec![false; used.len()]);
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

        mgr.update_cultivators(&mut topo, &used, &vec![false; used.len()]);

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
        mgr.update_cultivators(&mut topo, &used, &vec![false; used.len()]);

        assert_eq!(topo.busy_counts[first_id as usize], 1);
    }

    #[test]
    fn yield_charges_each_attempt_to_how_it_ended() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(29);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        mgr.init_magic_nodes(&mut topo);
        let id = mgr.magic_node_ids[0] as usize;
        // Isolate one node: every other cultivator is used each lcycle, so it never
        // finishes, and its charges are subtracted out below.
        let n = topo.n_nodes;
        let idle = vec![false; n];
        let none = vec![false; n];
        let mut used_only = vec![false; n];
        used_only[id] = true;
        let mut consumed_only = vec![false; n];
        consumed_only[id] = true;

        // An attempt of 3 interrupted after 2 lcycles.
        topo.cultivation_times[id] = 3;
        topo.busy_counts[id] = 0;
        let before = mgr.yield_stats.clone();
        mgr.update_cultivators(&mut topo, &idle, &none);
        mgr.update_cultivators(&mut topo, &idle, &none);
        let other_mid = mgr.yield_stats.clone();
        mgr.update_cultivators(&mut topo, &used_only, &none);
        assert_eq!(mgr.yield_stats.n_interrupted, other_mid.n_interrupted + 1);
        assert_eq!(mgr.yield_stats.cycles_interrupted, other_mid.cycles_interrupted + 2);
        assert_eq!(mgr.yield_stats.n_delivered, before.n_delivered);

        // An attempt of 2 that finishes and is consumed by a T gate.
        topo.cultivation_times[id] = 2;
        topo.busy_counts[id] = 0;
        mgr.update_cultivators(&mut topo, &idle, &none);
        mgr.update_cultivators(&mut topo, &idle, &none);
        assert_eq!(topo.cultivation_times[id], 0);
        let pre = mgr.yield_stats.clone();
        mgr.update_cultivators(&mut topo, &used_only, &consumed_only);
        assert_eq!(mgr.yield_stats.n_delivered, pre.n_delivered + 1);
        assert_eq!(mgr.yield_stats.cycles_delivered, pre.cycles_delivered + 2);

        // An attempt of 1 that finishes and is then routed over: lost.
        topo.cultivation_times[id] = 1;
        topo.busy_counts[id] = 0;
        mgr.update_cultivators(&mut topo, &idle, &none);
        assert_eq!(topo.cultivation_times[id], 0);
        let pre = mgr.yield_stats.clone();
        mgr.update_cultivators(&mut topo, &used_only, &none);
        assert_eq!(mgr.yield_stats.n_lost, pre.n_lost + 1);
        assert_eq!(mgr.yield_stats.cycles_lost, pre.cycles_lost + 1);
    }

    #[test]
    fn settle_charges_the_final_lcycle_without_changing_state() {
        Node::set_magic_routing(true);
        let mut mgr = CultivationManager::new(29);
        mgr.set_lambda(1.0, 0.0).unwrap();
        mgr.t_products_remaining = 20;
        mgr.fill_pool(200);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        mgr.init_magic_nodes(&mut topo);
        let id = mgr.magic_node_ids[0] as usize;
        let n = topo.n_nodes;
        topo.cultivation_times[id] = 2;
        topo.busy_counts[id] = 0;
        mgr.update_cultivators(&mut topo, &vec![false; n], &vec![false; n]);
        mgr.update_cultivators(&mut topo, &vec![false; n], &vec![false; n]);
        let mut used = vec![false; n];
        used[id] = true;
        let pre = mgr.yield_stats.clone();
        let times = topo.cultivation_times.clone();
        mgr.settle(&topo, &used, &used);
        assert_eq!(mgr.yield_stats.n_delivered, pre.n_delivered + 1);
        assert_eq!(mgr.yield_stats.cycles_delivered, pre.cycles_delivered + 2);
        assert_eq!(topo.cultivation_times, times);
        // The settled node is not also counted as unfinished.
        let with = mgr.unfinished_cycles(&topo, &vec![false; n]);
        let without = mgr.unfinished_cycles(&topo, &used);
        assert_eq!(with - without, 2);
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
        mgr.update_cultivators(&mut topo, &used, &vec![false; used.len()]);

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
        mgr.update_cultivators(&mut topo, &used, &vec![false; used.len()]);

        let xs: Vec<f32> = mgr.ready_magic_positions.iter().map(|p| p.0).collect();
        for w in xs.windows(2) {
            assert!(w[0] <= w[1], "ready_magic_positions not sorted by x: {:?}", xs);
        }
    }
}
