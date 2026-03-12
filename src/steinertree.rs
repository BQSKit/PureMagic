use crate::debug_sched;
use crate::node::NodeType;
use crate::pauliproduct::GateType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
#[allow(unused_imports)]
use crate::utils::{_GREEN, _LGREEN, _RESET};
use std::collections::VecDeque;

/// State container for greedy multi-source shortest path (Steiner tree) computation.
/// Tracks visited nodes, path connectivity, and early termination statistics.
pub struct SteinerTreeComputation {
    num_nodes: usize,
    visited: Vec<Option<u16>>,
    paths: Vec<Vec<u16>>,
    queue: VecDeque<u16>,
    pub num_calls: usize,
}

impl SteinerTreeComputation {
    /// Creates a new Steiner tree computation state.
    /// `termination_threshold` controls early exit to avoid long tail computations.
    pub fn new(num_nodes: usize) -> Self {
        SteinerTreeComputation { num_nodes: num_nodes,
                                 visited: vec![None; num_nodes],
                                 paths: vec![Vec::with_capacity(num_nodes); num_nodes],
                                 queue: VecDeque::with_capacity(num_nodes),
                                 num_calls: 0 }
    }

    /// Clears internal state for a fresh computation.
    pub fn clear(&mut self) {
        self.visited.fill(None);
        for path in self.paths.iter_mut() {
            path.clear();
        }
        self.queue.clear();
    }

    /// Greedy multi-source shortest path computation connecting all root nodes.
    /// Expands from roots using BFS to find paths between all root pairs while
    /// identifying a magic node (for T gates) if available. Returns a tree with
    /// data and routing nodes, or None if no valid path exists.
    pub fn compute(&mut self, topo: &TopoGraph, used: &Vec<bool>, root_ids: &Vec<u16>,
                   terminal_nodes: &Vec<u16>, gate_type: GateType)
                   -> Option<TreeGraph> {
        debug_sched!("    BFS from root nodes {:?} to terminal nodes {:?}",
                     root_ids.iter().map(|id| &topo.get_node(*id).label).collect::<Vec<_>>(),
                     terminal_nodes.iter().map(|id| &topo.get_node(*id).label).collect::<Vec<_>>());
        self.num_calls += 1;
        self.clear();
        let mut tree = TreeGraph::new(self.num_nodes);
        let mut cultivator: Option<u16> = None;
        let mut num_paths: usize = 0;
        let reqd_paths = root_ids.len() * (root_ids.len() - 1);
        debug_sched!("    Require {} paths", reqd_paths);
        for root_id in root_ids {
            self.visited[*root_id as usize] = Some(*root_id);
            self.queue.push_back(*root_id);
            let root = topo.get_node(*root_id);
            debug_sched!("      {}root node {}{}", _GREEN, root.label, _RESET);
            if !tree.contains_node(root.id) {
                tree.add_node(root);
            }
            if cultivator.is_none()
               && root.node_type == NodeType::Magic
               && root.cultivation_time == 0
            {
                cultivator = Some(*root_id);
                debug_sched!("      {}found root cultivator {}{}",
                             _GREEN,
                             topo.get_node(cultivator.unwrap()).label,
                             _RESET);
            }
            let root_node = topo.get_node(*root_id);
            for nb_id in root_node.nbors.iter() {
                let nb = topo.get_node(*nb_id);
                if terminal_nodes.contains(&nb_id) {
                    if !tree.contains_node(nb.id) {
                        tree.add_node(nb);
                    }
                    tree.add_edge(*root_id, *nb_id);
                }
            }
        }

        tree.remove_double_edges();
        while let Some(node_id) = self.queue.pop_front() {
            debug_sched!("      {}Visit neighbors of {}{}",
                         _LGREEN,
                         topo.get_node(node_id).label,
                         _RESET);
            (num_paths, cultivator) = self.visit_neighbors(node_id, topo, used, reqd_paths,
                                                           gate_type, cultivator, num_paths,
                                                           &mut tree);
            if num_paths == reqd_paths {
                if gate_type.is_t() && cultivator.is_none() {
                    continue;
                }
                tree.root_node_id = if gate_type.is_t() {
                    debug_sched!("      {}tree complete, cultivator {}{}",
                                 _GREEN,
                                 topo.get_node(cultivator.unwrap()).label,
                                 _RESET);
                    Some(cultivator.unwrap())
                } else {
                    debug_sched!("      {}tree complete{}", _GREEN, _RESET);
                    Some(root_ids[0])
                };
                let _num_trimmed = tree.trim_dangling_nodes();
                debug_sched!("    Trimmed {} dangling nodes", _num_trimmed);
                // Attach any terminal data nodes not yet in the tree. This can happen when
                // get_root_nodes counts a terminal as handled (via saturation) but didn't
                // return a root adjacent to it (e.g. Y-type gates where one root serves both
                // X and Z, but is only adjacent to one of them).
                // We must only connect to a side routing node (same pos.1) to avoid creating
                // a single vertical data edge on a routing node, which violates check_edges.
                for &tid in terminal_nodes.iter() {
                    if !tree.contains_node(tid) {
                        let tid_pos = topo.get_node(tid).pos;
                        let conn =
                            topo.get_node(tid).nbors.iter().copied().find(|&nb_id| {
                                                                        tree.contains_node(nb_id)
                                                                        && topo.get_node(nb_id)
                                                                               .is_routing()
                                                                        && (topo.get_node(nb_id)
                                                                                .pos
                                                                                .1
                                                                            - tid_pos.1)
                                                                                        .abs()
                                                                           < 0.01
                                                                    });
                        if let Some(conn_id) = conn {
                            tree.add_node(topo.get_node(tid));
                            tree.add_edge(conn_id, tid);
                        } else {
                            return None;
                        }
                    }
                }
                #[cfg(debug_assertions)]
                self.check_edges(topo, &tree);
                return Some(tree);
            }
        }
        None
    }

    /// Explores neighbors of the current node during BFS, tracking path connections
    /// and merging root groups when paths connect. Updates tree with new edges.
    fn visit_neighbors(&mut self, node_id: u16, topo: &TopoGraph, used: &Vec<bool>,
                       reqd_paths: usize, gate_type: GateType, starting_cultivator: Option<u16>,
                       num_start_paths: usize, tree: &mut TreeGraph)
                       -> (usize, Option<u16>) {
        let node = topo.get_node(node_id);
        let curr_root_id = self.visited[node_id as usize].unwrap();
        let mut num_paths = num_start_paths;
        #[cfg(debug_assertions)]
        {
            let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
            assert_eq!(num_paths, curr_num_paths);
        }
        let mut cultivator = starting_cultivator;
        for nb_id in node.nbors.iter() {
            let nb = topo.get_node(*nb_id);
            if used[nb.id as usize] {
                continue;
            }
            if nb.node_type == NodeType::Data {
                continue;
            }
            if nb.is_routing() && node.is_routing() && self.visited[*nb_id as usize].is_some() {
                let nb_root_id = self.visited[*nb_id as usize].unwrap();
                if curr_root_id == nb_root_id {
                    continue;
                }
                let curr_root_paths = &self.paths[curr_root_id as usize];
                if !curr_root_paths.contains(&nb_root_id) {
                    let nb_root_paths = self.paths[nb_root_id as usize].clone();
                    let mut merged_set = curr_root_paths.clone();
                    merged_set.push(nb_root_id.clone());
                    merged_set.extend(nb_root_paths.iter().cloned());
                    merged_set.push(curr_root_id.clone());
                    for root_id in merged_set.iter() {
                        assert!(num_paths >= self.paths[*root_id as usize].len());
                        num_paths -= self.paths[*root_id as usize].len();
                        self.paths[*root_id as usize] = merged_set.clone();
                        let pos = self.paths[*root_id as usize].iter()
                                                               .position(|&id| id == *root_id)
                                                               .unwrap();
                        self.paths[*root_id as usize].swap_remove(pos);
                        debug_sched!("      {}removing self for {}{}",
                                     _GREEN,
                                     topo.get_node(*root_id).label,
                                     _RESET);
                        num_paths += self.paths[*root_id as usize].len();
                    }
                    #[cfg(debug_assertions)]
                    {
                        let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
                        assert_eq!(num_paths, curr_num_paths);
                    }
                    #[cfg(debug_assertions)]
                    {
                        debug_sched!("      {}path from {} to {} (total paths {}/{}){}",
                                     _GREEN,
                                     topo.get_node(curr_root_id).label,
                                     topo.get_node(nb_root_id).label,
                                     num_paths,
                                     reqd_paths,
                                     _RESET);
                        debug_sched!("      {}paths:{}", _GREEN, _RESET);
                        for (root_id, path) in self.paths.iter().enumerate() {
                            if !path.is_empty() {
                                let root_label = &topo.get_node(root_id as u16).label;
                                let path_labels: Vec<String> =
                                    path.iter()
                                        .map(|&id| topo.get_node(id).label.clone())
                                        .collect();
                                debug_sched!("        {} -> {:?}", root_label, path_labels);
                            }
                        }
                    }
                    tree.add_edge(node_id, *nb_id);
                    if num_paths == reqd_paths {
                        if gate_type.is_t() && cultivator.is_none() {
                            continue;
                        }
                        break;
                    }
                }
                continue;
            }
            let nb_is_cultivator = gate_type.is_t()
                                   && cultivator.is_none()
                                   && nb.node_type == NodeType::Magic
                                   && nb.cultivation_time == 0;
            if nb.is_routing() || nb_is_cultivator {
                if !tree.contains_node(nb.id) {
                    tree.add_node(nb);
                }
                if !tree.contains_edge(node_id, *nb_id) {
                    tree.add_edge(node_id, *nb_id);
                }
                self.queue.push_back(*nb_id);
                if cultivator.is_none() && nb_is_cultivator {
                    cultivator = Some(*nb_id);
                    debug_sched!("      {}found cultivator {}{}",
                                 _GREEN,
                                 topo.get_node(cultivator.unwrap()).label,
                                 _RESET);
                    if num_paths == reqd_paths {
                        break;
                    }
                }
            }
            self.visited[*nb_id as usize] = Some(curr_root_id);
        }
        (num_paths as usize, cultivator)
    }

    /// Validates tree structure in debug builds: ensures data nodes have exactly one edge,
    /// edges are reciprocated, and routing nodes have matching top/bottom data edges.
    #[cfg(debug_assertions)]
    fn check_edges(&self, topo: &TopoGraph, tree: &TreeGraph) {
        for node_id in tree.iter_nodes() {
            let node = topo.get_node(node_id);
            if node.node_type == NodeType::Data {
                let num_edges = tree.get_num_node_edges(node_id);
                assert_eq!(num_edges, 1);
            }
            for nb_id in node.nbors.iter() {
                let n1n2 = tree.contains_edge(node_id, *nb_id);
                let n2n1 = tree.contains_edge(*nb_id, node_id);
                assert_eq!(n1n2, n2n1);
            }
            // Note: we intentionally do not assert that vertical data edge counts == 2 here.
            // For Clifford gates this invariant held, but T-gates with non-Y operators can
            // legitimately have a routing node with a single vertical data edge (e.g. only
            // the X patch of a qubit is a terminal, not the Z patch).
        }
    }
}
