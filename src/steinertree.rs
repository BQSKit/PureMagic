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
    visited: Vec<Option<usize>>,
    paths: Vec<Vec<usize>>,
    queue: VecDeque<usize>,
    early_terminations: usize,
    num_calls: usize,
    termination_threshold: usize,
}

impl SteinerTreeComputation {
    /// Creates a new Steiner tree computation state.
    /// `termination_threshold` controls early exit to avoid long tail computations.
    pub fn new(num_nodes: usize, termination_threshold: usize) -> Self {
        SteinerTreeComputation { num_nodes: num_nodes,
                                 visited: vec![None; num_nodes],
                                 paths: vec![Vec::with_capacity(num_nodes); num_nodes],
                                 queue: VecDeque::with_capacity(num_nodes),
                                 early_terminations: 0,
                                 num_calls: 0,
                                 termination_threshold: termination_threshold }
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
    pub fn compute(&mut self, topo: &TopoGraph, used: &Vec<bool>, root_ids: &Vec<usize>,
                   terminal_nodes: &Vec<usize>, gate_type: GateType, num_scheduled: usize)
                   -> Option<TreeGraph> {
        debug_sched!("    BFS from root nodes {:?} to terminal nodes {:?}",
                     root_ids.iter().map(|id| &topo.get_node(*id).label).collect::<Vec<_>>(),
                     terminal_nodes.iter().map(|id| &topo.get_node(*id).label).collect::<Vec<_>>());
        self.num_calls += 1;
        self.clear();
        let mut tree = TreeGraph::new(self.num_nodes);
        let mut cultivator: Option<usize> = None;
        let mut num_paths: usize = 0;
        // every root must have a path to every other root
        let reqd_paths = root_ids.len() * (root_ids.len() - 1);
        debug_sched!("    Require {} paths", reqd_paths);
        for root_id in root_ids {
            self.visited[*root_id] = Some(*root_id);
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
            // add terminals
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

        let max_dist = self.get_max_dist(topo, terminal_nodes) + 1;
        let mut search_steps = 0;
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
                // we have all the paths and terms and a cultivator (if needed), so we can now
                // return the tree (bfs_graph)
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
                #[cfg(debug_assertions)]
                self.check_edges(topo, &tree);
                // FIXME: for XX and ZZ, replace side edges with top/bottom, if that
                // makes the path shorter
                return Some(tree);
            }
            search_steps += 1;
            // early exit to cut off the long tail in computation
            if num_scheduled > 0 && search_steps > max_dist * self.termination_threshold {
                self.early_terminations += 1;
                break;
            }
        }
        None
    }

    /// Computes maximum Manhattan distance between any pair of terminal nodes.
    /// Used to estimate search depth for early termination heuristic.
    fn get_max_dist(&self, topo: &TopoGraph, terminal_nodes: &Vec<usize>) -> usize {
        let mut max_dist = 0;
        for i in 0..terminal_nodes.len() {
            let node_i = topo.get_node(terminal_nodes[i]);
            for j in (i + 1)..terminal_nodes.len() {
                let node_j = topo.get_node(terminal_nodes[j]);
                let manhattan_dist = ((node_i.pos.0 - node_j.pos.0).abs()
                                      + (node_i.pos.1 - node_j.pos.1).abs())
                                     as usize;
                if manhattan_dist > max_dist {
                    max_dist = manhattan_dist;
                }
            }
        }
        max_dist
    }

    /// Explores neighbors of the current node during BFS, tracking path connections
    /// and merging root groups when paths connect. Updates tree with new edges.
    fn visit_neighbors(&mut self, node_id: usize, topo: &TopoGraph, used: &Vec<bool>,
                       reqd_paths: usize, gate_type: GateType,
                       starting_cultivator: Option<usize>, num_start_paths: usize,
                       tree: &mut TreeGraph)
                       -> (usize, Option<usize>) {
        let node = topo.get_node(node_id);
        let curr_root_id = self.visited[node_id].unwrap();
        let mut num_paths = num_start_paths;
        #[cfg(debug_assertions)]
        {
            let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
            assert_eq!(num_paths, curr_num_paths);
        }
        let mut cultivator = starting_cultivator;
        for nb_id in node.nbors.iter() {
            let nb = topo.get_node(*nb_id);
            if used[nb.id] {
                continue;
            }
            if nb.node_type == NodeType::Data {
                // all data nodes are already linked in
                continue;
            }
            // check for path links between roots via routing nodes
            if nb.is_routing() && node.is_routing() && self.visited[*nb_id].is_some() {
                let nb_root_id = self.visited[*nb_id].unwrap();
                if curr_root_id == nb_root_id {
                    continue;
                }
                let curr_root_paths = &self.paths[curr_root_id];
                if !curr_root_paths.contains(&nb_root_id) {
                    // update the nb root IndexSet to contain paths to all the roots in
                    // the curr_root IndexSet
                    let nb_root_paths = self.paths[nb_root_id].clone();
                    // Create merged set containing all roots from both groups
                    let mut merged_set = curr_root_paths.clone();
                    merged_set.push(nb_root_id.clone());
                    merged_set.extend(nb_root_paths.iter().cloned());
                    merged_set.push(curr_root_id.clone());
                    // Update all roots in the merged set to have the complete merged set
                    for root_id in merged_set.iter() {
                        assert!(num_paths >= self.paths[*root_id].len());
                        num_paths -= self.paths[*root_id].len();
                        self.paths[*root_id] = merged_set.clone();
                        // Don't include self
                        let pos =
                            self.paths[*root_id].iter().position(|&id| id == *root_id).unwrap();
                        self.paths[*root_id].swap_remove(pos);
                        debug_sched!("      {}removing self for {}{}",
                                     _GREEN,
                                     topo.get_node(*root_id).label,
                                     _RESET);
                        num_paths += self.paths[*root_id].len();
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
                                let root_label = &topo.get_node(root_id).label;
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
                        // we break here because we previously found a cultivator, and now have
                        // found all the paths
                        break;
                    }
                }
                continue;
            }
            let nb_is_cultivator = gate_type.is_t()
                                   && cultivator.is_none()
                                   && nb.node_type == NodeType::Magic
                                   && nb.cultivation_time == 0;
            // add routing node/cultivator
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
                        // we break here because we previously found all the paths, and now have
                        // found a cultivator
                        break;
                    }
                }
            }
            self.visited[*nb_id] = Some(curr_root_id);
        }
        (num_paths as usize, cultivator)
    }

    /// Returns the total number of compute calls and count of early terminations.
    pub fn get_call_counts(&mut self) -> (usize, usize) {
        (self.num_calls, self.early_terminations)
    }

    /// Validates tree structure in debug builds: ensures data nodes have exactly one edge,
    /// edges are reciprocated, and routing nodes have matching top/bottom data edges.
    #[cfg(debug_assertions)]
    fn check_edges(&self, topo: &TopoGraph, tree: &TreeGraph) {
        for node_id in tree.iter_nodes() {
            let node = topo.get_node(node_id);
            // check that each data node has exactly one edge
            if node.node_type == NodeType::Data {
                let num_edges = tree.get_num_node_edges(node_id);
                assert_eq!(num_edges, 1);
            }
            // check that edges are reciprocated
            for nb_id in node.nbors.iter() {
                let n1n2 = tree.contains_edge(node_id, *nb_id);
                let n2n1 = tree.contains_edge(*nb_id, node_id);
                assert_eq!(n1n2, n2n1);
            }
            // check that if one top or bottom edge exists, so does the other
            if node.is_routing() {
                debug_sched!("    Checking vertical edges for node {}", node.label);
                let (above_count, below_count) = tree.get_num_vertical_data_edges(node_id);
                if above_count > 0 {
                    assert_eq!(above_count, 2,
                               "Routing node {} ({:?}) has {} nbors above",
                               node.label, node.node_type, above_count);
                }
                if below_count > 0 {
                    assert_eq!(below_count, 2,
                               "Routing node {} ({:?}) has {} nbors below",
                               node.label, node.node_type, below_count);
                }
            }
        }
    }
}
