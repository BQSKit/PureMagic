use crate::debug_sched;
use crate::node::NodeType;
use crate::pauliproduct::GateType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
#[allow(unused_imports)]
use colored::Colorize;
use std::collections::VecDeque;

#[inline]
fn is_cultivator_candidate(
    gate_type: GateType, cultivator: Option<u16>, cultivation_time: i32,
) -> bool {
    gate_type.is_t() && cultivator.is_none() && cultivation_time == 0
}

/// State container for greedy multi-source shortest path (Steiner tree) computation.
pub(crate) struct SteinerTreeComputation {
    num_nodes: usize,
    visited: Vec<Option<u16>>,
    paths: Vec<Vec<u16>>,
    queue: VecDeque<u16>,
    pub num_calls: usize,
}

impl SteinerTreeComputation {
    pub(crate) fn new(num_nodes: usize) -> Self {
        SteinerTreeComputation {
            num_nodes: num_nodes,
            visited: vec![None; num_nodes],
            paths: vec![Vec::with_capacity(num_nodes); num_nodes],
            queue: VecDeque::with_capacity(num_nodes),
            num_calls: 0,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.visited.fill(None);
        for path in self.paths.iter_mut() {
            path.clear();
        }
        self.queue.clear();
    }

    /// Greedy BFS connecting all root nodes; returns a routing tree or None.
    pub(crate) fn compute(
        &mut self, topo: &TopoGraph, used: &Vec<bool>, root_ids: &Vec<u16>,
        terminal_nodes: &Vec<u16>, gate_type: GateType,
    ) -> Option<TreeGraph> {
        debug_sched!(
            "    BFS from root nodes {:?} to terminal nodes {:?}",
            root_ids.iter().map(|id| topo.get_label(*id)).collect::<Vec<_>>(),
            terminal_nodes.iter().map(|id| topo.get_label(*id)).collect::<Vec<_>>()
        );
        self.num_calls += 1;
        self.clear();
        let reqd_paths = root_ids.len() * (root_ids.len() - 1);
        debug_sched!("    Require {} paths", reqd_paths);

        let (mut tree, mut cultivator) = self.init_bfs_from_roots(root_ids, terminal_nodes, topo);

        let mut num_paths: usize = 0;
        while let Some(node_id) = self.queue.pop_front() {
            debug_sched!(
                "      {}",
                format!("Visit neighbors of {}", topo.get_label(node_id)).bright_green()
            );
            (num_paths, cultivator) = self.visit_neighbors(
                node_id, topo, used, reqd_paths, gate_type, cultivator, num_paths, &mut tree,
            );
            if num_paths == reqd_paths {
                if gate_type.is_t() && cultivator.is_none() {
                    continue;
                }
                tree.root_node_id = if gate_type.is_t() {
                    debug_sched!(
                        "      {}",
                        format!(
                            "tree complete, cultivator {}",
                            topo.get_label(cultivator.unwrap())
                        )
                        .green()
                    );
                    Some(cultivator.unwrap())
                } else {
                    debug_sched!("      {}", "tree complete".green());
                    Some(root_ids[0])
                };
                let _num_trimmed = tree.trim_dangling_nodes();
                debug_sched!("    Trimmed {} dangling nodes", _num_trimmed);
                if Self::attach_missing_terminals(terminal_nodes, topo, &mut tree).is_none() {
                    return None;
                }
                #[cfg(debug_assertions)]
                self.check_edges(topo, &tree);
                return Some(tree);
            }
        }
        None
    }

    fn init_bfs_from_roots(
        &mut self, root_ids: &[u16], terminal_nodes: &[u16], topo: &TopoGraph,
    ) -> (TreeGraph, Option<u16>) {
        let mut tree = TreeGraph::new(self.num_nodes);
        let mut cultivator: Option<u16> = None;
        for root_id in root_ids {
            self.visited[*root_id as usize] = Some(*root_id);
            self.queue.push_back(*root_id);
            let root = topo.get_node(*root_id);
            debug_sched!("      {}", format!("root node {}", topo.get_label(*root_id)).green());
            if !tree.contains_node(root.id) {
                tree.add_node(root, topo.get_label(*root_id));
            }
            if cultivator.is_none()
                && root.node_type == NodeType::Magic
                && topo.cultivation_times[*root_id as usize] == 0
            {
                cultivator = Some(*root_id);
                debug_sched!(
                    "      {}",
                    format!("found root cultivator {}", topo.get_label(cultivator.unwrap()))
                        .green()
                );
            }
            for nb_id in topo.get_node(*root_id).nbors_slice().iter() {
                let nb = topo.get_node(*nb_id);
                if terminal_nodes.contains(nb_id) {
                    if !tree.contains_node(nb.id) {
                        tree.add_node(nb, topo.get_label(*nb_id));
                    }
                    tree.add_edge(*root_id, *nb_id);
                }
            }
        }
        tree.remove_double_edges();
        (tree, cultivator)
    }

    /// Attaches terminal data nodes not yet in `tree` via a same-row routing neighbor.
    /// Returns `Some(())` on success, or `None` if any terminal cannot be connected.
    fn attach_missing_terminals(
        terminal_nodes: &[u16], topo: &TopoGraph, tree: &mut TreeGraph,
    ) -> Option<()> {
        for &tid in terminal_nodes.iter() {
            if !tree.contains_node(tid) {
                let tid_pos = topo.get_node(tid).pos;
                let conn = topo.get_node(tid).nbors_slice().iter().copied().find(|&nb_id| {
                    tree.contains_node(nb_id)
                        && topo.get_node(nb_id).is_routing()
                        && (topo.get_node(nb_id).pos.1 - tid_pos.1).abs() < 0.01
                });
                if let Some(conn_id) = conn {
                    tree.add_node(topo.get_node(tid), topo.get_label(tid));
                    tree.add_edge(conn_id, tid);
                } else {
                    return None;
                }
            }
        }
        Some(())
    }

    fn visit_neighbors(
        &mut self, node_id: u16, topo: &TopoGraph, used: &Vec<bool>, reqd_paths: usize,
        gate_type: GateType, starting_cultivator: Option<u16>, num_start_paths: usize,
        tree: &mut TreeGraph,
    ) -> (usize, Option<u16>) {
        let node = topo.get_node(node_id);
        let curr_root_id = self.visited[node_id as usize].unwrap();
        let mut num_paths = num_start_paths;
        #[cfg(debug_assertions)]
        {
            let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
            assert_eq!(num_paths, curr_num_paths);
        }
        let mut cultivator = starting_cultivator;
        for nb_id in node.nbors_slice().iter() {
            let nb = topo.get_node(*nb_id);
            if used[nb.id as usize] {
                continue;
            }
            if nb.node_type == NodeType::Data {
                continue;
            }
            if !topo.use_magic_routing && nb.node_type == NodeType::Magic {
                if !is_cultivator_candidate(
                    gate_type,
                    cultivator,
                    topo.cultivation_times[nb.id as usize],
                ) {
                    continue;
                }
            }
            if nb.is_routing() && node.is_routing() && self.visited[*nb_id as usize].is_some() {
                let nb_root_id = self.visited[*nb_id as usize].unwrap();
                if curr_root_id == nb_root_id {
                    continue;
                }
                num_paths = self.merge_root_groups(
                    curr_root_id,
                    nb_root_id,
                    num_paths,
                    reqd_paths,
                    node_id,
                    *nb_id,
                    topo,
                    tree,
                );
                if num_paths == reqd_paths {
                    if gate_type.is_t() && cultivator.is_none() {
                        continue;
                    }
                    break;
                }
                continue;
            }
            let nb_is_cultivator = nb.node_type == NodeType::Magic
                && is_cultivator_candidate(
                    gate_type,
                    cultivator,
                    topo.cultivation_times[nb.id as usize],
                );
            if let Some(new_cultivator) = self.expand_new_neighbor(
                node_id,
                *nb_id,
                nb,
                curr_root_id,
                nb_is_cultivator,
                cultivator,
                topo,
                tree,
            ) {
                cultivator = Some(new_cultivator);
                if num_paths == reqd_paths {
                    break;
                }
            }
        }
        (num_paths as usize, cultivator)
    }

    fn merge_root_groups(
        &mut self, curr_root_id: u16, nb_root_id: u16, num_start_paths: usize,
        #[cfg_attr(not(debug_assertions), allow(unused_variables))] reqd_paths: usize,
        node_id: u16, nb_id: u16,
        #[cfg_attr(not(debug_assertions), allow(unused_variables))] topo: &TopoGraph,
        tree: &mut TreeGraph,
    ) -> usize {
        let mut num_paths = num_start_paths;
        let curr_root_paths = &self.paths[curr_root_id as usize];
        if curr_root_paths.contains(&nb_root_id) {
            return num_paths;
        }
        let nb_root_paths = self.paths[nb_root_id as usize].clone();
        let mut merged_set = curr_root_paths.clone();
        merged_set.push(nb_root_id);
        merged_set.extend(nb_root_paths.iter().cloned());
        merged_set.push(curr_root_id);
        for root_id in merged_set.iter() {
            assert!(num_paths >= self.paths[*root_id as usize].len());
            num_paths -= self.paths[*root_id as usize].len();
            self.paths[*root_id as usize] = merged_set.clone();
            let pos = self.paths[*root_id as usize].iter().position(|&id| id == *root_id).unwrap();
            self.paths[*root_id as usize].swap_remove(pos);
            debug_sched!(
                "      {}",
                format!("removing self for {}", topo.get_label(*root_id)).green()
            );
            num_paths += self.paths[*root_id as usize].len();
        }
        #[cfg(debug_assertions)]
        {
            let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
            assert_eq!(num_paths, curr_num_paths);
        }
        #[cfg(debug_assertions)]
        {
            debug_sched!(
                "      {}",
                format!(
                    "path from {} to {} (total paths {}/{})",
                    topo.get_label(curr_root_id),
                    topo.get_label(nb_root_id),
                    num_paths,
                    reqd_paths,
                )
                .green()
            );
            debug_sched!("      {}", "paths:".green());
            for (root_id, path) in self.paths.iter().enumerate() {
                if !path.is_empty() {
                    let root_label = topo.get_label(root_id as u16);
                    let path_labels: Vec<String> =
                        path.iter().map(|&id| topo.get_label(id).to_string()).collect();
                    debug_sched!("        {} -> {:?}", root_label, path_labels);
                }
            }
        }
        tree.add_edge(node_id, nb_id);
        num_paths
    }

    fn expand_new_neighbor(
        &mut self, node_id: u16, nb_id: u16, nb: &crate::node::Node, curr_root_id: u16,
        nb_is_cultivator: bool, cultivator: Option<u16>, topo: &TopoGraph, tree: &mut TreeGraph,
    ) -> Option<u16> {
        if !nb.is_routing() && !nb_is_cultivator {
            return None;
        }
        if !tree.contains_node(nb.id) {
            tree.add_node(nb, topo.get_label(nb_id));
        }
        if !tree.contains_edge(node_id, nb_id) {
            tree.add_edge(node_id, nb_id);
        }
        self.queue.push_back(nb_id);
        self.visited[nb_id as usize] = Some(curr_root_id);
        if cultivator.is_none() && nb_is_cultivator {
            debug_sched!("      {}", format!("found cultivator {}", topo.get_label(nb_id)).green());
            return Some(nb_id);
        }
        None
    }

    #[cfg(debug_assertions)]
    fn check_edges(&self, topo: &TopoGraph, tree: &TreeGraph) {
        for node_id in tree.iter_nodes() {
            let node = topo.get_node(node_id);
            if node.node_type == NodeType::Data {
                let num_edges = tree.get_num_node_edges(node_id);
                assert_eq!(num_edges, 1);
            }
            for nb_id in node.nbors_slice().iter() {
                let n1n2 = tree.contains_edge(node_id, *nb_id);
                let n2n1 = tree.contains_edge(*nb_id, node_id);
                assert_eq!(n1n2, n2n1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{Node, NodeType};
    use crate::topograph::TopoGraph;

    #[test]
    fn new_initialises_with_zero_calls() {
        let stree = SteinerTreeComputation::new(10);
        assert_eq!(stree.num_calls, 0);
    }

    #[test]
    fn clear_resets_state() {
        let mut stree = SteinerTreeComputation::new(5);
        stree.queue.push_back(1);
        stree.visited[0] = Some(0);
        stree.clear();
        assert!(stree.queue.is_empty());
        assert!(stree.visited.iter().all(|v| v.is_none()));
    }

    #[test]
    fn compute_cx_gate_finds_tree_between_two_data_qubits() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        let num_nodes = topo.num_nodes;
        let used = vec![false; num_nodes];

        let root0 = topo.get_data_node_id(0, 'X');
        let root1 = topo.get_data_node_id(1, 'X');

        let roots: Vec<u16> = [root0, root1]
            .iter()
            .filter_map(|&did| {
                topo.get_node(did)
                    .nbors_slice()
                    .iter()
                    .copied()
                    .find(|&nb| topo.get_node(nb).node_type == NodeType::Magic)
            })
            .collect();

        if roots.len() < 2 {
            return;
        }

        let terminals = vec![root0, root1];
        let mut stree = SteinerTreeComputation::new(num_nodes);
        let result = stree.compute(&topo, &used, &roots, &terminals, GateType::CX);
        assert_eq!(stree.num_calls, 1);
        let _ = result;
    }

    #[test]
    fn compute_increments_num_calls() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        let num_nodes = topo.num_nodes;
        let used = vec![false; num_nodes];
        let root0 = topo.get_data_node_id(0, 'X');
        let roots: Vec<u16> = topo
            .get_node(root0)
            .nbors_slice()
            .iter()
            .copied()
            .filter(|&nb| topo.get_node(nb).node_type == NodeType::Magic)
            .take(1)
            .collect();

        if roots.is_empty() {
            return;
        }

        let terminals = vec![root0];
        let mut stree = SteinerTreeComputation::new(num_nodes);
        let _ = stree.compute(&topo, &used, &roots, &terminals, GateType::M);
        let _ = stree.compute(&topo, &used, &roots, &terminals, GateType::M);
        assert_eq!(stree.num_calls, 2);
    }

    #[test]
    fn compute_t_gate_returns_none_when_no_magic_ready() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        let magic_ids: Vec<u16> =
            topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).map(|n| n.id).collect();
        for id in &magic_ids {
            topo.cultivation_times[*id as usize] = 10;
            topo.busy_counts[*id as usize] = 1;
        }

        let num_nodes = topo.num_nodes;
        let used = vec![false; num_nodes];

        let root0 = topo.get_data_node_id(0, 'X');
        let roots: Vec<u16> = topo
            .get_node(root0)
            .nbors_slice()
            .iter()
            .copied()
            .filter(|&nb| topo.get_node(nb).is_routing())
            .take(1)
            .collect();

        if roots.is_empty() {
            return;
        }

        let terminals = vec![root0];
        let mut stree = SteinerTreeComputation::new(num_nodes);
        let result = stree.compute(&topo, &used, &roots, &terminals, GateType::T);
        assert!(result.is_none());
    }

    #[test]
    fn compute_t_gate_succeeds_when_magic_ready() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);

        let magic_ids: Vec<u16> =
            topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).map(|n| n.id).collect();
        for id in &magic_ids {
            topo.cultivation_times[*id as usize] = 5;
            topo.busy_counts[*id as usize] = 1;
        }
        if let Some(&first_magic) = magic_ids.first() {
            topo.cultivation_times[first_magic as usize] = 0;
            topo.busy_counts[first_magic as usize] = 0;
        }

        let num_nodes = topo.num_nodes;
        let used = vec![false; num_nodes];

        let root0 = topo.get_data_node_id(0, 'X');
        let roots: Vec<u16> = topo
            .get_node(root0)
            .nbors_slice()
            .iter()
            .copied()
            .filter(|&nb| topo.get_node(nb).is_routing())
            .take(1)
            .collect();

        if roots.is_empty() {
            return;
        }

        let terminals = vec![root0];
        let mut stree = SteinerTreeComputation::new(num_nodes);
        let result = stree.compute(&topo, &used, &roots, &terminals, GateType::T);
        assert!(result.is_some());
        if let Some(tree) = result {
            assert!(tree.root_node_id.is_some());
            let root_id = tree.root_node_id.unwrap();
            assert_eq!(topo.get_node(root_id).node_type, NodeType::Magic);
        }
    }
}
