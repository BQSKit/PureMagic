use crate::debug_sched;
use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
#[cfg(debug_assertions)]
use crate::utils::{GREEN, RESET};
use std::collections::VecDeque;

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
    pub fn new(num_nodes: usize, termination_threshold: usize) -> Self {
        SteinerTreeComputation { num_nodes: num_nodes,
                                 visited: vec![None; num_nodes],
                                 paths: vec![Vec::with_capacity(num_nodes); num_nodes],
                                 queue: VecDeque::with_capacity(num_nodes),
                                 early_terminations: 0,
                                 num_calls: 0,
                                 termination_threshold: termination_threshold }
    }

    pub fn clear(&mut self) {
        self.visited.fill(None);
        for path in self.paths.iter_mut() {
            path.clear();
        }
        self.queue.clear();
    }

    // this can be viewed as a greedy multi-source shortest path algorithm
    pub fn get_steiner_tree(&mut self, topo: &TopoGraph, used: &Vec<bool>,
                            root_ids: &Vec<usize>, terminal_nodes: &Vec<usize>, is_tgate: bool,
                            num_scheduled: usize)
                            -> Option<TreeGraph> {
        debug_sched!("    BFS from nodes {:?} to nodes {:?}", root_ids, terminal_nodes);
        self.num_calls += 1;
        self.clear();
        let mut tree = TreeGraph::new(self.num_nodes);
        let mut cultivator: Option<usize> = None;
        let mut num_paths: usize = 0;
        debug_sched!("    Number of root labels {}", root_ids.len());
        // every root must have a path to every other root
        let reqd_paths = root_ids.len() * (root_ids.len() - 1);
        debug_sched!("    Require {} paths", reqd_paths);
        for root_id in root_ids {
            debug_sched!("      {}root node {}{}", GREEN, root_id, RESET);
            self.visited[*root_id] = Some(*root_id);
            self.queue.push_back(*root_id);
            let root = topo.get_node(*root_id);
            tree.add_node(root.id, root.is_routing());
            if cultivator.is_none()
               && root.node_type == NodeType::Magic
               && root.cultivation_time == 0
            {
                cultivator = Some(*root_id);
                debug_sched!("      {}found root cultivator {}{}",
                             GREEN,
                             cultivator.unwrap(),
                             RESET);
            }
            // add terminals
            let root_node = topo.get_node(*root_id);
            for nb_id in root_node.nbors.iter() {
                let nb = topo.get_node(*nb_id);
                if terminal_nodes.contains(&nb_id) {
                    tree.add_node(nb.id, nb.is_routing());
                    tree.add_edge(*root_id, *nb_id);
                    debug_sched!("      {}add node {}{}", GREEN, nb_id, RESET);
                    debug_sched!("      {}add edge {}->{}{}", GREEN, root_id, nb_id, RESET);
                }
            }
        }

        let max_dist = self.get_max_dist(topo, terminal_nodes) + 1;
        let mut search_steps = 0;
        while let Some(node_id) = self.queue.pop_front() {
            (num_paths, cultivator) = self.visit_neighbors(node_id, topo, used, reqd_paths,
                                                           is_tgate, cultivator, num_paths,
                                                           &mut tree);
            if num_paths == reqd_paths {
                if is_tgate && cultivator.is_none() {
                    continue;
                }
                // we have all the paths and terms and a cultivator (if needed), so we can now
                // return the tree (bfs_graph)
                tree.root_node_id = if is_tgate {
                    debug_sched!("      {}tree complete, cultivator {}{}",
                                 GREEN,
                                 cultivator.unwrap(),
                                 RESET);
                    Some(cultivator.unwrap())
                } else {
                    debug_sched!("      {}tree complete{}", GREEN, RESET);
                    Some(root_ids[0])
                };
                let _num_trimmed = tree.trim_dangling_nodes();
                debug_sched!("    Trimmed {} dangling nodes", _num_trimmed);
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

    fn visit_neighbors(&mut self, node_id: usize, topo: &TopoGraph, used: &Vec<bool>,
                       reqd_paths: usize, is_tgate: bool, starting_cultivator: Option<usize>,
                       num_start_paths: usize, tree: &mut TreeGraph)
                       -> (usize, Option<usize>) {
        let node = topo.get_node(node_id);
        let curr_root_id = self.visited[node_id].unwrap();
        let mut num_paths = num_start_paths;
        #[cfg(debug_assertions)]
        {
            let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
            debug_assert_eq!(num_paths, curr_num_paths);
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
                        num_paths += self.paths[*root_id].len();
                    }
                    #[cfg(debug_assertions)]
                    {
                        let curr_num_paths = self.paths.iter().map(|set| set.len()).sum::<usize>();
                        debug_assert_eq!(num_paths, curr_num_paths);
                    }
                    debug_sched!("      {}path from {} to {} (total paths {}/{}){}",
                                 GREEN,
                                 curr_root_id,
                                 nb_root_id,
                                 num_paths,
                                 reqd_paths,
                                 RESET);
                    debug_sched!("      {}paths:{:?}{}", GREEN, self.paths, RESET);
                    tree.add_edge(node_id, *nb_id);
                    debug_sched!("      {}add edge {}->{}{}", GREEN, node_id, nb_id, RESET);
                    if num_paths == reqd_paths {
                        if is_tgate && cultivator.is_none() {
                            continue;
                        }
                        // we break here because we previously found a cultivator, and now have
                        // found all the paths
                        break;
                    }
                }
                continue;
            }
            let nb_is_cultivator = is_tgate
                                   && cultivator.is_none()
                                   && nb.node_type == NodeType::Magic
                                   && nb.cultivation_time == 0;
            // add routing node/cultivator
            if nb.is_routing() || nb_is_cultivator {
                tree.add_node(nb.id, nb.is_routing());
                tree.add_edge(node_id, *nb_id);
                debug_sched!("      {}add node {}{}", GREEN, nb_id, RESET);
                debug_sched!("      {}add edge {}->{}{}", GREEN, node_id, nb_id, RESET);
                self.queue.push_back(*nb_id);
                if cultivator.is_none() && nb_is_cultivator {
                    cultivator = Some(*nb_id);
                    debug_sched!("      {}found clutivator {}{}",
                                 GREEN,
                                 cultivator.unwrap(),
                                 RESET);
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

    pub fn get_call_counts(&mut self) -> (usize, usize) {
        (self.num_calls, self.early_terminations)
    }
}
