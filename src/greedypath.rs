use crate::astar::PathResult;
use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;

pub struct GreedyPathComputation {
    visited: Vec<bool>,
    backtracked: Vec<bool>,
    pub num_calls: usize,
}

impl GreedyPathComputation {
    pub fn new(num_nodes: usize) -> Self {
        GreedyPathComputation {
            visited: vec![false; num_nodes],
            backtracked: vec![false; num_nodes],
            num_calls: 0,
        }
    }

    /// Greedy walk from root to the nearest ready magic node.
    /// Always steps to the open neighbour closest to the target magic node.
    /// On dead-end, backtracks along the walked path (marking nodes as "revisited")
    /// until it finds a node with open neighbours, then resumes greedy walk.
    /// Terminates when it reaches a ready magic node or exhausts all options.
    /// When `plotting` is false, marks `used[]` directly and returns `PathFound(None)`.
    /// When `plotting` is true, builds and returns `PathFound(Some(tree))`.
    /// Returns `NoPath` if no path exists.
    pub fn compute(
        &mut self, terminal_ids: &[u16], root_ids: &[u16], topo: &TopoGraph, used: &mut Vec<bool>,
        ready_magic_positions: &[(f32, f32)], plotting: bool,
    ) -> PathResult {
        self.num_calls += 1;
        self.visited.fill(false);
        self.backtracked.fill(false);
        let root_id = root_ids[0];
        debug_assert!(!used[root_id as usize]);
        let ready_idx = heuristic(topo.get_node(root_id).pos, ready_magic_positions).1;
        let target_pos = ready_magic_positions[ready_idx as usize];
        let mut path: Vec<u16> = vec![root_id];
        self.visited[root_id as usize] = true;

        loop {
            let current = *path.last().unwrap();
            let current_node = topo.get_node(current);
            if current_node.node_type == NodeType::Magic
                && topo.cultivation_times[current_node.id as usize] == 0
                && !used[current as usize]
            {
                if !plotting {
                    // Mark path nodes used directly; skip TreeGraph allocation.
                    for &nid in &path {
                        used[nid as usize] = true;
                    }
                    for &tid in terminal_ids {
                        used[tid as usize] = true;
                    }
                    return PathResult::PathFound(None);
                }
                return PathResult::PathFound(Some(build_tree(
                    &path,
                    terminal_ids,
                    root_ids,
                    topo,
                )));
            }

            // Find the best open neighbour: not used, not data, not visited, not revisited,
            // closest manhattan distance to target.
            // When use_magic_routing is false, magic nodes may only be stepped onto as the
            // final goal (ready + unused); they must not be used as routing intermediaries.
            let best_nb = current_node
                .nbors_slice()
                .iter()
                .copied()
                .filter(|&nb_id| {
                    if used[nb_id as usize] {
                        return false;
                    }
                    if self.visited[nb_id as usize] || self.backtracked[nb_id as usize] {
                        return false;
                    }
                    let nb = topo.get_node(nb_id);
                    if nb.node_type == NodeType::Data {
                        return false;
                    }
                    if !topo.use_magic_routing && nb.node_type == NodeType::Magic {
                        // Only allow stepping onto a magic node if it is the goal
                        let is_goal = topo.cultivation_times[nb_id as usize] == 0;
                        if !is_goal {
                            return false;
                        }
                    }
                    true
                })
                .min_by_key(|&nb_id| manhattan_dist(topo.get_node(nb_id).pos, target_pos));

            if let Some(nb_id) = best_nb {
                self.visited[nb_id as usize] = true;
                path.push(nb_id);
            } else {
                self.visited[current as usize] = false;
                self.backtracked[current as usize] = true;
                path.pop();
                if path.is_empty() {
                    return PathResult::NoPath;
                }
                loop {
                    let backtrack_node = *path.last().unwrap();
                    let has_open_nb =
                        topo.get_node(backtrack_node).nbors_slice().iter().copied().any(|nb_id| {
                            if used[nb_id as usize]
                                || self.visited[nb_id as usize]
                                || self.backtracked[nb_id as usize]
                            {
                                return false;
                            }
                            let nb = topo.get_node(nb_id);
                            if nb.node_type == NodeType::Data {
                                return false;
                            }
                            if nb.node_type == NodeType::Magic {
                                // Not-yet-ready magic nodes are never usable
                                if topo.cultivation_times[nb_id as usize] > 0 {
                                    return false;
                                }
                                // When not using magic routing, magic nodes are only valid
                                // as the final goal, not as routing intermediaries
                                if !topo.use_magic_routing {
                                    return false;
                                }
                            }
                            true
                        });
                    if has_open_nb {
                        break;
                    } else {
                        self.visited[backtrack_node as usize] = false;
                        self.backtracked[backtrack_node as usize] = true;
                        path.pop();
                        if path.is_empty() {
                            return PathResult::NoPath;
                        }
                    }
                }
            }
        }
    }
}

/// Builds a TreeGraph from the walked path and attaches terminal nodes.
fn build_tree(path: &[u16], terminal_ids: &[u16], root_ids: &[u16], topo: &TopoGraph) -> TreeGraph {
    let mut tree = TreeGraph::new(topo.num_nodes);
    tree.root_node_id = Some(*path.last().unwrap());
    for &node_id in path {
        if !tree.contains_node(node_id) {
            tree.add_node(topo.get_node(node_id), topo.get_label(node_id));
        }
    }
    for window in path.windows(2) {
        tree.add_edge(window[0], window[1]);
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
            }
        }
        if i < terminal_ids.len() {
            let tid = terminal_ids[i];
            if !tree.contains_node(tid) {
                tree.add_node(topo.get_node(tid), topo.get_label(tid));
            }
            tree.add_edge(root_id, tid);
        }
    }
    tree
}

fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> (u32, u16) {
    ready_magic_positions
        .iter()
        .enumerate()
        .map(|(idx, &mp)| (manhattan_dist(mp, pos), idx as u16))
        .min_by(|(da, _), (db, _)| da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap()
}

fn manhattan_dist(p1: (f32, f32), p2: (f32, f32)) -> u32 {
    ((p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()) as u32
}
