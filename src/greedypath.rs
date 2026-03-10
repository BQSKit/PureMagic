use crate::node::NodeType;
use crate::topograph::TopoGraph;
use crate::treegraph::TreeGraph;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

// Per-rayon-thread reusable backtrack buffer for compute_parallel.
// Avoids per-task allocation and allocator contention when threads run concurrently.
thread_local! {
    static BACKTRACKED: RefCell<Vec<bool>> = RefCell::new(Vec::new());
}

pub struct GreedyPathComputation {
    visited: Vec<bool>,
    backtracked: Vec<bool>,
}

impl GreedyPathComputation {
    pub fn new(num_nodes: usize) -> Self {
        GreedyPathComputation { visited: vec![false; num_nodes],
                                backtracked: vec![false; num_nodes] }
    }

    /// Greedy walk from root to the nearest ready magic node.
    /// Always steps to the open neighbour closest to the target magic node.
    /// On dead-end, backtracks along the walked path (marking nodes as "revisited")
    /// until it finds a node with open neighbours, then resumes greedy walk.
    /// Terminates when it reaches a ready magic node or exhausts all options.
    pub fn compute(&mut self, terminal_ids: &[usize], root_ids: &[usize], topo: &TopoGraph,
                   used: &[bool], ready_magic_positions: &[(f32, f32)])
                   -> Option<TreeGraph> {
        self.visited.fill(false);
        self.backtracked.fill(false);
        let root_id = root_ids[0];
        debug_assert!(!used[root_id]);
        // Pick target magic node using heuristic from root
        let (_, ready_idx) = heuristic(topo.get_node(root_id).pos, ready_magic_positions);
        let target_pos = ready_magic_positions[ready_idx];
        // path stack: the current walk from root to current node
        let mut path: Vec<usize> = vec![root_id];
        self.visited[root_id] = true;

        loop {
            let current = *path.last().unwrap();
            let current_node = topo.get_node(current);
            // Check if we reached a ready magic node
            if current_node.node_type == NodeType::Magic
               && current_node.cultivation_time == 0
               && !used[current]
            {
                return Some(build_tree(&path, terminal_ids, root_ids, topo));
            }

            // Find the best open neighbour: not used, not data, not visited, not revisited,
            // closest manhattan distance to target
            let best_nb =
                current_node.nbors
                            .iter()
                            .copied()
                            .filter(|&nb_id| {
                                if used[nb_id] {
                                    return false;
                                }
                                if self.visited[nb_id] || self.backtracked[nb_id] {
                                    return false;
                                }
                                let nb = topo.get_node(nb_id);
                                if nb.node_type == NodeType::Data {
                                    return false;
                                }
                                true
                            })
                            .min_by_key(|&nb_id| {
                                manhattan_dist(topo.get_node(nb_id).pos, target_pos)
                            });

            if let Some(nb_id) = best_nb {
                // Step forward
                self.visited[nb_id] = true;
                path.push(nb_id);
            } else {
                // Dead-end: backtrack
                self.visited[current] = false;
                self.backtracked[current] = true;
                path.pop();
                if path.is_empty() {
                    return None;
                }
                // Walk back further until we find a node with open neighbours
                loop {
                    let backtrack_node = *path.last().unwrap();
                    let has_open_nb =
                        topo.get_node(backtrack_node).nbors.iter().copied().any(|nb_id| {
                                                                               !used[nb_id]
                                && !self.visited[nb_id] && !self.backtracked[nb_id]
                                && topo.get_node(nb_id).node_type != NodeType::Data
                                && !(topo.get_node(nb_id).node_type == NodeType::Magic
                                     && topo.get_node(nb_id).cultivation_time > 0)
                                                                           });
                    if has_open_nb {
                        break;
                    } else {
                        self.visited[backtrack_node] = false;
                        self.backtracked[backtrack_node] = true;
                        path.pop();
                        if path.is_empty() {
                            return None;
                        }
                    }
                }
            }
        }
    }
}

/// Parallel greedy walk for use in multi-threaded T-gate scheduling.
///
/// Identical in logic to `GreedyPathComputation::compute`, but uses a shared
/// `&[AtomicBool]` for the visited set so that concurrent walks on different
/// products cannot claim the same routing node. Ownership of a node is
/// established via compare-and-swap; on backtrack the node is released back.
///
/// `backtracked` is a per-thread scratch buffer (caller-owned, reset on entry).
/// `target_magic_pos` is the pre-assigned target magic node position for this
/// walk; threads are given different positions so they tend toward different
/// areas of the topology, reducing contention.
pub fn compute_parallel(terminal_ids: &[usize], root_ids: &[usize], topo: &TopoGraph,
                        used: &[bool], shared_visited: &[AtomicBool],
                        target_magic_pos: (f32, f32))
                        -> Option<TreeGraph> {
    BACKTRACKED.with(|b| {
                   let mut bt = b.borrow_mut();
                   let num_nodes = shared_visited.len();
                   if bt.len() < num_nodes {
                       bt.resize(num_nodes, false);
                   } else {
                       bt.fill(false);
                   }
                   compute_parallel_impl(terminal_ids,
                                         root_ids,
                                         topo,
                                         used,
                                         shared_visited,
                                         &mut bt,
                                         target_magic_pos)
               })
}

fn compute_parallel_impl(terminal_ids: &[usize], root_ids: &[usize], topo: &TopoGraph,
                         used: &[bool], shared_visited: &[AtomicBool],
                         backtracked: &mut Vec<bool>, target_magic_pos: (f32, f32))
                         -> Option<TreeGraph> {
    let root_id = root_ids[0];
    debug_assert!(!used[root_id]);
    // Atomically claim the root. If another thread already owns it, give up immediately.
    if shared_visited[root_id].compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                              .is_err()
    {
        return None;
    }
    let mut path: Vec<usize> = vec![root_id];

    loop {
        let current = *path.last().unwrap();
        let current_node = topo.get_node(current);
        // Reached a ready magic node that we own (current is in path, claimed via CAS).
        if current_node.node_type == NodeType::Magic
           && current_node.cultivation_time == 0
           && !used[current]
        {
            return Some(build_tree(&path, terminal_ids, root_ids, topo));
        }

        // Find the best open neighbour: not globally used, not claimed by any thread,
        // not a data node, not backtracked by this thread; closest to target.
        let best_nb =
            current_node.nbors
                        .iter()
                        .copied()
                        .filter(|&nb_id| {
                            if used[nb_id] || shared_visited[nb_id].load(Ordering::Relaxed) {
                                return false;
                            }
                            if backtracked[nb_id] {
                                return false;
                            }
                            topo.get_node(nb_id).node_type != NodeType::Data
                        })
                        .min_by_key(|&nb_id| {
                            manhattan_dist(topo.get_node(nb_id).pos, target_magic_pos)
                        });

        if let Some(nb_id) = best_nb {
            // Attempt to claim the neighbour atomically. If another thread beat us to it,
            // re-enter the loop: the node now appears claimed and will be filtered out.
            if shared_visited[nb_id].compare_exchange(false,
                                                      true,
                                                      Ordering::AcqRel,
                                                      Ordering::Relaxed)
                                    .is_ok()
            {
                path.push(nb_id);
            }
        } else {
            // Dead-end: release current node back to the shared pool, mark as locally
            // backtracked so this thread won't revisit it.
            shared_visited[current].store(false, Ordering::Release);
            backtracked[current] = true;
            path.pop();
            if path.is_empty() {
                return None;
            }
            // Continue releasing upward until we find a node with open neighbours.
            loop {
                let bt_node = *path.last().unwrap();
                let has_open = topo.get_node(bt_node).nbors.iter().copied().any(|nb_id| {
                                                                               !used[nb_id]
                        && !shared_visited[nb_id].load(Ordering::Relaxed)
                        && !backtracked[nb_id]
                        && topo.get_node(nb_id).node_type != NodeType::Data
                        && !(topo.get_node(nb_id).node_type == NodeType::Magic
                             && topo.get_node(nb_id).cultivation_time > 0)
                                                                           });
                if has_open {
                    break;
                }
                shared_visited[bt_node].store(false, Ordering::Release);
                backtracked[bt_node] = true;
                path.pop();
                if path.is_empty() {
                    return None;
                }
            }
        }
    }
}

/// Builds a TreeGraph from the walked path and attaches terminal nodes.
fn build_tree(path: &[usize], terminal_ids: &[usize], root_ids: &[usize], topo: &TopoGraph)
              -> TreeGraph {
    let mut tree = TreeGraph::new(topo.num_nodes);
    tree.root_node_id = Some(*path.last().unwrap());
    // Add all path nodes and edges
    for &node_id in path {
        if !tree.contains_node(node_id) {
            tree.add_node(topo.get_node(node_id));
        }
    }
    for window in path.windows(2) {
        tree.add_edge(window[0], window[1]);
    }
    // Attach any additional root nodes not already on the path
    for (i, &root_id) in root_ids.iter().enumerate() {
        if !tree.contains_node(root_id) {
            let conn = topo.get_node(root_id)
                           .nbors
                           .iter()
                           .copied()
                           .find(|&nb_id| tree.contains_node(nb_id));
            if let Some(conn_id) = conn {
                tree.add_node(topo.get_node(root_id));
                tree.add_edge(conn_id, root_id);
            }
        }
        // Attach terminal to its root
        if i < terminal_ids.len() {
            let tid = terminal_ids[i];
            if !tree.contains_node(tid) {
                tree.add_node(topo.get_node(tid));
            }
            tree.add_edge(root_id, tid);
        }
    }
    tree
}

fn heuristic(pos: (f32, f32), ready_magic_positions: &[(f32, f32)]) -> (u32, usize) {
    ready_magic_positions.iter()
                         .enumerate()
                         .map(|(idx, &mp)| (manhattan_dist(mp, pos), idx))
                         .min_by(|(da, _), (db, _)| {
                             da.partial_cmp(db).unwrap_or(std::cmp::Ordering::Equal)
                         })
                         .unwrap()
}

fn manhattan_dist(p1: (f32, f32), p2: (f32, f32)) -> u32 {
    ((p1.0 - p2.0).abs() + (p1.1 - p2.1).abs()).floor() as u32
}
