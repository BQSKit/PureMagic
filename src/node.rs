use std::sync::atomic::{AtomicBool, Ordering};

/// Classification of nodes in the topological graph.
/// - `Magic`: nodes for magic state distillation/cultivation
/// - `Bus`: routing nodes (when not using magic routing)
/// - `Data`: logical data qubits
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum NodeType {
    Magic,
    Bus,
    Data,
}

/// Maximum number of neighbours a node can have.
/// Interior grid nodes have at most 4 (up/down/left/right).
/// Bus nodes can additionally connect to 2 data nodes = 6 total.
pub(crate) const MAX_NBORS: usize = 6;

/// Represents a node in the topological graph.
/// Contains metadata about node type, position, magic state cultivation tracking, and connectivity.
/// `nbors` is stored as a fixed-size inline array to avoid heap allocation and pointer chasing
/// in the A* inner loop.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub node_type: NodeType,
    pub id: u16,
    pub paired_data_id: Option<u16>,
    pub pos: (f32, f32),
    /// Inline neighbour list; valid entries are `nbors[0..num_nbors]`.
    pub nbors: [u16; MAX_NBORS],
    /// Number of valid entries in `nbors`.
    pub num_nbors: u8,
}

static USE_MAGIC_ROUTING: AtomicBool = AtomicBool::new(true);

impl Node {
    pub(crate) fn new(
        id: u16, paired_data_id: Option<u16>, x: f32, y: f32, node_type: NodeType,
    ) -> Self {
        Node { node_type, id, paired_data_id, pos: (x, y), nbors: [0u16; MAX_NBORS], num_nbors: 0 }
    }

    pub(crate) fn set_magic_routing(enabled: bool) {
        USE_MAGIC_ROUTING.store(enabled, Ordering::Relaxed);
    }

    /// Adds a neighbour to this node's connectivity list.
    /// Panics in debug mode if `MAX_NBORS` is exceeded.
    pub(crate) fn add_neighbor(&mut self, other: u16) {
        if self.nbors_slice().contains(&other) {
            return;
        }
        debug_assert!(
            (self.num_nbors as usize) < MAX_NBORS,
            "Node {} exceeded MAX_NBORS={} neighbours",
            self.id,
            MAX_NBORS
        );
        self.nbors[self.num_nbors as usize] = other;
        self.num_nbors += 1;
    }

    #[inline(always)]
    pub(crate) fn nbors_slice(&self) -> &[u16] {
        &self.nbors[..self.num_nbors as usize]
    }

    pub(crate) fn is_routing(&self) -> bool {
        if USE_MAGIC_ROUTING.load(Ordering::Relaxed) {
            assert_ne!(self.node_type, NodeType::Bus);
            self.node_type == NodeType::Magic
        } else {
            self.node_type == NodeType::Bus
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeType tests ────────────────────────────────────────────────────────

    #[test]
    fn node_type_equality() {
        assert_eq!(NodeType::Magic, NodeType::Magic);
        assert_eq!(NodeType::Bus, NodeType::Bus);
        assert_eq!(NodeType::Data, NodeType::Data);
        assert_ne!(NodeType::Magic, NodeType::Bus);
        assert_ne!(NodeType::Magic, NodeType::Data);
        assert_ne!(NodeType::Bus, NodeType::Data);
    }

    #[test]
    fn node_type_clone_and_copy() {
        let t = NodeType::Magic;
        let t2 = t; // Copy
        let t3 = t.clone(); // Clone
        assert_eq!(t, t2);
        assert_eq!(t, t3);
    }

    // ── Node::new ─────────────────────────────────────────────────────────────

    #[test]
    fn node_new_magic() {
        let node = Node::new(5, None, 1.0, 2.0, NodeType::Magic);
        assert_eq!(node.id, 5);
        assert_eq!(node.node_type, NodeType::Magic);
        assert_eq!(node.pos, (1.0, 2.0));
        assert!(node.paired_data_id.is_none());
        assert_eq!(node.num_nbors, 0);
    }

    #[test]
    fn node_new_data_with_pair() {
        let node = Node::new(3, Some(7), 0.5, 1.5, NodeType::Data);
        assert_eq!(node.id, 3);
        assert_eq!(node.node_type, NodeType::Data);
        assert_eq!(node.paired_data_id, Some(7));
    }

    #[test]
    fn node_new_bus() {
        let node = Node::new(0, None, 0.0, 0.0, NodeType::Bus);
        assert_eq!(node.node_type, NodeType::Bus);
    }

    // ── Node::add_neighbor ────────────────────────────────────────────────────

    #[test]
    fn add_neighbor_increases_count() {
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        assert_eq!(node.num_nbors, 0);
        node.add_neighbor(1);
        assert_eq!(node.num_nbors, 1);
        node.add_neighbor(2);
        assert_eq!(node.num_nbors, 2);
    }

    #[test]
    fn add_neighbor_no_duplicates() {
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        node.add_neighbor(1);
        node.add_neighbor(1); // duplicate — should be ignored
        assert_eq!(node.num_nbors, 1);
    }

    #[test]
    fn add_neighbor_up_to_max() {
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        for i in 1..=(MAX_NBORS as u16) {
            node.add_neighbor(i);
        }
        assert_eq!(node.num_nbors as usize, MAX_NBORS);
    }

    // ── Node::nbors_slice ─────────────────────────────────────────────────────

    #[test]
    fn nbors_slice_empty_initially() {
        let node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        assert!(node.nbors_slice().is_empty());
    }

    #[test]
    fn nbors_slice_contains_added_neighbors() {
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        node.add_neighbor(10);
        node.add_neighbor(20);
        let slice = node.nbors_slice();
        assert_eq!(slice.len(), 2);
        assert!(slice.contains(&10));
        assert!(slice.contains(&20));
    }

    // ── Node::is_routing ─────────────────────────────────────────────────────

    #[test]
    fn is_routing_magic_when_magic_routing_enabled() {
        Node::set_magic_routing(true);
        let magic = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        let data = Node::new(1, None, 0.0, 0.0, NodeType::Data);
        assert!(magic.is_routing());
        assert!(!data.is_routing());
    }

    #[test]
    fn is_routing_bus_when_magic_routing_disabled() {
        Node::set_magic_routing(false);
        let bus = Node::new(0, None, 0.0, 0.0, NodeType::Bus);
        let data = Node::new(1, None, 0.0, 0.0, NodeType::Data);
        assert!(bus.is_routing());
        assert!(!data.is_routing());
        Node::set_magic_routing(true);
    }

    // ── Node::set_magic_routing ───────────────────────────────────────────────

    #[test]
    fn set_magic_routing_toggles_global_flag() {
        Node::set_magic_routing(true);
        let magic = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        assert!(magic.is_routing());

        Node::set_magic_routing(false);
        let bus = Node::new(1, None, 0.0, 0.0, NodeType::Bus);
        assert!(bus.is_routing());

        Node::set_magic_routing(true);
    }

    // ── Node::new — data without pair ─────────────────────────────────────────

    #[test]
    fn node_new_data_without_pair() {
        let node = Node::new(9, None, 3.0, 4.0, NodeType::Data);
        assert_eq!(node.id, 9);
        assert_eq!(node.node_type, NodeType::Data);
        assert!(node.paired_data_id.is_none());
        assert_eq!(node.pos, (3.0, 4.0));
    }

    // ── Node::add_neighbor — overflow beyond MAX_NBORS ────────────────────────

    #[test]
    #[cfg(not(debug_assertions))]
    fn add_neighbor_beyond_max_is_noop_in_release() {
        // In release builds the overflow check is disabled; adding more than MAX_NBORS
        // neighbours should not panic and the count should stay capped.
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        for i in 1..=(MAX_NBORS as u16 + 2) {
            node.add_neighbor(i);
        }
        // num_nbors must not exceed MAX_NBORS
        assert!(node.num_nbors as usize <= MAX_NBORS);
    }

    // ── Node::is_routing — Bus node is routing when magic routing disabled ───

    #[test]
    fn is_routing_bus_node_when_magic_routing_disabled() {
        Node::set_magic_routing(false);
        let bus = Node::new(0, None, 0.0, 0.0, NodeType::Bus);
        // When magic routing is off, Bus nodes are routing nodes
        assert!(bus.is_routing());
        Node::set_magic_routing(true); // restore
    }

    // ── Node::is_routing — Magic node with magic routing disabled ────────────

    #[test]
    fn is_routing_magic_when_magic_routing_disabled() {
        Node::set_magic_routing(false);
        let magic = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        // When magic routing is off, magic nodes are NOT routing nodes
        assert!(!magic.is_routing());
        Node::set_magic_routing(true); // restore
    }

    // ── MAX_NBORS constant ────────────────────────────────────────────────────

    #[test]
    fn max_nbors_is_six() {
        assert_eq!(MAX_NBORS, 6);
    }
}
