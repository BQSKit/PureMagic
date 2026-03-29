use std::sync::atomic::{AtomicBool, Ordering};

/// Classification of nodes in the topological graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum NodeType {
    Magic,
    Bus,
    Data,
}

/// Maximum number of neighbours a node can have (4 grid + 2 data = 6).
pub(crate) const MAX_NBORS: usize = 6;

/// A node in the topological graph.
/// `nbors` is a fixed-size inline array to avoid heap allocation in the A* inner loop.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub node_type: NodeType,
    pub id: u16,
    pub paired_data_id: Option<u16>,
    pub pos: (f32, f32),
    pub nbors: [u16; MAX_NBORS],
    pub num_nbors: u8,
}

/// Global flag controlling whether magic nodes act as routing intermediaries.
/// Set once at startup via `Node::set_magic_routing`; read on every A* expansion.
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

    /// Returns true if this node is a valid routing intermediary in the current mode.
    /// In magic-routing mode only Magic nodes route; Bus nodes must not exist.
    /// In bus-routing mode only Bus nodes route.
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
        let t2 = t;
        let t3 = t.clone();
        assert_eq!(t, t2);
        assert_eq!(t, t3);
    }

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
        node.add_neighbor(1);
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

    #[test]
    fn node_new_data_without_pair() {
        let node = Node::new(9, None, 3.0, 4.0, NodeType::Data);
        assert_eq!(node.id, 9);
        assert_eq!(node.node_type, NodeType::Data);
        assert!(node.paired_data_id.is_none());
        assert_eq!(node.pos, (3.0, 4.0));
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn add_neighbor_beyond_max_is_noop_in_release() {
        let mut node = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        for i in 1..=(MAX_NBORS as u16 + 2) {
            node.add_neighbor(i);
        }
        assert!(node.num_nbors as usize <= MAX_NBORS);
    }

    #[test]
    fn is_routing_bus_node_when_magic_routing_disabled() {
        Node::set_magic_routing(false);
        let bus = Node::new(0, None, 0.0, 0.0, NodeType::Bus);
        assert!(bus.is_routing());
        Node::set_magic_routing(true);
    }

    #[test]
    fn is_routing_magic_when_magic_routing_disabled() {
        Node::set_magic_routing(false);
        let magic = Node::new(0, None, 0.0, 0.0, NodeType::Magic);
        assert!(!magic.is_routing());
        Node::set_magic_routing(true);
    }

    #[test]
    fn max_nbors_is_six() {
        assert_eq!(MAX_NBORS, 6);
    }
}
