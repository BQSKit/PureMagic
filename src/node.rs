use std::sync::atomic::{AtomicBool, Ordering};

/// Classification of nodes in the topological graph.
/// - `Magic`: nodes for magic state distillation/cultivation
/// - `Bus`: routing nodes (when not using magic routing)
/// - `Data`: logical data qubits
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Magic,
    Bus,
    Data,
}

/// Maximum number of neighbours a node can have.
/// Interior grid nodes have at most 4 (up/down/left/right).
/// Bus nodes can additionally connect to 2 data nodes = 6 total.
pub const MAX_NBORS: usize = 6;

/// Represents a node in the topological graph.
/// Contains metadata about node type, position, magic state cultivation tracking, and connectivity.
/// `nbors` is stored as a fixed-size inline array to avoid heap allocation and pointer chasing
/// in the A* inner loop.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
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
    /// Creates a new node with the given properties and empty neighbour set.
    pub fn new(id: u16, paired_data_id: Option<u16>, x: f32, y: f32, node_type: NodeType) -> Self {
        Node { node_type, id, paired_data_id, pos: (x, y), nbors: [0u16; MAX_NBORS], num_nbors: 0 }
    }

    /// Global switch to enable/disable magic routing (vs bus routing).
    pub fn set_magic_routing(enabled: bool) {
        USE_MAGIC_ROUTING.store(enabled, Ordering::Relaxed);
    }

    /// Adds a neighbour to this node's connectivity list.
    /// Panics in debug mode if `MAX_NBORS` is exceeded.
    pub fn add_neighbor(&mut self, other: u16) {
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

    /// Returns the slice of valid neighbour IDs.
    #[inline(always)]
    pub fn nbors_slice(&self) -> &[u16] {
        &self.nbors[..self.num_nbors as usize]
    }

    /// Returns true if this node is a routing node (magic or bus depending on config).
    pub fn is_routing(&self) -> bool {
        if USE_MAGIC_ROUTING.load(Ordering::Relaxed) {
            assert_ne!(self.node_type, NodeType::Bus);
            self.node_type == NodeType::Magic
        } else {
            self.node_type == NodeType::Bus
        }
    }
}
