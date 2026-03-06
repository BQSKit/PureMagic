use indexmap::IndexSet;
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

/// Represents a node in the topological graph.
/// Contains metadata about node type, position, magic state cultivation tracking, and connectivity.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub node_type: NodeType,
    pub id: usize,
    pub label: String,
    pub paired_data_id: Option<usize>,
    pub pos: (f32, f32),
    pub busy_count: i32,
    pub cultivation_time: i32,
    pub nbors: IndexSet<usize>,
}

static USE_MAGIC_ROUTING: AtomicBool = AtomicBool::new(true);

impl Node {
    /// Creates a new node with the given properties and empty neighbor set.
    pub fn new(id: usize, paired_data_id: Option<usize>, label: String, x: f32, y: f32,
               node_type: NodeType, busy_count: i32, cultivation_time: i32)
               -> Self {
        Node { node_type,
               id: id,
               label: label,
               paired_data_id: paired_data_id,
               pos: (x, y),
               busy_count,
               cultivation_time,
               nbors: IndexSet::new() }
    }

    /// Global switch to enable/disable magic routing (vs bus routing).
    pub fn set_magic_routing(enabled: bool) {
        USE_MAGIC_ROUTING.store(enabled, Ordering::Relaxed);
    }

    /// Adds a neighbor to this node's connectivity list.
    pub fn add_neighbor(&mut self, other: usize) {
        self.nbors.insert(other);
    }

    /// Returns true if this magic node is currently cultivating (in progress).
    pub fn is_cultivating(&self) -> bool {
        self.cultivation_time > 0 && self.busy_count < self.cultivation_time
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
