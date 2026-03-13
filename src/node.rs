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
    pub id: u16,
    pub paired_data_id: Option<u16>,
    pub pos: (f32, f32),
    pub nbors: Vec<u16>,
}

static USE_MAGIC_ROUTING: AtomicBool = AtomicBool::new(true);

impl Node {
    /// Creates a new node with the given properties and empty neighbor set.
    pub fn new(id: u16, paired_data_id: Option<u16>, x: f32, y: f32,
               node_type: NodeType)
               -> Self {
        Node { node_type,
               id: id,
               paired_data_id: paired_data_id,
               pos: (x, y),
               nbors: Vec::new() }
    }

    /// Global switch to enable/disable magic routing (vs bus routing).
    pub fn set_magic_routing(enabled: bool) {
        USE_MAGIC_ROUTING.store(enabled, Ordering::Relaxed);
    }

    /// Adds a neighbor to this node's connectivity list.
    pub fn add_neighbor(&mut self, other: u16) {
        if !self.nbors.contains(&other) {
            self.nbors.push(other);
        }
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
