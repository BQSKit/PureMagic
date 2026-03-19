use crate::fn_timer;
use crate::node::{Node, NodeType};
use crate::pauliproduct::PauliProduct;
use crate::treegraph::TreeGraph;
use indexmap::IndexMap;
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::Write;
use std::io::{self, BufRead};
use std::path::Path;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Plot helper types ────────────────────────────────────────────────────────

/// Geometry for one double-data-qubit group (X row + Z row sharing a column).
struct DataGroup {
    col: f32, // integer column centre
    y_x: f32, // y of X row (dashed border)
    y_z: f32, // y of Z row (solid border)
    x_left_id: u16,
    x_right_id: u16,
    z_left_id: u16,
    z_right_id: u16,
}

/// Per-node metadata computed once and reused across draw passes.
struct NodeDrawInfo {
    x: f32,
    y: f32,
    half_x: f32,
    half_y: f32,
    #[allow(dead_code)]
    is_data_x: bool,
    border_color_idx: Option<usize>,
    #[allow(dead_code)]
    is_root: bool,
    label_text: String,
}

// ── TopoGraph ─────────────────────────────────────────────────────────────────

/// Represents the topological layout of a surface code quantum processor.
/// Contains data, magic, and routing qubits arranged in a 2D grid.
/// Supports both magic routing and bus routing architectures.
pub struct TopoGraph {
    nodes: Vec<Node>,
    pub labels: Vec<String>,
    node_ids_from_labels: IndexMap<String, u16>,
    // Fast lookup for data nodes: indexed by qubit number, [0] = X node id, [1] = Z node id
    data_node_ids: Vec<[u16; 2]>,
    node_grid: Vec<Vec<Option<String>>>,
    num_cols: usize,
    num_rows: usize,
    topo_fname: String,
    circuit_fname: String,
    pub use_magic_routing: bool,
    pub num_data_qubits: usize,
    pub num_bus_qubits: usize,
    pub num_magic_qubits: usize,
    pub num_qubits: usize,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub busy_counts: Vec<i32>,
    pub cultivation_times: Vec<i32>,
    pub sides_only: bool,
}

impl TopoGraph {
    /// Creates an empty topology graph.
    pub fn new() -> Self {
        TopoGraph {
            nodes: Vec::new(),
            labels: Vec::new(),
            node_ids_from_labels: IndexMap::new(),
            data_node_ids: Vec::new(),
            node_grid: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            num_data_qubits: 0,
            num_bus_qubits: 0,
            num_magic_qubits: 0,
            num_qubits: 0,
            num_edges: 0,
            num_nodes: 0,
            circuit_fname: String::new(),
            topo_fname: String::new(),
            use_magic_routing: true,
            busy_counts: Vec::new(),
            cultivation_times: Vec::new(),
            sides_only: false,
        }
    }

    pub fn get_label(&self, id: u16) -> &str {
        &self.labels[id as usize]
    }

    /// Initializes topology from file or generates a synthetic layout.
    /// Sets up node metadata, qubit pairings, and edge connectivity.
    pub fn set_topo(
        &mut self, min_num_qubits: usize, circuit_fname: &String, topo_fname: &String, rseed: &u32,
        use_magic_routing: bool, ancilla_rows: usize, sides_only: bool,
    ) {
        self.circuit_fname = circuit_fname.to_string();
        self.topo_fname = topo_fname.to_string();
        self.use_magic_routing = use_magic_routing;
        self.sides_only = sides_only;
        Node::set_magic_routing(use_magic_routing);

        if !self.topo_fname.is_empty() {
            if let Err(e) = self.read_topo_from_file(rseed, sides_only) {
                eprintln!("Error reading topology file: {}", e);
            }
        } else if !use_magic_routing {
            if ancilla_rows == 0 {
                self.gen_compact_bus_routing_topo(min_num_qubits, sides_only);
            } else {
                self.gen_bus_routing_topo(min_num_qubits, sides_only);
            }
        } else {
            self.gen_pure_magic_topo(min_num_qubits, ancilla_rows, sides_only);
        }
        let node_ids: Vec<u16> = self.nodes.iter().map(|node| node.id).collect();
        for node_id in node_ids {
            let node = self.get_node(node_id);
            if node.node_type == NodeType::Data {
                let label = self.get_label(node_id);
                let qubit = label
                    .chars()
                    .skip(1)
                    .take_while(|c| c.is_numeric())
                    .collect::<String>()
                    .parse::<usize>()
                    .ok()
                    .unwrap();
                let term = label.chars().last().map(|c| c.to_string()).unwrap();
                let pair_qubit = if qubit % 2 == 0 { qubit + 1 } else { qubit - 1 };
                let paired_node_label = format!("d{}{}", pair_qubit, term);
                self.get_node_mut(node_id).paired_data_id =
                    self.node_ids_from_labels.get(&paired_node_label).copied();
            }
        }
        // Build fast data-node lookup: label format is "d{qubit}{basis}" where basis is 'X' or 'Z'
        self.data_node_ids.clear();
        for node in &self.nodes {
            if node.node_type == NodeType::Data {
                let label = &self.labels[node.id as usize];
                let basis: char = label.chars().last().unwrap();
                let qubit: usize = label[1..label.len() - 1].parse().unwrap();
                let basis_idx: usize = if basis == 'X' { 0 } else { 1 };
                if qubit >= self.data_node_ids.len() {
                    self.data_node_ids.resize(qubit + 1, [u16::MAX; 2]);
                }
                self.data_node_ids[qubit][basis_idx] = node.id;
            }
        }
        self.update_statistics();
        self.print_statistics();
    }

    /// Loads topology from a file describing node labels and grid positions.
    /// Supports randomized data node pairing for scheduling variation.
    pub fn read_topo_from_file(&mut self, rseed: &u32, sides_only: bool) -> io::Result<()> {
        let _timer = fn_timer!();
        let mut rows = Vec::new();
        let file = File::open(&self.topo_fname)?;
        for line in io::BufReader::new(file).lines() {
            let line = line?;
            let row: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if !row.is_empty() {
                rows.push(row);
            }
        }
        self.num_rows = rows.len();
        self.num_cols = rows[0].len();
        self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

        for (row_i, row) in rows.iter().enumerate() {
            for (col_i, col) in row.iter().enumerate() {
                self.node_grid[col_i][row_i] = Some(col.clone());
                if self.use_magic_routing && col == "b" {
                    self.node_grid[col_i][row_i] = Some("m".to_string());
                }
            }
        }
        let mut pair_indices = Vec::new();
        let mut num_data_nodes = 0;
        for col in 0..self.num_cols {
            for row in 0..self.num_rows {
                if let Some(ref node) = self.node_grid[col][row] {
                    if node.starts_with('d') && node.ends_with('X') {
                        pair_indices.push(num_data_nodes);
                        num_data_nodes += 4;
                    }
                }
            }
        }
        if *rseed != 0 {
            let timer_seed =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
            let mut rng = StdRng::seed_from_u64(timer_seed);
            pair_indices.shuffle(&mut rng);
        }
        println!("Data node order {:?}", pair_indices);
        let mut di = 0;
        for col in 0..self.num_cols {
            for row in 0..self.num_rows {
                if let Some(ref node) = self.node_grid[col][row] {
                    if node.starts_with('d') {
                        let op = node.chars().nth(1).unwrap_or('X');
                        let pair_di =
                            if op == 'X' { pair_indices[di] } else { pair_indices[di] + 2 };
                        self.add_double_data_qubit(pair_di, col, row, op == 'X');
                        if op == 'Z' {
                            di += 1;
                        }
                    } else {
                        let node_type = match node.chars().next() {
                            Some('m') => NodeType::Magic,
                            Some('b') => NodeType::Bus,
                            _ => continue,
                        };
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                    }
                }
            }
        }
        self.set_edges(sides_only);
        println!("Read topology with dimensions: {} {}", self.num_cols, self.num_rows);

        Ok(())
    }

    /// Generates a bus routing topology: data qubits with dedicated bus columns for routing.
    fn gen_bus_routing_topo(&mut self, min_num_qubits: usize, sides_only: bool) {
        let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let bus_rows = patch_rows + 1;
        let qubits_per_col = 2 * patch_rows;
        let num_data_cols = ((min_num_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;
        self.num_cols = 2 * num_data_cols + 3;
        self.num_rows = 2 + 2 * patch_rows + bus_rows;
        self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

        self.add_border_row(0);
        self.add_border_column(0);

        let max_qi =
            if min_num_qubits % 2 == 0 { 2 * min_num_qubits } else { 2 * min_num_qubits + 1 };
        let mut qi = 0;
        for col in 1..self.num_cols - 1 {
            if col % 2 == 0 {
                for row in 1..self.num_rows - 1 {
                    if row % 3 + 1 == 2 {
                        let node_type =
                            if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus };
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                    } else {
                        if qi < max_qi {
                            self.add_double_data_qubit(qi, col, row, row % 3 + 1 == 3);
                            qi += 2;
                        } else {
                            self.node_grid[col][row] =
                                Some(self.add_qubit(col, row, NodeType::Magic));
                        }
                    }
                }
            } else {
                let node_type =
                    if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus };
                for row in 1..self.num_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                }
            }
        }
        self.add_border_column(self.num_cols - 1);
        self.add_border_row(self.num_rows - 1);
        self.set_edges(sides_only);
        println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
    }

    /// Generates a compact bus routing topology without separate bus columns.
    fn gen_compact_bus_routing_topo(&mut self, min_num_qubits: usize, sides_only: bool) {
        let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let qubits_per_col = 2 * patch_rows;
        let num_data_cols = ((min_num_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;
        self.num_cols = 2 * num_data_cols + 1;
        self.num_rows = 3 + 2 * patch_rows;
        self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

        self.add_border_row_compact(0);
        let max_qi =
            if min_num_qubits % 2 == 0 { 2 * min_num_qubits } else { 2 * min_num_qubits + 1 };
        let mut qi = 0;
        for col in 0..self.num_cols {
            if col % 2 == 1 {
                for row in 1..self.num_rows - 1 {
                    if qi < max_qi && row < self.num_rows - 2 {
                        self.add_double_data_qubit(qi, col, row, row % 2 == 1);
                        qi += 2;
                    } else {
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
                    }
                }
                let row = self.num_rows - 1;
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
            } else {
                let node_type =
                    if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus };
                for row in 1..self.num_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                }
            }
        }
        self.add_border_row_compact(self.num_rows - 1);
        self.set_edges(sides_only);
        println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
    }

    /// Adds magic/bus nodes along a border row connecting to adjacent nodes.
    fn add_border_row(&mut self, row: usize) {
        let node_type = if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus };
        self.node_grid[0][row] = Some(self.add_qubit(0, row, node_type));
        self.node_grid[self.num_cols - 1][row] =
            Some(self.add_qubit(self.num_cols - 1, row, node_type));
        for col in 1..self.num_cols - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    /// Adds border nodes for compact bus topology (alternating magic/bus columns).
    fn add_border_row_compact(&mut self, row: usize) {
        for col in 0..self.num_cols {
            if col % 2 == 0 {
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
            } else {
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
            }
        }
    }

    /// Adds magic nodes down a border column.
    fn add_border_column(&mut self, col: usize) {
        for row in 1..self.num_rows - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    /// Generates a pure magic topology: all non-data qubits are magic nodes.
    /// `ancilla_rows` controls spacing between data qubit rows.
    pub fn gen_pure_magic_topo(
        &mut self, min_num_qubits: usize, ancilla_rows: usize, sides_only: bool,
    ) {
        let row_spacing = ancilla_rows + 1;
        let col_spacing = if ancilla_rows == 0 { 2 } else { ancilla_rows + 1 };
        let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let patch_cols = ((min_num_qubits as f64) / ((2 * patch_rows) as f64)).ceil() as usize;
        self.num_rows = patch_rows * (1 + row_spacing) + row_spacing - 1;
        if ancilla_rows == 0 {
            self.num_rows += 1;
        }
        self.num_cols = patch_cols * col_spacing + col_spacing - 1;
        self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];
        let mut qi = 0;
        let max_qi =
            if min_num_qubits % 2 == 0 { 2 * min_num_qubits } else { 2 * min_num_qubits + 1 };
        let row_gap = 1 + row_spacing;
        for col in 0..self.num_cols {
            for row in 0..self.num_rows {
                if col % col_spacing == col_spacing - 1 {
                    if (row % row_gap == row_spacing || row % row_gap == row_spacing - 1)
                        && !(ancilla_rows == 0 && row == self.num_rows - 1)
                    {
                        if qi < max_qi {
                            let is_x = row % row_gap == row_spacing - 1;
                            self.add_double_data_qubit(qi, col, row, is_x);
                            qi += 2;
                        } else {
                            self.node_grid[col][row] =
                                Some(self.add_qubit(col, row, NodeType::Magic));
                        }
                    } else {
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
                    }
                } else {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
                }
            }
        }
        self.set_edges(sides_only);
        println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
    }

    /// Adds a pair of data qubits (X and Z basis) at the given position.
    /// Both qubits share a combined label in the grid but have separate nodes.
    fn add_double_data_qubit(&mut self, qi: usize, col: usize, row: usize, is_x: bool) {
        let q = if is_x { qi / 2 } else { qi / 2 - 1 };
        let op = if is_x { 'X' } else { 'Z' };
        let label1 = format!("d{}{}", q, op);
        let id1 = self.num_nodes as u16;
        let node1 = Node::new(
            id1,
            None,
            col as f32 - 0.25,
            (self.num_rows - 1 - row) as f32,
            NodeType::Data,
        );
        self.nodes.push(node1);
        self.labels.push(label1.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.node_ids_from_labels.insert(label1, id1);
        self.num_nodes += 1;
        let id2 = self.num_nodes as u16;
        let label2 = format!("d{}{}", q + 1, op);
        let node2 = Node::new(
            id2,
            None,
            col as f32 + 0.25,
            (self.num_rows - 1 - row) as f32,
            NodeType::Data,
        );
        self.nodes.push(node2);
        self.labels.push(label2.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.node_ids_from_labels.insert(label2, id2);
        let combined_label = format!("d{}/{}{}", q, q + 1, op);
        self.node_grid[col][row] = Some(combined_label.clone());
        self.num_nodes += 1;
    }

    /// Creates and adds a single node (magic, bus, or data) at grid position (col, row).
    fn add_qubit(&mut self, col: usize, row: usize, node_type: NodeType) -> String {
        let ch = match node_type {
            NodeType::Magic => "m",
            NodeType::Bus => "b",
            NodeType::Data => "d",
        };

        let label = format!("{}{}-{}", ch, col, row);
        let node = Node::new(
            self.num_nodes as u16,
            None,
            col as f32,
            (self.num_rows - 1 - row) as f32,
            node_type,
        );
        self.nodes.push(node);
        self.labels.push(label.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.node_ids_from_labels.insert(label.clone(), self.num_nodes as u16);
        self.num_nodes += 1;
        label
    }

    /// Establishes edges between adjacent nodes (4-connectivity with optional vertical data edges).
    fn set_edges(&mut self, sides_only: bool) {
        let mut edges_to_add = Vec::new();
        let mut vert_data_edges_to_add = Vec::new();

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    if col > 0 {
                        if let Some(ref left_label) = self.node_grid[col - 1][row] {
                            edges_to_add.push((label.clone(), left_label.clone()));
                        }
                    }
                    if !sides_only {
                        if row > 1 {
                            if label.starts_with('d') && label.ends_with('Z') {
                                if let Some(ref up_label) = self.node_grid[col][row - 2] {
                                    if up_label.starts_with('b') || up_label.starts_with('m') {
                                        vert_data_edges_to_add
                                            .push((label.clone(), up_label.clone()));
                                    }
                                }
                            }
                        }
                        if row < self.num_rows - 2 {
                            if label.starts_with('d') && label.ends_with('X') {
                                if let Some(ref up_label) = self.node_grid[col][row + 2] {
                                    if up_label.starts_with('b') || up_label.starts_with('m') {
                                        vert_data_edges_to_add
                                            .push((label.clone(), up_label.clone()));
                                    }
                                }
                            }
                        }
                    }
                    if row > 0 {
                        if let Some(ref up_label) = self.node_grid[col][row - 1] {
                            if !label.starts_with('d') && !up_label.starts_with('d') {
                                edges_to_add.push((label.clone(), up_label.clone()));
                            }
                        }
                    }
                }
            }
        }
        // Add all edges
        for (label1, label2) in edges_to_add {
            if label1.starts_with('d') {
                if let Some(ref d) = self.get_data_label_side(&label1, true) {
                    let n1 = self.node_ids_from_labels.get(d).unwrap();
                    let n2 = self.node_ids_from_labels.get(&label2).unwrap();
                    self.add_edge(*n1, *n2);
                }
            } else if label2.starts_with('d') {
                if let Some(ref d) = self.get_data_label_side(&label2, false) {
                    let n1 = self.node_ids_from_labels.get(d).unwrap();
                    let n2 = self.node_ids_from_labels.get(&label1).unwrap();
                    self.add_edge(*n2, *n1);
                }
            } else {
                let n1 = self.node_ids_from_labels.get(&label1).unwrap();
                let n2 = self.node_ids_from_labels.get(&label2).unwrap();
                self.add_edge(*n1, *n2);
            }
        }
        for (label1, label2) in vert_data_edges_to_add {
            let (data_label, bus_label) =
                if label1.starts_with('d') { (label1, label2) } else { (label2, label1) };
            let (data_label1, data_label2) = self.get_data_labels(&data_label).unwrap();
            let data_node_id1 = self.node_ids_from_labels.get(&data_label1).unwrap().clone();
            let data_node_id2 = self.node_ids_from_labels.get(&data_label2).unwrap().clone();
            let bus_node_id = self.node_ids_from_labels.get(&bus_label).unwrap().clone();
            self.get_node_mut(bus_node_id).add_neighbor(data_node_id1);
            self.get_node_mut(bus_node_id).add_neighbor(data_node_id2);
            self.get_node_mut(data_node_id1).add_neighbor(bus_node_id);
            self.get_node_mut(data_node_id2).add_neighbor(bus_node_id);
        }
    }

    /// Extracts the left or right side data node label from a double data qubit label.
    fn get_data_label_side(&self, label: &str, left: bool) -> Option<String> {
        // Find indices of numbers and operator
        let d_pos = label.find('d')?;
        let slash_pos = label.find('/')?;
        let op_pos = label.find(|c: char| c == 'X' || c == 'Z')?;
        // Extract the numbers and operator
        let first_num = &label[d_pos + 1..slash_pos];
        let second_num = &label[slash_pos + 1..op_pos];
        let operator = &label[op_pos..=op_pos];
        if left {
            return Some(format!("d{}{}", first_num, operator));
        } else {
            return Some(format!("d{}{}", second_num, operator));
        }
    }

    /// Extracts both left and right data node labels from a double data qubit label.
    fn get_data_labels(&self, label: &str) -> Option<(String, String)> {
        // Find indices of numbers and operator
        let d_pos = label.find('d')?;
        let slash_pos = label.find('/')?;
        let op_pos = label.find(|c: char| c == 'X' || c == 'Z')?;
        // Extract the numbers and operator
        let first_num = &label[d_pos + 1..slash_pos];
        let second_num = &label[slash_pos + 1..op_pos];
        let operator = &label[op_pos..=op_pos];
        Some((format!("d{}{}", first_num, operator), format!("d{}{}", second_num, operator)))
    }

    /// Recomputes qubit counts and builds fast data node lookup by qubit and basis.
    pub fn update_statistics(&mut self) {
        let mut data_count = 0;
        let mut magic_count = 0;
        let mut bus_count = 0;

        for node in &self.nodes {
            match node.node_type {
                NodeType::Data => data_count += 1,
                NodeType::Magic => magic_count += 1,
                NodeType::Bus => bus_count += 1,
            }
        }
        self.num_data_qubits = data_count / 2;
        self.num_magic_qubits = magic_count;
        self.num_bus_qubits = bus_count;
        self.num_qubits = self.num_data_qubits + self.num_bus_qubits + self.num_magic_qubits;
    }

    /// Prints qubit type distribution to stdout.
    fn print_statistics(&mut self) {
        let total = self.num_qubits as f64;
        println!("Number of qubits:");
        println!(
            "  data:         {} ({:.3})",
            self.num_data_qubits,
            self.num_data_qubits as f64 / total
        );
        println!(
            "  bus:          {} ({:.3})",
            self.num_bus_qubits,
            self.num_bus_qubits as f64 / total
        );
        println!(
            "  magic:        {} ({:.3})",
            self.num_magic_qubits,
            self.num_magic_qubits as f64 / total
        );
        println!("  total:        {}", self.num_qubits);
    }

    /// Retrieves a node by its ID.
    pub fn get_node(&self, id: u16) -> &Node {
        &self.nodes[id as usize]
    }

    /// Retrieves a mutable reference to a node by its ID.
    pub fn get_node_mut(&mut self, id: u16) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// Returns an iterator over all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    /// Creates a bidirectional edge between two nodes.
    pub fn add_edge(&mut self, node_id1: u16, node_id2: u16) {
        self.get_node_mut(node_id1).add_neighbor(node_id2);
        self.get_node_mut(node_id2).add_neighbor(node_id1);
        self.num_edges += 1;
    }

    /// Fast lookup of a data node by qubit number and basis (X or Z).
    pub fn get_data_node_id(&self, qubit: u16, basis: char) -> u16 {
        let basis_idx: usize = if basis == 'X' { 0 } else { 1 };
        self.data_node_ids[qubit as usize][basis_idx]
    }

    /// Returns true if this magic node is currently cultivating (in progress).
    pub fn is_cultivating(&self, node_id: u16) -> bool {
        self.cultivation_times[node_id as usize] > 0
            && self.busy_counts[node_id as usize] < self.cultivation_times[node_id as usize]
    }

    /// Writes topology grid to a text file
    pub fn print(&self) -> io::Result<()> {
        let topo_path = Path::new(&self.circuit_fname);
        let topo_stem = topo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("topo");
        let output_fname = format!("{}.topo.txt", topo_stem);
        let mut file = File::create(&output_fname)?;

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    //write!(file, "{:8}  ", label)?;
                    if label.starts_with('d') {
                        write!(
                            file,
                            "{}{} ",
                            label.chars().nth(0).unwrap_or(' '),
                            label.chars().last().unwrap_or(' ')
                        )?;
                    } else {
                        write!(file, "{}  ", label.chars().nth(0).unwrap_or(' '))?;
                    }
                }
            }
            writeln!(file)?;
        }

        println!("Wrote topology to {}", output_fname);
        Ok(())
    }

    /// Plots the topology with scheduled Pauli product paths highlighted.
    /// Generates PNG with nodes colored by type and edges colored by path.
    pub fn plot(
        &self, fname_added: &str, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>)],
        title_str: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = fn_timer!();
        let topo_path = Path::new(&self.circuit_fname);
        let topo_stem = topo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("topo");
        let plot_fname = format!("{}{}.png", topo_stem, fname_added);

        let root = BitMapBackend::new(
            &plot_fname,
            (self.num_cols as u32 * 100, self.num_rows as u32 * 100),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(-1f32..self.num_cols as f32, -1f32..self.num_rows as f32)?;

        let path_colors = self.make_path_colors(pauli_product_paths.len());
        let data_groups = self.build_data_groups();
        let node_infos = self.build_node_infos(pauli_product_paths);
        let routing_pos_color = self.build_routing_pos_color(&node_infos);

        self.draw_node_fills(&mut chart, &node_infos, &path_colors)?;

        self.draw_node_borders(&mut chart, &node_infos, pauli_product_paths)?;

        self.draw_data_borders(
            &mut chart,
            &data_groups,
            &node_infos,
            &path_colors,
            &routing_pos_color,
        )?;
        self.draw_labels(
            &mut chart,
            &data_groups,
            &node_infos,
            &path_colors,
            &routing_pos_color,
            pauli_product_paths,
        )?;
        if !title_str.is_empty() {
            let font_size = (6.0 * (self.num_rows as f64).sqrt()) as u32;
            for (i, line) in title_str.split('\n').enumerate() {
                draw_text(
                    &mut chart,
                    line,
                    -0.5,
                    -0.8 - (i as f32 * 0.33),
                    ("sans-serif", font_size).into_font(),
                )?;
            }
        }
        root.present()?;
        println!("Plotted topology to {}", plot_fname);
        Ok(())
    }

    // ── Private plot helpers ──────────────────────────────────────────────────

    /// Builds the color palette for product paths.
    fn make_path_colors(&self, num_paths: usize) -> Vec<RGBAColor> {
        let n = num_paths.max(1);
        (0..n)
            .map(|i| {
                let hue = (i as f64) / (n as f64);
                let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
                RGBColor(r, g, b).to_rgba()
            })
            .collect()
    }

    /// Builds the list of data-qubit groups (X row + Z row pairs).
    fn build_data_groups(&self) -> Vec<DataGroup> {
        // Map from rounded (x*10, y*10) → node id for data nodes.
        let data_pos_map: std::collections::HashMap<(i32, i32), u16> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Data)
            .map(|n| {
                let (px, py) = n.pos;
                (((px * 10.0).round() as i32, (py * 10.0).round() as i32), n.id)
            })
            .collect();

        let mut groups = Vec::new();
        let mut seen: std::collections::HashSet<u16> = std::collections::HashSet::new();

        for node in &self.nodes {
            if node.node_type != NodeType::Data {
                continue;
            }
            if seen.contains(&node.id) {
                continue;
            }
            let label = &self.labels[node.id as usize];
            if !label.ends_with('X') {
                continue;
            } // anchor on X nodes

            let (x, y) = node.pos;
            let xi = (x * 10.0).round() as i32;
            let yi = (y * 10.0).round() as i32;

            // Find the X-row partner (same y, x differs by 0.5 = 5 in *10 units).
            let partner_xi = if data_pos_map.contains_key(&(xi + 5, yi)) {
                xi + 5
            } else if data_pos_map.contains_key(&(xi - 5, yi)) {
                xi - 5
            } else {
                continue;
            };
            let (x_left_id, x_right_id) = if xi < partner_xi {
                (
                    node.id,
                    match data_pos_map.get(&(partner_xi, yi)) {
                        Some(&id) => id,
                        None => continue,
                    },
                )
            } else {
                (
                    match data_pos_map.get(&(partner_xi, yi)) {
                        Some(&id) => id,
                        None => continue,
                    },
                    node.id,
                )
            };

            // Find the Z-row partners (adjacent y, same x positions).
            let left_xi = xi.min(partner_xi);
            let right_xi = xi.max(partner_xi);
            let mut z_found = None;
            for &zy in &[yi - 10, yi + 10] {
                if let (Some(&zl), Some(&zr)) =
                    (data_pos_map.get(&(left_xi, zy)), data_pos_map.get(&(right_xi, zy)))
                {
                    if self.labels[zl as usize].ends_with('Z')
                        && self.labels[zr as usize].ends_with('Z')
                    {
                        z_found = Some((zy, zl, zr));
                        break;
                    }
                }
            }
            let (zy, z_left_id, z_right_id) = match z_found {
                Some(v) => v,
                None => continue,
            };

            let partner_x = if xi < partner_xi { x + 0.5 } else { x - 0.5 };
            let col_center = if x < partner_x { x + 0.25 } else { x - 0.25 };

            seen.insert(x_left_id);
            seen.insert(x_right_id);
            seen.insert(z_left_id);
            seen.insert(z_right_id);
            groups.push(DataGroup {
                col: col_center,
                y_x: yi as f32 / 10.0,
                y_z: zy as f32 / 10.0,
                x_left_id,
                x_right_id,
                z_left_id,
                z_right_id,
            });
        }
        groups
    }

    /// Builds per-node draw metadata (position, color index, label text).
    fn build_node_infos(
        &self, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>)],
    ) -> Vec<NodeDrawInfo> {
        self.nodes
            .iter()
            .map(|node| {
                let (x, y) = node.pos;
                let (half_x, half_y) = if node.node_type == NodeType::Data {
                    (0.25f32, 0.5f32)
                } else {
                    (0.5f32, 0.5f32)
                };
                let label = &self.labels[node.id as usize];
                let is_data_x = node.node_type == NodeType::Data && label.ends_with('X');
                let mut border_color_idx = None;
                let mut is_root = false;
                let mut path_is_t = false;
                for (i, (pp, path_graph)) in pauli_product_paths.iter().enumerate() {
                    if path_graph.contains_node(node.id) {
                        border_color_idx = Some(i);
                        is_root = path_graph.root_node_id == Some(node.id);
                        path_is_t = pp.gate_type.is_t();
                        break;
                    }
                }
                let label_text = match node.node_type {
                    NodeType::Data => String::new(),
                    NodeType::Magic => {
                        if border_color_idx.is_some() {
                            if is_root && path_is_t { "T".to_string() } else { String::new() }
                        } else if self.is_cultivating(node.id) {
                            (self.cultivation_times[node.id as usize]
                                - self.busy_counts[node.id as usize])
                                .to_string()
                        } else if pauli_product_paths.is_empty() {
                            label.clone()
                        } else {
                            "T".to_string()
                        }
                    }
                    NodeType::Bus => {
                        if pauli_product_paths.is_empty() && border_color_idx.is_none() {
                            label.clone()
                        } else {
                            String::new()
                        }
                    }
                };
                NodeDrawInfo {
                    x,
                    y,
                    half_x,
                    half_y,
                    is_data_x,
                    border_color_idx,
                    is_root,
                    label_text,
                }
            })
            .collect()
    }

    /// Builds a map from rounded node position → color index for non-data path nodes.
    /// Used to check if a routing node adjacent to a data group edge is in the same product.
    fn build_routing_pos_color(
        &self, node_infos: &[NodeDrawInfo],
    ) -> std::collections::HashMap<(i32, i32), usize> {
        self.nodes
            .iter()
            .zip(node_infos.iter())
            .filter(|(n, _)| n.node_type != NodeType::Data)
            .filter_map(|(n, info)| {
                info.border_color_idx.map(|ci| {
                    let (px, py) = n.pos;
                    (((px * 10.0).round() as i32, (py * 10.0).round() as i32), ci)
                })
            })
            .collect()
    }

    /// Pass 1: draw filled rectangles for all nodes.
    fn draw_node_fills(
        &self, chart: &mut PlotChart, node_infos: &[NodeDrawInfo], path_colors: &[RGBAColor],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            let fill = match node.node_type {
                NodeType::Magic => {
                    if let Some(ci) = info.border_color_idx {
                        path_colors[ci].mix(0.35).filled()
                    } else {
                        RGBColor(0xFF, 0xDD, 0x44).mix(0.25).filled()
                    }
                }
                NodeType::Bus => {
                    if let Some(ci) = info.border_color_idx {
                        path_colors[ci].mix(0.35).filled()
                    } else {
                        RGBColor(0x88, 0x88, 0x88).mix(0.15).filled()
                    }
                }
                NodeType::Data => RGBColor(0x44, 0x88, 0xFF).mix(0.25).filled(),
            };
            draw_rect(chart, info.x, info.y, info.half_x, info.half_y, fill)?;
        }
        Ok(())
    }

    /// Pass 2 + 2b + 2c: draw solid borders for all nodes and path outlines.
    fn draw_node_borders(
        &self, chart: &mut PlotChart, node_infos: &[NodeDrawInfo],
        pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Pass 2 + 2b: black border for all non-data nodes; grey separator for path nodes.
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            if node.node_type == NodeType::Data {
                continue;
            }
            let (x, y, hx, hy) = (info.x, info.y, info.half_x, info.half_y);
            draw_rect(chart, x, y, hx, hy, BLACK.stroke_width(1))?;
            if info.border_color_idx.is_some() {
                draw_rect(chart, x, y, hx, hy, RGBColor(0xAA, 0xAA, 0xAA).stroke_width(1))?;
            }
        }
        for (_, path_graph) in pauli_product_paths.iter() {
            let path_positions: std::collections::HashSet<(i32, i32)> = path_graph
                .iter_nodes()
                .map(|id| {
                    let (px, py) = self.nodes[id as usize].pos;
                    ((px * 10.0).round() as i32, (py * 10.0).round() as i32)
                })
                .collect();
            for id in path_graph.iter_nodes() {
                let node = &self.nodes[id as usize];
                if node.node_type == NodeType::Data {
                    continue;
                }
                let info = &node_infos[id as usize];
                let (x, y, hx, hy) = (info.x, info.y, info.half_x, info.half_y);
                let xi = (x * 10.0).round() as i32;
                let yi = (y * 10.0).round() as i32;
                let step_x = (hx * 2.0 * 10.0).round() as i32;
                let step_y = (hy * 2.0 * 10.0).round() as i32;
                for &(p1, p2, dx, dy) in &[
                    ((x - hx, y - hy), (x + hx, y - hy), 0, -step_y),
                    ((x - hx, y + hy), (x + hx, y + hy), 0, step_y),
                    ((x - hx, y - hy), (x - hx, y + hy), -step_x, 0),
                    ((x + hx, y - hy), (x + hx, y + hy), step_x, 0),
                ] {
                    if !path_positions.contains(&(xi + dx, yi + dy)) {
                        draw_line(chart, p1, p2, BLACK.stroke_width(2))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Pass 2c + 3: draw colored solid borders (Z-row) and dashed borders (X-row) for data groups.
    fn draw_data_borders(
        &self, chart: &mut PlotChart, data_groups: &[DataGroup], node_infos: &[NodeDrawInfo],
        path_colors: &[RGBAColor],
        routing_pos_color: &std::collections::HashMap<(i32, i32), usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for group in data_groups {
            let (left, right) = (group.col - 0.5, group.col + 0.5);
            let (y_top, y_bot) = (group.y_x.max(group.y_z), group.y_x.min(group.y_z));
            let xl_ci = node_infos[group.x_left_id as usize].border_color_idx;
            let xr_ci = node_infos[group.x_right_id as usize].border_color_idx;
            let zl_ci = node_infos[group.z_left_id as usize].border_color_idx;
            let zr_ci = node_infos[group.z_right_id as usize].border_color_idx;

            // Solid: top edge (if routing node above is in same path and !sides_only).
            let top_ci = (!self.sides_only)
                .then(|| {
                    let key =
                        ((group.col * 10.0).round() as i32, ((y_top + 1.0) * 10.0).round() as i32);
                    xl_ci
                        .or(xr_ci)
                        .filter(|&ci| routing_pos_color.get(&key).map_or(false, |&r| r == ci))
                })
                .flatten();
            draw_line(
                chart,
                (left, y_top + 0.5),
                (right, y_top + 0.5),
                ci_color(top_ci, path_colors).stroke_width(2),
            )?;
            // Solid: Z-row left and right sides.
            draw_line(
                chart,
                (left, group.y_z - 0.5),
                (left, group.y_z + 0.5),
                ci_color(zl_ci, path_colors).stroke_width(2),
            )?;
            draw_line(
                chart,
                (right, group.y_z - 0.5),
                (right, group.y_z + 0.5),
                ci_color(zr_ci, path_colors).stroke_width(2),
            )?;

            // Dashed: bottom edge (if routing node below is in same path and !sides_only).
            let bot_ci = (!self.sides_only)
                .then(|| {
                    let key =
                        ((group.col * 10.0).round() as i32, ((y_bot - 1.0) * 10.0).round() as i32);
                    zl_ci
                        .or(zr_ci)
                        .filter(|&ci| routing_pos_color.get(&key).map_or(false, |&r| r == ci))
                })
                .flatten();
            draw_dashed(
                chart,
                false,
                left,
                y_bot - 0.5,
                right - left,
                ci_color(bot_ci, path_colors),
            )?;
            // Dashed: X-row left and right sides.
            draw_dashed(chart, true, left, group.y_x - 0.5, 1.0, ci_color(xl_ci, path_colors))?;
            draw_dashed(chart, true, right, group.y_x - 0.5, 1.0, ci_color(xr_ci, path_colors))?;
        }
        Ok(())
    }

    /// Pass 4: draw all text labels (node labels, data group labels, side labels, product labels).
    fn draw_labels(
        &self, chart: &mut PlotChart, data_groups: &[DataGroup], node_infos: &[NodeDrawInfo],
        path_colors: &[RGBAColor],
        routing_pos_color: &std::collections::HashMap<(i32, i32), usize>,
        pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Non-data node labels.
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            if node.node_type == NodeType::Data || info.label_text.is_empty() {
                continue;
            }
            let font = if info.label_text == "T" {
                ("sans-serif", 20, FontStyle::Bold).into_font()
            } else {
                ("sans-serif", 18).into_font()
            };
            draw_text(chart, &info.label_text, info.x - 0.17, info.y + 0.09, font)?;
        }
        // Data group qubit-number labels.
        for group in data_groups {
            let hy = 0.5f32;
            let y_top = group.y_x.max(group.y_z);
            let y_bot = group.y_x.min(group.y_z);
            let x_is_top = group.y_x > group.y_z;
            let (x_row_y, z_row_y) = if x_is_top { (y_top, y_bot) } else { (y_bot, y_top) };
            let x_label = self.labels[group.x_left_id as usize]
                .trim_end_matches('X')
                .trim_start_matches('d')
                .to_string();
            let z_label = self.labels[group.z_right_id as usize]
                .trim_end_matches('Z')
                .trim_start_matches('d')
                .to_string();
            let label_x = group.col - 0.17;
            draw_text(chart, &x_label, label_x, x_row_y + 0.09, ("sans-serif", 24).into_font())?;
            draw_text(chart, &z_label, label_x, z_row_y + 0.09, ("sans-serif", 24).into_font())?;

            // Side labels (boxed X/Z letters on colored edges).
            let xl_ci = node_infos[group.x_left_id as usize].border_color_idx;
            let xr_ci = node_infos[group.x_right_id as usize].border_color_idx;
            let zl_ci = node_infos[group.z_left_id as usize].border_color_idx;
            let zr_ci = node_infos[group.z_right_id as usize].border_color_idx;
            let (left, right) = (group.col - 0.5, group.col + 0.5);
            let (top, bot) = (y_top + hy, y_bot - hy);

            // Top/bottom labels suppressed when sides_only.
            if !self.sides_only {
                let top_key =
                    ((group.col * 10.0).round() as i32, ((top + 0.5) * 10.0).round() as i32);
                if let Some(ci) = xl_ci
                    .or(xr_ci)
                    .filter(|&ci| routing_pos_color.get(&top_key).map_or(false, |&r| r == ci))
                {
                    draw_boxed_label(chart, "Z", group.col, top, path_colors[ci])?;
                }
                let bot_key =
                    ((group.col * 10.0).round() as i32, ((bot - 0.5) * 10.0).round() as i32);
                if let Some(ci) = zl_ci
                    .or(zr_ci)
                    .filter(|&ci| routing_pos_color.get(&bot_key).map_or(false, |&r| r == ci))
                {
                    draw_boxed_label(chart, "X", group.col, bot, path_colors[ci])?;
                }
            }
            // Side labels always shown (X on dashed X-row sides, Z on solid Z-row sides).
            for &(letter, x, y, ci) in &[
                ("X", left, x_row_y, xl_ci),
                ("X", right, x_row_y, xr_ci),
                ("Z", left, z_row_y, zl_ci),
                ("Z", right, z_row_y, zr_ci),
            ] {
                if let Some(ci) = ci {
                    draw_boxed_label(chart, letter, x, y, path_colors[ci])?;
                }
            }
        }
        // Product operator + ID labels.
        for (i, (pp, path_graph)) in pauli_product_paths.iter().enumerate() {
            // Find the widest non-data row to place the label.
            let mut row_map: std::collections::HashMap<i32, Vec<f32>> =
                std::collections::HashMap::new();
            for id in path_graph.iter_nodes() {
                let node = &self.nodes[id as usize];
                if node.node_type != NodeType::Data {
                    let (px, py) = node.pos;
                    row_map.entry((py * 10.0).round() as i32).or_default().push(px);
                }
            }
            if let Some((&row_key, xs)) = row_map.iter().max_by_key(|(_, xs)| xs.len()) {
                let row_y = row_key as f32 / 10.0;
                let center_x = (xs.iter().cloned().fold(f32::INFINITY, f32::min)
                    + xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                    / 2.0;
                let product_str = pp.to_operator_str();
                let tw = product_str.len() as f32 * 0.125;
                draw_rect_coords(
                    chart,
                    center_x - tw / 2.0 - 0.05,
                    row_y - 0.15,
                    center_x + tw / 2.0 + 0.05,
                    row_y + 0.15,
                    path_colors[i].mix(0.2).filled(),
                )?;
                draw_text(
                    chart,
                    &product_str,
                    center_x - tw / 2.0,
                    row_y + 0.09,
                    ("sans-serif", 22).into_font(),
                )?;
                let id_str = pp.id.to_string();
                let id_w = id_str.len() as f32 * 0.10;
                draw_text(
                    chart,
                    &id_str,
                    center_x - id_w / 2.0,
                    row_y - 0.12,
                    ("sans-serif", 14).into_font(),
                )?;
            }
        }
        Ok(())
    }
}

// ── Module-level plot drawing helpers ─────────────────────────────────────────

/// Type alias for the chart context used in plot helpers.
type PlotChart<'a> = plotters::prelude::ChartContext<
    'a,
    BitMapBackend<'a>,
    plotters::prelude::Cartesian2d<
        plotters::coord::types::RangedCoordf32,
        plotters::coord::types::RangedCoordf32,
    >,
>;

/// Returns the path color for index `ci`, or black if `None`.
fn ci_color(ci: Option<usize>, path_colors: &[RGBAColor]) -> RGBAColor {
    ci.map_or_else(|| BLACK.to_rgba(), |i| path_colors[i])
}

/// Draws a single filled/stroked rectangle centered at (cx, cy) with half-extents (hx, hy).
fn draw_rect(
    chart: &mut PlotChart, cx: f32, cy: f32, hx: f32, hy: f32, style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Rectangle::new(
        [(cx - hx, cy - hy), (cx + hx, cy + hy)],
        style,
    )))?;
    Ok(())
}

/// Draws a rectangle given explicit corner coordinates.
fn draw_rect_coords(
    chart: &mut PlotChart, x0: f32, y0: f32, x1: f32, y1: f32, style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Rectangle::new([(x0, y0), (x1, y1)], style)))?;
    Ok(())
}

/// Draws a straight line between two points.
fn draw_line(
    chart: &mut PlotChart, p1: (f32, f32), p2: (f32, f32), style: ShapeStyle,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(LineSeries::new(vec![p1, p2], style))?;
    Ok(())
}

/// Draws a text label at position (x, y).
fn draw_text<'a>(
    chart: &mut PlotChart, text: &str, x: f32, y: f32, font: plotters::style::FontDesc<'a>,
) -> Result<(), Box<dyn std::error::Error>> {
    chart.draw_series(std::iter::once(Text::new(text.to_string(), (x, y), font)))?;
    Ok(())
}

/// Draws a dashed line. If `vertical` is true the line runs along Y from (x0, y0);
/// otherwise it runs along X. `len` is the total length. Erases underneath first.
fn draw_dashed(
    chart: &mut PlotChart, vertical: bool, x0: f32, y0: f32, len: f32, c: RGBAColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let end = if vertical { (x0, y0 + len) } else { (x0 + len, y0) };
    draw_line(chart, (x0, y0), end, WHITE.stroke_width(3))?;
    let (dash, gap) = (0.08f32, 0.06f32);
    let mut pos = 0.0f32;
    let mut on = true;
    while pos < len {
        let seg = if on { dash } else { gap }.min(len - pos);
        if on {
            let (p1, p2) = if vertical {
                ((x0, y0 + pos), (x0, y0 + pos + seg))
            } else {
                ((x0 + pos, y0), (x0 + pos + seg, y0))
            };
            draw_line(chart, p1, p2, c.stroke_width(2))?;
        }
        pos += seg;
        on = !on;
    }
    Ok(())
}

/// Draws a white-filled, colored-outlined box with a centered letter inside.
fn draw_boxed_label(
    chart: &mut PlotChart, letter: &str, cx: f32, cy: f32, c: RGBAColor,
) -> Result<(), Box<dyn std::error::Error>> {
    let (bh, bw) = (0.18f32, 0.13f32);
    draw_rect_coords(chart, cx - bw, cy - bh, cx + bw, cy + bh, WHITE.filled())?;
    draw_rect_coords(
        chart,
        cx - bw,
        cy - bh,
        cx + bw,
        cy + bh,
        Into::<ShapeStyle>::into(c).stroke_width(2),
    )?;
    // Use Text directly here because .color() returns TextStyle, not FontDesc.
    chart.draw_series(std::iter::once(Text::new(
        letter.to_string(),
        (cx - bw * 0.45, cy + bh * 0.35),
        ("sans-serif", 28u32).into_font().color(&c),
    )))?;
    Ok(())
}

/// Converts HSV color space to RGB for plotting.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match (h * 6.0).floor() as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8)
}
