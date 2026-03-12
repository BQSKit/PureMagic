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
#[cfg(debug_assertions)]
use std::io::Write;
use std::io::{self, BufRead};
use std::path::Path;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents the topological layout of a surface code quantum processor.
/// Contains data, magic, and routing qubits arranged in a 2D grid.
/// Supports both magic routing and bus routing architectures.
pub struct TopoGraph {
    nodes: Vec<Node>,
    node_ids_from_labels: IndexMap<String, u16>,
    // Fast lookup for data nodes: indexed by qubit number, [0] = X node id, [1] = Z node id
    data_node_ids: Vec<[u16; 2]>,
    node_grid: Vec<Vec<Option<String>>>,
    num_cols: usize,
    num_rows: usize,
    topo_fname: String,
    circuit_fname: String,
    use_magic_routing: bool,
    pub num_data_qubits: usize,
    pub num_bus_qubits: usize,
    pub num_magic_qubits: usize,
    pub num_qubits: usize,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub busy_counts: Vec<i32>,
    pub cultivation_times: Vec<i32>,
}

impl TopoGraph {
    /// Creates an empty topology graph.
    pub fn new() -> Self {
        TopoGraph { nodes: Vec::new(),
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
                    cultivation_times: Vec::new() }
    }

    /// Initializes topology from file or generates a synthetic layout.
    /// Sets up node metadata, qubit pairings, and edge connectivity.
    pub fn set_topo(&mut self, min_num_qubits: usize, circuit_fname: &String,
                    topo_fname: &String, rseed: &u32, use_magic_routing: bool,
                    ancilla_rows: usize, sides_only: bool) {
        self.circuit_fname = circuit_fname.to_string();
        self.topo_fname = topo_fname.to_string();
        self.use_magic_routing = use_magic_routing;
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
                let qubit = node.label
                                .chars()
                                .skip(1)
                                .take_while(|c| c.is_numeric())
                                .collect::<String>()
                                .parse::<usize>()
                                .ok()
                                .unwrap();
                let term = node.label.chars().last().map(|c| c.to_string()).unwrap();
                let pair_qubit = if qubit % 2 == 0 { qubit + 1 } else { qubit - 1 };
                let paired_node_label = format!("d{}{}", pair_qubit, term);
                self.get_node_mut(node_id).paired_data_id =
                    self.node_ids_from_labels.get(&paired_node_label).copied();
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
    pub fn gen_pure_magic_topo(&mut self, min_num_qubits: usize, ancilla_rows: usize,
                               sides_only: bool) {
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
        let node1 = Node::new(id1,
                              None,
                              label1.to_string(),
                              col as f32 - 0.25,
                              (self.num_rows - 1 - row) as f32,
                              NodeType::Data);
        self.nodes.push(node1);
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.node_ids_from_labels.insert(label1, id1);
        self.num_nodes += 1;
        let id2 = self.num_nodes as u16;
        let label2 = format!("d{}{}", q + 1, op);
        let node2 = Node::new(id2,
                              None,
                              label2.to_string(),
                              col as f32 + 0.25,
                              (self.num_rows - 1 - row) as f32,
                              NodeType::Data);
        self.nodes.push(node2);
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
        let node = Node::new(self.num_nodes as u16,
                             None,
                             label.to_string(),
                             col as f32,
                             (self.num_rows - 1 - row) as f32,
                             node_type);
        self.nodes.push(node);
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
                                        vert_data_edges_to_add.push((label.clone(),
                                                                     up_label.clone()));
                                    }
                                }
                            }
                        }
                        if row < self.num_rows - 2 {
                            if label.starts_with('d') && label.ends_with('X') {
                                if let Some(ref up_label) = self.node_grid[col][row + 2] {
                                    if up_label.starts_with('b') || up_label.starts_with('m') {
                                        vert_data_edges_to_add.push((label.clone(),
                                                                     up_label.clone()));
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

        // Build fast data-node lookup: label format is "d{qubit}{basis}" where basis is 'X' or 'Z'
        self.data_node_ids.clear();
        for node in &self.nodes {
            if node.node_type == NodeType::Data {
                let label = &node.label;
                let basis: char = label.chars().last().unwrap();
                let qubit: usize = label[1..label.len() - 1].parse().unwrap();
                let basis_idx: usize = if basis == 'X' { 0 } else { 1 };
                if qubit >= self.data_node_ids.len() {
                    self.data_node_ids.resize(qubit + 1, [u16::MAX; 2]);
                }
                self.data_node_ids[qubit][basis_idx] = node.id;
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
        println!("  data:         {} ({:.3})",
                 self.num_data_qubits,
                 self.num_data_qubits as f64 / total);
        println!("  bus:          {} ({:.3})",
                 self.num_bus_qubits,
                 self.num_bus_qubits as f64 / total);
        println!("  magic:        {} ({:.3})",
                 self.num_magic_qubits,
                 self.num_magic_qubits as f64 / total);
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

    /// Returns a mutable iterator over all nodes.
    //pub fn iter_nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
    //    self.nodes.iter_mut()
    //}

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

    /// Writes topology grid to a text file (debug builds only).
    #[cfg(debug_assertions)]
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
                        write!(file,
                               "{}{} ",
                               label.chars().nth(0).unwrap_or(' '),
                               label.chars().last().unwrap_or(' '))?;
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
    pub fn plot(&self, fname_added: &str, pauli_product_paths: &[(PauliProduct, Rc<TreeGraph>)],
                title_str: &str)
                -> Result<(), Box<dyn std::error::Error>> {
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
        let mut chart = ChartBuilder::on(&root).margin(10)
                                               .set_label_area_size(LabelAreaPosition::Bottom, 50)
                                               .build_cartesian_2d(-1f32..self.num_cols as f32,
                                                                   -1f32..self.num_rows as f32)?;
        chart.draw_series(std::iter::once(Rectangle::new([(-0.5, -0.5),
                                                          (self.num_cols as f32 - 0.5,
                                                           self.num_rows as f32 - 0.5)],
                                                         RGBColor(220, 220, 220).filled())))?;
        for row in 0..=self.num_rows {
            chart.draw_series(LineSeries::new(vec![(-0.5, row as f32 - 0.5),
                                                   (self.num_cols as f32 - 0.5,
                                                    row as f32 - 0.5),],
                                              WHITE.stroke_width(3)))?;
        }
        for col in 0..=self.num_cols {
            chart.draw_series(LineSeries::new(vec![(col as f32 - 0.5, -0.5),
                                                   (col as f32 - 0.5,
                                                    self.num_rows as f32 - 0.5),],
                                              WHITE.stroke_width(3)))?;
        }
        let num_colors = pauli_product_paths.len().max(1);
        let path_colors: Vec<RGBAColor> =
            (0..num_colors).map(|i| {
                               let hue = (i as f64) / (num_colors as f64);
                               let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
                               RGBColor(r, g, b).to_rgba()
                           })
                           .collect();
        for node in &self.nodes {
            for nb_id in &node.nbors {
                let nb = &self.nodes[*nb_id as usize];
                let mut edge_color = &BLACK.mix(0.5).to_rgba();
                let mut stroke_width = 1;
                for (i, (_, path_graph)) in pauli_product_paths.iter().enumerate() {
                    if path_graph.contains_edge(node.id, *nb_id) {
                        edge_color = &path_colors[i];
                        stroke_width = 6;
                        break;
                    }
                }
                if node.pos.0 != nb.pos.0
                   && node.pos.1 != nb.pos.1
                   && nb.node_type == NodeType::Data
                {
                    continue;
                }
                let edge_points = self.generate_edge_points(node.pos, nb.pos);
                let mix = if node.pos.0 != nb.pos.0 && node.pos.1 < nb.pos.1 { 0.5 } else { 1.0 };
                chart.draw_series(LineSeries::new(edge_points,
                                                  edge_color.mix(mix).stroke_width(stroke_width)))?;
            }
        }
        for node in &self.nodes {
            let (x, y) = node.pos;
            let mut border_color = None;
            let mut root_node = None;
            for (i, (_, path_graph)) in pauli_product_paths.iter().enumerate() {
                if path_graph.contains_node(node.id) {
                    border_color = Some(&path_colors[i]);
                    root_node = path_graph.root_node_id.clone();
                    break;
                }
            }
            let node_color = match node.node_type {
                                 NodeType::Magic => {
                                     if border_color == None || Some(node.id.clone()) == root_node {
                                         RGBColor(0xFF, 0xBB, 0x99)
                                     } else {
                                         RGBColor(0xAA, 0xAA, 0xAA)
                                     }
                                 }
                                 NodeType::Bus => RGBColor(0xAA, 0xAA, 0xAA),
                                 NodeType::Data => RGBColor(0x99, 0x99, 0xFF),
                             }.filled();
            chart.draw_series(std::iter::once(Circle::new((x as f32, y as f32), 22, node_color)))?;
            if let Some(color) = border_color {
                chart.draw_series(std::iter::once(Circle::new((x as f32, y as f32),
                                                              22,
                                                              color.stroke_width(3))))?;
            }
            let label_text = match node.node_type {
                NodeType::Data => node.label.clone(),
                NodeType::Magic => {
                    if border_color == None {
                        if self.is_cultivating(node.id) {
                            (self.cultivation_times[node.id as usize]
                             - self.busy_counts[node.id as usize])
                                                                  .to_string()
                        } else if pauli_product_paths.is_empty() {
                            node.label.clone()
                        } else {
                            "  R".to_string()
                        }
                    } else {
                        node.label.clone()
                    }
                }
                NodeType::Bus => {
                    if border_color == None {
                        node.label.clone()
                    } else {
                        "  B".to_string()
                    }
                }
            };
            let font_style = if label_text.contains('R') {
                ("sans-serif", 18, FontStyle::Bold).into_font()
            } else {
                ("sans-serif", 18).into_font()
            };
            chart.draw_series(std::iter::once(Text::new(label_text,
                                                        (x as f32 - 0.17, y as f32 + 0.09),
                                                        font_style)))?;
        }
        for (i, (pp, path_graph)) in pauli_product_paths.iter().enumerate() {
            if let Some(first_data_node_id) =
                path_graph.iter_nodes()
                          .filter(|id| matches!(self.nodes[*id as usize].node_type, NodeType::Data))
                          .next()
            {
                let (x, y) = self.nodes[first_data_node_id as usize].pos;
                let product_str = pp.to_operator_str();
                let text_width = product_str.len() as f32 * 0.125;
                chart.draw_series(std::iter::once(Rectangle::new([(x as f32 - 0.3,
                                                                   y as f32 + 0.3),
                                                                  (x as f32 - 0.3
                                                                   + text_width,
                                                                   y as f32 + 0.55)],
                                                                 path_colors[i].mix(0.2)
                                                                               .filled())))?;
                chart.draw_series(std::iter::once(Text::new(product_str,
                                                            (x as f32 - 0.2, y as f32 + 0.5),
                                                            ("sans-serif", 22).into_font())))?;
            }
        }
        if !title_str.is_empty() {
            let lines: Vec<&str> = title_str.split('\n').collect();
            for (i, line) in lines.iter().enumerate() {
                chart.draw_series(std::iter::once(Text::new(line.to_string(),
                                                            (-0.5, -0.8 - (i as f32 * 0.33)),
                                                            ("sans-serif",
                                                             (6.0 * (self.num_rows as f64).sqrt())
                                                             as u32)
                                                                    .into_font())))?;
            }
        }
        root.present()?;
        println!("Plotted topology to {}", plot_fname);
        Ok(())
    }

    /// Generates edge path points for drawing (straight for aligned nodes, curved for diagonal).
    fn generate_edge_points(&self, pos1: (f32, f32), pos2: (f32, f32)) -> Vec<(f32, f32)> {
        let (x1, y1) = pos1;
        let (x2, y2) = pos2;
        if x1 == x2 || y1 == y2 {
            vec![(x1, y1), (x2, y2)]
        } else {
            let mid_x = (x1 + x2) / 2.0;
            let mid_y = (y1 + y2) / 2.0;
            let curve_offset = if x1 < x2 { 0.2 } else { -0.2 };
            let control_x = mid_x + curve_offset;
            let control_y = mid_y;
            let num_points = 10;
            (0..=num_points).map(|i| {
                                let t = i as f32 / num_points as f32;
                                let one_minus_t = 1.0 - t;
                                let x = one_minus_t.powi(2) * x1
                                        + 2.0 * one_minus_t * t * control_x
                                        + t.powi(2) * x2;
                                let y = one_minus_t.powi(2) * y1
                                        + 2.0 * one_minus_t * t * control_y
                                        + t.powi(2) * y2;
                                (x, y)
                            })
                            .collect()
        }
    }
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
