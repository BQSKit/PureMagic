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
        let num_colors = pauli_product_paths.len().max(1);
        let path_colors: Vec<RGBAColor> = (0..num_colors)
            .map(|i| {
                let hue = (i as f64) / (num_colors as f64);
                let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
                RGBColor(r, g, b).to_rgba()
            })
            .collect();
        // Collect per-node metadata needed across multiple draw passes.
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

        // Build a map from (x*10 rounded, y*10 rounded) → node id for data nodes,
        // so we can find the partner nodes in a double-data-qubit group.
        // A double data qubit group has 4 nodes:
        //   X-left  (col-0.25, row_x),  X-right (col+0.25, row_x)  — X row (dashed)
        //   Z-left  (col-0.25, row_z),  Z-right (col+0.25, row_z)  — Z row (solid)
        // where row_x and row_z are adjacent (differ by 1.0 in plot y).
        // We identify groups by the column center (col) and the pair of rows.
        // For each data node, we store which group it belongs to and its role.
        // group_id is (col*10 as i32, min_row*10 as i32) — unique per group.
        let data_pos_map: std::collections::HashMap<(i32, i32), u16> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Data)
            .map(|n| {
                let (px, py) = n.pos;
                ((px * 10.0).round() as i32, (py * 10.0).round() as i32, n.id)
            })
            .map(|(xi, yi, id)| ((xi, yi), id))
            .collect();

        // For each data node, find its group: the 4 nodes sharing the same column center.
        // col_center = round(x) (the integer column), x is col±0.25.
        // The X-left and Z-left share x=col-0.25; X-right and Z-right share x=col+0.25.
        // X and Z rows are adjacent: one has label ending 'X', other 'Z'.
        // We identify the group by (col_center*10, min_y*10).
        // For each data node, compute group_id and role (x_left, x_right, z_left, z_right).
        // We'll store: group_id → (x_left_id, x_right_id, z_left_id, z_right_id, col_x, col_y_x, col_y_z)
        // where col_y_x is the y of the X row and col_y_z is the y of the Z row.
        // Build groups: key = (col_center*10, min_y*10)
        struct DataGroup {
            col: f32, // integer column center
            y_x: f32, // y of X row (dashed, higher y = visually upper)
            y_z: f32, // y of Z row (solid, lower y = visually lower)
            x_left_id: u16,
            x_right_id: u16,
            z_left_id: u16,
            z_right_id: u16,
        }
        let mut data_groups: Vec<DataGroup> = Vec::new();
        // Track which node ids have been assigned to a group
        let mut grouped_data_ids: std::collections::HashSet<u16> = std::collections::HashSet::new();
        for node in &self.nodes {
            if node.node_type != NodeType::Data {
                continue;
            }
            if grouped_data_ids.contains(&node.id) {
                continue;
            }
            let label = &self.labels[node.id as usize];
            if !label.ends_with('X') {
                continue;
            } // process X nodes as group anchors
            let (x, y) = node.pos;
            let xi = (x * 10.0).round() as i32;
            let yi = (y * 10.0).round() as i32;
            // This is an X node. Find its X partner (same y, x differs by 0.5 = 5 in *10 units)
            // X-left has x = col-0.25, X-right has x = col+0.25
            // The partner is at xi ± 5 (same yi)
            // Find partner: try both xi+5 and xi-5
            let partner_xi = if data_pos_map.contains_key(&(xi + 5, yi)) {
                xi + 5
            } else if data_pos_map.contains_key(&(xi - 5, yi)) {
                xi - 5
            } else {
                continue;
            }; // no partner found, skip
            let (x_left_id, x_right_id) = if xi < partner_xi {
                (
                    node.id,
                    *match data_pos_map.get(&(partner_xi, yi)) {
                        Some(id) => id,
                        None => continue,
                    },
                )
            } else {
                (
                    *match data_pos_map.get(&(partner_xi, yi)) {
                        Some(id) => id,
                        None => continue,
                    },
                    node.id,
                )
            };
            // Find Z partners: same x positions, adjacent y (yi ± 10)
            let z_yi_candidates = [yi - 10, yi + 10];
            let mut z_yi = None;
            for &zy in &z_yi_candidates {
                let left_xi = xi.min(partner_xi);
                let right_xi = xi.max(partner_xi);
                if let (Some(&zl), Some(&zr)) =
                    (data_pos_map.get(&(left_xi, zy)), data_pos_map.get(&(right_xi, zy)))
                {
                    let zl_label = &self.labels[zl as usize];
                    let zr_label = &self.labels[zr as usize];
                    if zl_label.ends_with('Z') && zr_label.ends_with('Z') {
                        z_yi = Some((zy, zl, zr));
                        break;
                    }
                }
            }
            let (zy, z_left_id, z_right_id) = match z_yi {
                Some(v) => v,
                None => continue,
            };
            // Compute col_center from actual node x position to avoid rounding drift.
            // The two X nodes are at col±0.25, so col = x + 0.25 (for left node) or x - 0.25 (for right).
            // Equivalently, col = (x_left + x_right) / 2 where x_left < x_right.
            // We know this node's x; the partner's x is at xi±5 in *10 units = ±0.5 in actual coords.
            // But we already have the actual x of this node, so:
            let partner_x = if xi < partner_xi { x + 0.5 } else { x - 0.5 };
            let col_center = if x < partner_x { x + 0.25 } else { x - 0.25 };
            let y_x = yi as f32 / 10.0;
            let y_z = zy as f32 / 10.0;
            grouped_data_ids.insert(x_left_id);
            grouped_data_ids.insert(x_right_id);
            grouped_data_ids.insert(z_left_id);
            grouped_data_ids.insert(z_right_id);
            data_groups.push(DataGroup {
                col: col_center,
                y_x,
                y_z,
                x_left_id,
                x_right_id,
                z_left_id,
                z_right_id,
            });
        }
        // Map from node id → group index (for suppressing internal borders and labels)
        let _node_to_group: std::collections::HashMap<u16, usize> = data_groups
            .iter()
            .enumerate()
            .flat_map(|(gi, g)| {
                vec![(g.x_left_id, gi), (g.x_right_id, gi), (g.z_left_id, gi), (g.z_right_id, gi)]
            })
            .collect();

        let node_infos: Vec<NodeDrawInfo> = self
            .nodes
            .iter()
            .map(|node| {
                let (x, y) = node.pos;
                // Data nodes are offset ±0.25 horizontally within their column,
                // so they are 0.25 wide (half=0.25) but still full height (half_y=0.5).
                // Non-data nodes fill a full cell in both dimensions.
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
                // Data node labels are handled at the group level (drawn once per row-pair),
                // so suppress per-node labels for data nodes in a group.
                let label_text = match node.node_type {
                    NodeType::Data => String::new(), // labels drawn per-group below
                    NodeType::Magic => {
                        if let Some(_) = border_color_idx {
                            // On a path: show "T" only if this is the magic root of a T product
                            if is_root && path_is_t { "T".to_string() } else { String::new() }
                        } else {
                            // Idle magic: always show cultivation countdown if cultivating
                            if self.is_cultivating(node.id) {
                                (self.cultivation_times[node.id as usize]
                                    - self.busy_counts[node.id as usize])
                                    .to_string()
                            } else if pauli_product_paths.is_empty() {
                                label.clone()
                            } else {
                                // Ready magic node while paths are shown: show "T"
                                "T".to_string()
                            }
                        }
                    }
                    NodeType::Bus => {
                        // When paths are being plotted, suppress all bus labels
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
            .collect();

        // Build a map from rounded position → border_color_idx for non-data nodes.
        // Used to check if a routing node adjacent to a data group edge is in the same product path.
        let routing_pos_color: std::collections::HashMap<(i32, i32), usize> = self
            .nodes
            .iter()
            .zip(node_infos.iter())
            .filter(|(n, _)| n.node_type != NodeType::Data)
            .filter_map(|(n, info)| {
                info.border_color_idx.map(|ci| {
                    let (px, py) = n.pos;
                    (((px * 10.0).round() as i32, (py * 10.0).round() as i32), ci)
                })
            })
            .collect();

        // Pass 1: draw all node fills.
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            let node_color = match node.node_type {
                NodeType::Magic => {
                    if let Some(ci) = info.border_color_idx {
                        // Both root and routing magic nodes on a path: fill with path color tint
                        path_colors[ci].mix(0.35).filled()
                    } else {
                        RGBColor(0xFF, 0xDD, 0x44).mix(0.25).filled() // low-alpha yellow for idle/ready
                    }
                }
                NodeType::Bus => {
                    if let Some(ci) = info.border_color_idx {
                        path_colors[ci].mix(0.35).filled() // fill with path color tint
                    } else {
                        RGBColor(0x88, 0x88, 0x88).mix(0.15).filled() // low-alpha grey
                    }
                }
                NodeType::Data => RGBColor(0x44, 0x88, 0xFF).mix(0.25).filled(), // low-alpha blue
            };
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (info.x - info.half_x, info.y - info.half_y),
                    (info.x + info.half_x, info.y + info.half_y),
                ],
                node_color,
            )))?;
        }

        // Pass 2: draw solid borders.
        // - Data nodes: borders drawn per-group (outer only) below
        // - All other nodes: full solid rectangle border
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            if node.node_type == NodeType::Data {
                continue;
            } // handled per-group
            let (x, y, hx, hy) = (info.x, info.y, info.half_x, info.half_y);
            // non-data node: black border
            let border_style = BLACK.stroke_width(1);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - hx, y - hy), (x + hx, y + hy)],
                border_style,
            )))?;
        }
        // Pass 2b: draw black outer border around each path as a group.
        // We draw a black rectangle for each path node, but use a faint separator
        // for internal edges (the per-node black border gives the outer group outline
        // because adjacent path nodes share edges — the outer edges get drawn once,
        // inner shared edges get drawn twice and appear as faint lines).
        // Strategy: draw faint grey per-node borders for path nodes (internal separators),
        // then draw the outer group border by finding the bounding box per path.
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            if node.node_type == NodeType::Data {
                continue;
            } // data borders handled per-group
            if info.border_color_idx.is_some() {
                let (x, y, hx, hy) = (info.x, info.y, info.half_x, info.half_y);
                // Faint separator lines between path nodes
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x - hx, y - hy), (x + hx, y + hy)],
                    RGBColor(0xAA, 0xAA, 0xAA).stroke_width(1),
                )))?;
            }
        }
        // Draw black outer border for each path.
        // For each path node, draw only the 4 sides that are NOT shared with another path node.
        // This correctly handles L-shaped and non-rectangular paths.
        for (_, path_graph) in pauli_product_paths.iter() {
            // Build a set of (rounded x*10, rounded y*10) for fast lookup
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
                } // data borders handled per-group
                let info = &node_infos[id as usize];
                let (x, y, hx, hy) = (info.x, info.y, info.half_x, info.half_y);
                let xi = (x * 10.0).round() as i32;
                let yi = (y * 10.0).round() as i32;
                // For non-data nodes (hx==0.5), neighbors are 1 unit away in x or y.
                // For data nodes (hx==0.25), the pair partner is 0.5 units away in x.
                // We check each of the 4 sides: if the neighbor cell in that direction
                // is also in the path, skip that side (it's internal).
                let step_x = (hx * 2.0 * 10.0).round() as i32; // 10 for non-data, 5 for data
                let step_y = (hy * 2.0 * 10.0).round() as i32; // always 10
                // Sides: (start, end, neighbor key offset)
                // bottom: y - hy edge, neighbor is below (yi - step_y)
                // top:    y + hy edge, neighbor is above (yi + step_y)
                // left:   x - hx edge, neighbor is left  (xi - step_x)
                // right:  x + hx edge, neighbor is right (xi + step_x)
                let sides: &[((f32, f32), (f32, f32), i32, i32)] = &[
                    ((x - hx, y - hy), (x + hx, y - hy), 0, -step_y), // bottom
                    ((x - hx, y + hy), (x + hx, y + hy), 0, step_y),  // top
                    ((x - hx, y - hy), (x - hx, y + hy), -step_x, 0), // left
                    ((x + hx, y - hy), (x + hx, y + hy), step_x, 0),  // right
                ];
                for &(p1, p2, dx, dy) in sides {
                    let neighbor = (xi + dx, yi + dy);
                    if !path_positions.contains(&neighbor) {
                        chart.draw_series(LineSeries::new(vec![p1, p2], BLACK.stroke_width(2)))?;
                    }
                }
            }
        }

        // Pass 2c: draw colored solid borders for data node groups.
        // Done AFTER Pass 2b so colored lines overwrite any black outer-border lines
        // that routing nodes may have drawn over the data group edges.
        //
        // Node → solid outer sides:
        //   x_left/x_right: top edge (shared, spans full group width)
        //   z_left:  left side of Z row
        //   z_right: right side of Z row
        for group in &data_groups {
            let hy = 0.5f32;
            let left = group.col - 0.5;
            let right = group.col + 0.5;
            let (y_top, _y_bot) =
                if group.y_x > group.y_z { (group.y_x, group.y_z) } else { (group.y_z, group.y_x) };
            let top = y_top + hy;
            let z_row_top = group.y_z + hy;
            let z_row_bot = group.y_z - hy;
            let xl_ci = node_infos[group.x_left_id as usize].border_color_idx;
            let xr_ci = node_infos[group.x_right_id as usize].border_color_idx;
            let zl_ci = node_infos[group.z_left_id as usize].border_color_idx;
            let zr_ci = node_infos[group.z_right_id as usize].border_color_idx;
            let color = |ci: Option<usize>| -> RGBAColor {
                if let Some(i) = ci { path_colors[i] } else { BLACK.to_rgba() }
            };
            // Top solid edge: only color if a routing node above the X row is in the same path,
            // and sides_only is not set.
            let top_adj_key =
                ((group.col * 10.0).round() as i32, ((y_top + 1.0) * 10.0).round() as i32);
            let top_edge_ci = if self.sides_only {
                None
            } else {
                xl_ci.or(xr_ci).filter(|&ci| {
                    routing_pos_color.get(&top_adj_key).map_or(false, |&rci| rci == ci)
                })
            };
            let top_color = color(top_edge_ci);
            chart.draw_series(LineSeries::new(
                vec![(left, top), (right, top)],
                top_color.stroke_width(2),
            ))?;
            // Left solid side of Z row: color from z_left
            chart.draw_series(LineSeries::new(
                vec![(left, z_row_bot), (left, z_row_top)],
                color(zl_ci).stroke_width(2),
            ))?;
            // Right solid side of Z row: color from z_right
            chart.draw_series(LineSeries::new(
                vec![(right, z_row_bot), (right, z_row_top)],
                color(zr_ci).stroke_width(2),
            ))?;
            let _ = hy;
        }

        // Pass 3: draw dashed borders for X-basis data qubit rows.
        // For each group, draw dashed lines on the outer edges of the X row:
        // top (if X is top row) or bottom (if X is bottom row), plus the mid line gets dashed too.
        // We draw white gaps first (to erase any solid border underneath), then black dashes.
        for group in &data_groups {
            let hy = 0.5f32;
            let left = group.col - 0.5;
            let right = group.col + 0.5;
            let width = right - left;
            let (y_top, y_bot) =
                if group.y_x > group.y_z { (group.y_x, group.y_z) } else { (group.y_z, group.y_x) };
            let bot = y_bot - hy;
            let x_row_bot = group.y_x - hy;
            // Per-node path colors
            let xl_ci = node_infos[group.x_left_id as usize].border_color_idx;
            let xr_ci = node_infos[group.x_right_id as usize].border_color_idx;
            let zl_ci = node_infos[group.z_left_id as usize].border_color_idx;
            let zr_ci = node_infos[group.z_right_id as usize].border_color_idx;
            let color = |ci: Option<usize>| -> RGBAColor {
                if let Some(i) = ci { path_colors[i] } else { BLACK.to_rgba() }
            };
            let dash = 0.08f32;
            let gap = 0.06f32;
            // Inline dashed horizontal line helper
            let draw_h_dash_colored = |chart: &mut plotters::prelude::ChartContext<
                '_,
                BitMapBackend<'_>,
                plotters::prelude::Cartesian2d<
                    plotters::coord::types::RangedCoordf32,
                    plotters::coord::types::RangedCoordf32,
                >,
            >,
                                       y: f32,
                                       x0: f32,
                                       len: f32,
                                       c: RGBAColor|
             -> Result<(), Box<dyn std::error::Error>> {
                chart.draw_series(LineSeries::new(
                    vec![(x0, y), (x0 + len, y)],
                    WHITE.stroke_width(3),
                ))?;
                let mut x = x0;
                let mut dist = 0.0f32;
                let mut on = true;
                while dist < len {
                    let seg = if on { dash } else { gap }.min(len - dist);
                    if on {
                        chart.draw_series(LineSeries::new(
                            vec![(x, y), (x + seg, y)],
                            c.stroke_width(2),
                        ))?;
                    }
                    x += seg;
                    dist += seg;
                    on = !on;
                }
                Ok(())
            };
            // Inline dashed vertical line helper
            let draw_v_dash_colored = |chart: &mut plotters::prelude::ChartContext<
                '_,
                BitMapBackend<'_>,
                plotters::prelude::Cartesian2d<
                    plotters::coord::types::RangedCoordf32,
                    plotters::coord::types::RangedCoordf32,
                >,
            >,
                                       x: f32,
                                       y0: f32,
                                       len: f32,
                                       c: RGBAColor|
             -> Result<(), Box<dyn std::error::Error>> {
                chart.draw_series(LineSeries::new(
                    vec![(x, y0), (x, y0 + len)],
                    WHITE.stroke_width(3),
                ))?;
                let mut y = y0;
                let mut dist = 0.0f32;
                let mut on = true;
                while dist < len {
                    let seg = if on { dash } else { gap }.min(len - dist);
                    if on {
                        chart.draw_series(LineSeries::new(
                            vec![(x, y), (x, y + seg)],
                            c.stroke_width(2),
                        ))?;
                    }
                    y += seg;
                    dist += seg;
                    on = !on;
                }
                Ok(())
            };
            // Bottom outer edge: dashed, only color if a routing node below the Z row is in the same path,
            // and sides_only is not set.
            let bot_adj_key =
                ((group.col * 10.0).round() as i32, ((y_bot - 1.0) * 10.0).round() as i32);
            let bot_edge_ci = if self.sides_only {
                None
            } else {
                zl_ci.or(zr_ci).filter(|&ci| {
                    routing_pos_color.get(&bot_adj_key).map_or(false, |&rci| rci == ci)
                })
            };
            let bot_color = color(bot_edge_ci);
            draw_h_dash_colored(&mut chart, bot, left, width, bot_color)?;
            // Left side of X row (half-height): dashed, color from x_left
            draw_v_dash_colored(&mut chart, left, x_row_bot, hy * 2.0, color(xl_ci))?;
            // Right side of X row (half-height): dashed, color from x_right
            draw_v_dash_colored(&mut chart, right, x_row_bot, hy * 2.0, color(xr_ci))?;
            let _ = (y_top, y_bot);
        }

        // Pass 4: draw all labels on top.
        // Non-data node labels
        for (node, info) in self.nodes.iter().zip(node_infos.iter()) {
            if node.node_type == NodeType::Data {
                continue;
            }
            if info.label_text.is_empty() {
                continue;
            }
            let font_style = if info.label_text == "T" {
                ("sans-serif", 20, FontStyle::Bold).into_font()
            } else {
                ("sans-serif", 18).into_font()
            };
            chart.draw_series(std::iter::once(Text::new(
                info.label_text.clone(),
                (info.x - 0.17, info.y + 0.09),
                font_style,
            )))?;
        }
        // Data node group labels: one label per row-pair, centered across both nodes.
        // X row: label = qubit number only (strip 'd' prefix and 'X' suffix, e.g. "d0X" → "0")
        // Z row: label = qubit number only (strip 'd' prefix and 'Z' suffix, e.g. "d1Z" → "1")
        for group in &data_groups {
            let hy = 0.5f32;
            let (y_top, y_bot) =
                if group.y_x > group.y_z { (group.y_x, group.y_z) } else { (group.y_z, group.y_x) };
            let x_is_top = group.y_x > group.y_z;
            // X row label: strip 'X' suffix and leading 'd' (e.g. "d0X" → "0")
            let x_left_label = &self.labels[group.x_left_id as usize];
            let x_label = x_left_label.trim_end_matches('X').trim_start_matches('d').to_string();
            // Z row label: strip 'Z' suffix and leading 'd' (e.g. "d1Z" → "1")
            let z_right_label = &self.labels[group.z_right_id as usize];
            let z_label = z_right_label.trim_end_matches('Z').trim_start_matches('d').to_string();
            let (x_row_y, z_row_y) = if x_is_top { (y_top, y_bot) } else { (y_bot, y_top) };
            // Center x across the full group width
            let label_x = group.col - 0.17;
            chart.draw_series(std::iter::once(Text::new(
                x_label,
                (label_x, x_row_y + 0.09),
                ("sans-serif", 24).into_font(),
            )))?;
            chart.draw_series(std::iter::once(Text::new(
                z_label,
                (label_x, z_row_y + 0.09),
                ("sans-serif", 24).into_font(),
            )))?;

            // Side labels: "X" on dashed sides, "Z" on solid sides — only on colored (in-product) sides.
            // Each label is drawn as: white-filled box outlined in product color, letter in product color.
            let xl_ci = node_infos[group.x_left_id as usize].border_color_idx;
            let xr_ci = node_infos[group.x_right_id as usize].border_color_idx;
            let zl_ci = node_infos[group.z_left_id as usize].border_color_idx;
            let zr_ci = node_infos[group.z_right_id as usize].border_color_idx;
            let left = group.col - 0.5;
            let right = group.col + 0.5;
            let top = y_top + hy;
            let bot = y_bot - hy;
            let side_font_size = 28u32;
            // Half-size of the label box in plot coordinates
            let bh = 0.18f32; // half-height of box
            let bw = 0.13f32; // half-width of box
            // Helper: draw a boxed letter at (cx, cy) with given color index
            let draw_boxed_label = |chart: &mut plotters::prelude::ChartContext<
                '_,
                BitMapBackend<'_>,
                plotters::prelude::Cartesian2d<
                    plotters::coord::types::RangedCoordf32,
                    plotters::coord::types::RangedCoordf32,
                >,
            >,
                                    letter: &str,
                                    cx: f32,
                                    cy: f32,
                                    ci: usize|
             -> Result<(), Box<dyn std::error::Error>> {
                let c = path_colors[ci];
                // White fill
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(cx - bw, cy - bh), (cx + bw, cy + bh)],
                    WHITE.filled(),
                )))?;
                // Colored outline
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(cx - bw, cy - bh), (cx + bw, cy + bh)],
                    Into::<ShapeStyle>::into(c).stroke_width(2),
                )))?;
                // Colored letter centered in box.
                // plotters text anchor is at the bottom-left (baseline).
                // Letter is too low → move baseline up by using a positive offset above center.
                let char_hw = bw * 0.45; // half char width estimate
                chart.draw_series(std::iter::once(Text::new(
                    letter.to_string(),
                    (cx - char_hw, cy + bh * 0.35),
                    ("sans-serif", side_font_size).into_font().color(&c),
                )))?;
                Ok(())
            };
            // Top solid edge → "Z" label: only if path runs through top edge and sides_only is not set
            if !self.sides_only {
                let top_adj_key =
                    ((group.col * 10.0).round() as i32, ((top + 0.5) * 10.0).round() as i32);
                if let Some(ci) = xl_ci.or(xr_ci).filter(|&ci| {
                    routing_pos_color.get(&top_adj_key).map_or(false, |&rci| rci == ci)
                }) {
                    draw_boxed_label(&mut chart, "Z", group.col, top, ci)?;
                }
                // Bottom dashed edge → "X" label: only if path runs through bottom edge
                let bot_adj_key2 =
                    ((group.col * 10.0).round() as i32, ((bot - 0.5) * 10.0).round() as i32);
                if let Some(ci) = zl_ci.or(zr_ci).filter(|&ci| {
                    routing_pos_color.get(&bot_adj_key2).map_or(false, |&rci| rci == ci)
                }) {
                    draw_boxed_label(&mut chart, "X", group.col, bot, ci)?;
                }
            }
            // Left dashed side of X row → "X" label on left edge, vertically centered in X row
            if let Some(ci) = xl_ci {
                draw_boxed_label(&mut chart, "X", left, x_row_y, ci)?;
            }
            // Right dashed side of X row → "X" label on right edge, vertically centered in X row
            if let Some(ci) = xr_ci {
                draw_boxed_label(&mut chart, "X", right, x_row_y, ci)?;
            }
            // Left solid side of Z row → "Z" label on left edge, vertically centered in Z row
            if let Some(ci) = zl_ci {
                draw_boxed_label(&mut chart, "Z", left, z_row_y, ci)?;
            }
            // Right solid side of Z row → "Z" label on right edge, vertically centered in Z row
            if let Some(ci) = zr_ci {
                draw_boxed_label(&mut chart, "Z", right, z_row_y, ci)?;
            }
        }
        for (i, (pp, path_graph)) in pauli_product_paths.iter().enumerate() {
            // Find the widest non-data row: group non-data path nodes by y-coordinate,
            // pick the row with the most nodes, and center the label on that row.
            // We exclude data nodes so the label never overlaps a data qubit cell.
            let mut row_map: std::collections::HashMap<i32, Vec<f32>> =
                std::collections::HashMap::new();
            for id in path_graph.iter_nodes() {
                let node = &self.nodes[id as usize];
                if node.node_type == NodeType::Data {
                    continue; // skip data nodes
                }
                let (px, py) = node.pos;
                // Use rounded y*10 as integer key to group by row
                let row_key = (py * 10.0).round() as i32;
                row_map.entry(row_key).or_default().push(px);
            }
            if let Some((_, xs)) = row_map.iter().max_by_key(|(_, xs)| xs.len()) {
                let row_y = {
                    // recover the actual y from the key
                    let key =
                        row_map.iter().max_by_key(|(_, xs)| xs.len()).map(|(k, _)| *k).unwrap();
                    key as f32 / 10.0
                };
                let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let center_x = (min_x + max_x) / 2.0;
                let product_str = pp.to_operator_str();
                let text_width = product_str.len() as f32 * 0.125;
                chart.draw_series(std::iter::once(Rectangle::new(
                    [
                        (center_x - text_width / 2.0 - 0.05, row_y - 0.15),
                        (center_x + text_width / 2.0 + 0.05, row_y + 0.15),
                    ],
                    path_colors[i].mix(0.2).filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    product_str,
                    (center_x - text_width / 2.0, row_y + 0.09),
                    ("sans-serif", 22).into_font(),
                )))?;
                // Product ID below the operator label
                let id_str = pp.id.to_string();
                let id_width = id_str.len() as f32 * 0.10;
                chart.draw_series(std::iter::once(Text::new(
                    id_str,
                    (center_x - id_width / 2.0, row_y - 0.12),
                    ("sans-serif", 14).into_font(),
                )))?;
            }
        }
        if !title_str.is_empty() {
            let lines: Vec<&str> = title_str.split('\n').collect();
            for (i, line) in lines.iter().enumerate() {
                chart.draw_series(std::iter::once(Text::new(
                    line.to_string(),
                    (-0.5, -0.8 - (i as f32 * 0.33)),
                    ("sans-serif", (6.0 * (self.num_rows as f64).sqrt()) as u32).into_font(),
                )))?;
            }
        }
        root.present()?;
        println!("Plotted topology to {}", plot_fname);
        Ok(())
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
