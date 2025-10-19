use crate::pauliproduct::PauliProduct;
use crate::utils::Timer;
use indexmap::{IndexMap, IndexSet};
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Magic,
    Bus,
    Data,
    Estabilizer,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub label: String,
    pub pos: (f64, f64),
    pub busy_count: i32,
    pub cultivation_time: i32,
    pub edges: IndexSet<String>,
    pub used: bool,
}

impl Node {
    fn new(label: String, x: f64, y: f64, node_type: NodeType, busy_count: i32,
           cultivation_time: i32)
           -> Self {
        Node { node_type,
               label,
               pos: (x, y),
               busy_count,
               cultivation_time,
               edges: IndexSet::new(),
               used: false }
    }

    fn add_edge(&mut self, other: &str) {
        self.edges.insert(other.to_string());
    }

    pub fn get_data_label_number(&self) -> Option<usize> {
        // Skip first character (node type)
        let after_type = self.label.get(1..)?;
        // For data nodes, parse number before operator
        if self.label.starts_with('d') {
            let op_pos = after_type.find(|c: char| c == 'X' || c == 'Z')?;
            return after_type[..op_pos].parse().ok();
        }
        None
    }

    pub fn is_cultivating(&self) -> bool {
        self.cultivation_time > 0 && self.busy_count < self.cultivation_time
    }
}

pub struct TopoGraph {
    nodes: IndexMap<String, Node>,
    node_grid: Vec<Vec<Option<String>>>,
    num_cols: usize,
    num_rows: usize,
    topo_fname: String,
    circuit_fname: String,
    use_magic_routing: bool,
    pub num_data_qubits: usize,
    pub num_bus_qubits: usize,
    pub num_magic_qubits: usize,
    pub num_estabilizer_qubits: usize,
    pub num_qubits: usize,
    pub num_edges: usize,
    pub num_nodes: usize,
}

impl TopoGraph {
    pub fn new() -> Self {
        TopoGraph { nodes: IndexMap::new(),
                    node_grid: Vec::new(),
                    num_cols: 0,
                    num_rows: 0,
                    num_data_qubits: 0,
                    num_bus_qubits: 0,
                    num_magic_qubits: 0,
                    num_estabilizer_qubits: 0,
                    num_qubits: 0,
                    num_edges: 0,
                    num_nodes: 0,
                    circuit_fname: String::new(),
                    topo_fname: String::new(),
                    use_magic_routing: true }
    }

    pub fn set_topo(&mut self, min_num_qubits: usize, circuit_fname: &String,
                    topo_fname: &String, rseed: &u32, use_magic_routing: bool,
                    ancilla_rows: usize) {
        let _timer = Timer::new("set_topo");
        self.circuit_fname = circuit_fname.to_string();
        self.topo_fname = topo_fname.to_string();
        self.use_magic_routing = use_magic_routing;

        if !self.topo_fname.is_empty() {
            if let Err(e) = self.read_topo_from_file(rseed) {
                eprintln!("Error reading topology file: {}", e);
            }
        } else if !use_magic_routing {
            // minimum layout with bus qubits
            let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
            let patch_rows = sq_dim / 2 + sq_dim % 2;
            let bus_rows = patch_rows + 1;

            let qubits_per_col = 2 * patch_rows;
            let num_data_cols = ((min_num_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;

            self.num_cols = 2 * num_data_cols + 3;
            self.num_rows = 2 + 2 * patch_rows + bus_rows;

            self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

            println!("Layout dimensions: {} {}", self.num_cols, self.num_rows);
            self.gen_topo(min_num_qubits);
        } else {
            // minimum layout with all magic qubits
            let spacing = ancilla_rows + 1;
            let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
            let patch_rows = sq_dim / 2 + sq_dim % 2;
            let patch_cols = ((min_num_qubits as f64) / ((2 * patch_rows) as f64)).ceil() as usize;
            //println!("sq dim {}", sq_dim);
            //println!("patch rows {}", patch_rows);
            //println!("patch cols {}", patch_cols);
            self.num_cols = patch_cols * spacing + spacing - 1;
            self.num_rows = patch_rows * (1 + spacing) + spacing - 1;
            self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];
            let mut qi = 0;
            let max_qi =
                if min_num_qubits % 2 == 0 { 2 * min_num_qubits } else { 2 * min_num_qubits + 1 };
            let row_gap = 1 + spacing;
            for col in 0..self.num_cols {
                for row in 0..self.num_rows {
                    if col % spacing == spacing - 1 {
                        // data column
                        if row % row_gap == spacing || row % row_gap == spacing - 1 {
                            if qi < max_qi {
                                let is_x = row % row_gap == spacing - 1;
                                self.add_double_data_qubit(qi, col, row, is_x);
                                qi += 2;
                            } else {
                                self.node_grid[col][row] =
                                    Some(self.add_qubit(col, row, NodeType::Magic));
                            }
                        } else {
                            let node_type = if row != 0
                                               && row != self.num_rows - 1
                                               && row % (2 * row_gap)
                                                  == row_gap + (ancilla_rows / 2)
                                               && col % (spacing * 2) == spacing - 1
                            {
                                NodeType::Estabilizer
                            } else {
                                NodeType::Magic
                            };
                            self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                        }
                    } else {
                        // magic column
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
                    }
                }
            }
            println!("Layout dimensions: {} {}", self.num_cols, self.num_rows);
            self.set_edges();
            println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
        }
        self.update_statistics();
    }

    pub fn read_topo_from_file(&mut self, rseed: &u32) -> io::Result<()> {
        let _timer = Timer::new("read_topo_from_file");
        // Read the grid layout
        let mut rows = Vec::new();
        let file = File::open(&self.topo_fname)?;
        for line in io::BufReader::new(file).lines() {
            let line = line?;
            let row: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if !row.is_empty() {
                rows.push(row);
            }
        }
        // Transpose grid from row-major to col-major order
        self.num_rows = rows.len();
        self.num_cols = rows[0].len();
        self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

        for (row_i, row) in rows.iter().enumerate() {
            for (col_i, col) in row.iter().enumerate() {
                self.node_grid[col_i][row_i] = Some(col.clone());
            }
        }
        // Count data nodes to create randomized mapping
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
            // Create randomized pairing for data nodes
            //let timer_seed = *rseed;
            let timer_seed =
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;

            let mut rng = StdRng::seed_from_u64(timer_seed);
            pair_indices.shuffle(&mut rng);
        }

        println!("Data node order {:?}", pair_indices);
        // Add nodes
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
                            Some('e') => NodeType::Estabilizer,
                            _ => continue,
                        };
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                    }
                }
            }
        }
        // Add edges
        self.set_edges();
        println!("Read topology with dimensions: {} {}", self.num_cols, self.num_rows);

        Ok(())
    }

    fn gen_topo(&mut self, min_num_qubits: usize) {
        self.add_border_row(0);
        self.add_border_column(0);

        let max_qi =
            if min_num_qubits % 2 == 0 { 2 * min_num_qubits } else { 2 * min_num_qubits + 1 };
        let mut qi = 0;
        for col in 1..self.num_cols - 1 {
            if col % 2 == 0 {
                // Data column
                for row in 1..self.num_rows - 1 {
                    if row % 3 + 1 == 2 {
                        if row != 1 && row != self.num_rows - 2 && col % 4 == 0 {
                            self.node_grid[col][row] =
                                Some(self.add_qubit(col, row, NodeType::Estabilizer));
                        } else {
                            let node_type = if self.use_magic_routing {
                                NodeType::Magic
                            } else {
                                NodeType::Bus
                            };
                            self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                        }
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
                // Bus column
                for row in 1..self.num_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                }
            }
        }

        self.add_border_column(self.num_cols - 1);
        self.add_border_row(self.num_rows - 1);
        self.set_edges();
        println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
    }

    fn add_border_row(&mut self, row: usize) {
        let node_type = if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus };
        // Add corner bus nodes
        self.node_grid[0][row] = Some(self.add_qubit(0, row, node_type));
        self.node_grid[self.num_cols - 1][row] =
            Some(self.add_qubit(self.num_cols - 1, row, node_type));
        // Add alternating magic nodes
        for col in 1..self.num_cols - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    fn add_border_column(&mut self, col: usize) {
        for row in 1..self.num_rows - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    fn add_double_data_qubit(&mut self, qi: usize, col: usize, row: usize, is_x: bool) {
        let q = if is_x { qi / 2 } else { qi / 2 - 1 };
        let op = if is_x { 'X' } else { 'Z' };
        let label1 = format!("d{}{}", q, op);
        let node1 = Node::new(label1.to_string(),
                              col as f64 - 0.25,
                              (self.num_rows - 1 - row) as f64,
                              NodeType::Data,
                              0,
                              0);
        self.nodes.insert(label1.to_string(), node1);
        let label2 = format!("d{}{}", q + 1, op);
        let node2 = Node::new(label2.to_string(),
                              col as f64 + 0.25,
                              (self.num_rows - 1 - row) as f64,
                              NodeType::Data,
                              0,
                              0);
        self.nodes.insert(label2.to_string(), node2);
        let combined_label = format!("d{}/{}{}", q, q + 1, op);
        self.node_grid[col][row] = Some(combined_label.clone());
        self.num_nodes += 2;
    }

    fn add_qubit(&mut self, col: usize, row: usize, node_type: NodeType) -> String {
        let ch = match node_type {
            NodeType::Magic => "m",
            NodeType::Bus => "b",
            NodeType::Data => "d",
            NodeType::Estabilizer => "e",
        };

        let label = format!("{}{}-{}", ch, col, row);
        let node = Node::new(label.to_string(),
                             col as f64,
                             (self.num_rows - 1 - row) as f64,
                             node_type,
                             0,
                             0);
        self.nodes.insert(label.to_string(), node);
        self.num_nodes += 1;
        label
    }

    fn set_edges(&mut self) {
        let mut edges_to_add = Vec::new();
        let mut vertical_data_edges_to_add = Vec::new();

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    // Add horizontal edges
                    if col > 0 {
                        if let Some(ref left_label) = self.node_grid[col - 1][row] {
                            edges_to_add.push((label.clone(), left_label.clone()));
                        }
                    }
                    // Add vertical edges
                    if row > 0 {
                        if let Some(ref up_label) = self.node_grid[col][row - 1] {
                            if !label.starts_with('d') && !up_label.starts_with('d') {
                                edges_to_add.push((label.clone(), up_label.clone()));
                            } else if label.starts_with('d')
                                      && (up_label.starts_with('b') || up_label.starts_with('m'))
                            {
                                vertical_data_edges_to_add.push((label.clone(), up_label.clone()));
                            } else if (label.starts_with('b') || label.starts_with('m'))
                                      && up_label.starts_with('d')
                            {
                                vertical_data_edges_to_add.push((label.clone(), up_label.clone()));
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
                    self.add_edge(d, &label2);
                }
            } else if label2.starts_with('d') {
                if let Some(ref d) = self.get_data_label_side(&label2, false) {
                    self.add_edge(&label1, &d);
                }
            } else {
                self.add_edge(&label1, &label2);
            }
        }
        for (label1, label2) in vertical_data_edges_to_add {
            let (data_label, bus_label) =
                if label1.starts_with('d') { (label1, label2) } else { (label2, label1) };
            let (data_label1, data_label2) = self.get_data_labels(&data_label).unwrap();
            self.get_node_mut(&bus_label).add_edge(&data_label1);
            self.get_node_mut(&bus_label).add_edge(&data_label2);
            self.get_node_mut(&data_label1).add_edge(&bus_label);
            self.get_node_mut(&data_label2).add_edge(&bus_label);
        }
    }

    pub fn get_paired_data_node(&self, node: &Node) -> &Node {
        let qubit = node.get_data_label_number().unwrap();
        let term = node.label.chars().last().unwrap();
        let pair_qubit = if qubit % 2 == 0 { qubit + 1 } else { qubit - 1 };
        let paired_node_label = format!("d{}{}", pair_qubit, term);
        self.get_node(&paired_node_label)
    }

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

    pub fn get_data_labels(&self, label: &str) -> Option<(String, String)> {
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

    fn update_statistics(&mut self) {
        let mut data_count = 0;
        let mut magic_count = 0;
        let mut bus_count = 0;
        let mut estabilizer_count = 0;

        for node in self.nodes.values() {
            match node.node_type {
                NodeType::Data => data_count += 1,
                NodeType::Magic => magic_count += 1,
                NodeType::Bus => bus_count += 1,
                NodeType::Estabilizer => estabilizer_count += 1,
            }
        }

        self.num_data_qubits = data_count / 2;
        self.num_magic_qubits = magic_count;
        self.num_bus_qubits = bus_count;
        self.num_estabilizer_qubits = estabilizer_count;
        self.num_qubits = self.num_data_qubits
                          + self.num_bus_qubits
                          + self.num_magic_qubits
                          + self.num_estabilizer_qubits;

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
        println!("  e-stabilizer: {} ({:.3})",
                 self.num_estabilizer_qubits,
                 self.num_estabilizer_qubits as f64 / total);
        println!("  total:        {}", self.num_qubits);
    }

    pub fn trim_dangling_nodes(&mut self, root_node: &str) {
        let mut num_trimmed = 0;
        loop {
            // Find dangling bus nodes
            let mut dangling_labels: Vec<String> = Vec::new();
            for (label, node) in self.nodes.iter() {
                // there is at most one path going into the bus/magic node
                if self.is_routing_node(node) && node.edges.len() <= 1 && node.label != root_node {
                    dangling_labels.push(label.clone());
                }
            }
            // Remove dangling nodes if any found
            if dangling_labels.is_empty() {
                break;
            } else {
                for label in dangling_labels {
                    self.remove_node(&label);
                    num_trimmed += 1;
                }
            }
        }
        log::info!("    Trimmed {} dangling nodes", num_trimmed);
    }

    pub fn is_routing_node(&self, node: &Node) -> bool {
        if self.use_magic_routing {
            return node.node_type == NodeType::Bus || node.node_type == NodeType::Magic;
        } else {
            return node.node_type == NodeType::Bus;
        }
    }

    pub fn get_node(&self, node_label: &str) -> &Node {
        self.nodes.get(node_label).expect(&format!("Node {} not found", node_label))
    }

    pub fn get_node_mut(&mut self, node_label: &str) -> &mut Node {
        self.nodes.get_mut(node_label).expect(&format!("Node {} not found", node_label))
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.nodes.iter().flat_map(|(node_label, node)| {
                             node.edges
                                 .iter()
                                 .map(move |edge_label| (node_label.as_str(), edge_label.as_str()))
                         })
    }

    pub fn iter_nodes_mut(&mut self) -> impl Iterator<Item = &mut Node> {
        self.nodes.values_mut()
    }

    pub fn contains_node(&self, node_label: &str) -> bool {
        self.nodes.contains_key(node_label)
    }

    pub fn contains_edge(&self, label1: &str, label2: &str) -> bool {
        if let Some(node) = self.nodes.get(label1) { node.edges.contains(label2) } else { false }
    }

    pub fn add_node(&mut self, node: Node) {
        let new_node = Node::new(node.label.to_string(),
                                 node.pos.0,
                                 node.pos.1,
                                 node.node_type,
                                 node.busy_count,
                                 node.cultivation_time);
        self.nodes.insert(node.label.to_string(), new_node);
        self.num_nodes += 1;
    }

    pub fn remove_node(&mut self, node_label: &str) {
        // Get edges to remove from neighbors
        let node = self.get_node(node_label);
        let edges_to_remove: Vec<(String, String)> =
            node.edges.iter().map(|neighbor| (neighbor.clone(), node_label.to_string())).collect();
        // Remove edges from neighbor nodes
        for (nb_label, edge_to_remove) in edges_to_remove {
            if let Some(nb) = self.nodes.get_mut(&nb_label) {
                nb.edges.swap_remove(&edge_to_remove);
                self.num_edges -= 1;
            }
        }
        // Remove the node itself
        if self.nodes.swap_remove(node_label).is_some() {
            self.num_nodes -= 1;
        }
    }

    pub fn add_edge(&mut self, label1: &str, label2: &str) {
        self.get_node_mut(label1).add_edge(label2);
        self.get_node_mut(label2).add_edge(label1);
        self.num_edges += 1;
    }

    pub fn remove_all_edges(&mut self, label: &str) {
        // Get all edges to remove
        let edges_to_remove: Vec<String> = if let Some(node) = self.nodes.get(label) {
            node.edges.iter().cloned().collect()
        } else {
            return;
        };
        // Remove edges from both ends
        for edge in edges_to_remove {
            if let Some(neighbor) = self.nodes.get_mut(&edge) {
                neighbor.edges.swap_remove(label);
            }
            if let Some(node) = self.nodes.get_mut(label) {
                node.edges.swap_remove(&edge);
            }
            self.num_edges -= 1;
        }
    }

    pub fn node_list(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

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

    pub fn plot(&self, fname_added: &str, pauli_product_paths: &[(PauliProduct, TopoGraph)],
                title_str: &str)
                -> Result<(), Box<dyn std::error::Error>> {
        //let _timer = Timer::new("plot");
        let topo_path = Path::new(&self.circuit_fname);
        let topo_stem = topo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("topo");
        let png_fname = format!("{}{}.png", topo_stem, fname_added);
        // Create output file
        let root = BitMapBackend::new(
            &png_fname,
            (self.num_cols as u32 * 100, self.num_rows as u32 * 100),
        )
        .into_drawing_area();

        root.fill(&WHITE)?;
        let mut chart =
            ChartBuilder::on(&root).margin(50).build_cartesian_2d(-1f32..self.num_cols as f32,
                                                                   -1f32..self.num_rows as f32)?;
        // Draw background
        chart.draw_series(std::iter::once(Rectangle::new([(-0.5, -0.5),
                                                          (self.num_cols as f32 - 0.5,
                                                           self.num_rows as f32 - 0.5)],
                                                         RGBColor(230, 230, 230).filled())))?;
        // Draw grid lines
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
        // Generate colors for Pauli product paths
        let num_colors = pauli_product_paths.len().max(1);
        let path_colors: Vec<RGBAColor> =
            (0..num_colors).map(|i| {
                               let hue = (i as f64) / (num_colors as f64);
                               let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
                               RGBColor(r, g, b).to_rgba()
                           })
                           .collect();
        // Draw edges with path coloring
        for node in self.nodes.values() {
            for edge in &node.edges {
                if let Some(other) = self.nodes.get(edge) {
                    let mut edge_color = &BLACK.mix(0.5).to_rgba();
                    let mut stroke_width = 1;
                    // Check if edge is part of any path
                    for (i, (_, path_graph)) in pauli_product_paths.iter().enumerate() {
                        if path_graph.contains_edge(&node.label, edge) {
                            edge_color = &path_colors[i];
                            stroke_width = 6;
                            break;
                        }
                    }
                    chart.draw_series(LineSeries::new(vec![(node.pos.0 as f32,
                                                            node.pos.1 as f32),
                                                           (other.pos.0 as f32,
                                                            other.pos.1 as f32),],
                                                      edge_color.stroke_width(stroke_width)))?;
                }
            }
        }
        // Draw nodes with path highlighting
        for node in self.nodes.values() {
            let (x, y) = node.pos;
            let mut border_color = None;
            // Check if node is part of any path
            for (i, (_, path_graph)) in pauli_product_paths.iter().enumerate() {
                if path_graph.contains_node(&node.label) {
                    border_color = Some(&path_colors[i]);
                    break;
                }
            }
            // Draw node circle
            chart.draw_series(std::iter::once(Circle::new((x as f32, y as f32),
                                                          22,
                                                          match node.node_type {
                                                              NodeType::Magic => {
                                                                  RGBColor(0xFF, 0xBB, 0x99)
                                                              }
                                                              NodeType::Bus => {
                                                                  RGBColor(0xAA, 0xAA, 0xAA)
                                                              }
                                                              NodeType::Data => {
                                                                  RGBColor(0x99, 0x99, 0xFF)
                                                              }
                                                              NodeType::Estabilizer => {
                                                                  RGBColor(0x99, 0xCC, 0x99)
                                                              }
                                                          }.filled())))?;
            // Draw border if part of a path
            if let Some(color) = border_color {
                chart.draw_series(std::iter::once(Circle::new((x as f32, y as f32),
                                                              22,
                                                              color.stroke_width(3))))?;
            }
            // Draw label
            let label_text = if node.is_cultivating() {
                node.busy_count.to_string()
            } else {
                node.label.clone()
            };
            chart.draw_series(std::iter::once(Text::new(label_text,
                                                        (x as f32 - 0.17, y as f32 + 0.09),
                                                        ("sans-serif", 18).into_font())))?;
        }
        // Draw Pauli product labels
        for (i, (pp, path_graph)) in pauli_product_paths.iter().enumerate() {
            if let Some(first_data_node) =
                path_graph.iter_nodes().filter(|n| matches!(n.node_type, NodeType::Data)).next()
            {
                let (x, y) = first_data_node.pos;
                let product_str = pp.get_product_str();
                let text_width = product_str.len() as f32 * 0.15;
                // Draw text background
                chart.draw_series(std::iter::once(Rectangle::new([(x as f32 - 0.3,
                                                                   y as f32 + 0.3),
                                                                  (x as f32 - 0.3
                                                                   + text_width,
                                                                   y as f32 + 0.55)],
                                                                 path_colors[i].mix(0.2)
                                                                               .filled())))?;
                // Draw product string
                chart.draw_series(std::iter::once(Text::new(product_str,
                                                            (x as f32 - 0.2, y as f32 + 0.5),
                                                            ("sans-serif", 22).into_font())))?;
            }
        }
        // Draw title
        if !title_str.is_empty() {
            // manually deal with new lines, since the plotting doesn't
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
        println!("Plotted topology to {}", png_fname);
        Ok(())
    }
}

// Helper function to convert HSV to RGB
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
