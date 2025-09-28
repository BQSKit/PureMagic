use crate::utils::Timer;
use plotters::prelude::*;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeType {
    Magic,
    Bus,
    Data,
    Ancilla,
    Estabilizer,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub label: String,
    pub pos: (f64, f64),
    pub busy_count: Option<i32>,
    pub edges: HashSet<String>,
    pub used: bool,
}

pub struct TopoGraph {
    nodes: HashMap<String, Node>,
    node_grid: Vec<Vec<Option<String>>>,
    num_cols: usize,
    num_rows: usize,
    num_data_qubits: usize,
    num_bus_qubits: usize,
    num_magic_qubits: usize,
    num_ancilla_qubits: usize,
    num_estabilizer_qubits: usize,
    num_qubits: usize,
    rng: StdRng,
    topo_fname: String,
    circuit_fname: String,
}

impl Node {
    fn new(label: String, x: f64, y: f64, node_type: NodeType) -> Self {
        Node {
            node_type,
            label,
            pos: (x, y),
            busy_count: if node_type == NodeType::Magic { Some(0) } else { None },
            edges: HashSet::new(),
            used: false,
        }
    }

    fn add_edge(&mut self, other: &str) {
        self.edges.insert(other.to_string());
    }
}

impl TopoGraph {
    pub fn new() -> Self {
        TopoGraph {
            nodes: HashMap::new(),
            node_grid: Vec::new(),
            num_cols: 0,
            num_rows: 0,
            num_data_qubits: 0,
            num_bus_qubits: 0,
            num_magic_qubits: 0,
            num_ancilla_qubits: 0,
            num_estabilizer_qubits: 0,
            num_qubits: 0,
            rng: StdRng::from_entropy(),
            circuit_fname: String::new(),
            topo_fname: String::new(),
        }
    }

    pub fn set_topo(
        &mut self,
        min_num_qubits: usize,
        circuit_fname: &String,
        topo_fname: &String,
        rng: StdRng,
    ) {
        let _timer = Timer::new("set_topo");
        self.rng = rng;
        self.circuit_fname = circuit_fname.to_string();
        self.topo_fname = topo_fname.to_string();

        if !self.topo_fname.is_empty() {
            if let Err(e) = self.read_topo_from_file() {
                eprintln!("Error reading topology file: {}", e);
            }
        } else {
            let sq_dim = (min_num_qubits as f64).sqrt().floor() as usize;
            let patch_rows = sq_dim / 2 + sq_dim % 2;
            let bus_rows = patch_rows + 1;

            let qubits_per_col = 2 * patch_rows;
            let num_data_cols = ((min_num_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;

            self.num_cols = 2 * num_data_cols + 3;
            self.num_rows = 2 + 2 * patch_rows + bus_rows;

            self.node_grid = vec![vec![None; self.num_rows]; self.num_cols];

            if self.num_cols > 0 && self.num_rows > 0 {
                println!("Layout dimensions: {} {}", self.num_cols, self.num_rows);
                self.gen_topo();
            }
        }
        self.update_statistics();
    }

    pub fn read_topo_from_file(&mut self) -> io::Result<()> {
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
        // Add nodes
        let mut di = 0;
        for col in 0..self.num_cols {
            for row in 0..self.num_rows {
                if let Some(ref node) = self.node_grid[col][row] {
                    if node.starts_with('d') {
                        let op = node.chars().nth(1).unwrap_or('X');
                        self.add_data_qubit(di, col, row, op == 'X');
                        di += 2;
                    } else {
                        let node_type = match node.chars().next() {
                            Some('m') => NodeType::Magic,
                            Some('b') => NodeType::Bus,
                            Some('a') => NodeType::Ancilla,
                            Some('e') => NodeType::Estabilizer,
                            _ => continue,
                        };
                        self.node_grid[col][row] = Some(self.add_node(col, row, node_type));
                    }
                }
            }
        }
        // Add edges
        self.set_edges();
        println!("Read topology with dimensions: {} {}", self.num_cols, self.num_rows);

        Ok(())
    }

    fn gen_topo(&mut self) {
        self.add_border_row(0);
        self.add_border_column(0);

        let mut qi = 0;
        for col in 1..self.num_cols - 1 {
            if col % 2 == 0 {
                // Data column
                for row in 1..self.num_rows - 1 {
                    if row % 3 + 1 == 2 {
                        if row != 1 && row != self.num_rows - 2 && col % 4 == 0 {
                            self.node_grid[col][row] =
                                Some(self.add_node(col, row, NodeType::Estabilizer));
                        } else {
                            self.node_grid[col][row] = Some(self.add_node(col, row, NodeType::Bus));
                        }
                    } else {
                        self.add_data_qubit(qi, col, row, row % 3 + 1 == 3);
                        qi += 2;
                    }
                }
            } else {
                // Bus column
                for row in 1..self.num_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_node(col, row, NodeType::Bus));
                }
            }
        }

        self.add_border_column(self.num_cols - 1);
        self.add_border_row(self.num_rows - 1);
        self.set_edges();
        println!("Generated topology with dimensions: {} {}", self.num_cols, self.num_rows);
    }

    fn add_border_row(&mut self, row: usize) {
        // Add corner bus nodes
        self.node_grid[0][row] = Some(self.add_node(0, row, NodeType::Bus));
        self.node_grid[self.num_cols - 1][row] =
            Some(self.add_node(self.num_cols - 1, row, NodeType::Bus));
        // Add alternating magic/ancilla nodes
        for col in 1..self.num_cols - 1 {
            self.node_grid[col][row] = Some(self.add_node(col, row, NodeType::Magic));
        }
    }

    fn add_border_column(&mut self, col: usize) {
        for row in 1..self.num_rows - 1 {
            self.node_grid[col][row] = Some(self.add_node(col, row, NodeType::Magic));
        }
    }

    fn add_data_qubit(&mut self, qi: usize, col: usize, row: usize, is_x: bool) {
        let q = if is_x { qi / 2 } else { qi / 2 - 1 };
        let op = if is_x { 'X' } else { 'Z' };
        let label1 = format!("d{}{}", q, op);
        let node1 = Node::new(
            label1.to_string(),
            col as f64 - 0.25,
            (self.num_rows - 1 - row) as f64,
            NodeType::Data,
        );
        self.nodes.insert(label1.to_string(), node1);
        let label2 = format!("d{}{}", q + 1, op);
        let node2 = Node::new(
            label2.to_string(),
            col as f64 + 0.25,
            (self.num_rows - 1 - row) as f64,
            NodeType::Data,
        );
        self.nodes.insert(label2.to_string(), node2);
        let combined_label = format!("d{}/{}{}", q, q + 1, op);
        self.node_grid[col][row] = Some(combined_label);
    }

    fn set_edges(&mut self) {
        let mut edges_to_add = Vec::new();

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    // Add horizontal edges
                    if col > 0 {
                        if let Some(ref left_label) = self.node_grid[col - 1][row] {
                            if !label.starts_with('m') || !left_label.starts_with('m') {
                                edges_to_add.push((label.clone(), left_label.clone()));
                            }
                        }
                    }
                    // Add vertical edges
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

    fn update_statistics(&mut self) {
        let mut data_count = 0;
        let mut magic_count = 0;
        let mut bus_count = 0;
        let mut ancilla_count = 0;
        let mut estabilizer_count = 0;

        for node in self.nodes.values() {
            match node.node_type {
                NodeType::Data => data_count += 1,
                NodeType::Magic => magic_count += 1,
                NodeType::Bus => bus_count += 1,
                NodeType::Ancilla => ancilla_count += 1,
                NodeType::Estabilizer => estabilizer_count += 1,
            }
        }

        self.num_data_qubits = data_count / 2;
        self.num_magic_qubits = magic_count;
        self.num_bus_qubits = bus_count;
        self.num_ancilla_qubits = ancilla_count;
        self.num_estabilizer_qubits = estabilizer_count;
        self.num_qubits = self.num_data_qubits
            + self.num_bus_qubits
            + self.num_magic_qubits
            + self.num_ancilla_qubits
            + self.num_estabilizer_qubits;

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
        println!(
            "  ancilla:      {} ({:.3})",
            self.num_ancilla_qubits,
            self.num_ancilla_qubits as f64 / total
        );
        println!(
            "  e-stabilizer: {} ({:.3})",
            self.num_estabilizer_qubits,
            self.num_estabilizer_qubits as f64 / total
        );
        println!("  total:        {}", self.num_qubits);
    }

    pub fn print(&self) -> io::Result<()> {
        let topo_path = Path::new(&self.circuit_fname);
        let topo_stem = topo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("topo");
        let output_fname = format!("{}.topo.txt", topo_stem);
        let mut file = File::create(&output_fname)?;

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    write!(file, "{:8}  ", label)?;
                    /*
                    if is_data_node(label) {
                        write!(
                            file,
                            "{}{} ",
                            label.chars().nth(0).unwrap_or(' '),
                            label.chars().last().unwrap_or(' ')
                        )?;
                    } else {
                        write!(file, "{}  ", label.chars().nth(0).unwrap_or(' '))?;
                    }
                     */
                }
            }
            writeln!(file)?;
        }

        println!("Wrote topology to {}", output_fname);
        Ok(())
    }

    pub fn plot(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _timer = Timer::new("plot");

        let topo_path = Path::new(&self.circuit_fname);
        let topo_stem = topo_path.file_stem().and_then(|s| s.to_str()).unwrap_or("topo");
        let png_fname = format!("{}.topo.png", topo_stem);
        // Create output files
        let root = BitMapBackend::new(
            &png_fname,
            (self.num_cols as u32 * 100, self.num_rows as u32 * 100),
        )
        .into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(-1f32..self.num_cols as f32, -1f32..self.num_rows as f32)?;
        // draw background grey
        chart.draw_series(std::iter::once(Rectangle::new(
            [(-0.5, -0.5), (self.num_cols as f32 - 0.5, self.num_rows as f32 - 0.5)],
            RGBColor(230, 230, 230).filled(),
        )))?;
        // draw white horizontal lines
        for row in 0..=self.num_rows {
            chart.draw_series(LineSeries::new(
                vec![(-0.5, row as f32 - 0.5), (self.num_cols as f32 - 0.5, row as f32 - 0.5)],
                WHITE.stroke_width(3),
            ))?;
        }
        // Draw white vertical lines
        for col in 0..=self.num_cols {
            chart.draw_series(LineSeries::new(
                vec![(col as f32 - 0.5, -0.5), (col as f32 - 0.5, self.num_rows as f32 - 0.5)],
                WHITE.stroke_width(3),
            ))?;
        }
        // Draw edges
        for node in self.nodes.values() {
            for edge in &node.edges {
                if let Some(other) = self.nodes.get(edge) {
                    chart.draw_series(LineSeries::new(
                        vec![
                            (node.pos.0 as f32, node.pos.1 as f32),
                            (other.pos.0 as f32, other.pos.1 as f32),
                        ],
                        &BLACK.mix(0.5),
                    ))?;
                }
            }
        }
        // Draw nodes
        for node in self.nodes.values() {
            let (x, y) = node.pos;
            chart.draw_series(std::iter::once(Circle::new(
                (x as f32, y as f32),
                22,
                match node.node_type {
                    NodeType::Magic => RGBColor(0xFF, 0xBB, 0x99),
                    NodeType::Bus => RGBColor(0xAA, 0xAA, 0xAA),
                    NodeType::Data => RGBColor(0x99, 0x99, 0xFF),
                    NodeType::Ancilla => RGBColor(0xFF, 0x88, 0xAA),
                    NodeType::Estabilizer => RGBColor(0x99, 0xCC, 0x99),
                }
                .filled(),
            )))?;
            chart.draw_series(std::iter::once(Text::new(
                node.label.clone(),
                (x as f32 - 0.17, y as f32 + 0.09),
                ("sans-serif", 18).into_font(),
            )))?;
        }
        root.present()?;
        println!("Plotted topology to {}", png_fname);
        Ok(())
    }

    pub fn trim_dangling_bus_nodes(&mut self) {
        loop {
            // Find dangling bus nodes
            let mut dangling_labels: Vec<String> = Vec::new();
            for (label, node) in self.nodes.iter() {
                // there is at most one path going into the bus node
                if node.node_type == NodeType::Bus && node.edges.len() <= 1 {
                    dangling_labels.push(label.clone());
                }
            }
            // Remove dangling nodes if any found
            if dangling_labels.is_empty() {
                break;
            } else {
                for label in dangling_labels {
                    self.nodes.remove(&label);
                }
            }
        }
    }

    pub fn get_node(&self, node_label: &str) -> &Node {
        self.nodes.get(node_label).expect("Node not found")
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn contains_node(&self, node_label: &str) -> bool {
        self.nodes.contains_key(node_label)
    }

    pub fn add_node(&mut self, col: usize, row: usize, node_type: NodeType) -> String {
        let ch = match node_type {
            NodeType::Magic => "m",
            NodeType::Ancilla => "a",
            NodeType::Bus => "b",
            NodeType::Data => "d",
            NodeType::Estabilizer => "e",
            _ => "",
        };

        let label = format!("{}{}-{}", ch, col, row);
        let node =
            Node::new(label.to_string(), col as f64, (self.num_rows - 1 - row) as f64, node_type);
        self.nodes.insert(label.to_string(), node);
        label
    }

    pub fn add_node_copied(&mut self, node: Node) {
        self.nodes.insert(node.label.to_string(), node);
    }

    pub fn add_edge(&mut self, label1: &str, label2: &str) {
        if let Some(node1) = self.nodes.get_mut(label1) {
            node1.add_edge(label2);
        }
        if let Some(node2) = self.nodes.get_mut(label2) {
            node2.add_edge(label1);
        }
    }
}
