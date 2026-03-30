use crate::node::{Node, NodeType};
use crate::pauliproduct::PauliProduct;
use crate::topograph_plotter::{DataGroup, TopoGraphPlotter};
use crate::treegraph::TreeGraph;
use indexmap::IndexMap;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Topological layout of a surface code quantum processor.
///
/// Nodes are either Data (logical qubit patches), Magic (magic state cultivators),
/// or Bus (routing ancilla in bus-routing mode). Each data qubit has two nodes:
/// one for the X stabiliser patch and one for the Z stabiliser patch.
pub(crate) struct TopoGraph {
    pub(crate) nodes: Vec<Node>,
    pub labels: Vec<String>,
    pub(crate) label_to_id: IndexMap<String, u16>,
    pub(crate) data_ids: Vec<[u16; 2]>,
    pub(crate) node_grid: Vec<Vec<Option<String>>>,
    pub(crate) n_cols: usize,
    pub(crate) n_rows: usize,
    pub(crate) topo_fname: String,
    pub(crate) circuit_fname: String,
    pub use_magic_routing: bool,
    pub n_data_qubits: usize,
    pub n_bus_qubits: usize,
    pub n_magic_qubits: usize,
    pub n_qubits: usize,
    pub n_edges: usize,
    pub n_nodes: usize,
    pub busy_counts: Vec<i32>,
    pub cultivation_times: Vec<i32>,
    pub sides_only: bool,
    pub(crate) data_groups: Vec<DataGroup>,
    pub(crate) node_to_group: HashMap<u16, usize>,
    pub(crate) data_pos_map: HashMap<(i32, i32), u16>,
}

impl TopoGraph {
    pub(crate) fn new() -> Self {
        TopoGraph {
            nodes: Vec::new(),
            labels: Vec::new(),
            label_to_id: IndexMap::new(),
            data_ids: Vec::new(),
            node_grid: Vec::new(),
            n_cols: 0,
            n_rows: 0,
            n_data_qubits: 0,
            n_bus_qubits: 0,
            n_magic_qubits: 0,
            n_qubits: 0,
            n_edges: 0,
            n_nodes: 0,
            circuit_fname: String::new(),
            topo_fname: String::new(),
            use_magic_routing: true,
            busy_counts: Vec::new(),
            cultivation_times: Vec::new(),
            sides_only: false,
            data_groups: Vec::new(),
            node_to_group: HashMap::new(),
            data_pos_map: HashMap::new(),
        }
    }

    pub(crate) fn label(&self, id: u16) -> &str {
        &self.labels[id as usize]
    }

    pub(crate) fn set_topo(
        &mut self, min_n_qubits: usize, circuit_fname: &String, topo_fname: &String, rseed: &u32,
        use_magic_routing: bool, ancilla_rows: usize, sides_only: bool,
    ) {
        self.circuit_fname = circuit_fname.to_string();
        self.topo_fname = topo_fname.to_string();
        self.use_magic_routing = use_magic_routing;
        self.sides_only = sides_only;
        Node::set_magic_routing(use_magic_routing);

        let has_topo_file = !self.topo_fname.is_empty();
        if has_topo_file {
            if let Err(e) = self.read_topo_from_file(rseed, sides_only) {
                eprintln!("Error reading topology file: {}", e);
            }
        } else if !use_magic_routing {
            if ancilla_rows == 0 {
                self.gen_compact_bus_routing_topo(min_n_qubits, sides_only);
            } else {
                self.gen_bus_routing_topo(min_n_qubits, sides_only);
            }
        } else {
            self.gen_pure_magic_topo(min_n_qubits, ancilla_rows, sides_only);
        }
        // Link each data node to its X/Z partner (same qubit, opposite basis).
        // Data nodes are generated in pairs: even qubit index = X, odd = Z.
        // The paired node has the adjacent qubit number and the same basis suffix.
        let node_ids: Vec<u16> = self.nodes.iter().map(|node| node.id).collect();
        for node_id in node_ids {
            let node = self.node(node_id);
            if node.node_type == NodeType::Data {
                let label = self.label(node_id);
                let qubit = label
                    .chars()
                    .skip(1)
                    .take_while(|c| c.is_numeric())
                    .collect::<String>()
                    .parse::<usize>()
                    .ok()
                    .unwrap();
                let term = label.chars().last().map(|c| c.to_string()).unwrap();
                // Even qubit pairs with qubit+1; odd qubit pairs with qubit-1.
                let pair_qubit = if qubit % 2 == 0 { qubit + 1 } else { qubit - 1 };
                let paired_node_label = format!("d{}{}", pair_qubit, term);
                self.node_mut(node_id).paired_data_id =
                    self.label_to_id.get(&paired_node_label).copied();
            }
        }
        self.data_ids.clear();
        for node in &self.nodes {
            if node.node_type == NodeType::Data {
                let label = &self.labels[node.id as usize];
                let basis: char = label.chars().last().unwrap();
                let qubit: usize = label[1..label.len() - 1].parse().unwrap();
                let basis_idx: usize = if basis == 'X' { 0 } else { 1 };
                if qubit >= self.data_ids.len() {
                    self.data_ids.resize(qubit + 1, [u16::MAX; 2]);
                }
                self.data_ids[qubit][basis_idx] = node.id;
            }
        }
        self.update_statistics();
        self.print_statistics();
        self.build_plot_cache();
    }

    pub(crate) fn build_plot_cache(&mut self) {
        self.data_pos_map = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::Data)
            .map(|n| {
                let (px, py) = n.pos;
                (((px * 10.0).round() as i32, (py * 10.0).round() as i32), n.id)
            })
            .collect();

        self.data_groups = self.build_data_groups();
        self.node_to_group.clear();
        for (gi, group) in self.data_groups.iter().enumerate() {
            self.node_to_group.insert(group.x_left_id, gi);
            self.node_to_group.insert(group.x_right_id, gi);
            self.node_to_group.insert(group.z_left_id, gi);
            self.node_to_group.insert(group.z_right_id, gi);
        }
    }

    fn build_data_groups(&self) -> Vec<DataGroup> {
        let data_pos_map = &self.data_pos_map;
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

    pub(crate) fn update_statistics(&mut self) {
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
        // Each logical qubit has two data nodes (X patch + Z patch).
        self.n_data_qubits = data_count / 2;
        self.n_magic_qubits = magic_count;
        self.n_bus_qubits = bus_count;
        self.n_qubits = self.n_data_qubits + self.n_bus_qubits + self.n_magic_qubits;
    }

    fn print_statistics(&mut self) {
        let total = self.n_qubits as f64;
        println!("Number of qubits:");
        println!(
            "  data:         {} ({:.3})",
            self.n_data_qubits,
            self.n_data_qubits as f64 / total
        );
        println!(
            "  bus:          {} ({:.3})",
            self.n_bus_qubits,
            self.n_bus_qubits as f64 / total
        );
        println!(
            "  magic:        {} ({:.3})",
            self.n_magic_qubits,
            self.n_magic_qubits as f64 / total
        );
        println!("  total:        {}", self.n_qubits);
    }

    pub(crate) fn node(&self, id: u16) -> &Node {
        &self.nodes[id as usize]
    }

    pub(crate) fn node_mut(&mut self, id: u16) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    pub(crate) fn iter_nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    pub(crate) fn add_edge(&mut self, node_id1: u16, node_id2: u16) {
        self.node_mut(node_id1).add_nb(node_id2);
        self.node_mut(node_id2).add_nb(node_id1);
        self.n_edges += 1;
    }

    pub(crate) fn get_data_node_id(&self, qubit: u16, basis: char) -> u16 {
        let basis_idx: usize = if basis == 'X' { 0 } else { 1 };
        self.data_ids[qubit as usize][basis_idx]
    }

    /// Returns true if this magic node is actively cultivating (started but not yet ready).
    /// A node is ready when `cultivation_time == 0`; it is cultivating when
    /// `busy_count < cultivation_time` (i.e. it has been assigned a time but not finished).
    pub(crate) fn is_cultivating(&self, node_id: u16) -> bool {
        self.cultivation_times[node_id as usize] > 0
            && self.busy_counts[node_id as usize] < self.cultivation_times[node_id as usize]
    }

    pub(crate) fn circuit_stem(&self) -> &str {
        Path::new(&self.circuit_fname).file_stem().and_then(|s| s.to_str()).unwrap_or("topo")
    }

    pub(crate) fn data_groups(&self) -> &[DataGroup] {
        &self.data_groups
    }

    pub(crate) fn node_to_group_map(&self) -> &HashMap<u16, usize> {
        &self.node_to_group
    }

    pub(crate) fn print(&self) -> io::Result<()> {
        let topo_stem = self.circuit_stem();
        let output_fname = format!("{}.topo.txt", topo_stem);
        let mut f = File::create(&output_fname)?;

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                if let Some(ref label) = self.node_grid[col][row] {
                    if label.starts_with('d') {
                        write!(
                            f,
                            "{}{} ",
                            label.chars().nth(0).unwrap_or(' '),
                            label.chars().last().unwrap_or(' ')
                        )?;
                    } else {
                        write!(f, "{}  ", label.chars().nth(0).unwrap_or(' '))?;
                    }
                }
            }
            writeln!(f)?;
        }

        println!("Wrote topology to {}", output_fname);
        Ok(())
    }

    pub(crate) fn plot(
        &self, fname_added: &str, pp_paths: &[(PauliProduct, Rc<TreeGraph>, u32)],
        title_str: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        TopoGraphPlotter::new(self).plot(fname_added, pp_paths, title_str)
    }

    fn read_topo_from_file(&mut self, rseed: &u32, sides_only: bool) -> io::Result<()> {
        use crate::fn_timer;
        let _timer = fn_timer!();
        let mut rows = Vec::new();
        let f = File::open(&self.topo_fname)?;
        for line in io::BufReader::new(f).lines() {
            let line = line?;
            let row: Vec<String> = line.split_whitespace().map(|s| s.to_string()).collect();
            if !row.is_empty() {
                rows.push(row);
            }
        }
        self.n_rows = rows.len();
        self.n_cols = rows[0].len();
        self.node_grid = vec![vec![None; self.n_rows]; self.n_cols];

        for (row_i, row) in rows.iter().enumerate() {
            for (col_i, col) in row.iter().enumerate() {
                self.node_grid[col_i][row_i] = Some(col.clone());
                if self.use_magic_routing && col == "b" {
                    self.node_grid[col_i][row_i] = Some("m".to_string());
                }
            }
        }
        let mut pair_indices = Vec::new();
        let mut n_data_nodes = 0;
        for col in 0..self.n_cols {
            for row in 0..self.n_rows {
                if let Some(ref node) = self.node_grid[col][row].clone() {
                    if node.starts_with('d') && node.ends_with('X') {
                        pair_indices.push(n_data_nodes);
                        n_data_nodes += 4;
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
        for col in 0..self.n_cols {
            for row in 0..self.n_rows {
                let node_opt = self.node_grid[col][row].clone();
                if let Some(ref node) = node_opt {
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
        println!("Read topology with dimensions: {} {}", self.n_cols, self.n_rows);
        Ok(())
    }

    /// Returns the node type used for routing nodes in the current routing mode.
    #[inline]
    fn routing_node_type(&self) -> NodeType {
        if self.use_magic_routing { NodeType::Magic } else { NodeType::Bus }
    }

    fn gen_bus_routing_topo(&mut self, min_n_qubits: usize, sides_only: bool) {
        let sq_dim = (min_n_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let bus_rows = patch_rows + 1;
        let qubits_per_col = 2 * patch_rows;
        let n_data_cols = ((min_n_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;
        self.n_cols = 2 * n_data_cols + 3;
        self.n_rows = 2 + 2 * patch_rows + bus_rows;
        self.node_grid = vec![vec![None; self.n_rows]; self.n_cols];

        self.add_border_row(0);
        self.add_border_column(0);

        let max_qi =
            if min_n_qubits % 2 == 0 { 2 * min_n_qubits } else { 2 * min_n_qubits + 1 };
        let mut qi = 0;
        for col in 1..self.n_cols - 1 {
            if col % 2 == 0 {
                for row in 1..self.n_rows - 1 {
                    if row % 3 + 1 == 2 {
                        let node_type = self.routing_node_type();
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                    } else if qi < max_qi {
                        self.add_double_data_qubit(qi, col, row, row % 3 + 1 == 3);
                        qi += 2;
                    } else {
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
                    }
                }
            } else {
                let node_type = self.routing_node_type();
                for row in 1..self.n_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                }
            }
        }
        self.add_border_column(self.n_cols - 1);
        self.add_border_row(self.n_rows - 1);
        self.set_edges(sides_only);
        println!("Generated topology with dimensions: {} {}", self.n_cols, self.n_rows);
    }

    fn gen_compact_bus_routing_topo(&mut self, min_n_qubits: usize, sides_only: bool) {
        let sq_dim = (min_n_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let qubits_per_col = 2 * patch_rows;
        let n_data_cols = ((min_n_qubits as f64) / (qubits_per_col as f64)).ceil() as usize;
        self.n_cols = 2 * n_data_cols + 1;
        self.n_rows = 3 + 2 * patch_rows;
        self.node_grid = vec![vec![None; self.n_rows]; self.n_cols];

        self.add_border_row_compact(0);
        let max_qi =
            if min_n_qubits % 2 == 0 { 2 * min_n_qubits } else { 2 * min_n_qubits + 1 };
        let mut qi = 0;
        for col in 0..self.n_cols {
            if col % 2 == 1 {
                for row in 1..self.n_rows - 1 {
                    if qi < max_qi && row < self.n_rows - 2 {
                        self.add_double_data_qubit(qi, col, row, row % 2 == 1);
                        qi += 2;
                    } else {
                        self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
                    }
                }
                let row = self.n_rows - 1;
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
            } else {
                let node_type = self.routing_node_type();
                for row in 1..self.n_rows - 1 {
                    self.node_grid[col][row] = Some(self.add_qubit(col, row, node_type));
                }
            }
        }
        self.add_border_row_compact(self.n_rows - 1);
        self.set_edges(sides_only);
        println!("Generated topology with dimensions: {} {}", self.n_cols, self.n_rows);
    }

    pub(crate) fn gen_pure_magic_topo(
        &mut self, min_n_qubits: usize, ancilla_rows: usize, sides_only: bool,
    ) {
        let row_spacing = ancilla_rows + 1;
        let col_spacing = if ancilla_rows == 0 { 2 } else { ancilla_rows + 1 };
        let sq_dim = (min_n_qubits as f64).sqrt().floor() as usize;
        let patch_rows = sq_dim / 2 + sq_dim % 2;
        let patch_cols = ((min_n_qubits as f64) / ((2 * patch_rows) as f64)).ceil() as usize;
        self.n_rows = patch_rows * (1 + row_spacing) + row_spacing - 1;
        if ancilla_rows == 0 {
            self.n_rows += 1;
        }
        self.n_cols = patch_cols * col_spacing + col_spacing - 1;
        self.node_grid = vec![vec![None; self.n_rows]; self.n_cols];
        let mut qi = 0;
        let max_qi =
            if min_n_qubits % 2 == 0 { 2 * min_n_qubits } else { 2 * min_n_qubits + 1 };
        let row_gap = 1 + row_spacing;
        for col in 0..self.n_cols {
            for row in 0..self.n_rows {
                if col % col_spacing == col_spacing - 1 {
                    if (row % row_gap == row_spacing || row % row_gap == row_spacing - 1)
                        && !(ancilla_rows == 0 && row == self.n_rows - 1)
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
        println!("Generated topology with dimensions: {} {}", self.n_cols, self.n_rows);
    }

    /// Adds a pair of data nodes (left and right) at grid position (col, row).
    /// `qi` is the base qubit index (even); the two nodes get labels `d{q}X/Z` and `d{q+1}X/Z`.
    /// Positions are offset by ±0.25 so the two nodes sit side-by-side within the column.
    fn add_double_data_qubit(&mut self, qi: usize, col: usize, row: usize, is_x: bool) {
        let q = if is_x { qi / 2 } else { qi / 2 - 1 };
        let op = if is_x { 'X' } else { 'Z' };
        let fname1 = format!("d{}{}", q, op);
        let id1 = self.n_nodes as u16;
        let node1 = Node::new(
            id1,
            None,
            col as f32 - 0.25,
            (self.n_rows - 1 - row) as f32,
            NodeType::Data,
        );
        self.nodes.push(node1);
        self.labels.push(fname1.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.label_to_id.insert(fname1, id1);
        self.n_nodes += 1;
        let id2 = self.n_nodes as u16;
        let fname2 = format!("d{}{}", q + 1, op);
        let node2 = Node::new(
            id2,
            None,
            col as f32 + 0.25,
            (self.n_rows - 1 - row) as f32,
            NodeType::Data,
        );
        self.nodes.push(node2);
        self.labels.push(fname2.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.label_to_id.insert(fname2, id2);
        let combined_label = format!("d{}/{}{}", q, q + 1, op);
        self.node_grid[col][row] = Some(combined_label.clone());
        self.n_nodes += 1;
    }

    fn add_qubit(&mut self, col: usize, row: usize, node_type: NodeType) -> String {
        let ch = match node_type {
            NodeType::Magic => "m",
            NodeType::Bus => "b",
            NodeType::Data => "d",
        };

        let label = format!("{}{}-{}", ch, col, row);
        let node = Node::new(
            self.n_nodes as u16,
            None,
            col as f32,
            (self.n_rows - 1 - row) as f32,
            node_type,
        );
        self.nodes.push(node);
        self.labels.push(label.clone());
        self.busy_counts.push(0);
        self.cultivation_times.push(0);
        self.label_to_id.insert(label.clone(), self.n_nodes as u16);
        self.n_nodes += 1;
        label
    }

    fn add_border_row(&mut self, row: usize) {
        let node_type = self.routing_node_type();
        self.node_grid[0][row] = Some(self.add_qubit(0, row, node_type));
        let last_col = self.n_cols - 1;
        self.node_grid[last_col][row] = Some(self.add_qubit(last_col, row, node_type));
        for col in 1..self.n_cols - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    fn add_border_row_compact(&mut self, row: usize) {
        for col in 0..self.n_cols {
            if col % 2 == 0 {
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
            } else {
                self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Bus));
            }
        }
    }

    fn add_border_column(&mut self, col: usize) {
        for row in 1..self.n_rows - 1 {
            self.node_grid[col][row] = Some(self.add_qubit(col, row, NodeType::Magic));
        }
    }

    /// Establishes edges between adjacent nodes (4-connectivity).
    ///
    /// Horizontal edges connect every node to its left nb.
    /// Vertical edges connect routing nodes to each other (not to data nodes directly).
    /// When `sides_only` is false, additional vertical edges connect Z data nodes to
    /// the routing node two rows above, and X data nodes to the routing node two rows
    /// below — these are the "top/bottom" connections used for Y-basis operators.
    fn set_edges(&mut self, sides_only: bool) {
        let mut edges_to_add = Vec::new();
        let mut vert_data_edges_to_add = Vec::new();

        for row in 0..self.n_rows {
            for col in 0..self.n_cols {
                if let Some(ref label) = self.node_grid[col][row].clone() {
                    if col > 0 {
                        if let Some(ref left_label) = self.node_grid[col - 1][row].clone() {
                            edges_to_add.push((label.clone(), left_label.clone()));
                        }
                    }
                    if !sides_only {
                        // Z data node: connect upward to the routing node 2 rows above.
                        if row > 1 {
                            if label.starts_with('d') && label.ends_with('Z') {
                                if let Some(ref up_label) = self.node_grid[col][row - 2].clone() {
                                    if up_label.starts_with('b') || up_label.starts_with('m') {
                                        vert_data_edges_to_add
                                            .push((label.clone(), up_label.clone()));
                                    }
                                }
                            }
                        }
                        // X data node: connect downward to the routing node 2 rows below.
                        if row < self.n_rows - 2 {
                            if label.starts_with('d') && label.ends_with('X') {
                                if let Some(ref up_label) = self.node_grid[col][row + 2].clone() {
                                    if up_label.starts_with('b') || up_label.starts_with('m') {
                                        vert_data_edges_to_add
                                            .push((label.clone(), up_label.clone()));
                                    }
                                }
                            }
                        }
                    }
                    if row > 0 {
                        if let Some(ref up_label) = self.node_grid[col][row - 1].clone() {
                            if !label.starts_with('d') && !up_label.starts_with('d') {
                                edges_to_add.push((label.clone(), up_label.clone()));
                            }
                        }
                    }
                }
            }
        }
        // For horizontal edges involving a double-data-qubit label (e.g. "d0/1X"),
        // connect only the left or right individual data node to the routing nb.
        for (label1, label2) in edges_to_add {
            if label1.starts_with('d') {
                if let Some(d) = Self::data_label_side(&label1, true) {
                    let n1 = *self.label_to_id.get(&d).unwrap();
                    let n2 = *self.label_to_id.get(&label2).unwrap();
                    self.add_edge(n1, n2);
                }
            } else if label2.starts_with('d') {
                if let Some(d) = Self::data_label_side(&label2, false) {
                    let n1 = *self.label_to_id.get(&d).unwrap();
                    let n2 = *self.label_to_id.get(&label1).unwrap();
                    self.add_edge(n2, n1);
                }
            } else {
                let n1 = *self.label_to_id.get(&label1).unwrap();
                let n2 = *self.label_to_id.get(&label2).unwrap();
                self.add_edge(n1, n2);
            }
        }
        // Vertical data edges connect both individual data nodes in a pair to the
        // routing node above/below (add_nb is used directly to avoid double-counting
        // n_edges, since these are not standard bidirectional topology edges).
        for (label1, label2) in vert_data_edges_to_add {
            let (data_label, bus_label) =
                if label1.starts_with('d') { (label1, label2) } else { (label2, label1) };
            let (data_label1, data_label2) = Self::data_labels(&data_label).unwrap();
            let data_node_id1 = *self.label_to_id.get(&data_label1).unwrap();
            let data_node_id2 = *self.label_to_id.get(&data_label2).unwrap();
            let bus_node_id = *self.label_to_id.get(&bus_label).unwrap();
            self.node_mut(bus_node_id).add_nb(data_node_id1);
            self.node_mut(bus_node_id).add_nb(data_node_id2);
            self.node_mut(data_node_id1).add_nb(bus_node_id);
            self.node_mut(data_node_id2).add_nb(bus_node_id);
        }
    }

    /// Parses a combined double-data-qubit label like `"d0/1X"` into its parts.
    fn parse_data_label_parts(label: &str) -> Option<(&str, &str, &str)> {
        let d_pos = label.find('d')?;
        let slash_pos = label.find('/')?;
        let op_pos = label.find(|c: char| c == 'X' || c == 'Z')?;
        let first_num = &label[d_pos + 1..slash_pos];
        let second_num = &label[slash_pos + 1..op_pos];
        let operator = &label[op_pos..=op_pos];
        Some((first_num, second_num, operator))
    }

    /// Extracts the left (`left=true`) or right (`left=false`) individual data node label.
    fn data_label_side(label: &str, left: bool) -> Option<String> {
        let (first_num, second_num, operator) = Self::parse_data_label_parts(label)?;
        if left {
            Some(format!("d{}{}", first_num, operator))
        } else {
            Some(format!("d{}{}", second_num, operator))
        }
    }

    /// Extracts both individual data node labels from a double data qubit label.
    fn data_labels(label: &str) -> Option<(String, String)> {
        let (first_num, second_num, operator) = Self::parse_data_label_parts(label)?;
        Some((format!("d{}{}", first_num, operator), format!("d{}{}", second_num, operator)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeType;

    #[test]
    fn new_creates_empty_topology() {
        let topo = TopoGraph::new();
        assert_eq!(topo.n_nodes, 0);
        assert_eq!(topo.n_edges, 0);
        assert_eq!(topo.n_data_qubits, 0);
        assert_eq!(topo.n_magic_qubits, 0);
        assert_eq!(topo.n_bus_qubits, 0);
    }

    #[test]
    fn gen_pure_magic_topo_has_enough_data_qubits() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        assert!(
            topo.n_data_qubits >= 4,
            "expected >= 4 data qubits, got {}",
            topo.n_data_qubits
        );
    }

    #[test]
    fn gen_pure_magic_topo_has_only_magic_and_data_nodes() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        assert_eq!(topo.n_bus_qubits, 0, "pure magic topo should have no bus qubits");
        assert!(topo.n_magic_qubits > 0, "pure magic topo should have magic qubits");
    }

    #[test]
    fn gen_pure_magic_topo_total_qubits_consistent() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        assert_eq!(
            topo.n_qubits,
            topo.n_data_qubits + topo.n_bus_qubits + topo.n_magic_qubits
        );
    }

    #[test]
    fn compact_bus_topo_has_bus_qubits() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, false, 0, false);
        assert!(topo.n_bus_qubits > 0);
        assert!(!topo.use_magic_routing);
        crate::node::Node::set_magic_routing(true);
    }

    #[test]
    fn compact_bus_topo_has_enough_data_qubits() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, false, 0, false);
        assert!(topo.n_data_qubits >= 4);
        crate::node::Node::set_magic_routing(true);
    }

    #[test]
    fn bus_routing_topo_has_bus_qubits() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, false, 1, false);
        assert!(topo.n_bus_qubits > 0);
        crate::node::Node::set_magic_routing(true);
    }

    #[test]
    fn add_edge_creates_bidirectional_connection() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let magic_ids: Vec<u16> =
            topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).map(|n| n.id).collect();
        let has_nbs = magic_ids.iter().any(|&id| !topo.node(id).nbs_slice().is_empty());
        assert!(has_nbs);
    }

    #[test]
    fn edges_are_symmetric() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        for node in topo.iter_nodes() {
            for &nb_id in node.nbs_slice() {
                let nb = topo.node(nb_id);
                assert!(
                    nb.nbs_slice().contains(&node.id),
                    "edge {}->{} is not symmetric",
                    node.id,
                    nb_id
                );
            }
        }
    }

    #[test]
    fn get_data_node_id_returns_valid_node() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let x_id = topo.get_data_node_id(0, 'X');
        let z_id = topo.get_data_node_id(0, 'Z');
        assert_ne!(x_id, u16::MAX);
        assert_ne!(z_id, u16::MAX);
        assert_ne!(x_id, z_id);
        assert_eq!(topo.node(x_id).node_type, NodeType::Data);
        assert_eq!(topo.node(z_id).node_type, NodeType::Data);
    }

    #[test]
    fn is_cultivating_false_when_cultivation_time_zero() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let magic_id =
            topo.iter_nodes().find(|n| n.node_type == NodeType::Magic).map(|n| n.id).unwrap();
        topo.cultivation_times[magic_id as usize] = 0;
        topo.busy_counts[magic_id as usize] = 0;
        assert!(!topo.is_cultivating(magic_id));
    }

    #[test]
    fn is_cultivating_true_when_busy_less_than_cultivation_time() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let magic_id =
            topo.iter_nodes().find(|n| n.node_type == NodeType::Magic).map(|n| n.id).unwrap();
        topo.cultivation_times[magic_id as usize] = 5;
        topo.busy_counts[magic_id as usize] = 2;
        assert!(topo.is_cultivating(magic_id));
    }

    #[test]
    fn is_cultivating_false_when_busy_equals_cultivation_time() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let magic_id =
            topo.iter_nodes().find(|n| n.node_type == NodeType::Magic).map(|n| n.id).unwrap();
        topo.cultivation_times[magic_id as usize] = 3;
        topo.busy_counts[magic_id as usize] = 3;
        assert!(!topo.is_cultivating(magic_id));
    }

    #[test]
    fn update_statistics_keeps_counts_consistent() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        let before_total = topo.n_qubits;
        topo.update_statistics();
        assert_eq!(topo.n_qubits, before_total, "update_statistics should be idempotent");
        assert_eq!(
            topo.n_qubits,
            topo.n_data_qubits + topo.n_bus_qubits + topo.n_magic_qubits
        );
    }

    #[test]
    fn get_label_returns_non_empty_string() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        for node in topo.iter_nodes() {
            let label = topo.label(node.id);
            assert!(!label.is_empty());
        }
    }

    #[test]
    fn data_node_labels_contain_basis() {
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"dummy".to_string(), &"".to_string(), &0, true, 1, false);
        for node in topo.iter_nodes() {
            if node.node_type == NodeType::Data {
                let label = topo.label(node.id);
                assert!(label.ends_with('X') || label.ends_with('Z'));
            }
        }
    }

    #[test]
    fn hsv_to_rgb_red() {
        let (r, g, b) = crate::topograph_plotter::hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn hsv_to_rgb_black() {
        let (r, g, b) = crate::topograph_plotter::hsv_to_rgb(0.0, 0.0, 0.0);
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn hsv_to_rgb_white() {
        let (r, g, b) = crate::topograph_plotter::hsv_to_rgb(0.0, 0.0, 1.0);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
    }

    #[test]
    fn set_topo_generates_topology_for_small_circuit() {
        Node::set_magic_routing(false);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"test".to_string(), &"".to_string(), &0, false, 0, false);
        assert!(topo.n_nodes > 0);
        assert!(topo.n_qubits > 0);
        Node::set_magic_routing(true);
    }

    #[test]
    fn set_topo_with_magic_routing_has_magic_nodes() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"test".to_string(), &"".to_string(), &0, true, 1, false);
        let magic_count = topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).count();
        assert!(magic_count > 0);
    }

    #[test]
    fn set_topo_without_magic_routing_has_bus_nodes() {
        Node::set_magic_routing(false);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"test".to_string(), &"".to_string(), &0, false, 0, false);
        let bus_count = topo.iter_nodes().filter(|n| n.node_type == NodeType::Bus).count();
        assert!(bus_count > 0);
        Node::set_magic_routing(true);
    }

    #[test]
    fn bus_routing_topo_has_data_qubits() {
        Node::set_magic_routing(false);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"test".to_string(), &"".to_string(), &0, false, 0, false);
        let data_count = topo.iter_nodes().filter(|n| n.node_type == NodeType::Data).count();
        assert!(data_count >= 4);
        Node::set_magic_routing(true);
    }

    #[test]
    fn update_statistics_data_count_matches_data_nodes() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"test".to_string(), &"".to_string(), &0, true, 1, false);
        topo.update_statistics();
        let actual_data_nodes = topo.iter_nodes().filter(|n| n.node_type == NodeType::Data).count();
        assert_eq!(topo.n_data_qubits * 2, actual_data_nodes);
    }

    #[test]
    fn update_statistics_magic_count_matches_magic_nodes() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"test".to_string(), &"".to_string(), &0, true, 1, false);
        topo.update_statistics();
        let actual_magic = topo.iter_nodes().filter(|n| n.node_type == NodeType::Magic).count();
        assert_eq!(topo.n_magic_qubits, actual_magic);
    }

    #[test]
    fn get_node_returns_correct_node() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"test".to_string(), &"".to_string(), &0, true, 1, false);
        for node in topo.iter_nodes() {
            let fetched = topo.node(node.id);
            assert_eq!(fetched.id, node.id);
        }
    }

    #[test]
    fn all_nodes_have_finite_positions() {
        Node::set_magic_routing(true);
        let mut topo = TopoGraph::new();
        topo.set_topo(4, &"test".to_string(), &"".to_string(), &0, true, 1, false);
        for node in topo.iter_nodes() {
            assert!(node.pos.0.is_finite());
            assert!(node.pos.1.is_finite());
        }
    }

    #[test]
    fn circuit_stem_returns_non_empty_string() {
        let mut topo = TopoGraph::new();
        topo.set_topo(2, &"mytest.trans".to_string(), &"".to_string(), &0, false, 0, false);
        let stem = topo.circuit_stem();
        assert!(!stem.is_empty());
    }
}
