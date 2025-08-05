extern crate env_logger;
extern crate log;

use log::{debug, warn};
use num::integer::gcd;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::time::Instant;

#[derive(Clone, Debug)]
struct PauliTerm {
    basis: char,
    phase: i32,
    qubit: i32,
}

#[derive(Clone, Debug)]
struct PauliProduct {
    terms: Vec<PauliTerm>,
    angle_numerator: i32,
    angle_denominator: i32,
    is_clifford: bool,
}

struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    fn new(name: &str) -> Self {
        println!("Starting {}", name);
        Timer {
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!("{} took {:?}", self.name, self.start.elapsed());
    }
}

impl PauliTerm {
    fn new(basis: char, phase: i32, qubit: i32) -> Self {
        PauliTerm {
            basis,
            phase,
            qubit,
        }
    }

    fn load_from_string(&mut self, s: &str) -> io::Result<()> {
        if !s.starts_with("Pauli") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid Pauli term format",
            ));
        }

        self.basis = s
            .chars()
            .nth(5)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing basis"))?;

        if !matches!(self.basis, 'X' | 'Y' | 'Z' | 'I') {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid basis"));
        }

        let phase_str = &s[7..9];
        self.phase = match phase_str {
            "+1" => 0,
            "+i" => 1,
            "-1" => 2,
            "-i" => 3,
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid phase")),
        };

        self.qubit = s[10..]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid qubit number"))?;

        Ok(())
    }
}

impl std::fmt::Display for PauliTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let phase = match self.phase {
            0 => "+1",
            1 => "+i",
            2 => "-1",
            3 => "-i",
            _ => unreachable!(),
        };
        write!(f, "Pauli{}({}){}", self.basis, phase, self.qubit)
    }
}

fn add_phase(phase1: i32, phase2: i32) -> i32 {
    (phase1 + phase2) % 4
}

impl PauliProduct {
    fn new() -> Self {
        PauliProduct {
            terms: Vec::new(),
            angle_numerator: 1,
            angle_denominator: 1,
            is_clifford: false,
        }
    }

    fn load_from_string(&mut self, s: &str) -> io::Result<()> {
        let parts: Vec<&str> = s.split('<').collect();
        if parts.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid format"));
        }

        // Parse terms
        for term_str in parts[0].split('.') {
            if !term_str.is_empty() {
                let mut term = PauliTerm::new('I', 0, 0);
                term.load_from_string(term_str)?;
                self.terms.push(term);
            }
        }

        // Parse angle
        if let Some(angle_str) = parts[1]
            .strip_prefix("Angle(")
            .and_then(|s| s.strip_suffix(")>"))
        {
            if angle_str.contains('/') {
                let nums: Vec<&str> = angle_str.split('/').collect();
                if nums[0].len() > 2 {
                    self.angle_numerator =
                        nums[0].trim_end_matches("pi").parse().map_err(|_| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Invalid numerator: ".to_owned() + angle_str,
                            )
                        })?;
                } else {
                    self.angle_numerator = 1;
                }
                self.angle_denominator = nums[1].parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid denominator")
                })?;
            } else {
                self.angle_numerator = angle_str
                    .trim_end_matches("pi")
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid angle"))?;
                self.angle_denominator = 1;
            }
        }
        let gcd_factor = gcd(self.angle_numerator, self.angle_denominator);
        let numerator = self.angle_numerator as f64 / gcd_factor as f64;
        let denominator = self.angle_denominator as f64 / gcd_factor as f64;
        self.angle_numerator = numerator.floor() as i32;
        self.angle_denominator = denominator.floor() as i32;
        self.is_clifford = matches!(self.angle_denominator, 1 | 2 | 4 | -1 | -2 | -4);
        Ok(())
    }

    fn commutes_with(&self, other: &PauliProduct) -> bool {
        let mut terms_map = HashMap::new();
        for term in &self.terms {
            terms_map.insert(term.qubit, term.basis);
        }
        let mut sum_signs = 0;
        for term in &other.terms {
            if let Some(&basis) = terms_map.get(&term.qubit) {
                if basis != term.basis && basis != 'I' && term.basis != 'I' {
                    sum_signs += 1;
                }
            }
        }
        sum_signs % 2 == 0
    }

    fn commute_right(&self, rhs: &PauliProduct) -> PauliProduct {
        if self.commutes_with(rhs) {
            return rhs.clone();
        }
        if !self.is_clifford {
            panic!("Currently only support commuting right of Clifford angles");
        }

        let mut new_prod = PauliProduct::new();
        new_prod.angle_numerator = rhs.angle_numerator;
        new_prod.angle_denominator = rhs.angle_denominator;
        new_prod.is_clifford = rhs.is_clifford;

        // Apply commutation rules
        for term in &rhs.terms {
            let mut new_term = term.clone();
            for self_term in &self.terms {
                if self_term.qubit == term.qubit {
                    match (self_term.basis, term.basis) {
                        ('X', 'Y') => {
                            new_term.basis = 'Z';
                            new_term.phase = add_phase(add_phase(new_term.phase, 1), 1);
                        }
                        ('X', 'Z') => {
                            new_term.basis = 'Y';
                            new_term.phase = add_phase(add_phase(new_term.phase, 3), 1);
                        }
                        ('Y', 'X') => {
                            new_term.basis = 'Z';
                            new_term.phase = add_phase(add_phase(new_term.phase, 3), 1);
                        }
                        ('Y', 'Z') => {
                            new_term.basis = 'X';
                            new_term.phase = add_phase(add_phase(new_term.phase, 1), 1);
                        }
                        ('Z', 'X') => {
                            new_term.basis = 'Y';
                            new_term.phase = add_phase(add_phase(new_term.phase, 1), 1);
                        }
                        ('Z', 'Y') => {
                            new_term.basis = 'X';
                            new_term.phase = add_phase(add_phase(new_term.phase, 3), 1);
                        }
                        _ => {}
                    }
                }
            }
            if self.angle_numerator > self.angle_denominator {
                new_term.phase = add_phase(new_term.phase, 2);
            }
            new_prod.terms.push(new_term);
        }

        new_prod
    }
}

impl std::fmt::Display for PauliProduct {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            write!(f, "{}", term)?;
        }

        if self.angle_denominator == 1 {
            write!(f, "<Angle({}pi)>", self.angle_numerator)
        } else if self.angle_numerator == 1 {
            write!(f, "<Angle(pi/{})>", self.angle_denominator)
        } else {
            write!(
                f,
                "<Angle({}pi/{})>",
                self.angle_numerator, self.angle_denominator
            )
        }
    }
}

#[derive(Debug)]
struct PauliProductDAG {
    products: Vec<PauliProduct>,
    children: Vec<HashSet<usize>>,
    parents: Vec<HashSet<usize>>,
    roots: HashSet<usize>,
    topological_order: Vec<usize>,
    max_qubit: i32,
    topo_steps: usize,
    update_topo_calls: usize,
    num_nodes: usize,
}

impl PauliProductDAG {
    fn new() -> Self {
        PauliProductDAG {
            products: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            roots: HashSet::new(),
            topological_order: Vec::new(),
            max_qubit: 0,
            topo_steps: 0,
            update_topo_calls: 0,
            num_nodes: 0,
        }
    }

    fn is_root(&self, node_id: usize) -> bool {
        self.parents[node_id].is_empty()
    }

    fn is_clifford(&self, node_id: usize) -> bool {
        self.products[node_id].is_clifford
    }

    fn is_bad_topo_order(&self, node_id: usize) -> bool {
        for &child_id in &self.children[node_id] {
            if self.topological_order[child_id] < self.topological_order[node_id] {
                return true;
            }
        }
        for &parent_id in &self.parents[node_id] {
            if self.topological_order[parent_id] > self.topological_order[node_id] {
                return true;
            }
        }
        false
    }

    fn swap_nodes(&mut self, node_id: usize, parent_id: usize) {
        self.children[parent_id].remove(&node_id);
        self.parents[parent_id].insert(node_id);
        self.parents[node_id].remove(&parent_id);
        self.children[node_id].insert(parent_id);

        let tmp = self.topological_order[node_id];
        self.topological_order[node_id] = self.topological_order[parent_id];
        self.topological_order[parent_id] = tmp;

        if self.roots.contains(&parent_id) {
            self.roots.remove(&parent_id);
            self.roots.insert(node_id);
        } else if self.parents[node_id].is_empty() {
            self.roots.insert(node_id);
        }
    }

    fn update_topological_order_starting_at(&mut self, node_id: usize) {
        self.update_topo_calls += 1;
        let offset = self.topological_order[node_id];

        let mut indegrees = vec![0; self.num_nodes];
        let mut subgraph_nodes = 0;

        for ni in 0..self.num_nodes {
            if self.topological_order[ni] >= offset {
                subgraph_nodes += 1;
                for &child_id in &self.children[ni] {
                    if self.topological_order[child_id] >= offset {
                        indegrees[child_id] += 1;
                    }
                }
            }
        }

        let mut new_order = Vec::with_capacity(subgraph_nodes);
        let mut queue = VecDeque::new();

        for ni in 0..self.num_nodes {
            if self.topological_order[ni] >= offset && indegrees[ni] == 0 {
                queue.push_back(ni);
            }
        }

        while let Some(current) = queue.pop_front() {
            new_order.push(current);
            self.topo_steps += 1;

            for &child_id in &self.children[current] {
                if self.topological_order[child_id] >= offset {
                    indegrees[child_id] -= 1;
                    if indegrees[child_id] == 0 {
                        queue.push_back(child_id);
                    }
                }
            }
        }

        for (ni, &node) in new_order.iter().enumerate() {
            self.topological_order[node] = ni + offset;
        }
    }

    fn load_node_from_string(&mut self, s: &str, node_id: usize) -> io::Result<()> {
        let tokens: Vec<&str> = s.split('\t').collect();
        const NUM_TOKENS: usize = 4;

        if tokens.len() != NUM_TOKENS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Incorrect number of tokens: expected {} but got {} for line: {}",
                    NUM_TOKENS,
                    tokens.len(),
                    s
                ),
            ));
        }

        let id: usize = tokens[0]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid node ID"))?;
        assert_eq!(id, node_id, "Node ID mismatch");

        self.products[node_id].load_from_string(tokens[1])?;

        // Parse children and parents
        self.children[node_id] = Self::parse_id_list(tokens[2])?;
        self.parents[node_id] = Self::parse_id_list(tokens[3])?;

        Ok(())
    }

    fn parse_id_list(s: &str) -> io::Result<HashSet<usize>> {
        if s == "[]" || s == "set()" {
            return Ok(HashSet::new());
        }

        let s = s
            .trim_start_matches(['[', '{'])
            .trim_end_matches([']', '}']);

        s.split(',')
            .map(|id| {
                id.trim()
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid ID in list"))
            })
            .collect()
    }

    fn load_from_file(&mut self, fname: &str) -> io::Result<()> {
        println!("Loading circuit from {}", fname);
        let file = File::open(fname)?;
        let reader = BufReader::new(file);
        let mut lines = Vec::new();

        // Skip header
        for line in reader.lines().skip(1) {
            lines.push(line?);
        }

        println!("Found {} lines in {}", lines.len(), fname);
        self.num_nodes = lines.len();
        self.children = vec![HashSet::new(); self.num_nodes];
        self.parents = vec![HashSet::new(); self.num_nodes];
        self.products = vec![PauliProduct::new(); self.num_nodes];
        self.topological_order = (0..self.num_nodes).collect();

        let mut num_cliffords = 0;
        let mut num_edges = 0;

        for (i, line) in lines.iter().enumerate() {
            self.load_node_from_string(line, i)?;

            if self.is_root(i) {
                self.roots.insert(i);
            }
            if self.is_clifford(i) {
                num_cliffords += 1;
            }

            for term in &self.products[i].terms {
                self.max_qubit = self.max_qubit.max(term.qubit);
            }
            num_edges += self.children[i].len();
        }

        println!(
            "Loaded {} products from {} of which {} are cliffords and {} are roots, with max qubit {}",
            self.num_nodes, fname, num_cliffords, self.roots.len(), self.max_qubit
        );
        println!(
            "Forms a dag with {} nodes and {} edges",
            self.num_nodes, num_edges
        );

        Ok(())
    }

    fn is_uncommuted_nonclifford(
        &self,
        node_id: usize,
        uncommuted_noncliffords: &BTreeSet<usize>,
    ) -> bool {
        if self.is_clifford(node_id) {
            return false;
        }
        if self.is_root(node_id) {
            return true;
        }
        if uncommuted_noncliffords.contains(&node_id) {
            return true;
        }
        for &child_id in &self.children[node_id] {
            if self.is_clifford(child_id) || uncommuted_noncliffords.contains(&child_id) {
                return true;
            }
        }
        false
    }

    fn done_commuting_nonclifford(
        &self,
        node_id: usize,
        uncommuted_noncliffords: &BTreeSet<usize>,
    ) -> bool {
        for &parent_id in &self.parents[node_id] {
            if self.is_clifford(parent_id) || uncommuted_noncliffords.contains(&parent_id) {
                return false;
            }
        }
        true
    }

    fn get_valid_parent_cliffords(&self, node_id: usize) -> Vec<usize> {
        let mut parent_cliffords = Vec::new();
        for &parent_id in &self.parents[node_id] {
            if self.is_clifford(parent_id) && !self.indirect_path_exists(parent_id, node_id) {
                parent_cliffords.push(parent_id);
            }
        }
        parent_cliffords
    }

    fn youngest_node(&self, nodes: &[usize]) -> usize {
        *nodes
            .iter()
            .max_by_key(|&&id| self.topological_order[id])
            .expect("Empty node list")
    }

    fn indirect_path_exists(&self, start: usize, end: usize) -> bool {
        if start == end {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        for &child_id in &self.children[start] {
            if child_id != end {
                queue.push_back(child_id);
                visited.insert(child_id);
            }
        }

        while let Some(current) = queue.pop_front() {
            for &child_id in &self.children[current] {
                if child_id == end {
                    return true;
                }
                if !visited.contains(&child_id) {
                    visited.insert(child_id);
                    queue.push_back(child_id);
                }
            }
        }
        false
    }

    fn commute_clifford_right(&mut self, clifford_id: usize, node_id: usize) {
        if !self.is_clifford(clifford_id) {
            debug!(
                "Node {} is not a clifford, skipping commute with {}",
                clifford_id, node_id
            );
            return;
        }
        let new_node_prod = self.products[clifford_id].commute_right(&self.products[node_id]);
        debug!(
            "Commuting clifford {} with nonclifford {}: {} -> {}",
            clifford_id, node_id, self.products[node_id], new_node_prod
        );
        self.products[node_id] = new_node_prod;
        self.swap_nodes(clifford_id, node_id);

        if self.is_bad_topo_order(node_id) || self.is_bad_topo_order(clifford_id) {
            self.update_topological_order_starting_at(node_id);
            assert!(!self.is_bad_topo_order(node_id) && !self.is_bad_topo_order(clifford_id));
        }
    }

    fn commute_all_cliffords(&mut self) {
        let _timer = Timer::new("commute_all_cliffords");
        let mut uncommuted_noncliffords = BTreeSet::new();

        for i in 0..self.num_nodes {
            if self.is_uncommuted_nonclifford(i, &uncommuted_noncliffords) {
                uncommuted_noncliffords.insert(i);
            }
        }

        if uncommuted_noncliffords.is_empty() {
            println!("No uncommuted non-Cliffords");
            return;
        }

        println!(
            "Commuting {} uncommuted noncliffords",
            uncommuted_noncliffords.len()
        );

        let mut num_commuted = 0;
        let num_uncommuted = uncommuted_noncliffords.len();
        let update_tick = (num_uncommuted as f64 / 100.0) as usize;
        let mut next_tick = update_tick;
        let mut i = 0;

        while !uncommuted_noncliffords.is_empty() {
            if num_commuted >= next_tick {
                println!("{} ", (num_commuted * 100 / num_uncommuted));
                std::io::stdout().flush().unwrap();
                next_tick = num_commuted + update_tick;
            }

            let mut finished_noncliffords = Vec::new();
            for &node_id in &uncommuted_noncliffords {
                if self.done_commuting_nonclifford(node_id, &uncommuted_noncliffords) {
                    debug!("Finished commuting nonclifford {}", node_id);
                    finished_noncliffords.push(node_id);
                    continue;
                }

                let parent_cliffords = self.get_valid_parent_cliffords(node_id);
                if parent_cliffords.is_empty() {
                    debug!("No valid parent cliffords for nonclifford {}", node_id);
                    continue;
                }

                // Check for loops
                for &parent_id in &self.parents[node_id] {
                    if self.children[node_id].contains(&parent_id) {
                        panic!("Loop detected");
                    }
                }

                let parent_id = self.youngest_node(&parent_cliffords);
                self.commute_clifford_right(parent_id, node_id);
            }

            for &nonclifford_id in &finished_noncliffords {
                uncommuted_noncliffords.remove(&nonclifford_id);
                debug!("Removed nonclifford {}", nonclifford_id);
            }

            num_commuted += finished_noncliffords.len();
            i += 1;
            debug!(
                "Iteration {}: Commuted {} noncliffords, remaining {}",
                i,
                finished_noncliffords.len(),
                uncommuted_noncliffords.len()
            );
            if i == 5 {
                break;
            }
        }

        println!();

        // Verify results
        for node_id in 0..self.num_nodes {
            if self.is_clifford(node_id) {
                for &child_id in &self.children[node_id] {
                    if !self.is_clifford(child_id) {
                        warn!(
                            "Found clifford {} with nonclifford child {}",
                            node_id, child_id
                        );
                    }
                }
            }
        }
        println!(
            "There were {} steps in {} calls to update the topological order",
            self.topo_steps, self.update_topo_calls
        );
    }
}

impl std::fmt::Display for PauliProductDAG {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "id\tproduct\tchildren\tparents")?;
        for i in 0..self.num_nodes {
            write!(f, "{}\t{}\t", i, self.products[i])?;
            write!(f, "[")?;
            let mut v = Vec::from_iter(self.children[i].iter().cloned());
            v.sort_unstable();
            for (j, &child) in v.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", child)?;
            }
            write!(f, "]\t")?;
            write!(f, "[")?;
            let mut v = Vec::from_iter(self.parents[i].iter().cloned());
            v.sort_unstable();
            for (j, &parent) in v.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", parent)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

fn main() -> io::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <input_file>", args[0]);
        std::process::exit(1);
    }

    let _timer = Timer::new("main");
    let mut dag = PauliProductDAG::new();

    dag.load_from_file(&args[1])?;

    let mut f = File::create(format!("{}-loaded.txt", args[1]))?;
    writeln!(f, "{}", dag)?;

    dag.commute_all_cliffords();

    let mut f = File::create(format!("{}-transpiled.txt", args[1]))?;
    writeln!(f, "{}", dag)?;

    Ok(())
}
