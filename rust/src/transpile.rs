extern crate env_logger;
extern crate log;

use clap::Parser;
use itertools::Itertools;
use log::{debug, warn};
use num::integer::gcd;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::FromPyObject;
use pyo3::Python;
use rayon::ThreadPoolBuilder;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, Write};
use std::time::{Duration, Instant};

struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    fn new(name: &str) -> Self {
        Timer {
            name: name.to_string(),
            start: Instant::now(),
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        println!(
            "\x1b[36mTiming: {} took {:.2} s\x1b[0m",
            self.name,
            self.start.elapsed().as_secs_f64()
        );
    }
}

#[derive(Debug)]
pub struct IntermittentTimer {
    start_time: Option<Instant>,
    total_elapsed: Duration,
    last_interval: Duration,
    name: String,
    interval_label: String,
}

impl IntermittentTimer {
    pub fn new(name: &str, interval_label: &str) -> Self {
        IntermittentTimer {
            start_time: None,
            total_elapsed: Duration::new(0, 0),
            last_interval: Duration::new(0, 0),
            name: name.to_string(),
            interval_label: interval_label.to_string(),
        }
    }

    pub fn done(&self) {
        println!(
            "\x1b[36mTiming: {} took {:.2} s\x1b[0m",
            self.name,
            self.total_elapsed.as_secs_f64()
        );
    }

    pub fn get_final(&self) -> String {
        format!("{}: {:.2}", self.name, self.total_elapsed.as_secs_f64())
    }

    pub fn start(&mut self) {
        if !self.interval_label.is_empty() {
            println!("{:<40}:", self.interval_label);
        }
        self.start_time = Some(Instant::now());
    }

    pub fn stop(&mut self) {
        if let Some(start) = self.start_time.take() {
            self.last_interval = start.elapsed();
            self.total_elapsed += self.last_interval;

            if !self.interval_label.is_empty() {
                println!("\x1b[34m{:.2} s\x1b[0m", self.last_interval.as_secs_f64());
            }
        }
    }

    pub fn get_interval(&self) -> f64 {
        self.last_interval.as_secs_f64()
    }
}

struct Circuit(PyObject);

impl Circuit {
    fn iter(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            // Convert circuit to iterator using __iter__
            let iter = self.0.as_ref(py).iter()?;
            // Collect all items into a Vec
            iter.map(|item| item.map(|x| x.into_py(py))).collect()
        })
    }
}

impl FromPyObject<'_> for Circuit {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Ok(Circuit(ob.into()))
    }
}

fn load_circuit(fname: &str) -> io::Result<Vec<String>> {
    let _timer = Timer::new("load_circuit");
    Python::with_gil(|py| -> io::Result<Vec<String>> {
        // Import required modules
        let bqskit_circuit = py.import("bqskit.ir.circuit")?;
        let bqskit_compiler = py.import("bqskit.compiler")?;
        let bqskit_passes = py.import("bqskit.passes")?;
        let mut file_read_timer = IntermittentTimer::new("reading circuit from file", "");
        file_read_timer.start();
        // Load and transform circuit
        let circuit = bqskit_circuit
            .getattr("Circuit")?
            .call_method1("from_file", (fname,))?;
        file_read_timer.stop();
        file_read_timer.done();
        let mut remove_measurements_timer =
            IntermittentTimer::new("removing measurements from circuit", "");
        remove_measurements_timer.start();
        // Remove measurements
        circuit.call_method0("remove_all_measurements")?;
        remove_measurements_timer.stop();
        remove_measurements_timer.done();
        let mut compile_timer = IntermittentTimer::new("compiling circuit", "");
        compile_timer.start();
        // Create the decomposition instance
        let decomp = bqskit_passes.getattr("ZXZXZDecomposition")?.call0()?;
        // Create ForEachBlockPass arguments
        let loop_body = PyList::new(py, &[decomp]);
        let filter_lambda = py.eval("lambda x: x.num_qudits == 1", None, None)?;
        // Create kwargs dictionary
        let kwargs = PyDict::new(py);
        kwargs.set_item("loop_body", loop_body)?;
        kwargs.set_item("collection_filter", filter_lambda)?;
        // Create passes list
        let foreach_pass = bqskit_passes
            .getattr("ForEachBlockPass")?
            .call((), Some(kwargs))?;
        let group_pass = bqskit_passes.getattr("GroupSingleQuditGatePass")?.call0()?;
        let passes = PyList::new(py, &[group_pass, foreach_pass]);
        // Compile circuit
        let compiler = bqskit_compiler.getattr("Compiler")?.call0()?;
        let circuit = compiler.call_method1("compile", (circuit, passes))?;
        // Unfold all
        circuit.call_method0("unfold_all")?;
        compile_timer.stop();
        compile_timer.done();
        let items = circuit
            .extract::<Circuit>()?
            .iter()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Python error: {}", e)))?;
        println!("Circuit has {} operations", items.len());
        let mut op_strings = Vec::new();
        for item in items.iter() {
            let item_str = item
                .as_ref(py)
                .str()
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("Python str error: {}", e))
                })?
                .extract::<String>()
                .map_err(|e| {
                    io::Error::new(io::ErrorKind::Other, format!("Python extract error: {}", e))
                })?;
            debug!("Operation: {}", item_str);
            op_strings.push(item_str);
        }
        Ok(op_strings)
    })
    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Python error: {}", e)))
}

fn basis_commutes_with(b1: char, b2: char) -> bool {
    b1 == 'I' || b2 == 'I' || b1 == b2
}

fn add_phase(phase1: i32, phase2: i32) -> i32 {
    (phase1 + phase2) % 4
}

fn multiply_pauli_ops(left: char, right: char) -> (char, i32) {
    match (left, right) {
        // Identity cases
        ('I', x) | (x, 'I') => (x, 0),

        // X cases
        ('X', 'X') => ('I', 0),
        ('X', 'Y') => ('Z', 1),
        ('X', 'Z') => ('Y', 3),

        // Y cases
        ('Y', 'Y') => ('I', 0),
        ('Y', 'X') => ('Z', 3),
        ('Y', 'Z') => ('X', 1),

        // Z cases
        ('Z', 'Z') => ('I', 0),
        ('Z', 'X') => ('Y', 1),
        ('Z', 'Y') => ('X', 3),

        _ => panic!("Invalid Pauli bases for commutation: {} {}", left, right),
    }
}

#[derive(Clone, Debug)]
struct PauliTerm {
    basis: char,
    phase: i32,
    qubit: i32,
}

impl PauliTerm {
    fn new(basis: char, phase: i32, qubit: i32) -> Self {
        PauliTerm {
            basis,
            phase,
            qubit,
        }
    }

    fn commute_right(&self, rhs: &PauliTerm, angle: &Angle) -> PauliTerm {
        assert!(self.qubit == rhs.qubit, "Qubits must match for commutation");
        if basis_commutes_with(self.basis, rhs.basis) {
            return rhs.clone();
        }
        // Create new term starting with combined phases
        let mut new_term = PauliTerm {
            basis: 'I',
            phase: add_phase(self.phase, rhs.phase),
            qubit: self.qubit,
        };
        let (new_basis, phase_shift) = multiply_pauli_ops(self.basis, rhs.basis);
        // Update term with commutation results
        new_term.basis = new_basis;
        new_term.phase = add_phase(new_term.phase, phase_shift);
        new_term.phase = add_phase(new_term.phase, 1);
        // Additional phase shift based on angle
        if angle.numerator > angle.denominator {
            new_term.phase = add_phase(new_term.phase, 2);
        }
        new_term
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

#[derive(Clone, Debug)]
struct Angle {
    numerator: i32,
    denominator: i32,
    is_clifford: bool,
}

impl Angle {
    fn new(numerator: i32, denominator: i32) -> Self {
        let gcd_factor = gcd(numerator, denominator);
        let numerator = (numerator as f64 / gcd_factor as f64).floor() as i32;
        let denominator = (denominator as f64 / gcd_factor as f64).floor() as i32;
        let is_clifford = matches!(denominator, 1 | 2 | 4 | -1 | -2 | -4);

        Angle {
            numerator,
            denominator,
            is_clifford,
        }
    }

    pub fn from_float(mut value: f64) -> Self {
        let pi2 = 2.0 * PI;
        // Normalize to [0, 2π) in units of π
        if value < 0.0 {
            value += pi2;
            // this can happen if the original value is very small
            if value == pi2 {
                value = pi2 - 1e-10;
            }
        }
        value = (value % pi2) / PI;
        // Find best rational approximation with limited denominator
        let (best_num, best_denom) = (1..=1000)
            .map(|denom| {
                let num = (value * denom as f64).round() as i32;
                let error = ((num as f64 / denom as f64) - value).abs();
                (num, denom, error)
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .map(|(num, denom, _)| (num, denom))
            .unwrap();
        Angle::new(best_num, best_denom)
    }
}

impl std::fmt::Display for Angle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominator == 1 {
            write!(f, "<Angle({}pi)>", self.numerator)
        } else if self.numerator == 1 {
            write!(f, "<Angle(pi/{})>", self.denominator)
        } else {
            write!(f, "<Angle({}pi/{})>", self.numerator, self.denominator)
        }
    }
}

#[derive(Clone, Debug)]
struct PauliProduct {
    terms: Vec<PauliTerm>,
    angle: Angle,
    qubit_cache: HashSet<i32>,
    layer: u32,
}

impl PauliProduct {
    fn new(terms: Vec<PauliTerm>, angle: Angle) -> Self {
        let qubit_cache = terms.iter().map(|term| term.qubit).collect();
        PauliProduct {
            terms,
            angle,
            qubit_cache,
            layer: 0,
        }
    }

    fn is_clifford(&self) -> bool {
        self.angle.is_clifford
    }

    fn commutes_with(&self, other: &PauliProduct) -> bool {
        let terms_map: HashMap<_, _> = self
            .terms
            .iter()
            .map(|term| (term.qubit, term.basis))
            .collect();
        other
            .terms
            .iter()
            .filter(|term| terms_map.contains_key(&term.qubit))
            .filter(|term| {
                let basis = terms_map[&term.qubit];
                basis != term.basis && basis != 'I' && term.basis != 'I'
            })
            .count()
            % 2
            == 0
    }

    fn commute_right(&self, rhs: &PauliProduct) -> PauliProduct {
        if self.commutes_with(rhs) {
            debug!("{} commutes with {}", self, rhs);
            return rhs.clone();
        }
        // Ensure we're commuting a Clifford angle rotation
        if !self.is_clifford() {
            panic!("Currently only support commuting right of Clifford angles");
        }
        // Calculate union of terms by qubit
        let mut all_terms_map: BTreeMap<i32, (Option<&PauliTerm>, Option<&PauliTerm>)> =
            BTreeMap::new();
        // Map left terms
        for term in &self.terms {
            all_terms_map.insert(term.qubit, (Some(term), None));
        }
        // Map right terms
        for term in &rhs.terms {
            all_terms_map
                .entry(term.qubit)
                .and_modify(|e| e.1 = Some(term))
                .or_insert((None, Some(term)));
        }
        // the new product will have the union of terms
        let mut new_terms = Vec::with_capacity(all_terms_map.len());
        // Process terms in order of increasing qubit number
        for (_, (left_term, right_term)) in all_terms_map {
            match (left_term, right_term) {
                (None, Some(right)) => {
                    // Only right term exists
                    new_terms.push(right.clone());
                }
                (Some(left), None) => {
                    // Only left term exists
                    new_terms.push(left.clone());
                }
                (Some(left), Some(right)) => {
                    // Both terms exist - apply commutation rules
                    new_terms.push(left.commute_right(right, &self.angle));
                }
                (None, None) => unreachable!("Map should not contain empty entries"),
            }
        }
        PauliProduct::new(new_terms, rhs.angle.clone())
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
        write!(f, "{}", self.angle)
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
    num_cliffords: usize,
    update_topo_calls: usize,
    num_nodes: usize,
    update_topo_timer: IntermittentTimer,
    swap_nodes_timer: IntermittentTimer,
    topo_sort_children: bool,
}

impl PauliProductDAG {
    fn new(topo_sort_children: bool) -> Self {
        PauliProductDAG {
            products: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            roots: HashSet::new(),
            topological_order: Vec::new(),
            max_qubit: 0,
            topo_steps: 0,
            num_cliffords: 0,
            update_topo_calls: 0,
            num_nodes: 0,
            update_topo_timer: IntermittentTimer::new("update_topo", ""),
            swap_nodes_timer: IntermittentTimer::new("swap_nodes", ""),
            topo_sort_children,
        }
    }

    fn is_root(&self, node_id: usize) -> bool {
        self.parents[node_id].is_empty()
    }

    fn is_clifford(&self, node_id: usize) -> bool {
        self.products[node_id].is_clifford()
    }

    fn is_bad_topo_order(&self, node_id: usize) -> bool {
        self.children[node_id]
            .iter()
            .any(|&child_id| self.topological_order[child_id] < self.topological_order[node_id])
            || self.parents[node_id].iter().any(|&parent_id| {
                self.topological_order[parent_id] > self.topological_order[node_id]
            })
    }

    fn collect_uncommuted_noncliffords(&self) -> BTreeSet<usize> {
        let mut uncommuted_noncliffords = BTreeSet::new();
        let mut visited = HashSet::new();

        for node_id in 0..self.num_nodes {
            // Skip if already processed or is Clifford
            if visited.contains(&node_id) || self.is_clifford(node_id) {
                continue;
            }
            // Check if node is uncommuted
            if self.is_root(node_id) || self.children[node_id].is_empty() {
                uncommuted_noncliffords.insert(node_id);
                continue;
            }

            // DFS to check children
            let mut stack = Vec::new();
            let mut node_visited = HashSet::new();
            stack.push(node_id);
            while let Some(current) = stack.pop() {
                if !node_visited.insert(current) {
                    continue;
                }
                for &child_id in &self.children[current] {
                    if self.is_clifford(child_id) || uncommuted_noncliffords.contains(&child_id) {
                        uncommuted_noncliffords.insert(node_id);
                        break;
                    }
                    stack.push(child_id);
                }
            }
            visited.extend(node_visited);
        }

        uncommuted_noncliffords
    }

    fn done_commuting_nonclifford(
        &self,
        node_id: usize,
        uncommuted_noncliffords: &mut BTreeSet<usize>,
    ) -> bool {
        !self.parents[node_id].iter().any(|&parent_id| {
            self.is_clifford(parent_id) || uncommuted_noncliffords.contains(&parent_id)
        })
    }

    fn indirect_path_exists(&self, start: usize, end: usize) -> bool {
        // Base case - path to self
        if start == end {
            return true;
        }
        assert!(
            !self.is_bad_topo_order(start) && !self.is_bad_topo_order(end),
            "Bad topological order detected"
        );
        let topo_index_start = self.topological_order[start];
        let topo_index_end = self.topological_order[end];
        assert!(topo_index_start < topo_index_end);
        // BFS traversal from start's children that are not end
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        for &child_id in &self.children[start] {
            if child_id != end {
                queue.push_back(child_id);
                visited.insert(child_id);
            }
        }
        // BFS traversal
        while let Some(current) = queue.pop_front() {
            for &child_id in &self.children[current] {
                // Found path to end
                if child_id == end {
                    return true;
                }

                if !visited.contains(&child_id) {
                    // Prune if end cannot depend on child_id - this is only a performance optimization
                    if self.topological_order[child_id] > topo_index_end {
                        continue;
                    }
                    visited.insert(child_id);
                    queue.push_back(child_id);
                }
            }
        }
        false
    }

    fn get_youngest_valid_parent_clifford(&self, node_id: usize) -> Option<usize> {
        self.parents[node_id]
            .iter()
            .filter(|&&parent_id| {
                self.is_clifford(parent_id) && !self.indirect_path_exists(parent_id, node_id)
            })
            .max_by_key(|&&id| self.topological_order[id])
            .copied()
    }

    fn get_relation_for_qubit(
        &self,
        node_id: usize,
        qubit: i32,
        from_children: bool,
    ) -> Option<usize> {
        let relations = if from_children {
            &self.children[node_id]
        } else {
            &self.parents[node_id]
        };
        relations
            .iter()
            .filter(|&&relation_id| self.products[relation_id].qubit_cache.contains(&qubit))
            .min_by_key(|&&relation_id| {
                if from_children {
                    self.topological_order[relation_id]
                } else {
                    std::usize::MAX - self.topological_order[relation_id]
                }
            })
            .copied()
    }

    fn get_related_qubits(
        &self,
        node_id: usize,
        related_id: usize,
        from_children: bool,
    ) -> Vec<i32> {
        self.products[node_id]
            .terms
            .iter()
            .filter_map(|term| {
                self.get_relation_for_qubit(node_id, term.qubit, from_children)
                    .and_then(|id| {
                        if id == related_id {
                            Some(term.qubit)
                        } else {
                            None
                        }
                    })
            })
            .collect()
    }

    fn should_erase_relation(
        &self,
        grandparent_id: usize,
        parent_id: usize,
        node_id: usize,
        from_children: bool,
    ) -> bool {
        if !self.children[grandparent_id].contains(&parent_id) {
            return false;
        }

        let related_qubits = if from_children {
            self.get_related_qubits(grandparent_id, parent_id, true)
        } else {
            self.get_related_qubits(parent_id, grandparent_id, false)
        };

        related_qubits
            .iter()
            .all(|&qubit| self.products[node_id].qubit_cache.contains(&qubit))
    }

    fn erase_relation(&mut self, parent_id: usize, node_id: usize) {
        assert!(self.children[parent_id].contains(&node_id));
        assert!(self.parents[node_id].contains(&parent_id));
        self.children[parent_id].remove(&node_id);
        self.parents[node_id].remove(&parent_id);
    }

    fn add_relation(&mut self, parent_id: usize, node_id: usize) {
        self.parents[node_id].insert(parent_id);
        self.children[parent_id].insert(node_id);
    }

    fn swap_nodes(&mut self, parent_id: usize, node_id: usize) {
        self.swap_nodes_timer.start();
        assert!(self.products[parent_id].is_clifford());
        assert!(self.children[parent_id].contains(&node_id));
        let shared_qubits: Vec<_> = self.products[parent_id]
            .qubit_cache
            .intersection(&self.products[node_id].qubit_cache)
            .copied()
            .collect();
        // Get relations maps for shared qubits only
        let grandparents_by_qubit: HashMap<_, _> = shared_qubits
            .iter()
            .filter_map(|&qubit| {
                self.get_relation_for_qubit(parent_id, qubit, false)
                    .map(|id| (qubit, id))
            })
            .collect();
        let children_by_qubit: HashMap<_, _> = shared_qubits
            .iter()
            .filter_map(|&qubit| {
                self.get_relation_for_qubit(node_id, qubit, true)
                    .map(|id| (qubit, id))
            })
            .collect();

        self.erase_relation(parent_id, node_id);
        self.add_relation(node_id, parent_id);

        for &qubit in &shared_qubits {
            // Update grandparent relationships
            if let Some(&grandparent_id) = grandparents_by_qubit.get(&qubit) {
                if self.should_erase_relation(grandparent_id, parent_id, node_id, true) {
                    self.erase_relation(grandparent_id, parent_id);
                }
                self.add_relation(grandparent_id, node_id);
            }
            // Update child relationships
            if let Some(&child_id) = children_by_qubit.get(&qubit) {
                if self.should_erase_relation(node_id, child_id, parent_id, false) {
                    self.erase_relation(node_id, child_id);
                }
                self.add_relation(parent_id, child_id);
            }
        }
        self.topological_order.swap(node_id, parent_id);
        if self.roots.remove(&parent_id) || self.parents[parent_id].is_empty() {
            self.roots.insert(node_id);
        }
        self.swap_nodes_timer.stop();
    }

    fn update_topological_order_starting_at(&mut self, node_id: usize) {
        debug!("Updating topological order starting at {}", node_id);
        debug!("Current topo order: {:?}", self.topological_order);
        self.update_topo_calls += 1;
        self.update_topo_timer.start();
        let offset = self.topological_order[node_id];
        let mut indegrees = vec![0; self.num_nodes];
        for ni in 0..self.num_nodes {
            if self.topological_order[ni] >= offset {
                for &child_id in &self.children[ni] {
                    if self.topological_order[child_id] >= offset {
                        indegrees[child_id] += 1;
                    }
                }
            }
        }
        let mut queue: VecDeque<_> = (0..self.num_nodes)
            .filter(|&ni| self.topological_order[ni] >= offset && indegrees[ni] == 0)
            .collect();

        let mut new_order_idx = offset;

        while let Some(current) = queue.pop_front() {
            debug!(
                "Popped node {} with topo order {}",
                current, self.topological_order[current]
            );
            self.topological_order[current] = new_order_idx;
            new_order_idx += 1;
            debug!("Append to new order {}", current);
            self.topo_steps += self.children[current].len();
            if self.topo_sort_children {
                self.children[current]
                    .iter()
                    .copied()
                    .filter(|&c| self.topological_order[c] >= offset)
                    .sorted_by_key(|&c| self.topological_order[c])
                    .for_each(|child_id| {
                        indegrees[child_id] -= 1;
                        if indegrees[child_id] == 0 {
                            queue.push_back(child_id);
                            debug!("From {} pushed node {}", current, child_id);
                        }
                    });
            } else {
                // this is much faster
                self.children[current]
                    .iter()
                    .filter(|&&child_id| self.topological_order[child_id] >= offset)
                    .for_each(|&child_id| {
                        indegrees[child_id] -= 1;
                        if indegrees[child_id] == 0 {
                            queue.push_back(child_id);
                            debug!("From {} pushed node {}", current, child_id);
                        }
                    });
            }
        }
        debug!("New topo order: {:?}", self.topological_order);
        self.update_topo_timer.stop();
    }

    fn parse_location(s: &str) -> io::Result<Vec<i32>> {
        let inner = s
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim_end_matches(','); // Handle trailing comma

        if inner.is_empty() {
            return Ok(Vec::new());
        }

        inner
            .split(',')
            .map(|x| {
                x.trim().parse::<i32>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Failed to parse location number: {}", e),
                    )
                })
            })
            .collect()
    }

    fn parse_param_from_op(s: &str) -> io::Result<Option<f64>> {
        let inner = s.trim_start_matches('[').trim_end_matches("])@");

        if inner.is_empty() {
            Ok(None)
        } else {
            inner.parse::<f64>().map(Some).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to parse float: {}", e),
                )
            })
        }
    }

    fn products_from_operation(op: &str) -> io::Result<Vec<PauliProduct>> {
        let parts: Vec<&str> = op.split('(').collect();
        let (gate_name, params, location) = (
            parts[0],
            PauliProductDAG::parse_param_from_op(parts[1])?,
            PauliProductDAG::parse_location(parts[2])?,
        );

        let gate = &gate_name[..gate_name.len() - 4]; // Remove "Gate" suffix

        // Define angles based on dagger flag
        let dagger = gate.ends_with("dg");
        let base_gate = if dagger {
            gate.trim_end_matches("dg")
        } else {
            gate
        };

        let (pi_2, pi_4, pi7_4, pi_8, pi15_8) = if dagger {
            (
                Angle::new(1, 2),
                Angle::new(7, 4),
                Angle::new(1, 4),
                Angle::new(15, 8),
                Angle::new(1, 8),
            )
        } else {
            (
                Angle::new(1, 2),
                Angle::new(1, 4),
                Angle::new(7, 4),
                Angle::new(1, 8),
                Angle::new(15, 8),
            )
        };

        // Unwrap location values after checking bounds
        let qubit0 = location.get(0).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Missing first location parameter",
            )
        })?;
        let qubit1 = location.get(1).unwrap_or(&-1);

        Ok(match base_gate {
            // Pauli gates (X, Y, Z) -> (pi/2)
            "X" | "Y" | "Z" => {
                vec![PauliProduct::new(
                    vec![PauliTerm::new(
                        base_gate.chars().next().unwrap(),
                        0,
                        *qubit0,
                    )],
                    pi_2.clone(),
                )]
            }
            // Single qubit Clifford gates
            "S" | "Sdg" => {
                vec![PauliProduct::new(
                    vec![PauliTerm::new(
                        'Z',
                        if gate == "Sdg" { -1 } else { 1 },
                        *qubit0,
                    )],
                    if base_gate == "s" {
                        pi_4.clone()
                    } else {
                        pi7_4.clone()
                    },
                )]
            }
            "SqrtX" => {
                vec![PauliProduct::new(
                    vec![PauliTerm::new('X', 0, *qubit0)],
                    pi_4.clone(),
                )]
            }
            "H" => {
                vec![
                    PauliProduct::new(vec![PauliTerm::new('Z', 0, *qubit0)], pi_4.clone()),
                    PauliProduct::new(vec![PauliTerm::new('X', 0, *qubit0)], pi_4.clone()),
                    PauliProduct::new(vec![PauliTerm::new('Z', 0, *qubit0)], pi_4.clone()),
                ]
            }
            // Two qubit Clifford gates
            "CNOT" => {
                vec![
                    PauliProduct::new(
                        if qubit0 < qubit1 {
                            vec![
                                PauliTerm::new('Z', 0, *qubit0),
                                PauliTerm::new('X', 0, *qubit1),
                            ]
                        } else {
                            vec![
                                PauliTerm::new('X', 0, *qubit1),
                                PauliTerm::new('Z', 0, *qubit0),
                            ]
                        },
                        pi_4.clone(),
                    ),
                    PauliProduct::new(vec![PauliTerm::new('Z', 0, *qubit0)], pi7_4.clone()),
                    PauliProduct::new(vec![PauliTerm::new('X', 0, *qubit1)], pi7_4.clone()),
                ]
            }
            "CZ" => {
                vec![
                    PauliProduct::new(
                        vec![
                            PauliTerm::new('Z', 0, *qubit0),
                            PauliTerm::new('Z', 0, *qubit1),
                        ],
                        pi_4.clone(),
                    ),
                    PauliProduct::new(vec![PauliTerm::new('Z', 0, *qubit0)], pi7_4.clone()),
                    PauliProduct::new(vec![PauliTerm::new('Z', 0, *qubit1)], pi7_4.clone()),
                ]
            }
            // T gates
            "T" | "Tdg" => {
                vec![PauliProduct::new(
                    vec![PauliTerm::new('Z', 0, *qubit0)],
                    if base_gate == "t" {
                        pi_8.clone()
                    } else {
                        pi15_8.clone()
                    },
                )]
            }
            // Rotation gates
            "RX" | "RY" | "RZ" => {
                let param = match params {
                    Some(a) => {
                        if dagger {
                            -a
                        } else {
                            a
                        }
                    }
                    None => panic!("Rotation gate requires angle parameter"),
                };
                let angle = Angle::from_float(param);
                let basis = match base_gate {
                    "RX" => 'X',
                    "RY" => 'Y',
                    "RZ" => 'Z',
                    _ => panic!(
                        "Invalid rotation gate: {} {} {}",
                        base_gate, gate, gate_name
                    ),
                };
                vec![PauliProduct::new(
                    vec![PauliTerm::new(basis, 0, *qubit0)],
                    angle,
                )]
            }
            "bar" => vec![], //  what is left of barrier after removing the "gate" ending
            _ => panic!("No known transpilation rule for {}", gate),
        })
    }

    fn from_circuit(&mut self, fname: &str) -> io::Result<()> {
        let op_strings = load_circuit(fname)?;
        for item_str in op_strings {
            let products = Self::products_from_operation(&item_str)?;
            if log::log_enabled!(log::Level::Debug) {
                for product in &products {
                    debug!("  {}", product);
                }
            }
            self.products.extend(products);
        }
        self.num_nodes = self.products.len();
        self.children = vec![HashSet::new(); self.num_nodes];
        self.parents = vec![HashSet::new(); self.num_nodes];
        self.topological_order = (0..self.num_nodes).collect();
        println!("Extracted {} Pauli products from circuit", self.num_nodes);
        for i in 0..self.num_nodes {
            if self.is_clifford(i) {
                self.roots.insert(i);
                self.num_cliffords += 1;
            }
            let mut prev_qubit = -1;
            for term in &self.products[i].terms {
                assert!(term.qubit >= 0);
                self.max_qubit = self.max_qubit.max(term.qubit);
                if term.qubit <= prev_qubit {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Qubit numbers must be in increasing order, found {} after {}",
                            term.qubit, prev_qubit
                        ),
                    ));
                }
                prev_qubit = term.qubit;
            }
        }
        let mut frontier: Vec<i32> = vec![-1; self.max_qubit as usize + 1];
        for i in 0..self.num_nodes {
            for term in &self.products[i].terms {
                let qubit = term.qubit as usize;
                if frontier[qubit] != -1 {
                    let parent_id = frontier[qubit] as usize;
                    assert!(parent_id != i);
                    self.children[parent_id].insert(i);
                    self.parents[i].insert(parent_id);
                }
                frontier[qubit] = i as i32; // Update frontier to current node
            }
        }
        Ok(())
    }

    fn commute_clifford_right(&mut self, parent_id: usize, node_id: usize) {
        assert!(self.is_clifford(parent_id));
        let new_node_prod = self.products[parent_id].commute_right(&self.products[node_id]);
        debug!(
            "Commuting clifford {} with nonclifford {}:\n   {} -> {}\n   topo order {} {}",
            parent_id,
            node_id,
            self.products[node_id],
            new_node_prod,
            self.topological_order[parent_id],
            self.topological_order[node_id]
        );
        self.products[node_id] = new_node_prod;
        self.swap_nodes(parent_id, node_id);
        debug!(
            "after swap: {} {}\n   topo order {} {}",
            self.products[parent_id],
            self.products[node_id],
            self.topological_order[parent_id],
            self.topological_order[node_id]
        );
        if self.is_bad_topo_order(node_id) || self.is_bad_topo_order(parent_id) {
            self.update_topological_order_starting_at(node_id);
            assert!(!self.is_bad_topo_order(node_id) && !self.is_bad_topo_order(parent_id));
        }
    }

    fn commute_all_cliffords(&mut self) {
        let _timer = Timer::new("commute_all_cliffords");
        let mut uncommuted_noncliffords = self.collect_uncommuted_noncliffords();
        if uncommuted_noncliffords.is_empty() {
            println!("No uncommuted non-Cliffords");
            return;
        }
        let num_uncommuted = uncommuted_noncliffords.len();
        print!("Commuting {} noncliffords:  00%", num_uncommuted);
        std::io::stdout().flush().unwrap();
        let mut num_commuted = 0;
        let update_tick = (num_uncommuted as f64 / 100.0) as usize;
        let mut next_tick = update_tick;
        let mut loops = 0;
        while !uncommuted_noncliffords.is_empty() {
            if num_commuted >= next_tick {
                print!("\x08\x08\x08{:02}%", (num_commuted * 100 / num_uncommuted));
                std::io::stdout().flush().unwrap();
                next_tick = num_commuted + update_tick;
            }
            let mut finished_noncliffords = Vec::new();
            // Create a temporary copy for iteration
            let current_noncliffords: Vec<_> = uncommuted_noncliffords.iter().copied().collect();
            for &node_id in &current_noncliffords {
                if self.done_commuting_nonclifford(node_id, &mut uncommuted_noncliffords) {
                    debug!("Finished commuting nonclifford {}", node_id);
                    finished_noncliffords.push(node_id);
                    continue;
                }
                if let Some(parent_id) = self.get_youngest_valid_parent_clifford(node_id) {
                    if self.children[node_id].contains(&parent_id) {
                        panic!("Loop detected");
                    }
                    debug!("youngest parent {}", parent_id);
                    self.commute_clifford_right(parent_id, node_id);
                }
            }
            for &nonclifford_id in &finished_noncliffords {
                uncommuted_noncliffords.remove(&nonclifford_id);
                debug!("Removed nonclifford {}", nonclifford_id);
            }
            num_commuted += finished_noncliffords.len();
            loops += 1;
            debug!(
                "Iteration {}: Commuted {} noncliffords, remaining {}",
                loops,
                finished_noncliffords.len(),
                uncommuted_noncliffords.len()
            );
        }
        println!("\x08\x08\x08{}%", 100);
        println!(
            "There were {} steps in {} calls to update the topological order",
            self.topo_steps, self.update_topo_calls
        );
    }

    fn verify_clifford_relations(&self) -> u32 {
        let num_cliffords = (0..self.num_nodes)
            .filter(|&node_id| self.is_clifford(node_id))
            .count() as u32;
        let num_failures = (0..self.num_nodes)
            .filter(|&node_id| self.is_clifford(node_id))
            .flat_map(|node_id| self.children[node_id].iter())
            .filter(|&&child_id| !self.is_clifford(child_id))
            .count() as u32;
        assert!(num_cliffords == self.num_cliffords as u32);
        num_failures
    }

    fn set_layers(&mut self) -> usize {
        let mut nodes_used = BTreeSet::new();
        let mut nodes_left: BTreeSet<_> = (0..self.num_nodes)
            .sorted_by_key(|&i| self.topological_order[i])
            .collect();
        let mut layer_idx = 0;
        while !nodes_left.is_empty() {
            let nodes_left_snapshot = nodes_left.clone();
            let nodes_used_snapshot = nodes_used.clone();
            for &node_id in &nodes_left_snapshot {
                // Check if all parents are used
                let all_parents_used = self.parents[node_id]
                    .iter()
                    .all(|&parent| nodes_used_snapshot.contains(&parent));

                if all_parents_used {
                    nodes_used.insert(node_id);
                    nodes_left.remove(&node_id);
                    self.products[node_id].layer = layer_idx;
                }
            }
            layer_idx += 1;
        }
        layer_idx as usize
    }
}

impl std::fmt::Display for PauliProductDAG {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "id\tproduct\tchildren\tparents")?;
        for i in 0..self.num_nodes {
            write!(f, "{}\t{}\t[", i, self.products[i])?;
            self.children[i]
                .iter()
                .sorted_unstable()
                .enumerate()
                .try_for_each(|(j, &child)| {
                    if j > 0 {
                        write!(f, ", ")?
                    }
                    write!(f, "{}", child)
                })?;

            write!(f, "]\t[")?;
            self.parents[i]
                .iter()
                .sorted_unstable()
                .enumerate()
                .try_for_each(|(j, &parent)| {
                    if j > 0 {
                        write!(f, ", ")?
                    }
                    write!(f, "{}", parent)
                })?;
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input circuit file path
    #[arg(help = "Path to the input .qasm circuit file")]
    input_file: String,
    /// Enable logging
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    /// Sort children topologically during updates
    #[arg(short = 's', long, default_value_t = false)]
    topo_sort_children: bool,
    /// Number of threads to use for parallel operations
    #[arg(short = 't', long, default_value_t = 8)]
    threads: usize,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    if args.verbose {
        env_logger::init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    let _timer = Timer::new("main");
    let mut dag = PauliProductDAG::new(args.topo_sort_children);

    dag.from_circuit(&args.input_file)?;
    let mut num_layers = dag.set_layers();
    println!("Circuit has {} layers", num_layers);

    let fname = format!("{}.compiled.txt", &args.input_file);
    println!("Saving compiled circuit to {}", fname);
    let mut f = File::create(fname)?;
    write!(f, "{}", dag)?;

    dag.commute_all_cliffords();
    num_layers = dag.set_layers();
    println!("Transpiled circuit has {} layers", num_layers);
    let num_failures = dag.verify_clifford_relations();
    if num_failures > 0 {
        warn!("Found {} failures", num_failures);
    }

    let fname = format!("{}.transpiled.txt", &args.input_file);
    println!("Saving transpiled circuit to {}", fname);
    write!(File::create(fname)?, "{}", dag)?;

    dag.update_topo_timer.done();
    dag.swap_nodes_timer.done();

    Ok(())
}
