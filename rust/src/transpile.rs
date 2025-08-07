extern crate env_logger;
extern crate log;

use clap::Parser;
use lazy_static::lazy_static;
use log::{debug, warn};
use num::integer::gcd;
#[cfg(feature = "pythonapi")]
use pyo3::prelude::*;
#[cfg(feature = "pythonapi")]
use pyo3::types::{PyDict, PyList};
#[cfg(feature = "pythonapi")]
use pyo3::FromPyObject;
#[cfg(feature = "pythonapi")]
use pyo3::Python;
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

#[cfg(feature = "pythonapi")]
struct Circuit(PyObject);

#[cfg(feature = "pythonapi")]
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

#[cfg(feature = "pythonapi")]
impl FromPyObject<'_> for Circuit {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        Ok(Circuit(ob.into()))
    }
}

#[cfg(feature = "pythonapi")]
fn load_circuit(fname: &str) -> io::Result<Circuit> {
    let _timer = Timer::new("load_circuit");
    // Initialize Python
    Python::with_gil(|py| -> PyResult<Circuit> {
        // Import required modules
        let bqskit_circuit = py.import("bqskit.ir.circuit")?;
        let bqskit_compiler = py.import("bqskit.compiler")?;
        let bqskit_passes = py.import("bqskit.passes")?;

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
        // Compile circuit
        let compiler = bqskit_compiler.getattr("Compiler")?.call0()?;
        let circuit = compiler.call_method1("compile", (circuit, passes))?;
        // Unfold all
        circuit.call_method0("unfold_all")?;
        compile_timer.stop();
        compile_timer.done();
        Ok(circuit.extract()?)
    })
    .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Python error: {}", e)))
}

// Constants needed for commutation rules
lazy_static! {
    static ref TERM_MUL: HashMap<(char, char), (char, i32)> = {
        let mut m = HashMap::new();
        // Identity
        m.insert(('I', 'I'), ('I', 0));
        m.insert(('I', 'X'), ('X', 0));
        m.insert(('I', 'Y'), ('Y', 0));
        m.insert(('I', 'Z'), ('Z', 0));
        // X Pauli
        m.insert(('X', 'X'), ('I', 0));
        m.insert(('X', 'I'), ('X', 0));
        m.insert(('X', 'Y'), ('Z', 1));
        m.insert(('X', 'Z'), ('Y', 3));
        // Y Pauli
        m.insert(('Y', 'Y'), ('I', 0));
        m.insert(('Y', 'X'), ('Z', 3));
        m.insert(('Y', 'I'), ('Y', 0));
        m.insert(('Y', 'Z'), ('X', 1));
        // Z Pauli
        m.insert(('Z', 'Z'), ('I', 0));
        m.insert(('Z', 'I'), ('Z', 0));
        m.insert(('Z', 'X'), ('Y', 1));
        m.insert(('Z', 'Y'), ('X', 3));
        m
    };
}

fn basis_commutes_with(b1: char, b2: char) -> bool {
    b1 == 'I' || b2 == 'I' || b1 == b2
}

fn add_phase(phase1: i32, phase2: i32) -> i32 {
    (phase1 + phase2) % 4
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
        // If qubits don't match or bases commute, return rhs unchanged
        if self.qubit != rhs.qubit || basis_commutes_with(self.basis, rhs.basis) {
            return rhs.clone();
        }
        // Create new term starting with combined phases
        let mut new_term = PauliTerm {
            basis: 'I',
            phase: add_phase(self.phase, rhs.phase),
            qubit: self.qubit,
        };
        // Look up the commutation result in the multiplication table
        let key = (self.basis, rhs.basis);
        let (new_basis, phase_shift) = TERM_MUL
            .get(&key)
            .expect("Invalid Pauli bases for commutation");
        // Update term with commutation results
        new_term.basis = *new_basis;
        new_term.phase = add_phase(new_term.phase, *phase_shift);
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
        // Normalize to [0, 2π) in units of π
        if value < 0.0 {
            value += 2.0 * PI;
        }
        value = (value % (2.0 * PI)) / PI;
        // Find best rational approximation with denominator <= 16
        let max_denom = 16;
        let mut best_num = 0;
        let mut best_denom = 1;
        let mut min_error = f64::MAX;

        for denom in 1..=max_denom {
            let num = (value * denom as f64).round() as i32;
            let error = ((num as f64 / denom as f64) - value).abs();

            if error < min_error {
                min_error = error;
                best_num = num;
                best_denom = denom;
            }
        }
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
}

impl PauliProduct {
    fn new(terms: Vec<PauliTerm>, angle: Angle) -> Self {
        PauliProduct { terms, angle }
    }

    fn is_clifford(&self) -> bool {
        self.angle.is_clifford
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
            debug!("{} commutes with {}", self, rhs);
            return rhs.clone();
        }
        // Ensure we're commuting a Clifford angle rotation
        if !self.is_clifford() {
            panic!("Currently only support commuting right of Clifford angles");
        }
        // Use BTreeMap to maintain sorted order by qubit
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
        // Create new product with same size as combined terms
        let mut new_prod = PauliProduct {
            terms: Vec::with_capacity(all_terms_map.len()),
            angle: rhs.angle.clone(),
        };
        // Process terms in order of increasing qubit number
        for (_, (left_term, right_term)) in all_terms_map {
            match (left_term, right_term) {
                (None, Some(right)) => {
                    // Only right term exists
                    new_prod.terms.push(right.clone());
                }
                (Some(left), None) => {
                    // Only left term exists
                    new_prod.terms.push(left.clone());
                }
                (Some(left), Some(right)) => {
                    // Both terms exist - apply commutation rules
                    new_prod.terms.push(left.commute_right(right, &self.angle));
                }
                (None, None) => unreachable!("Map should not contain empty entries"),
            }
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
    update_topo_calls: usize,
    num_nodes: usize,
    update_topo_timer: IntermittentTimer,
    swap_nodes_timer: IntermittentTimer,
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
            update_topo_timer: IntermittentTimer::new("update_topo", ""),
            swap_nodes_timer: IntermittentTimer::new("swap_nodes", ""),
        }
    }

    fn is_root(&self, node_id: usize) -> bool {
        self.parents[node_id].is_empty()
    }

    fn is_clifford(&self, node_id: usize) -> bool {
        self.products[node_id].is_clifford()
    }

    fn involves_qubit(&self, node_id: usize, qubit: i32) -> bool {
        self.products[node_id]
            .terms
            .iter()
            .any(|term| term.qubit == qubit)
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

    fn is_uncommuted_nonclifford(
        &self,
        node_id: usize,
        uncommuted_noncliffords: &mut BTreeSet<usize>,
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
        if self.children[node_id].is_empty() {
            uncommuted_noncliffords.insert(node_id);
            return true;
        }
        for &child_id in &self.children[node_id] {
            if self.is_clifford(child_id)
                || self.is_uncommuted_nonclifford(child_id, uncommuted_noncliffords)
            {
                uncommuted_noncliffords.insert(node_id);
                return true;
            }
        }
        false
    }

    fn done_commuting_nonclifford(
        &self,
        node_id: usize,
        uncommuted_noncliffords: &mut BTreeSet<usize>,
    ) -> bool {
        for &parent_id in &self.parents[node_id] {
            if self.is_clifford(parent_id) || uncommuted_noncliffords.contains(&parent_id) {
                return false;
            }
        }
        true
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
        if topo_index_start > topo_index_end {
            return false;
        }
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
                    // Prune if end cannot depend on child_id
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

    fn relations_by_qubit(&self, node_id: usize, from_children: bool) -> HashMap<i32, usize> {
        let mut relation_map = HashMap::new();
        let relations = if from_children {
            &self.children[node_id]
        } else {
            &self.parents[node_id]
        };
        // Find relations by qubit
        for term in &self.products[node_id].terms {
            let mut selected_id = None;
            let mut selected_order = if from_children { std::usize::MAX } else { 0 };
            for &relation_id in relations {
                if self.involves_qubit(relation_id, term.qubit) {
                    if selected_id.is_none()
                        || (from_children && self.topological_order[relation_id] < selected_order)
                        || (!from_children && self.topological_order[relation_id] > selected_order)
                    {
                        selected_id = Some(relation_id);
                        selected_order = self.topological_order[relation_id];
                    }
                }
            }
            if selected_id.is_some() {
                relation_map.insert(term.qubit, selected_id.unwrap());
            }
        }
        relation_map
    }

    fn erase_related(
        &mut self,
        grandparent_id: usize,
        parent_id: usize,
        node_id: usize,
        from_children: bool,
    ) {
        debug!(
            "Erasing related {} {} {} from_children {}",
            grandparent_id, parent_id, node_id, from_children
        );
        // Only proceed if there's a relationship to erase
        if !self.children[grandparent_id].contains(&parent_id) {
            return;
        }
        // Find shared qubits between the three nodes
        let mut related_qubits = Vec::new();
        if from_children {
            // Check qubits from grandparent's perspective
            for (qubit, related_id) in self.relations_by_qubit(grandparent_id, true) {
                if related_id == parent_id {
                    related_qubits.push(qubit);
                }
            }
        } else {
            // Check qubits from parent's perspective
            for (qubit, related_id) in self.relations_by_qubit(parent_id, false) {
                if related_id == grandparent_id {
                    related_qubits.push(qubit);
                }
            }
        }
        // Only erase the relationship if all qubits are involved in the node
        let all_qubits_in_node = related_qubits
            .iter()
            .all(|&qubit| self.involves_qubit(node_id, qubit));
        if all_qubits_in_node {
            self.children[grandparent_id].remove(&parent_id);
            self.parents[parent_id].remove(&grandparent_id);
        }
    }

    fn swap_nodes(&mut self, param_node_id: usize, param_parent_id: usize) {
        /*
        If commuting a clifford through, update the node products beforehand.

        grandparents -> parent -> node -> children

                             |-------?---------v
        grandparents -?-> node -> *parent -?-> children
                   |------?--------^
        */
        self.swap_nodes_timer.start();
        let mut node_id = param_node_id;
        let mut parent_id = param_parent_id;
        // Check if parent is actually a child
        if self.children[node_id].contains(&parent_id) {
            std::mem::swap(&mut node_id, &mut parent_id);
            debug!("  parent is child, swapped: {} {}", node_id, parent_id);
        }
        //let _timer = Timer::new("by_qubit");
        // Find the parents associated with each of node's qubits
        let parent_parents_by_qubit = self.relations_by_qubit(parent_id, false);
        let node_children_by_qubit = self.relations_by_qubit(node_id, true);
        // Update basic relationships
        self.children[parent_id].remove(&node_id);
        self.parents[parent_id].insert(node_id);
        self.parents[node_id].remove(&parent_id);
        self.children[node_id].insert(parent_id);
        //let _timer = Timer::new("shared_qubits");
        // Only shared qubits need to be updated
        let mut node_qubits = HashSet::new();
        for term in &self.products[node_id].terms {
            node_qubits.insert(term.qubit);
        }

        let shared_qubits: Vec<i32> = self.products[parent_id]
            .terms
            .iter()
            .filter(|term| node_qubits.contains(&term.qubit))
            .map(|term| term.qubit)
            .collect();

        for qubit in shared_qubits {
            // What grandparents should now point at node?
            if let Some(&grandparent_id) = parent_parents_by_qubit.get(&qubit) {
                // Update the relationship between grandparent and parent
                self.erase_related(grandparent_id, parent_id, node_id, true);
                self.children[grandparent_id].insert(node_id);
                self.parents[node_id].insert(grandparent_id);
            }
            // What children should now be pointed to by parent?
            if let Some(&child_id) = node_children_by_qubit.get(&qubit) {
                // Update the relationship between node and child
                self.erase_related(node_id, child_id, parent_id, false);
                self.children[parent_id].insert(child_id);
                self.parents[child_id].insert(parent_id);
            }
        }
        // Swap topological order
        self.topological_order.swap(node_id, parent_id);
        // Update roots
        if self.roots.contains(&parent_id) {
            self.roots.remove(&parent_id);
            self.roots.insert(node_id);
        } else if self.parents[node_id].is_empty() {
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
            debug!(
                "Popped node {} with topo order {}",
                current, self.topological_order[current]
            );
            new_order.push(current);
            debug!("Append to new order {}", current);
            let mut sorted_children: Vec<_> = self.children[current].iter().copied().collect();
            sorted_children.sort_by_key(|&c| self.topological_order[c]);
            for &child_id in &sorted_children {
                self.topo_steps += 1;
                if self.topological_order[child_id] >= offset {
                    indegrees[child_id] -= 1;
                    if indegrees[child_id] == 0 {
                        queue.push_back(child_id);
                        debug!("From {} pushed node {}", current, child_id);
                    }
                }
            }
        }
        for (ni, &node) in new_order.iter().enumerate() {
            self.topological_order[node] = ni + offset;
        }
        debug!("New topo order: {:?}", self.topological_order);
        self.update_topo_timer.stop();
    }

    #[cfg(feature = "pythonapi")]
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

    #[cfg(feature = "pythonapi")]
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

    #[cfg(feature = "pythonapi")]
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

    #[cfg(feature = "pythonapi")]
    fn from_circuit(&mut self, fname: &str) -> io::Result<()> {
        let circuit = load_circuit(fname)?;
        let items = circuit
            .iter()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Python error: {}", e)))?;

        println!("Circuit has {} operations", items.len());

        for (_i, item) in items.iter().enumerate() {
            let item_str = Python::with_gil(|py| -> io::Result<String> {
                item.as_ref(py)
                    .str()
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::Other, format!("Python str error: {}", e))
                    })?
                    .extract::<String>()
                    .map_err(|e| {
                        io::Error::new(io::ErrorKind::Other, format!("Python extract error: {}", e))
                    })
            })?;
            debug!("Operation: {}", item_str);
            let products = Self::products_from_operation(&item_str)?;
            for product in &products {
                debug!("  {}", product);
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

    fn commute_clifford_right(&mut self, clifford_id: usize, node_id: usize) {
        if !self.is_clifford(clifford_id) {
            return;
        }
        let new_node_prod = self.products[clifford_id].commute_right(&self.products[node_id]);
        debug!(
            "Commuting clifford {} with nonclifford {}:\n   {} -> {}\n   topo order {} {}",
            clifford_id,
            node_id,
            self.products[node_id],
            new_node_prod,
            self.topological_order[clifford_id],
            self.topological_order[node_id]
        );
        self.products[node_id] = new_node_prod;
        self.swap_nodes(clifford_id, node_id);
        debug!(
            "after swap: {} {}\n   topo order {} {}",
            self.products[clifford_id],
            self.products[node_id],
            self.topological_order[clifford_id],
            self.topological_order[node_id]
        );
        if self.is_bad_topo_order(node_id) || self.is_bad_topo_order(clifford_id) {
            self.update_topological_order_starting_at(node_id);
            assert!(!self.is_bad_topo_order(node_id) && !self.is_bad_topo_order(clifford_id));
        }
    }

    fn commute_all_cliffords(&mut self) {
        let _timer = Timer::new("commute_all_cliffords");
        let mut uncommuted_noncliffords = BTreeSet::new();
        for i in 0..self.num_nodes {
            self.is_uncommuted_nonclifford(i, &mut uncommuted_noncliffords);
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
        let update_tick = (num_uncommuted as f64 / 20.0) as usize;
        let mut next_tick = update_tick;
        let mut loops = 0;
        while !uncommuted_noncliffords.is_empty() {
            if num_commuted >= next_tick {
                print!("{} ", (num_commuted * 100 / num_uncommuted));
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
                let parent_cliffords = self.get_valid_parent_cliffords(node_id);
                //let all_parents_cliffords = Vec::from_iter(self.parents[node_id].iter().cloned());
                //debug!(
                //    "node_id {} valid parents {:?} parents {:?}",
                //    node_id, parent_cliffords, all_parents_cliffords
                //);
                if parent_cliffords.is_empty() {
                    continue;
                }
                // Check for loops
                for &parent_id in &self.parents[node_id] {
                    if self.children[node_id].contains(&parent_id) {
                        panic!("Loop detected");
                    }
                }
                let parent_id = self.youngest_node(&parent_cliffords);
                if parent_cliffords.len() > 1 {
                    debug!("youngest parent {}", parent_id);
                }
                self.commute_clifford_right(parent_id, node_id);
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

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input circuit file path
    #[arg(help = "Path to the input circuit file")]
    input_file: String,

    /// Enable logging
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    if args.verbose {
        env_logger::init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    }

    let _timer = Timer::new("main");
    let mut dag = PauliProductDAG::new();

    #[cfg(feature = "pythonapi")]
    {
        dag.from_circuit(&args.input_file)?;
        let fname = format!("{}.compiled.txt", &args.input_file);
        println!("Saving compiled circuit to {}", fname);
        let mut f = File::create(fname)?;
        write!(f, "{}", dag)?;
    }

    //dag.load_from_file(&args.input_file)?;
    //let fname = format!("{}-loaded.txt", &args.input_file);
    //println!("Saving loaded circuit to {}", fname);
    //let mut f = File::create(fname)?;
    //writeln!(f, "{}", dag)?;

    dag.commute_all_cliffords();

    let fname = format!("{}.transpiled.txt", &args.input_file);
    println!("Saving transpiled circuit to {}", fname);
    let mut f = File::create(fname)?;
    write!(f, "{}", dag)?;

    dag.update_topo_timer.done();
    dag.swap_nodes_timer.done();

    Ok(())
}
