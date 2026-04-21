#!/usr/bin/env -S cargo run --bin transpile --
//! Transpile a Clifford+T circuit (QASM) to Pauli basis measurements (.trans).
//!
//! Input must be a `.cliffordt.qasm` file produced by `compile_circuit.py`.

use clap::Parser;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[allow(dead_code)]
mod pauliproduct;
mod tableau;
#[allow(dead_code)]
#[macro_use]
mod utils;

use tableau::{Gate1Q, Gate2Q, PauliString, Tableau};
use utils::Timer;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about = "Transpile a Clifford+T circuit to Pauli basis measurements")]
struct Args {
    #[arg(short, long = "input_file")]
    input_file: String,

    /// Defaults to the input stem with a .trans suffix.
    #[arg(short, long = "output_file", default_value = "")]
    output_file: String,

    /// Maximum Pauli product weight; -1 = no limit.
    #[arg(short = 'm', long = "max_width", default_value = "-1")]
    max_width: i32,
}

// ─────────────────────────────────────────────────────────────────────────────
// QASM gate representation
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum QasmGate {
    Clifford1Q { gate: Gate1Q, qubit: usize },
    Clifford2Q { gate: Gate2Q, control: usize, target: usize },
    T { qubit: usize },
    Tdg { qubit: usize },
    Measure { qubit: usize },
    Barrier,
}

impl QasmGate {
    fn qubits(&self) -> Vec<usize> {
        match self {
            QasmGate::Clifford1Q { qubit, .. } => vec![*qubit],
            QasmGate::Clifford2Q { control, target, .. } => {
                let mut v = vec![*control, *target];
                v.sort_unstable();
                v
            }
            QasmGate::T { qubit } => vec![*qubit],
            QasmGate::Tdg { qubit } => vec![*qubit],
            QasmGate::Measure { qubit } => vec![*qubit],
            QasmGate::Barrier => vec![],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpiler output types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Sign {
    Plus,
    Minus,
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sign::Plus => write!(f, "+"),
            Sign::Minus => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone)]
struct TransPauli {
    sign: Sign,
    /// One char per qubit: 'I', 'X', 'Y', 'Z'.
    ops: Vec<char>,
    /// "T" or "M".
    label: String,
}

impl TransPauli {
    fn from_pauli_string(ps: &PauliString, label: &str) -> Self {
        let sign = if ps.sign { Sign::Minus } else { Sign::Plus };
        let ops: Vec<char> = (0..ps.n).map(|q| ps.pauli_at(q)).collect();
        TransPauli { sign, ops, label: label.to_string() }
    }
}

impl fmt::Display for TransPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sign)?;
        for &c in &self.ops {
            if c == 'I' {
                write!(f, "_")?;
            } else {
                write!(f, "{}", c)?;
            }
        }
        write!(f, "<{}>", self.label)
    }
}

#[derive(Debug, Clone)]
struct TransClifford {
    /// One char per qubit: '_', 'X', or 'Z'.
    ops: Vec<char>,
    /// Gate name: "CX", "S", "Sdg", "SX", "SXdg".
    name: String,
}

impl TransClifford {
    /// Returns `None` for gates not emitted in the .trans format (X, Z, H, Y, CZ, SWAP).
    fn from_qasm_gate(gate: &QasmGate, n_qubits: usize) -> Option<Self> {
        let mut ops = vec!['_'; n_qubits];
        let name = match gate {
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control, target } => {
                ops[*control] = 'Z';
                ops[*target] = 'X';
                "CX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit } => {
                ops[*qubit] = 'Z';
                "S".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit } => {
                ops[*qubit] = 'Z';
                "Sdg".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit } => {
                ops[*qubit] = 'X';
                "SX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SXdg, qubit } => {
                ops[*qubit] = 'X';
                "SXdg".to_string()
            }
            _ => return None,
        };
        Some(TransClifford { ops, name })
    }
}

impl fmt::Display for TransClifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for &c in &self.ops {
            write!(f, "{}", c)?;
        }
        write!(f, "<{}>", self.name)
    }
}

#[derive(Debug, Clone)]
enum TransItem {
    Pauli(TransPauli),
    Clifford(TransClifford),
}

impl fmt::Display for TransItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransItem::Pauli(p) => write!(f, "{}", p),
            TransItem::Clifford(c) => write!(f, "{}", c),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// QASM parser
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a `.cliffordt.qasm` file into a gate list and qubit count.
/// Unrecognised lines (headers, custom gate defs, creg, etc.) are silently skipped.
fn parse_qasm(path: &str) -> io::Result<(usize, Vec<QasmGate>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut n_qubits = 0usize;
    let mut gates = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
            || line.starts_with("creg")
            || line.starts_with("gate ")
            || line.starts_with('{')
            || line.starts_with('}')
            || line.starts_with("U(")
        {
            continue;
        }

        if line.starts_with("barrier") {
            gates.push(QasmGate::Barrier);
            continue;
        }

        if line.starts_with("qreg") {
            if let Some(n) = parse_qreg(line) {
                n_qubits = n;
            }
            continue;
        }

        if line.starts_with("measure") {
            if let Some(q) = parse_single_qubit_index(line) {
                gates.push(QasmGate::Measure { qubit: q });
            }
            continue;
        }

        if let Some(g) = try_parse_2q(line) {
            gates.push(g);
            continue;
        }
        if let Some(g) = try_parse_1q(line) {
            gates.push(g);
            continue;
        }
    }

    let gates = reorder_by_cycles(gates, n_qubits);
    Ok((n_qubits, gates))
}

/// Reorder gates into bqskit's cycle-based order: pack each gate into the
/// earliest cycle where all its qubits are free, then sort within each cycle
/// by minimum qubit index.  This is a semantics-preserving reordering of
/// independent (commuting) gates.
fn reorder_by_cycles(gates: Vec<QasmGate>, n_qubits: usize) -> Vec<QasmGate> {
    if n_qubits == 0 {
        return gates;
    }

    let mut next_cycle = vec![0usize; n_qubits];
    let mut cycles: Vec<Vec<(usize, QasmGate)>> = Vec::new();

    for gate in gates {
        if matches!(gate, QasmGate::Barrier) {
            // Advance all qubits to the same cycle, place barrier, then advance past it.
            let sync = next_cycle.iter().copied().max().unwrap_or(0);
            if sync >= cycles.len() {
                cycles.resize_with(sync + 1, Vec::new);
            }
            cycles[sync].push((usize::MAX, gate));
            let after = sync + 1;
            for q in next_cycle.iter_mut() {
                *q = after;
            }
            continue;
        }

        let qubits = gate.qubits();
        let cycle = qubits.iter().map(|&q| next_cycle[q]).max().unwrap_or(0);
        if cycle >= cycles.len() {
            cycles.resize_with(cycle + 1, Vec::new);
        }
        let min_q = *qubits.iter().min().unwrap();
        cycles[cycle].push((min_q, gate));
        for &q in &qubits {
            next_cycle[q] = cycle + 1;
        }
    }

    let mut result = Vec::new();
    for cycle in cycles {
        let mut sorted = cycle;
        sorted.sort_by_key(|(min_q, _)| *min_q);
        for (_, gate) in sorted {
            if !matches!(gate, QasmGate::Barrier) {
                result.push(gate);
            }
        }
    }
    result
}

fn parse_qreg(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_single_qubit_index(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

fn parse_two_qubit_indices(line: &str) -> Option<(usize, usize)> {
    let mut indices = line.split('[').skip(1);
    let a: usize = indices.next()?.split(']').next()?.parse().ok()?;
    let b: usize = indices.next()?.split(']').next()?.parse().ok()?;
    Some((a, b))
}

fn try_parse_2q(line: &str) -> Option<QasmGate> {
    let lower = line.to_lowercase();
    if lower.starts_with("cx ") || lower.starts_with("cx\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CX, control: a, target: b });
    }
    if lower.starts_with("cz ") || lower.starts_with("cz\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::CZ, control: a, target: b });
    }
    if lower.starts_with("swap ") || lower.starts_with("swap\t") {
        let (a, b) = parse_two_qubit_indices(line)?;
        return Some(QasmGate::Clifford2Q { gate: Gate2Q::Swap, control: a, target: b });
    }
    None
}

fn try_parse_1q(line: &str) -> Option<QasmGate> {
    let mut parts = line.splitn(2, |c: char| c.is_whitespace());
    let name = parts.next()?.trim().to_lowercase();
    let name = name.trim_end_matches(';');
    let q = parse_single_qubit_index(line)?;
    let gate = match name {
        "h" => QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: q },
        "s" => QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: q },
        "sdg" => QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit: q },
        "sx" => QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit: q },
        "sxdg" => QasmGate::Clifford1Q { gate: Gate1Q::SXdg, qubit: q },
        "x" => QasmGate::Clifford1Q { gate: Gate1Q::X, qubit: q },
        "y" => QasmGate::Clifford1Q { gate: Gate1Q::Y, qubit: q },
        "z" => QasmGate::Clifford1Q { gate: Gate1Q::Z, qubit: q },
        "t" => QasmGate::T { qubit: q },
        "tdg" => QasmGate::Tdg { qubit: q },
        _ => return None,
    };
    Some(gate)
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-qubit Clifford optimizer
// ─────────────────────────────────────────────────────────────────────────────
//
// Each of the 24 single-qubit Cliffords is uniquely identified by the pair
// (X-image, Z-image) of its conjugation action on the Pauli group, where each
// image is one of the 6 signed non-identity Paulis {±X, ±Y, ±Z}.
//
// We encode a signed Pauli as a u8:
//   bits [1:0] = Pauli type: 0=X, 1=Y, 2=Z
//   bit  [2]   = sign: 0=+, 1=−
//
// A sequence of single-qubit gates is compressed by composing their actions
// into a single (x_img, z_img) state, then looking up the minimal gate
// sequence that produces that state from a precomputed table of all 24
// elements of the single-qubit Clifford group.
//
// Minimal sequences use only {S, Sdg, SX, SXdg, X, Z}.  Of these, X and Z
// are not emitted in the .trans format and are silently dropped.

/// Encode a signed single-qubit Pauli as a u8.
/// Pauli type: 0=X, 1=Y, 2=Z.  Sign bit 2: 0=+, 1=−.
#[inline]
fn pauli_code(pauli: u8, neg: bool) -> u8 {
    pauli | ((neg as u8) << 2)
}

/// Apply a single-qubit gate to a Clifford state `(x_img, z_img)` and return
/// the updated state.  Each image is a [`pauli_code`]-encoded signed Pauli
/// representing where the gate sends X (resp. Z) under conjugation.
fn apply_1q_to_state(gate: Gate1Q, x_img: u8, z_img: u8) -> (u8, u8) {
    // Conjugation rules (P → G·P·G†) for each gate on the three Paulis:
    //   H:    X→Z,   Y→-Y,  Z→X
    //   S:    X→Y,   Y→-X,  Z→Z
    //   Sdg:  X→-Y,  Y→X,   Z→Z
    //   SX:   X→X,   Y→-Z,  Z→-Y
    //   SXdg: X→X,   Y→Z,   Z→Y
    //   X:    X→X,   Y→-Y,  Z→-Z
    //   Y:    X→-X,  Y→Y,   Z→-Z
    //   Z:    X→-X,  Y→-Y,  Z→Z
    //
    // For a composed state (x_img, z_img), prepending gate G updates each
    // image independently: new_x_img = G(x_img), new_z_img = G(z_img).
    fn apply_gate_to_pauli(gate: Gate1Q, p: u8) -> u8 {
        let pauli = p & 0x3;
        let neg = (p >> 2) & 1 != 0;
        let (new_pauli, extra_neg) = match gate {
            Gate1Q::H => match pauli {
                0 => (2, false), // X→Z
                1 => (1, true),  // Y→-Y
                2 => (0, false), // Z→X
                _ => unreachable!(),
            },
            Gate1Q::S => match pauli {
                0 => (1, false), // X→Y
                1 => (0, true),  // Y→-X
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
            Gate1Q::Sdg => match pauli {
                0 => (1, true),  // X→-Y
                1 => (0, false), // Y→X
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
            Gate1Q::SX => match pauli {
                0 => (0, false), // X→X
                1 => (2, true),  // Y→-Z
                2 => (1, true),  // Z→-Y
                _ => unreachable!(),
            },
            Gate1Q::SXdg => match pauli {
                0 => (0, false), // X→X
                1 => (2, false), // Y→Z
                2 => (1, false), // Z→Y
                _ => unreachable!(),
            },
            Gate1Q::X => match pauli {
                0 => (0, false), // X→X
                1 => (1, true),  // Y→-Y
                2 => (2, true),  // Z→-Z
                _ => unreachable!(),
            },
            Gate1Q::Y => match pauli {
                0 => (0, true),  // X→-X
                1 => (1, false), // Y→Y
                2 => (2, true),  // Z→-Z
                _ => unreachable!(),
            },
            Gate1Q::Z => match pauli {
                0 => (0, true),  // X→-X
                1 => (1, true),  // Y→-Y
                2 => (2, false), // Z→Z
                _ => unreachable!(),
            },
        };
        pauli_code(new_pauli, neg ^ extra_neg)
    }
    (apply_gate_to_pauli(gate, x_img), apply_gate_to_pauli(gate, z_img))
}

/// Compose a sequence of single-qubit gates into a single Clifford state
/// `(x_img, z_img)` by applying each gate in order to the identity state
/// `(X+, Z+)`.
fn simulate_gate_sequence(gates: &[Gate1Q]) -> (u8, u8) {
    let mut x_img = pauli_code(0, false); // X+
    let mut z_img = pauli_code(2, false); // Z+
    for &g in gates {
        let (nx, nz) = apply_1q_to_state(g, x_img, z_img);
        x_img = nx;
        z_img = nz;
    }
    (x_img, z_img)
}

/// Build a lookup table mapping each of the 24 single-qubit Clifford states
/// `(x_img, z_img)` to a minimal gate sequence that produces it.
///
/// The 24 sequences cover all elements of the single-qubit Clifford group,
/// using only gates from {I, S, Sdg, SX, SXdg, X, Z} (at most 4 gates each).
/// The table is constructed by simulating each sequence and recording the
/// resulting state; no two sequences produce the same state.
fn build_clifford_table() -> std::collections::HashMap<(u8, u8), Vec<Gate1Q>> {
    use Gate1Q::*;
    let sequences: &[&[Gate1Q]] = &[
        &[],             // I
        &[X],            // X
        &[X, Z],         // X·Z
        &[Z],            // Z
        &[S, SX, S],     // S·SX·S  (= H up to global phase)
        &[S],            // S
        &[Sdg],          // Sdg
        &[S, X],         // S·X
        &[Sdg, X],       // Sdg·X
        &[Z, SX, S],     // Z·SX·S
        &[SX, S],        // SX·S
        &[S, SX, Z],     // S·SX·Z
        &[Z, SX, Z],     // Z·SX·Z
        &[Z, SX, Z, X],  // Z·SX·Z·X
        &[Z, SXdg],      // Z·SXdg
        &[Z, SX],        // Z·SX
        &[Sdg, SX, S],   // Sdg·SX·S
        &[Sdg, SX, Sdg], // Sdg·SX·Sdg
        &[S, SX, Sdg],   // S·SX·Sdg
        &[S, SXdg, Z],   // S·SXdg·Z
        &[S, SXdg],      // S·SXdg
        &[S, SX],        // S·SX
        &[SX, Sdg],      // SX·Sdg
        &[Z, SX, Sdg],   // Z·SX·Sdg
    ];
    let mut table = std::collections::HashMap::new();
    for seq in sequences {
        let state = simulate_gate_sequence(seq);
        table.insert(state, seq.to_vec());
    }
    table
}

/// Reduce a sequence of single-qubit Clifford gates on one qubit to the
/// minimal equivalent sequence.  Returns an empty vec for the identity.
///
/// The lookup table is built once per thread and cached for subsequent calls.
fn optimize_single_qubit_sequence(gates: &[Gate1Q]) -> Vec<Gate1Q> {
    if gates.is_empty() {
        return vec![];
    }
    let state = simulate_gate_sequence(gates);
    use std::cell::RefCell;
    thread_local! {
        static TABLE: RefCell<Option<std::collections::HashMap<(u8, u8), Vec<Gate1Q>>>> =
            RefCell::new(None);
    }
    TABLE.with(|t| {
        let mut borrow = t.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(build_clifford_table());
        }
        borrow.as_ref().unwrap().get(&state).cloned().unwrap_or_default()
    })
}

/// Reduce a sequence of Clifford gates to a minimal equivalent sequence and
/// return the result as [`TransClifford`] items ready for the .trans output.
///
/// Single-qubit gates on each qubit are accumulated independently and
/// compressed into a minimal sequence using the 24-element Clifford group
/// lookup.  When a 2-qubit gate is encountered, the pending single-qubit
/// sequences on its two qubits are flushed and optimized first; the 2-qubit
/// gate is then emitted unchanged.  Any remaining single-qubit sequences are
/// flushed at the end.
///
/// Only CX, S, Sdg, SX, and SXdg are emitted; X and Z are Pauli corrections
/// that do not need to be scheduled and are omitted from the output.
fn optimize_clifford_sequence(gates: &[QasmGate], n_qubits: usize) -> Vec<TransClifford> {
    let mut per_qubit: Vec<Vec<Gate1Q>> = vec![Vec::new(); n_qubits];
    let mut result: Vec<TransClifford> = Vec::new();

    fn flush_qubit(
        qubit: usize, per_qubit: &mut Vec<Vec<Gate1Q>>, result: &mut Vec<TransClifford>,
        n_qubits: usize,
    ) {
        let seq = std::mem::take(&mut per_qubit[qubit]);
        if seq.is_empty() {
            return;
        }
        for g in optimize_single_qubit_sequence(&seq) {
            let qasm_gate = QasmGate::Clifford1Q { gate: g, qubit };
            if let Some(tc) = TransClifford::from_qasm_gate(&qasm_gate, n_qubits) {
                result.push(tc);
            }
        }
    }

    for gate in gates {
        match gate {
            QasmGate::Clifford1Q { gate: g, qubit } => {
                per_qubit[*qubit].push(*g);
            }
            QasmGate::Clifford2Q { gate: g2, control, target } => {
                flush_qubit(*control, &mut per_qubit, &mut result, n_qubits);
                flush_qubit(*target, &mut per_qubit, &mut result, n_qubits);
                let qasm_gate =
                    QasmGate::Clifford2Q { gate: *g2, control: *control, target: *target };
                if let Some(tc) = TransClifford::from_qasm_gate(&qasm_gate, n_qubits) {
                    result.push(tc);
                }
            }
            _ => {}
        }
    }

    for qubit in 0..n_qubits {
        flush_qubit(qubit, &mut per_qubit, &mut result, n_qubits);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpiler
// ─────────────────────────────────────────────────────────────────────────────

fn transpile(n_qubits: usize, gates: &[QasmGate], max_weight: i32) -> Vec<TransItem> {
    let effective_max_weight = if max_weight <= 0 { n_qubits + 1 } else { max_weight as usize };

    let mut tableau = Tableau::new(n_qubits);
    let mut ops: Vec<TransItem> = Vec::new();
    let mut clifford_queue: Vec<QasmGate> = Vec::new();

    let has_t = gates.iter().any(|g| matches!(g, QasmGate::T { .. } | QasmGate::Tdg { .. }));
    if !has_t {
        eprintln!("Warning: circuit has no T gates");
    }

    // Append Z-basis measurements for any qubit not already measured.
    let mut gates_with_measurements = gates.to_vec();
    let measured: Vec<bool> = {
        let mut m = vec![false; n_qubits];
        for g in gates {
            if let QasmGate::Measure { qubit } = g {
                m[*qubit] = true;
            }
        }
        m
    };
    for (q, &has_m) in measured.iter().enumerate() {
        if !has_m {
            gates_with_measurements.push(QasmGate::Measure { qubit: q });
        }
    }

    let mut last_weight: Option<usize> = None;

    fn flush_clifford_queue(
        clifford_queue: &mut Vec<QasmGate>, ops: &mut Vec<TransItem>, tableau: &mut Tableau,
        n_qubits: usize,
    ) {
        let optimized = optimize_clifford_sequence(clifford_queue, n_qubits);
        for c in optimized {
            ops.push(TransItem::Clifford(c));
        }
        clifford_queue.clear();
        *tableau = Tableau::new(n_qubits);
    }

    for gate in &gates_with_measurements {
        match gate {
            QasmGate::Clifford1Q { gate: g, qubit } => {
                clifford_queue.push(gate.clone());
                tableau.prepend_1q_correct(*g, *qubit);
            }
            QasmGate::Clifford2Q { gate: g, control, target } => {
                clifford_queue.push(gate.clone());
                tableau.prepend_2q(*g, *control, *target);
            }
            QasmGate::T { qubit } => {
                let pre_pauli = make_z_pauli(n_qubits, *qubit, false);
                let conjugated = tableau.conjugate(&pre_pauli);
                let weight = conjugated.weight();
                last_weight = Some(weight);
                if weight > effective_max_weight {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                    let fresh_pauli = make_z_pauli(n_qubits, *qubit, false);
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "T")));
                } else {
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "T")));
                }
            }
            QasmGate::Tdg { qubit } => {
                // Tdg uses a negative-sign Z Pauli (−Z rotation).
                let pre_pauli = make_z_pauli(n_qubits, *qubit, true);
                let conjugated = tableau.conjugate(&pre_pauli);
                let weight = conjugated.weight();
                last_weight = Some(weight);
                if weight > effective_max_weight {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                    let fresh_pauli = make_z_pauli(n_qubits, *qubit, true);
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "T")));
                } else {
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "T")));
                }
            }
            QasmGate::Measure { qubit } => {
                if last_weight.is_none() {
                    eprintln!("Warning: measurement on qubit {} before any T gate", qubit);
                }
                let pre_pauli = make_z_pauli(n_qubits, *qubit, false);
                let conjugated = tableau.conjugate(&pre_pauli);
                // Flush only when the preceding T gate's conjugated weight exceeded
                // max_weight; the measurement's own conjugated weight is not checked.
                let should_flush = last_weight.map_or(false, |w| w > effective_max_weight);
                if should_flush {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, n_qubits);
                    let fresh_pauli = make_z_pauli(n_qubits, *qubit, false);
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&fresh_pauli, "M")));
                } else {
                    ops.push(TransItem::Pauli(TransPauli::from_pauli_string(&conjugated, "M")));
                }
            }
            QasmGate::Barrier => {}
        }
    }

    ops
}

fn make_z_pauli(n: usize, q: usize, negative: bool) -> PauliString {
    let mut ps = PauliString::identity(n);
    ps.z_bits[q] = true;
    ps.sign = negative;
    ps
}

// ─────────────────────────────────────────────────────────────────────────────
// Output writer
// ─────────────────────────────────────────────────────────────────────────────

fn write_trans(output_path: &str, items: &[TransItem]) -> io::Result<(usize, usize)> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut n_ts = 0usize;
    let mut n_cliffords = 0usize;

    for item in items {
        match item {
            TransItem::Pauli(p) => {
                writeln!(writer, "{}", p)?;
                if p.label == "T" {
                    n_ts += 1;
                }
            }
            TransItem::Clifford(c) => {
                writeln!(writer, "{}", c)?;
                n_cliffords += 1;
            }
        }
    }

    Ok((n_ts, n_cliffords))
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

fn count_stats(items: &[TransItem]) -> (usize, usize, usize, f64) {
    let n_cliffords = items.iter().filter(|i| matches!(i, TransItem::Clifford(_))).count();
    let pps: Vec<&TransPauli> = items
        .iter()
        .filter_map(|i| {
            if let TransItem::Pauli(p) = i {
                if p.label == "T" { Some(p) } else { None }
            } else {
                None
            }
        })
        .collect();
    let n_pps = pps.len();
    let avg_weight = if n_pps > 0 {
        let tot_weight: usize =
            pps.iter().map(|p| p.ops.iter().filter(|&&c| c != 'I').count()).sum();
        tot_weight as f64 / n_pps as f64
    } else {
        0.0
    };
    (n_cliffords, n_pps, items.len(), avg_weight)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let _total_timer = Timer::new("transpilation");

    let input_file = &args.input_file;
    if !input_file.ends_with(".cliffordt.qasm") {
        return Err(format!(
            "Input file must be a .cliffordt.qasm file produced by compile_circuit.py, got: {}",
            input_file
        )
        .into());
    }

    println!("Loading compiled circuit from {}", input_file);
    let _load_timer = Timer::new("load circuit");
    let (n_qubits, gates) = parse_qasm(input_file)?;
    drop(_load_timer);

    let tot_gates = gates.iter().filter(|g| !matches!(g, QasmGate::Measure { .. })).count();
    let clifford_gates = gates
        .iter()
        .filter(|g| matches!(g, QasmGate::Clifford1Q { .. } | QasmGate::Clifford2Q { .. }))
        .count();

    println!("Circuit has {} gates on {} qubits", tot_gates, n_qubits);

    let mut gate_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for g in &gates {
        let name = match g {
            QasmGate::Clifford1Q { gate, .. } => format!("{:?}", gate),
            QasmGate::Clifford2Q { gate, .. } => format!("{:?}", gate),
            QasmGate::T { .. } => "T".to_string(),
            QasmGate::Tdg { .. } => "Tdg".to_string(),
            QasmGate::Measure { .. } => "Measure".to_string(),
            QasmGate::Barrier => continue,
        };
        gate_names.insert(name);
    }
    println!("Gate set: {}", gate_names.iter().cloned().collect::<Vec<_>>().join(", "));

    let _tableau_timer = Timer::new("tableau");
    let items = transpile(n_qubits, &gates, args.max_width);
    drop(_tableau_timer);

    let (n_cliffords, n_pps, tot_post, avg_weight) = count_stats(&items);
    let tot_delta = tot_gates as i64 - tot_post as i64;
    let clifford_delta = clifford_gates as i64 - n_cliffords as i64;

    if tot_gates > 0 {
        println!("Circuit length:    {} (before) -> {} (after transpilation)", tot_gates, tot_post);
        println!(
            "  Overall reduction: {} operations removed ({:.1}% reduction)",
            tot_delta,
            100.0 * tot_delta as f64 / tot_gates as f64
        );
    } else {
        println!("  Overall reduction: N/A (empty circuit)");
    }
    println!(
        "  Clifford gates:    {} (before) -> {} (after transpilation)",
        clifford_gates, n_cliffords
    );
    if clifford_gates > 0 {
        println!(
            "  Clifford reduction: {} gates removed ({:.1}% reduction)",
            clifford_delta,
            100.0 * clifford_delta as f64 / clifford_gates as f64
        );
    } else {
        println!("  Clifford reduction: N/A (no Cliffords before transpilation)");
    }
    println!("  Non-Clifford Pauli products: {}", n_pps);
    println!("  Average Pauli product weight: {:.2}", avg_weight);

    let output_stem = if args.output_file.is_empty() {
        let p = Path::new(input_file);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        // stem is "foo.cliffordt" — strip the inner extension
        if stem.ends_with(".cliffordt") {
            stem[..stem.len() - ".cliffordt".len()].to_string()
        } else {
            stem.to_string()
        }
    } else {
        args.output_file.clone()
    };
    let output_path = if args.output_file.is_empty() {
        format!("{}.trans", output_stem)
    } else {
        args.output_file
    };

    let (n_ts, n_cliffords_written) = write_trans(&output_path, &items)?;
    println!("Wrote transpiled circuit to {}", output_path);
    println!("Wrote {} T gates and {} Cliffords", n_ts, n_cliffords_written);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_qasm(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".cliffordt.qasm").unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    // ── QASM parser ───────────────────────────────────────────────────────────

    #[test]
    fn parse_qasm_basic() {
        let f = write_qasm(&[
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "qreg q[2];",
            "h q[0];",
            "t q[1];",
            "cx q[0], q[1];",
        ]);
        let (n, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(n, 2);
        assert_eq!(gates.len(), 3);
        assert!(matches!(gates[0], QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 }));
        assert!(matches!(gates[1], QasmGate::T { qubit: 1 }));
        assert!(matches!(
            gates[2],
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 }
        ));
    }

    #[test]
    fn parse_qasm_sdg_tdg() {
        let f = write_qasm(&["qreg q[1];", "sdg q[0];", "tdg q[0];"]);
        let (_, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(matches!(gates[0], QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit: 0 }));
        assert!(matches!(gates[1], QasmGate::Tdg { qubit: 0 }));
    }

    // ── make_z_pauli ──────────────────────────────────────────────────────────

    #[test]
    fn make_z_pauli_positive() {
        let p = make_z_pauli(3, 1, false);
        assert_eq!(p.pauli_at(0), 'I');
        assert_eq!(p.pauli_at(1), 'Z');
        assert_eq!(p.pauli_at(2), 'I');
        assert!(!p.sign);
    }

    #[test]
    fn make_z_pauli_negative() {
        let p = make_z_pauli(2, 0, true);
        assert_eq!(p.pauli_at(0), 'Z');
        assert!(p.sign);
    }

    // ── TransPauli display ────────────────────────────────────────────────────

    #[test]
    fn trans_pauli_display_t_gate() {
        let ps = PauliString::from_str("+ZII");
        let tp = TransPauli::from_pauli_string(&ps, "T");
        assert_eq!(format!("{}", tp), "+Z__<T>");
    }

    #[test]
    fn trans_pauli_display_measurement() {
        let ps = PauliString::from_str("-IZI");
        let tp = TransPauli::from_pauli_string(&ps, "M");
        assert_eq!(format!("{}", tp), "-_Z_<M>");
    }

    // ── TransClifford display ─────────────────────────────────────────────────

    #[test]
    fn trans_clifford_cx_display() {
        let gate = QasmGate::Clifford2Q { gate: Gate2Q::CX, control: 0, target: 1 };
        let tc = TransClifford::from_qasm_gate(&gate, 2).unwrap();
        assert_eq!(format!("{}", tc), "+ZX<CX>");
    }

    #[test]
    fn trans_clifford_s_display() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 1 };
        let tc = TransClifford::from_qasm_gate(&gate, 3).unwrap();
        assert_eq!(format!("{}", tc), "+_Z_<S>");
    }

    #[test]
    fn trans_clifford_x_is_dropped() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::X, qubit: 0 };
        assert!(TransClifford::from_qasm_gate(&gate, 2).is_none());
    }

    #[test]
    fn trans_clifford_z_is_dropped() {
        let gate = QasmGate::Clifford1Q { gate: Gate1Q::Z, qubit: 0 };
        assert!(TransClifford::from_qasm_gate(&gate, 2).is_none());
    }

    // ── Transpiler ────────────────────────────────────────────────────────────

    #[test]
    fn transpile_single_t_gate_no_cliffords() {
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Plus);
            assert_eq!(p.ops[0], 'Z');
            assert_eq!(p.ops[1], 'I');
        }
    }

    #[test]
    fn transpile_h_then_t_gives_x_pauli() {
        // H maps Z→X, so T (a Z rotation) becomes an X rotation after conjugation.
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 }, QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.ops[0], 'X');
        }
    }

    #[test]
    fn transpile_tdg_gives_negative_sign() {
        let gates = vec![QasmGate::Tdg { qubit: 0 }];
        let items = transpile(1, &gates, -1);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Minus);
            assert_eq!(p.ops[0], 'Z');
        }
    }

    #[test]
    fn transpile_measurements_appended_for_all_qubits() {
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        let m_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "M")).collect();
        assert_eq!(m_items.len(), 2);
    }

    #[test]
    fn transpile_max_weight_1_flushes_cliffords() {
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 0 }, QasmGate::T { qubit: 0 }];
        let items_unlimited = transpile(2, &gates, -1);
        let cliffords_unlimited: Vec<_> =
            items_unlimited.iter().filter(|i| matches!(i, TransItem::Clifford(_))).collect();
        assert_eq!(cliffords_unlimited.len(), 0);
    }

    // ── count_stats ───────────────────────────────────────────────────────────

    #[test]
    fn count_stats_basic() {
        let items = vec![
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                ops: vec!['Z', 'I'],
                label: "T".to_string(),
            }),
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                ops: vec!['Z', 'Z'],
                label: "T".to_string(),
            }),
            TransItem::Clifford(TransClifford { ops: vec!['Z', '_'], name: "S".to_string() }),
        ];
        let (n_cliffords, n_paulis, tot, avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 1);
        assert_eq!(n_paulis, 2);
        assert_eq!(tot, 3);
        assert!((avg_weight - 1.5).abs() < 1e-9);
    }

    // ── count_stats — all Cliffords ───────────────────────────────────────────

    #[test]
    fn count_stats_all_cliffords() {
        let items = vec![
            TransItem::Clifford(TransClifford { ops: vec!['X', '_'], name: "CX".to_string() }),
            TransItem::Clifford(TransClifford { ops: vec!['Z', '_'], name: "S".to_string() }),
        ];
        let (n_cliffords, n_paulis, tot, _avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 2);
        assert_eq!(n_paulis, 0);
        assert_eq!(tot, 2);
    }

    // ── count_stats — empty ───────────────────────────────────────────────────

    #[test]
    fn count_stats_empty() {
        let items: Vec<TransItem> = vec![];
        let (n_cliffords, n_paulis, tot, avg_weight) = count_stats(&items);
        assert_eq!(n_cliffords, 0);
        assert_eq!(n_paulis, 0);
        assert_eq!(tot, 0);
        assert_eq!(avg_weight, 0.0);
    }

    // ── make_z_pauli — weight ─────────────────────────────────────────────────

    #[test]
    fn make_z_pauli_has_weight_one() {
        let ps = make_z_pauli(4, 2, false);
        assert_eq!(ps.weight(), 1);
    }

    #[test]
    fn make_z_pauli_correct_qubit() {
        let ps = make_z_pauli(4, 1, false);
        assert_eq!(ps.pauli_at(1), 'Z');
        assert_eq!(ps.pauli_at(0), 'I');
        assert_eq!(ps.pauli_at(2), 'I');
    }

    // ── parse_qasm — sdg and tdg gates ───────────────────────────────────────

    #[test]
    fn parse_qasm_sdg_and_tdg_gates() {
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "sdg q[0];", "tdg q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(n_qubits, 2);
        assert_eq!(gates.len(), 2);
    }

    // ── transpile — SX gate ───────────────────────────────────────────────────

    #[test]
    fn transpile_sx_gate_produces_clifford() {
        // Clifford gates are only flushed when a T gate's conjugated weight exceeds
        // max_weight. Use a 2-qubit circuit: cx entangles qubits so that T on q[1]
        // has weight 2 (> max_weight=1), triggering a flush of the clifford queue
        // which contains the preceding sx gate.
        let f =
            write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "cx q[0],q[1];", "sx q[0];", "t q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 1);
        // SX is a Clifford; should produce at least one Clifford item
        let has_clifford = items.iter().any(|i| matches!(i, TransItem::Clifford(_)));
        assert!(has_clifford, "SX gate should produce a Clifford TransItem");
    }

    // ── transpile — measurement appended for all qubits ──────────────────────

    #[test]
    fn transpile_two_qubit_circuit_appends_two_measurements() {
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "t q[0];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 0);
        let measurement_count = items
            .iter()
            .filter(|i| matches!(i, TransItem::Pauli(p) if p.label.contains('M')))
            .count();
        assert_eq!(measurement_count, 2, "should append one measurement per qubit");
    }

    // ── TransPauli display — negative sign ────────────────────────────────────

    #[test]
    fn trans_pauli_display_negative_sign() {
        let ps = make_z_pauli(2, 0, true); // negative
        let tp = TransPauli::from_pauli_string(&ps, "T");
        let s = format!("{}", tp);
        assert!(s.starts_with('-'), "negative pauli should start with '-': {}", s);
    }

    // ── TransClifford — CX display ────────────────────────────────────────────

    #[test]
    fn trans_clifford_cx_has_two_qubit_paulis() {
        // Clifford gates are only flushed when a T gate's conjugated weight exceeds
        // max_weight. After cx q[0],q[1], Z on q[1] maps to Z_0*Z_1 (weight 2).
        // With max_weight=1, weight 2 > 1 triggers a flush of the clifford queue
        // which contains the CX gate.
        let f = write_qasm(&["OPENQASM 2.0;", "qreg q[2];", "cx q[0],q[1];", "t q[1];"]);
        let (n_qubits, gates) = parse_qasm(f.path().to_str().unwrap()).unwrap();
        let items = transpile(n_qubits, &gates, 1);
        let cx_item = items
            .iter()
            .find(|i| if let TransItem::Clifford(c) = i { c.name == "CX" } else { false });
        assert!(cx_item.is_some(), "CX gate should produce a CX Clifford item");
        if let Some(TransItem::Clifford(c)) = cx_item {
            assert_eq!(c.ops.len(), 2, "CX should have 2 qubit entries");
        }
    }
}
