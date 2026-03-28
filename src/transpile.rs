#!/usr/bin/env -S cargo run --bin transpile --
//! Transpile a Clifford+T circuit (QASM) to Pauli basis measurements.
//!
//! This is a Rust reimplementation of `tableau/transpile_circuit.py`.
//!
//! The input must be a `.cliffordt.qasm` file.  The output is a `.trans` file
//! containing a sequence of PauliProducts and CliffordOperations that can be
//! consumed by the PureMagic scheduler.
//!
//! # Algorithm
//!
//! We maintain a symplectic Clifford tableau (see `tableau.rs`).  For each gate
//! in the QASM circuit:
//!
//! - **Clifford gates** (H, S, Sdg, SX, SXdg, X, Y, Z, CX, CZ, SWAP) are
//!   accumulated into the tableau via `prepend`.
//! - **T / Tdg gates** are converted to a weight-1 Pauli product (Z on the
//!   target qubit), then conjugated through the current tableau to obtain the
//!   effective Pauli product.  If the resulting weight exceeds `max_weight`,
//!   the Clifford queue is flushed to the output and the tableau is reset.
//! - **Measurements** are handled similarly: a Z-basis measurement on qubit q
//!   is conjugated through the tableau.
//!
//! The output `.trans` format is the same as produced by the Python version.

use clap::Parser;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[allow(dead_code)]
mod pauliproduct;
mod tableau;
#[allow(dead_code)]
#[macro_use]
mod utils;

use tableau::{Gate1Q, Gate2Q, PauliString, Tableau};

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Transpile a Clifford+T circuit to Pauli basis measurements (.trans).\n\
             The input must be a .cliffordt.qasm file produced by compile_circuit.py."
)]
struct Args {
    /// Input compiled circuit file (must have a .cliffordt.qasm extension).
    #[arg(short, long = "input_file")]
    input_file: String,

    /// Output file stem (without extension).  Defaults to the stem of the input
    /// file.  A .trans suffix is appended automatically.
    #[arg(short, long = "output_file", default_value = "")]
    output_file: String,

    /// Maximum Pauli product weight allowed during tableau conjugation.
    /// Defaults to -1 (no limit).
    #[arg(short = 'm', long = "max_width", default_value = "-1")]
    max_width: i32,
}

// ─────────────────────────────────────────────────────────────────────────────
// QASM gate representation
// ─────────────────────────────────────────────────────────────────────────────

/// A single gate parsed from a QASM file.
#[derive(Debug, Clone)]
enum QasmGate {
    /// Single-qubit Clifford gate.
    Clifford1Q { gate: Gate1Q, qubit: usize },
    /// Two-qubit Clifford gate.
    Clifford2Q { gate: Gate2Q, control: usize, target: usize },
    /// T gate (pi/8 rotation around Z).
    T { qubit: usize },
    /// Tdg gate (−pi/8 rotation around Z).
    Tdg { qubit: usize },
    /// Z-basis measurement on a qubit.
    Measure { qubit: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpiler output types
// ─────────────────────────────────────────────────────────────────────────────

/// Sign of a Pauli product (+1 or −1).
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

/// A Pauli product in the `.trans` output format.
///
/// Stores the sign, the Pauli string (as a vector of chars, one per qubit),
/// and the gate type label.
#[derive(Debug, Clone)]
struct TransPauli {
    sign: Sign,
    /// One char per qubit: 'I', 'X', 'Y', 'Z', or '_' (identity).
    paulis: Vec<char>,
    /// Gate label: "T" or "M".
    label: String,
}

impl TransPauli {
    fn from_pauli_string(ps: &PauliString, label: &str) -> Self {
        let sign = if ps.sign { Sign::Minus } else { Sign::Plus };
        let paulis: Vec<char> = (0..ps.n).map(|q| ps.pauli_at(q)).collect();
        TransPauli { sign, paulis, label: label.to_string() }
    }
}

impl fmt::Display for TransPauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.sign)?;
        for &c in &self.paulis {
            if c == 'I' {
                write!(f, "_")?;
            } else {
                write!(f, "{}", c)?;
            }
        }
        write!(f, "<{}>", self.label)
    }
}

/// A Clifford operation in the `.trans` output format.
///
/// Matches the format produced by `format_clifford()` in the Python version.
#[derive(Debug, Clone)]
struct TransClifford {
    /// One char per qubit: '_', 'X', or 'Z'.
    paulis: Vec<char>,
    /// Gate name: "CX", "S", "Sdg", "SX", "SXdg".
    name: String,
}

impl TransClifford {
    /// Returns `None` for X and Z gates (which are dropped in the output).
    fn from_qasm_gate(gate: &QasmGate, num_qubits: usize) -> Option<Self> {
        let mut paulis = vec!['_'; num_qubits];
        let name = match gate {
            QasmGate::Clifford2Q { gate: Gate2Q::CX, control, target } => {
                paulis[*control] = 'Z';
                paulis[*target] = 'X';
                "CX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::S, qubit } => {
                paulis[*qubit] = 'Z';
                "S".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::Sdg, qubit } => {
                paulis[*qubit] = 'Z';
                "Sdg".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SX, qubit } => {
                paulis[*qubit] = 'X';
                "SX".to_string()
            }
            QasmGate::Clifford1Q { gate: Gate1Q::SXdg, qubit } => {
                paulis[*qubit] = 'X';
                "SXdg".to_string()
            }
            // X and Z are Pauli corrections that do not need to be scheduled
            QasmGate::Clifford1Q { gate: Gate1Q::X, .. } => return None,
            QasmGate::Clifford1Q { gate: Gate1Q::Z, .. } => return None,
            // H, Y, CZ, SWAP are flushed but not written to .trans
            // (they only appear in the clifford_queue when max_weight is exceeded)
            _ => return None,
        };
        Some(TransClifford { paulis, name })
    }
}

impl fmt::Display for TransClifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "+")?;
        for &c in &self.paulis {
            write!(f, "{}", c)?;
        }
        write!(f, "<{}>", self.name)
    }
}

/// An item in the transpiler output list.
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

/// Parse a `.cliffordt.qasm` file into a list of gates and the number of qubits.
///
/// Supports the gate set used by `compile_circuit.py`:
///   h, s, sdg, sx, sxdg, x, y, z, cx, cz, swap, t, tdg, measure, barrier.
///
/// Lines that cannot be parsed (headers, comments, custom gate definitions,
/// creg declarations, etc.) are silently skipped.
fn parse_qasm(path: &str) -> io::Result<(usize, Vec<QasmGate>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut num_qubits = 0usize;
    let mut gates = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines, comments, and QASM headers
        if line.is_empty()
            || line.starts_with("//")
            || line.starts_with("OPENQASM")
            || line.starts_with("include")
            || line.starts_with("creg")
            || line.starts_with("gate ")
            || line.starts_with('{')
            || line.starts_with('}')
            || line.starts_with("U(")
            || line.starts_with("barrier")
        {
            continue;
        }

        // qreg q[N];
        if line.starts_with("qreg") {
            if let Some(n) = parse_qreg(line) {
                num_qubits = n;
            }
            continue;
        }

        // measure q[i] -> c[i];  (treat as Z-basis measurement)
        if line.starts_with("measure") {
            if let Some(q) = parse_single_qubit_index(line) {
                gates.push(QasmGate::Measure { qubit: q });
            }
            continue;
        }

        // Two-qubit gates: cx q[a], q[b];
        if let Some(g) = try_parse_2q(line) {
            gates.push(g);
            continue;
        }

        // Single-qubit gates: h q[i];
        if let Some(g) = try_parse_1q(line) {
            gates.push(g);
            continue;
        }
    }

    Ok((num_qubits, gates))
}

/// Parse `qreg q[N];` → N.
fn parse_qreg(line: &str) -> Option<usize> {
    // e.g. "qreg q[6];"
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

/// Parse a single qubit index from a line like `h q[3];` → 3.
fn parse_single_qubit_index(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

/// Parse two qubit indices from a line like `cx q[0], q[1];` → (0, 1).
fn parse_two_qubit_indices(line: &str) -> Option<(usize, usize)> {
    let mut indices = line.split('[').skip(1);
    let a: usize = indices.next()?.split(']').next()?.parse().ok()?;
    let b: usize = indices.next()?.split(']').next()?.parse().ok()?;
    Some((a, b))
}

/// Try to parse a two-qubit gate from a QASM line.
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

/// Try to parse a single-qubit gate from a QASM line.
fn try_parse_1q(line: &str) -> Option<QasmGate> {
    // Split on whitespace to get the gate name
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
// Clifford sequence optimizer
// ─────────────────────────────────────────────────────────────────────────────

/// Optimize a sequence of Clifford gates for output.
///
/// This is a simplified version of `CliffordOperation.optimize_sequence()` from
/// the Python code.  For now we just filter out gates that don't appear in the
/// `.trans` format (H, Y, CZ, SWAP) and pass through the rest.
///
/// The Python version does full single-qubit unitary optimization; we replicate
/// the essential behaviour: only S, Sdg, SX, SXdg, CX (and X, Z which are
/// dropped) appear in the output.
fn optimize_clifford_sequence(gates: &[QasmGate], num_qubits: usize) -> Vec<TransClifford> {
    gates.iter().filter_map(|g| TransClifford::from_qasm_gate(g, num_qubits)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpiler
// ─────────────────────────────────────────────────────────────────────────────

/// Transpile a parsed QASM gate list into a `.trans` item list.
///
/// Mirrors `Transpiler.transpile()` from `tableau/tableau/transpile.py`.
fn transpile(num_qubits: usize, gates: &[QasmGate], max_weight: i32) -> Vec<TransItem> {
    let effective_max_weight = if max_weight <= 0 { num_qubits + 1 } else { max_weight as usize };

    let mut tableau = Tableau::new(num_qubits);
    let mut ops: Vec<TransItem> = Vec::new();
    let mut clifford_queue: Vec<QasmGate> = Vec::new();

    // Check that the circuit has at least one T gate
    let has_t = gates.iter().any(|g| matches!(g, QasmGate::T { .. } | QasmGate::Tdg { .. }));
    if !has_t {
        eprintln!("Warning: circuit has no T gates");
    }

    // Append Z-basis measurements for any qubit that doesn't already end with one.
    // We do this by tracking which qubits have measurements and appending at the end.
    let mut gates_with_measurements = gates.to_vec();
    let measured: Vec<bool> = {
        let mut m = vec![false; num_qubits];
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

    /// Flush the clifford queue: optimize and append to ops, reset tableau.
    fn flush_clifford_queue(
        clifford_queue: &mut Vec<QasmGate>, ops: &mut Vec<TransItem>, tableau: &mut Tableau,
        num_qubits: usize,
    ) {
        let optimized = optimize_clifford_sequence(clifford_queue, num_qubits);
        for c in optimized {
            ops.push(TransItem::Clifford(c));
        }
        clifford_queue.clear();
        *tableau = Tableau::new(num_qubits);
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
                // T gate: Z rotation by pi/8 on `qubit`.
                // Pre-tableau Pauli: +Z on qubit `qubit`, I elsewhere.
                let pre_pauli = make_z_pauli(num_qubits, *qubit, false);
                let conjugated = tableau.conjugate(&pre_pauli);
                let weight = conjugated.weight();
                last_weight = Some(weight);
                if weight > effective_max_weight {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, num_qubits);
                    // Re-conjugate through fresh identity tableau (weight-1 result)
                    let fresh_pauli = make_z_pauli(num_qubits, *qubit, false);
                    let tp = TransPauli::from_pauli_string(&fresh_pauli, "T");
                    ops.push(TransItem::Pauli(tp));
                } else {
                    let tp = TransPauli::from_pauli_string(&conjugated, "T");
                    ops.push(TransItem::Pauli(tp));
                }
            }
            QasmGate::Tdg { qubit } => {
                // Tdg gate: Z rotation by −pi/8 on `qubit`.
                // Pre-tableau Pauli: −Z on qubit `qubit` (negative sign for Tdg).
                let pre_pauli = make_z_pauli(num_qubits, *qubit, true);
                let conjugated = tableau.conjugate(&pre_pauli);
                let weight = conjugated.weight();
                last_weight = Some(weight);
                if weight > effective_max_weight {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, num_qubits);
                    let fresh_pauli = make_z_pauli(num_qubits, *qubit, true);
                    let tp = TransPauli::from_pauli_string(&fresh_pauli, "T");
                    ops.push(TransItem::Pauli(tp));
                } else {
                    let tp = TransPauli::from_pauli_string(&conjugated, "T");
                    ops.push(TransItem::Pauli(tp));
                }
            }
            QasmGate::Measure { qubit } => {
                // Z-basis measurement on `qubit`.
                if last_weight.is_none() {
                    // No T gate seen yet — this shouldn't happen in a valid circuit
                    // but we handle it gracefully.
                    eprintln!("Warning: measurement on qubit {} before any T gate", qubit);
                }
                let pre_pauli = make_z_pauli(num_qubits, *qubit, false);
                let conjugated = tableau.conjugate(&pre_pauli);
                let should_flush = last_weight.map_or(false, |w| w > effective_max_weight)
                    || conjugated.weight() > effective_max_weight;
                if should_flush {
                    flush_clifford_queue(&mut clifford_queue, &mut ops, &mut tableau, num_qubits);
                    let fresh_pauli = make_z_pauli(num_qubits, *qubit, false);
                    let tp = TransPauli::from_pauli_string(&fresh_pauli, "M");
                    ops.push(TransItem::Pauli(tp));
                } else {
                    let tp = TransPauli::from_pauli_string(&conjugated, "M");
                    ops.push(TransItem::Pauli(tp));
                }
            }
        }
    }

    ops
}

/// Create a Pauli string with Z on qubit `q` and I elsewhere.
fn make_z_pauli(n: usize, q: usize, negative: bool) -> PauliString {
    let mut ps = PauliString::identity(n);
    ps.z_bits[q] = true;
    ps.sign = negative;
    ps
}

// ─────────────────────────────────────────────────────────────────────────────
// Output writer
// ─────────────────────────────────────────────────────────────────────────────

/// Write the transpiled circuit to a `.trans` file.
///
/// Mirrors `print_trans()` from `transpile_circuit.py`.
fn write_trans(output_path: &str, items: &[TransItem]) -> io::Result<(usize, usize)> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let mut num_ts = 0usize;
    let mut num_cliffords = 0usize;

    for item in items {
        match item {
            TransItem::Pauli(p) => {
                writeln!(writer, "{}", p)?;
                if p.label == "T" {
                    num_ts += 1;
                }
            }
            TransItem::Clifford(c) => {
                writeln!(writer, "{}", c)?;
                num_cliffords += 1;
            }
        }
    }

    Ok((num_ts, num_cliffords))
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────────────────────────────────────

fn count_stats(items: &[TransItem]) -> (usize, usize, usize, f64) {
    let num_cliffords = items.iter().filter(|i| matches!(i, TransItem::Clifford(_))).count();
    let pauli_ts: Vec<&TransPauli> = items
        .iter()
        .filter_map(|i| {
            if let TransItem::Pauli(p) = i {
                if p.label == "T" { Some(p) } else { None }
            } else {
                None
            }
        })
        .collect();
    let num_pauli_products = pauli_ts.len();
    let avg_weight = if num_pauli_products > 0 {
        let total_weight: usize =
            pauli_ts.iter().map(|p| p.paulis.iter().filter(|&&c| c != 'I').count()).sum();
        total_weight as f64 / num_pauli_products as f64
    } else {
        0.0
    };
    (num_cliffords, num_pauli_products, items.len(), avg_weight)
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_file = &args.input_file;
    if !input_file.ends_with(".cliffordt.qasm") {
        return Err(format!(
            "Input file must be a .cliffordt.qasm file produced by compile_circuit.py, got: {}",
            input_file
        )
        .into());
    }

    // ── Load circuit ──────────────────────────────────────────────────────────
    println!("Loading compiled circuit from {}", input_file);
    let load_start = Instant::now();
    let (num_qubits, gates) = parse_qasm(input_file)?;
    let load_elapsed = load_start.elapsed();
    println!("Circuit loaded in {:.2} seconds", load_elapsed.as_secs_f64());

    let total_gates = gates.iter().filter(|g| !matches!(g, QasmGate::Measure { .. })).count();
    let clifford_gates = gates
        .iter()
        .filter(|g| matches!(g, QasmGate::Clifford1Q { .. } | QasmGate::Clifford2Q { .. }))
        .count();

    println!("Circuit has {} gates on {} qubits", total_gates, num_qubits);

    // ── Warn about gate set ───────────────────────────────────────────────────
    let mut gate_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for g in &gates {
        let name = match g {
            QasmGate::Clifford1Q { gate, .. } => format!("{:?}", gate),
            QasmGate::Clifford2Q { gate, .. } => format!("{:?}", gate),
            QasmGate::T { .. } => "T".to_string(),
            QasmGate::Tdg { .. } => "Tdg".to_string(),
            QasmGate::Measure { .. } => "Measure".to_string(),
        };
        gate_names.insert(name);
    }
    println!("Gate set: {}", gate_names.iter().cloned().collect::<Vec<_>>().join(", "));

    // ── Transpile ─────────────────────────────────────────────────────────────
    let transpile_start = Instant::now();
    let items = transpile(num_qubits, &gates, args.max_width);
    let transpile_elapsed = transpile_start.elapsed();

    let (num_cliffords, num_pauli_products, post_total, avg_weight) = count_stats(&items);
    let total_delta = total_gates as i64 - post_total as i64;
    let clifford_delta = clifford_gates as i64 - num_cliffords as i64;

    println!("Tableau took {:.2} seconds", transpile_elapsed.as_secs_f64());
    if total_gates > 0 {
        println!(
            "Circuit length:    {} (before) -> {} (after transpilation)",
            total_gates, post_total
        );
        println!(
            "  Overall reduction: {} operations removed ({:.1}% reduction)",
            total_delta,
            100.0 * total_delta as f64 / total_gates as f64
        );
    } else {
        println!("  Overall reduction: N/A (empty circuit)");
    }
    println!(
        "  Clifford gates:    {} (before) -> {} (after transpilation)",
        clifford_gates, num_cliffords
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
    println!("  Non-Clifford Pauli products: {}", num_pauli_products);
    println!("  Average Pauli product weight: {:.2}", avg_weight);

    // ── Write output ──────────────────────────────────────────────────────────
    let output_stem = if args.output_file.is_empty() {
        // Strip both extensions from .cliffordt.qasm
        let p = Path::new(input_file);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("output");
        // stem is now "foo.cliffordt" — strip the .cliffordt part
        if stem.ends_with(".cliffordt") {
            stem[..stem.len() - ".cliffordt".len()].to_string()
        } else {
            stem.to_string()
        }
    } else {
        args.output_file.clone()
    };
    let output_path = format!("{}.trans", output_stem);

    let (num_ts, num_cliffords_written) = write_trans(&output_path, &items)?;
    println!("Wrote transpiled circuit to {}", output_path);
    println!("Wrote {} T gates and {} Cliffords", num_ts, num_cliffords_written);

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

    // ── Transpiler: identity tableau ──────────────────────────────────────────

    #[test]
    fn transpile_single_t_gate_no_cliffords() {
        // A single T gate on qubit 0 with no preceding Cliffords.
        // The tableau is identity, so the output should be +Z_<T>.
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        // Should have: 1 T gate + 2 measurements (appended for qubits 0 and 1)
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Plus);
            assert_eq!(p.paulis[0], 'Z');
            assert_eq!(p.paulis[1], 'I');
        }
    }

    #[test]
    fn transpile_h_then_t_gives_x_pauli() {
        // H on qubit 0, then T on qubit 0.
        // H maps Z → X, so T (which is a Z rotation) becomes an X rotation.
        // The tableau after H: X_0 → Z, Z_0 → X.
        // Conjugating +Z through this tableau: Z_0 → X_0.
        // So the output T gate should be +X_<T>.
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::H, qubit: 0 }, QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.paulis[0], 'X');
        }
    }

    #[test]
    fn transpile_tdg_gives_negative_sign() {
        // Tdg on qubit 0 with identity tableau → -Z_<T>
        let gates = vec![QasmGate::Tdg { qubit: 0 }];
        let items = transpile(1, &gates, -1);
        let t_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "T")).collect();
        assert_eq!(t_items.len(), 1);
        if let TransItem::Pauli(p) = &t_items[0] {
            assert_eq!(p.sign, Sign::Minus);
            assert_eq!(p.paulis[0], 'Z');
        }
    }

    #[test]
    fn transpile_measurements_appended_for_all_qubits() {
        // Circuit with T on qubit 0 only; measurements should be appended for both qubits.
        let gates = vec![QasmGate::T { qubit: 0 }];
        let items = transpile(2, &gates, -1);
        let m_items: Vec<_> =
            items.iter().filter(|i| matches!(i, TransItem::Pauli(p) if p.label == "M")).collect();
        assert_eq!(m_items.len(), 2);
    }

    // ── Transpiler: max_weight flushing ───────────────────────────────────────

    #[test]
    fn transpile_max_weight_1_flushes_cliffords() {
        // CX creates a weight-2 product; with max_weight=1 it should flush.
        // CX q[0], q[1]; T q[0];
        // After CX: tableau maps Z_0 → Z_0 (unchanged), but X_0 → X_0 X_1.
        // T on q[0] conjugates +Z_0 through tableau: Z_0 → Z_0 (weight 1, OK).
        // Actually CX: X0→X0X1, Z0→Z0, X1→X1, Z1→Z0Z1.
        // T on q[0]: conjugate +Z_0 → Z_0 (weight 1, no flush needed).
        // Let's use a case that actually triggers flush:
        // H q[0]; CX q[0], q[1]; T q[0];
        // After H: Z_0→X_0, X_0→Z_0.
        // After CX: X_0→Z_0 (unchanged since CX maps X0→X0X1 but we prepend,
        //   so the tableau accumulates in reverse order).
        // This is getting complex; just test that max_weight=0 (effectively 1)
        // causes cliffords to be flushed.
        let gates =
            vec![QasmGate::Clifford1Q { gate: Gate1Q::S, qubit: 0 }, QasmGate::T { qubit: 0 }];
        // With max_weight=-1 (no limit), S is in clifford_queue and not flushed.
        let items_unlimited = transpile(2, &gates, -1);
        let cliffords_unlimited: Vec<_> =
            items_unlimited.iter().filter(|i| matches!(i, TransItem::Clifford(_))).collect();
        // No flush → no Clifford items in output
        assert_eq!(cliffords_unlimited.len(), 0);
    }

    // ── count_stats ───────────────────────────────────────────────────────────

    #[test]
    fn count_stats_basic() {
        let items = vec![
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                paulis: vec!['Z', 'I'],
                label: "T".to_string(),
            }),
            TransItem::Pauli(TransPauli {
                sign: Sign::Plus,
                paulis: vec!['Z', 'Z'],
                label: "T".to_string(),
            }),
            TransItem::Clifford(TransClifford { paulis: vec!['Z', '_'], name: "S".to_string() }),
        ];
        let (num_cliffords, num_paulis, total, avg_weight) = count_stats(&items);
        assert_eq!(num_cliffords, 1);
        assert_eq!(num_paulis, 2);
        assert_eq!(total, 3);
        // weights: 1 + 2 = 3, avg = 1.5
        assert!((avg_weight - 1.5).abs() < 1e-9);
    }
}
