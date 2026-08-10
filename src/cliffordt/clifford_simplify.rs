//! Stage 4: canonicalize every maximal run of Clifford-only gates that
//! sits between T/Tdg gates (or circuit/multi-qubit-gate boundaries)
//! anywhere in the final compiled circuit.
//!
//! `gauge_collapse` (Stage 1) only rewrites a whole single-qubit block when
//! its *entire* composed unitary is exactly Clifford -- it never fires on a
//! synthesized block, since those genuinely contain T gates. This pass
//! instead looks at the finer-grained runs *between* T gates, each of which
//! is still guaranteed to live on a single qubit and compose to an exact
//! Clifford element, so the same `CliffordTable` lookup gauge_collapse
//! already uses is sufficient here too -- no stabilizer/tableau formalism
//! needed, since nothing here ever needs to commute a gate across qubits.

use crate::cliffordt::clifford::{CliffordTable, circuit_from_word};
use crate::cliffordt::qgate_circuit::{Circuit, Gate, Operation};

/// Replace `buffer`'s gates (all on `qubit`) with their canonical shortest
/// word if one exists and is *strictly* shorter than what's already there,
/// then clear it. A no-op on an empty buffer.
fn flush(
    buffer: &mut Vec<Gate>, qubit: usize, table: &CliffordTable, tol: f64, out: &mut Vec<Operation>,
) {
    if buffer.is_empty() {
        return;
    }
    let target = circuit_from_word(buffer).get_unitary();
    // Every gate in `buffer` is itself a Clifford generator, so this is
    // guaranteed to hit one of the table's 24 entries barring pathological
    // floating-point edge cases; `None` falls back to the original gates
    // unchanged rather than panicking. Strictly-shorter (not `<=`) matters:
    // swapping in a same-length-but-different word buys nothing in gate
    // count and isn't a bit-identical reconstruction of the original, so it
    // would introduce ~1e-16 floating-point churn on runs that were already
    // optimal for zero benefit.
    let replacement = match table.exact_match(&target, tol) {
        Some(word) if word.len() < buffer.len() => word,
        _ => buffer.clone(),
    };
    for gate in replacement {
        out.push(Operation { gate, qubits: vec![qubit] });
    }
    buffer.clear();
}

/// See module doc comment.
pub fn simplify_clifford_runs(circuit: &Circuit, table: &CliffordTable, tol: f64) -> Circuit {
    let mut buffers: Vec<Vec<Gate>> = vec![Vec::new(); circuit.n_qubits];
    let mut out = Vec::with_capacity(circuit.ops.len());

    for op in &circuit.ops {
        if op.qubits.len() == 1 && op.gate.is_clifford() {
            buffers[op.qubits[0]].push(op.gate.clone());
        } else {
            for &q in &op.qubits {
                flush(&mut buffers[q], q, table, tol, &mut out);
            }
            out.push(op.clone());
        }
    }
    for (q, buffer) in buffers.iter_mut().enumerate() {
        flush(buffer, q, table, tol, &mut out);
    }

    Circuit { n_qubits: circuit.n_qubits, ops: out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    fn push_all(circuit: &mut Circuit, gates: &[(Gate, &[usize])]) {
        for (gate, qubits) in gates {
            circuit.push(gate.clone(), qubits.to_vec());
        }
    }

    #[test]
    fn identity_producing_run_is_dropped() {
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        push_all(&mut c, &[(Gate::H, &[0]), (Gate::H, &[0])]);
        let simplified = simplify_clifford_runs(&c, &table, 1e-10);
        assert!(simplified.ops.is_empty());
    }

    #[test]
    fn s_to_the_fourth_collapses_to_nothing() {
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        push_all(&mut c, &[(Gate::S, &[0]), (Gate::S, &[0]), (Gate::S, &[0]), (Gate::S, &[0])]);
        let simplified = simplify_clifford_runs(&c, &table, 1e-10);
        assert!(simplified.ops.is_empty());
    }

    #[test]
    fn clifford_run_between_t_gates_is_shortened_without_changing_unitary() {
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        // A deliberately redundant Clifford prefix (H,H,S,S,S,S = identity)
        // in front of a T, then a redundant suffix after it.
        push_all(
            &mut c,
            &[
                (Gate::H, &[0]),
                (Gate::H, &[0]),
                (Gate::S, &[0]),
                (Gate::S, &[0]),
                (Gate::S, &[0]),
                (Gate::S, &[0]),
                (Gate::T, &[0]),
                (Gate::X, &[0]),
                (Gate::X, &[0]),
                (Gate::H, &[0]),
            ],
        );
        let simplified = simplify_clifford_runs(&c, &table, 1e-10);
        assert!(simplified.ops.len() < c.ops.len());
        assert!(distance(&c.get_unitary(), &simplified.get_unitary()) < 1e-9);
    }

    #[test]
    fn t_and_tdg_gates_are_never_touched() {
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        push_all(
            &mut c,
            &[
                (Gate::H, &[0]),
                (Gate::H, &[0]),
                (Gate::T, &[0]),
                (Gate::S, &[0]),
                (Gate::Sdg, &[0]),
                (Gate::Tdg, &[0]),
                (Gate::X, &[0]),
            ],
        );
        let simplified = simplify_clifford_runs(&c, &table, 1e-10);
        let t_gates: Vec<&Gate> = simplified
            .ops
            .iter()
            .map(|op| &op.gate)
            .filter(|g| matches!(g, Gate::T | Gate::Tdg))
            .collect();
        assert_eq!(t_gates, vec![&Gate::T, &Gate::Tdg]);
    }

    #[test]
    fn multi_qubit_gate_boundaries_are_respected() {
        let table = CliffordTable::build();
        let mut c = Circuit::new(2);
        push_all(
            &mut c,
            &[
                (Gate::H, &[0]),
                (Gate::H, &[0]), // qubit 0: identity padding before the Cx
                (Gate::S, &[1]),
                (Gate::S, &[1]),
                (Gate::S, &[1]),
                (Gate::S, &[1]), // qubit 1: identity padding before the Cx
                (Gate::Cx, &[0, 1]),
                (Gate::X, &[0]),
                (Gate::X, &[0]), // qubit 0: identity padding after the Cx
            ],
        );
        let simplified = simplify_clifford_runs(&c, &table, 1e-10);
        assert_eq!(
            simplified.ops.iter().map(|op| op.gate.clone()).collect::<Vec<_>>(),
            vec![Gate::Cx]
        );
        assert!(distance(&c.get_unitary(), &simplified.get_unitary()) < 1e-9);
    }

    #[test]
    fn never_increases_gate_count() {
        let table = CliffordTable::build();
        let samples: Vec<Circuit> = vec![
            {
                let mut c = Circuit::new(1);
                push_all(&mut c, &[(Gate::H, &[0]), (Gate::T, &[0]), (Gate::S, &[0])]);
                c
            },
            {
                let mut c = Circuit::new(1);
                push_all(
                    &mut c,
                    &[(Gate::X, &[0]), (Gate::Y, &[0]), (Gate::Z, &[0]), (Gate::T, &[0])],
                );
                c
            },
        ];
        for sample in samples {
            let simplified = simplify_clifford_runs(&sample, &table, 1e-10);
            assert!(simplified.ops.len() <= sample.ops.len());
            assert!(distance(&sample.get_unitary(), &simplified.get_unitary()) < 1e-9);
        }
    }
}
