//! Stage 1: compose consecutive single-qubit gates on each qubit into one
//! `Block` operation, bounded by any multi-qubit gate touching that qubit.
//!
//! Mirrors bqskit's `GroupSingleQuditGatePass`. Assumes the input circuit's
//! single-qubit gates are already drawn from this pipeline's own `Gate`
//! vocabulary (Clifford generators + `Rz`) -- any prior decomposition from
//! an arbitrary front-end gate set into that vocabulary is a precondition
//! of this pipeline, not one of its six stages.

use std::collections::HashMap;

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

pub fn group_single_qubit_gates(circuit: &Circuit) -> Circuit {
    let mut out = Circuit::new(circuit.n_qubits);
    let mut buffers: HashMap<usize, Circuit> = HashMap::new();

    for op in &circuit.ops {
        if op.qubits.len() == 1 {
            let q = op.qubits[0];
            let buf = buffers.entry(q).or_insert_with(|| Circuit::new(1));
            buf.push(op.gate.clone(), vec![0]);
        } else {
            for &q in &op.qubits {
                flush(&mut out, &mut buffers, q);
            }
            out.ops.push(op.clone());
        }
    }
    for q in 0..circuit.n_qubits {
        flush(&mut out, &mut buffers, q);
    }
    out
}

fn flush(out: &mut Circuit, buffers: &mut HashMap<usize, Circuit>, q: usize) {
    if let Some(buf) = buffers.remove(&q) {
        if !buf.ops.is_empty() {
            out.push(Gate::Block(Box::new(buf)), vec![q]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    #[test]
    fn single_qubit_run_becomes_one_block() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![0]);
        c.push(Gate::S, vec![0]);
        let grouped = group_single_qubit_gates(&c);
        assert_eq!(grouped.ops.len(), 1);
        assert!(matches!(grouped.ops[0].gate, Gate::Block(_)));
        // Composed action must be unchanged.
        assert!(distance(&c.get_unitary(), &grouped.get_unitary()) < 1e-12);
    }

    #[test]
    fn two_qubit_gate_splits_runs_into_separate_blocks() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![1]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::S, vec![0]);
        c.push(Gate::X, vec![1]);
        let grouped = group_single_qubit_gates(&c);
        // block(q0: H), block(q1: T), CX, block(q0: S), block(q1: X)
        assert_eq!(grouped.ops.len(), 5);
        assert!(matches!(grouped.ops[0].gate, Gate::Block(_)));
        assert!(matches!(grouped.ops[1].gate, Gate::Block(_)));
        assert!(matches!(grouped.ops[2].gate, Gate::Cx));
        assert!(matches!(grouped.ops[3].gate, Gate::Block(_)));
        assert!(matches!(grouped.ops[4].gate, Gate::Block(_)));
        assert!(distance(&c.get_unitary(), &grouped.get_unitary()) < 1e-12);
    }

    #[test]
    fn unfold_after_group_recovers_original_flat_gates() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![1]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::S, vec![0]);
        let grouped = group_single_qubit_gates(&c);
        let unfolded = grouped.unfold();
        assert_eq!(unfolded.ops.len(), 4);
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-12);
    }

    #[test]
    fn qubit_with_no_single_qubit_gates_produces_no_empty_block() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![0, 1]);
        let grouped = group_single_qubit_gates(&c);
        assert_eq!(grouped.ops.len(), 1);
        assert!(matches!(grouped.ops[0].gate, Gate::Cx));
    }
}
