//! Circuit statistics for reporting, matching the depth/T-depth model
//! established for `compile_cliffordt.py`: per-qubit running depth
//! counters, each gate taking the max over its touched qubits (+1 always
//! for `depth`, +1 only when the gate is T/Tdg for `t_depth`), then
//! writing that value back to every touched qubit regardless.

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

pub struct Stats {
    pub qubits: usize,
    pub gates: usize,
    pub depth: usize,
    pub t_count: usize,
    pub t_depth: usize,
    pub cx_count: usize,
    pub clifford_count: usize,
}

pub fn compute_stats(circuit: &Circuit) -> Stats {
    let n = circuit.n_qubits;
    let mut depth_counters = vec![0usize; n];
    let mut t_depth_counters = vec![0usize; n];
    let mut t_count = 0usize;
    let mut cx_count = 0usize;
    let mut clifford_count = 0usize;

    for op in &circuit.ops {
        let is_t = matches!(op.gate, Gate::T | Gate::Tdg);
        if is_t {
            t_count += 1;
        }
        if matches!(op.gate, Gate::Cx) {
            cx_count += 1;
        }
        if op.gate.is_clifford() {
            clifford_count += 1;
        }

        let max_depth = op.qubits.iter().map(|&q| depth_counters[q]).max().unwrap_or(0);
        let new_depth = max_depth + 1;
        for &q in &op.qubits {
            depth_counters[q] = new_depth;
        }

        let max_t_depth = op.qubits.iter().map(|&q| t_depth_counters[q]).max().unwrap_or(0);
        let new_t_depth = max_t_depth + if is_t { 1 } else { 0 };
        for &q in &op.qubits {
            t_depth_counters[q] = new_t_depth;
        }
    }

    Stats {
        qubits: n,
        gates: circuit.ops.len(),
        depth: depth_counters.into_iter().max().unwrap_or(0),
        t_count,
        t_depth: t_depth_counters.into_iter().max().unwrap_or(0),
        cx_count,
        clifford_count,
    }
}

/// (number of Stage 1 `Block`s formed, total single-qubit gates grouped
/// inside them). Blocking's own contribution is invisible to gate/Rz
/// counts -- it only repackages gates into `Block`s, never removes any
/// (`total_gate_count` recurses into `Block`, so the total is unchanged
/// by construction) -- so this is the metric that actually reflects what
/// the stage did, rather than a delta that's always zero.
pub fn block_stats(circuit: &Circuit) -> (usize, usize) {
    let mut num_blocks = 0;
    let mut grouped_gates = 0;
    for op in &circuit.ops {
        if let Gate::Block(inner) = &op.gate {
            num_blocks += 1;
            grouped_gates += inner.ops.len();
        }
    }
    (num_blocks, grouped_gates)
}

/// Gate kinds that shouldn't survive to a finished Clifford+T circuit --
/// anything counted here means an earlier stage left work undone (a stray
/// `Rz`/`U3` that never reached Stage 6, or a `Block` that was never
/// unfolded), not a legitimate part of the result. Mirrors
/// `non_basis_ops` in `compile_cliffordt.py`, recursing into any leftover
/// `Block` defensively (the pipeline always unfolds before returning, so
/// this should never actually fire in practice).
pub fn non_basis_ops(circuit: &Circuit) -> Vec<(&'static str, usize)> {
    fn gate_name(gate: &Gate) -> Option<&'static str> {
        match gate {
            Gate::Rz(_) => Some("rz"),
            Gate::U3(..) => Some("u"),
            Gate::Block(_) => Some("block"),
            _ => None,
        }
    }

    fn walk(circuit: &Circuit, counts: &mut Vec<(&'static str, usize)>) {
        for op in &circuit.ops {
            if let Gate::Block(inner) = &op.gate {
                walk(inner, counts);
            }
            if let Some(name) = gate_name(&op.gate) {
                match counts.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, n)) => *n += 1,
                    None => counts.push((name, 1)),
                }
            }
        }
    }

    let mut counts = Vec::new();
    walk(circuit, &mut counts);
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_stats_counts_blocks_and_their_grouped_gates() {
        let mut inner_a = Circuit::new(1);
        inner_a.push(Gate::H, vec![0]);
        inner_a.push(Gate::T, vec![0]);
        let mut inner_b = Circuit::new(1);
        inner_b.push(Gate::S, vec![0]);
        let mut c = Circuit::new(2);
        c.push(Gate::Block(Box::new(inner_a)), vec![0]);
        c.push(Gate::Block(Box::new(inner_b)), vec![1]);
        c.push(Gate::Cx, vec![0, 1]);
        assert_eq!(block_stats(&c), (2, 3));
    }

    #[test]
    fn block_stats_is_zero_for_a_circuit_with_no_blocks() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        assert_eq!(block_stats(&c), (0, 0));
    }

    #[test]
    fn non_basis_ops_is_empty_for_a_pure_clifford_t_circuit() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        assert!(non_basis_ops(&c).is_empty());
    }

    #[test]
    fn non_basis_ops_flags_a_leftover_rz() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.3), vec![0]);
        assert_eq!(non_basis_ops(&c), vec![("rz", 1)]);
    }

    #[test]
    fn non_basis_ops_recurses_into_a_leftover_block() {
        let mut inner = Circuit::new(1);
        inner.push(Gate::Rz(0.1), vec![0]);
        let mut c = Circuit::new(1);
        c.push(Gate::Block(Box::new(inner)), vec![0]);
        let found = non_basis_ops(&c);
        assert!(found.contains(&("block", 1)));
        assert!(found.contains(&("rz", 1)));
    }

    #[test]
    fn sequential_single_qubit_gates_have_depth_equal_to_gate_count() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![0]);
        c.push(Gate::T, vec![0]);
        let s = compute_stats(&c);
        assert_eq!(s.depth, 3);
        assert_eq!(s.t_count, 2);
        assert_eq!(s.t_depth, 2);
    }

    #[test]
    fn parallel_gates_on_different_qubits_share_a_layer() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![1]);
        let s = compute_stats(&c);
        assert_eq!(s.depth, 1);
        assert_eq!(s.t_depth, 1);
    }

    #[test]
    fn non_t_gates_do_not_advance_t_depth() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::S, vec![0]);
        c.push(Gate::X, vec![0]);
        let s = compute_stats(&c);
        assert_eq!(s.depth, 3);
        assert_eq!(s.t_depth, 0);
    }

    #[test]
    fn cx_count_only_counts_literal_cx_gates() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Cz, vec![0, 1]);
        let s = compute_stats(&c);
        assert_eq!(s.cx_count, 1);
    }
}
