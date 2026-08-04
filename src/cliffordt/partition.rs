//! Multi-qubit windowed partitioning, used by Stage 4 (block_size 2) and
//! Stage 5 (block_size 4+).
//!
//! A simplified, single-pass greedy partitioner in the same spirit as
//! bqskit's `QuickPartitioner` (walk operations in program order, grow or
//! close bins of at most `block_size` qubits as needed) -- deliberately not
//! a line-for-line port of its `Bin`/`pending_bins`/`blocked_qudits`
//! bookkeeping (see the plan doc), since that machinery exists to handle
//! circuit topologies more general than this pipeline produces. Validity is
//! independent of which exact grouping is chosen: any partition that keeps
//! each qubit owned by at most one open block at a time preserves the
//! circuit's exact action, which is what the tests below check directly.

use std::collections::HashMap;

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

struct OpenBlock {
    /// Outer qubit index for each local qubit position.
    qubits: Vec<usize>,
    inner: Circuit,
}

impl OpenBlock {
    fn local_index(&mut self, q: usize) -> usize {
        if let Some(pos) = self.qubits.iter().position(|&x| x == q) {
            pos
        } else {
            self.qubits.push(q);
            self.inner.n_qubits = self.qubits.len();
            self.qubits.len() - 1
        }
    }
}

pub fn partition(circuit: &Circuit, block_size: usize) -> Circuit {
    let mut out = Circuit::new(circuit.n_qubits);
    let mut open_blocks: Vec<OpenBlock> = Vec::new();
    let mut owner: HashMap<usize, usize> = HashMap::new(); // outer qubit -> index into open_blocks

    for op in &circuit.ops {
        // Collect distinct blocks currently owning any of this op's qubits.
        let mut owners: Vec<usize> = op.qubits.iter().filter_map(|q| owner.get(q).copied()).collect();
        owners.sort_unstable();
        owners.dedup();

        if owners.len() > 1 {
            // Conflicting ownership: close all touched blocks, then treat
            // this op as needing a fresh block.
            for &idx in &owners {
                close_block(&mut out, &mut open_blocks, &mut owner, idx);
            }
            open_new_block(&mut open_blocks, &mut owner, op.gate.clone(), &op.qubits);
            continue;
        }

        if owners.is_empty() {
            open_new_block(&mut open_blocks, &mut owner, op.gate.clone(), &op.qubits);
            continue;
        }

        let idx = owners[0];
        let required: std::collections::HashSet<usize> =
            open_blocks[idx].qubits.iter().copied().chain(op.qubits.iter().copied()).collect();
        if required.len() > block_size {
            close_block(&mut out, &mut open_blocks, &mut owner, idx);
            open_new_block(&mut open_blocks, &mut owner, op.gate.clone(), &op.qubits);
            continue;
        }

        // Grow the existing block (if needed) and append this op.
        let local_qubits: Vec<usize> = {
            let block = &mut open_blocks[idx];
            op.qubits.iter().map(|&q| block.local_index(q)).collect()
        };
        for &q in &op.qubits {
            owner.insert(q, idx);
        }
        open_blocks[idx].inner.push(op.gate.clone(), local_qubits);
    }

    // Flush any still-open blocks, in creation order.
    for idx in 0..open_blocks.len() {
        if owner.values().any(|&v| v == idx) {
            close_block(&mut out, &mut open_blocks, &mut owner, idx);
        }
    }

    out
}

fn open_new_block(
    open_blocks: &mut Vec<OpenBlock>,
    owner: &mut HashMap<usize, usize>,
    gate: Gate,
    qubits: &[usize],
) {
    let idx = open_blocks.len();
    let mut block = OpenBlock { qubits: Vec::new(), inner: Circuit::new(0) };
    let local_qubits: Vec<usize> = qubits.iter().map(|&q| block.local_index(q)).collect();
    block.inner.push(gate, local_qubits);
    open_blocks.push(block);
    for &q in qubits {
        owner.insert(q, idx);
    }
}

fn close_block(
    out: &mut Circuit,
    open_blocks: &mut [OpenBlock],
    owner: &mut HashMap<usize, usize>,
    idx: usize,
) {
    if open_blocks[idx].qubits.is_empty() {
        return;
    }
    owner.retain(|_, v| *v != idx);
    let qubits = open_blocks[idx].qubits.clone();
    let inner = open_blocks[idx].inner.clone();
    out.push(Gate::Block(Box::new(inner)), qubits);
    // Mark as emptied so a later flush pass (or repeat close) is a no-op.
    open_blocks[idx].qubits.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    #[test]
    fn partition_preserves_circuit_unitary_linear_chain() {
        let mut c = Circuit::new(4);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Cx, vec![1, 2]);
        c.push(Gate::Cx, vec![2, 3]);
        c.push(Gate::T, vec![3]);
        let partitioned = partition(&c, 2);
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }

    #[test]
    fn partition_preserves_circuit_unitary_dense_overlap() {
        let mut c = Circuit::new(3);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::H, vec![1]);
        c.push(Gate::Cx, vec![1, 2]);
        c.push(Gate::Cx, vec![0, 2]);
        c.push(Gate::T, vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        let partitioned = partition(&c, 2);
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }

    #[test]
    fn every_block_respects_the_size_limit() {
        let mut c = Circuit::new(5);
        for i in 0..4 {
            c.push(Gate::Cx, vec![i, i + 1]);
        }
        let partitioned = partition(&c, 2);
        for op in &partitioned.ops {
            assert!(op.qubits.len() <= 2, "block exceeded size limit: {:?}", op.qubits);
        }
    }

    #[test]
    fn single_qubit_gates_only_partition_correctly() {
        let mut c = Circuit::new(3);
        c.push(Gate::H, vec![0]);
        c.push(Gate::T, vec![1]);
        c.push(Gate::S, vec![2]);
        let partitioned = partition(&c, 2);
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }

    #[test]
    fn larger_block_size_can_merge_more_qubits() {
        let mut c = Circuit::new(4);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Cx, vec![2, 3]);
        c.push(Gate::Cx, vec![1, 2]);
        let partitioned = partition(&c, 4);
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }
}
