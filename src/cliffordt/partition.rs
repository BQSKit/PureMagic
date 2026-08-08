//! Multi-qubit windowed partitioning, used by Stage 2 (block_size 2) and
//! Stage 3 (block_size 4+).
//!
//! A faithful port of bqskit's `QuickPartitioner`: a single pass over
//! operations in topological ("cycle") order, growing "bins" of qubits up
//! to `block_size`, but with two mechanisms a naive greedy pass lacks:
//!
//! - **Per-qubit partial closing + `blocked` qubits**: a bin can lose just
//!   one of its qubits to a competing bin while staying open on the rest,
//!   and once two bins have interacted this way, each is permanently
//!   forbidden from later reclaiming any qubit the other now owns -- this
//!   is what prevents an inconsistent (cyclic) ordering between blocks,
//!   not a size heuristic.
//! - **Deferred placement (`pending`/`dividing_line`) + retroactive
//!   merging**: a bin that's fully closed doesn't get written to the
//!   output immediately -- it waits until nothing earlier remains
//!   unplaced on any of its qubits, and when it *is* placed, it first
//!   merges with whatever's currently at the output's "rear" if the two
//!   blocks' qubit sets are in a subset/superset relation. This is what
//!   lets several separately-detected bins combine into one genuinely
//!   wide window.
//!
//! An earlier, simpler version of this module (single-pass, close-on-
//! overflow, no partial closing, no merging) preserved circuit correctness
//! but produced far smaller/more fragmented windows than bqskit's real
//! algorithm -- confirmed to starve a wide-window joint optimizer of the
//! multi-Rz windows it needs (`0/52 blocks improved` on a real circuit).
//! This module fixes that.
//!
//! bqskit's `BarrierBin` (a dedicated case for barrier/measurement/reset
//! gates) has no equivalent here: this pipeline's `Circuit` model has no
//! such gate at all (`qasm.rs`'s loader drops those QASM lines entirely
//! rather than representing them as ops), so that whole case is omitted.
//!
//! bqskit's "cycle" is just the earliest topological layer at which all of
//! an op's qubits are simultaneously free -- the same quantity
//! `stats.rs`'s `compute_stats` computes per-op via per-qubit running
//! depth counters. Computed once up front here (`assign_cycles`) rather
//! than needing a grid-based `Circuit` structure.

use std::collections::{HashMap, HashSet};

use crate::cliffordt::qgate_circuit::{Circuit, Gate, Operation};

/// A partition candidate under construction.
struct Bin {
    /// All qubits ever added to this bin -- permanent, never shrinks.
    qubits: Vec<usize>,
    /// Subset of `qubits` still open to accept new ops.
    active: HashSet<usize>,
    /// Cycle each qubit first joined this bin.
    starts: HashMap<usize, usize>,
    /// Cycle each qubit was deactivated in this bin (`None` while active).
    ends: HashMap<usize, Option<usize>>,
    /// Qubits this bin is permanently forbidden from claiming, because it
    /// has already interacted with another bin that currently owns (or is
    /// itself blocked on) them -- see the module doc.
    blocked: HashSet<usize>,
    /// Ops assigned to this bin, in program order, still globally indexed
    /// (remapped to local indices only once finally emitted).
    ops: Vec<Operation>,
}

impl Bin {
    fn new() -> Self {
        Bin {
            qubits: Vec::new(),
            active: HashSet::new(),
            starts: HashMap::new(),
            ends: HashMap::new(),
            blocked: HashSet::new(),
            ops: Vec::new(),
        }
    }

    fn add_op(&mut self, op: &Operation, cycle: usize) {
        for &q in &op.qubits {
            if !self.qubits.contains(&q) {
                self.qubits.push(q);
                self.active.insert(q);
                self.starts.insert(q, cycle);
                self.ends.insert(q, None);
            }
        }
        self.ops.push(op.clone());
    }

    /// Mirrors `Bin.can_accommodate`: not blocked (unless already ours and
    /// active), not reaching back into a qubit we've already closed, and
    /// not growing past `max(block_size, our current size)`.
    fn can_accommodate(&self, qubits: &[usize], block_size: usize) -> bool {
        if qubits.iter().any(|q| self.blocked.contains(q) && !self.active.contains(q)) {
            return false;
        }
        let overlapping_active =
            qubits.iter().all(|q| !self.qubits.contains(q) || self.active.contains(q));
        if !overlapping_active {
            return false;
        }
        let size_limit = block_size.max(self.qubits.len());
        let mut union: HashSet<usize> = self.qubits.iter().copied().collect();
        union.extend(qubits.iter().copied());
        union.len() <= size_limit
    }
}

/// Deactivate `qubits` in `bin` (recording their closing cycle), clearing
/// the global `active_bin_of` slot for any qubit this bin loses. Returns
/// true if the bin has no active qubits left afterward.
fn close_bin_qubits(
    bin: &mut Bin, bin_idx: usize, qubits: &[usize], cycle: usize,
    active_bin_of: &mut [Option<usize>],
) -> bool {
    for &q in qubits {
        if bin.active.remove(&q) {
            bin.ends.insert(q, Some(cycle.saturating_sub(1)));
        }
        if active_bin_of[q] == Some(bin_idx) {
            active_bin_of[q] = None;
        }
    }
    bin.active.is_empty()
}

/// A bin that has been fully placed into the output (or tombstoned by a
/// later retroactive merge into a subsequent block).
struct PlacedBlock {
    qubits: Vec<usize>,  // sorted
    ops: Vec<Operation>, // still globally-qubit-indexed
}

/// Per-op cycle assignment (earliest layer at which all its qubits are
/// simultaneously free) plus each qubit's first-touched cycle -- mirrors
/// bqskit's `circuit._front[q].cycle`, needed to seed `dividing_line`
/// correctly (a qubit's first real operation isn't necessarily at cycle 0
/// if it's first touched by a multi-qubit gate shared with an
/// already-busy qubit).
fn assign_cycles(circuit: &Circuit) -> (Vec<usize>, Vec<usize>, usize) {
    let n = circuit.n_qubits;
    let mut next_cycle = vec![0usize; n];
    let mut cycles = Vec::with_capacity(circuit.ops.len());
    let mut first_cycle = vec![0usize; n];
    let mut touched = vec![false; n];
    for op in &circuit.ops {
        let cycle = op.qubits.iter().map(|&q| next_cycle[q]).max().unwrap_or(0);
        cycles.push(cycle);
        for &q in &op.qubits {
            next_cycle[q] = cycle + 1;
            if !touched[q] {
                touched[q] = true;
                first_cycle[q] = cycle;
            }
        }
    }
    let final_cycle = next_cycle.iter().copied().max().unwrap_or(0);
    (cycles, first_cycle, final_cycle)
}

pub fn partition(circuit: &Circuit, block_size: usize) -> Circuit {
    let n = circuit.n_qubits;
    let (cycles, first_cycle, final_cycle) = assign_cycles(circuit);

    let mut bins: Vec<Bin> = Vec::new();
    let mut active_bin_of: Vec<Option<usize>> = vec![None; n];
    let mut pending: Vec<usize> = Vec::new();
    let mut dividing_line: Vec<usize> = first_cycle;
    let mut placed: Vec<Option<PlacedBlock>> = Vec::new();
    let mut rear_of: HashMap<usize, usize> = HashMap::new();

    for (i, op) in circuit.ops.iter().enumerate() {
        let cycle = cycles[i];

        let mut overlapping: Vec<usize> =
            op.qubits.iter().filter_map(|&q| active_bin_of[q]).collect();
        overlapping.sort_unstable();
        overlapping.dedup();

        let admissible: Vec<usize> = overlapping
            .iter()
            .copied()
            .filter(|&idx| bins[idx].can_accommodate(&op.qubits, block_size))
            .collect();

        for &idx in &overlapping {
            if !admissible.contains(&idx)
                && close_bin_qubits(&mut bins[idx], idx, &op.qubits, cycle, &mut active_bin_of)
            {
                pending.push(idx);
            }
        }

        let selected = if admissible.is_empty() {
            let idx = bins.len();
            bins.push(Bin::new());
            idx
        } else {
            admissible
                .iter()
                .copied()
                .find(|&idx| op.qubits.iter().all(|q| bins[idx].qubits.contains(q)))
                .unwrap_or(admissible[0])
        };

        for &idx in &admissible {
            if idx != selected
                && close_bin_qubits(&mut bins[idx], idx, &op.qubits, cycle, &mut active_bin_of)
            {
                pending.push(idx);
            }
        }

        bins[selected].add_op(op, cycle);
        for &q in &op.qubits {
            active_bin_of[q] = Some(selected);
        }

        // Circular-dependency propagation: any other currently-open bin
        // that already shares history (directly, or via its own blocked
        // set) with the selected bin's qubits is permanently forbidden
        // from claiming any of them, and inherits the selected bin's own
        // blocked set too, so the relation stays transitive.
        let selected_qubits: HashSet<usize> = bins[selected].qubits.iter().copied().collect();
        let selected_blocked: HashSet<usize> = bins[selected].blocked.iter().copied().collect();
        let mut other_active: Vec<usize> = active_bin_of.iter().filter_map(|x| *x).collect();
        other_active.sort_unstable();
        other_active.dedup();
        for idx in other_active {
            if idx == selected {
                continue;
            }
            let overlaps = {
                let b2 = &bins[idx];
                b2.blocked.iter().chain(b2.qubits.iter()).any(|q| selected_qubits.contains(q))
            };
            if overlaps {
                let b2 = &mut bins[idx];
                b2.blocked.extend(selected_qubits.iter().copied());
                b2.blocked.extend(selected_blocked.iter().copied());
            }
        }

        flush_pending(&mut pending, &mut bins, &mut dividing_line, &mut placed, &mut rear_of);
    }

    // Force-close every still-open bin on all its qubits, then drain
    // pending completely.
    let mut still_active: Vec<usize> = active_bin_of.iter().filter_map(|x| *x).collect();
    still_active.sort_unstable();
    still_active.dedup();
    for idx in still_active {
        let qubits = bins[idx].qubits.clone();
        if close_bin_qubits(&mut bins[idx], idx, &qubits, final_cycle, &mut active_bin_of) {
            pending.push(idx);
        }
    }
    flush_pending(&mut pending, &mut bins, &mut dividing_line, &mut placed, &mut rear_of);
    debug_assert!(
        pending.is_empty(),
        "partition: pending bins failed to fully drain -- this is a bug in the port"
    );

    let mut out = Circuit::new(n);
    for block in placed.into_iter().flatten() {
        let PlacedBlock { qubits, ops } = block;
        let local: HashMap<usize, usize> =
            qubits.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let mut inner = Circuit::new(qubits.len());
        for op in ops {
            let local_qubits: Vec<usize> = op.qubits.iter().map(|q| local[q]).collect();
            inner.push(op.gate, local_qubits);
        }
        out.push(Gate::Block(Box::new(inner)), qubits);
    }
    out
}

/// Repeatedly place any pending bin whose recorded start matches the
/// current dividing line on every qubit it uses, merging with the output's
/// current rear frontier first when the two blocks' qubit sets are in a
/// subset/superset relation (mirrors `process_pending_bins`).
fn flush_pending(
    pending: &mut Vec<usize>, bins: &mut [Bin], dividing_line: &mut [usize],
    placed: &mut Vec<Option<PlacedBlock>>, rear_of: &mut HashMap<usize, usize>,
) {
    loop {
        let ready_pos = pending
            .iter()
            .position(|&idx| bins[idx].starts.iter().all(|(&q, &start)| dividing_line[q] == start));
        let Some(pos) = ready_pos else { break };
        let idx = pending.remove(pos);

        let bin_ends = std::mem::take(&mut bins[idx].ends);
        let mut ops = std::mem::take(&mut bins[idx].ops);
        let mut qset: Vec<usize> = std::mem::take(&mut bins[idx].qubits);

        loop {
            let mut candidates: Vec<usize> =
                qset.iter().filter_map(|q| rear_of.get(q).copied()).collect();
            candidates.sort_unstable();
            candidates.dedup();

            let mut merged = false;
            for cand_idx in candidates {
                let Some(cand) = &placed[cand_idx] else { continue };
                let is_rear = cand.qubits.iter().all(|q| rear_of.get(q) == Some(&cand_idx));
                if !is_rear {
                    continue;
                }
                let cand_qubits: HashSet<usize> = cand.qubits.iter().copied().collect();
                let this_qubits: HashSet<usize> = qset.iter().copied().collect();

                if cand_qubits.iter().all(|q| this_qubits.contains(q)) {
                    // cand's qubits subset of this bin's -- cand came
                    // first chronologically, so its ops go before ours.
                    let cand = placed[cand_idx].take().unwrap();
                    let mut new_ops = cand.ops;
                    new_ops.append(&mut ops);
                    ops = new_ops;
                    merged = true;
                    break;
                } else if this_qubits.iter().all(|q| cand_qubits.contains(q)) {
                    // This bin's qubits subset of cand's -- same ordering
                    // (cand first), but widen our qubit set to cand's.
                    let cand = placed[cand_idx].take().unwrap();
                    let mut new_ops = cand.ops;
                    new_ops.append(&mut ops);
                    ops = new_ops;
                    qset = cand.qubits;
                    merged = true;
                    break;
                }
            }
            if !merged {
                break;
            }
        }

        qset.sort_unstable();
        qset.dedup();
        let new_idx = placed.len();
        for &q in &qset {
            rear_of.insert(q, new_idx);
        }
        placed.push(Some(PlacedBlock { qubits: qset, ops }));

        for (&q, &end) in &bin_ends {
            dividing_line[q] = end.map(|e| e + 1).unwrap_or(usize::MAX);
        }
    }
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

    /// Several 1- and 2-qubit gates crossing qubit ownership in a tangled
    /// pattern, checked only for the two invariants that must hold
    /// regardless of which exact grouping the algorithm picks.
    #[test]
    fn tangled_ownership_preserves_unitary_and_size_limit() {
        let mut c = Circuit::new(6);
        c.push(Gate::Cx, vec![1, 4]);
        c.push(Gate::H, vec![2]);
        c.push(Gate::T, vec![3]);
        c.push(Gate::S, vec![5]);
        c.push(Gate::Cx, vec![5, 0]);
        c.push(Gate::H, vec![1]);
        c.push(Gate::T, vec![3]);
        c.push(Gate::Cx, vec![2, 0]);
        c.push(Gate::Cx, vec![0, 3]);
        c.push(Gate::Cx, vec![3, 5]);
        c.push(Gate::H, vec![1]);
        c.push(Gate::S, vec![1]);

        let partitioned = partition(&c, 3);
        for op in &partitioned.ops {
            assert!(op.qubits.len() <= 3, "block exceeded size limit: {:?}", op.qubits);
        }
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }

    /// Regression test for the bug this port fixes: the old single-pass
    /// greedy partitioner closed a block the instant any op touched a new
    /// qubit near the size limit, and never regrouped afterward -- so a
    /// qubit repeatedly interleaved with others (exactly the shape a
    /// downstream wide-window joint-optimization consumer would need)
    /// ended up fragmented into many 1-2 qubit blocks with only 1-2 Rz
    /// gates each, leaving nothing to jointly optimize. The real
    /// algorithm's retroactive merging should instead recognize all of
    /// this fits in one 4-qubit window.
    #[test]
    fn repeated_qubit_interleaved_with_others_forms_one_wide_block() {
        let mut c = Circuit::new(4);
        c.push(Gate::Rz(0.1), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(0.2), vec![0]);
        c.push(Gate::Cx, vec![1, 2]);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Cx, vec![2, 3]);
        c.push(Gate::Rz(0.4), vec![0]);

        let partitioned = partition(&c, 4);
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);

        let max_rz_in_one_block = partitioned
            .ops
            .iter()
            .map(|op| match &op.gate {
                Gate::Block(inner) => inner.ops.iter().filter(|o| o.gate.is_rz()).count(),
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        assert!(
            max_rz_in_one_block > 2,
            "expected a wide block gathering more than 2 of the 4 Rz gates, got at most {max_rz_in_one_block} in any one block"
        );
    }

    /// Exercises the `blocked` circular-dependency mechanism: qubit 1
    /// briefly belongs to bin A (with qubit 0), then gets taken over by
    /// bin B (with qubit 2) once a size-limited bin can't accommodate a
    /// three-way touch. A later op touching both bins' qubits at once
    /// must not be allowed to silently reunify them in a way that would
    /// require an inconsistent ordering -- it should instead close both
    /// and start fresh. Only the invariants that must hold regardless of
    /// the exact partition are checked (unitary preservation, size
    /// limit); the point is that this shape doesn't panic or corrupt the
    /// circuit.
    #[test]
    fn blocked_qubits_prevents_invalid_reunification() {
        let mut c = Circuit::new(3);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::H, vec![2]);
        c.push(Gate::Cx, vec![1, 2]);
        c.push(Gate::Cx, vec![0, 2]);

        let partitioned = partition(&c, 2);
        for op in &partitioned.ops {
            assert!(op.qubits.len() <= 2, "block exceeded size limit: {:?}", op.qubits);
        }
        let unfolded = partitioned.unfold();
        assert!(distance(&c.get_unitary(), &unfolded.get_unitary()) < 1e-10);
    }
}
