//! Stage 2's per-block search: `ScanningGateRemovalPass`.
//!
//! Scans a block's gates from one side, tentatively removing each one and
//! renumerically re-fitting the remaining continuous parameters (via
//! `instantiate.rs`'s shared NLS engine) against the block's original
//! target unitary -- not just checking the unmodified remainder, which is
//! what makes this more than a static redundancy check (see the plan
//! doc's Context section).
//!
//! That per-op loop has a structural blind spot: it only ever tries
//! removing ONE op at a time, so a block that's redundant only as a whole
//! -- e.g. two adjacent identical parameter-free `Cx` gates, which cancel
//! to the identity together but neither removal alone gets anywhere close
//! -- is invisible to it (confirmed on `qft_n63.qasm`, where 595 of 1953
//! Stage 2 blocks were exactly `Cx(a,b); Cx(a,b)`, entirely dead weight
//! left over once `phase_merge.rs` cancelled the diagonal gate that used
//! to sit between them, none of which this pass removed). `whole_block_
//! is_redundant` checks that specific case directly before the per-op loop
//! even runs.

use crate::cliffordt::instantiate::instantiate_multistart;
use crate::cliffordt::matrix::{distance, identity, infidelity, Unitary};
use crate::cliffordt::qgate_circuit::{Circuit, Operation};

pub struct ScanConfig {
    pub success_threshold: f64,
    pub start_from_left: bool,
    pub n_starts: usize,
    pub max_iters: usize,
    pub seed: u64,
    /// When true, also accept a removal whose resulting circuit is within
    /// `success_threshold` *infidelity* of the block's target, not just
    /// within `success_threshold` operator-norm `distance` -- matching
    /// bqskit's own `ScanningGateRemovalPass`, which uses
    /// `HilbertSchmidtResidualsGenerator` (a cost quadratic in the operator
    /// error) against the same nominal threshold. Since infidelity scales
    /// roughly as `distance^2` near a match, this tolerates angular
    /// deviations up to roughly `sqrt(success_threshold)` instead of
    /// `success_threshold` directly -- confirmed as the actual mechanism
    /// behind bqskit's real-rotation-count reduction on `qft_n63.qasm`
    /// (1275 -> ~680), which our own operator-norm-only criterion missed
    /// almost entirely. Off by default: the resulting per-block error is
    /// always measured exactly and returned (see `scanning_gate_removal`'s
    /// return type) rather than assumed, so the pipeline's additive error
    /// bound stays rigorous either way -- but the actual error with this on
    /// can be far larger per approximate cancellation than `epsilon`
    /// itself, which is why it's opt-in rather than always applied.
    pub approx_cancel: bool,
}

impl Default for ScanConfig {
    fn default() -> Self {
        ScanConfig { success_threshold: 1e-8, start_from_left: true, n_starts: 4, max_iters: 100, seed: 0, approx_cancel: false }
    }
}

/// Whether `built` is an acceptable stand-in for `target` under `config`,
/// and the *real* operator-norm cost of using it either way -- the accept
/// decision may use the looser infidelity criterion (`approx_cancel`), but
/// the returned cost is always the true `distance`, never assumed to be
/// `success_threshold`-sized. Callers fold this into the pipeline's
/// additive error-bound accounting.
fn accept(target: &Unitary, built: &Unitary, config: &ScanConfig) -> (bool, f64) {
    let d = distance(target, built);
    let ok = if config.approx_cancel { infidelity(target, built) < config.success_threshold } else { d < config.success_threshold };
    (ok, d)
}

/// Returns the rewritten block and the actual operator-norm error it now
/// carries relative to `circuit`'s own original unitary (0.0 if nothing
/// was changed) -- the caller is expected to add this into the pipeline's
/// running error-bound total, the same way Stage 4 already does for final
/// synthesis.
pub fn scanning_gate_removal(circuit: &Circuit, config: &ScanConfig) -> (Circuit, f64) {
    let target: Unitary = circuit.get_unitary();

    // Whole-block check first -- see module docs for why the per-op loop
    // below can never find this case on its own.
    let (ok, d) = accept(&target, &identity(1 << circuit.n_qubits), config);
    if ok {
        return (Circuit::new(circuit.n_qubits), d);
    }

    let mut slots: Vec<Option<Operation>> = circuit.ops.iter().cloned().map(Some).collect();
    let mut last_error = 0.0;

    let order: Vec<usize> =
        if config.start_from_left { (0..slots.len()).collect() } else { (0..slots.len()).rev().collect() };

    for idx in order {
        let removed = slots[idx].take();
        let candidate = build_circuit(circuit.n_qubits, &slots);

        let fit = instantiate_multistart(
            &candidate,
            &target,
            config.n_starts,
            config.max_iters,
            config.seed ^ idx as u64,
            config.success_threshold,
        );

        let mut fitted = candidate.clone();
        fitted.set_params(&fit.params);
        let (ok, d) = accept(&target, &fitted.get_unitary(), config);

        if ok {
            apply_params_to_slots(&mut slots, &fit.params);
            last_error = d;
        } else {
            slots[idx] = removed;
        }
    }

    (build_circuit(circuit.n_qubits, &slots), last_error)
}

fn build_circuit(n_qubits: usize, slots: &[Option<Operation>]) -> Circuit {
    let mut c = Circuit::new(n_qubits);
    for slot in slots.iter().flatten() {
        c.ops.push(slot.clone());
    }
    c
}

fn apply_params_to_slots(slots: &mut [Option<Operation>], params: &[f64]) {
    let mut it = params.iter();
    for slot in slots.iter_mut().flatten() {
        if slot.gate.is_rz() {
            if let Some(&v) = it.next() {
                slot.gate.set_param(v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::qgate_circuit::Gate;

    #[test]
    fn opposite_rz_rotations_collapse_via_reinstantiation() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Rz(-0.3), vec![0]);
        let original_unitary = c.get_unitary();

        let (result, error) = scanning_gate_removal(&c, &ScanConfig::default());
        assert!(
            result.ops.len() < c.ops.len(),
            "expected at least one gate removed, got {} ops",
            result.ops.len()
        );
        assert!(distance(&original_unitary, &result.get_unitary()) < 1e-6);
        // The returned error must be the real, correctly-computed distance
        // -- not just "some small untracked value" -- and consistent with
        // what we just measured directly above.
        assert!(error >= 0.0 && error < 1e-6, "reported error {error} inconsistent with the actual distance");
    }

    #[test]
    fn single_gate_with_no_redundancy_is_not_removed() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        let (result, error) = scanning_gate_removal(&c, &ScanConfig::default());
        assert_eq!(result.ops.len(), 1);
        assert_eq!(error, 0.0, "nothing changed, so no error should be reported");
    }

    #[test]
    fn result_always_preserves_the_original_unitary() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.77), vec![0]);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(-0.77), vec![0]);
        let original_unitary = c.get_unitary();
        let (result, _error) = scanning_gate_removal(&c, &ScanConfig::default());
        assert!(distance(&original_unitary, &result.get_unitary()) < 1e-6);
    }

    /// Regression test for the bug this fixes: two adjacent identical `Cx`
    /// gates and nothing else -- the exact shape found dead in `qft_n63.qasm`
    /// blocks left over by `phase_merge.rs`. Neither individual removal
    /// ever gets close to identity, so only a whole-block check catches it.
    #[test]
    fn two_identical_adjacent_cx_gates_cancel_to_an_empty_block() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![1, 0]);
        c.push(Gate::Cx, vec![1, 0]);
        let (result, _error) = scanning_gate_removal(&c, &ScanConfig::default());
        assert!(result.ops.is_empty(), "expected the whole redundant block to be dropped, got {:?}", result.ops);
    }

    /// A block with the same two `Cx` gates plus a genuine, non-redundant
    /// `Rz` between them must NOT be emptied -- the whole-block check must
    /// only fire when the block is truly the identity.
    #[test]
    fn cx_sandwiched_real_rotation_is_not_dropped() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![1, 0]);
        c.push(Gate::Rz(0.6), vec![0]);
        c.push(Gate::Cx, vec![1, 0]);
        let original_unitary = c.get_unitary();
        let (result, _error) = scanning_gate_removal(&c, &ScanConfig::default());
        assert!(!result.ops.is_empty(), "a genuine ZZ-type rotation must not be dropped");
        assert!(distance(&original_unitary, &result.get_unitary()) < 1e-6);
    }

    /// The motivating case: a `Cx; Rz(theta); Cx` block where theta is
    /// close to a multiple of 2*pi by more than epsilon in angle (so it's
    /// NOT within operator-norm epsilon of the identity) but well within
    /// epsilon *infidelity* -- the actual mechanism behind bqskit's own
    /// ScanningGateRemovalPass finding reductions ours doesn't (see module
    /// docs). Mirrors a real removed block found on qft_n63.qasm.
    #[test]
    fn approx_cancel_drops_a_near_clifford_block_the_default_config_must_not() {
        let theta = 2.0 * std::f64::consts::PI - 2e-4;
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![1, 0]);
        c.push(Gate::Rz(theta), vec![0]);
        c.push(Gate::Cx, vec![1, 0]);
        let target = c.get_unitary();

        // Sanity check: this angle really is outside plain operator-norm
        // epsilon of the identity (otherwise this test wouldn't be
        // distinguishing the two configs at all).
        let default_config = ScanConfig::default();
        assert!(distance(&target, &identity(4)) > default_config.success_threshold);

        let (result, _error) = scanning_gate_removal(&c, &default_config);
        assert!(!result.ops.is_empty(), "without approx_cancel, this near-but-not-exact block must survive");

        let approx_config = ScanConfig { approx_cancel: true, ..ScanConfig::default() };
        let (result, error) = scanning_gate_removal(&c, &approx_config);
        assert!(result.ops.is_empty(), "with approx_cancel, this near-Clifford block should be dropped entirely");
        // The reported error must be the REAL operator-norm cost of that
        // approximation, not silently reported as epsilon-sized -- this is
        // the whole point of "properly accounted".
        let actual_distance = distance(&target, &identity(4));
        assert!(
            (error - actual_distance).abs() < 1e-12,
            "returned error {error} should equal the real distance {actual_distance}"
        );
        assert!(error > default_config.success_threshold, "the accounted error should visibly exceed epsilon, not hide it");
    }
}
