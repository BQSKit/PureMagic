//! Exactly merges/cancels redundant diagonal (`rz`-family) rotations via
//! their phase-polynomial "parity" -- the t-par technique of Amy, Maslov,
//! and Mosca -- ported from `merge_phase_polynomial` in
//! `data_processing/compile_cliffordt.py` (its own doc comment has the full
//! derivation).
//!
//! Two diagonal single-qubit gates (`Z`/`S`/`Sdg`/`T`/`Tdg`/`Rz`) anywhere
//! in a `{Cx, diagonal}` region of the circuit commute and add exactly
//! whenever they act on the same XOR-parity of the original input qubits
//! at the time each is applied -- regardless of which physical qubit holds
//! that parity or what runs in between, since `Cx` and every diagonal gate
//! commute freely with each other. This matters a lot for CX-ladder-
//! decomposed controlled-phase gates (`cx; rz(-a); cx; rz(a)`), exactly how
//! QFT-family circuits' controlled-phase gates look after decomposition:
//! the Python reference measured this dropping the number of rotations
//! that actually need gridsynth on a 32-qubit QFT from 1350 to 522.
//!
//! Tracks each qubit's parity as a `BTreeSet<usize>` (a set of "symbols"),
//! not a fixed-width bitmask -- the Python reference uses an arbitrary-
//! precision int specifically because a non-diagonal gate mints a fresh
//! symbol never reused elsewhere, and a large circuit with many such gates
//! can need more symbols than any fixed width comfortably holds.
//! XOR-of-bitmask is exactly symmetric-difference-of-sets, and
//! `BTreeSet<usize>` already implements `Hash + Eq` (ordered, so hashing is
//! deterministic), so it works directly as a `HashMap` key.
//!
//! Any gate not in the recognized `{Cx, diagonal}` set (`H`/`X`/`Y`/`U3`/
//! `Cz`/`Swap`/anything else) is treated conservatively: it resets every
//! qubit it touches to a fresh, never-reused parity symbol, so it can only
//! cause a missed merge, never an incorrect one -- deliberately matching
//! the Python reference's own conservative treatment (which never
//! special-cases `Cz`, for instance, even though it is itself diagonal).

use std::collections::{BTreeSet, HashMap};

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

/// Exact-match tolerance for dropping a merged rotation that cancelled to
/// (numerically) zero -- matches the value used elsewhere in this pipeline
/// (`synthesize.rs::EXACTNESS_FLOOR`) and in the Python reference.
const EXACTNESS_FLOOR: f64 = 1e-12;

type Parity = BTreeSet<usize>;

/// The angle one occurrence of a diagonal single-qubit gate contributes,
/// or `None` if `gate` isn't part of the recognized diagonal family.
/// Deliberately excludes `Id` (matching the Python reference leaving `id`
/// out of its own `DIAGONAL_1Q_ANGLE` table) -- conservative, since
/// treating it as diagonal-with-zero-angle would be correct too, but this
/// stays a faithful port rather than an independent improvement.
fn diagonal_angle(gate: &Gate) -> Option<f64> {
    match gate {
        Gate::Z => Some(std::f64::consts::PI),
        Gate::S => Some(std::f64::consts::FRAC_PI_2),
        Gate::Sdg => Some(-std::f64::consts::FRAC_PI_2),
        Gate::T => Some(std::f64::consts::FRAC_PI_4),
        Gate::Tdg => Some(-std::f64::consts::FRAC_PI_4),
        Gate::Rz(theta) => Some(*theta),
        _ => None,
    }
}

/// Merge every group of diagonal gates sharing a parity into one `Rz` at
/// the position of the group's last occurrence, dropping the group
/// entirely if its summed angle cancels to (numerically) zero. See module
/// docs for the algorithm. An exact rewrite: the result's unitary matches
/// `circuit`'s exactly (up to floating-point rounding), never approximated.
pub fn merge_phase_polynomial(circuit: &Circuit) -> Circuit {
    let n = circuit.n_qubits;
    let mut parity: Vec<Parity> = (0..n).map(|i| BTreeSet::from([i])).collect();
    let mut next_symbol = n;

    let mut group_key: Vec<Option<Parity>> = Vec::with_capacity(circuit.ops.len());
    let mut group_total: HashMap<Parity, f64> = HashMap::new();
    let mut group_last_index: HashMap<Parity, usize> = HashMap::new();

    for (i, op) in circuit.ops.iter().enumerate() {
        match &op.gate {
            Gate::Cx => {
                let (control, target) = (op.qubits[0], op.qubits[1]);
                parity[target] = parity[target].symmetric_difference(&parity[control]).copied().collect();
                group_key.push(None);
            }
            gate if diagonal_angle(gate).is_some() => {
                let key = parity[op.qubits[0]].clone();
                *group_total.entry(key.clone()).or_insert(0.0) += diagonal_angle(gate).unwrap();
                group_last_index.insert(key.clone(), i);
                group_key.push(Some(key));
            }
            _ => {
                for &q in &op.qubits {
                    parity[q] = BTreeSet::from([next_symbol]);
                    next_symbol += 1;
                }
                group_key.push(None);
            }
        }
    }

    let mut out = Circuit::new(n);
    for (i, op) in circuit.ops.iter().enumerate() {
        match &group_key[i] {
            None => out.ops.push(op.clone()),
            Some(key) => {
                if group_last_index[key] != i {
                    continue; // an earlier occurrence of this parity already covers it
                }
                let total = group_total[key].rem_euclid(2.0 * std::f64::consts::PI);
                if total > EXACTNESS_FLOOR {
                    out.push(Gate::Rz(total), vec![op.qubits[0]]);
                }
            }
        }
    }
    out
}

/// How many `Rz` occurrences in `circuit` are NOT within `tol` of a
/// multiple of pi/4 -- a cheap, gridsynth-free proxy for "genuinely costly"
/// rotations, mirroring the Python reference's `count_real_rotations`.
/// `Z`/`S`/`Sdg`/`T`/`Tdg` never need checking (already exact pi/4
/// multiples by construction); `U3` occurrences are untouched by
/// `merge_phase_polynomial` either way, so they don't need to factor into
/// a comparison between a merged and unmerged candidate.
pub fn count_real_rotations(circuit: &Circuit, tol: f64) -> usize {
    circuit
        .ops
        .iter()
        .filter(|op| match op.gate {
            Gate::Rz(theta) => {
                let k = theta / std::f64::consts::FRAC_PI_4;
                (k - k.round()).abs() >= tol
            }
            _ => false,
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    fn rz(theta: f64, q: usize) -> (Gate, Vec<usize>) {
        (Gate::Rz(theta), vec![q])
    }

    fn build(ops: Vec<(Gate, Vec<usize>)>, n: usize) -> Circuit {
        let mut c = Circuit::new(n);
        for (gate, qubits) in ops {
            c.push(gate, qubits);
        }
        c
    }

    /// The standard 2-CX controlled-phase decomposition (matches
    /// qelib1.inc's own `crz`): `rz(a/2) c; cx c,t; rz(-a/2) t; cx c,t;
    /// rz(a/2) t;` -- three Rz occurrences that should merge to fewer.
    fn cu1_ladder(control: usize, target: usize, angle: f64) -> Vec<(Gate, Vec<usize>)> {
        vec![
            rz(angle / 2.0, control),
            (Gate::Cx, vec![control, target]),
            rz(-angle / 2.0, target),
            (Gate::Cx, vec![control, target]),
            rz(angle / 2.0, target),
        ]
    }

    #[test]
    fn cx_ladder_controlled_phase_gates_merge_and_preserve_unitary() {
        let mut ops = cu1_ladder(0, 1, 0.7);
        ops.extend(cu1_ladder(0, 2, 1.3));
        let original = build(ops.clone(), 3);

        let merged = merge_phase_polynomial(&original);

        let rz_count = |c: &Circuit| c.ops.iter().filter(|op| op.gate.is_rz()).count();
        assert!(rz_count(&merged) < rz_count(&original), "merge should reduce the Rz count");
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }

    #[test]
    fn same_qubit_no_cx_merges_into_one_summed_rotation() {
        let original = build(vec![rz(0.3, 0), rz(0.4, 0)], 1);
        let merged = merge_phase_polynomial(&original);
        assert_eq!(merged.ops.len(), 1);
        assert!(matches!(merged.ops[0].gate, Gate::Rz(a) if (a - 0.7).abs() < 1e-12));
    }

    #[test]
    fn non_diagonal_gate_between_same_qubit_rotations_blocks_the_merge() {
        let original = build(vec![rz(0.3, 0), (Gate::H, vec![0]), rz(0.4, 0)], 1);
        let merged = merge_phase_polynomial(&original);
        let rz_count = merged.ops.iter().filter(|op| op.gate.is_rz()).count();
        assert_eq!(rz_count, 2, "an H in between must reset parity, not let the two Rz merge");
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }

    #[test]
    fn cz_between_same_qubit_rotations_blocks_the_merge() {
        let original = build(vec![rz(0.3, 0), (Gate::Cz, vec![0, 1]), rz(0.4, 0)], 2);
        let merged = merge_phase_polynomial(&original);
        let rz_count = merged.ops.iter().filter(|op| op.gate.is_rz()).count();
        assert_eq!(rz_count, 2, "Cz is treated as an unrecognized reset, like the Python reference");
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }

    #[test]
    fn swap_between_same_qubit_rotations_blocks_the_merge() {
        let original = build(vec![rz(0.3, 0), (Gate::Swap, vec![0, 1]), rz(0.4, 0)], 2);
        let merged = merge_phase_polynomial(&original);
        let rz_count = merged.ops.iter().filter(|op| op.gate.is_rz()).count();
        assert_eq!(rz_count, 2);
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }

    #[test]
    fn opposite_angles_on_the_same_parity_cancel_to_no_gate_at_all() {
        let original = build(vec![rz(0.55, 0), rz(-0.55, 0)], 1);
        let merged = merge_phase_polynomial(&original);
        assert!(merged.ops.is_empty(), "a cancelled rotation should be dropped, not emitted as Rz(0)");
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }

    #[test]
    fn count_real_rotations_ignores_pi_4_multiples() {
        let c = build(vec![rz(std::f64::consts::FRAC_PI_4, 0), rz(0.4, 0)], 1);
        assert_eq!(count_real_rotations(&c, 1e-9), 1);
    }

    #[test]
    fn qft_shaped_circuit_reduces_real_rotation_count_after_merge() {
        // Mirrors qft_n63.qasm's structure: several CX-ladder-decomposed
        // controlled-phase gates all targeting qubit 0, each one's ladder
        // touching the qubit that already accumulated phase from earlier
        // ones -- the actual shape that motivated this pass.
        let mut ops = Vec::new();
        for (k, &control) in [1, 2, 3, 4].iter().enumerate() {
            ops.extend(cu1_ladder(control, 0, std::f64::consts::PI / 2f64.powi(k as i32 + 1)));
        }
        let original = build(ops, 5);
        let merged = merge_phase_polynomial(&original);

        let tol = 1e-8;
        let before = count_real_rotations(&original, tol);
        let after = count_real_rotations(&merged, tol);
        assert!(after < before, "expected fewer real rotations after merging (before={before}, after={after})");
        assert!(distance(&original.get_unitary(), &merged.get_unitary()) < 1e-12);
    }
}
