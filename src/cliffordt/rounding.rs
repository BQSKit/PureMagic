//! Stage 1: angle rounding onto the discrete π/4 grid, mirroring bqskit-ft's
//! `RoundToDiscreteZPass`.

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

/// If `angle` (mod 2π) sits within `epsilon` of a multiple of π/4, return
/// the fixed gate sequence realizing that exact multiple; otherwise `None`.
pub fn round_angle(angle: f64, epsilon: f64) -> Option<Vec<Gate>> {
    let two_pi = std::f64::consts::TAU;
    let normalized = angle.rem_euclid(two_pi);
    let pi_over_4 = std::f64::consts::FRAC_PI_4;
    let value = (normalized / pi_over_4).round();
    let rounded_angle = value * pi_over_4;
    let residual = (normalized - rounded_angle).abs();
    if residual > epsilon {
        return None;
    }

    let v = (value as i64).rem_euclid(8);
    let gates = match v {
        0 => vec![],
        1 => vec![Gate::T],
        2 => vec![Gate::S],
        3 => vec![Gate::S, Gate::T],
        4 => vec![Gate::Z],
        5 => vec![Gate::Sdg, Gate::Tdg],
        6 => vec![Gate::Sdg],
        _ => vec![Gate::Tdg],
    };
    Some(gates)
}

/// Replace every `Rz` operation whose angle is within `epsilon` of a
/// multiple of π/4 with its exact fixed-gate realization; leaves any `Rz`
/// that isn't close enough untouched, along with every other gate.
pub fn round_to_discrete_z(circuit: &Circuit, epsilon: f64) -> Circuit {
    let mut out = Circuit::new(circuit.n_qubits);
    for op in &circuit.ops {
        if let Gate::Rz(angle) = op.gate {
            if let Some(gates) = round_angle(angle, epsilon) {
                for g in gates {
                    out.push(g, op.qubits.clone());
                }
                continue;
            }
        }
        out.ops.push(op.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    const EPS: f64 = 1e-8;

    #[test]
    fn zero_angle_rounds_to_no_gates() {
        assert_eq!(round_angle(0.0, EPS), Some(vec![]));
    }

    #[test]
    fn pi_over_4_rounds_to_t() {
        assert_eq!(round_angle(std::f64::consts::FRAC_PI_4, EPS), Some(vec![Gate::T]));
    }

    #[test]
    fn pi_over_2_rounds_to_s() {
        assert_eq!(round_angle(std::f64::consts::FRAC_PI_2, EPS), Some(vec![Gate::S]));
    }

    #[test]
    fn negative_pi_over_4_rounds_to_tdg() {
        assert_eq!(round_angle(-std::f64::consts::FRAC_PI_4, EPS), Some(vec![Gate::Tdg]));
    }

    #[test]
    fn pi_rounds_to_z() {
        assert_eq!(round_angle(std::f64::consts::PI, EPS), Some(vec![Gate::Z]));
    }

    #[test]
    fn angle_far_from_grid_does_not_round() {
        assert_eq!(round_angle(0.123456, EPS), None);
    }

    #[test]
    fn angle_within_epsilon_of_grid_point_still_rounds() {
        let nudged = std::f64::consts::FRAC_PI_4 + EPS / 2.0;
        assert_eq!(round_angle(nudged, EPS), Some(vec![Gate::T]));
    }

    #[test]
    fn rounded_gate_word_matches_original_angle_within_tolerance() {
        for (angle, _label) in [
            (std::f64::consts::FRAC_PI_4, "pi/4"),
            (3.0 * std::f64::consts::FRAC_PI_4, "3pi/4"),
            (5.0 * std::f64::consts::FRAC_PI_4, "5pi/4"),
        ] {
            let word = round_angle(angle, EPS).expect("should round");
            let mut c = Circuit::new(1);
            for g in word {
                c.push(g, vec![0]);
            }
            let mut target = Circuit::new(1);
            target.push(Gate::Rz(angle), vec![0]);
            assert!(distance(&target.get_unitary(), &c.get_unitary()) < 1e-6);
        }
    }

    #[test]
    fn round_to_discrete_z_replaces_only_rounded_rz_gates() {
        let mut circuit = Circuit::new(1);
        circuit.push(Gate::H, vec![0]);
        circuit.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        circuit.push(Gate::Rz(0.123456), vec![0]);
        let result = round_to_discrete_z(&circuit, EPS);
        // H, T (replacing the first Rz), Rz(0.123456) untouched
        assert_eq!(result.ops.len(), 3);
        assert!(matches!(result.ops[1].gate, Gate::T));
        assert!(matches!(result.ops[2].gate, Gate::Rz(_)));
    }
}
