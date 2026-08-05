//! Stage 5: TRbO-style joint gauge-freedom optimization.
//!
//! For a (typically multi-qubit) block, jointly search over its several
//! free `Rz` angles -- holding the block's overall unitary fixed within
//! `success_threshold` -- for an assignment where as many angles as
//! possible land exactly on a discrete grid (T-representable points
//! first, then Clifford-only "remainder" points), rather than rounding
//! each angle independently (that's Stage 3). Mirrors the actual
//! `trbo`/`tcount`/`clift` Python package read earlier this session: a
//! smooth "smallest-N-deviation" relaxation of "how many angles are
//! exactly rounded," found via multi-start nonlinear least squares and a
//! binary search over N, then a deterministic snap-to-grid.

use nalgebra::{DMatrix, DVector};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use crate::cliffordt::instantiate::{fidelity_residuals, fidelity_residuals_and_jacobian, instantiate_multistart, lm_fit_multistart};
use crate::cliffordt::matrix::{distance, Unitary};
use crate::cliffordt::qgate_circuit::{Circuit, Gate};

pub struct TrboConfig {
    pub success_threshold: f64,
    pub multistarts: usize,
    pub max_iters: usize,
    pub seed: u64,
}

impl Default for TrboConfig {
    fn default() -> Self {
        // Matches the actual upstream `trbo` package's own default
        // (`TReductionByOptimiationPass.__init__`'s `multistarts: int = 64`,
        // used unmodified by `compile_cliffordt.py`'s call site) -- our
        // hand-rolled finite-difference LM solver is weaker per-attempt
        // than trbo's Ceres-backed one, so matching its restart count
        // matters more here, not less.
        TrboConfig { success_threshold: 1e-8, multistarts: 64, max_iters: 150, seed: 0 }
    }
}

#[derive(Clone, Copy)]
enum Discretization {
    T,
    Cliff,
}

impl Discretization {
    fn period(self) -> f64 {
        match self {
            Discretization::T => FRAC_PI_4,
            Discretization::Cliff => FRAC_PI_2,
        }
    }
}

/// Exact gate sequence for a value that's already been decided to be
/// rounded to the nearest multiple of `disc`'s period -- mirrors
/// `trbo`'s `clift.circuit_for_rounded_val`.
fn rounded_gate(val: f64, disc: Discretization) -> Vec<Gate> {
    let val = val.rem_euclid(TAU);
    match disc {
        Discretization::Cliff => {
            let rv = (val * 2.0 / PI).round() as i64 % 4;
            match rv {
                1 => vec![Gate::S],
                2 => vec![Gate::Z],
                3 => vec![Gate::Sdg],
                _ => vec![],
            }
        }
        Discretization::T => {
            let rv = (val * 4.0 / PI).round() as i64 % 8;
            let mut gates = Vec::new();
            if rv < 4 {
                if rv >= 2 {
                    gates.push(Gate::S);
                }
                if rv % 2 == 1 {
                    gates.push(Gate::T);
                }
            } else if rv > 4 {
                if rv <= 6 {
                    gates.push(Gate::Sdg);
                }
                if rv % 2 == 1 {
                    gates.push(Gate::Tdg);
                }
            } else {
                gates.push(Gate::Z);
            }
            gates
        }
    }
}

/// Deviation of a single angle from the nearest multiple of `period`.
fn deviation_single(p: f64, period: f64) -> f64 {
    let shifted = (p - period / 2.0).rem_euclid(period);
    (shifted - period / 2.0).abs() / 2.0
}

/// Deviation of each angle from the nearest multiple of `period` -- mirrors
/// `tcount.get_deviation_arr` (without a blacklist: every parameter passed
/// in here is already known to be a genuine free Rz angle).
fn deviation(params: &[f64], period: f64) -> Vec<f64> {
    params.iter().map(|&p| deviation_single(p, period)).collect()
}

/// `d(deviation)/dp`, exact almost everywhere (`deviation` is piecewise
/// linear with slope +-1/2; the derivative is undefined only exactly at the
/// wrap/kink points, a measure-zero set no floating-point parameter lands
/// on in practice).
fn deviation_derivative(p: f64, period: f64) -> f64 {
    let shifted = (p - period / 2.0).rem_euclid(period);
    (shifted - period / 2.0).signum() / 2.0
}

/// Fidelity residuals (global-phase-invariant, see `fidelity_residuals`)
/// concatenated with a `dim`-scaled residual per one of the `n_round`
/// smallest angle-deviations (mirrors `SumResidualsGenerator` of
/// `MatrixDistanceCost` + `RoundSmallestNResiduals`).
fn combined_residuals(circuit_template: &Circuit, target: &Unitary, params: &[f64], period: f64, n_round: usize) -> DVector<f64> {
    let fid = fidelity_residuals(circuit_template, target, params);
    if n_round == 0 {
        return fid;
    }
    let dim = (1usize << circuit_template.n_qubits) as f64;
    let mut devs = deviation(params, period);
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out = DVector::zeros(fid.len() + n_round);
    out.rows_mut(0, fid.len()).copy_from(&fid);
    for (i, &d) in devs[..n_round].iter().enumerate() {
        out[fid.len() + i] = d * dim;
    }
    out
}

/// `combined_residuals`'s value *and* its analytic Jacobian, computed
/// together off one shared `fidelity_residuals_and_jacobian` call: its rows
/// unchanged, plus one row per one of the `n_round` smallest deviations --
/// each such row is `deviation_derivative` (scaled by `dim`) in the column
/// of the *specific* parameter that produced that smallest deviation, zero
/// in every other column. Valid almost everywhere: which parameter lands
/// among the `n_round` smallest is a locally constant, discrete selection
/// (`combined_residuals` re-sorts at every evaluation, so a parameter
/// crossing another's rank is itself already handled the next time the
/// residual/Jacobian pair is (re-)evaluated at the new point -- exactly how
/// a numerical Jacobian would behave here too, just cheaper).
fn combined_residuals_and_jacobian(
    circuit_template: &Circuit,
    target: &Unitary,
    params: &[f64],
    period: f64,
    n_round: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let (fid, fid_jac) = fidelity_residuals_and_jacobian(circuit_template, target, params);
    if n_round == 0 {
        return (fid, fid_jac);
    }
    let dim = (1usize << circuit_template.n_qubits) as f64;
    let n = params.len();
    let mut idx_dev: Vec<(usize, f64)> = params.iter().enumerate().map(|(i, &p)| (i, deviation_single(p, period))).collect();
    idx_dev.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let m = fid.len();
    let mut r = DVector::zeros(m + n_round);
    r.rows_mut(0, m).copy_from(&fid);
    let mut jac = DMatrix::zeros(m + n_round, n);
    jac.view_mut((0, 0), (m, n)).copy_from(&fid_jac);
    for (row, &(param_idx, d)) in idx_dev[..n_round].iter().enumerate() {
        r[m + row] = d * dim;
        jac[(m + row, param_idx)] = deviation_derivative(params[param_idx], period) * dim;
    }
    (r, jac)
}

/// Jointly fit `circuit_template`'s Rz angles (biased toward getting
/// `n_round` of them close to `disc`'s grid) against `target`.
///
/// Before launching the expensive multistart search, tries `known_good`
/// (typically the previous binary-search probe's accepted params) and the
/// template's own unmodified params via a single, optimization-free
/// residual evaluation each -- mirrors `trbo`'s own `validated_optimization`
/// ("Try known good sets of parameters before introducing randomness"),
/// which is often enough on its own (`n_round=0` always qualifies; angles
/// already exactly on the grid from an earlier stage need no refitting).
/// `combined_residuals`'s Euclidean norm is the Frobenius norm of the
/// (phase-aligned) fidelity part, which upper-bounds `matrix::distance`'s
/// spectral norm -- so this check is a safe (never a false positive)
/// stand-in for the `distance(...) < success_threshold` test the caller
/// ultimately relies on.
fn fit_with_rounding_bias(
    circuit_template: &Circuit,
    target: &Unitary,
    n_round: usize,
    disc: Discretization,
    config: &TrboConfig,
    seed: u64,
    known_good: &[Vec<f64>],
) -> Vec<f64> {
    let n_rz = circuit_template.num_params();
    let period = disc.period();

    let original_params = circuit_template.params();
    for guess in known_good.iter().chain(std::iter::once(&original_params)) {
        if guess.len() != n_rz {
            continue;
        }
        let cost = combined_residuals(circuit_template, target, guess, period, n_round).norm();
        if cost < config.success_threshold {
            return guess.clone();
        }
    }

    let extra_starts = vec![vec![0.0; n_rz], original_params];

    let template = circuit_template.clone();
    let target_c = target.clone();
    let fit = lm_fit_multistart(
        move |p| combined_residuals_and_jacobian(&template, &target_c, p, period, n_round),
        n_rz,
        &extra_starts,
        config.multistarts,
        config.max_iters,
        seed,
        config.success_threshold,
    );
    fit.params
}

/// Binary search over how many angles can be rounded to `disc`'s grid
/// while keeping the block within `config.success_threshold` of `target`;
/// then perform the actual rounding. Mirrors
/// `TReductionByOptimiationPass.optimize_for_discretization`.
fn optimize_for_discretization(circuit: &Circuit, target: &Unitary, disc: Discretization, config: &TrboConfig) -> Circuit {
    let n_rz = circuit.num_params();
    if n_rz == 0 {
        return circuit.clone();
    }

    let mut low = 0usize;
    let mut high = n_rz;
    let mut best_n = 0usize;
    let mut best_params = circuit.params();

    while low <= high {
        let mid = low + (high - low) / 2;
        let seed = config.seed ^ (mid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let known_good = [best_params.clone()];
        let params = fit_with_rounding_bias(circuit, target, mid, disc, config, seed, &known_good);

        let mut fitted = circuit.clone();
        fitted.set_params(&params);
        let ok = distance(target, &fitted.get_unitary()) < config.success_threshold;

        if ok {
            best_n = mid;
            best_params = params;
            low = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            high = mid - 1;
        }
    }

    if best_n == 0 {
        return circuit.clone();
    }

    // Round the `best_n` angles closest to the grid.
    let devs = deviation(&best_params, disc.period());
    let mut order: Vec<usize> = (0..devs.len()).collect();
    order.sort_by(|&a, &b| devs[a].partial_cmp(&devs[b]).unwrap());
    let to_round: std::collections::HashSet<usize> = order[..best_n].iter().copied().collect();

    let mut fitted = circuit.clone();
    fitted.set_params(&best_params);
    let mut out = Circuit::new(circuit.n_qubits);
    let mut rz_idx = 0usize;
    for op in &fitted.ops {
        if let Gate::Rz(angle) = op.gate {
            if to_round.contains(&rz_idx) {
                for g in rounded_gate(angle, disc) {
                    out.push(g, op.qubits.clone());
                }
            } else {
                out.push(Gate::Rz(angle), op.qubits.clone());
            }
            rz_idx += 1;
        } else {
            out.ops.push(op.clone());
        }
    }

    // Verify, with a fidelity-only fallback re-fit of whatever Rz angles
    // remain if the snap pushed things slightly out of tolerance.
    if distance(target, &out.get_unitary()) > config.success_threshold && out.num_params() > 0 {
        let refit =
            instantiate_multistart(&out, target, config.multistarts, config.max_iters, config.seed, config.success_threshold);
        if refit.distance < distance(target, &out.get_unitary()) {
            out.set_params(&refit.params);
        }
    }

    out
}

/// `true` if `b` is a "better" (fewer expensive resources) circuit than
/// `a` -- mirrors `trbo.clift.better_min_t_count_circuit`, restricted to
/// this pipeline's own gate vocabulary (no non-Clifford+T+Rz gates ever
/// appear here, so that tier of the Python comparator is always tied).
fn better_circuit(a: &Circuit, b: &Circuit) -> bool {
    let count = |c: &Circuit, pred: fn(&Gate) -> bool| c.ops.iter().filter(|op| pred(&op.gate)).count();
    let rz = |g: &Gate| g.is_rz();
    let t = |g: &Gate| matches!(g, Gate::T | Gate::Tdg);
    let multi_clifford = |g: &Gate| g.is_clifford() && g.num_qubits() > 1;
    let total_clifford = |g: &Gate| g.is_clifford();

    for pred in [rz, t, multi_clifford, total_clifford] {
        let (ac, bc) = (count(a, pred), count(b, pred));
        if bc < ac {
            return true;
        }
        if bc > ac {
            return false;
        }
    }
    false
}

/// Optimize one block: try the T-representable discretization first, then
/// (on top of whatever that achieves) the Clifford-only "remainder" pass,
/// keeping whichever result is better by `better_circuit` at each step.
pub fn trbo_optimize(circuit: &Circuit, config: &TrboConfig) -> Circuit {
    let target = circuit.get_unitary();
    let mut best = circuit.clone();

    let after_t = optimize_for_discretization(circuit, &target, Discretization::T, config);
    if better_circuit(&best, &after_t) {
        best = after_t;
    }

    if best.num_params() > 0 {
        let after_cliff = optimize_for_discretization(&best, &target, Discretization::Cliff, config);
        if better_circuit(&best, &after_cliff) {
            best = after_cliff;
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauge_freedom_rounds_at_least_one_of_two_equivalent_angles() {
        // Same-axis rotations commute and add: any (x, y) with x + y ==
        // pi/4 (mod 2*pi) reproduces the same target. Neither starting
        // angle is individually nice.
        let target_angle = std::f64::consts::FRAC_PI_4;
        let x0 = 0.37;
        let y0 = target_angle - x0;

        let mut circuit = Circuit::new(1);
        circuit.push(Gate::Rz(x0), vec![0]);
        circuit.push(Gate::Rz(y0), vec![0]);
        let original_unitary = circuit.get_unitary();

        let config = TrboConfig { success_threshold: 1e-8, multistarts: 12, max_iters: 200, seed: 3 };
        let result = trbo_optimize(&circuit, &config);

        assert!(distance(&original_unitary, &result.get_unitary()) < 1e-6);
        assert!(
            result.num_params() < circuit.num_params(),
            "expected at least one angle to round away, {} Rz gates remain",
            result.num_params()
        );
    }

    #[test]
    fn already_optimal_block_is_left_alone_or_improved_never_worsened() {
        let mut circuit = Circuit::new(1);
        circuit.push(Gate::T, vec![0]);
        let original_unitary = circuit.get_unitary();
        let config = TrboConfig::default();
        let result = trbo_optimize(&circuit, &config);
        assert!(distance(&original_unitary, &result.get_unitary()) < 1e-6);
        assert!(!better_circuit(&result, &circuit), "a pure-Clifford+T circuit should not get worse");
    }

    #[test]
    fn better_circuit_prefers_fewer_rz_gates() {
        let mut a = Circuit::new(1);
        a.push(Gate::Rz(0.3), vec![0]);
        let b = Circuit::new(1);
        assert!(better_circuit(&a, &b));
        assert!(!better_circuit(&b, &a));
    }
}
