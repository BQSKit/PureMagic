//! Shared nonlinear-least-squares fitting engine.
//!
//! The generic core (`lm_fit`/`lm_fit_multistart`) is a hand-rolled
//! Levenberg-Marquardt solver over `nalgebra` operating on an arbitrary
//! real-valued residual function, with a finite-difference Jacobian
//! (simpler and lower-risk than analytic per-gate gradients for a first
//! correct implementation; a straightforward place to optimize later).
//!
//! `instantiate_from`/`instantiate_multistart` specialize it to "fit a
//! circuit template's Rz angles (plus a free global phase, mirroring
//! TRbO's own `GlobalPhaseGate` trick) to a target unitary" -- used by
//! Stage 4's `ScanningGateRemovalPass` re-fit. Stage 5 (TRbO) builds its
//! own residual closure (fidelity + rounding-cost terms) on top of the same
//! generic core -- see the plan doc's Context section for why these two
//! turned out to need the same primitive.

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::cliffordt::matrix::{Unitary, C64};
use crate::cliffordt::qgate_circuit::Circuit;

fn jacobian_fd(residual_fn: &(impl Fn(&[f64]) -> DVector<f64> + Sync), params: &[f64]) -> DMatrix<f64> {
    let n = params.len();
    let r0 = residual_fn(params);
    let m = r0.len();
    let mut j = DMatrix::zeros(m, n);
    const H: f64 = 1e-6;
    for k in 0..n {
        let mut p = params.to_vec();
        p[k] += H;
        let r_plus = residual_fn(&p);
        p[k] -= 2.0 * H;
        let r_minus = residual_fn(&p);
        let col = (r_plus - r_minus) / (2.0 * H);
        j.set_column(k, &col);
    }
    j
}

/// Result of one fit: the parameter vector and the resulting residual norm.
pub struct FitResult {
    pub params: Vec<f64>,
    pub cost_norm: f64,
}

/// Single Levenberg-Marquardt run of `residual_fn` from `x0`.
pub fn lm_fit(residual_fn: impl Fn(&[f64]) -> DVector<f64> + Sync, x0: &[f64], max_iters: usize) -> FitResult {
    let n = x0.len();
    let mut params = DVector::from_column_slice(x0);
    let mut lambda = 1e-3_f64;
    let mut r = residual_fn(params.as_slice());
    let mut cost = r.norm_squared();

    for _ in 0..max_iters {
        if cost < 1e-24 {
            break;
        }
        let j = jacobian_fd(&residual_fn, params.as_slice());
        let jt = j.transpose();
        let jtj = &jt * &j;
        let jtr = &jt * &r;

        let mut improved = false;
        for _ in 0..30 {
            let mut a = jtj.clone();
            for i in 0..n {
                let diag = a[(i, i)].max(1e-12);
                a[(i, i)] += lambda * diag;
            }
            let delta = match a.lu().solve(&(-&jtr)) {
                Some(d) => d,
                None => {
                    lambda *= 10.0;
                    continue;
                }
            };
            let new_params = &params + &delta;
            let new_r = residual_fn(new_params.as_slice());
            let new_cost = new_r.norm_squared();
            if new_cost < cost {
                params = new_params;
                r = new_r;
                cost = new_cost;
                lambda = (lambda / 10.0).max(1e-14);
                improved = true;
                break;
            }
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
        if !improved {
            break;
        }
    }

    FitResult { params: params.as_slice().to_vec(), cost_norm: cost.max(0.0).sqrt() }
}

/// A simple deterministic pseudo-random generator (xorshift), used so
/// multi-start seeding doesn't require pulling in a separate RNG crate
/// dependency for this module. Not cryptographic; fine for optimizer
/// restarts.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x as f64 / u64::MAX as f64) * std::f64::consts::TAU
    }
}

/// Run several independent Levenberg-Marquardt fits of `residual_fn` (in
/// parallel via rayon) from `extra_starts` plus `n_random_starts` random
/// points, returning the best (lowest-cost) result.
pub fn lm_fit_multistart(
    residual_fn: impl Fn(&[f64]) -> DVector<f64> + Sync,
    n_params: usize,
    extra_starts: &[Vec<f64>],
    n_random_starts: usize,
    max_iters: usize,
    seed: u64,
) -> FitResult {
    let mut starts: Vec<Vec<f64>> = extra_starts.to_vec();
    for i in 0..n_random_starts {
        let mut rng = Xorshift64(seed.wrapping_add(i as u64).wrapping_mul(2685821657736338717).max(1));
        starts.push((0..n_params).map(|_| rng.next_f64()).collect());
    }

    starts
        .par_iter()
        .map(|x0| lm_fit(&residual_fn, x0, max_iters))
        .reduce(
            || FitResult { params: vec![0.0; n_params], cost_norm: f64::INFINITY },
            |a, b| if a.cost_norm <= b.cost_norm { a } else { b },
        )
}

/// Residual vector (real, imag interleaved over every matrix entry) for
/// `circuit_template` with its Rz angles set to `params[..num_params]` and
/// an extra free global phase `params[num_params]`, against `target`.
pub fn fidelity_residuals(circuit_template: &Circuit, target: &Unitary, params: &[f64]) -> DVector<f64> {
    let n_rz = circuit_template.num_params();
    let mut c = circuit_template.clone();
    c.set_params(&params[..n_rz]);
    let phase = params[n_rz];
    let rot = C64::new(phase.cos(), phase.sin());
    let built = c.get_unitary();
    let dim = built.nrows();
    let mut r = DVector::zeros(2 * dim * dim);
    let mut k = 0;
    for i in 0..dim {
        for j in 0..dim {
            let diff = built[(i, j)] * rot - target[(i, j)];
            r[k] = diff.re;
            r[k + 1] = diff.im;
            k += 2;
        }
    }
    r
}

/// Fit `circuit_template`'s Rz angles (plus a free global phase) to best
/// match `target`, from several random starts plus the template's current
/// parameters and an all-zero start (often already close).
pub fn instantiate_multistart(
    circuit_template: &Circuit,
    target: &Unitary,
    n_starts: usize,
    max_iters: usize,
    seed: u64,
) -> FitResultDistance {
    let n_rz = circuit_template.num_params();
    let n = n_rz + 1;

    let mut extra_starts = vec![vec![0.0; n]];
    let mut current = circuit_template.params();
    current.push(0.0);
    extra_starts.push(current);

    let template = circuit_template.clone();
    let target = target.clone();
    let fit = lm_fit_multistart(
        move |p| fidelity_residuals(&template, &target, p),
        n,
        &extra_starts,
        n_starts,
        max_iters,
        seed,
    );
    FitResultDistance { params: fit.params[..n_rz].to_vec(), distance: fit.cost_norm }
}

/// Like `FitResult`, but with the residual norm relabeled as a "distance"
/// (i.e. an operator-norm-scale error), and with the global phase already
/// dropped -- the shape callers of the circuit-specific `instantiate_*`
/// functions want.
pub struct FitResultDistance {
    pub params: Vec<f64>,
    pub distance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::qgate_circuit::Gate;

    #[test]
    fn instantiate_recovers_known_rz_angle() {
        let mut template = Circuit::new(1);
        template.push(Gate::Rz(0.0), vec![0]);
        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::Rz(0.9), vec![0]);
        let target = target_circuit.get_unitary();

        let result = instantiate_multistart(&template, &target, 4, 100, 42);
        assert!(result.distance < 1e-6, "distance too large: {}", result.distance);

        let mut fitted = template.clone();
        fitted.set_params(&result.params);
        assert!(crate::cliffordt::matrix::distance(&target, &fitted.get_unitary()) < 1e-6);
    }

    #[test]
    fn instantiate_fits_multiple_free_angles() {
        let mut template = Circuit::new(1);
        template.push(Gate::Rz(0.0), vec![0]);
        template.push(Gate::H, vec![0]);
        template.push(Gate::Rz(0.0), vec![0]);

        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::Rz(0.4), vec![0]);
        target_circuit.push(Gate::H, vec![0]);
        target_circuit.push(Gate::Rz(-1.1), vec![0]);
        let target = target_circuit.get_unitary();

        let result = instantiate_multistart(&template, &target, 8, 200, 7);
        assert!(result.distance < 1e-6, "distance too large: {}", result.distance);
    }

    #[test]
    fn instantiate_reports_large_distance_when_template_cannot_reach_target() {
        let template = Circuit::new(1);
        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::Rz(1.3), vec![0]);
        let target = target_circuit.get_unitary();
        let result = instantiate_multistart(&template, &target, 4, 50, 1);
        assert!(result.distance > 0.1);
    }
}
