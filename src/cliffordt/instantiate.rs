//! Shared nonlinear-least-squares fitting engine.
//!
//! The generic core (`lm_fit`/`lm_fit_multistart`) wraps the `rust-cv`
//! `levenberg-marquardt` crate -- a nalgebra-native port of MINPACK's
//! trust-region Levenberg-Marquardt, verified bit-identical to MINPACK on
//! its own test suite -- rather than the hand-rolled "bump lambda by 10x
//! and retry" solver this module started with. Callers supply one combined
//! `&[f64] -> (residuals, jacobian)` closure rather than two separate ones:
//! `fidelity_residuals_and_jacobian` computes both together, its analytic
//! derivative built from `Circuit::unitary_and_rz_derivatives`'s
//! prefix/suffix per-gate derivatives. `NlsProblem` below caches that pair
//! per parameter point (the crate's own trust-region loop always calls
//! `residuals()` then `jacobian()` once each at the *same* current point
//! before moving to a trial point elsewhere, which correctly invalidates
//! the cache via `set_params`) -- so the shared prefix/suffix sweep behind
//! both only happens once per point, not once per trait method call.
//!
//! `instantiate_from`/`instantiate_multistart` specialize it to "fit a
//! circuit template's Rz angles (plus a free global phase) to a target
//! unitary" -- used by Stage 2's re-fit. Other callers can build their own
//! combined closures (e.g. fidelity plus extra cost terms) on top of the
//! same generic core.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cliffordt::matrix::{C64, Unitary};
use crate::cliffordt::qgate_circuit::Circuit;

/// Adapts a combined `&[f64] -> (DVector<f64>, DMatrix<f64>)`
/// residual-and-Jacobian closure to the crate's `LeastSquaresProblem`
/// trait, caching its one result per parameter point (see module docs).
struct NlsProblem<'a, C>
where
    C: Fn(&[f64]) -> (DVector<f64>, DMatrix<f64>) + Sync,
{
    combined_fn: &'a C,
    params: DVector<f64>,
    cache: RefCell<Option<(DVector<f64>, DMatrix<f64>)>>,
}

impl<'a, C> NlsProblem<'a, C>
where
    C: Fn(&[f64]) -> (DVector<f64>, DMatrix<f64>) + Sync,
{
    fn new(combined_fn: &'a C, params: DVector<f64>) -> Self {
        NlsProblem { combined_fn, params, cache: RefCell::new(None) }
    }

    fn ensure_cached(&self) {
        let mut cache = self.cache.borrow_mut();
        if cache.is_none() {
            *cache = Some((self.combined_fn)(self.params.as_slice()));
        }
    }
}

impl<'a, C> LeastSquaresProblem<f64, Dyn, Dyn> for NlsProblem<'a, C>
where
    C: Fn(&[f64]) -> (DVector<f64>, DMatrix<f64>) + Sync,
{
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params = x.clone();
        *self.cache.get_mut() = None;
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        self.ensure_cached();
        Some(self.cache.borrow().as_ref().unwrap().0.clone())
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        self.ensure_cached();
        Some(self.cache.borrow().as_ref().unwrap().1.clone())
    }
}

/// Result of one fit: the parameter vector and the resulting residual norm.
pub struct FitResult {
    pub params: Vec<f64>,
    pub cost_norm: f64,
}

/// Single Levenberg-Marquardt run of `residual_fn` from `x0`, via the
/// `levenberg-marquardt` crate. Skips the solve entirely once `stop_early`
/// is already set (some other start in the same `lm_fit_multistart` batch
/// already reached `success_threshold`) -- only prevents launching a new
/// solve; one already in flight keeps running to completion.
pub fn lm_fit(
    combined_fn: impl Fn(&[f64]) -> (DVector<f64>, DMatrix<f64>) + Sync, x0: &[f64],
    max_iters: usize, success_threshold: f64, stop_early: &AtomicBool,
) -> FitResult {
    let n = x0.len();
    if n == 0 {
        // Nothing to vary (e.g. a template with zero free Rz angles, now
        // that the global phase is resolved analytically inside
        // `residual_fn` rather than fit as an extra parameter) -- a 0x0
        // normal-equations solve isn't something the solver handles, and
        // there's nothing to optimize anyway.
        let (r, _) = combined_fn(x0);
        return FitResult { params: Vec::new(), cost_norm: r.norm_squared().max(0.0).sqrt() };
    }
    if stop_early.load(Ordering::Relaxed) {
        let (r, _) = combined_fn(x0);
        return FitResult { params: x0.to_vec(), cost_norm: r.norm_squared().max(0.0).sqrt() };
    }

    let problem = NlsProblem::new(&combined_fn, DVector::from_column_slice(x0));
    // `with_tol` governs *convergence detection* (stop once further steps
    // stop helping), not "good enough" -- set tight so the solver runs to
    // genuine numerical convergence rather than a loose relative-change
    // criterion. `with_patience` caps total residual evaluations at
    // `max_iters * (n + 1)`, this module's previous "at most `max_iters`
    // outer iterations" bound translated into the crate's own units.
    let (result, report) =
        LevenbergMarquardt::new().with_tol(1e-14).with_patience(max_iters).minimize(problem);
    let cost_norm = (2.0 * report.objective_function).sqrt();

    if cost_norm < success_threshold {
        stop_early.store(true, Ordering::Relaxed);
    }

    FitResult { params: result.params.as_slice().to_vec(), cost_norm }
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
/// points, returning the best (lowest-cost) result. All starts share one
/// `AtomicBool`: the moment any of them reaches `success_threshold`, the
/// rest stop at their next iteration boundary instead of grinding on to
/// `max_iters` -- pass `f64::INFINITY` to disable this and always run every
/// start to its own convergence (e.g. when the caller genuinely wants the
/// best achievable fit, not just "good enough").
pub fn lm_fit_multistart(
    combined_fn: impl Fn(&[f64]) -> (DVector<f64>, DMatrix<f64>) + Sync, n_params: usize,
    extra_starts: &[Vec<f64>], n_random_starts: usize, max_iters: usize, seed: u64,
    success_threshold: f64,
) -> FitResult {
    let mut starts: Vec<Vec<f64>> = extra_starts.to_vec();
    for i in 0..n_random_starts {
        let mut rng =
            Xorshift64(seed.wrapping_add(i as u64).wrapping_mul(2685821657736338717).max(1));
        starts.push((0..n_params).map(|_| rng.next_f64()).collect());
    }

    let stop_early = AtomicBool::new(false);
    starts
        .par_iter()
        .map(|x0| lm_fit(&combined_fn, x0, max_iters, success_threshold, &stop_early))
        .reduce(
            || FitResult { params: vec![0.0; n_params], cost_norm: f64::INFINITY },
            |a, b| if a.cost_norm <= b.cost_norm { a } else { b },
        )
}

/// Global-phase-invariant residual vector (real, imag interleaved over
/// every matrix entry) for `circuit_template` with its Rz angles set to
/// `params`, against `target`, and its analytic Jacobian, computed
/// together off one shared `Circuit::unitary_and_rz_derivatives` call (one
/// forward + one backward sweep over the circuit's gates, total) instead
/// of two independent sweeps -- this is the combined closure `NlsProblem`
/// caches per parameter point. The aligning phase is computed analytically
/// (see below) rather than fit as an extra free parameter: adding a fitted
/// phase parameter instead would force the optimizer to also resolve an
/// extra periodic dimension that has a closed-form answer at every
/// evaluation -- a strictly harder landscape for no benefit. Differentiates
/// through the phase-alignment
/// term analytically too: writing `s = tr(built^dagger target)`, the
/// aligning phase's `rot = exp(i*phase) = s/|s|` (a standard identity for
/// `phase = arg(s)`), and for a *real* parameter `theta`, `d(rot)/dtheta =
/// (ds - rot * Re(ds * conj(rot))) / |s|` (derived from `|s|^2 = s *
/// conj(s)` and the quotient rule).
///
/// `s = 0` exactly (e.g. `built` and `target` differing by a Pauli, which
/// has zero trace -- a common case in this pipeline's own Clifford-heavy
/// gate set, not a rare corner) makes `rot`/`d_rot` a division by zero;
/// `global_phase_between`'s `atan2`-based `arg()` sidesteps this silently
/// (`atan2(0, 0) = 0` by convention), but this direct `s/|s|` form doesn't.
/// Falls back to `rot = 1` (matching that same `arg(0) = 0` convention) and
/// `d_rot = 0` there -- `s = 0` is also a genuine singular point of
/// `arg(s(theta))` itself (undefined slope, not just an unlucky division),
/// so there is no "more correct" derivative to fall back to; the residual
/// value stays finite and well-defined, only its derivative through this
/// one term is approximated as flat at that exact point.
pub fn fidelity_residuals_and_jacobian(
    circuit_template: &Circuit, target: &Unitary, params: &[f64],
) -> (DVector<f64>, DMatrix<f64>) {
    const SINGULAR_EPS: f64 = 1e-12;
    let mut c = circuit_template.clone();
    c.set_params(params);
    let (built, derivs) = c.unitary_and_rz_derivatives();
    let dim = built.nrows();

    let s = (built.adjoint() * target).trace();
    let s_abs = s.norm();
    let rot = if s_abs > SINGULAR_EPS { s / C64::new(s_abs, 0.0) } else { C64::new(1.0, 0.0) };

    let mut r = DVector::zeros(2 * dim * dim);
    let mut row = 0usize;
    for i in 0..dim {
        for j in 0..dim {
            let diff = built[(i, j)] * rot - target[(i, j)];
            r[row] = diff.re;
            r[row + 1] = diff.im;
            row += 2;
        }
    }

    let n = params.len();
    let mut jac = DMatrix::zeros(2 * dim * dim, n);
    for (k, d_built) in derivs.iter().enumerate() {
        let ds = (d_built.adjoint() * target).trace();
        let d_rot = if s_abs > SINGULAR_EPS {
            let cross = (ds * rot.conj()).re;
            (ds - rot * C64::new(cross, 0.0)) / C64::new(s_abs, 0.0)
        } else {
            C64::new(0.0, 0.0)
        };

        let mut row = 0usize;
        for i in 0..dim {
            for j in 0..dim {
                let d_diff = d_built[(i, j)] * rot + built[(i, j)] * d_rot;
                jac[(row, k)] = d_diff.re;
                jac[(row + 1, k)] = d_diff.im;
                row += 2;
            }
        }
    }
    (r, jac)
}

/// Fit `circuit_template`'s Rz angles to best match `target` (up to global
/// phase, resolved analytically inside `fidelity_residuals_and_jacobian`),
/// from several random starts plus the template's current parameters and an
/// all-zero start (often already close). Returns just the best-fit
/// parameters -- callers that also want the achieved fit quality can
/// recompute it directly (e.g. via `matrix::distance` against the target
/// unitary with those parameters applied).
pub fn instantiate_multistart(
    circuit_template: &Circuit, target: &Unitary, n_starts: usize, max_iters: usize, seed: u64,
    success_threshold: f64,
) -> Vec<f64> {
    let n_rz = circuit_template.num_params();

    let extra_starts = vec![vec![0.0; n_rz], circuit_template.params()];

    let template = circuit_template.clone();
    let target = target.clone();
    let fit = lm_fit_multistart(
        move |p| fidelity_residuals_and_jacobian(&template, &target, p),
        n_rz,
        &extra_starts,
        n_starts,
        max_iters,
        seed,
        success_threshold,
    );
    fit.params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::qgate_circuit::Gate;

    #[test]
    fn zero_trace_overlap_does_not_produce_nan() {
        // An empty (identity) circuit against an H target: trace(I^dagger *
        // H) = trace(H) = 0 exactly -- a common case in this pipeline's own
        // Clifford-heavy gate set (X, Y, Z, H all have zero trace), not a
        // contrived corner. Regression test for a real bug: the direct
        // `s / |s|` formula divides by zero here, and `f64::max` silently
        // turns the resulting NaN cost into a false "perfect fit" instead
        // of the correct large distance.
        let template = Circuit::new(1);
        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::H, vec![0]);
        let target = target_circuit.get_unitary();

        let (r, jac) = fidelity_residuals_and_jacobian(&template, &target, &[]);
        assert!(r.iter().all(|v| v.is_finite()), "residual contains NaN/Inf: {r:?}");
        assert_eq!(jac.ncols(), 0);
        assert!(r.norm() > 0.5, "identity should be far from H, got residual norm {}", r.norm());
    }

    #[test]
    fn fidelity_jacobian_matches_finite_difference() {
        let mut template = Circuit::new(2);
        template.push(Gate::H, vec![0]);
        template.push(Gate::Rz(0.2), vec![0]);
        template.push(Gate::Cx, vec![0, 1]);
        template.push(Gate::Rz(-0.5), vec![1]);
        template.push(Gate::S, vec![1]);
        template.push(Gate::Rz(0.9), vec![0]);

        let mut target_circuit = template.clone();
        target_circuit.set_params(&[0.33, -1.1, 0.7]);
        let target = target_circuit.get_unitary();

        let params = vec![0.6, 0.1, -0.4];
        let (_, analytic) = fidelity_residuals_and_jacobian(&template, &target, &params);

        const H: f64 = 1e-6;
        for k in 0..params.len() {
            let mut p_plus = params.clone();
            p_plus[k] += H;
            let mut p_minus = params.clone();
            p_minus[k] -= H;
            let (r_plus, _) = fidelity_residuals_and_jacobian(&template, &target, &p_plus);
            let (r_minus, _) = fidelity_residuals_and_jacobian(&template, &target, &p_minus);
            let numeric_col = (r_plus - r_minus) / (2.0 * H);
            let diff = (analytic.column(k) - numeric_col).norm();
            assert!(diff < 1e-6, "column {k}: analytic vs numeric Jacobian differ by {diff}");
        }
    }

    #[test]
    fn instantiate_recovers_known_rz_angle() {
        let mut template = Circuit::new(1);
        template.push(Gate::Rz(0.0), vec![0]);
        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::Rz(0.9), vec![0]);
        let target = target_circuit.get_unitary();

        let params = instantiate_multistart(&template, &target, 4, 100, 42, 1e-8);

        let mut fitted = template.clone();
        fitted.set_params(&params);
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

        let params = instantiate_multistart(&template, &target, 8, 200, 7, 1e-8);
        let mut fitted = template.clone();
        fitted.set_params(&params);
        assert!(crate::cliffordt::matrix::distance(&target, &fitted.get_unitary()) < 1e-6);
    }

    #[test]
    fn instantiate_reports_large_distance_when_template_cannot_reach_target() {
        let template = Circuit::new(1);
        let mut target_circuit = Circuit::new(1);
        target_circuit.push(Gate::Rz(1.3), vec![0]);
        let target = target_circuit.get_unitary();
        let params = instantiate_multistart(&template, &target, 4, 50, 1, 1e-8);
        let mut fitted = template.clone();
        fitted.set_params(&params);
        assert!(crate::cliffordt::matrix::distance(&target, &fitted.get_unitary()) > 0.1);
    }
}
