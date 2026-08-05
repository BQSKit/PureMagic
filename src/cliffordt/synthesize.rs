//! Stage 6: final synthesis of whatever continuous rotation survives all
//! earlier stages, via the three-tier dispatch described throughout this
//! session -- exact Clifford hit (free) -> near-Clifford numerical-
//! stability guard -> cyclosynth's joint ZYZ lattice search -> rsgridsynth
//! fallback (used directly for the independent per-axis path, and as the
//! safety net when cyclosynth is skipped or fails).

use std::sync::Mutex;

use cyclosynth::synthesis::angle::Angle;
use cyclosynth::synthesis::Synthesizer;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;

use crate::cliffordt::clifford::CliffordTable;
use crate::cliffordt::matrix::{Unitary, C64};
use crate::cliffordt::qgate_circuit::{Circuit, Gate};

/// rsgridsynth keeps its arbitrary-precision search state in a
/// process-global `AtomicUsize` (`PREC_BITS` in its own `common.rs`),
/// read/bumped by `gridsynth_gates` itself rather than threaded through as
/// per-call state. That's fine sequentially, but under this pipeline's
/// block-level parallelism (`Circuit::for_each_block`), concurrent calls on
/// different threads can read or reset each other's in-flight precision
/// state, making the exact (still individually valid, within-epsilon) gate
/// sequence depend on scheduling -- confirmed empirically (two runs with
/// the same seed produced the same T-count but a different gate sequence
/// in one block). Serializing every call through this lock restores
/// reproducibility for a given seed. Gridsynth's own search dominates each
/// block's cost anyway (up to 3 calls per block, ZYZ decomposition and
/// bookkeeping around it is cheap by comparison), so this reverts Stage 6
/// specifically to close to its pre-parallelization sequential time --
/// small relative to total runtime, and Stage 4/5 (which never call
/// gridsynth) are unaffected.
static GRIDSYNTH_LOCK: Mutex<()> = Mutex::new(());

pub struct SynthConfig {
    pub epsilon: f64,
    pub seed: u64,
    /// Whether to try cyclosynth's joint synthesis before falling back to
    /// independent per-axis gridsynth.
    pub use_cyclosynth: bool,
}

impl Default for SynthConfig {
    fn default() -> Self {
        SynthConfig { epsilon: 1e-8, seed: 0, use_cyclosynth: false }
    }
}

/// Exact-match tolerance -- tighter than the synthesis epsilon, since an
/// "exact" hit should cost genuinely zero approximation error, not merely
/// stay within the overall budget.
const EXACTNESS_FLOOR: f64 = 1e-12;
/// A block within this many multiples of epsilon of a Clifford point is
/// treated as "too close to call" for cyclosynth's lattice search (which
/// can behave unpredictably right at a near-singular point) and routed to
/// the independent-axis fallback instead.
const NEAR_CLIFFORD_MARGIN: f64 = 10.0;

/// Extract ZYZ Euler angles `(theta, phi, lam)` such that
/// `target ~ Rz(phi) * Ry(theta) * Rz(lam)` (up to global phase) --
/// the "qiskit/bqskit" U3 convention cyclosynth's `synthesize_u3` expects.
pub fn zyz_angles(target: &Unitary) -> (f64, f64, f64) {
    let det = target[(0, 0)] * target[(1, 1)] - target[(0, 1)] * target[(1, 0)];
    let phase = det.arg() / 2.0;
    let norm = C64::new(phase.cos(), phase.sin());
    let a = target[(0, 0)] / norm;
    let c = target[(1, 0)] / norm;
    let d = target[(1, 1)] / norm;

    let theta = 2.0 * c.norm().atan2(a.norm());

    const TINY: f64 = 1e-12;
    if c.norm() < TINY {
        (theta, 2.0 * d.arg(), 0.0)
    } else if a.norm() < TINY {
        (theta, 0.0, -2.0 * c.arg())
    } else {
        let sum = 2.0 * d.arg();
        let diff = 2.0 * c.arg();
        (theta, (sum + diff) / 2.0, (sum - diff) / 2.0)
    }
}

/// Run rsgridsynth on `Rz(theta)` and return the resulting gate word in
/// circuit (temporal) order. rsgridsynth's own string convention is
/// "leftmost = last applied" (built up by successive right-multiplication
/// during its search), the opposite of cyclosynth's -- reversed here so
/// every caller in this module can just push characters left-to-right.
fn gridsynth_rz_word(theta: f64, epsilon: f64, seed: u64) -> Vec<Gate> {
    let _guard = GRIDSYNTH_LOCK.lock().unwrap();
    let mut config = config_from_theta_epsilon(theta, epsilon, seed, false, true);
    let result = gridsynth_gates(&mut config);
    result
        .gates
        .chars()
        .rev()
        .filter_map(|c| match c {
            'H' => Some(Gate::H),
            'S' => Some(Gate::S),
            'T' => Some(Gate::T),
            'X' => Some(Gate::X),
            'W' => None, // pure global phase, no physical gate
            _ => None,
        })
        .collect()
}

/// Approximate `Ry(theta)` via a fixed Clifford change of basis around an
/// Rz synthesis: `S*H` conjugates Z to Y (established directly: H maps
/// Z->X, then S maps X->Y), so `Ry(theta) = (S*H) * Rz(theta) * (S*H)^dagger`.
/// In circuit/temporal order that's `(S*H)^dagger` first, then the Rz
/// word, then `S*H`.
fn ry_word_via_rz(theta: f64, epsilon: f64, seed: u64) -> Vec<Gate> {
    let mut word = vec![Gate::Sdg, Gate::H]; // (S*H)^dagger = H * Sdg (matrix); circuit order Sdg,H
    word.extend(gridsynth_rz_word(theta, epsilon, seed));
    word.push(Gate::H); // S*H (matrix); circuit order H,S
    word.push(Gate::S);
    word
}

/// Independent per-axis synthesis: decompose into ZYZ Euler angles and
/// synthesize each axis separately via rsgridsynth.
///
/// Uses the *full* `config.epsilon` per axis, not a conservatively-split
/// epsilon/3 -- matching `data_processing/compile_cliffordt.py`'s own
/// `gridsynth_precision`, which computes one precision from the whole
/// synthesis epsilon and reuses it for every Rz gate regardless of how
/// many end up needing synthesis in a block. A per-axis split was tried
/// initially (worst-case triangle-inequality accounting) but measurably
/// inflated T-count relative to the Python pipeline for no corresponding
/// fidelity benefit the accumulated-error-bound bookkeeping doesn't
/// already cover.
pub fn gridsynth_unitary(target: &Unitary, config: &SynthConfig) -> Circuit {
    let (theta, phi, lam) = zyz_angles(target);

    let mut circuit = Circuit::new(1);
    for gate in gridsynth_rz_word(lam, config.epsilon, config.seed) {
        circuit.push(gate, vec![0]);
    }
    for gate in ry_word_via_rz(theta, config.epsilon, config.seed.wrapping_add(1)) {
        circuit.push(gate, vec![0]);
    }
    for gate in gridsynth_rz_word(phi, config.epsilon, config.seed.wrapping_add(2)) {
        circuit.push(gate, vec![0]);
    }
    circuit
}

/// Cyclosynth's joint ZYZ lattice search; `None` if it fails to find a
/// result (falls back to `gridsynth_unitary` at the call site).
///
/// Empirically confirmed (via round-trip testing against `zyz_angles`,
/// whose own convention is separately validated by direct matrix
/// reconstruction) that `synthesize_u3`'s actual behavior does not match
/// its doc comment ("U3(θ,φ,λ) ≡ ZYZ(α=φ, β=θ, γ=λ)", i.e.
/// `Rz(phi)*Ry(theta)*Rz(lam)`): it actually produces
/// `Rz(phi_arg)*Ry(-theta_arg)*Rz(lam_arg)` with `phi_arg`/`lam_arg`
/// swapped relative to the doc's own labels. Compensated here by calling
/// it with `(-theta, lam, phi)` instead of `(theta, phi, lam)`, which was
/// verified to reproduce the intended `Rz(phi)*Ry(theta)*Rz(lam)` target
/// to cyclosynth's own reported precision.
fn try_cyclosynth(target: &Unitary, config: &SynthConfig) -> Option<Circuit> {
    let (theta, phi, lam) = zyz_angles(target);
    let synth = Synthesizer::new(config.epsilon, false);
    let result = synth.synthesize_u3(Angle::Rad(-theta), Angle::Rad(lam), Angle::Rad(phi))?;
    let gates = result.gates?;

    let mut circuit = Circuit::new(1);
    for c in gates.chars() {
        // Uppercase = the gate; lowercase = its dagger (confirmed against
        // cyclosynth's own source -- e.g. `synthesis/decomposer.rs`'s
        // `'s' => U2T::s().dagger()`, `'t' => U2T::t().dagger()` -- not
        // documented in the public doc comment, which only lists the
        // uppercase alphabet.
        let gate = match c {
            'H' => Gate::H,
            'S' => Gate::S,
            's' => Gate::Sdg,
            'T' => Gate::T,
            't' => Gate::Tdg,
            'X' => Gate::X,
            'Y' => Gate::Y,
            'Z' => Gate::Z,
            _ => continue,
        };
        circuit.push(gate, vec![0]);
    }
    Some(circuit)
}

/// Full three-tier dispatch for one single-qubit block's target unitary.
pub fn synthesize_block(target: &Unitary, clifford_table: &CliffordTable, config: &SynthConfig) -> Circuit {
    if let Some(word) = clifford_table.exact_match(target, EXACTNESS_FLOOR) {
        return crate::cliffordt::clifford::circuit_from_word(&word);
    }

    if config.use_cyclosynth {
        let near_clifford = clifford_table.nearest_distance(target) < NEAR_CLIFFORD_MARGIN * config.epsilon;
        if !near_clifford {
            if let Some(circuit) = try_cyclosynth(target, config) {
                return circuit;
            }
        }
    }

    gridsynth_unitary(target, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    fn ry_matrix(theta: f64) -> Unitary {
        let h = theta / 2.0;
        Unitary::from_row_slice(
            2,
            2,
            &[C64::new(h.cos(), 0.0), C64::new(-h.sin(), 0.0), C64::new(h.sin(), 0.0), C64::new(h.cos(), 0.0)],
        )
    }

    fn zyz_matrix(theta: f64, phi: f64, lam: f64) -> Unitary {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(lam), vec![0]);
        let ry = ry_matrix(theta);
        // Insert Ry directly via a Block wrapping a synthetic single-gate
        // circuit is overkill; just multiply matrices directly here.
        let rz_lam = Gate::Rz(lam).matrix();
        let rz_phi = Gate::Rz(phi).matrix();
        let _ = c; // silence unused warning from the scratch circuit above
        rz_phi * ry * rz_lam
    }

    #[test]
    fn zyz_angles_round_trip_on_hadamard() {
        let h = Gate::H.matrix();
        let (theta, phi, lam) = zyz_angles(&h);
        let rebuilt = zyz_matrix(theta, phi, lam);
        assert!(distance(&h, &rebuilt) < 1e-10);
    }

    #[test]
    fn zyz_angles_round_trip_on_t_gate() {
        let t = Gate::T.matrix();
        let (theta, phi, lam) = zyz_angles(&t);
        let rebuilt = zyz_matrix(theta, phi, lam);
        assert!(distance(&t, &rebuilt) < 1e-10);
    }

    #[test]
    fn zyz_angles_round_trip_theta_near_zero() {
        let target = Gate::Rz(0.42).matrix();
        let (theta, phi, lam) = zyz_angles(&target);
        assert!(theta.abs() < 1e-8);
        let rebuilt = zyz_matrix(theta, phi, lam);
        assert!(distance(&target, &rebuilt) < 1e-10);
    }

    #[test]
    fn zyz_angles_round_trip_general_rotation() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(1.7), vec![0]);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(-0.6), vec![0]);
        let target = c.get_unitary();
        let (theta, phi, lam) = zyz_angles(&target);
        let rebuilt = zyz_matrix(theta, phi, lam);
        assert!(distance(&target, &rebuilt) < 1e-10);
    }

    #[test]
    fn synthesize_block_exact_clifford_costs_zero_t_gates() {
        let table = CliffordTable::build();
        let target = Gate::S.matrix();
        let config = SynthConfig::default();
        let result = synthesize_block(&target, &table, &config);
        assert!(!result.ops.iter().any(|op| matches!(op.gate, Gate::T | Gate::Tdg)));
        assert!(distance(&target, &result.get_unitary()) < 1e-8);
    }

    #[test]
    fn gridsynth_unitary_approximates_generic_rotation_within_epsilon() {
        let target = Gate::Rz(0.123456789).matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 1, use_cyclosynth: false };
        let result = gridsynth_unitary(&target, &config);
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn gridsynth_unitary_approximates_hadamard_within_epsilon() {
        let target = Gate::H.matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 2, use_cyclosynth: false };
        let result = gridsynth_unitary(&target, &config);
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synthesize_block_generic_target_via_gridsynth_within_epsilon() {
        let table = CliffordTable::build();
        let target = Gate::Rz(0.37).matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 5, use_cyclosynth: false };
        let result = synthesize_block(&target, &table, &config);
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synthesize_block_generic_target_via_cyclosynth_within_epsilon() {
        let table = CliffordTable::build();
        let target = Gate::Rz(0.37).matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 5, use_cyclosynth: true };
        let result = synthesize_block(&target, &table, &config);
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synthesize_block_via_cyclosynth_handles_nonzero_theta() {
        // Regression test: cyclosynth's `synthesize_u3` does not actually
        // match its own doc comment for targets with a genuine Ry (theta)
        // component -- a target built purely from Rz gates (theta == 0)
        // can't expose this, since Ry(0) is the identity regardless of
        // sign convention. This target (Rz then H) has theta == pi/2.
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.6), vec![0]);
        c.push(Gate::H, vec![0]);
        let target = c.get_unitary();
        let config = SynthConfig { epsilon: 1e-6, seed: 3, use_cyclosynth: true };
        let result = synthesize_block(&target, &table, &config);
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }
}
