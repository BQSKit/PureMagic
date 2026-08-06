//! Stage 4: final synthesis of whatever continuous rotation survives all
//! earlier stages, via a three-tier dispatch: exact Clifford hit (free) ->
//! near-Clifford numerical-stability guard -> cyclosynth's joint ZYZ
//! lattice search -> rsgridsynth fallback (used directly for the
//! independent per-axis path, and as the safety net when cyclosynth is
//! skipped or fails).

use std::collections::HashMap;
use std::sync::Mutex;

use cyclosynth::synthesis::Synthesizer;
use cyclosynth::synthesis::angle::Angle;
use rsgridsynth::config::config_from_theta_epsilon;
use rsgridsynth::gridsynth::gridsynth_gates;

use crate::cliffordt::clifford::CliffordTable;
use crate::cliffordt::matrix::{C64, Unitary};
use crate::cliffordt::qgate_circuit::{Circuit, Gate};

/// rsgridsynth keeps its arbitrary-precision search state in a
/// process-global `AtomicUsize` (`PREC_BITS` in its own `common.rs`) rather
/// than per-call state, so concurrent calls on different threads can read
/// or reset each other's in-flight precision state, making the (still
/// individually valid, within-epsilon) gate sequence depend on scheduling.
/// Serializing every call through this lock restores reproducibility for a
/// given seed; gridsynth's own search dominates each block's cost anyway,
/// so this has little effect on Stage 4's total runtime.
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
/// can behave unpredictably right at a near-singular point -- a per-prefix
/// L²-LLL blowup, not the SE search itself) and routed to the
/// independent-axis fallback instead. Empirically tuned as a time/T-count
/// tradeoff on QFT-shaped circuits; not re-validated against other
/// near-Clifford angle distributions.
const NEAR_CLIFFORD_MARGIN: f64 = 300.0;

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
/// epsilon/3: a per-axis split measurably inflated T-count for no
/// corresponding fidelity benefit.
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
/// `synthesize_u3`'s actual behavior does not match its own doc comment
/// ("U3(θ,φ,λ) ≡ ZYZ(α=φ, β=θ, γ=λ)", i.e. `Rz(phi)*Ry(theta)*Rz(lam)`):
/// it actually produces `Rz(phi_arg)*Ry(-theta_arg)*Rz(lam_arg)` with
/// `phi_arg`/`lam_arg` swapped relative to the doc's own labels.
/// Compensated here by calling it with `(-theta, lam, phi)` instead of
/// `(theta, phi, lam)`.
fn try_cyclosynth(target: &Unitary, config: &SynthConfig) -> Option<Circuit> {
    let (theta, phi, lam) = zyz_angles(target);
    let synth = Synthesizer::new(config.epsilon, false);
    let result = synth.synthesize_u3(Angle::Rad(-theta), Angle::Rad(lam), Angle::Rad(phi))?;
    let gates = result.gates?;

    let mut circuit = Circuit::new(1);
    for c in gates.chars() {
        // Uppercase = the gate; lowercase = its dagger -- not documented in
        // cyclosynth's own public doc comment, which only lists the
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

/// The expensive part of the full three-tier dispatch for one single-qubit
/// block's target unitary: near-Clifford guard -> cyclosynth -> gridsynth
/// fallback. `synthesize_block_cached` is the actual entry point (it adds
/// the exact-Clifford check ahead of this and caches this part -- the
/// exact-Clifford check is already cheap, a lookup against 24 entries, so
/// there's no benefit caching it too, only the actual numerical search).
fn synthesize_non_clifford(
    target: &Unitary, clifford_table: &CliffordTable, config: &SynthConfig,
) -> Circuit {
    if config.use_cyclosynth {
        let near_clifford =
            clifford_table.nearest_distance(target) < NEAR_CLIFFORD_MARGIN * config.epsilon;
        if !near_clifford {
            if let Some(circuit) = try_cyclosynth(target, config) {
                return circuit;
            }
        }
    }

    gridsynth_unitary(target, config)
}

/// Hashable, global-phase-invariant, rounded key for a 2x2 unitary: round
/// magnitudes first so the pivot tie-break is deterministic (unitarity
/// forces `|a|==|d|` and `|b|==|c|`), pick the largest-magnitude entry as
/// pivot, divide out its (unrounded) phase, then round every entry to 7
/// decimals -- coarse enough to absorb floating-point noise accumulated by
/// earlier stages, but far finer than any epsilon this pipeline
/// synthesizes to.
///
/// Keyed on the matrix directly, not a ZYZ-angle decomposition: two
/// matrices that are "the same" rotation can have genuinely different
/// (theta, phi, lam) representations at a branch boundary (e.g. `phi = pi`
/// vs `phi = -pi`), which an angle-based key would wrongly treat as
/// distinct.
fn canonical_key(target: &Unitary) -> [(i64, i64); 4] {
    const SCALE: f64 = 1e7;
    let round = |x: f64| (x * SCALE).round() as i64;

    let mut pivot = 0usize;
    let mut best_rounded_abs = f64::MIN;
    for i in 0..4 {
        let v = target[(i / 2, i % 2)];
        let rounded_abs = (v.norm() * SCALE).round() / SCALE;
        if rounded_abs > best_rounded_abs {
            best_rounded_abs = rounded_abs;
            pivot = i;
        }
    }
    let phase = target[(pivot / 2, pivot % 2)].arg();
    let rot = C64::new(phase.cos(), phase.sin());

    let mut out = [(0i64, 0i64); 4];
    for i in 0..4 {
        let v = target[(i / 2, i % 2)] / rot;
        out[i] = (round(v.re), round(v.im));
    }
    out
}

/// Cache of already-synthesized non-Clifford single-qubit targets, keyed by
/// `canonical_key`. Avoids redundant cyclosynth/gridsynth searches for
/// rotations that repeat across a circuit.
#[derive(Default)]
pub struct SynthCache(Mutex<HashMap<[(i64, i64); 4], Circuit>>);

impl SynthCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct targets synthesized so far -- used to weight
    /// progress by genuine new synthesis work rather than plain call count,
    /// since most repeated rotations are instant cache hits and a call
    /// count would race through them and then stall on the rare expensive
    /// misses.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// Like `synthesize_block`, but looks up/populates `cache` around the
/// expensive non-Clifford path. Safe to call concurrently: a cache miss
/// racing with another thread's miss on the same angle just does the
/// (correct, just redundant) work twice rather than corrupting anything.
pub fn synthesize_block_cached(
    target: &Unitary, clifford_table: &CliffordTable, config: &SynthConfig, cache: &SynthCache,
) -> Circuit {
    if let Some(word) = clifford_table.exact_match(target, EXACTNESS_FLOOR) {
        return crate::cliffordt::clifford::circuit_from_word(&word);
    }

    let key = canonical_key(target);
    if let Some(hit) = cache.0.lock().unwrap().get(&key) {
        return hit.clone();
    }

    let circuit = synthesize_non_clifford(target, clifford_table, config);
    cache.0.lock().unwrap().insert(key, circuit.clone());
    circuit
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
            &[
                C64::new(h.cos(), 0.0),
                C64::new(-h.sin(), 0.0),
                C64::new(h.sin(), 0.0),
                C64::new(h.cos(), 0.0),
            ],
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
        let _ = c;
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
        let result = synthesize_block_cached(&target, &table, &config, &SynthCache::new());
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
        let result = synthesize_block_cached(&target, &table, &config, &SynthCache::new());
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synthesize_block_generic_target_via_cyclosynth_within_epsilon() {
        let table = CliffordTable::build();
        let target = Gate::Rz(0.37).matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 5, use_cyclosynth: true };
        let result = synthesize_block_cached(&target, &table, &config, &SynthCache::new());
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synthesize_block_via_cyclosynth_handles_nonzero_theta() {
        // A target built purely from Rz gates has theta == 0, where the
        // sign-convention bug in try_cyclosynth's doc comment can't show up
        // (Ry(0) is the identity either way). This target (Rz then H) has
        // theta == pi/2.
        let table = CliffordTable::build();
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.6), vec![0]);
        c.push(Gate::H, vec![0]);
        let target = c.get_unitary();
        let config = SynthConfig { epsilon: 1e-6, seed: 3, use_cyclosynth: true };
        let result = synthesize_block_cached(&target, &table, &config, &SynthCache::new());
        assert!(distance(&target, &result.get_unitary()) < 3e-6);
    }

    #[test]
    fn synth_cache_reuses_result_for_repeated_angle() {
        let table = CliffordTable::build();
        let target = Gate::Rz(0.37).matrix();
        let config = SynthConfig { epsilon: 1e-6, seed: 5, use_cyclosynth: false };
        let cache = SynthCache::new();
        let first = synthesize_block_cached(&target, &table, &config, &cache);
        let second = synthesize_block_cached(&target, &table, &config, &cache);
        assert_eq!(first, second, "cached result should be byte-identical to the first synthesis");
    }

    #[test]
    fn synth_cache_absorbs_floating_point_noise() {
        // Two targets that are "the same" rotation up to noise far smaller
        // than 1e-7 must still hit the same cache entry -- an
        // exact-bit-pattern key would treat these as distinct and
        // undercache real duplicates.
        let table = CliffordTable::build();
        let target = Gate::Rz(0.37).matrix();
        let mut noisy_circuit = Circuit::new(1);
        noisy_circuit.push(Gate::Rz(0.37 + 1e-10), vec![0]);
        let noisy_target = noisy_circuit.get_unitary();

        let config = SynthConfig { epsilon: 1e-6, seed: 5, use_cyclosynth: false };
        let cache = SynthCache::new();
        let first = synthesize_block_cached(&target, &table, &config, &cache);
        let second = synthesize_block_cached(&noisy_target, &table, &config, &cache);
        assert_eq!(
            first, second,
            "noise far below the 1e-7 bucket width should still hit the cache"
        );
    }

    #[test]
    fn synth_cache_key_ignores_global_phase() {
        // canonical_key must be global-phase invariant: a target multiplied
        // by an arbitrary unit phase is physically the identical gate.
        let target = Gate::Rz(0.55).matrix();
        let phased = target.map(|v| v * C64::new(0.0, 1.0));
        assert_eq!(canonical_key(&target), canonical_key(&phased));
    }
}
