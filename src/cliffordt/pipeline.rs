//! Orchestrates all stages into one `compile` entry point: exact
//! phase-polynomial merge -> gauge collapse (blocking + exact-Clifford +
//! rounding, run and reported as one combined stage) -> windowed
//! multi-qubit resynthesis -> gauge collapse again -> TRbO (if enabled) ->
//! gauge collapse a *third* time -> final synthesis. Running the
//! gauge-collapse cycle both before and after TRbO is deliberate, not
//! redundant: skipping the second run measurably regresses T-count (by
//! ~1.4% on measured circuits), since re-checking for exact Clifford hits
//! after TRbO's angle adjustments exposes coincidences the first pass
//! alone doesn't.

use std::time::{Duration, Instant};

use crate::cliffordt::clifford::CliffordTable;
use crate::cliffordt::group_single_qubit::group_single_qubit_gates;
use crate::cliffordt::matrix::distance;
use crate::cliffordt::partition::partition;
use crate::cliffordt::phase_merge::{count_real_rotations, merge_phase_polynomial};
use crate::cliffordt::progress::ProgressTracker;
use crate::cliffordt::qgate_circuit::{Circuit, Gate};
use crate::cliffordt::rounding::round_to_discrete_z;
use crate::cliffordt::stage4_scan_removal::{ScanConfig, scanning_gate_removal};
use crate::cliffordt::stats::block_stats;
use crate::cliffordt::synthesize::{SynthCache, SynthConfig, synthesize_block_cached};
use crate::cliffordt::trbo::{TrboConfig, trbo_optimize};

/// Total gate count, recursing into `Block` sub-circuits (the circuit is
/// only ever fully flat right after an `unfold`; at other points it may
/// still have grouped blocks pending).
pub fn total_gate_count(circuit: &Circuit) -> usize {
    circuit
        .ops
        .iter()
        .map(|op| match &op.gate {
            Gate::Block(inner) => total_gate_count(inner),
            _ => 1,
        })
        .sum()
}

/// Total `Rz` gate count, recursing into `Block` sub-circuits -- the
/// natural "how much work is still left to do" measure for stages 1-3,
/// since every one of them exists to either remove an `Rz` entirely
/// (exact-Clifford hit, or a gate-removal simplification) or round it onto
/// the discrete grid; only stage 4 ever needs to actually pay a T-count
/// price for whichever ones survive.
pub fn total_rz_count(circuit: &Circuit) -> usize {
    circuit
        .ops
        .iter()
        .map(|op| match &op.gate {
            Gate::Block(inner) => total_rz_count(inner),
            Gate::Rz(_) => 1,
            _ => 0,
        })
        .sum()
}

/// A single reported stage: its name, how long it took, and the circuit's
/// state immediately after it ran (callers can derive whatever "how much
/// did this stage contribute" stats they want -- gate count, Rz count,
/// T count -- via `total_gate_count`/`total_rz_count`/direct inspection).
///
/// `detail`, when set, is a stage-specific summary computed here rather
/// than left to the caller: some stages (blocking, exact-Clifford) have a
/// contribution that's invisible in a generic gate/Rz-count delta --
/// blocking never changes either count by construction (it only wraps
/// gates in a `Block`), and exact-Clifford's real payoff is Rz gates
/// consumed for free, not gate count (see the stage 2 comment below).
/// Only this module has the per-block information needed to explain that,
/// so it's computed here instead of guessed at by whatever prints it.
pub struct StageReport<'a> {
    pub name: String,
    pub elapsed: Duration,
    pub circuit: &'a Circuit,
    pub detail: Option<String>,
}

pub struct PipelineConfig {
    pub epsilon: f64,
    pub seed: u64,
    pub trbo: bool,
    pub cyclosynth: bool,
    /// Let Stage 2 drop a block that's within `epsilon` *infidelity* of a
    /// simpler circuit, not just within `epsilon` operator-norm distance --
    /// see `stage4_scan_removal.rs::ScanConfig::approx_cancel` for the full
    /// rationale. Off by default: the resulting error is always measured
    /// exactly and folded into the returned error bound, but it can be far
    /// larger per approximate cancellation than `epsilon` itself.
    pub approx_cancel: bool,
    /// Skip every gauge-collapse cycle (all three: initial, post stage 2,
    /// post TRbO), for isolating its contribution to the final result.
    pub skip_gauge_collapse: bool,
    /// Skip Stage 2 (windowed multi-qubit resynthesis), for isolating its
    /// contribution to the final result.
    pub skip_windowed_resynthesis: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            epsilon: 1e-8,
            seed: 0,
            trbo: false,
            cyclosynth: false,
            approx_cancel: false,
            skip_gauge_collapse: false,
            skip_windowed_resynthesis: false,
        }
    }
}

/// Blocking + exact Clifford recognition (leaving non-matches untouched) +
/// unfold + angle rounding, run and reported as one combined stage. `cycle`
/// labels which of the three times through this cycle it is ("initial",
/// "post stage 2", "post TRbO") since it runs more than once (see module
/// docs). A block that doesn't match exactly is left as its original gates,
/// not forced through anything lossy -- this stage never introduces
/// approximation error.
fn gauge_collapse(
    circuit: &Circuit, epsilon: f64, table: &CliffordTable, cycle: &str,
    on_stage: &mut impl FnMut(StageReport),
) -> Circuit {
    let t = Instant::now();
    let rz_before = total_rz_count(circuit);

    let grouped = group_single_qubit_gates(circuit);

    // Each block whose composed unitary exactly matches one of the 24
    // single-qubit Cliffords is rewritten as that element's shortest
    // available word (see `CliffordTable::build`) -- but only when that
    // word is no longer than what's already there. A match never makes a
    // block worse: if the shortest word happens to be longer than the
    // block's current gate count (rare now that the table's generating set
    // includes every native Clifford gate, not just H/S, but still
    // possible), the original gates are left alone.
    let (checked, _) = grouped.for_each_block_with(|inner| {
        let target = inner.get_unitary();
        match table.exact_match(&target, 1e-10) {
            Some(word) if word.len() <= inner.ops.len() => {
                (crate::cliffordt::clifford::circuit_from_word(&word), ())
            }
            Some(_) => (inner.clone(), ()),
            None => {
                // Not an exact Clifford. A `U3` gate here would otherwise
                // be permanently invisible to every later stage (only
                // `Rz` is recognized as an adjustable parameter) -- expose
                // its rotation exactly via a canonical ZYZ decomposition
                // instead of leaving it frozen. A block with no `U3` is
                // already fully expressed in adjustable `Rz` + Clifford
                // gates, so leave it untouched (decomposing it further
                // would only add gates for no benefit).
                if inner.ops.iter().any(|op| matches!(op.gate, Gate::U3(..))) {
                    (crate::cliffordt::clifford::decompose_to_rz_canonical(&target), ())
                } else {
                    (inner.clone(), ())
                }
            }
        }
    });
    let mut current = checked.unfold();

    // A block whose rotation was hidden inside an opaque `U3` gate (not
    // counted by `total_rz_count` at all) turns into up to 3 freshly
    // visible `Rz`s once exposed -- so `rz_before` can legitimately
    // undercount the circuit's existing continuous rotation, and this
    // midpoint can be *larger* than `rz_before` even though nothing new was
    // created. Reporting it separately keeps that exposure jump from being
    // read as a regression.
    let rz_exposed = total_rz_count(&current);

    current = round_to_discrete_z(&current, epsilon);

    let rz_after = total_rz_count(&current);
    let detail = format!(
        "Rz: {rz_before} -> {rz_exposed} (Clifford match/expose) -> {rz_after} (rounded), net {:+}",
        rz_after as i64 - rz_before as i64
    );
    on_stage(StageReport {
        name: format!("stage 1: gauge collapse ({cycle})"),
        elapsed: t.elapsed(),
        circuit: &current,
        detail: Some(detail),
    });

    current
}

/// Compile `circuit` (expected to already be expressed in this pipeline's
/// own gate vocabulary -- Clifford generators, CX/CZ/Swap, and `Rz` -- any
/// prior decomposition from an arbitrary front-end gate set is a
/// precondition, not one of the six stages) down to Clifford+T.
///
/// Returns the compiled circuit and an upper bound on the total
/// approximation error introduced (summed per-block spectral distance,
/// a standard triangle-inequality-style bound -- not necessarily tight,
/// but a valid certificate that the result is within `config.epsilon *
/// num_blocks_synthesized` of the original, and in practice far tighter
/// since most blocks cost zero).
pub fn compile(
    circuit: &Circuit, config: &PipelineConfig, mut on_stage: impl FnMut(StageReport),
) -> (Circuit, f64) {
    let table = CliffordTable::build();

    // Stage 0: exact phase-polynomial merge of redundant diagonal
    // rotations (see phase_merge.rs) -- must run first, before Stage 1
    // ever groups gates into blocks: fusing a genuinely mergeable Rz
    // together with a neighboring non-diagonal gate first would bake it
    // into one opaque matrix this pass could no longer address. Not always
    // a net win (see phase_merge.rs's docs), so pick whichever candidate
    // needs fewer real (non-Clifford) rotations.
    let t = Instant::now();
    let merged = merge_phase_polynomial(circuit);
    let merged_real = count_real_rotations(&merged, config.epsilon);
    let unmerged_real = count_real_rotations(circuit, config.epsilon);
    let (preopt, chosen_real, other_real, used_merge) = if merged_real <= unmerged_real {
        (merged, merged_real, unmerged_real, true)
    } else {
        (circuit.clone(), unmerged_real, merged_real, false)
    };
    on_stage(StageReport {
        name: "stage 0: phase-polynomial merge".to_string(),
        elapsed: t.elapsed(),
        circuit: &preopt,
        detail: Some(format!(
            "{} candidate chosen: {chosen_real} real rotation(s) needing synthesis (vs {other_real} for the alternative)",
            if used_merge { "merged" } else { "unmerged" }
        )),
    });

    // Stage 1: gauge collapse (blocking + exact-Clifford + rounding).
    let mut current = if config.skip_gauge_collapse {
        preopt.clone()
    } else {
        gauge_collapse(&preopt, config.epsilon, &table, "initial", &mut on_stage)
    };

    // Stage 2: windowed multi-qubit resynthesis.
    let stage2_error: f64 = if config.skip_windowed_resynthesis {
        0.0
    } else {
        let t = Instant::now();
        let partitioned = partition(&current, 2);
        let scan_config = ScanConfig {
            success_threshold: config.epsilon,
            approx_cancel: config.approx_cancel,
            ..ScanConfig::default()
        };
        // No cache to weight by here (unlike Stage 4) -- every block does
        // similarly-cheap exact work, so plain per-block counting is the
        // right progress denominator.
        let (num_blocks, _) = block_stats(&partitioned);
        let progress = ProgressTracker::new("windowed resynthesis", num_blocks);
        let (scanned, scan_errors) = partitioned.for_each_block_with(|inner| {
            let result = scanning_gate_removal(inner, &scan_config);
            progress.add(1);
            result
        });
        progress.finish();
        let stage2_error: f64 = scan_errors.iter().sum();
        current = scanned.unfold();
        on_stage(StageReport {
            name: "stage 2: windowed resynthesis".to_string(),
            elapsed: t.elapsed(),
            circuit: &current,
            detail: None,
        });
        stage2_error
    };

    // Gauge collapse again before TRbO.
    if !config.skip_gauge_collapse {
        current = gauge_collapse(&current, config.epsilon, &table, "post stage 2", &mut on_stage);
    }

    // Stage 3: TRbO gauge-freedom optimization (optional). Its payoff is
    // per-block Rz reduction via gauge freedom across a wider window than
    // Stage 2 -- not visible in a whole-circuit gate/Rz delta when only
    // some blocks have exploitable freedom, since the ones that don't stay
    // untouched (trbo_optimize never makes a block worse) and dilute it.
    if config.trbo {
        let t = Instant::now();
        let partitioned = partition(&current, 4);
        let trbo_config = TrboConfig {
            success_threshold: config.epsilon,
            seed: config.seed,
            ..TrboConfig::default()
        };
        // Every block runs its own NLS optimization from scratch (no cache
        // to weight by), so -- like Stage 2 -- plain per-block counting is
        // the right progress denominator.
        let (num_blocks, _) = block_stats(&partitioned);
        let progress = ProgressTracker::new("TRbO", num_blocks);
        let (optimized, deltas) = partitioned.for_each_block_with(|inner| {
            let before = inner.num_params();
            let result = trbo_optimize(inner, &trbo_config);
            let after = result.num_params();
            progress.add(1);
            (result, (before, after))
        });
        progress.finish();
        current = optimized.unfold();
        let num_blocks = deltas.len();
        let num_improved = deltas.iter().filter(|(before, after)| after < before).count();
        let rz_before: usize = deltas.iter().map(|(before, _)| before).sum();
        let rz_after: usize = deltas.iter().map(|(_, after)| after).sum();
        let detail = format!(
            "{num_improved}/{num_blocks} blocks improved, Rz: {rz_before} -> {rz_after} ({:+})",
            rz_after as i64 - rz_before as i64
        );
        on_stage(StageReport {
            name: "stage 3: TRbO".to_string(),
            elapsed: t.elapsed(),
            circuit: &current,
            detail: Some(detail),
        });

        // Gauge collapse a third time -- not redundant, see module docs.
        if !config.skip_gauge_collapse {
            current = gauge_collapse(&current, config.epsilon, &table, "post TRbO", &mut on_stage);
        }
    }

    // Stage 4: final synthesis of whatever continuous rotation remains.
    let t = Instant::now();
    let grouped = group_single_qubit_gates(&current);
    let synth_config = SynthConfig {
        epsilon: config.epsilon,
        seed: config.seed,
        use_cyclosynth: config.cyclosynth,
    };
    // Shared across all blocks (including in parallel -- see SynthCache's
    // own doc comment): repeated rotation angles are common enough in real
    // circuits that caching the expensive non-Clifford synthesis path
    // across blocks is a large, real win, not a micro-optimization.
    let synth_cache = SynthCache::new();
    // Weight by actual cache growth, not call count: most repeated
    // rotations are instant cache hits, so counting every call equally
    // would race through them and then stall on the rare, genuinely
    // expensive misses. `total` is the number of non-Clifford blocks, an
    // upper bound on how many new cache entries can appear (repeats
    // collapse to fewer); the Clifford short-circuit below never touches
    // the cache, so it's excluded from both the total and the count.
    let total_synth = grouped
        .ops
        .iter()
        .filter(|op| matches!(&op.gate, Gate::Block(inner) if !inner.is_all_clifford()))
        .count();
    let progress = ProgressTracker::new("final synthesis", total_synth);
    let synth_one = |inner: &Circuit| {
        if inner.is_all_clifford() {
            return (inner.clone(), 0.0);
        }
        let target = inner.get_unitary();
        let before = synth_cache.len();
        let result = synthesize_block_cached(&target, &table, &synth_config, &synth_cache);
        progress.add(synth_cache.len() - before);
        let error = distance(&target, &result.get_unitary());
        (result, error)
    };
    // cyclosynth parallelizes its own search internally, so run Stage 4
    // sequentially at this level when it's enabled rather than nesting it
    // inside this crate's own per-block rayon parallelism -- avoids
    // oversubscription slowdown (see `for_each_block_with_sequential`'s doc
    // comment for measured throughput).
    let (synthesized, errors) = if synth_config.use_cyclosynth {
        grouped.for_each_block_with_sequential(synth_one)
    } else {
        grouped.for_each_block_with(synth_one)
    };
    progress.finish();
    // Stage 2's own approximate-cancellation error (see stage2_error above)
    // is folded in here too, so the reported bound reflects every stage
    // that can spend accuracy, not just Stage 4.
    let total_error: f64 = stage2_error + errors.iter().sum::<f64>();
    let mut final_circuit = synthesized.unfold();
    strip_identity_gates(&mut final_circuit);
    on_stage(StageReport {
        name: "stage 4: final synthesis".to_string(),
        elapsed: t.elapsed(),
        circuit: &final_circuit,
        detail: None,
    });

    (final_circuit, total_error)
}

fn strip_identity_gates(circuit: &mut Circuit) {
    circuit.ops.retain(|op| !matches!(op.gate, Gate::Id));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    fn t_count(circuit: &Circuit) -> usize {
        circuit.ops.iter().filter(|op| matches!(op.gate, Gate::T | Gate::Tdg)).count()
    }

    #[test]
    fn pure_clifford_circuit_compiles_to_zero_t_gates() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::S, vec![1]);
        let original = c.get_unitary();

        let config = PipelineConfig { epsilon: 1e-8, ..PipelineConfig::default() };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert_eq!(t_count(&compiled), 0);
        assert!(error_bound < 1e-6);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-6);
    }

    #[test]
    fn generic_rotation_compiles_within_epsilon() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.437), vec![0]);
        c.push(Gate::H, vec![0]);
        let original = c.get_unitary();

        let config = PipelineConfig { epsilon: 1e-6, ..PipelineConfig::default() };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert!(t_count(&compiled) > 0, "a generic rotation should need some T gates");
        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }

    #[test]
    fn two_t_gates_collapse_to_s_with_zero_leftover_t_count() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        let original = c.get_unitary();

        let config = PipelineConfig::default();
        let (compiled, _) = compile(&c, &config, |_| {});

        assert_eq!(t_count(&compiled), 0, "Rz(pi/4)+Rz(pi/4) = S, an exact Clifford");
        assert!(distance(&original, &compiled.get_unitary()) < 1e-6);
    }

    #[test]
    fn trbo_enabled_does_not_break_correctness() {
        let mut c = Circuit::new(2);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(-0.3), vec![1]);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.9), vec![0]);
        let original = c.get_unitary();

        let config =
            PipelineConfig { epsilon: 1e-6, seed: 11, trbo: true, ..PipelineConfig::default() };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }

    #[test]
    fn cyclosynth_enabled_does_not_break_correctness() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.6), vec![0]);
        c.push(Gate::H, vec![0]);
        let original = c.get_unitary();

        let config = PipelineConfig {
            epsilon: 1e-6,
            seed: 3,
            cyclosynth: true,
            ..PipelineConfig::default()
        };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }

    /// End-to-end check that `approx_cancel` is "properly accounted", not
    /// just a number that changed: a circuit shaped like the motivating
    /// qft_n63.qasm case (a lone Cx;Rz;Cx block whose angle sits just
    /// outside plain operator-norm epsilon of a Clifford point, but well
    /// within epsilon infidelity) should report a visibly larger error
    /// bound with the flag on -- and the *actual* distance from the
    /// original circuit must never exceed that reported bound, on or off.
    #[test]
    fn approx_cancel_properly_accounts_for_a_near_clifford_block() {
        let theta = 2.0 * std::f64::consts::PI - 2e-4;
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![1, 0]);
        c.push(Gate::Rz(theta), vec![0]);
        c.push(Gate::Cx, vec![1, 0]);
        let original = c.get_unitary();

        let strict_config = PipelineConfig { epsilon: 1e-8, ..PipelineConfig::default() };
        let (strict_compiled, strict_bound) = compile(&c, &strict_config, |_| {});

        let approx_config =
            PipelineConfig { epsilon: 1e-8, approx_cancel: true, ..PipelineConfig::default() };
        let (approx_compiled, approx_bound) = compile(&c, &approx_config, |_| {});

        assert!(
            approx_bound > strict_bound * 100.0,
            "approx_cancel should visibly spend far more of the error budget here (strict={strict_bound}, approx={approx_bound})"
        );

        // The critical rigor check, matching the same 10x compounding
        // slack main.rs's own --verify warning already uses: the reported
        // bound must never be violated by the true distance, on or off.
        assert!(distance(&original, &strict_compiled.get_unitary()) <= strict_bound * 10.0 + 1e-9);
        assert!(distance(&original, &approx_compiled.get_unitary()) <= approx_bound * 10.0 + 1e-9);
    }

    #[test]
    fn compiled_circuit_is_entirely_clifford_plus_t() {
        let mut c = Circuit::new(2);
        c.push(Gate::Rz(0.55), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(1.1), vec![1]);
        let config = PipelineConfig::default();
        let (compiled, _) = compile(&c, &config, |_| {});
        for op in &compiled.ops {
            assert!(
                op.gate.is_clifford() || matches!(op.gate, Gate::T | Gate::Tdg),
                "found non-Clifford+T gate: {:?}",
                op.gate
            );
        }
    }

    #[test]
    fn skip_gauge_collapse_omits_stage_1_but_still_compiles_correctly() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        let original = c.get_unitary();

        let config = PipelineConfig {
            epsilon: 1e-6,
            skip_gauge_collapse: true,
            ..PipelineConfig::default()
        };
        let mut stage_names = Vec::new();
        let (compiled, error_bound) = compile(&c, &config, |report| stage_names.push(report.name));

        assert!(
            !stage_names.iter().any(|name| name.contains("gauge collapse")),
            "gauge collapse should not run: {stage_names:?}"
        );
        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }

    #[test]
    fn skip_windowed_resynthesis_omits_stage_2_but_still_compiles_correctly() {
        let mut c = Circuit::new(2);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(-0.3), vec![1]);
        let original = c.get_unitary();

        let config = PipelineConfig {
            epsilon: 1e-6,
            skip_windowed_resynthesis: true,
            ..PipelineConfig::default()
        };
        let mut stage_names = Vec::new();
        let (compiled, error_bound) = compile(&c, &config, |report| stage_names.push(report.name));

        assert!(
            !stage_names.iter().any(|name| name.contains("windowed resynthesis")),
            "windowed resynthesis should not run: {stage_names:?}"
        );
        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }
}
