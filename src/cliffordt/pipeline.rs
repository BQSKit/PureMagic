//! Orchestrates all stages into one `compile` entry point: exact
//! phase-polynomial merge -> gauge collapse (blocking + exact-Clifford +
//! rounding, run and reported as one combined stage) -> windowed
//! multi-qubit resynthesis -> gauge collapse again -> final synthesis ->
//! Clifford-run simplification (canonicalize the Clifford framing final
//! synthesis leaves around each T gate).

use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::cliffordt::clifford::CliffordTable;
use crate::cliffordt::clifford_simplify::simplify_clifford_runs;
use crate::cliffordt::group_single_qubit::group_single_qubit_gates;
use crate::cliffordt::matrix::{Unitary, distance};
use crate::cliffordt::partition::partition;
use crate::cliffordt::phase_merge::{count_costly_euler_axes, merge_phase_polynomial};
use crate::cliffordt::progress::ProgressTracker;
use crate::cliffordt::qgate_circuit::{Circuit, Gate};
use crate::cliffordt::rounding::round_to_discrete_z;
use crate::cliffordt::stage4_scan_removal::{ScanConfig, scanning_gate_removal};
use crate::cliffordt::stats::block_stats;
use crate::cliffordt::synthesize::{
    SynthCache, SynthConfig, prepopulate_targets, synthesize_block_cached,
};

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
/// natural "how much work is still left to do" measure for stages 1-2,
/// since every one of them exists to either remove an `Rz` entirely
/// (exact-Clifford hit, or a gate-removal simplification) or round it onto
/// the discrete grid; only stage 3 ever needs to actually pay a T-count
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
    /// Skip cyclosynth's joint synthesis for Stage 3, using its independent
    /// per-axis Rz synthesis instead. Cyclosynth's joint search is the
    /// default (this defaults to `false`, i.e. not skipped) since it
    /// consistently produces equal-or-lower T-counts than the independent
    /// per-axis route.
    pub skip_cyclosynth: bool,
    /// Skip every gauge-collapse cycle (both: initial and post stage 2),
    /// for isolating its contribution to the final result.
    pub skip_gauge_collapse: bool,
    /// Skip Stage 2 (windowed multi-qubit resynthesis), for isolating its
    /// contribution to the final result.
    pub skip_windowed_resynthesis: bool,
    /// Skip Stage 0 (phase-polynomial merge), for isolating its
    /// contribution to the final result.
    pub skip_phase_merge: bool,
    /// Skip Stage 4 (Clifford-run simplification), for isolating its
    /// contribution to the final result.
    pub skip_clifford_simplify: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            epsilon: 1e-8,
            skip_cyclosynth: false,
            skip_gauge_collapse: false,
            skip_windowed_resynthesis: false,
            skip_phase_merge: false,
            skip_clifford_simplify: false,
        }
    }
}

/// Blocking + exact Clifford recognition (leaving non-matches untouched) +
/// unfold + angle rounding, run and reported as one combined stage. `cycle`
/// labels which of the two times through this cycle it is ("initial",
/// "post stage 2") since it runs twice (see module docs). A block that
/// doesn't match exactly is left as its original gates,
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
    // a net win (see phase_merge.rs's docs): merging can turn a block-edge
    // Clifford rotation into a costly one elsewhere without changing how
    // many Rz's look "real" pre-blocking, so the two candidates are compared
    // by costly Euler axis count *after* blocking, not raw rotation count.
    let preopt = if config.skip_phase_merge {
        circuit.clone()
    } else {
        let t = Instant::now();
        let merged = merge_phase_polynomial(circuit);
        let merged_cost = count_costly_euler_axes(&merged, config.epsilon);
        let unmerged_cost = count_costly_euler_axes(circuit, config.epsilon);
        let (preopt, chosen_cost, other_cost, used_merge) = if merged_cost <= unmerged_cost {
            (merged, merged_cost, unmerged_cost, true)
        } else {
            (circuit.clone(), unmerged_cost, merged_cost, false)
        };
        on_stage(StageReport {
            name: "stage 0: phase-polynomial merge".to_string(),
            elapsed: t.elapsed(),
            circuit: &preopt,
            detail: Some(format!(
                "{} candidate chosen: {chosen_cost} costly Euler axis(es) needing synthesis (vs {other_cost} for the alternative)",
                if used_merge { "merged" } else { "unmerged" }
            )),
        });
        preopt
    };

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
        let scan_config = ScanConfig { success_threshold: config.epsilon, ..ScanConfig::default() };
        // No cache to weight by here (unlike Stage 3) -- every block does
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

    // Gauge collapse again, to catch exact-Clifford matches Stage 2's
    // resynthesis exposed.
    if !config.skip_gauge_collapse {
        current = gauge_collapse(&current, config.epsilon, &table, "post stage 2", &mut on_stage);
    }

    // Stage 3: final synthesis of whatever continuous rotation remains.
    let t = Instant::now();
    let grouped = group_single_qubit_gates(&current);
    let synth_config =
        SynthConfig { epsilon: config.epsilon, use_cyclosynth: !config.skip_cyclosynth };
    // Shared across all blocks (including in parallel -- see SynthCache's
    // own doc comment): repeated rotation angles are common enough in real
    // circuits that caching the expensive non-Clifford synthesis path
    // across blocks is a large, real win, not a micro-optimization.
    let synth_cache = SynthCache::new();
    // One entry per non-Clifford block's target unitary, in program order --
    // computed in parallel (cheap per block) but collected into an
    // order-preserving Vec (`par_iter().collect()` into a Vec always
    // preserves original index order, as `for_each_block_with` itself
    // already relies on), so the *order* `prepopulate_targets` dedups
    // against below reflects the circuit's own block order, not Stage 3's
    // scheduling.
    let non_clifford_targets: Vec<Unitary> = grouped
        .ops
        .par_iter()
        .filter_map(|op| match &op.gate {
            Gate::Block(inner) if !inner.is_all_clifford() => Some(inner.get_unitary()),
            _ => None,
        })
        .collect();
    let progress = ProgressTracker::new("final synthesis", non_clifford_targets.len());
    // Deterministically pick and synthesize one representative target per
    // canonical_key *before* Stage 3's per-block parallel dispatch below --
    // see `prepopulate_targets`'s doc comment for why this replaces a race.
    prepopulate_targets(&non_clifford_targets, &table, &synth_config, &synth_cache, || {
        progress.add(1)
    });
    let synth_one = |inner: &Circuit| {
        if inner.is_all_clifford() {
            return (inner.clone(), 0.0);
        }
        let target = inner.get_unitary();
        let result = synthesize_block_cached(&target, &table, &synth_config, &synth_cache);
        let error = distance(&target, &result.get_unitary());
        (result, error)
    };
    // Stage 4 is entirely cyclosynth now (joint search by default, its
    // independent per-axis Rz synthesis with `--skip-cyclosynth`), and
    // cyclosynth parallelizes its own search internally -- so Stage 3 always runs
    // sequentially at this level rather than nesting it inside this crate's
    // own per-block rayon parallelism, which would oversubscribe the
    // thread pool (see `for_each_block_with_sequential`'s doc comment for
    // measured throughput).
    let (synthesized, errors) = grouped.for_each_block_with_sequential(synth_one);
    progress.finish();
    // Stage 2's own approximate-cancellation error (see stage2_error above)
    // is folded in here too, so the reported bound reflects every stage
    // that can spend accuracy, not just Stage 3.
    let total_error: f64 = stage2_error + errors.iter().sum::<f64>();
    let mut final_circuit = synthesized.unfold();
    strip_identity_gates(&mut final_circuit);
    on_stage(StageReport {
        name: "stage 3: final synthesis".to_string(),
        elapsed: t.elapsed(),
        circuit: &final_circuit,
        detail: None,
    });

    // Stage 4: canonicalize the Clifford framing final synthesis leaves
    // around each T gate. Exact and tolerance-gated (see
    // clifford_simplify.rs), so it never spends any of `total_error`'s
    // accuracy budget.
    if !config.skip_clifford_simplify {
        let t = Instant::now();
        final_circuit = simplify_clifford_runs(&final_circuit, &table, 1e-10);
        on_stage(StageReport {
            name: "stage 4: clifford simplification".to_string(),
            elapsed: t.elapsed(),
            circuit: &final_circuit,
            detail: None,
        });
    }

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
    fn cyclosynth_enabled_does_not_break_correctness() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.6), vec![0]);
        c.push(Gate::H, vec![0]);
        let original = c.get_unitary();

        let config =
            PipelineConfig { epsilon: 1e-6, skip_cyclosynth: false, ..PipelineConfig::default() };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }

    #[test]
    fn clifford_simplification_reduces_or_preserves_clifford_count() {
        // Two separate `compile()` calls can legitimately pick different
        // (equally valid) synthesis representatives for the same input --
        // Stage 3's dispatch dedups via a `HashSet` keyed on canonical
        // angle buckets, and Rust's default hasher is randomized per
        // process, so which representative "wins" a tie is not guaranteed
        // stable across calls. Comparing clifford_count between two
        // independent `compile()` runs would therefore be comparing two
        // potentially different Stage 0-3 outputs, not measuring Stage 4's
        // own effect. Instead: compile once with Stage 4 off to get a
        // single fixed "raw" circuit, then apply `simplify_clifford_runs`
        // to that same circuit directly and compare before/after.
        use crate::cliffordt::clifford_simplify::simplify_clifford_runs;
        use crate::cliffordt::stats::compute_stats;

        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.37), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(1.1), vec![1]);
        c.push(Gate::H, vec![1]);
        let original = c.get_unitary();

        let config = PipelineConfig {
            epsilon: 1e-6,
            skip_cyclosynth: false,
            skip_clifford_simplify: true,
            ..PipelineConfig::default()
        };
        let (raw, error_bound) = compile(&c, &config, |_| {});
        let table = CliffordTable::build();
        let simplified = simplify_clifford_runs(&raw, &table, 1e-10);

        assert!(compute_stats(&simplified).clifford_count <= compute_stats(&raw).clifford_count);
        assert!(distance(&original, &raw.get_unitary()) < error_bound * 10.0);
        // Exact, tolerance-gated rewrite -- must not spend any extra
        // accuracy budget beyond ordinary floating-point noise.
        assert!(distance(&original, &simplified.get_unitary()) < (error_bound * 10.0).max(1e-9));
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

    #[test]
    fn skip_phase_merge_omits_stage_0_but_still_compiles_correctly() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(0.2), vec![0]);
        c.push(Gate::Rz(0.3), vec![0]);
        let original = c.get_unitary();

        let config =
            PipelineConfig { epsilon: 1e-6, skip_phase_merge: true, ..PipelineConfig::default() };
        let mut stage_names = Vec::new();
        let (compiled, error_bound) = compile(&c, &config, |report| stage_names.push(report.name));

        assert!(
            !stage_names.iter().any(|name| name.contains("phase-polynomial merge")),
            "phase-polynomial merge should not run: {stage_names:?}"
        );
        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
    }
}
