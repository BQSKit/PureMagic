//! Orchestrates all six stages into one `compile` entry point, in the same
//! relative order as `build_bqskit_workflow` in
//! `data_processing/compile_cliffordt.py`: blocking -> exact-Clifford ->
//! rounding (the "gauge collapse" cycle) -> windowed multi-qubit
//! resynthesis -> gauge collapse again -> TRbO (if enabled) -> gauge
//! collapse a *second* time -> final synthesis. Running the gauge-collapse
//! cycle both before and after TRbO is deliberate, not redundant -- this
//! session's earlier work on the actual Python pipeline found removing the
//! second run regressed T-count by ~1.4%, since re-checking for exact
//! Clifford hits after TRbO's angle adjustments exposes coincidences the
//! first pass alone doesn't.

use std::time::{Duration, Instant};

use crate::cliffordt::clifford::CliffordTable;
use crate::cliffordt::group_single_qubit::group_single_qubit_gates;
use crate::cliffordt::matrix::distance;
use crate::cliffordt::partition::partition;
use crate::cliffordt::qgate_circuit::{Circuit, Gate};
use crate::cliffordt::rounding::round_to_discrete_z;
use crate::cliffordt::stage4_scan_removal::{scanning_gate_removal, ScanConfig};
use crate::cliffordt::stats::block_stats;
use crate::cliffordt::synthesize::{synthesize_block, SynthConfig};
use crate::cliffordt::trbo::{trbo_optimize, TrboConfig};

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
/// natural "how much work is still left to do" measure for stages 1-5,
/// since every one of them exists to either remove an `Rz` entirely
/// (exact-Clifford hit, or a gate-removal simplification) or round it onto
/// the discrete grid; only stage 6 ever needs to actually pay a T-count
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
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig { epsilon: 1e-8, seed: 0, trbo: false, cyclosynth: false }
    }
}

/// Stage 1 (blocking) + Stage 2 (exact Clifford, leaving non-matches
/// untouched) + unfold + Stage 3 (angle rounding), each timed and reported
/// separately. `cycle` labels which of the three times through this cycle
/// it is ("initial", "post stage 4", "post TRbO") since it runs more than
/// once (see module docs). A block that doesn't match exactly is left as
/// its original gates, not forced through anything lossy -- this stage
/// never introduces approximation error.
fn gauge_collapse(
    circuit: &Circuit,
    epsilon: f64,
    table: &CliffordTable,
    cycle: &str,
    on_stage: &mut impl FnMut(StageReport),
) -> Circuit {
    let t = Instant::now();
    let mut current = group_single_qubit_gates(circuit);
    let (num_blocks, grouped_gates) = block_stats(&current);
    let avg_block = if num_blocks > 0 { grouped_gates as f64 / num_blocks as f64 } else { 0.0 };
    let detail =
        format!("{grouped_gates} single-qubit gates grouped into {num_blocks} blocks (avg {avg_block:.1} gates/block)");
    on_stage(StageReport {
        name: format!("stage 1: blocking ({cycle})"),
        elapsed: t.elapsed(),
        circuit: &current,
        detail: Some(detail),
    });

    // Stage 2: each block whose composed unitary exactly matches one of
    // the 24 single-qubit Cliffords is rewritten as that element's
    // shortest available word (see `CliffordTable::build`) -- but only
    // when that word is no longer than what's already there. A match never
    // makes a block worse: if the shortest word happens to be longer than
    // the block's current gate count (rare now that the table's generating
    // set includes every native Clifford gate, not just H/S, but still
    // possible), the original gates are left alone. Mirrors
    // `collapse_clifford_blocks`'s `len(shortest) > len(block): keep
    // original` guard in the Python reference.
    let t = Instant::now();
    let mut blocks_matched = 0usize;
    let mut size_before = 0usize;
    let mut size_after = 0usize;
    let mut rz_consumed = 0usize;
    let mut blocks_exposed = 0usize;
    let checked = current.for_each_block(|inner| {
        let target = inner.get_unitary();
        match table.exact_match(&target, 1e-10) {
            Some(word) if word.len() <= inner.ops.len() => {
                blocks_matched += 1;
                size_before += inner.ops.len();
                size_after += word.len();
                rz_consumed += inner.ops.iter().filter(|op| op.gate.is_rz()).count();
                crate::cliffordt::clifford::circuit_from_word(&word)
            }
            Some(_) => inner.clone(),
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
                    blocks_exposed += 1;
                    crate::cliffordt::clifford::decompose_to_rz_canonical(&target)
                } else {
                    inner.clone()
                }
            }
        }
    });
    current = checked.unfold();
    let detail = format!(
        "{blocks_matched}/{num_blocks} blocks matched an exact Clifford{}, {rz_consumed} Rz gates consumed for free{}",
        if blocks_matched > 0 {
            format!(
                " (avg size {:.1} -> {:.1} gates)",
                size_before as f64 / blocks_matched as f64,
                size_after as f64 / blocks_matched as f64
            )
        } else {
            String::new()
        },
        if blocks_exposed > 0 {
            format!(", {blocks_exposed} blocks with a hidden U3 rotation exposed as adjustable Rz angles")
        } else {
            String::new()
        }
    );
    on_stage(StageReport {
        name: format!("stage 2: exact Clifford ({cycle})"),
        elapsed: t.elapsed(),
        circuit: &current,
        detail: Some(detail),
    });

    let t = Instant::now();
    current = round_to_discrete_z(&current, epsilon);
    on_stage(StageReport {
        name: format!("stage 3: rounding ({cycle})"),
        elapsed: t.elapsed(),
        circuit: &current,
        detail: None,
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
    circuit: &Circuit,
    config: &PipelineConfig,
    mut on_stage: impl FnMut(StageReport),
) -> (Circuit, f64) {
    let table = CliffordTable::build();

    // Gauge collapse (Stages 1-3).
    let mut current = gauge_collapse(circuit, config.epsilon, &table, "initial", &mut on_stage);

    // Stage 4: windowed multi-qubit resynthesis.
    let t = Instant::now();
    let partitioned = partition(&current, 2);
    let scan_config = ScanConfig { success_threshold: config.epsilon, ..ScanConfig::default() };
    let scanned = partitioned.for_each_block(|inner| scanning_gate_removal(inner, &scan_config));
    current = scanned.unfold();
    on_stage(StageReport {
        name: "stage 4: windowed resynthesis".to_string(),
        elapsed: t.elapsed(),
        circuit: &current,
        detail: None,
    });

    // Gauge collapse again before TRbO.
    current = gauge_collapse(&current, config.epsilon, &table, "post stage 4", &mut on_stage);

    // Stage 5: TRbO gauge-freedom optimization (optional). Its payoff is
    // per-block Rz reduction via gauge freedom across a wider window than
    // Stage 4 -- not visible in a whole-circuit gate/Rz delta when only
    // some blocks have exploitable freedom, since the ones that don't stay
    // untouched (trbo_optimize never makes a block worse) and dilute it.
    if config.trbo {
        let t = Instant::now();
        let partitioned = partition(&current, 4);
        let trbo_config = TrboConfig { success_threshold: config.epsilon, seed: config.seed, ..TrboConfig::default() };
        let mut num_blocks = 0usize;
        let mut num_improved = 0usize;
        let mut rz_before = 0usize;
        let mut rz_after = 0usize;
        let optimized = partitioned.for_each_block(|inner| {
            num_blocks += 1;
            let before = inner.num_params();
            let result = trbo_optimize(inner, &trbo_config);
            let after = result.num_params();
            rz_before += before;
            rz_after += after;
            if after < before {
                num_improved += 1;
            }
            result
        });
        current = optimized.unfold();
        let detail = format!(
            "{num_improved}/{num_blocks} blocks improved, Rz: {rz_before} -> {rz_after} ({:+})",
            rz_after as i64 - rz_before as i64
        );
        on_stage(StageReport {
            name: "stage 5: TRbO".to_string(),
            elapsed: t.elapsed(),
            circuit: &current,
            detail: Some(detail),
        });

        // Gauge collapse a second time -- not redundant, see module docs.
        current = gauge_collapse(&current, config.epsilon, &table, "post TRbO", &mut on_stage);
    }

    // Stage 6: final synthesis of whatever continuous rotation remains.
    let t = Instant::now();
    let grouped = group_single_qubit_gates(&current);
    let mut total_error = 0.0_f64;
    let synth_config = SynthConfig { epsilon: config.epsilon, seed: config.seed, use_cyclosynth: config.cyclosynth };
    let synthesized = grouped.for_each_block(|inner| {
        if inner.is_all_clifford() {
            return inner.clone();
        }
        let target = inner.get_unitary();
        let result = synthesize_block(&target, &table, &synth_config);
        total_error += distance(&target, &result.get_unitary());
        result
    });
    let mut final_circuit = synthesized.unfold();
    strip_identity_gates(&mut final_circuit);
    on_stage(StageReport {
        name: "stage 6: final synthesis".to_string(),
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

        let config = PipelineConfig { epsilon: 1e-8, seed: 0, trbo: false, cyclosynth: false };
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

        let config = PipelineConfig { epsilon: 1e-6, seed: 0, trbo: false, cyclosynth: false };
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

        let config = PipelineConfig { epsilon: 1e-6, seed: 11, trbo: true, cyclosynth: false };
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

        let config = PipelineConfig { epsilon: 1e-6, seed: 3, trbo: false, cyclosynth: true };
        let (compiled, error_bound) = compile(&c, &config, |_| {});

        assert!(error_bound < 1e-5);
        assert!(distance(&original, &compiled.get_unitary()) < 1e-5);
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
}

