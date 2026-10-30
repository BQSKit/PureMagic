//! Compile an arbitrary circuit down to Clifford+T.
//!
//! Runs a five-stage compilation pipeline. Terminal output includes a git
//! banner, backend params, per-circuit before/after stats, and a timing
//! breakdown.

mod cliffordt;
#[allow(dead_code)]
mod utils;

use std::fs::File;
use std::io::{BufWriter, Write};

use clap::Parser;
use rayon::prelude::*;

use cyclosynth::synthesis::diag;

use cliffordt::matrix::distance;
use cliffordt::pipeline::{PipelineConfig, compile, total_gate_count, total_rz_count};
use cliffordt::qasm::load_qasm;
use cliffordt::qasm_write::write_qasm;
use cliffordt::qgate_circuit::{Circuit, Gate};
use cliffordt::stats::{Stats, compute_stats, non_basis_ops};
use cliffordt::synthesize::{SynthCache, rz_t_count};
use utils::Timer;

#[derive(Parser, Debug)]
#[command(author, version, about = "Compile a circuit to Clifford+T")]
struct Args {
    /// Input .qasm file(s).
    #[arg(required = true)]
    inputs: Vec<String>,

    /// Output file (single input) or directory (multiple inputs); default
    /// is <input>.cliffordt.qasm next to the input.
    #[arg(short, long)]
    output: Option<String>,

    #[arg(short, long, default_value_t = 1e-8)]
    epsilon: f64,

    /// Skip cyclosynth's joint synthesis for Stage 3 (the default), falling
    /// back to its independent per-axis Rz synthesis instead.
    #[arg(long)]
    skip_cyclosynth: bool,

    /// Recompute exact unitary fidelity against the original circuit
    /// after compiling (only practical for small qubit counts).
    #[arg(long)]
    verify: bool,

    /// Skip every gauge-collapse cycle (Stage 1, run initially and post
    /// stage 2), to isolate its contribution to the result.
    #[arg(long)]
    skip_gauge_collapse: bool,

    /// Skip Stage 2 (windowed multi-qubit resynthesis), to isolate its
    /// contribution to the result.
    #[arg(long)]
    skip_windowed_resynthesis: bool,

    /// Skip Stage 0 (phase-polynomial merge), to isolate its contribution
    /// to the result.
    #[arg(long)]
    skip_phase_merge: bool,

    /// Skip Stage 4 (Clifford-run simplification), to isolate its
    /// contribution to the result.
    #[arg(long)]
    skip_clifford_simplify: bool,

    /// Also write the Clifford+Rz circuit Stage 3 receives -- the program
    /// after every rotation-level optimization, before T synthesis -- as
    /// <stem>.rotations.qasm beside the Clifford+T output, for back ends that
    /// take rotations rather than T gates; and <stem>.rotations.tcount, one
    /// "angle T-count" line per distinct non-Clifford Rz angle, from the same
    /// independent-axis synthesis `--skip-cyclosynth` uses, so such a back end
    /// pays exactly this synthesizer's cost per axis.
    #[arg(long)]
    emit_rotations: bool,
}

/// Distinct Rz angles that need synthesis (not multiples of pi/4), in first
/// appearance order.
fn costly_rz_angles(circuit: &Circuit) -> Vec<f64> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for op in &circuit.ops {
        if let Gate::Rz(theta) = op.gate {
            let quarter = theta / std::f64::consts::FRAC_PI_4;
            if (quarter - quarter.round()).abs() > 1e-12 && seen.insert(theta.to_bits()) {
                out.push(theta);
            }
        }
    }
    out
}

/// Where `--emit-rotations` writes, given the Clifford+T destination.
fn rotations_path(destination: &str) -> String {
    match destination.strip_suffix(".cliffordt.qasm") {
        Some(stem) => format!("{stem}.rotations.qasm"),
        None => format!("{destination}.rotations.qasm"),
    }
}

#[cfg(test)]
mod costly_rz_angles_tests {
    use super::{Circuit, Gate, costly_rz_angles};
    use crate::cliffordt::synthesize::{SynthCache, rz_t_count};

    #[test]
    fn keeps_each_non_clifford_angle_once_in_order() {
        let mut c = Circuit::new(2);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_2), vec![1]); // Clifford
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![1]); // a T gate
        c.push(Gate::Rz(-0.7), vec![1]);
        c.push(Gate::Rz(0.3), vec![1]); // repeat
        assert_eq!(costly_rz_angles(&c), vec![0.3, -0.7]);
    }

    #[test]
    fn a_generic_axis_costs_tens_of_t_gates() {
        // ~3 log2(1/eps) for Ross-Selinger at 1e-8.
        let t = rz_t_count(0.123456789, 1e-8, &SynthCache::new());
        assert!((60..=100).contains(&t), "{t}");
    }
}

#[cfg(test)]
mod rotations_path_tests {
    use super::rotations_path;

    #[test]
    fn replaces_the_cliffordt_suffix() {
        assert_eq!(rotations_path("data/dnn_n8.cliffordt.qasm"), "data/dnn_n8.rotations.qasm");
    }

    #[test]
    fn appends_to_an_explicit_output_name() {
        assert_eq!(rotations_path("out.qasm"), "out.qasm.rotations.qasm");
    }
}

fn output_path(input: &str, output: &Option<String>, multiple_inputs: bool) -> String {
    let stem = std::path::Path::new(input).file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    match output {
        Some(o) if multiple_inputs => {
            format!("{}/{}.cliffordt.qasm", o.trim_end_matches('/'), stem)
        }
        Some(o) => o.clone(),
        None => {
            // `Path::parent()` returns `Some("")` -- not `None` -- for a bare
            // filename with no directory component, so an empty parent must
            // also map to "." or the joined path resolves to filesystem root
            // instead of the current directory.
            let parent = match std::path::Path::new(input).parent().and_then(|p| p.to_str()) {
                Some(p) if !p.is_empty() => p,
                _ => ".",
            };
            format!("{parent}/{stem}.cliffordt.qasm")
        }
    }
}

#[cfg(test)]
mod output_path_tests {
    use super::output_path;

    #[test]
    fn bare_filename_writes_next_to_current_directory_not_filesystem_root() {
        let path = output_path("dnn_n8.qasm", &None, false);
        assert_eq!(path, "./dnn_n8.cliffordt.qasm");
    }

    #[test]
    fn filename_with_directory_writes_alongside_input() {
        let path = output_path("data/all_compiled/dnn_n8.qasm", &None, false);
        assert_eq!(path, "data/all_compiled/dnn_n8.cliffordt.qasm");
    }

    #[test]
    fn explicit_output_overrides_default_for_single_input() {
        let path = output_path("dnn_n8.qasm", &Some("out.qasm".to_string()), false);
        assert_eq!(path, "out.qasm");
    }

    #[test]
    fn explicit_output_directory_used_for_multiple_inputs() {
        let path = output_path("data/dnn_n8.qasm", &Some("outdir".to_string()), true);
        assert_eq!(path, "outdir/dnn_n8.cliffordt.qasm");
    }
}

fn report_line(before: &Stats, after: &Stats) -> String {
    format!(
        "  {} qubits, {} gates -> {} gates (T={}, clifford={}, cx={}, depth={}, T-depth={})",
        before.qubits,
        before.gates,
        after.gates,
        after.t_count,
        after.clifford_count,
        after.cx_count,
        after.depth,
        after.t_depth
    )
}

fn main() {
    // Claim rayon's global thread pool (a lazily-built, process-wide
    // singleton) with 16 MiB stacks before any other rayon user -- our own
    // pipeline stages or cyclosynth's deep recursive search -- can win that
    // race with the default 2 MiB stacks and silently leave cyclosynth
    // running on undersized stacks, causing a stack overflow.
    let _ = rayon::ThreadPoolBuilder::new().stack_size(16 * 1024 * 1024).build_global();

    let args = Args::parse();

    utils::print_banner("compile_cliffordt (rust)");
    println!(
        "backend: rust (epsilon={:e}, skip_cyclosynth={}, skip_gauge_collapse={}, skip_windowed_resynthesis={}, skip_phase_merge={}, skip_clifford_simplify={})",
        args.epsilon,
        args.skip_cyclosynth,
        args.skip_gauge_collapse,
        args.skip_windowed_resynthesis,
        args.skip_phase_merge,
        args.skip_clifford_simplify
    );

    let mut failures = 0usize;
    for input in &args.inputs {
        println!("=== {input}");
        let _total_timer = Timer::new("total");

        let load_timer = Timer::new("load");
        let circuit = match load_qasm(input) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERROR {input}: failed to load: {e}");
                failures += 1;
                continue;
            }
        };
        drop(load_timer);

        let before = compute_stats(&circuit);
        let original_unitary =
            if args.verify && circuit.n_qubits <= 10 { Some(circuit.get_unitary()) } else { None };

        let config = PipelineConfig {
            epsilon: args.epsilon,
            skip_cyclosynth: args.skip_cyclosynth,
            skip_gauge_collapse: args.skip_gauge_collapse,
            skip_windowed_resynthesis: args.skip_windowed_resynthesis,
            skip_phase_merge: args.skip_phase_merge,
            skip_clifford_simplify: args.skip_clifford_simplify,
        };

        diag::coarse_profile_reset();
        diag::reset_se_top_branch_histogram();
        let compile_timer = Timer::new("compile");
        let mut prev_gates = total_gate_count(&circuit);
        let mut prev_rz = total_rz_count(&circuit);
        // The last circuit reported before Stage 3 is the one it synthesizes;
        // with every earlier stage skipped, that is the input itself.
        let mut pre_synthesis = None;
        let (compiled, error_bound) = compile(&circuit, &config, |report| {
            if args.emit_rotations
                && !report.name.starts_with("stage 3")
                && !report.name.starts_with("stage 4")
            {
                pre_synthesis = Some(report.circuit.clone());
            }
            let gates = total_gate_count(report.circuit);
            let rz = total_rz_count(report.circuit);
            match &report.detail {
                // Some stages (blocking, exact-Clifford) have effects a
                // generic gate/Rz delta can't show or misrepresents -- see
                // pipeline.rs's StageReport doc comment.
                Some(detail) => {
                    println!(
                        "  [{}] {:.2}s -- {}",
                        report.name,
                        report.elapsed.as_secs_f64(),
                        detail
                    );
                }
                None => {
                    println!(
                        "  [{}] {:.2}s -- gates: {} -> {} ({:+}), Rz remaining: {} -> {} ({:+})",
                        report.name,
                        report.elapsed.as_secs_f64(),
                        prev_gates,
                        gates,
                        gates as i64 - prev_gates as i64,
                        prev_rz,
                        rz,
                        rz as i64 - prev_rz as i64,
                    );
                }
            }
            prev_gates = gates;
            prev_rz = rz;
        });
        drop(compile_timer);
        println!("  [cyclosynth profile] {}", diag::coarse_profile_line());
        if diag::trace_enabled() {
            diag::dump_se_top_branch_histogram();
        }

        let after = compute_stats(&compiled);
        println!("{}", report_line(&before, &after));
        println!("  upper error bound: {:.2e}", error_bound);

        // Always run (no flag) -- cheap, and a broken basis (a stray
        // Rz/U3/Block) is worth surfacing regardless of --verify.
        let non_basis = non_basis_ops(&compiled);
        if non_basis.is_empty() {
            println!("  basis check passed (h, s, sdg, x, y, z, t, tdg, cx, cz, swap)");
        } else {
            let detail: Vec<String> =
                non_basis.iter().map(|(name, n)| format!("{name}: {n}")).collect();
            eprintln!(
                "WARNING {input}: FAILED basis check, output is not Clifford+T: {}",
                detail.join(", ")
            );
        }

        if args.verify {
            let verify_timer = Timer::new("verify");
            match &original_unitary {
                Some(original) => {
                    let d = distance(original, &compiled.get_unitary());
                    println!("  verified distance from original: {:.2e}", d);
                    // error_bound already sums each synthesized block's own
                    // epsilon budget, so comparing against it (not a flat
                    // epsilon) avoids false-positive warnings on circuits
                    // with many leftover rotations. Floored at 1e-12: a
                    // circuit that reduces to exact Cliffords has
                    // error_bound == 0.0 exactly, but Stage 4's cross-qubit
                    // gate reordering (safe -- gates on disjoint qubits
                    // commute) still does floating-point work after that
                    // bound is finalized, so `d` lands around 1e-16 rather
                    // than bit-exact zero -- measured, not guessed. Kept
                    // three orders of magnitude above that measured noise
                    // (not something looser like 1e-9) so the check still
                    // means something at a tight --epsilon: this comparison
                    // is itself limited to about 1e-15 by dense f64 unitary
                    // arithmetic regardless of epsilon, so there's no floor
                    // that stays meaningful all the way down to that limit,
                    // but 1e-12 leaves real headroom before hitting it.
                    if d > (error_bound * 10.0).max(1e-12) {
                        eprintln!(
                            "WARNING {input}: distance from original exceeds 10x the computed upper error bound"
                        );
                    }
                }
                None => {
                    println!(
                        "  --verify requested but circuit has >10 qubits; skipped (dense unitary would be too large)"
                    );
                }
            }
            drop(verify_timer);
        }

        let destination = output_path(input, &args.output, args.inputs.len() > 1);
        let write_timer = Timer::new("write");
        let write_result = File::create(&destination).and_then(|f| {
            let mut writer = BufWriter::new(f);
            write_qasm(&compiled, &mut writer)
        });
        if let Err(e) = write_result {
            eprintln!("ERROR {input}: failed to write {destination}: {e}");
            failures += 1;
            continue;
        }
        drop(write_timer);

        println!("  wrote {destination}");

        if args.emit_rotations {
            let rotations = pre_synthesis.take().unwrap_or_else(|| circuit.clone()).unfold();
            let path = rotations_path(&destination);
            let write_result = File::create(&path).and_then(|f| {
                let mut writer = BufWriter::new(f);
                write_qasm(&rotations, &mut writer)
            });
            if let Err(e) = write_result {
                eprintln!("ERROR {input}: failed to write {path}: {e}");
                failures += 1;
                continue;
            }
            println!("  wrote {path} (Rz={})", total_rz_count(&rotations));

            let angles = costly_rz_angles(&rotations);
            let cache = SynthCache::new();
            let counts: Vec<usize> =
                angles.par_iter().map(|&a| rz_t_count(a, args.epsilon, &cache)).collect();
            let tcount_path = format!("{}.tcount", path);
            let write_result = File::create(&tcount_path).and_then(|f| {
                let mut writer = BufWriter::new(f);
                for (angle, t) in angles.iter().zip(&counts) {
                    writeln!(writer, "{angle} {t}")?;
                }
                writer.flush()
            });
            if let Err(e) = write_result {
                eprintln!("ERROR {input}: failed to write {tcount_path}: {e}");
                failures += 1;
                continue;
            }
            let by_angle: std::collections::HashMap<u64, usize> =
                angles.iter().map(|a| a.to_bits()).zip(counts.iter().copied()).collect();
            let per_axis_total: usize = rotations
                .ops
                .iter()
                .filter_map(|op| match op.gate {
                    Gate::Rz(t) => by_angle.get(&t.to_bits()).copied(),
                    _ => None,
                })
                .sum();
            println!(
                "  wrote {tcount_path} ({} distinct angles, {per_axis_total} T if each Rz is synthesized alone)",
                angles.len()
            );
        }
    }

    if failures > 0 {
        eprintln!("{failures} of {} input(s) failed", args.inputs.len());
        std::process::exit(1);
    }
}
