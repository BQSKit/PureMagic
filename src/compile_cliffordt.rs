//! Compile an arbitrary circuit down to Clifford+T.
//!
//! Runs a four-stage compilation pipeline. Terminal output includes a git
//! banner, backend params, per-circuit before/after stats, and a timing
//! breakdown.

mod cliffordt;
#[allow(dead_code)]
mod utils;

use std::fs::File;
use std::io::BufWriter;

use clap::Parser;

use cliffordt::matrix::distance;
use cliffordt::pipeline::{PipelineConfig, compile, total_gate_count, total_rz_count};
use cliffordt::qasm::load_qasm;
use cliffordt::qasm_write::write_qasm;
use cliffordt::stats::{Stats, compute_stats, non_basis_ops};
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

    /// Enable cyclosynth's joint synthesis for Stage 3, instead of its
    /// independent per-axis Rz synthesis.
    #[arg(long)]
    cyclosynth: bool,

    /// Let Stage 2 drop a block that's within epsilon *infidelity* of a
    /// simpler circuit, not just within epsilon operator-norm distance.
    /// Each approximation is measured exactly and folded into the reported
    /// upper error bound, but real per-cancellation error can be far
    /// larger than epsilon itself -- off by default.
    #[arg(long)]
    approx_cancel: bool,

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

    let git_sha = env!("VERGEN_GIT_SHA");
    let short_sha = &git_sha[..git_sha.len().min(8)];
    println!(
        "compile_cliffordt (rust) - Git branch: {} | Commit: {} | Built: {}",
        env!("VERGEN_GIT_BRANCH"),
        short_sha,
        env!("VERGEN_BUILD_TIMESTAMP")
    );
    println!(
        "backend: rust (epsilon={:e}, cyclosynth={}, approx_cancel={}, skip_gauge_collapse={}, skip_windowed_resynthesis={}, skip_phase_merge={})",
        args.epsilon,
        args.cyclosynth,
        args.approx_cancel,
        args.skip_gauge_collapse,
        args.skip_windowed_resynthesis,
        args.skip_phase_merge
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
            cyclosynth: args.cyclosynth,
            approx_cancel: args.approx_cancel,
            skip_gauge_collapse: args.skip_gauge_collapse,
            skip_windowed_resynthesis: args.skip_windowed_resynthesis,
            skip_phase_merge: args.skip_phase_merge,
        };

        let compile_timer = Timer::new("compile");
        let mut prev_gates = total_gate_count(&circuit);
        let mut prev_rz = total_rz_count(&circuit);
        let (compiled, error_bound) = compile(&circuit, &config, |report| {
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
                    // with many leftover rotations.
                    if d > error_bound * 10.0 {
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
    }

    if failures > 0 {
        eprintln!("{failures} of {} input(s) failed", args.inputs.len());
        std::process::exit(1);
    }
}
