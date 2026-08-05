//! Compile an arbitrary circuit down to Clifford+T.
//!
//! Pure-Rust reimplementation of `data_processing/compile_cliffordt.py`'s
//! bqskit backend's six-stage pipeline (see
//! `/home/vscode/.claude/plans/starry-mixing-lecun.md` for the design).
//! Terminal output deliberately mirrors that script's shape (git banner,
//! backend params, per-circuit before/after stats, timing breakdown) so
//! the two are easy to compare side by side.

mod cliffordt;

use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use clap::Parser;

use cliffordt::matrix::distance;
use cliffordt::pipeline::{compile, total_gate_count, total_rz_count, PipelineConfig};
use cliffordt::qasm::load_qasm;
use cliffordt::qasm_write::write_qasm;
use cliffordt::stats::{compute_stats, non_basis_ops, Stats};

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

    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Enable TRbO-style joint gauge-freedom optimization (Stage 5).
    #[arg(long)]
    trbo: bool,

    /// Enable cyclosynth's joint synthesis for Stage 6, instead of
    /// independent per-axis gridsynth.
    #[arg(long)]
    cyclosynth: bool,

    /// Recompute exact unitary fidelity against the original circuit
    /// after compiling (only practical for small qubit counts).
    #[arg(long)]
    verify: bool,
}

fn output_path(input: &str, output: &Option<String>, multiple_inputs: bool) -> String {
    let stem = std::path::Path::new(input).file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    match output {
        Some(o) if multiple_inputs => format!("{}/{}.cliffordt.qasm", o.trim_end_matches('/'), stem),
        Some(o) => o.clone(),
        None => {
            // `Path::parent()` returns `Some("")` -- not `None` -- for a
            // bare filename with no directory component (e.g. "foo.qasm"
            // run from its own directory), so the `None` fallback alone
            // doesn't catch that case; an empty parent must also map to
            // "." or the joined path gets a leading '/' and resolves to
            // filesystem root instead of the current directory.
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
    // Claim rayon's global thread pool with 16 MiB worker stacks before
    // anything else -- our own pipeline's own rayon usage (Stages 1-5) or
    // cyclosynth's own `ensure_rayon_stack` -- can win that race first.
    // Rayon's global pool is a process-wide singleton built lazily on first
    // use; whichever caller's `build_global` runs first wins, silently (a
    // losing caller's request just becomes a no-op, per rayon's own docs).
    // cyclosynth's recursive "optimal mode" search needs stacks this large
    // (its own `synthesis::mod.rs::ensure_rayon_stack` says so directly:
    // its parallel search nests per-prefix scratch frames deep enough to
    // overflow rayon's default 2 MiB stacks) -- but since this pipeline's
    // own Stage 1-5 rayon usage runs first and would otherwise win that
    // race with the *default* stack size, cyclosynth's own request was
    // silently losing every time, leaving it to run on undersized stacks.
    // That's the real cause of a stack-overflow crash fixed differently
    // (by serializing Stage 6 instead) in an earlier commit -- this claims
    // the pool correctly instead of just reducing contention around the
    // underlying problem.
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
        "backend: rust (epsilon={:e}, seed={}, trbo={}, cyclosynth={})",
        args.epsilon, args.seed, args.trbo, args.cyclosynth
    );

    let mut failures = 0usize;
    for input in &args.inputs {
        println!("=== {input}");
        let total_start = Instant::now();

        let load_start = Instant::now();
        let circuit = match load_qasm(input) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERROR {input}: failed to load: {e}");
                failures += 1;
                continue;
            }
        };
        let load_time = load_start.elapsed();

        let before = compute_stats(&circuit);
        let original_unitary = if args.verify && circuit.n_qubits <= 10 { Some(circuit.get_unitary()) } else { None };

        let config = PipelineConfig { epsilon: args.epsilon, seed: args.seed, trbo: args.trbo, cyclosynth: args.cyclosynth };

        let compile_start = Instant::now();
        let mut prev_gates = total_gate_count(&circuit);
        let mut prev_rz = total_rz_count(&circuit);
        let (compiled, error_bound) = compile(&circuit, &config, |report| {
            let gates = total_gate_count(report.circuit);
            let rz = total_rz_count(report.circuit);
            match &report.detail {
                // Some stages (blocking, exact-Clifford) have a
                // contribution a generic gate/Rz delta can't show, or
                // actively misrepresents -- see pipeline.rs's StageReport
                // doc comment for why.
                Some(detail) => {
                    println!("  [{}] {:.2}s -- {}", report.name, report.elapsed.as_secs_f64(), detail);
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
        let compile_time = compile_start.elapsed();

        let after = compute_stats(&compiled);
        println!("{}", report_line(&before, &after));
        println!("  upper error bound: {:.2e}", error_bound);

        // Basis check: always, no flag needed -- cheap (no simulation), and
        // a broken basis (a stray Rz/U3/Block that never reached Stage 6)
        // is worth surfacing regardless of whether --verify was requested.
        let non_basis = non_basis_ops(&compiled);
        if non_basis.is_empty() {
            println!("  basis check passed (h, s, sdg, x, y, z, t, tdg, cx, cz, swap)");
        } else {
            let detail: Vec<String> = non_basis.iter().map(|(name, n)| format!("{name}: {n}")).collect();
            eprintln!("WARNING {input}: FAILED basis check, output is not Clifford+T: {}", detail.join(", "));
        }

        let mut verify_time = std::time::Duration::ZERO;
        if args.verify {
            let verify_start = Instant::now();
            match &original_unitary {
                Some(original) => {
                    let d = distance(original, &compiled.get_unitary());
                    println!("  verified distance from original: {:.2e}", d);
                    // error_bound already accounts for how many blocks were
                    // synthesized (each consuming its own epsilon budget),
                    // unlike a flat multiple of epsilon -- comparing against
                    // that certificate, not the single-gate epsilon, avoids
                    // false-positive warnings on circuits with many leftover
                    // rotations, where legitimately accumulated error can
                    // exceed 10x a single gate's epsilon.
                    if d > error_bound * 10.0 {
                        eprintln!("WARNING {input}: distance from original exceeds 10x the computed upper error bound");
                    }
                }
                None => {
                    println!("  --verify requested but circuit has >10 qubits; skipped (dense unitary would be too large)");
                }
            }
            verify_time = verify_start.elapsed();
        }

        let destination = output_path(input, &args.output, args.inputs.len() > 1);
        let write_start = Instant::now();
        let write_result = File::create(&destination).and_then(|f| {
            let mut writer = BufWriter::new(f);
            write_qasm(&compiled, &mut writer)
        });
        if let Err(e) = write_result {
            eprintln!("ERROR {input}: failed to write {destination}: {e}");
            failures += 1;
            continue;
        }
        let write_time = write_start.elapsed();
        let total_time = total_start.elapsed();

        println!("  load: {:.2}s", load_time.as_secs_f64());
        println!("  compile: {:.2}s", compile_time.as_secs_f64());
        println!("  verify: {:.2}s", verify_time.as_secs_f64());
        println!("  write: {:.2}s", write_time.as_secs_f64());
        println!("  total: {:.2}s", total_time.as_secs_f64());
        println!("  wrote {destination}");
    }

    if failures > 0 {
        eprintln!("{failures} of {} input(s) failed", args.inputs.len());
        std::process::exit(1);
    }
}
