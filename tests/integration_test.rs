//! Integration tests for the `puremagic` and `gen_circuit` binaries.
//!
//! These tests build the binaries via `cargo build` (done automatically by the test harness
//! when using `cargo test --test integration_test`) and then invoke them through
//! `std::process::Command`, checking exit codes, stdout content, and generated output files.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use tempfile::TempDir;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Returns the path to a compiled binary in the Cargo target directory.
fn bin_path(name: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("target").join("debug").join(name)
}

/// Path to a fixture file shipped with the integration tests.
fn fixture(name: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir).join("tests").join("fixtures").join(name)
}

/// Build both binaries exactly once across all parallel test threads.
///
/// When `cargo test` runs integration tests in parallel every test calls this
/// function, but `OnceLock` ensures the actual `cargo build` subprocess is
/// spawned only once.  Without this guard each parallel test would launch its
/// own `cargo build`, causing them to contend for the Cargo package-cache and
/// artifact-directory locks and print repeated
/// "Blocking waiting for file lock on …" messages to stderr.
fn build_binaries() {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let status = Command::new("cargo")
            .args(["build", "--bins"])
            .current_dir(manifest_dir)
            .status()
            .expect("failed to run cargo build");
        assert!(status.success(), "cargo build --bins failed");
    });
}

/// Run `puremagic` with the given extra args in `workdir`.
/// Returns `(exit_success, stdout, stderr)`.
fn run_puremagic(extra_args: &[&str], workdir: &Path) -> (bool, String, String) {
    build_binaries();
    let output = Command::new(bin_path("puremagic"))
        .args(extra_args)
        .current_dir(workdir)
        .output()
        .expect("failed to spawn puremagic");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status.success(), stdout, stderr)
}

/// Run `gen_circuit` with the given extra args in `workdir`.
/// Returns `(exit_success, stdout, stderr)`.
fn run_gen_circuit(extra_args: &[&str], workdir: &Path) -> (bool, String, String) {
    build_binaries();
    let output = Command::new(bin_path("gen_circuit"))
        .args(extra_args)
        .current_dir(workdir)
        .output()
        .expect("failed to spawn gen_circuit");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status.success(), stdout, stderr)
}

// ── puremagic: basic smoke tests ──────────────────────────────────────────────

#[test]
fn puremagic_exits_zero_on_tiny_trans_circuit() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic exited non-zero; stderr:\n{}", stderr);
}

#[test]
fn puremagic_schedules_all_products() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    // tiny.trans has 4 T-gates + 2 CX gates = 6 products.
    // T gate failures add extra scheduled attempts, so the count is >= 6.
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert!(
        scheduled_n.map(|n| n >= 6).unwrap_or(false),
        "expected 'Scheduled N in ...' with N >= 6 in stdout, got:\n{}",
        stdout
    );
}

#[test]
fn puremagic_produces_schedule_file() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    // The scheduler writes <stem>.schedule in the CWD.
    let schedule_file = tmp.path().join("tiny.schedule");
    assert!(schedule_file.exists(), "expected schedule file {:?} to be created", schedule_file);
}

#[test]
fn puremagic_schedule_file_contains_header_and_steps() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    let schedule_file = tmp.path().join("tiny.schedule");
    let contents = std::fs::read_to_string(&schedule_file).unwrap();
    assert!(
        contents.contains("# Total products:"),
        "schedule file missing '# Total products:' header; contents:\n{}",
        contents
    );
    // At least one lcycle line starts with a digit.
    assert!(
        contents.lines().any(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit())),
        "schedule file has no lcycle lines; contents:\n{}",
        contents
    );
}

#[test]
fn puremagic_reports_parallelism() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    assert!(stdout.contains("Parallelism:"), "expected 'Parallelism:' in stdout, got:\n{}", stdout);
}

#[test]
fn puremagic_reports_scheduling_efficiency() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    assert!(
        stdout.contains("Scheduling efficiency:"),
        "expected 'Scheduling efficiency:' in stdout, got:\n{}",
        stdout
    );
}

#[test]
fn puremagic_with_magic_routing_flag_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    // clap uses kebab-case for long flags derived from snake_case field names.
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--use-magic-routing"], tmp.path());
    assert!(ok, "puremagic --use-magic-routing failed; stderr:\n{}", stderr);
}

#[test]
fn puremagic_with_fixed_rseed_is_deterministic() {
    let tmp1 = TempDir::new().unwrap();
    let tmp2 = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let args = ["--circuit", circuit.to_str().unwrap(), "--rseed", "42"];
    let (ok1, stdout1, _) = run_puremagic(&args, tmp1.path());
    let (ok2, stdout2, _) = run_puremagic(&args, tmp2.path());
    assert!(ok1 && ok2, "puremagic failed with fixed rseed");
    // The "Scheduled N in M lcycles" line must be identical across runs.
    let extract_scheduled_line = |s: &str| {
        s.lines().find(|l| l.contains("Scheduled") && l.contains("lcycles")).map(str::to_owned)
    };
    assert_eq!(
        extract_scheduled_line(&stdout1),
        extract_scheduled_line(&stdout2),
        "puremagic output differs between runs with the same rseed"
    );
}

#[test]
fn puremagic_fails_on_nonexistent_circuit() {
    let tmp = TempDir::new().unwrap();
    let (ok, _stdout, _stderr) =
        run_puremagic(&["--circuit", "nonexistent_file.trans"], tmp.path());
    assert!(!ok, "puremagic should exit non-zero for a missing circuit file");
}

#[test]
fn puremagic_with_larger_fixture_circuit_exits_zero() {
    let tmp = TempDir::new().unwrap();
    // Uses a fixture with 20 T-gates + 3 CX gates on 4 qubits — no external data dir needed.
    let circuit = fixture("small_4q.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed on small_4q.trans; stderr:\n{}", stderr);
}

#[test]
fn puremagic_larger_fixture_schedules_all_products() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("small_4q.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed on small_4q.trans; stderr:\n{}", stderr);
    // small_4q.trans has 20 T-gates + 3 CX gates = 23 products.
    // T gate failures add extra scheduled attempts, so the count is >= 23.
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert!(
        scheduled_n.map(|n| n >= 23).unwrap_or(false),
        "expected 'Scheduled N in ...' with N >= 23 in stdout, got:\n{}",
        stdout
    );
}

// ── puremagic: topology options ───────────────────────────────────────────────

#[test]
fn puremagic_with_ancilla_rows_2_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) = run_puremagic(
        &["--circuit", circuit.to_str().unwrap(), "--use-magic-routing", "--ancilla-rows", "2"],
        tmp.path(),
    );
    assert!(ok, "puremagic --ancilla-rows 2 failed; stderr:\n{}", stderr);
}

#[test]
fn puremagic_with_sides_only_flag_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--sides_only"], tmp.path());
    assert!(ok, "puremagic --sides_only failed; stderr:\n{}", stderr);
}

// ── gen_circuit binary tests ──────────────────────────────────────────────────

#[test]
fn gen_circuit_exits_zero_with_defaults() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) =
        run_gen_circuit(&["--output", out_file.to_str().unwrap()], tmp.path());
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
}

#[test]
fn gen_circuit_produces_output_file() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) =
        run_gen_circuit(&["--output", out_file.to_str().unwrap()], tmp.path());
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    assert!(out_file.exists(), "gen_circuit did not create output file {:?}", out_file);
}

#[test]
fn gen_circuit_output_file_is_non_empty() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) =
        run_gen_circuit(&["--output", out_file.to_str().unwrap()], tmp.path());
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    let metadata = std::fs::metadata(&out_file).unwrap();
    assert!(metadata.len() > 0, "gen_circuit produced an empty output file");
}

#[test]
fn gen_circuit_output_lines_have_trans_format() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_gen_circuit(
        &[
            "--output",
            out_file.to_str().unwrap(),
            "--random-products",
            "10",
            "--random-qubits",
            "4",
        ],
        tmp.path(),
    );
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    let contents = std::fs::read_to_string(&out_file).unwrap();
    // Every non-empty line should start with '+' or '-' and end with a gate marker like '<T>'.
    for line in contents.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            line.starts_with('+') || line.starts_with('-'),
            "line does not start with +/-: {:?}",
            line
        );
        assert!(
            line.contains('<') && line.ends_with('>'),
            "line missing gate type marker: {:?}",
            line
        );
    }
}

#[test]
fn gen_circuit_respects_random_products_count() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_gen_circuit(
        &[
            "--output",
            out_file.to_str().unwrap(),
            "--random-products",
            "15",
            "--random-qubits",
            "4",
        ],
        tmp.path(),
    );
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    let contents = std::fs::read_to_string(&out_file).unwrap();
    let line_count = contents.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(line_count, 15, "expected 15 product lines, got {}", line_count);
}

#[test]
fn gen_circuit_stdout_reports_product_count() {
    let tmp = TempDir::new().unwrap();
    let out_file = tmp.path().join("out.trans");
    let (ok, stdout, stderr) = run_gen_circuit(
        &["--output", out_file.to_str().unwrap(), "--random-products", "8", "--random-qubits", "4"],
        tmp.path(),
    );
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    assert!(stdout.contains("8 products"), "expected '8 products' in stdout, got:\n{}", stdout);
}

// ── end-to-end pipeline: gen_circuit → puremagic ─────────────────────────────

#[test]
fn pipeline_gen_circuit_then_schedule_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit_file = tmp.path().join("generated.trans");

    // Step 1: generate a small circuit.
    let (gen_ok, _stdout, gen_stderr) = run_gen_circuit(
        &[
            "--output",
            circuit_file.to_str().unwrap(),
            "--random-products",
            "20",
            "--random-qubits",
            "4",
        ],
        tmp.path(),
    );
    assert!(gen_ok, "gen_circuit failed; stderr:\n{}", gen_stderr);
    assert!(circuit_file.exists(), "gen_circuit did not create circuit file");

    // Step 2: schedule the generated circuit.
    let (sched_ok, _stdout, sched_stderr) =
        run_puremagic(&["--circuit", circuit_file.to_str().unwrap()], tmp.path());
    assert!(sched_ok, "puremagic failed on generated circuit; stderr:\n{}", sched_stderr);
}

#[test]
fn pipeline_all_products_scheduled_in_generated_circuit() {
    let tmp = TempDir::new().unwrap();
    let circuit_file = tmp.path().join("generated.trans");

    let (gen_ok, _stdout, gen_stderr) = run_gen_circuit(
        &[
            "--output",
            circuit_file.to_str().unwrap(),
            "--random-products",
            "10",
            "--random-qubits",
            "4",
        ],
        tmp.path(),
    );
    assert!(gen_ok, "gen_circuit failed; stderr:\n{}", gen_stderr);

    let (sched_ok, stdout, sched_stderr) =
        run_puremagic(&["--circuit", circuit_file.to_str().unwrap()], tmp.path());
    assert!(sched_ok, "puremagic failed; stderr:\n{}", sched_stderr);
    // Verify the scheduler reports scheduling all 10 products (>= 10 due to T gate failure retries).
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert!(
        scheduled_n.map(|n| n >= 10).unwrap_or(false),
        "expected 'Scheduled N in ...' with N >= 10 in stdout, got:\n{}",
        stdout
    );
}

#[test]
fn pipeline_schedule_file_created_for_generated_circuit() {
    let tmp = TempDir::new().unwrap();
    let circuit_file = tmp.path().join("generated.trans");

    let (gen_ok, _stdout, gen_stderr) = run_gen_circuit(
        &[
            "--output",
            circuit_file.to_str().unwrap(),
            "--random-products",
            "5",
            "--random-qubits",
            "4",
        ],
        tmp.path(),
    );
    assert!(gen_ok, "gen_circuit failed; stderr:\n{}", gen_stderr);

    let (sched_ok, _stdout, sched_stderr) =
        run_puremagic(&["--circuit", circuit_file.to_str().unwrap()], tmp.path());
    assert!(sched_ok, "puremagic failed; stderr:\n{}", sched_stderr);

    let schedule_file = tmp.path().join("generated.schedule");
    assert!(
        schedule_file.exists(),
        "expected schedule file {:?} to be created after pipeline run",
        schedule_file
    );
}

// ── T gate failure reporting ──────────────────────────────────────────────────

/// The scheduler must always print a "T gate failures:" line in stdout.
#[test]
fn puremagic_reports_t_gate_failures_line() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    assert!(
        stdout.contains("T gate failures:"),
        "expected 'T gate failures:' in stdout, got:\n{}",
        stdout
    );
}

/// The "T gate failures:" line must contain a fraction and a percentage in parentheses.
/// Expected format: "T gate failures: N/M (P.P%)"
#[test]
fn puremagic_t_gate_failures_line_has_fraction_and_percent() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    let line = stdout
        .lines()
        .find(|l| l.contains("T gate failures:"))
        .expect("no 'T gate failures:' line in stdout");
    // Must contain a '/' (fraction) and a '%' (percentage).
    assert!(line.contains('/'), "T gate failures line missing '/': {:?}", line);
    assert!(line.contains('%'), "T gate failures line missing '%': {:?}", line);
}

/// With a fixed rseed the T gate failure count must be identical across two runs.
#[test]
fn puremagic_t_gate_failures_deterministic_with_fixed_rseed() {
    let tmp1 = TempDir::new().unwrap();
    let tmp2 = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let args = ["--circuit", circuit.to_str().unwrap(), "--rseed", "42"];
    let (ok1, stdout1, _) = run_puremagic(&args, tmp1.path());
    let (ok2, stdout2, _) = run_puremagic(&args, tmp2.path());
    assert!(ok1 && ok2, "puremagic failed with fixed rseed");
    let extract_failures =
        |s: &str| s.lines().find(|l| l.contains("T gate failures:")).map(str::to_owned);
    assert_eq!(
        extract_failures(&stdout1),
        extract_failures(&stdout2),
        "T gate failures line differs between runs with the same rseed"
    );
}

/// The failure count reported must not exceed the total number of T gates in the circuit.
/// tiny.trans has 4 T gates, so failures must be in [0, 4].
#[test]
fn puremagic_t_gate_failures_bounded_by_total_t_gates() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    // Use rseed=0 for a deterministic run.
    let (ok, stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--rseed", "0"], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    let line = stdout
        .lines()
        .find(|l| l.contains("T gate failures:"))
        .expect("no 'T gate failures:' line in stdout");
    // Parse "T gate failures: N/M (P%)" — extract N and M.
    // The line looks like: "T gate failures: 2/4 (50.0%)"
    let after_colon = line.split(':').nth(1).expect("no colon in T gate failures line").trim();
    let fraction = after_colon.split_whitespace().next().expect("no fraction token");
    let mut parts = fraction.split('/');
    let failures: usize =
        parts.next().and_then(|s| s.parse().ok()).expect("could not parse failure count");
    let total: usize =
        parts.next().and_then(|s| s.parse().ok()).expect("could not parse total T gate count");
    assert_eq!(total, 4, "expected 4 total T gates in tiny.trans, got {}", total);
    assert!(failures <= total, "T gate failures {} exceeds total T gates {}", failures, total);
}

/// The larger fixture (small_4q.trans, 20 T gates) must also report T gate failures correctly.
#[test]
fn puremagic_t_gate_failures_reported_for_larger_circuit() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("small_4q.trans");
    let (ok, stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--rseed", "7"], tmp.path());
    assert!(ok, "puremagic failed on small_4q.trans; stderr:\n{}", stderr);
    let line = stdout
        .lines()
        .find(|l| l.contains("T gate failures:"))
        .expect("no 'T gate failures:' line in stdout");
    // Parse total T gate count from "T gate failures: N/M (P%)"
    let after_colon = line.split(':').nth(1).expect("no colon in T gate failures line").trim();
    let fraction = after_colon.split_whitespace().next().expect("no fraction token");
    let total: usize = fraction
        .split('/')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("could not parse total T gate count");
    // small_4q.trans has 20 T gates.
    assert_eq!(total, 20, "expected 20 total T gates in small_4q.trans, got {}", total);
}

/// Different rseeds must (with overwhelming probability) produce different failure counts
/// for a circuit with many T gates.  We run 10 seeds and require at least 2 distinct values.
#[test]
fn puremagic_t_gate_failures_vary_across_seeds() {
    let circuit = fixture("small_4q.trans");
    let circuit_str = circuit.to_str().unwrap();
    let counts: Vec<usize> = (0u32..10)
        .map(|seed| {
            let tmp = TempDir::new().unwrap();
            let seed_str = seed.to_string();
            let (ok, stdout, stderr) =
                run_puremagic(&["--circuit", circuit_str, "--rseed", &seed_str], tmp.path());
            assert!(ok, "puremagic failed with rseed {}; stderr:\n{}", seed, stderr);
            let line = stdout
                .lines()
                .find(|l| l.contains("T gate failures:"))
                .expect("no 'T gate failures:' line in stdout");
            let after_colon =
                line.split(':').nth(1).expect("no colon in T gate failures line").trim();
            let fraction = after_colon.split_whitespace().next().expect("no fraction token");
            fraction
                .split('/')
                .next()
                .and_then(|s| s.parse().ok())
                .expect("could not parse failure count")
        })
        .collect();
    let distinct = counts.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "T gate failures never varied across 10 seeds for small_4q.trans: {:?}",
        counts
    );
}

// ── helpers: transpile binary ─────────────────────────────────────────────────

/// Run `transpile` with the given extra args in `workdir`.
/// Returns `(exit_success, stdout, stderr)`.
fn run_transpile(extra_args: &[&str], workdir: &Path) -> (bool, String, String) {
    build_binaries();
    let output = Command::new(bin_path("transpile"))
        .args(extra_args)
        .current_dir(workdir)
        .output()
        .expect("failed to spawn transpile");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    (output.status.success(), stdout, stderr)
}

// ── transpile: basic smoke tests ──────────────────────────────────────────────

/// `transpile` must exit zero when given a valid `.cliffordt.qasm` file.
#[test]
fn transpile_exits_zero_on_valid_qasm() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile exited non-zero; stderr:\n{}", stderr);
}

/// `transpile` must create the output `.trans` file.
#[test]
fn transpile_produces_output_file() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    assert!(out.exists(), "transpile did not create output file {:?}", out);
}

/// The output `.trans` file must be non-empty.
#[test]
fn transpile_output_file_is_non_empty() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    let meta = std::fs::metadata(&out).unwrap();
    assert!(meta.len() > 0, "transpile produced an empty output file");
}

/// Every non-empty line in the `.trans` output must start with `+` or `-`
/// and end with a gate-type marker like `<T>`, `<M>`, `<CX>`, etc.
#[test]
fn transpile_output_lines_have_trans_format() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    let contents = std::fs::read_to_string(&out).unwrap();
    for line in contents.lines().filter(|l| !l.trim().is_empty()) {
        assert!(
            line.starts_with('+') || line.starts_with('-'),
            "trans line does not start with +/-: {:?}",
            line
        );
        assert!(
            line.contains('<') && line.ends_with('>'),
            "trans line missing gate-type marker: {:?}",
            line
        );
    }
}

/// `transpile` stdout must report the number of T gates written.
#[test]
fn transpile_stdout_reports_t_gate_count() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    assert!(stdout.contains("T gates"), "expected 'T gates' in transpile stdout, got:\n{}", stdout);
}

/// `transpile` stdout must report the number of Clifford gates written.
#[test]
fn transpile_stdout_reports_clifford_count() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    assert!(
        stdout.contains("Cliffords"),
        "expected 'Cliffords' in transpile stdout, got:\n{}",
        stdout
    );
}

/// `transpile` stdout must report the average Pauli product weight.
#[test]
fn transpile_stdout_reports_average_weight() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    assert!(
        stdout.contains("Average Pauli product weight"),
        "expected 'Average Pauli product weight' in transpile stdout, got:\n{}",
        stdout
    );
}

/// `transpile` must exit non-zero when the input file does not end with `.cliffordt.qasm`.
#[test]
fn transpile_rejects_non_cliffordt_qasm_input() {
    let tmp = TempDir::new().unwrap();
    // Use a plain .trans file — not a .cliffordt.qasm file.
    let input = fixture("tiny.trans");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, _stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(!ok, "transpile should exit non-zero for a non-.cliffordt.qasm input");
}

/// `transpile` must exit non-zero when the input file does not exist.
#[test]
fn transpile_fails_on_nonexistent_input() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, _stderr) = run_transpile(
        &["--input_file", "nonexistent.cliffordt.qasm", "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(!ok, "transpile should exit non-zero for a missing input file");
}

/// With a fixed `--max_width`, every non-Clifford line in the output must have
/// at most that many non-`_` characters in the Pauli string (i.e. weight ≤ max_width).
#[test]
fn transpile_max_width_limits_pauli_weight() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let max_w = 1i32;
    let (ok, _stdout, stderr) = run_transpile(
        &[
            "--input_file",
            input.to_str().unwrap(),
            "--output_file",
            out.to_str().unwrap(),
            "--max_width",
            &max_w.to_string(),
        ],
        tmp.path(),
    );
    assert!(ok, "transpile --max_width {} failed; stderr:\n{}", max_w, stderr);
    let contents = std::fs::read_to_string(&out).unwrap();
    for line in contents.lines().filter(|l| !l.trim().is_empty()) {
        // Only check T-gate lines (not CX/S/etc. Clifford lines).
        if !line.ends_with("<T>") {
            continue;
        }
        // The Pauli string is between the sign character and the '<'.
        let pauli_str = &line[1..line.rfind('<').unwrap()];
        let weight = pauli_str.chars().filter(|&c| c != '_').count();
        assert!(
            weight <= max_w as usize,
            "T-gate line weight {} exceeds max_width {}: {:?}",
            weight,
            max_w,
            line
        );
    }
}

/// Two runs of `transpile` with the same input must produce identical output files
/// (transpilation is deterministic — no randomness involved).
#[test]
fn transpile_output_is_deterministic() {
    let tmp1 = TempDir::new().unwrap();
    let tmp2 = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out1 = tmp1.path().join("out.trans");
    let out2 = tmp2.path().join("out.trans");
    let args = ["--input_file", input.to_str().unwrap()];
    let (ok1, _, _) =
        run_transpile(&[args[0], args[1], "--output_file", out1.to_str().unwrap()], tmp1.path());
    let (ok2, _, _) =
        run_transpile(&[args[0], args[1], "--output_file", out2.to_str().unwrap()], tmp2.path());
    assert!(ok1 && ok2, "transpile failed in determinism test");
    let c1 = std::fs::read_to_string(&out1).unwrap();
    let c2 = std::fs::read_to_string(&out2).unwrap();
    assert_eq!(c1, c2, "transpile produced different output on two identical runs");
}

/// The `.trans` output produced by `transpile` must be schedulable by `puremagic`
/// (end-to-end pipeline: transpile → puremagic).
#[test]
fn pipeline_transpile_then_puremagic_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let trans_out = tmp.path().join("tiny.trans");

    // Step 1: transpile the QASM file.
    let (trans_ok, _stdout, trans_stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", trans_out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(trans_ok, "transpile failed in pipeline test; stderr:\n{}", trans_stderr);
    assert!(trans_out.exists(), "transpile did not create .trans file");

    // Step 2: schedule the transpiled circuit.
    let (sched_ok, _stdout, sched_stderr) =
        run_puremagic(&["--circuit", trans_out.to_str().unwrap()], tmp.path());
    assert!(sched_ok, "puremagic failed on transpile output; stderr:\n{}", sched_stderr);
}

/// The `.trans` output from `transpile` must contain at least one `<T>` line
/// (the tiny fixture has T gates).
#[test]
fn transpile_output_contains_t_gate_lines() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    let contents = std::fs::read_to_string(&out).unwrap();
    assert!(
        contents.lines().any(|l| l.ends_with("<T>")),
        "expected at least one <T> line in transpile output, got:\n{}",
        contents
    );
}

/// The `.trans` output from `transpile` must contain measurement lines (`<M>`).
/// The transpiler always appends one measurement per qubit at the end.
#[test]
fn transpile_output_contains_measurement_lines() {
    let tmp = TempDir::new().unwrap();
    let input = fixture("tiny.cliffordt.qasm");
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, stderr) = run_transpile(
        &["--input_file", input.to_str().unwrap(), "--output_file", out.to_str().unwrap()],
        tmp.path(),
    );
    assert!(ok, "transpile failed; stderr:\n{}", stderr);
    let contents = std::fs::read_to_string(&out).unwrap();
    assert!(
        contents.lines().any(|l| l.ends_with("<M>")),
        "expected at least one <M> line in transpile output, got:\n{}",
        contents
    );
}

// ── puremagic: --no-t-failures flag ──────────────────────────────────────────

/// `puremagic --no-t-failures` must exit zero.
#[test]
fn puremagic_no_t_failures_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--no-t-failures"], tmp.path());
    assert!(ok, "puremagic --no-t-failures exited non-zero; stderr:\n{}", stderr);
}

/// With `--no-t-failures`, the "T gate failures:" line must report 0 failures.
#[test]
fn puremagic_no_t_failures_reports_zero_failures() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--no-t-failures"], tmp.path());
    assert!(ok, "puremagic --no-t-failures failed; stderr:\n{}", stderr);
    let line = stdout
        .lines()
        .find(|l| l.contains("T gate failures:"))
        .expect("no 'T gate failures:' line in stdout");
    // Parse "T gate failures: N/M (P%)" — N must be 0.
    let after_colon = line.split(':').nth(1).expect("no colon in T gate failures line").trim();
    let fraction = after_colon.split_whitespace().next().expect("no fraction token");
    let failures: usize = fraction
        .split('/')
        .next()
        .and_then(|s| s.parse().ok())
        .expect("could not parse failure count");
    assert_eq!(failures, 0, "--no-t-failures should report 0 failures, got {}", failures);
}

/// With `--no-t-failures`, the scheduled count must equal exactly the number of
/// products in the circuit (no extra retry attempts).
#[test]
fn puremagic_no_t_failures_scheduled_count_equals_products() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--no-t-failures"], tmp.path());
    assert!(ok, "puremagic --no-t-failures failed; stderr:\n{}", stderr);
    // tiny.trans has exactly 6 products; with no failures the count must be exactly 6.
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert_eq!(
        scheduled_n,
        Some(6),
        "expected exactly 6 scheduled products with --no-t-failures, got {:?}; stdout:\n{}",
        scheduled_n,
        stdout
    );
}

// ── puremagic: bus routing (default, no --use-magic-routing) ─────────────────

/// Without `--use-magic-routing`, `puremagic` uses bus routing and must still exit zero.
#[test]
fn puremagic_bus_routing_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    // Explicitly omit --use-magic-routing to exercise the bus routing code path.
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic (bus routing) exited non-zero; stderr:\n{}", stderr);
}

/// Bus routing must schedule all products.
#[test]
fn puremagic_bus_routing_schedules_all_products() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("small_4q.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic (bus routing) failed; stderr:\n{}", stderr);
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert!(
        scheduled_n.map(|n| n >= 23).unwrap_or(false),
        "bus routing: expected Scheduled N >= 23, got:\n{}",
        stdout
    );
}

// ── puremagic: --ancilla-rows 0 ───────────────────────────────────────────────

/// `--ancilla-rows 0` (compact magic topology) must exit zero.
#[test]
fn puremagic_ancilla_rows_0_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    let (ok, _stdout, stderr) = run_puremagic(
        &["--circuit", circuit.to_str().unwrap(), "--use-magic-routing", "--ancilla-rows", "0"],
        tmp.path(),
    );
    assert!(ok, "puremagic --ancilla-rows 0 exited non-zero; stderr:\n{}", stderr);
}

// ── puremagic: numeric value assertions ───────────────────────────────────────

/// The parallelism value reported by `puremagic` must be a positive floating-point number.
#[test]
fn puremagic_parallelism_value_is_positive() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("small_4q.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    // Line format: "Parallelism: 1.234x"
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Parallelism:"))
        .expect("no 'Parallelism:' line in stdout");
    let value_str = line
        .split_whitespace()
        .nth(1)
        .expect("no value token after 'Parallelism:'")
        .trim_end_matches('x');
    let value: f64 = value_str.parse().expect("parallelism value is not a float");
    assert!(value > 0.0, "parallelism value must be > 0, got {}", value);
}

/// The scheduling efficiency value must be a float in (0, 1].
#[test]
fn puremagic_scheduling_efficiency_value_is_in_range() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("small_4q.trans");
    let (ok, stdout, stderr) = run_puremagic(&["--circuit", circuit.to_str().unwrap()], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    // Line format: "Scheduling efficiency: 0.456"
    let line = stdout
        .lines()
        .find(|l| l.starts_with("Scheduling efficiency:"))
        .expect("no 'Scheduling efficiency:' line in stdout");
    let value_str =
        line.split_whitespace().nth(2).expect("no value token after 'Scheduling efficiency:'");
    let value: f64 = value_str.parse().expect("scheduling efficiency value is not a float");
    assert!(value > 0.0, "scheduling efficiency must be > 0, got {}", value);
    assert!(value <= 1.0, "scheduling efficiency must be <= 1.0, got {}", value);
}

// ── puremagic: seed variation ─────────────────────────────────────────────────

/// Different rseeds must (with overwhelming probability) produce different total
/// lcycle counts for a circuit with many T gates.
/// We run 8 seeds and require at least 2 distinct lcycle counts.
#[test]
fn puremagic_different_rseeds_produce_different_lcycle_counts() {
    let circuit = fixture("small_4q.trans");
    let circuit_str = circuit.to_str().unwrap();
    let lcycle_counts: Vec<usize> = (0u32..8)
        .map(|seed| {
            let tmp = TempDir::new().unwrap();
            let seed_str = seed.to_string();
            let (ok, stdout, stderr) =
                run_puremagic(&["--circuit", circuit_str, "--rseed", &seed_str], tmp.path());
            assert!(ok, "puremagic failed with rseed {}; stderr:\n{}", seed, stderr);
            stdout
                .lines()
                .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
                .and_then(|l| {
                    // "Scheduled N in M logical cycles, ..."
                    let mut it = l.split_whitespace();
                    it.next(); // "Scheduled"
                    it.next(); // N (scheduled count)
                    it.next(); // "in"
                    it.next().and_then(|s| s.parse().ok()) // M (lcycles)
                })
                .expect("could not parse lcycle count from stdout")
        })
        .collect();
    let distinct = lcycle_counts.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "lcycle counts never varied across 8 seeds for small_4q.trans: {:?}",
        lcycle_counts
    );
}

// ── gen_circuit: additional tests ────────────────────────────────────────────

/// `gen_circuit` with a fixed `--rseed` must produce identical output on two runs.
#[test]
fn gen_circuit_with_fixed_rseed_is_deterministic() {
    let tmp1 = TempDir::new().unwrap();
    let tmp2 = TempDir::new().unwrap();
    let out1 = tmp1.path().join("out.trans");
    let out2 = tmp2.path().join("out.trans");
    let (ok1, _, stderr1) = run_gen_circuit(
        &[
            "--output",
            out1.to_str().unwrap(),
            "--random-products",
            "20",
            "--random-qubits",
            "8",
            "--rseed",
            "77",
        ],
        tmp1.path(),
    );
    let (ok2, _, stderr2) = run_gen_circuit(
        &[
            "--output",
            out2.to_str().unwrap(),
            "--random-products",
            "20",
            "--random-qubits",
            "8",
            "--rseed",
            "77",
        ],
        tmp2.path(),
    );
    assert!(ok1, "gen_circuit run 1 failed; stderr:\n{}", stderr1);
    assert!(ok2, "gen_circuit run 2 failed; stderr:\n{}", stderr2);
    let c1 = std::fs::read_to_string(&out1).unwrap();
    let c2 = std::fs::read_to_string(&out2).unwrap();
    assert_eq!(c1, c2, "gen_circuit produced different output on two runs with the same rseed");
}

/// `gen_circuit` stdout must mention the qubit count matching `--random-qubits`.
#[test]
fn gen_circuit_stdout_reports_qubit_count() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("out.trans");
    let (ok, stdout, stderr) = run_gen_circuit(
        &["--output", out.to_str().unwrap(), "--random-products", "10", "--random-qubits", "16"],
        tmp.path(),
    );
    assert!(ok, "gen_circuit failed; stderr:\n{}", stderr);
    assert!(
        stdout.contains("16 qubits"),
        "expected '16 qubits' in gen_circuit stdout, got:\n{}",
        stdout
    );
}

/// `gen_circuit` must reject `--spread-probability` values outside [0, 1].
#[test]
fn gen_circuit_rejects_invalid_spread_probability() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, _stderr) = run_gen_circuit(
        &["--output", out.to_str().unwrap(), "--spread-probability", "1.5"],
        tmp.path(),
    );
    assert!(!ok, "gen_circuit should exit non-zero for spread-probability > 1.0");
}

/// `gen_circuit` must reject `--decay-factor` values outside [0, 1].
#[test]
fn gen_circuit_rejects_invalid_decay_factor() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("out.trans");
    let (ok, _stdout, _stderr) =
        run_gen_circuit(&["--output", out.to_str().unwrap(), "--decay-factor", "-0.1"], tmp.path());
    assert!(!ok, "gen_circuit should exit non-zero for decay-factor < 0.0");
}

/// All products must still be scheduled even when T gates fail (recovery lcycle completes them).
/// Verify that "Scheduled N in" reports >= the base product count (extra = T failure retries).
#[test]
fn puremagic_all_products_scheduled_despite_t_gate_failures() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    // rseed=1 is likely to produce at least one failure.
    let (ok, stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--rseed", "1"], tmp.path());
    assert!(ok, "puremagic failed; stderr:\n{}", stderr);
    // tiny.trans has 6 products total (4 T + 2 CX); T failures add extra retries so count >= 6.
    let scheduled_n: Option<usize> = stdout
        .lines()
        .find(|l| l.contains("Scheduled") && l.contains("logical cycles"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok());
    assert!(
        scheduled_n.map(|n| n >= 6).unwrap_or(false),
        "expected 'Scheduled N in ...' with N >= 6 in stdout even with T gate failures, got:\n{}",
        stdout
    );
}
