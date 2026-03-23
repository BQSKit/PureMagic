//! Integration tests for the `puremagic` and `gen_circuit` binaries.
//!
//! These tests build the binaries via `cargo build` (done automatically by the test harness
//! when using `cargo test --test integration_test`) and then invoke them through
//! `std::process::Command`, checking exit codes, stdout content, and generated output files.

use std::path::{Path, PathBuf};
use std::process::Command;
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

/// Build both binaries once (idempotent — cargo is a no-op if already up to date).
fn build_binaries() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let status = Command::new("cargo")
        .args(["build", "--bins"])
        .current_dir(manifest_dir)
        .status()
        .expect("failed to run cargo build");
    assert!(status.success(), "cargo build --bins failed");
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
    assert!(
        stdout.contains("Scheduled 6 in"),
        "expected 'Scheduled 6 in ...' in stdout, got:\n{}",
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
    // At least one timestep line starts with a digit.
    assert!(
        contents.lines().any(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit())),
        "schedule file has no timestep lines; contents:\n{}",
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
fn puremagic_with_greedy_path_flag_exits_zero() {
    let tmp = TempDir::new().unwrap();
    let circuit = fixture("tiny.trans");
    // This flag uses underscore (long = "use_greedy") not kebab-case.
    let (ok, _stdout, stderr) =
        run_puremagic(&["--circuit", circuit.to_str().unwrap(), "--use_greedy"], tmp.path());
    assert!(ok, "puremagic --use_greedy failed; stderr:\n{}", stderr);
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
    // The "Scheduled N in M timesteps" line must be identical across runs.
    let extract_scheduled_line = |s: &str| {
        s.lines().find(|l| l.contains("Scheduled") && l.contains("timesteps")).map(str::to_owned)
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
    assert!(
        stdout.contains("Scheduled 23 in"),
        "expected 'Scheduled 23 in ...' in stdout, got:\n{}",
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
    // Verify the scheduler reports scheduling all 10 products.
    assert!(
        stdout.contains("Scheduled 10 in"),
        "expected 'Scheduled 10 in ...' in stdout, got:\n{}",
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
