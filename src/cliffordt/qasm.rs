//! Minimal OpenQASM 2.0 subset loader for this pipeline's own gate
//! vocabulary (Clifford generators, `rz`, a general `u`/`u3` for whatever
//! hasn't been pre-decomposed, and `cx`/`cz`/`swap`).
//!
//! Deliberately narrower than a full QASM parser -- matches the line-based
//! parsing style already used by `transpile.rs`'s own `parse_qasm` (split
//! on `[`/`]` for qubit indices, lowercase-prefix match for gate names),
//! extended to handle parameterized gates, which that parser never needed
//! since its input is always already-compiled Clifford+T.
//!
//! Deliberately fails loudly (a hard `io::Error`, naming the exact line and
//! reason) on anything it can't parse, rather than silently skipping it --
//! a silently-dropped gate or unparsed angle produces a *different, wrong*
//! circuit that still "compiles" and even "verifies" successfully against
//! itself, which is a far worse failure mode than a load error.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Error, ErrorKind};

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

pub fn load_qasm(path: &str) -> io::Result<Circuit> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut circuit = Circuit::new(0);

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let stripped = strip_comment(&line);
        let stripped = stripped.trim();

        if stripped.is_empty()
            || stripped.starts_with("OPENQASM")
            || stripped.starts_with("include")
            || stripped.starts_with("creg")
            || stripped.starts_with("gate ")
            || stripped.starts_with('{')
            || stripped.starts_with('}')
            || stripped.starts_with("barrier")
            || stripped.starts_with("measure")
        {
            continue;
        }

        if stripped.starts_with("qreg") {
            if let Some(n) = bracketed_number(stripped) {
                circuit = Circuit::new(n);
            }
            continue;
        }

        let (name, params, qubits) = parse_gate_line(stripped)
            .map_err(|e| parse_error(line_no, &line, &e))?;
        push_gate(&mut circuit, &name, &params, &qubits)
            .map_err(|e| parse_error(line_no, &line, &e))?;
    }

    Ok(circuit)
}

fn parse_error(line_no: usize, line: &str, reason: &str) -> Error {
    Error::new(ErrorKind::InvalidData, format!("line {}: {reason} (in {line:?})", line_no + 1))
}

fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(idx) => &line[..idx],
        None => line,
    }
}

fn bracketed_number(line: &str) -> Option<usize> {
    let start = line.find('[')? + 1;
    let end = line.find(']')?;
    line[start..end].parse().ok()
}

/// Every `q[N]` (or `qreg_name[N]`) occurrence's index, in order.
fn qubit_indices(line: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('[') {
        if let Some(end) = rest[start..].find(']') {
            if let Ok(idx) = rest[start + 1..start + end].parse::<usize>() {
                out.push(idx);
            }
            rest = &rest[start + end + 1..];
        } else {
            break;
        }
    }
    out
}

/// Split `name(p0, p1, ...) q[a], q[b];` into (name, params, qubit indices).
/// A gate name followed by unparseable parameters is a hard error, not a
/// silently-empty parameter list.
fn parse_gate_line(line: &str) -> Result<(String, Vec<f64>, Vec<usize>), String> {
    let line = line.trim_end_matches(';').trim();
    if line.is_empty() {
        return Err("empty statement".to_string());
    }

    let (name, params, rest_after_paren) = if let Some(open) = line.find('(') {
        let close = line.find(')').ok_or_else(|| "unmatched '(' in gate call".to_string())?;
        let name = line[..open].trim().to_lowercase();
        let mut params = Vec::new();
        for raw in line[open + 1..close].split(',') {
            let raw = raw.trim();
            params.push(
                parse_angle_expr(raw)
                    .ok_or_else(|| format!("could not parse angle expression '{raw}' for gate '{name}'"))?,
            );
        }
        (name, params, &line[close + 1..])
    } else {
        let split = line
            .find(char::is_whitespace)
            .ok_or_else(|| format!("expected a qubit argument after gate name in '{line}'"))?;
        (line[..split].trim().to_lowercase(), Vec::new(), &line[split..])
    };

    let qubits = qubit_indices(rest_after_paren);
    if qubits.is_empty() {
        return Err(format!("no qubit operand found for gate '{name}'"));
    }
    Ok((name, params, qubits))
}

/// Parse a numeric angle expression: plain floats, and `pi`-relative forms
/// commonly emitted by QASM exporters -- `pi`, `-pi/2`, `pi/4`, `3*pi/8`,
/// and (qiskit's own `qasm2.dump` convention) `pi*0.35`, pi first.
fn parse_angle_expr(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let lower = s.to_lowercase();
    let (sign, lower) = if let Some(stripped) = lower.strip_prefix('-') { (-1.0, stripped) } else { (1.0, lower.as_str()) };
    if lower == "pi" {
        return Some(sign * std::f64::consts::PI);
    }
    if let Some(rest) = lower.strip_prefix("pi/") {
        let denom: f64 = rest.parse().ok()?;
        return Some(sign * std::f64::consts::PI / denom);
    }
    if let Some(rest) = lower.strip_prefix("pi*") {
        let numer: f64 = rest.parse().ok()?;
        return Some(sign * numer * std::f64::consts::PI);
    }
    if let Some(rest) = lower.strip_suffix("*pi") {
        let numer: f64 = rest.parse().ok()?;
        return Some(sign * numer * std::f64::consts::PI);
    }
    None
}

/// Push the gate named `name` (already lowercased) onto `circuit`. An
/// unrecognized name, or too few parameters/qubits for a recognized one, is
/// a hard error -- silently dropping it would leave a different, wrong
/// circuit that still looks like it loaded successfully.
fn push_gate(circuit: &mut Circuit, name: &str, params: &[f64], qubits: &[usize]) -> Result<(), String> {
    match name {
        "h" => circuit.push(Gate::H, vec![qubits[0]]),
        "x" => circuit.push(Gate::X, vec![qubits[0]]),
        "y" => circuit.push(Gate::Y, vec![qubits[0]]),
        "z" => circuit.push(Gate::Z, vec![qubits[0]]),
        "s" => circuit.push(Gate::S, vec![qubits[0]]),
        "sdg" => circuit.push(Gate::Sdg, vec![qubits[0]]),
        "t" => circuit.push(Gate::T, vec![qubits[0]]),
        "tdg" => circuit.push(Gate::Tdg, vec![qubits[0]]),
        "id" => circuit.push(Gate::Id, vec![qubits[0]]),
        "rz" | "u1" => {
            let theta = *params.first().ok_or("rz/u1 needs 1 parameter")?;
            circuit.push(Gate::Rz(theta), vec![qubits[0]]);
        }
        "u" | "u3" => {
            if params.len() < 3 {
                return Err(format!("u/u3 needs 3 parameters, got {}", params.len()));
            }
            circuit.push(Gate::U3(params[0], params[1], params[2]), vec![qubits[0]]);
        }
        "u2" => {
            if params.len() < 2 {
                return Err(format!("u2 needs 2 parameters, got {}", params.len()));
            }
            circuit.push(Gate::U3(std::f64::consts::FRAC_PI_2, params[0], params[1]), vec![qubits[0]]);
        }
        // Rx(theta) = U3(theta, -pi/2, pi/2); Ry(theta) = U3(theta, 0, 0) --
        // standard qiskit identities, exact (not approximated).
        "rx" => {
            let theta = *params.first().ok_or("rx needs 1 parameter")?;
            circuit.push(Gate::U3(theta, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2), vec![qubits[0]]);
        }
        "ry" => {
            let theta = *params.first().ok_or("ry needs 1 parameter")?;
            circuit.push(Gate::U3(theta, 0.0, 0.0), vec![qubits[0]]);
        }
        // sx = Rx(pi/2), sxdg = Rx(-pi/2), both up to the global phase this
        // pipeline's phase-invariant distance never cares about.
        "sx" => {
            circuit.push(
                Gate::U3(std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                vec![qubits[0]],
            );
        }
        "sxdg" => {
            circuit.push(
                Gate::U3(-std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                vec![qubits[0]],
            );
        }
        "cx" | "cnot" => {
            if qubits.len() < 2 {
                return Err("cx needs 2 qubit operands".to_string());
            }
            circuit.push(Gate::Cx, vec![qubits[0], qubits[1]]);
        }
        "cz" => {
            if qubits.len() < 2 {
                return Err("cz needs 2 qubit operands".to_string());
            }
            circuit.push(Gate::Cz, vec![qubits[0], qubits[1]]);
        }
        "swap" => {
            if qubits.len() < 2 {
                return Err("swap needs 2 qubit operands".to_string());
            }
            circuit.push(Gate::Swap, vec![qubits[0], qubits[1]]);
        }
        "ccx" | "toffoli" | "ccnot" => {
            if qubits.len() < 3 {
                return Err("ccx needs 3 qubit operands".to_string());
            }
            push_toffoli(circuit, qubits[0], qubits[1], qubits[2]);
        }
        "cswap" | "fredkin" => {
            if qubits.len() < 3 {
                return Err("cswap needs 3 qubit operands".to_string());
            }
            let (control, t1, t2) = (qubits[0], qubits[1], qubits[2]);
            // CSWAP(control; t1, t2) = CX(t2,t1) . CCX(control,t1,t2) . CX(t2,t1),
            // the standard identity turning a controlled-swap into a Toffoli
            // sandwiched between two CNOTs.
            circuit.push(Gate::Cx, vec![t2, t1]);
            push_toffoli(circuit, control, t1, t2);
            circuit.push(Gate::Cx, vec![t2, t1]);
        }
        other => return Err(format!("unsupported gate '{other}'")),
    }
    Ok(())
}

/// Standard 7-T Toffoli (CCX) decomposition into this pipeline's own
/// vocabulary (H, Cx, T, Tdg) -- exact, no approximation, so this is a
/// front-end gate identity rather than something Stage 6 needs to
/// synthesize. `a`/`b` are the two controls, `c` is the target (Nielsen &
/// Chuang Fig. 4.9 -- the same circuit qiskit's own `CCXGate.definition`
/// emits).
fn push_toffoli(circuit: &mut Circuit, a: usize, b: usize, c: usize) {
    circuit.push(Gate::H, vec![c]);
    circuit.push(Gate::Cx, vec![b, c]);
    circuit.push(Gate::Tdg, vec![c]);
    circuit.push(Gate::Cx, vec![a, c]);
    circuit.push(Gate::T, vec![c]);
    circuit.push(Gate::Cx, vec![b, c]);
    circuit.push(Gate::Tdg, vec![c]);
    circuit.push(Gate::Cx, vec![a, c]);
    circuit.push(Gate::T, vec![b]);
    circuit.push(Gate::T, vec![c]);
    circuit.push(Gate::H, vec![c]);
    circuit.push(Gate::Cx, vec![a, b]);
    circuit.push(Gate::T, vec![a]);
    circuit.push(Gate::Tdg, vec![b]);
    circuit.push(Gate::Cx, vec![a, b]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::{distance, identity, C64};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn ccx_decomposition_matches_toffoli_matrix() {
        let mut c = Circuit::new(3);
        push_toffoli(&mut c, 0, 1, 2);
        // Qubit 0 is the MSB: index = 4*a + 2*b + c, so a=b=1 is rows/cols 6/7.
        let mut expected = identity(8);
        expected.swap_columns(6, 7);
        assert!(distance(&expected, &c.get_unitary()) < 1e-9);
    }

    #[test]
    fn cswap_decomposition_matches_fredkin_matrix() {
        let mut c = Circuit::new(3);
        push_gate(&mut c, "cswap", &[], &[0, 1, 2]).unwrap();
        // control=0 is the MSB; control=1 swaps t1/t2, i.e. rows/cols 5/6.
        let mut expected = identity(8);
        expected.swap_columns(5, 6);
        assert!(distance(&expected, &c.get_unitary()) < 1e-9);
    }

    #[test]
    fn toffoli_gate_name_alias_is_accepted() {
        let mut c = Circuit::new(3);
        push_gate(&mut c, "toffoli", &[], &[0, 1, 2]).unwrap();
        assert_eq!(c.ops.len(), 15);
    }

    #[test]
    fn fredkin_gate_name_alias_is_accepted() {
        let mut c = Circuit::new(3);
        push_gate(&mut c, "fredkin", &[], &[0, 1, 2]).unwrap();
        assert_eq!(c.ops.len(), 17);
    }

    fn rx_matrix(theta: f64) -> crate::cliffordt::matrix::Unitary {
        let h = theta / 2.0;
        crate::cliffordt::matrix::Unitary::from_row_slice(
            2,
            2,
            &[C64::new(h.cos(), 0.0), C64::new(0.0, -h.sin()), C64::new(0.0, -h.sin()), C64::new(h.cos(), 0.0)],
        )
    }

    fn ry_matrix(theta: f64) -> crate::cliffordt::matrix::Unitary {
        let h = theta / 2.0;
        crate::cliffordt::matrix::Unitary::from_row_slice(
            2,
            2,
            &[C64::new(h.cos(), 0.0), C64::new(-h.sin(), 0.0), C64::new(h.sin(), 0.0), C64::new(h.cos(), 0.0)],
        )
    }

    fn write_qasm(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn loads_basic_gates_and_qubit_count() {
        let f = write_qasm(&[
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            "qreg q[2];",
            "h q[0];",
            "cx q[0],q[1];",
            "rz(0.5) q[1];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.n_qubits, 2);
        assert_eq!(c.ops.len(), 3);
        assert!(matches!(c.ops[0].gate, Gate::H));
        assert!(matches!(c.ops[1].gate, Gate::Cx));
        assert!(matches!(c.ops[2].gate, Gate::Rz(a) if (a - 0.5).abs() < 1e-12));
    }

    #[test]
    fn parses_pi_relative_angles() {
        let f = write_qasm(&["qreg q[1];", "rz(pi/4) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(matches!(c.ops[0].gate, Gate::Rz(a) if (a - std::f64::consts::FRAC_PI_4).abs() < 1e-12));
    }

    #[test]
    fn parses_pi_first_multiplication_form() {
        // qiskit's own qasm2.dump emits `pi*0.35...`, pi first -- distinct
        // from the `0.35*pi` form above. This exact form once made every
        // rx/ry/rz/u3 gate in a real input file silently disappear.
        let f = write_qasm(&["qreg q[1];", "rz(pi*0.25) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(matches!(c.ops[0].gate, Gate::Rz(a) if (a - std::f64::consts::FRAC_PI_4).abs() < 1e-12));
    }

    #[test]
    fn parses_u_gate_with_three_params() {
        let f = write_qasm(&["qreg q[1];", "u(0.1,0.2,0.3) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        match c.ops[0].gate {
            Gate::U3(t, p, l) => {
                assert!((t - 0.1).abs() < 1e-12);
                assert!((p - 0.2).abs() < 1e-12);
                assert!((l - 0.3).abs() < 1e-12);
            }
            _ => panic!("expected U3 gate"),
        }
    }

    #[test]
    fn ignores_comments_and_barriers() {
        let f = write_qasm(&[
            "// a comment",
            "qreg q[2];",
            "h q[0]; // inline comment",
            "barrier q[0],q[1];",
            "cx q[0],q[1];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.ops.len(), 2);
    }

    #[test]
    fn rx_matches_standard_rx_matrix() {
        let f = write_qasm(&["qreg q[1];", "rx(0.8) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(distance(&rx_matrix(0.8), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn ry_matches_standard_ry_matrix() {
        let f = write_qasm(&["qreg q[1];", "ry(1.3) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(distance(&ry_matrix(1.3), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn sx_matches_rx_pi_over_2_up_to_phase() {
        let f = write_qasm(&["qreg q[1];", "sx q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(distance(&rx_matrix(std::f64::consts::FRAC_PI_2), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn sxdg_matches_rx_negative_pi_over_2_up_to_phase() {
        let f = write_qasm(&["qreg q[1];", "sxdg q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(distance(&rx_matrix(-std::f64::consts::FRAC_PI_2), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn unsupported_gate_is_a_hard_error_not_a_silent_skip() {
        let f = write_qasm(&["qreg q[1];", "frobnicate q[0];"]);
        let result = load_qasm(f.path().to_str().unwrap());
        assert!(result.is_err(), "an unsupported gate must fail loudly, not silently disappear");
    }

    #[test]
    fn unparseable_angle_is_a_hard_error_not_a_silent_skip() {
        let f = write_qasm(&["qreg q[1];", "rz(not_a_number) q[0];"]);
        let result = load_qasm(f.path().to_str().unwrap());
        assert!(result.is_err(), "an unparseable angle must fail loudly, not silently produce an empty param list");
    }
}
