//! Minimal OpenQASM 2.0 subset loader for this pipeline's own gate
//! vocabulary (Clifford generators, `rz`, a general `u`/`u3` for whatever
//! hasn't been pre-decomposed, and `cx`/`cz`/`swap`), plus the specific
//! OpenQASM 3 syntax forms (`qubit[N] name;` register declarations, possibly
//! more than one per file; `bit[N] name;`; assignment-style measurement)
//! needed to load MQT Bench's OQASM3 exports -- not a general OpenQASM 3
//! parser, and other OQASM3-only constructs (control flow, subroutines,
//! gate modifiers, ...) are still unsupported and fail loudly like anything
//! else this module doesn't recognize.
//!
//! Deliberately narrower than a full QASM parser -- matches the line-based
//! parsing style already used by `transpile.rs`'s own `parse_qasm` (split
//! on `[`/`]` for qubit indices, lowercase-prefix match for gate names),
//! extended to handle parameterized gates.
//!
//! Also expands user-defined `gate NAME(params) qargs { body }` macros
//! (`GateDef`/`BodyOp`/`expand_gate_call`) inline at every call site, plus
//! built-in identities for the qelib1.inc extensions (`rx`/`ry`/`sx`/
//! `sxdg`/`cry`/`crz`/`rzz`) actually seen in real inputs, since
//! `include "qelib1.inc"` itself is never read (just skipped) rather than
//! resolved from an external file.
//!
//! Deliberately fails loudly (a hard `io::Error`, naming the exact line and
//! reason) on anything it can't parse, rather than silently skipping it --
//! a silently-dropped gate or unparsed angle produces a *different, wrong*
//! circuit that still "compiles" and even "verifies" successfully against
//! itself, which is a far worse failure mode than a load error.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Error, ErrorKind};

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

/// A user-defined `gate NAME(params) qargs { body }` template, registered
/// while parsing and expanded inline at every call site -- QASM's `gate`
/// blocks are macros, not run-time entities, so there's no need to represent
/// them as anything richer than "how to rewrite a call into built-in ops".
struct GateDef {
    param_formals: Vec<String>,
    qubit_formals: Vec<String>,
    body: Vec<BodyOp>,
}

/// One statement inside a `gate` body, still in terms of the definition's
/// own formal parameter/qubit names -- resolved against a specific call's
/// actual arguments by `expand_gate_call`.
struct BodyOp {
    name: String,
    param_exprs: Vec<String>,
    qubit_formals: Vec<String>,
}

pub fn load_qasm(path: &str) -> io::Result<Circuit> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut circuit = Circuit::new(0);
    let mut defs: HashMap<String, GateDef> = HashMap::new();
    let mut defining: Option<GateDef> = None;
    let mut defining_name = String::new();
    let mut register_offsets: HashMap<String, usize> = HashMap::new();
    let mut total_qubits: usize = 0;

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let stripped = strip_comment(&line);
        let stripped = stripped.trim();

        if let Some(def) = defining.as_mut() {
            if stripped == "{" {
                continue;
            }
            if stripped == "}" {
                defs.insert(std::mem::take(&mut defining_name), defining.take().unwrap());
                continue;
            }
            if stripped.is_empty() {
                continue;
            }
            let body_op = parse_body_op(stripped).map_err(|e| parse_error(line_no, &line, &e))?;
            def.body.push(body_op);
            continue;
        }

        if stripped.is_empty()
            || stripped.starts_with("OPENQASM")
            || stripped.starts_with("include")
            || stripped.starts_with("creg")
            || stripped.starts_with("bit[")
            || stripped.starts_with('{')
            || stripped.starts_with('}')
            || stripped.starts_with("barrier")
            || stripped.starts_with("measure")
            || stripped.contains(" = measure")
        {
            continue;
        }

        if stripped.starts_with("gate ") {
            let (name, param_formals, qubit_formals) =
                parse_gate_header(stripped).map_err(|e| parse_error(line_no, &line, &e))?;
            defining_name = name;
            defining = Some(GateDef { param_formals, qubit_formals, body: Vec::new() });
            continue;
        }

        if stripped.starts_with("qreg") || stripped.starts_with("qubit[") {
            let (name, n) =
                parse_register_decl(stripped).map_err(|e| parse_error(line_no, &line, &e))?;
            register_offsets.insert(name, total_qubits);
            total_qubits += n;
            circuit = Circuit::new(total_qubits);
            continue;
        }

        let (name, params, qubit_refs) =
            parse_gate_line(stripped).map_err(|e| parse_error(line_no, &line, &e))?;
        let mut qubits = Vec::with_capacity(qubit_refs.len());
        for (reg_name, local_idx) in &qubit_refs {
            let offset = *register_offsets.get(reg_name).ok_or_else(|| {
                parse_error(
                    line_no,
                    &line,
                    &format!("reference to undeclared register '{reg_name}'"),
                )
            })?;
            qubits.push(offset + local_idx);
        }
        expand_gate_call(&mut circuit, &defs, &name, &params, &qubits)
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

/// Parse a register declaration in either OpenQASM 2 (`qreg NAME[N];`) or
/// OpenQASM 3 (`qubit[N] NAME;`) spelling into (name, size) -- the two
/// spellings put the name and the bracketed size on opposite sides.
fn parse_register_decl(line: &str) -> Result<(String, usize), String> {
    let line = line.trim_end_matches(';').trim();
    let open =
        line.find('[').ok_or_else(|| format!("expected '[' in register declaration '{line}'"))?;
    let close =
        line.find(']').ok_or_else(|| format!("expected ']' in register declaration '{line}'"))?;
    let size = line[open + 1..close]
        .parse::<usize>()
        .map_err(|_| format!("could not parse register size in '{line}'"))?;
    let name = if line.starts_with("qreg") {
        line[..open].trim_start_matches("qreg").trim().to_string()
    } else {
        line[close + 1..].trim().to_string()
    };
    if name.is_empty() {
        return Err(format!("register declaration '{line}' has no name"));
    }
    Ok((name, size))
}

/// Every qubit reference in a gate call's argument list, as (register name,
/// local index) pairs, in order -- e.g. `"a[0], b[0], cin[0]"` yields
/// `[("a",0), ("b",0), ("cin",0)]`. Resolving these to global indices needs
/// the register offset table only `load_qasm` maintains, so that's done by
/// the caller, not here.
fn named_qubit_refs(line: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('[') {
        let name_start = rest[..start]
            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
            .map(|i| i + 1)
            .unwrap_or(0);
        let name = rest[name_start..start].trim().to_string();
        let Some(end) = rest[start..].find(']') else { break };
        if let Ok(idx) = rest[start + 1..start + end].parse::<usize>() {
            out.push((name, idx));
        }
        rest = &rest[start + end + 1..];
    }
    out
}

/// Split `name(p0, p1, ...) q[a], q[b];` into (name, params, qubit refs).
/// A gate name followed by unparseable parameters is a hard error, not a
/// silently-empty parameter list. Qubit references are (register name,
/// local index) pairs, not resolved global indices -- see `named_qubit_refs`.
fn parse_gate_line(line: &str) -> Result<(String, Vec<f64>, Vec<(String, usize)>), String> {
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
            params.push(parse_angle_expr(raw).ok_or_else(|| {
                format!("could not parse angle expression '{raw}' for gate '{name}'")
            })?);
        }
        (name, params, &line[close + 1..])
    } else {
        let split = line
            .find(char::is_whitespace)
            .ok_or_else(|| format!("expected a qubit argument after gate name in '{line}'"))?;
        (line[..split].trim().to_lowercase(), Vec::new(), &line[split..])
    };

    let qubits = named_qubit_refs(rest_after_paren);
    if qubits.is_empty() {
        return Err(format!("no qubit operand found for gate '{name}'"));
    }
    Ok((name, params, qubits))
}

/// Parse a numeric angle expression: plain floats, and `pi`-relative forms
/// commonly emitted by QASM exporters -- `pi`, `-pi/2`, `pi/4`, `3*pi/8`,
/// `7*pi/2` (qiskit's own `qasm2.dump` convention for a rational multiple of
/// pi with a numerator other than 1), and `pi*0.35`, pi first.
fn parse_angle_expr(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let lower = s.to_lowercase();
    let (sign, lower) = if let Some(stripped) = lower.strip_prefix('-') {
        (-1.0, stripped)
    } else {
        (1.0, lower.as_str())
    };

    // Peel off at most one trailing "/denom" first -- e.g. "7*pi/2" becomes
    // main="7*pi", denom=2.0 -- so the multiply-form checks below don't also
    // need to special-case a divisor tacked on the end of them.
    let (main, denom) = match lower.rfind('/') {
        Some(idx) => (&lower[..idx], lower[idx + 1..].parse::<f64>().ok()?),
        None => (lower, 1.0),
    };

    let numer = if main == "pi" {
        1.0
    } else if let Some(rest) = main.strip_prefix("pi*") {
        rest.parse().ok()?
    } else if let Some(rest) = main.strip_suffix("*pi") {
        rest.parse().ok()?
    } else {
        return None;
    };

    Some(sign * numer * std::f64::consts::PI / denom)
}

/// Split `gate NAME(p0, p1, ...) q0, q1, ...` (an optional trailing `{`
/// already handled by the caller not mattering here, since it's stripped)
/// into (name, formal parameter names, formal qubit names). Unlike
/// `parse_gate_line`, qubit arguments here are bare formal names (`q0`), not
/// bracketed register indices -- a definition's qubits aren't bound to any
/// actual index until a call site supplies one.
fn parse_gate_header(line: &str) -> Result<(String, Vec<String>, Vec<String>), String> {
    let rest = line.strip_prefix("gate").unwrap_or(line).trim();
    let rest = rest.trim_end_matches('{').trim();

    let (name, params, qubits_part) = if let Some(open) = rest.find('(') {
        let close = rest.find(')').ok_or_else(|| "unmatched '(' in gate definition".to_string())?;
        let name = rest[..open].trim().to_lowercase();
        let params: Vec<String> = rest[open + 1..close]
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        (name, params, rest[close + 1..].trim())
    } else {
        let split = rest
            .find(char::is_whitespace)
            .ok_or_else(|| format!("expected qubit arguments in gate definition '{line}'"))?;
        (rest[..split].trim().to_lowercase(), Vec::new(), rest[split..].trim())
    };

    let qubits: Vec<String> =
        qubits_part.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if qubits.is_empty() {
        return Err(format!("gate definition '{name}' declares no qubit arguments"));
    }
    Ok((name, params, qubits))
}

/// Parse one statement inside a `gate` body into a template `BodyOp`: like
/// `parse_gate_line`, but qubit arguments are bare formal names (`q0`), not
/// bracketed indices, and parameter expressions are kept as raw text --
/// they may reference the enclosing definition's own formal parameters
/// (e.g. `lambda/2`), which can't be resolved until a call site binds them.
fn parse_body_op(line: &str) -> Result<BodyOp, String> {
    let line = line.trim_end_matches(';').trim();
    if line.is_empty() {
        return Err("empty statement in gate body".to_string());
    }

    let (name, param_exprs, rest_after_paren) = if let Some(open) = line.find('(') {
        let close =
            line.find(')').ok_or_else(|| "unmatched '(' in gate body statement".to_string())?;
        let name = line[..open].trim().to_lowercase();
        let param_exprs: Vec<String> =
            line[open + 1..close].split(',').map(|s| s.trim().to_string()).collect();
        (name, param_exprs, line[close + 1..].trim())
    } else {
        let split = line
            .find(char::is_whitespace)
            .ok_or_else(|| format!("expected a qubit argument after gate name in '{line}'"))?;
        (line[..split].trim().to_lowercase(), Vec::new(), line[split..].trim())
    };

    let qubit_formals: Vec<String> = rest_after_paren
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if qubit_formals.is_empty() {
        return Err(format!("no qubit operand found in gate body statement for '{name}'"));
    }
    Ok(BodyOp { name, param_exprs, qubit_formals })
}

/// Resolve a gate body's parameter expression against a specific call's
/// bound formal-parameter values. Beyond a bare (optionally negated) formal
/// name, this only understands `name/N` and `name*N` (either order) --
/// exactly the forms qelib1.inc's own `cry`/`crz` bodies use (`lambda/2`,
/// `-lambda/2`) -- not a general arithmetic evaluator; anything fancier
/// falls through to `parse_angle_expr` and, failing that, is a hard error
/// rather than a silent guess (see module docs).
fn eval_param_expr(expr: &str, bindings: &[(String, f64)]) -> Option<f64> {
    let trimmed = expr.trim();
    for (name, value) in bindings {
        let (sign, rest) =
            if let Some(r) = trimmed.strip_prefix('-') { (-1.0, r.trim()) } else { (1.0, trimmed) };
        if rest == name {
            return Some(sign * value);
        }
        if let Some(denom_str) = rest.strip_prefix(name.as_str()).and_then(|r| r.strip_prefix('/'))
        {
            let denom: f64 = denom_str.trim().parse().ok()?;
            return Some(sign * value / denom);
        }
        if let Some(numer_str) = rest.strip_prefix(name.as_str()).and_then(|r| r.strip_prefix('*'))
        {
            let numer: f64 = numer_str.trim().parse().ok()?;
            return Some(sign * value * numer);
        }
        if let Some(numer_str) = rest.strip_suffix(&format!("*{name}")) {
            let numer: f64 = numer_str.trim().parse().ok()?;
            return Some(sign * numer * value);
        }
    }
    parse_angle_expr(trimmed)
}

/// Map a body statement's formal qubit names to actual circuit indices via
/// the enclosing definition's own formal qubit list and a specific call's
/// actual qubits (matched by position).
fn resolve_body_qubits(
    def: &GateDef, body_qubit_formals: &[String], actual_qubits: &[usize],
) -> Result<Vec<usize>, String> {
    body_qubit_formals
        .iter()
        .map(|formal| {
            def.qubit_formals
                .iter()
                .position(|f| f == formal)
                .map(|idx| actual_qubits[idx])
                .ok_or_else(|| format!("gate body references unknown qubit '{formal}'"))
        })
        .collect()
}

/// Push the effect of calling gate `name` with `params`/`qubits` onto
/// `circuit`: expands it against `defs` if it's a user-defined `gate`
/// (recursively, so a custom gate's body may itself call another custom
/// gate), otherwise falls through to `push_gate`'s built-in vocabulary. A
/// file that defines its own version of a name this module also knows as a
/// built-in takes precedence, since only names actually declared via
/// `gate` end up in `defs` in the first place.
fn expand_gate_call(
    circuit: &mut Circuit, defs: &HashMap<String, GateDef>, name: &str, params: &[f64],
    qubits: &[usize],
) -> Result<(), String> {
    let Some(def) = defs.get(name) else {
        return push_gate(circuit, name, params, qubits);
    };
    if def.qubit_formals.len() != qubits.len() {
        return Err(format!(
            "gate '{name}' expects {} qubit argument(s), got {}",
            def.qubit_formals.len(),
            qubits.len()
        ));
    }
    let bindings: Vec<(String, f64)> =
        def.param_formals.iter().cloned().zip(params.iter().copied()).collect();
    for body_op in &def.body {
        let resolved_params: Vec<f64> = body_op
            .param_exprs
            .iter()
            .map(|expr| {
                eval_param_expr(expr, &bindings).ok_or_else(|| {
                    format!(
                        "could not resolve parameter expression '{expr}' in body of gate '{name}'"
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let resolved_qubits = resolve_body_qubits(def, &body_op.qubit_formals, qubits)?;
        expand_gate_call(circuit, defs, &body_op.name, &resolved_params, &resolved_qubits)?;
    }
    Ok(())
}

/// Push the gate named `name` (already lowercased) onto `circuit`. An
/// unrecognized name, or too few parameters/qubits for a recognized one, is
/// a hard error -- silently dropping it would leave a different, wrong
/// circuit that still looks like it loaded successfully.
fn push_gate(
    circuit: &mut Circuit, name: &str, params: &[f64], qubits: &[usize],
) -> Result<(), String> {
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
            circuit
                .push(Gate::U3(std::f64::consts::FRAC_PI_2, params[0], params[1]), vec![qubits[0]]);
        }
        // Rx(theta) = U3(theta, -pi/2, pi/2); Ry(theta) = U3(theta, 0, 0) --
        // standard qiskit identities, exact (not approximated).
        "rx" => {
            let theta = *params.first().ok_or("rx needs 1 parameter")?;
            circuit.push(
                Gate::U3(theta, -std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                vec![qubits[0]],
            );
        }
        "ry" => {
            let theta = *params.first().ok_or("ry needs 1 parameter")?;
            circuit.push(Gate::U3(theta, 0.0, 0.0), vec![qubits[0]]);
        }
        // sx = Rx(pi/2), sxdg = Rx(-pi/2), both up to the global phase this
        // pipeline's phase-invariant distance never cares about.
        "sx" => {
            circuit.push(
                Gate::U3(
                    std::f64::consts::FRAC_PI_2,
                    -std::f64::consts::FRAC_PI_2,
                    std::f64::consts::FRAC_PI_2,
                ),
                vec![qubits[0]],
            );
        }
        "sxdg" => {
            circuit.push(
                Gate::U3(
                    -std::f64::consts::FRAC_PI_2,
                    -std::f64::consts::FRAC_PI_2,
                    std::f64::consts::FRAC_PI_2,
                ),
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
        // Standard qelib1.inc extensions, not otherwise redefined by the
        // input file itself (see `expand_gate_call`'s doc comment for when
        // a file's own `gate cry ...` shadows this) -- exact identities,
        // not approximated, matching qelib1.inc's own decompositions.
        "cry" => {
            if qubits.len() < 2 {
                return Err("cry needs 2 qubit operands".to_string());
            }
            let lambda = *params.first().ok_or("cry needs 1 parameter")?;
            let (a, b) = (qubits[0], qubits[1]);
            circuit.push(Gate::U3(lambda / 2.0, 0.0, 0.0), vec![b]);
            circuit.push(Gate::Cx, vec![a, b]);
            circuit.push(Gate::U3(-lambda / 2.0, 0.0, 0.0), vec![b]);
            circuit.push(Gate::Cx, vec![a, b]);
        }
        "crz" => {
            if qubits.len() < 2 {
                return Err("crz needs 2 qubit operands".to_string());
            }
            let lambda = *params.first().ok_or("crz needs 1 parameter")?;
            let (a, b) = (qubits[0], qubits[1]);
            circuit.push(Gate::Rz(lambda / 2.0), vec![b]);
            circuit.push(Gate::Cx, vec![a, b]);
            circuit.push(Gate::Rz(-lambda / 2.0), vec![b]);
            circuit.push(Gate::Cx, vec![a, b]);
        }
        // rzz(theta) = CX(a,b); u1(theta) b; CX(a,b) in qelib1.inc -- u1 and
        // rz are the same gate up to a global phase this pipeline's own
        // phase-invariant distance never cares about (see the "rz" | "u1"
        // case above, which already treats them identically).
        "rzz" => {
            if qubits.len() < 2 {
                return Err("rzz needs 2 qubit operands".to_string());
            }
            let theta = *params.first().ok_or("rzz needs 1 parameter")?;
            let (a, b) = (qubits[0], qubits[1]);
            circuit.push(Gate::Cx, vec![a, b]);
            circuit.push(Gate::Rz(theta), vec![b]);
            circuit.push(Gate::Cx, vec![a, b]);
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
/// front-end gate identity rather than something Stage 4 needs to
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
    use crate::cliffordt::matrix::{C64, distance, identity};
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

    /// qiskit's `qasm2.dump` emits `N*pi/M` for a rational multiple of pi
    /// whose numerator isn't 1 (e.g. `7*pi/2`) -- a combined
    /// multiply-then-divide form none of the parser's other branches match.
    #[test]
    fn angle_expr_handles_numerator_and_denominator_together() {
        let pi = std::f64::consts::PI;
        assert!((parse_angle_expr("7*pi/2").unwrap() - 7.0 * pi / 2.0).abs() < 1e-12);
        assert!((parse_angle_expr("-7*pi/2").unwrap() - (-7.0 * pi / 2.0)).abs() < 1e-12);
        assert!((parse_angle_expr("3*pi/8").unwrap() - 3.0 * pi / 8.0).abs() < 1e-12);
        assert!((parse_angle_expr("pi*3/8").unwrap() - 3.0 * pi / 8.0).abs() < 1e-12);
    }

    #[test]
    fn angle_expr_still_handles_simple_forms() {
        let pi = std::f64::consts::PI;
        assert!((parse_angle_expr("pi").unwrap() - pi).abs() < 1e-12);
        assert!((parse_angle_expr("-pi").unwrap() - (-pi)).abs() < 1e-12);
        assert!((parse_angle_expr("pi/4").unwrap() - pi / 4.0).abs() < 1e-12);
        assert!((parse_angle_expr("pi*0.35").unwrap() - pi * 0.35).abs() < 1e-12);
        assert!((parse_angle_expr("1.23").unwrap() - 1.23).abs() < 1e-12);
        assert!(parse_angle_expr("banana").is_none());
        assert!(parse_angle_expr("pi/").is_none());
    }

    fn rx_matrix(theta: f64) -> crate::cliffordt::matrix::Unitary {
        let h = theta / 2.0;
        crate::cliffordt::matrix::Unitary::from_row_slice(
            2,
            2,
            &[
                C64::new(h.cos(), 0.0),
                C64::new(0.0, -h.sin()),
                C64::new(0.0, -h.sin()),
                C64::new(h.cos(), 0.0),
            ],
        )
    }

    fn ry_matrix(theta: f64) -> crate::cliffordt::matrix::Unitary {
        let h = theta / 2.0;
        crate::cliffordt::matrix::Unitary::from_row_slice(
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

    fn rz_matrix(theta: f64) -> crate::cliffordt::matrix::Unitary {
        let h = theta / 2.0;
        crate::cliffordt::matrix::Unitary::from_row_slice(
            2,
            2,
            &[
                C64::new(h.cos(), -h.sin()),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(h.cos(), h.sin()),
            ],
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
        assert!(
            matches!(c.ops[0].gate, Gate::Rz(a) if (a - std::f64::consts::FRAC_PI_4).abs() < 1e-12)
        );
    }

    #[test]
    fn parses_pi_first_multiplication_form() {
        // qiskit's own qasm2.dump emits `pi*0.35...`, pi first -- distinct
        // from the `0.35*pi` form above.
        let f = write_qasm(&["qreg q[1];", "rz(pi*0.25) q[0];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert!(
            matches!(c.ops[0].gate, Gate::Rz(a) if (a - std::f64::consts::FRAC_PI_4).abs() < 1e-12)
        );
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
        assert!(
            result.is_err(),
            "an unparseable angle must fail loudly, not silently produce an empty param list"
        );
    }

    #[test]
    fn cry_builtin_matches_controlled_ry_matrix() {
        let f = write_qasm(&["qreg q[2];", "cry(0.7) q[0],q[1];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        let ry = ry_matrix(0.7);
        let mut expected = identity(4);
        for i in 0..2 {
            for j in 0..2 {
                expected[(2 + i, 2 + j)] = ry[(i, j)];
            }
        }
        assert!(distance(&expected, &c.get_unitary()) < 1e-9);
    }

    #[test]
    fn crz_builtin_matches_controlled_rz_matrix() {
        let f = write_qasm(&["qreg q[2];", "crz(1.1) q[0],q[1];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        let rz = rz_matrix(1.1);
        let mut expected = identity(4);
        for i in 0..2 {
            for j in 0..2 {
                expected[(2 + i, 2 + j)] = rz[(i, j)];
            }
        }
        assert!(distance(&expected, &c.get_unitary()) < 1e-9);
    }

    #[test]
    fn rzz_builtin_matches_zz_rotation_matrix() {
        let f = write_qasm(&["qreg q[2];", "rzz(0.9) q[0],q[1];"]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        let h: f64 = 0.9 / 2.0;
        let diag = [
            C64::new(h.cos(), -h.sin()),
            C64::new(h.cos(), h.sin()),
            C64::new(h.cos(), h.sin()),
            C64::new(h.cos(), -h.sin()),
        ];
        let mut expected = identity(4);
        for (i, d) in diag.iter().enumerate() {
            expected[(i, i)] = *d;
        }
        assert!(distance(&expected, &c.get_unitary()) < 1e-9);
    }

    #[test]
    fn custom_gate_definition_expands_at_call_site() {
        let f = write_qasm(&[
            "qreg q[2];",
            "gate bell q0,q1",
            "{",
            " h q0;",
            " cx q0,q1;",
            "}",
            "bell q[0],q[1];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.ops.len(), 2);
        assert!(matches!(c.ops[0].gate, Gate::H));
        assert_eq!(c.ops[0].qubits, vec![0]);
        assert!(matches!(c.ops[1].gate, Gate::Cx));
        assert_eq!(c.ops[1].qubits, vec![0, 1]);
    }

    #[test]
    fn custom_gate_body_can_reference_formal_parameter_with_arithmetic() {
        // Mirrors qelib1.inc's own `crz` body (`lambda/2`, `-lambda/2`) --
        // under a name that isn't itself a recognized built-in, to isolate
        // eval_param_expr's substitution from push_gate's own "crz" case.
        let f = write_qasm(&[
            "qreg q[2];",
            "gate myrot(theta) a,b",
            "{",
            " rz(theta/2) b;",
            " cx a,b;",
            " rz(-theta/2) b;",
            " cx a,b;",
            "}",
            "myrot(0.8) q[0],q[1];",
        ]);
        let custom = load_qasm(f.path().to_str().unwrap()).unwrap();

        let f2 = write_qasm(&["qreg q[2];", "crz(0.8) q[0],q[1];"]);
        let builtin = load_qasm(f2.path().to_str().unwrap()).unwrap();

        assert!(distance(&custom.get_unitary(), &builtin.get_unitary()) < 1e-12);
    }

    #[test]
    fn user_defined_gate_shadows_a_builtin_name() {
        let f = write_qasm(&[
            "qreg q[2];",
            "gate cry(lambda) a,b",
            "{",
            " cx a,b;",
            "}",
            "cry(0.5) q[0],q[1];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(
            c.ops.len(),
            1,
            "the file's own gate definition should override the built-in cry"
        );
        assert!(matches!(c.ops[0].gate, Gate::Cx));
    }

    #[test]
    fn custom_gate_body_literal_angle_ignoring_unused_formal_param() {
        // Reproduces the shape qiskit's older qasm2 export actually emits
        // for a bound RYYGate: a formal parameter the body never
        // references, an embedded literal angle instead, and a call site
        // that passes its own (different, likewise unused) value.
        let f = write_qasm(&[
            "qreg q[5];",
            "gate ryy(param0) q0,q1",
            "{",
            " rx(pi/2) q0;",
            " rx(pi/2) q1;",
            " cx q0,q1;",
            " rz(1.2345) q1;",
            " cx q0,q1;",
            " rx(-pi/2) q0;",
            " rx(-pi/2) q1;",
            "}",
            "ryy(9.999) q[3],q[4];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.ops.len(), 7);
        for op in &c.ops {
            for q in &op.qubits {
                assert!(
                    *q == 3 || *q == 4,
                    "body qubits must resolve to the call's actual qubits, not 0/1"
                );
            }
        }
    }

    #[test]
    fn multiple_custom_gate_definitions_with_distinct_names_stay_independent() {
        let f = write_qasm(&[
            "qreg q[2];",
            "gate g1 q0,q1",
            "{",
            " cx q0,q1;",
            "}",
            "gate g2 q0,q1",
            "{",
            " cx q1,q0;",
            "}",
            "g1 q[0],q[1];",
            "g2 q[0],q[1];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.ops.len(), 2);
        assert_eq!(c.ops[0].qubits, vec![0, 1]);
        assert_eq!(c.ops[1].qubits, vec![1, 0]);
    }

    /// Mirrors MQT Bench's OpenQASM 3 export shape: multiple named
    /// `qubit[N]` registers (not one flat `qreg`), a `bit[N]` classical
    /// declaration, and assignment-style measurement -- none of which
    /// OpenQASM 2's `qreg`/`creg`/`measure ... ->` spellings use.
    #[test]
    fn loads_oqasm3_multi_register_declarations_and_measurement() {
        let f = write_qasm(&[
            "OPENQASM 3.0;",
            "include \"stdgates.inc\";",
            "qubit[2] a;",
            "qubit[3] b;",
            "bit[5] meas;",
            "h a[0];",
            "cx a[1],b[0];",
            "meas[0] = measure a[0];",
        ]);
        let c = load_qasm(f.path().to_str().unwrap()).unwrap();
        assert_eq!(c.n_qubits, 5);
        assert_eq!(c.ops.len(), 2, "the bit[] decl and assignment-measure must not become ops");
        assert!(matches!(c.ops[0].gate, Gate::H));
        assert_eq!(c.ops[0].qubits, vec![0]);
        assert!(matches!(c.ops[1].gate, Gate::Cx));
        // a occupies offset 0..2, b starts right after at offset 2.
        assert_eq!(c.ops[1].qubits, vec![1, 2]);
    }

    #[test]
    fn gate_call_referencing_undeclared_register_is_a_hard_error() {
        let f = write_qasm(&["qubit[2] a;", "h b[0];"]);
        let result = load_qasm(f.path().to_str().unwrap());
        assert!(result.is_err(), "a reference to a never-declared register must fail loudly");
    }
}
