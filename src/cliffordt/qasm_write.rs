//! Write a (flat, Clifford+T) `Circuit` back out as OpenQASM 2.0.

use std::io::{self, Write};

use crate::cliffordt::qgate_circuit::{Circuit, Gate};

pub fn write_qasm(circuit: &Circuit, out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "OPENQASM 2.0;")?;
    writeln!(out, "include \"qelib1.inc\";")?;
    writeln!(out, "qreg q[{}];", circuit.n_qubits)?;
    for op in &circuit.ops {
        let line = match &op.gate {
            Gate::Id => continue,
            Gate::H => format!("h q[{}];", op.qubits[0]),
            Gate::X => format!("x q[{}];", op.qubits[0]),
            Gate::Y => format!("y q[{}];", op.qubits[0]),
            Gate::Z => format!("z q[{}];", op.qubits[0]),
            Gate::S => format!("s q[{}];", op.qubits[0]),
            Gate::Sdg => format!("sdg q[{}];", op.qubits[0]),
            Gate::T => format!("t q[{}];", op.qubits[0]),
            Gate::Tdg => format!("tdg q[{}];", op.qubits[0]),
            Gate::Rz(theta) => format!("rz({theta}) q[{}];", op.qubits[0]),
            Gate::U3(theta, phi, lam) => format!("u({theta},{phi},{lam}) q[{}];", op.qubits[0]),
            Gate::Cx => format!("cx q[{}],q[{}];", op.qubits[0], op.qubits[1]),
            Gate::Cz => format!("cz q[{}],q[{}];", op.qubits[0], op.qubits[1]),
            Gate::Swap => format!("swap q[{}],q[{}];", op.qubits[0], op.qubits[1]),
            Gate::Block(inner) => {
                // Should not appear in a fully unfolded circuit; unfold
                // defensively so output is always valid QASM regardless.
                write_qasm(&inner.unfold(), out)?;
                continue;
            }
        };
        writeln!(out, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;
    use crate::cliffordt::qasm::load_qasm;
    use tempfile::NamedTempFile;

    #[test]
    fn round_trips_through_qasm() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::T, vec![1]);
        c.push(Gate::Rz(0.4), vec![0]);

        let mut buf = Vec::new();
        write_qasm(&c, &mut buf).unwrap();

        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&buf).unwrap();
        let reloaded = load_qasm(f.path().to_str().unwrap()).unwrap();

        assert_eq!(reloaded.n_qubits, c.n_qubits);
        assert!(distance(&c.get_unitary(), &reloaded.get_unitary()) < 1e-12);
    }
}
