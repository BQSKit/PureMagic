//! Circuit representation for the pre-Clifford+T pipeline: gates that may
//! still carry a continuous parameter (`Rz`), or be an opaque sub-circuit
//! block (mirrors bqskit's `CircuitGate`, produced by partitioning/blocking).
//!
//! Deliberately distinct from `circuit.rs`'s `PauliProduct`-based model,
//! which represents *already-transpiled* Pauli-product circuits for lattice
//! surgery scheduling -- a different stage of the overall pipeline.

use rayon::prelude::*;

use crate::cliffordt::matrix::{identity, Unitary, C64};

fn re(x: f64) -> C64 {
    C64::new(x, 0.0)
}
fn cis(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    Id,
    H,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
    /// exp(-i*theta*Z/2), the standard geometric-rotation convention.
    Rz(f64),
    /// A general single-qubit unitary `Rz(phi)*Ry(theta)*Rz(lam)` (the
    /// qiskit/bqskit U3 convention, matching `synthesize::zyz_angles`),
    /// used only to represent an as-loaded input gate exactly -- Stage 1
    /// composes it into a block's unitary immediately, so it never
    /// survives as a free NLS parameter the way `Rz` does.
    U3(f64, f64, f64),
    Cx,
    Cz,
    Swap,
    /// An opaque sub-circuit treated as a single gate, produced by
    /// partitioning/blocking -- mirrors bqskit's `CircuitGate`.
    Block(Box<Circuit>),
}

impl Gate {
    pub fn is_clifford(&self) -> bool {
        matches!(
            self,
            Gate::Id | Gate::H | Gate::S | Gate::Sdg | Gate::X | Gate::Y | Gate::Z | Gate::Cx | Gate::Cz | Gate::Swap
        )
    }

    pub fn is_rz(&self) -> bool {
        matches!(self, Gate::Rz(_))
    }

    pub fn num_qubits(&self) -> usize {
        match self {
            Gate::Cx | Gate::Cz | Gate::Swap => 2,
            Gate::Block(c) => c.n_qubits,
            _ => 1,
        }
    }

    pub fn param(&self) -> Option<f64> {
        if let Gate::Rz(t) = self {
            Some(*t)
        } else {
            None
        }
    }

    pub fn set_param(&mut self, v: f64) {
        if let Gate::Rz(t) = self {
            *t = v;
        }
    }

    pub fn matrix(&self) -> Unitary {
        let s = re(std::f64::consts::FRAC_1_SQRT_2);
        let z = re(0.0);
        let o = re(1.0);
        match self {
            Gate::Id => identity(2),
            Gate::H => Unitary::from_row_slice(2, 2, &[s, s, s, -s]),
            Gate::S => Unitary::from_row_slice(2, 2, &[o, z, z, C64::new(0.0, 1.0)]),
            Gate::Sdg => Unitary::from_row_slice(2, 2, &[o, z, z, C64::new(0.0, -1.0)]),
            Gate::T => Unitary::from_row_slice(2, 2, &[o, z, z, cis(std::f64::consts::FRAC_PI_4)]),
            Gate::Tdg => Unitary::from_row_slice(2, 2, &[o, z, z, cis(-std::f64::consts::FRAC_PI_4)]),
            Gate::X => Unitary::from_row_slice(2, 2, &[z, o, o, z]),
            Gate::Y => Unitary::from_row_slice(2, 2, &[z, C64::new(0.0, -1.0), C64::new(0.0, 1.0), z]),
            Gate::Z => Unitary::from_row_slice(2, 2, &[o, z, z, -o]),
            Gate::Rz(theta) => {
                let h = theta / 2.0;
                Unitary::from_row_slice(2, 2, &[cis(-h), z, z, cis(h)])
            }
            Gate::U3(theta, phi, lam) => {
                let rz_phi = Gate::Rz(*phi).matrix();
                let rz_lam = Gate::Rz(*lam).matrix();
                let h = theta / 2.0;
                let ry = Unitary::from_row_slice(2, 2, &[re(h.cos()), re(-h.sin()), re(h.sin()), re(h.cos())]);
                rz_phi * ry * rz_lam
            }
            Gate::Cx => cx_matrix(),
            Gate::Cz => cz_matrix(),
            Gate::Swap => swap_matrix(),
            Gate::Block(c) => c.get_unitary(),
        }
    }
}

fn cx_matrix() -> Unitary {
    let o = re(1.0);
    let z = re(0.0);
    #[rustfmt::skip]
    let m = Unitary::from_row_slice(4, 4, &[
        o, z, z, z,
        z, o, z, z,
        z, z, z, o,
        z, z, o, z,
    ]);
    m
}

fn cz_matrix() -> Unitary {
    let o = re(1.0);
    let z = re(0.0);
    #[rustfmt::skip]
    let m = Unitary::from_row_slice(4, 4, &[
        o, z, z, z,
        z, o, z, z,
        z, z, o, z,
        z, z, z, -o,
    ]);
    m
}

fn swap_matrix() -> Unitary {
    let o = re(1.0);
    let z = re(0.0);
    #[rustfmt::skip]
    let m = Unitary::from_row_slice(4, 4, &[
        o, z, z, z,
        z, z, o, z,
        z, o, z, z,
        z, z, z, o,
    ]);
    m
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub gate: Gate,
    /// Qubit indices this gate acts on; `qubits[0]` is the gate matrix's
    /// own qubit 0.
    pub qubits: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Circuit {
    pub n_qubits: usize,
    pub ops: Vec<Operation>,
}

impl Circuit {
    pub fn new(n_qubits: usize) -> Self {
        Circuit { n_qubits, ops: Vec::new() }
    }

    pub fn push(&mut self, gate: Gate, qubits: Vec<usize>) {
        debug_assert_eq!(gate.num_qubits(), qubits.len());
        self.ops.push(Operation { gate, qubits });
    }

    /// Full 2^n_qubits x 2^n_qubits unitary of the whole circuit, gates
    /// applied in program order (ops[0] first). Uses `apply_gate_inplace`
    /// (direct slice application) rather than `embed` + a dense multiply
    /// per gate -- for a circuit with many thousands of gates (e.g. what
    /// `--verify` builds), the dense path's O(dim^3)-per-gate cost makes
    /// it impractically slow even at a qubit count small enough that the
    /// unitary itself is a perfectly reasonable size.
    pub fn get_unitary(&self) -> Unitary {
        let dim = 1usize << self.n_qubits;
        let mut u = identity(dim);
        for op in &self.ops {
            let local = op.gate.matrix();
            crate::cliffordt::matrix::apply_gate_inplace(&mut u, &local, &op.qubits, self.n_qubits);
        }
        u
    }

    /// Continuous parameters (one per `Rz` gate), in program order --
    /// mirrors bqskit's `circuit.params` convention used throughout the
    /// NLS instantiation code.
    pub fn params(&self) -> Vec<f64> {
        self.ops.iter().filter_map(|op| op.gate.param()).collect()
    }

    pub fn set_params(&mut self, params: &[f64]) {
        let mut it = params.iter();
        for op in &mut self.ops {
            if op.gate.is_rz() {
                if let Some(&v) = it.next() {
                    op.gate.set_param(v);
                }
            }
        }
    }

    pub fn num_params(&self) -> usize {
        self.ops.iter().filter(|op| op.gate.is_rz()).count()
    }

    /// The circuit's unitary (same value `get_unitary` returns) together
    /// with analytic `d(unitary)/d(theta)` for each `Rz` gate, in the same
    /// program order as `params()`. Used by `instantiate.rs`'s analytic
    /// Jacobian instead of perturbing each parameter and rebuilding the
    /// whole unitary from scratch (this module's old approach): each
    /// derivative is `Suffix_k * (-i/2 * Z) * embed(Rz(theta_k)) * Prefix_k`
    /// where `Prefix_k`/`Suffix_k` are the composed unitaries of every gate
    /// strictly before/after gate `k` -- so computing every derivative
    /// costs one forward sweep (building all prefixes -- the last of which
    /// *is* the circuit's unitary, returned here instead of paying for a
    /// separate `get_unitary()` call) plus one backward sweep (all
    /// suffixes) over the circuit's actual gate count, not one full rebuild
    /// per free parameter.
    pub fn unitary_and_rz_derivatives(&self) -> (Unitary, Vec<Unitary>) {
        let dim = 1usize << self.n_qubits;

        // prefixes[k] = product of ops[0..k) (ops[0] applied first, i.e.
        // rightmost in the matrix product) -- built incrementally via the
        // same left-multiply `apply_gate_inplace` uses for `get_unitary`.
        // prefixes[n] is exactly what `get_unitary` would return.
        let mut prefixes: Vec<Unitary> = Vec::with_capacity(self.ops.len() + 1);
        prefixes.push(identity(dim));
        for op in &self.ops {
            let mut u = prefixes.last().unwrap().clone();
            crate::cliffordt::matrix::apply_gate_inplace(&mut u, &op.gate.matrix(), &op.qubits, self.n_qubits);
            prefixes.push(u);
        }
        let built = prefixes.last().unwrap().clone();

        // suffixes[k] = product of ops[k..n) (i.e. everything from k
        // onward, ops[k] applied first so it's rightmost); suffixes[n] = I.
        // Built back-to-front: suffixes[k] = suffixes[k+1] * embed(ops[k]).
        // Uses the dense `embed` (not the in-place slice trick) since this
        // is a right-multiply, not a left-multiply -- fine at these block
        // sizes (dim <= 16 or so).
        let n = self.ops.len();
        let mut suffixes: Vec<Unitary> = vec![identity(dim); n + 1];
        for k in (0..n).rev() {
            let embedded = crate::cliffordt::matrix::embed(&self.ops[k].gate.matrix(), &self.ops[k].qubits, self.n_qubits);
            suffixes[k] = &suffixes[k + 1] * &embedded;
        }

        let mut derivs = Vec::with_capacity(self.num_params());
        for (k, op) in self.ops.iter().enumerate() {
            if let Gate::Rz(theta) = op.gate {
                // d/dtheta Rz(theta) = -i/2 * Z * Rz(theta).
                let d_local = Gate::Z.matrix() * Gate::Rz(theta).matrix() * C64::new(0.0, -0.5);
                let d_embedded = crate::cliffordt::matrix::embed(&d_local, &op.qubits, self.n_qubits);
                derivs.push(&suffixes[k + 1] * &d_embedded * &prefixes[k]);
            }
        }
        (built, derivs)
    }

    pub fn is_all_clifford(&self) -> bool {
        self.ops.iter().all(|op| op.gate.is_clifford())
    }

    /// Expand every top-level `Block` operation into its inner ops, with
    /// qubit indices remapped to this circuit's qubits, recursively.
    /// Mirrors bqskit's `UnfoldPass`.
    pub fn unfold(&self) -> Circuit {
        let mut out = Circuit::new(self.n_qubits);
        for op in &self.ops {
            if let Gate::Block(inner) = &op.gate {
                let unfolded_inner = inner.unfold();
                for inner_op in &unfolded_inner.ops {
                    let remapped: Vec<usize> = inner_op.qubits.iter().map(|&q| op.qubits[q]).collect();
                    out.ops.push(Operation { gate: inner_op.gate.clone(), qubits: remapped });
                }
            } else {
                out.ops.push(op.clone());
            }
        }
        out
    }

    /// Apply `f` to the inner circuit of every top-level `Block` operation,
    /// in parallel across blocks (they're independent windows by
    /// construction -- that's the whole point of partitioning); non-block
    /// operations pass through unchanged, and `f` also returns a per-block
    /// value of type `T`, collected (in block order, one entry per `Block`
    /// op) for the caller to aggregate afterward -- since `f` runs in
    /// parallel across blocks, it can't just mutate a captured counter the
    /// way a sequential `for` loop could. Mirrors bqskit's
    /// `ForEachBlockPass`, but bqskit distributes this work across a
    /// process pool per block, not just within one block's own solver
    /// calls -- `f` must be `Sync` (no captured mutable state) for the
    /// same reason.
    pub fn for_each_block_with<T: Send>(&self, f: impl Fn(&Circuit) -> (Circuit, T) + Sync) -> (Circuit, Vec<T>) {
        let results: Vec<(Operation, Option<T>)> = self
            .ops
            .par_iter()
            .map(|op| {
                if let Gate::Block(inner) = &op.gate {
                    let (new_inner, extra) = f(inner);
                    (Operation { gate: Gate::Block(Box::new(new_inner)), qubits: op.qubits.clone() }, Some(extra))
                } else {
                    (op.clone(), None)
                }
            })
            .collect();

        let mut ops = Vec::with_capacity(results.len());
        let mut extras = Vec::new();
        for (op, extra) in results {
            ops.push(op);
            if let Some(e) = extra {
                extras.push(e);
            }
        }
        (Circuit { n_qubits: self.n_qubits, ops }, extras)
    }

    /// Like `for_each_block_with`, but processes blocks one at a time on the
    /// current thread instead of in parallel via rayon. Needed for `f`
    /// closures that already parallelize internally (e.g. cyclosynth's own
    /// `rayon::scope`/`par_iter`-based lattice search): nesting many
    /// concurrent outer blocks, each *also* trying to use the whole rayon
    /// pool internally, oversubscribes it badly -- measured on
    /// `dnn_n8.qasm --cyclosynth`, individual calls that take ~2-3s with
    /// exclusive access to the pool ballooned to 55-111s when 20 of them
    /// ran at once via the parallel path, even after fixing the separate
    /// stack-overflow risk this same oversubscription caused (a properly-
    /// sized global pool stops the crash, but not this slowdown). Going
    /// sequential at this level lets each call have the whole pool to
    /// itself: Stage 6 dropped from 111.6s to 42.2s on the same circuit,
    /// with the slowest individual call dropping from 111.5s to 3.0s.
    pub fn for_each_block_with_sequential<T>(&self, f: impl Fn(&Circuit) -> (Circuit, T)) -> (Circuit, Vec<T>) {
        let mut ops = Vec::with_capacity(self.ops.len());
        let mut extras = Vec::new();
        for op in &self.ops {
            if let Gate::Block(inner) = &op.gate {
                let (new_inner, extra) = f(inner);
                ops.push(Operation { gate: Gate::Block(Box::new(new_inner)), qubits: op.qubits.clone() });
                extras.push(extra);
            } else {
                ops.push(op.clone());
            }
        }
        (Circuit { n_qubits: self.n_qubits, ops }, extras)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cliffordt::matrix::distance;

    #[test]
    fn t_squared_equals_s() {
        let mut c = Circuit::new(1);
        c.push(Gate::T, vec![0]);
        c.push(Gate::T, vec![0]);
        let mut s_circuit = Circuit::new(1);
        s_circuit.push(Gate::S, vec![0]);
        assert!(distance(&s_circuit.get_unitary(), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn h_squared_is_identity() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::H, vec![0]);
        assert!(distance(&identity(2), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn rz_pi_over_4_is_t_up_to_global_phase() {
        let mut c = Circuit::new(1);
        c.push(Gate::Rz(std::f64::consts::FRAC_PI_4), vec![0]);
        let mut t_circuit = Circuit::new(1);
        t_circuit.push(Gate::T, vec![0]);
        assert!(distance(&t_circuit.get_unitary(), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn cx_on_two_qubit_circuit_matches_matrix() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![0, 1]);
        let expected = Gate::Cx.matrix();
        assert!(distance(&expected, &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn cx_twice_is_identity() {
        let mut c = Circuit::new(2);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Cx, vec![0, 1]);
        assert!(distance(&identity(4), &c.get_unitary()) < 1e-12);
    }

    #[test]
    fn params_round_trip() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::X, vec![0]);
        c.push(Gate::Rz(1.2), vec![0]);
        assert_eq!(c.params(), vec![0.3, 1.2]);
        c.set_params(&[0.7, -0.4]);
        assert_eq!(c.params(), vec![0.7, -0.4]);
    }

    #[test]
    fn rz_unitary_derivatives_matches_finite_difference() {
        let mut c = Circuit::new(2);
        c.push(Gate::H, vec![0]);
        c.push(Gate::Rz(0.3), vec![0]);
        c.push(Gate::Cx, vec![0, 1]);
        c.push(Gate::Rz(-0.7), vec![1]);
        c.push(Gate::S, vec![1]);
        c.push(Gate::Rz(1.1), vec![0]);

        let (built, derivs) = c.unitary_and_rz_derivatives();
        assert!(distance(&built, &c.get_unitary()) < 1e-12);
        let params = c.params();
        assert_eq!(derivs.len(), params.len());

        const H: f64 = 1e-6;
        for (k, d_analytic) in derivs.iter().enumerate() {
            let mut p_plus = params.clone();
            p_plus[k] += H;
            let mut p_minus = params.clone();
            p_minus[k] -= H;
            let mut c_plus = c.clone();
            c_plus.set_params(&p_plus);
            let mut c_minus = c.clone();
            c_minus.set_params(&p_minus);
            let numeric = (c_plus.get_unitary() - c_minus.get_unitary()) / C64::new(2.0 * H, 0.0);
            let diff_norm = (d_analytic - &numeric).norm();
            assert!(diff_norm < 1e-6, "param {k}: analytic vs numeric derivative differ by {diff_norm}");
        }
    }

    #[test]
    fn block_gate_composes_into_parent_unitary() {
        let mut inner = Circuit::new(1);
        inner.push(Gate::H, vec![0]);
        let mut outer = Circuit::new(1);
        outer.push(Gate::Block(Box::new(inner)), vec![0]);
        let mut h_circuit = Circuit::new(1);
        h_circuit.push(Gate::H, vec![0]);
        assert!(distance(&h_circuit.get_unitary(), &outer.get_unitary()) < 1e-12);
    }

    #[test]
    fn is_all_clifford_detects_rz() {
        let mut c = Circuit::new(1);
        c.push(Gate::H, vec![0]);
        assert!(c.is_all_clifford());
        c.push(Gate::Rz(0.1), vec![0]);
        assert!(!c.is_all_clifford());
    }
}
