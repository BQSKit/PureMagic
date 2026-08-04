//! Complex unitary matrices and the distance metric used throughout the
//! Clifford+T pipeline: a global-phase-aligned spectral-norm distance.
//!
//! Mirrors `global_phase_between`/`spectral_error` from
//! `data_processing/compile_cliffordt.py`.

use nalgebra::{Complex, DMatrix};

pub type C64 = Complex<f64>;
pub type Unitary = DMatrix<C64>;

pub fn identity(dim: usize) -> Unitary {
    Unitary::identity(dim, dim)
}

/// Kronecker (tensor) product of two square matrices. Only used by this
/// module's own tests, as an independent way to build the expected result
/// for `embed`/`apply_gate_inplace` cases -- not part of the pipeline
/// itself.
#[cfg(test)]
fn kron(a: &Unitary, b: &Unitary) -> Unitary {
    a.kronecker(b)
}

/// Phase gamma minimizing ||target - exp(i*gamma) * built||, i.e.
/// arg(tr(built^dagger @ target)).
pub fn global_phase_between(target: &Unitary, built: &Unitary) -> f64 {
    let prod = built.adjoint() * target;
    prod.trace().arg()
}

/// ||exp(i*phase) * built - target||_2 (spectral / operator norm).
pub fn spectral_error(target: &Unitary, built: &Unitary, phase: f64) -> f64 {
    let rot = C64::new(phase.cos(), phase.sin());
    let diff = built.map(|v| v * rot) - target;
    spectral_norm(&diff)
}

/// Largest singular value of `m`, via the largest eigenvalue of m^dagger m
/// (avoids pulling in a full complex SVD for what's always a small matrix).
pub fn spectral_norm(m: &Unitary) -> f64 {
    let gram = m.adjoint() * m;
    let gram_re = gram.map(|v| v.re);
    let eigs = gram_re.symmetric_eigenvalues();
    eigs.iter().cloned().fold(0.0_f64, f64::max).max(0.0).sqrt()
}

/// Global-phase-aligned spectral distance between `built` and `target`:
/// the error that will actually be reported once `built`'s global phase is
/// set to its optimal value.
pub fn distance(target: &Unitary, built: &Unitary) -> f64 {
    let phase = global_phase_between(target, built);
    spectral_error(target, built, phase)
}

/// Bit of qubit `q` (0 = most significant) within an `n`-qubit basis index.
/// Only used by `embed` (test-only, see its own doc comment).
#[cfg(test)]
fn bit(x: usize, q: usize, n: usize) -> usize {
    (x >> (n - 1 - q)) & 1
}

/// Sub-index formed by reading the bits at `qubits` (in the given order,
/// first entry = most significant) out of a full `n`-qubit basis index.
/// Only used by `embed` (test-only, see its own doc comment).
#[cfg(test)]
fn local_index(x: usize, qubits: &[usize], n: usize) -> usize {
    let mut idx = 0;
    for &q in qubits {
        idx = (idx << 1) | bit(x, q, n);
    }
    idx
}

/// Left-multiply `u` in place by `gate` embedded on `qubits`, i.e.
/// `u := embed(gate, qubits, n_qubits) * u`, without ever forming the full
/// (mostly-zero) embedded matrix.
///
/// `embed` followed by a dense multiply costs O(dim^3) per gate regardless
/// of how few qubits the gate actually touches -- fine for the small
/// per-block unitaries the compilation pipeline itself works with, but
/// prohibitive for `Circuit::get_unitary` on a whole circuit with many
/// thousands of gates (which is exactly what `--verify` needs to do).
/// This applies `gate` directly to the O(dim / 2^k) disjoint `2^k`-sized
/// slices of each of `u`'s `dim` columns that it actually touches, costing
/// O(dim^2 * 2^k) total -- for a 1- or 2-qubit gate, a large constant-factor
/// speedup over the dense path, growing with circuit size regardless of how
/// many gates are applied (each gate is still one full pass over `u`, but a
/// much cheaper one).
pub fn apply_gate_inplace(u: &mut Unitary, gate: &Unitary, qubits: &[usize], n_qubits: usize) {
    let k = qubits.len();
    let dim = u.nrows();
    let local_dim = 1usize << k;
    let outer_qubits: Vec<usize> = (0..n_qubits).filter(|q| !qubits.contains(q)).collect();
    let n_outer = outer_qubits.len();

    let mut local_vals = vec![C64::new(0.0, 0.0); local_dim];
    let mut row_indices = vec![0usize; local_dim];

    for col in 0..dim {
        for outer_idx in 0..(1usize << n_outer) {
            let mut base = 0usize;
            for (i, &q) in outer_qubits.iter().enumerate() {
                let b = (outer_idx >> (n_outer - 1 - i)) & 1;
                base |= b << (n_qubits - 1 - q);
            }

            for local_idx in 0..local_dim {
                let mut row = base;
                for (i, &q) in qubits.iter().enumerate() {
                    let b = (local_idx >> (k - 1 - i)) & 1;
                    row |= b << (n_qubits - 1 - q);
                }
                row_indices[local_idx] = row;
                local_vals[local_idx] = u[(row, col)];
            }

            for out_idx in 0..local_dim {
                let mut sum = C64::new(0.0, 0.0);
                for (in_idx, &val) in local_vals.iter().enumerate() {
                    sum += gate[(out_idx, in_idx)] * val;
                }
                u[(row_indices[out_idx], col)] = sum;
            }
        }
    }
}

/// Embed a k-qubit gate matrix -- acting on `qubits` (in the given order,
/// `qubits[0]` is the gate matrix's own qubit 0) -- into an `n_qubits`-qubit
/// space. Entries where the full basis states disagree on any qubit outside
/// `qubits` are zero (the gate acts as identity there).
///
/// Superseded by `apply_gate_inplace` for actual circuit composition (see
/// its doc comment for why); kept as the simple, obviously-correct
/// reference implementation `apply_gate_inplace`'s own tests check against,
/// so test-only rather than removed.
#[cfg(test)]
fn embed(gate: &Unitary, qubits: &[usize], n_qubits: usize) -> Unitary {
    let k = qubits.len();
    debug_assert_eq!(gate.nrows(), 1usize << k);
    debug_assert_eq!(gate.ncols(), 1usize << k);
    let dim = 1usize << n_qubits;
    let mut out = Unitary::zeros(dim, dim);
    for row in 0..dim {
        for col in 0..dim {
            let mut differs = false;
            for q in 0..n_qubits {
                if !qubits.contains(&q) && bit(row, q, n_qubits) != bit(col, q, n_qubits) {
                    differs = true;
                    break;
                }
            }
            if differs {
                continue;
            }
            let gr = local_index(row, qubits, n_qubits);
            let gc = local_index(col, qubits, n_qubits);
            out[(row, col)] = gate[(gr, gc)];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h() -> Unitary {
        let s = C64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        Unitary::from_row_slice(2, 2, &[s, s, s, -s])
    }

    #[test]
    fn identity_has_zero_distance_from_itself() {
        let id = identity(2);
        assert!(distance(&id, &id) < 1e-12);
    }

    #[test]
    fn h_squared_is_identity() {
        let h = h();
        let h2 = &h * &h;
        assert!(distance(&identity(2), &h2) < 1e-12);
    }

    #[test]
    fn distinct_unitaries_have_nonzero_distance() {
        let id = identity(2);
        let h = h();
        assert!(distance(&id, &h) > 0.5);
    }

    #[test]
    fn global_phase_alignment_ignores_pure_phase_difference() {
        let id = identity(2);
        let phased = id.map(|v| v * C64::new(0.0, 1.0)); // i * I
        assert!(distance(&id, &phased) < 1e-12);
    }

    #[test]
    fn kron_of_identities_is_identity() {
        let id2 = identity(2);
        let id4 = kron(&id2, &id2);
        assert_eq!(id4.nrows(), 4);
        assert!(distance(&identity(4), &id4) < 1e-12);
    }

    fn x() -> Unitary {
        Unitary::from_row_slice(
            2,
            2,
            &[C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        )
    }

    #[test]
    fn embed_on_lsb_qubit_matches_kron_identity_then_gate() {
        let id2 = identity(2);
        let expected = kron(&id2, &x());
        let embedded = embed(&x(), &[1], 2);
        assert!(distance(&expected, &embedded) < 1e-12);
    }

    #[test]
    fn embed_on_msb_qubit_matches_kron_gate_then_identity() {
        let id2 = identity(2);
        let expected = kron(&x(), &id2);
        let embedded = embed(&x(), &[0], 2);
        assert!(distance(&expected, &embedded) < 1e-12);
    }

    #[test]
    fn embed_full_qubit_list_in_order_is_identity_mapping() {
        let cx = Unitary::from_row_slice(
            4,
            4,
            &[
                C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0),
            ],
        );
        let embedded = embed(&cx, &[0, 1], 2);
        assert!(distance(&cx, &embedded) < 1e-12);
    }

    #[test]
    fn embed_reversed_qubit_order_swaps_control_and_target() {
        // Relabeling a 2-qubit gate's own qubit order (matrix-qubit-0 <->
        // matrix-qubit-1) is equivalent to conjugating the gate by SWAP.
        let cx = Unitary::from_row_slice(
            4,
            4,
            &[
                C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0),
            ],
        );
        let swap = Unitary::from_row_slice(
            4,
            4,
            &[
                C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0),
                C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0),
            ],
        );
        let embedded = embed(&cx, &[1, 0], 2);
        let expected = &swap * &cx * &swap;
        assert!(distance(&expected, &embedded) < 1e-12);
    }

    fn cx4() -> Unitary {
        let o = C64::new(1.0, 0.0);
        let z = C64::new(0.0, 0.0);
        #[rustfmt::skip]
        let m = Unitary::from_row_slice(4, 4, &[
            o, z, z, z,
            z, o, z, z,
            z, z, z, o,
            z, z, o, z,
        ]);
        m
    }

    #[test]
    fn apply_gate_inplace_matches_embed_then_multiply_single_qubit() {
        let mut u = identity(8);
        apply_gate_inplace(&mut u, &h(), &[1], 3);
        let expected = embed(&h(), &[1], 3) * identity(8);
        assert!(distance(&expected, &u) < 1e-12);
    }

    #[test]
    fn apply_gate_inplace_matches_embed_then_multiply_two_qubit_noncontiguous() {
        let mut u = identity(8);
        apply_gate_inplace(&mut u, &cx4(), &[2, 0], 3);
        let expected = embed(&cx4(), &[2, 0], 3) * identity(8);
        assert!(distance(&expected, &u) < 1e-12);
    }

    #[test]
    fn apply_gate_inplace_composes_correctly_over_multiple_gates() {
        // Build the same 3-qubit circuit (H on q1, then CX(q2,q0), then H on
        // q0) two ways: via the fast in-place path, and via the slow
        // embed+multiply path, and check they agree.
        let mut fast = identity(8);
        apply_gate_inplace(&mut fast, &h(), &[1], 3);
        apply_gate_inplace(&mut fast, &cx4(), &[2, 0], 3);
        apply_gate_inplace(&mut fast, &h(), &[0], 3);

        let mut slow = identity(8);
        slow = embed(&h(), &[1], 3) * slow;
        slow = embed(&cx4(), &[2, 0], 3) * slow;
        slow = embed(&h(), &[0], 3) * slow;

        assert!(distance(&slow, &fast) < 1e-10);
    }

    #[test]
    fn apply_gate_inplace_on_non_identity_starting_point() {
        let mut fast = h().kronecker(&identity(2)).kronecker(&identity(2));
        let mut slow = fast.clone();
        apply_gate_inplace(&mut fast, &cx4(), &[0, 2], 3);
        slow = embed(&cx4(), &[0, 2], 3) * slow;
        assert!(distance(&slow, &fast) < 1e-10);
    }
}
