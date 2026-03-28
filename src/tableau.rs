#![allow(dead_code)]
//! Symplectic tableau for tracking Clifford conjugation of Pauli operators.
//!
//! This is a Rust reimplementation of the functionality provided by `stim.Tableau`
//! and `stim.PauliString` as used in `tableau/tableau/transpile.py`.
//!
//! # Representation
//!
//! An n-qubit tableau stores 2n rows (one X-row and one Z-row per qubit).
//! Each row is a Pauli string of length n, represented as two bit-vectors
//! (x_bits and z_bits) plus a sign bit.
//!
//! Row `2*q`   = image of X_q under the accumulated Clifford conjugation.
//! Row `2*q+1` = image of Z_q under the accumulated Clifford conjugation.
//!
//! # Pauli encoding
//!
//! | x_bit | z_bit | Pauli |
//! |-------|-------|-------|
//! |   0   |   0   |   I   |
//! |   1   |   0   |   X   |
//! |   1   |   1   |   Y   |
//! |   0   |   1   |   Z   |
//!
//! # Sign convention
//!
//! `sign = false` → +1,  `sign = true` → −1.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// PauliString
// ─────────────────────────────────────────────────────────────────────────────

/// A Pauli string on `n` qubits with a ±1 sign.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PauliString {
    pub n: usize,
    /// x_bits[q] = true  ↔  X component on qubit q
    pub x_bits: Vec<bool>,
    /// z_bits[q] = true  ↔  Z component on qubit q
    pub z_bits: Vec<bool>,
    /// sign: false = +1, true = −1
    pub sign: bool,
}

impl PauliString {
    /// Create an all-identity Pauli string of length `n` with sign +1.
    pub(crate) fn identity(n: usize) -> Self {
        PauliString { n, x_bits: vec![false; n], z_bits: vec![false; n], sign: false }
    }

    /// Create a Pauli string from a character string like `"+IXYZ"` or `"ZXI"`.
    ///
    /// The optional leading `+` or `-` sets the sign.  Each subsequent character
    /// must be one of `I`, `X`, `Y`, `Z`.
    pub(crate) fn from_str(s: &str) -> Self {
        let s = s.trim();
        let (sign, chars) = if s.starts_with('-') {
            (true, &s[1..])
        } else if s.starts_with('+') {
            (false, &s[1..])
        } else {
            (false, s)
        };
        let n = chars.len();
        let mut x_bits = vec![false; n];
        let mut z_bits = vec![false; n];
        for (i, c) in chars.chars().enumerate() {
            match c {
                'I' => {}
                'X' => x_bits[i] = true,
                'Y' => {
                    x_bits[i] = true;
                    z_bits[i] = true;
                }
                'Z' => z_bits[i] = true,
                _ => panic!("Invalid Pauli character '{}'", c),
            }
        }
        PauliString { n, x_bits, z_bits, sign }
    }

    /// Return the Pauli character at qubit `q`.
    pub(crate) fn pauli_at(&self, q: usize) -> char {
        match (self.x_bits[q], self.z_bits[q]) {
            (false, false) => 'I',
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        }
    }

    /// Number of non-identity Pauli operators (weight).
    pub(crate) fn weight(&self) -> usize {
        (0..self.n).filter(|&q| self.x_bits[q] || self.z_bits[q]).count()
    }

    /// Multiply two Pauli strings element-wise using a 4-phase accumulator.
    ///
    /// Phase is tracked mod 4 (0 = +1, 1 = +i, 2 = −1, 3 = −i).
    /// The result must have phase 0 or 2 (i.e. ±1) for the product to be
    /// a valid Hermitian Pauli string.
    ///
    /// Single-qubit Pauli multiplication phases (from standard table):
    ///   X·Y = +iZ  → phase +1
    ///   Y·Z = +iX  → phase +1
    ///   Z·X = +iY  → phase +1
    ///   Y·X = -iZ  → phase +3 (= -i)
    ///   Z·Y = -iX  → phase +3
    ///   X·Z = -iY  → phase +3
    ///   All others (including I·anything, same·same) → phase 0
    pub(crate) fn mul(&self, other: &PauliString) -> PauliString {
        let n = self.n.max(other.n);
        // Start with the signs of both operands (each sign=true contributes phase 2)
        let mut phase: i32 = 0;
        if self.sign {
            phase += 2;
        }
        if other.sign {
            phase += 2;
        }

        let mut x_bits = vec![false; n];
        let mut z_bits = vec![false; n];

        for q in 0..n {
            let ax = if q < self.n { self.x_bits[q] } else { false };
            let az = if q < self.n { self.z_bits[q] } else { false };
            let bx = if q < other.n { other.x_bits[q] } else { false };
            let bz = if q < other.n { other.z_bits[q] } else { false };

            x_bits[q] = ax ^ bx;
            z_bits[q] = az ^ bz;

            // Phase contribution from single-qubit Pauli product (a)·(b):
            // Encode: I=0, X=1, Y=3, Z=2  (this is the standard symplectic encoding)
            // The phase table for non-trivial products:
            //   X·Y = iZ  (+1),  Y·X = -iZ (+3)
            //   Y·Z = iX  (+1),  Z·Y = -iX (+3)
            //   Z·X = iY  (+1),  X·Z = -iY (+3)
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        phase = phase.rem_euclid(4);
        let sign = match phase {
            0 => false,
            2 => true,
            p => panic!(
                "Non-Hermitian phase {} in Pauli product (self={}, other={})",
                p, self, other
            ),
        };
        PauliString { n, x_bits, z_bits, sign }
    }

    /// Multiply `self` by `(i · x_img · z_img)` where the factor of `i` comes
    /// from the identity Y = i·X·Z.
    ///
    /// When conjugating a Y operator through a tableau:
    ///   U·Y_q·U† = U·(i·X_q·Z_q)·U† = i·(U·X_q·U†)·(U·Z_q·U†)
    ///
    /// This function computes `self · i · x_img · z_img` using a full 4-phase
    /// accumulator, then asserts the result is ±1 (phase 0 or 2 mod 4).
    pub(crate) fn mul_with_y_phase(&self, x_img: &PauliString, z_img: &PauliString) -> PauliString {
        let n = self.n.max(x_img.n).max(z_img.n);
        // Accumulate phase from all three factors: self, i (=+1 phase unit), x_img, z_img
        let mut phase: i32 = 0;
        if self.sign {
            phase += 2;
        }
        // The extra factor of i from Y = i·X·Z
        phase += 1;
        if x_img.sign {
            phase += 2;
        }
        if z_img.sign {
            phase += 2;
        }

        // Compute x_img · z_img Pauli part and accumulate phase
        let mut xz_x = vec![false; n];
        let mut xz_z = vec![false; n];
        for q in 0..n {
            let ax = if q < x_img.n { x_img.x_bits[q] } else { false };
            let az = if q < x_img.n { x_img.z_bits[q] } else { false };
            let bx = if q < z_img.n { z_img.x_bits[q] } else { false };
            let bz = if q < z_img.n { z_img.z_bits[q] } else { false };
            xz_x[q] = ax ^ bx;
            xz_z[q] = az ^ bz;
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        // Now compute self · (x_img · z_img) Pauli part and accumulate phase
        let mut result_x = vec![false; n];
        let mut result_z = vec![false; n];
        for q in 0..n {
            let ax = if q < self.n { self.x_bits[q] } else { false };
            let az = if q < self.n { self.z_bits[q] } else { false };
            let bx = xz_x[q];
            let bz = xz_z[q];
            result_x[q] = ax ^ bx;
            result_z[q] = az ^ bz;
            phase += single_qubit_mul_phase(ax, az, bx, bz);
        }

        phase = phase.rem_euclid(4);
        let sign = match phase {
            0 => false,
            2 => true,
            p => panic!(
                "Non-Hermitian phase {} when conjugating Y (x_img={}, z_img={})",
                p, x_img, z_img
            ),
        };
        PauliString { n, x_bits: result_x, z_bits: result_z, sign }
    }
}

/// Returns the phase contribution (0, 1, 2, or 3 mod 4) from multiplying
/// single-qubit Paulis encoded as (x_bit, z_bit):
///   I=(0,0), X=(1,0), Y=(1,1), Z=(0,1)
///
/// Non-zero contributions:
///   X·Y = +iZ  → +1
///   Y·Z = +iX  → +1
///   Z·X = +iY  → +1
///   Y·X = -iZ  → +3
///   Z·Y = -iX  → +3
///   X·Z = -iY  → +3
fn single_qubit_mul_phase(ax: bool, az: bool, bx: bool, bz: bool) -> i32 {
    // Only non-identity pairs contribute phase.
    // We use the cyclic rule: for the "forward" cycle X→Y→Z→X, the product
    // of consecutive elements gives +i times the next; reversed gives -i.
    match (ax, az, bx, bz) {
        // X·Y = iZ
        (true, false, true, true) => 1,
        // Y·Z = iX
        (true, true, false, true) => 1,
        // Z·X = iY
        (false, true, true, false) => 1,
        // Y·X = -iZ
        (true, true, true, false) => 3,
        // Z·Y = -iX
        (false, true, true, true) => 3,
        // X·Z = -iY
        (true, false, false, true) => 3,
        // All other cases (I·anything, same·same, etc.) contribute 0
        _ => 0,
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.sign { "-" } else { "+" })?;
        for q in 0..self.n {
            write!(f, "{}", self.pauli_at(q))?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tableau
// ─────────────────────────────────────────────────────────────────────────────

/// Symplectic Clifford tableau on `n` qubits.
///
/// Stores the images of all X_q and Z_q generators under the accumulated
/// Clifford unitary.  Applying a gate `prepend`s it (i.e. the gate is applied
/// *before* the existing tableau, matching stim's `tableau.prepend(gate, qubits)`).
#[derive(Debug, Clone)]
pub(crate) struct Tableau {
    pub n: usize,
    /// rows[2*q]   = image of X_q
    /// rows[2*q+1] = image of Z_q
    rows: Vec<PauliString>,
}

impl Tableau {
    /// Create the identity tableau on `n` qubits.
    pub(crate) fn new(n: usize) -> Self {
        let mut rows = Vec::with_capacity(2 * n);
        for q in 0..n {
            // X_q image: X on qubit q
            let mut xrow = PauliString::identity(n);
            xrow.x_bits[q] = true;
            rows.push(xrow);
            // Z_q image: Z on qubit q
            let mut zrow = PauliString::identity(n);
            zrow.z_bits[q] = true;
            rows.push(zrow);
        }
        Tableau { n, rows }
    }

    /// Return the number of qubits.
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Conjugate a Pauli string through this tableau.
    ///
    /// For each qubit q where the input has a non-identity Pauli:
    ///   X_q → rows[2*q]
    ///   Z_q → rows[2*q+1]
    ///   Y_q = i·X_q·Z_q → i·rows[2*q]·rows[2*q+1]
    ///
    /// The results are multiplied together (with phase tracking) to give the
    /// output Pauli string.
    pub(crate) fn conjugate(&self, pauli: &PauliString) -> PauliString {
        let n = self.n.max(pauli.n);
        let mut result = PauliString::identity(n);
        if pauli.sign {
            result.sign = true;
        }
        for q in 0..pauli.n {
            let (px, pz) = (pauli.x_bits[q], pauli.z_bits[q]);
            if !px && !pz {
                continue; // identity on this qubit
            }
            let x_image = self.extended_row(2 * q, n);
            let z_image = self.extended_row(2 * q + 1, n);
            if px && pz {
                // Y_q = i·X_q·Z_q
                result = result.mul_with_y_phase(&x_image, &z_image);
            } else if px {
                // X_q
                result = result.mul(&x_image);
            } else {
                // Z_q
                result = result.mul(&z_image);
            }
        }
        result
    }

    /// Get row `r` extended to length `n` (padding with identity).
    fn extended_row(&self, r: usize, n: usize) -> PauliString {
        let row = &self.rows[r];
        if row.n == n {
            return row.clone();
        }
        let mut x_bits = row.x_bits.clone();
        let mut z_bits = row.z_bits.clone();
        x_bits.resize(n, false);
        z_bits.resize(n, false);
        PauliString { n, x_bits, z_bits, sign: row.sign }
    }

    /// Prepend a single-qubit gate to this tableau on qubit `q`.
    ///
    /// "Prepend" means the gate is applied *before* the existing tableau,
    /// matching stim's `tableau.prepend(gate, [q])`.
    ///
    /// Updates rows[2*q] and rows[2*q+1] using the gate's Clifford conjugation
    /// rules for X_q and Z_q.
    pub(crate) fn prepend_1q_correct(&mut self, gate: Gate1Q, q: usize) {
        let old_x = self.rows[2 * q].clone();
        let old_z = self.rows[2 * q + 1].clone();

        let (gx_sign, gx_x, gx_z) = gate.x_image_correct();
        let (gz_sign, gz_x, gz_z) = gate.z_image_correct();

        self.rows[2 * q] = combine_rows(&old_x, gx_x, &old_z, gx_z, gx_sign);
        self.rows[2 * q + 1] = combine_rows(&old_x, gz_x, &old_z, gz_z, gz_sign);
    }

    /// Prepend a two-qubit gate to this tableau on qubits `q0`, `q1`.
    pub(crate) fn prepend_2q(&mut self, gate: Gate2Q, q0: usize, q1: usize) {
        let old_x0 = self.rows[2 * q0].clone();
        let old_z0 = self.rows[2 * q0 + 1].clone();
        let old_x1 = self.rows[2 * q1].clone();
        let old_z1 = self.rows[2 * q1 + 1].clone();

        let (new_x0, new_z0, new_x1, new_z1) = gate.apply(&old_x0, &old_z0, &old_x1, &old_z1);

        self.rows[2 * q0] = new_x0;
        self.rows[2 * q0 + 1] = new_z0;
        self.rows[2 * q1] = new_x1;
        self.rows[2 * q1 + 1] = new_z1;
    }
}

/// Combine rows: compute the image of a generator after prepending a gate.
///
/// The gate maps a generator (X or Z) to `sign * P` where P is one of:
///   - I  (use_x=false, use_z=false)
///   - X  (use_x=true,  use_z=false)  → old_x
///   - Z  (use_x=false, use_z=true)   → old_z
///   - Y  (use_x=true,  use_z=true)   → i * old_x * old_z  (Y = i·X·Z)
///
/// When the image is Y (both use_x and use_z), we must use `mul_with_y_phase`
/// to correctly account for the i factor in Y = i·X·Z.
fn combine_rows(
    row_x: &PauliString, use_x: bool, row_z: &PauliString, use_z: bool, sign: bool,
) -> PauliString {
    let n = row_x.n;
    let mut result = PauliString::identity(n);
    result.sign = sign;
    if use_x && use_z {
        // Image is Y = i·X·Z: use mul_with_y_phase to handle the i factor
        result = result.mul_with_y_phase(row_x, row_z);
    } else if use_x {
        result = result.mul(row_x);
    } else if use_z {
        result = result.mul(row_z);
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-qubit gates
// ─────────────────────────────────────────────────────────────────────────────

/// Single-qubit Clifford gates supported by the transpiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Gate1Q {
    H,
    S,
    Sdg,
    SX,
    SXdg,
    X,
    Y,
    Z,
}

impl Gate1Q {
    /// Returns `(sign_negative, x_bit, z_bit)` for the image of X under this gate.
    ///
    /// Standard Clifford conjugation rules:
    ///   H:    X → Z
    ///   S:    X → Y   (= +Y, x=1,z=1, sign=false)
    ///   Sdg:  X → -Y  (= -Y, x=1,z=1, sign=true)
    ///   SX:   X → X
    ///   SXdg: X → X
    ///   X:    X → X
    ///   Y:    X → -X
    ///   Z:    X → -X
    pub(crate) fn x_image_correct(&self) -> (bool, bool, bool) {
        // (sign_negative, x_bit, z_bit)
        match self {
            Gate1Q::H => (false, false, true),    // X → Z
            Gate1Q::S => (false, true, true),     // X → Y
            Gate1Q::Sdg => (true, true, true),    // X → -Y
            Gate1Q::SX => (false, true, false),   // X → X
            Gate1Q::SXdg => (false, true, false), // X → X
            Gate1Q::X => (false, true, false),    // X → X
            Gate1Q::Y => (true, true, false),     // X → -X
            Gate1Q::Z => (true, true, false),     // X → -X
        }
    }

    /// Returns `(sign_negative, x_bit, z_bit)` for the image of Z under this gate.
    ///
    /// Standard Clifford conjugation rules:
    ///   H:    Z → X
    ///   S:    Z → Z
    ///   Sdg:  Z → Z
    ///   SX:   Z → -Y  (x=1,z=1, sign=true)
    ///   SXdg: Z → Y   (x=1,z=1, sign=false)
    ///   X:    Z → -Z
    ///   Y:    Z → -Z
    ///   Z:    Z → Z
    pub(crate) fn z_image_correct(&self) -> (bool, bool, bool) {
        match self {
            Gate1Q::H => (false, true, false),   // Z → X
            Gate1Q::S => (false, false, true),   // Z → Z
            Gate1Q::Sdg => (false, false, true), // Z → Z
            Gate1Q::SX => (true, true, true),    // Z → -Y
            Gate1Q::SXdg => (false, true, true), // Z → Y
            Gate1Q::X => (true, false, true),    // Z → -Z
            Gate1Q::Y => (true, false, true),    // Z → -Z
            Gate1Q::Z => (false, false, true),   // Z → Z
        }
    }
}

/// Two-qubit Clifford gates supported by the transpiler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Gate2Q {
    CX, // CNOT: control=q0, target=q1
    CZ,
    Swap,
}

impl Gate2Q {
    /// Compute the new images of X0, Z0, X1, Z1 after prepending this gate.
    ///
    /// CX (CNOT, control=q0, target=q1):
    ///   X0 → X0·X1,  Z0 → Z0,  X1 → X1,  Z1 → Z0·Z1
    ///
    /// CZ:
    ///   X0 → X0·Z1,  Z0 → Z0,  X1 → Z0·X1,  Z1 → Z1
    ///
    /// SWAP:
    ///   X0 → X1,  Z0 → Z1,  X1 → X0,  Z1 → Z0
    fn apply(
        &self, x0: &PauliString, z0: &PauliString, x1: &PauliString, z1: &PauliString,
    ) -> (PauliString, PauliString, PauliString, PauliString) {
        match self {
            Gate2Q::CX => (x0.mul(x1), z0.clone(), x1.clone(), z0.mul(z1)),
            Gate2Q::CZ => (x0.mul(z1), z0.clone(), z0.mul(x1), z1.clone()),
            Gate2Q::Swap => (x1.clone(), z1.clone(), x0.clone(), z0.clone()),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PauliString::from_str ─────────────────────────────────────────────────

    #[test]
    fn pauli_string_from_str_identity() {
        let p = PauliString::from_str("III");
        assert_eq!(p.n, 3);
        assert!(!p.sign);
        assert_eq!(p.weight(), 0);
    }

    #[test]
    fn pauli_string_from_str_xyz() {
        let p = PauliString::from_str("+XYZ");
        assert_eq!(p.n, 3);
        assert!(!p.sign);
        assert_eq!(p.pauli_at(0), 'X');
        assert_eq!(p.pauli_at(1), 'Y');
        assert_eq!(p.pauli_at(2), 'Z');
    }

    #[test]
    fn pauli_string_from_str_negative() {
        let p = PauliString::from_str("-XZ");
        assert!(p.sign);
        assert_eq!(p.pauli_at(0), 'X');
        assert_eq!(p.pauli_at(1), 'Z');
    }

    #[test]
    fn pauli_string_weight() {
        let p = PauliString::from_str("+XIZIY");
        assert_eq!(p.weight(), 3);
    }

    // ── PauliString::mul ──────────────────────────────────────────────────────

    #[test]
    fn pauli_mul_xx_is_identity() {
        let x = PauliString::from_str("+X");
        let result = x.mul(&x);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_zz_is_identity() {
        let z = PauliString::from_str("+Z");
        let result = z.mul(&z);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_yy_is_identity() {
        let y = PauliString::from_str("+Y");
        let result = y.mul(&y);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_zx_gives_y_with_phase() {
        // (XY) * (YX): qubit 0: X*Y = iZ, qubit 1: Y*X = -iZ
        // total phase: i * (-i) = 1 (+1), result = ZZ
        let xy = PauliString::from_str("+XY");
        let yx = PauliString::from_str("+YX");
        let result = xy.mul(&yx);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn pauli_mul_negative_signs() {
        // (-X) * (+X) = -I
        let neg_x = PauliString::from_str("-X");
        let pos_x = PauliString::from_str("+X");
        let result = neg_x.mul(&pos_x);
        assert_eq!(result.pauli_at(0), 'I');
        assert!(result.sign); // -I
    }

    // ── Tableau identity ──────────────────────────────────────────────────────

    #[test]
    fn tableau_identity_conjugates_x_trivially() {
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XII");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'I');
        assert_eq!(result.pauli_at(2), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_z_trivially() {
        let t = Tableau::new(2);
        let p = PauliString::from_str("+IZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_y_correctly() {
        // Identity tableau: Y → Y
        let t = Tableau::new(1);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign);
    }

    #[test]
    fn tableau_identity_conjugates_xyz_trivially() {
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XYZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Y');
        assert_eq!(result.pauli_at(2), 'Z');
        assert!(!result.sign);
    }

    // ── H gate ────────────────────────────────────────────────────────────────

    #[test]
    fn h_swaps_x_and_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        // H X H† = Z
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
        // H Z H† = X
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn h_twice_is_identity() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn h_maps_y_to_minus_y() {
        // H Y H† = -Y
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign); // -Y
    }

    // ── S gate ────────────────────────────────────────────────────────────────

    #[test]
    fn s_maps_x_to_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign);
    }

    #[test]
    fn s_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn s_maps_y_to_minus_x() {
        // S Y S† = -X  (since S X S† = Y and S Z S† = Z, so S Y S† = S(iXZ)S† = i·Y·Z = -X)
        // Actually: Y = iXZ, S Y S† = i(SXS†)(SZS†) = i·Y·Z = i·(iXZ)·Z = i·iX·Z·Z = -X
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        let y = PauliString::from_str("+Y");
        let result = t.conjugate(&y);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(result.sign); // -X
    }

    // ── Sdg gate ──────────────────────────────────────────────────────────────

    #[test]
    fn sdg_maps_x_to_minus_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign); // -Y
    }

    #[test]
    fn sdg_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── S then Sdg = identity ─────────────────────────────────────────────────

    #[test]
    fn s_sdg_is_identity() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::S, 0);
        t.prepend_1q_correct(Gate1Q::Sdg, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── SX gate ───────────────────────────────────────────────────────────────

    #[test]
    fn sx_maps_x_to_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SX, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn sx_maps_z_to_minus_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SX, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign); // -Y
    }

    #[test]
    fn sxdg_maps_z_to_y() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::SXdg, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(!result.sign); // +Y
    }

    // ── Z gate ────────────────────────────────────────────────────────────────

    #[test]
    fn z_maps_x_to_minus_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Z, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(result.sign); // -X
    }

    #[test]
    fn z_maps_z_to_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::Z, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── X gate ────────────────────────────────────────────────────────────────

    #[test]
    fn x_maps_z_to_minus_z() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::X, 0);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(result.sign); // -Z
    }

    #[test]
    fn x_maps_x_to_x() {
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::X, 0);
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
    }

    // ── CX gate ───────────────────────────────────────────────────────────────

    #[test]
    fn cx_maps_x0_to_x0x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_z1_to_z0z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let z1 = PauliString::from_str("+IZ");
        let result = t.conjugate(&z1);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_z0_to_z0() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let z0 = PauliString::from_str("+ZI");
        let result = t.conjugate(&z0);
        assert_eq!(result.pauli_at(0), 'Z');
        assert_eq!(result.pauli_at(1), 'I');
        assert!(!result.sign);
    }

    #[test]
    fn cx_maps_x1_to_x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CX, 0, 1);
        let x1 = PauliString::from_str("+IX");
        let result = t.conjugate(&x1);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    // ── CZ gate ───────────────────────────────────────────────────────────────

    #[test]
    fn cz_maps_x0_to_x0z1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::CZ, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Z');
        assert!(!result.sign);
    }

    // ── SWAP gate ─────────────────────────────────────────────────────────────

    #[test]
    fn swap_maps_x0_to_x1() {
        let mut t = Tableau::new(2);
        t.prepend_2q(Gate2Q::Swap, 0, 1);
        let x0 = PauliString::from_str("+XI");
        let result = t.conjugate(&x0);
        assert_eq!(result.pauli_at(0), 'I');
        assert_eq!(result.pauli_at(1), 'X');
        assert!(!result.sign);
    }

    // ── Multi-qubit Y conjugation ─────────────────────────────────────────────

    #[test]
    fn identity_conjugates_multi_qubit_y() {
        // Identity tableau on 3 qubits: XYZ → XYZ
        let t = Tableau::new(3);
        let p = PauliString::from_str("+XYZ");
        let result = t.conjugate(&p);
        assert_eq!(result.pauli_at(0), 'X');
        assert_eq!(result.pauli_at(1), 'Y');
        assert_eq!(result.pauli_at(2), 'Z');
        assert!(!result.sign);
    }

    // ── S^4 = identity ────────────────────────────────────────────────────────

    #[test]
    fn s_four_times_is_identity() {
        let mut t = Tableau::new(1);
        for _ in 0..4 {
            t.prepend_1q_correct(Gate1Q::S, 0);
        }
        let x = PauliString::from_str("+X");
        let result = t.conjugate(&x);
        assert_eq!(result.pauli_at(0), 'X');
        assert!(!result.sign);
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        assert_eq!(result.pauli_at(0), 'Z');
        assert!(!result.sign);
    }

    // ── H·S·H = Sdg (up to global phase) ─────────────────────────────────────

    #[test]
    fn h_s_h_maps_z_to_x() {
        // H·S·H: Z → H(S(H(Z))) = H(S(X)) = H(Y) = -Y... let's just verify
        // the tableau gives consistent results.
        let mut t = Tableau::new(1);
        t.prepend_1q_correct(Gate1Q::H, 0);
        t.prepend_1q_correct(Gate1Q::S, 0);
        t.prepend_1q_correct(Gate1Q::H, 0);
        // H·S·H = SX (up to global phase)
        // SX: X→X, Z→-Y
        let z = PauliString::from_str("+Z");
        let result = t.conjugate(&z);
        // H·S·H·Z·H·S†·H = SX·Z·SX† = -Y
        assert_eq!(result.pauli_at(0), 'Y');
        assert!(result.sign); // -Y
    }
}
