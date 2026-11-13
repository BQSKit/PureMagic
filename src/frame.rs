//! Clifford frame for `--t-correction frame`: failed T corrections are never applied to the data,
//! they are tracked exactly and folded into every later operation.
//!
//! The frame W relates the physical state to the ideal one: actual = W · ideal. To apply an
//! ideal rotation or measurement about Q, the hardware applies it about W·Q·W†.
//!
//! - A T about P that fails applies R_P(−π/8) = R_P(π/8)·R_P(−π/4), so W ← W·R_P(−π/4).
//! - A physical Clifford G (S, SX, CX lines in the `.trans`) runs unchanged, so W ← G·W·G†.
//!
//! Only which qubits and letters W·Q·W† has matter for scheduling, and those do not depend on
//! signs, so the frame is a symplectic (GF(2)) tableau without phases. Row `2q` is the image of
//! X_q, row `2q+1` the image of Z_q. Paulis are bitsets of 64-qubit words.

use crate::pauliproduct::Operator;

/// A Pauli operator without phase, as x and z bitsets: I = (0,0), X = (1,0), Y = (1,1), Z = (0,1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SPauli {
    pub(crate) x: Vec<u64>,
    pub(crate) z: Vec<u64>,
}

impl SPauli {
    pub(crate) fn identity(words: usize) -> Self {
        SPauli { x: vec![0; words], z: vec![0; words] }
    }

    pub(crate) fn from_ops(words: usize, ops: &[Operator]) -> Self {
        let mut p = SPauli::identity(words);
        for op in ops {
            p.set(op.qubit as usize, op.basis);
        }
        p
    }

    pub(crate) fn set(&mut self, q: usize, letter: char) {
        let (w, b) = (q / 64, 1u64 << (q % 64));
        let (x, z) = match letter {
            'X' => (true, false),
            'Y' => (true, true),
            'Z' => (false, true),
            _ => (false, false),
        };
        if x {
            self.x[w] |= b;
        } else {
            self.x[w] &= !b;
        }
        if z {
            self.z[w] |= b;
        } else {
            self.z[w] &= !b;
        }
    }

    #[inline]
    pub(crate) fn bits(&self, q: usize) -> (bool, bool) {
        let (w, s) = (q / 64, q % 64);
        ((self.x[w] >> s) & 1 == 1, (self.z[w] >> s) & 1 == 1)
    }

    pub(crate) fn letter(&self, q: usize) -> char {
        match self.bits(q) {
            (false, false) => '_',
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        }
    }

    /// Multiplies by `other`, ignoring phase.
    #[inline]
    pub(crate) fn mul_assign(&mut self, other: &SPauli) {
        for (a, b) in self.x.iter_mut().zip(&other.x) {
            *a ^= b;
        }
        for (a, b) in self.z.iter_mut().zip(&other.z) {
            *a ^= b;
        }
    }

    /// True if the two Paulis anticommute (odd symplectic product).
    #[inline]
    pub(crate) fn anticommutes(&self, other: &SPauli) -> bool {
        let mut parity = 0u32;
        for w in 0..self.x.len() {
            parity ^= ((self.x[w] & other.z[w]) ^ (self.z[w] & other.x[w])).count_ones() & 1;
        }
        parity == 1
    }

    #[cfg(test)]
    pub(crate) fn weight(&self) -> usize {
        (0..self.x.len()).map(|w| (self.x[w] | self.z[w]).count_ones() as usize).sum()
    }

    /// Qubits the Pauli acts on, ascending.
    pub(crate) fn support(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for w in 0..self.x.len() {
            let mut m = self.x[w] | self.z[w];
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                out.push(w * 64 + b);
                m &= m - 1;
            }
        }
        out
    }

    pub(crate) fn to_ops(&self) -> Vec<Operator> {
        self.support()
            .into_iter()
            .map(|q| Operator { qubit: q as u16, basis: self.letter(q) })
            .collect()
    }

    /// `_XYZ`-style string over `n` qubits (for debug-build scheduler traces).
    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    pub(crate) fn to_label(&self, n: usize) -> String {
        (0..n).map(|q| self.letter(q)).collect()
    }

    /// Conjugates by a physical Clifford gate: P → G·P·G† (letters only; the same map as
    /// G†·P·G for these gates).
    pub(crate) fn conjugate_by(&mut self, gate: FrameGate) {
        match gate {
            FrameGate::S(q) => {
                // R_Z(pi/4): X -> Y, Y -> X, Z -> Z.
                let (x, z) = self.bits(q);
                self.put(q, x, z ^ x);
            }
            FrameGate::SX(q) => {
                // R_X(pi/4): Z -> Y, Y -> Z, X -> X.
                let (x, z) = self.bits(q);
                self.put(q, x ^ z, z);
            }
            FrameGate::CX(c, t) => {
                // X_c -> X_c X_t, Z_t -> Z_c Z_t.
                let (xc, zc) = self.bits(c);
                let (xt, zt) = self.bits(t);
                self.put(t, xt ^ xc, zt);
                self.put(c, xc, zc ^ zt);
            }
        }
    }

    #[inline]
    fn put(&mut self, q: usize, x: bool, z: bool) {
        let (w, b) = (q / 64, 1u64 << (q % 64));
        if x {
            self.x[w] |= b;
        } else {
            self.x[w] &= !b;
        }
        if z {
            self.z[w] |= b;
        } else {
            self.z[w] &= !b;
        }
    }
}

/// A physical Clifford from the `.trans` file, as it acts on Paulis. S and Sdg act alike up to
/// sign, as do SX and SXdg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameGate {
    S(usize),
    SX(usize),
    CX(usize, usize),
}

impl FrameGate {
    fn qubits(&self) -> Vec<usize> {
        match *self {
            FrameGate::S(q) | FrameGate::SX(q) => vec![q],
            FrameGate::CX(c, t) => vec![c, t],
        }
    }
}

/// The Clifford frame W (see the module docs).
#[derive(Debug, Clone)]
pub(crate) struct CliffordFrame {
    /// Number of qubits (read by debug-build scheduler traces).
    #[cfg_attr(not(any(test, debug_assertions)), allow(dead_code))]
    pub(crate) n: usize,
    pub(crate) words: usize,
    /// rows[2q] = W X_q W†, rows[2q+1] = W Z_q W†.
    rows: Vec<SPauli>,
    /// False while W is the identity (nothing failed yet): images need no work.
    nontrivial: bool,
}

impl CliffordFrame {
    pub(crate) fn new(n: usize) -> Self {
        let words = (n + 63) / 64;
        let mut rows = Vec::with_capacity(2 * n);
        for q in 0..n {
            let mut x = SPauli::identity(words);
            x.set(q, 'X');
            rows.push(x);
            let mut z = SPauli::identity(words);
            z.set(q, 'Z');
            rows.push(z);
        }
        CliffordFrame { n, words, rows, nontrivial: false }
    }

    /// W·P·W† for an ideal Pauli P (letters only).
    pub(crate) fn image(&self, p: &SPauli) -> SPauli {
        if !self.nontrivial {
            return p.clone();
        }
        let mut out = SPauli::identity(self.words);
        for q in p.support() {
            let (x, z) = p.bits(q);
            if x {
                out.mul_assign(&self.rows[2 * q]);
            }
            if z {
                out.mul_assign(&self.rows[2 * q + 1]);
            }
        }
        out
    }

    /// A T about ideal Pauli P failed: W ← W·R_P(−π/4). A generator g that anticommutes with P
    /// becomes ∝ g·P, so its image is multiplied by W·P·W†.
    pub(crate) fn absorb_failed_t(&mut self, p: &SPauli) {
        let executed = self.image(p);
        for q in p.support() {
            let (x, z) = p.bits(q);
            if z {
                // X_q anticommutes with Z or Y on q.
                self.rows[2 * q].mul_assign(&executed);
            }
            if x {
                // Z_q anticommutes with X or Y on q.
                self.rows[2 * q + 1].mul_assign(&executed);
            }
        }
        self.nontrivial = true;
    }

    /// A physical Clifford G ran unchanged: W ← G·W·G†. New image of a generator g is
    /// G·W·(G†gG)·W†·G†: first the rows of G's qubits are recomputed from W's images of G†gG,
    /// then every row is conjugated by G.
    pub(crate) fn absorb_physical_clifford(&mut self, gate: FrameGate) {
        if !self.nontrivial {
            // G·I·G† = I.
            return;
        }
        let qs = gate.qubits();
        let mut updates: Vec<(usize, SPauli)> = Vec::new();
        for &q in &qs {
            for (r, letter) in [(2 * q, 'X'), (2 * q + 1, 'Z')] {
                let mut g = SPauli::identity(self.words);
                g.set(q, letter);
                g.conjugate_by(gate);
                updates.push((r, self.image(&g)));
            }
        }
        for (r, row) in updates {
            self.rows[r] = row;
        }
        for row in &mut self.rows {
            row.conjugate_by(gate);
        }
    }
}

#[cfg(test)]
#[path = "tableau.rs"]
#[allow(dead_code)]
mod tableau_ref;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn sp(label: &str) -> SPauli {
        let mut p = SPauli::identity(1);
        for (q, c) in label.chars().enumerate() {
            p.set(q, c);
        }
        p
    }

    #[test]
    fn letters_and_commutation() {
        assert!(sp("X").anticommutes(&sp("Z")));
        assert!(sp("Y").anticommutes(&sp("Z")));
        assert!(!sp("XX").anticommutes(&sp("ZZ")));
        assert!(sp("XY").anticommutes(&sp("XZ")));
        assert_eq!(sp("_Y_Z").weight(), 2);
        assert_eq!(sp("_Y_Z").to_label(4), "_Y_Z");
        assert_eq!(sp("_Y_Z").support(), vec![1, 3]);
    }

    #[test]
    fn gate_maps_match_the_transpiler_tableau() {
        // Independent check of the symplectic rules against tableau.rs (signed, Vec<bool>).
        use tableau_ref::{Gate1Q, Gate2Q, PauliString, Tableau};
        for (gate, fg) in [
            (Gate1Q::S, FrameGate::S(0)),
            (Gate1Q::Sdg, FrameGate::S(0)),
            (Gate1Q::SX, FrameGate::SX(0)),
            (Gate1Q::SXdg, FrameGate::SX(0)),
        ] {
            for l in ['X', 'Y', 'Z'] {
                let mut t = Tableau::new(1);
                t.prepend_1q_correct(gate, 0);
                let want = t.conjugate(&PauliString::from_str(&l.to_string())).pauli_at(0);
                let mut p = sp(&l.to_string());
                p.conjugate_by(fg);
                let want = if want == 'I' { '_' } else { want };
                assert_eq!(p.letter(0), want, "{:?} on {}", gate, l);
            }
        }
        for a in ['_', 'X', 'Y', 'Z'] {
            for b in ['_', 'X', 'Y', 'Z'] {
                let label: String = [a, b].iter().collect();
                if label == "__" {
                    continue;
                }
                let mut t = Tableau::new(2);
                t.prepend_2q(Gate2Q::CX, 0, 1);
                let want = t.conjugate(&PauliString::from_str(&label.replace('_', "I")));
                let mut p = sp(&label);
                p.conjugate_by(FrameGate::CX(0, 1));
                for q in 0..2 {
                    let w = want.pauli_at(q);
                    assert_eq!(p.letter(q), if w == 'I' { '_' } else { w }, "CX on {}", label);
                }
            }
        }
    }

    #[test]
    fn single_failure_examples() {
        // Failed T about Z0Z1; later X0 runs as Y0Z1 (the multi-qubit example in the docs).
        let mut f = CliffordFrame::new(2);
        f.absorb_failed_t(&sp("ZZ"));
        assert_eq!(f.image(&sp("X_")).to_label(2), "YZ");
        assert_eq!(f.image(&sp("ZZ")).to_label(2), "ZZ");
        assert_eq!(f.image(&sp("Z_")).to_label(2), "Z_");
        // Two failures on one qubit about Z then X: X runs as Y, then the S about Z (a physical
        // Clifford) leaves the frame cycling X -> Y.
        let mut g = CliffordFrame::new(1);
        g.absorb_failed_t(&sp("Z"));
        assert_eq!(g.image(&sp("X")).to_label(1), "Y");
        g.absorb_failed_t(&sp("X"));
        assert_eq!(g.image(&sp("Z")).to_label(1), "X");
    }

    /// Slow reference: keep the events in order and conjugate from the newest one inward.
    /// W_k Q W_k† with W_k = W_{k-1}·R_P  ->  W_{k-1} (R_P Q R_P†) W_{k-1}†,
    /// and W_k = G W_{k-1} G†  ->  G (W_{k-1} (G† Q G) W_{k-1}†) G†.
    enum Ev {
        Fail(SPauli),
        Gate(FrameGate),
    }

    fn reference(events: &[Ev], q: &SPauli) -> SPauli {
        let mut p = q.clone();
        let mut outer = Vec::new();
        for ev in events.iter().rev() {
            match ev {
                Ev::Fail(axis) => {
                    if p.anticommutes(axis) {
                        p.mul_assign(axis);
                    }
                }
                Ev::Gate(g) => {
                    p.conjugate_by(*g);
                    outer.push(*g);
                }
            }
        }
        for g in outer.iter().rev() {
            p.conjugate_by(*g);
        }
        p
    }

    fn random_pauli(rng: &mut StdRng, n: usize, words: usize) -> SPauli {
        let mut p = SPauli::identity(words);
        while p.weight() == 0 {
            for q in 0..n {
                if rng.gen_bool(0.4) {
                    p.set(q, ['X', 'Y', 'Z'][rng.gen_range(0..3)]);
                }
            }
        }
        p
    }

    #[test]
    fn tableau_matches_the_event_reference_on_random_histories() {
        let mut rng = StdRng::seed_from_u64(7);
        for &n in &[1usize, 2, 3, 5, 70, 130] {
            for _ in 0..40 {
                let mut f = CliffordFrame::new(n);
                let mut events = Vec::new();
                for _ in 0..rng.gen_range(1..25) {
                    if rng.gen_bool(0.6) {
                        let p = random_pauli(&mut rng, n, f.words);
                        f.absorb_failed_t(&p);
                        events.push(Ev::Fail(p));
                    } else {
                        let g = match rng.gen_range(0..3) {
                            0 => FrameGate::S(rng.gen_range(0..n)),
                            1 => FrameGate::SX(rng.gen_range(0..n)),
                            _ if n > 1 => {
                                let c = rng.gen_range(0..n);
                                let mut t = rng.gen_range(0..n);
                                while t == c {
                                    t = rng.gen_range(0..n);
                                }
                                FrameGate::CX(c, t)
                            }
                            _ => FrameGate::S(0),
                        };
                        f.absorb_physical_clifford(g);
                        events.push(Ev::Gate(g));
                    }
                    for _ in 0..4 {
                        let q = random_pauli(&mut rng, n, f.words);
                        assert_eq!(f.image(&q), reference(&events, &q), "n={}", n);
                    }
                }
            }
        }
    }

    #[test]
    fn conjugation_preserves_commutation() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut f = CliffordFrame::new(9);
        for _ in 0..30 {
            let p = random_pauli(&mut rng, 9, 1);
            f.absorb_failed_t(&p);
            let a = random_pauli(&mut rng, 9, 1);
            let b = random_pauli(&mut rng, 9, 1);
            assert_eq!(a.anticommutes(&b), f.image(&a).anticommutes(&f.image(&b)));
        }
    }
}
