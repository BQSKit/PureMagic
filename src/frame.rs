//! The Clifford corrections that failed T injections owe, tracked exactly.
//!
//! A T gate injected about the Pauli `P` fails half the time, and then the
//! correction `S_P = exp(-i pi/4 P)` is owed: one Clifford on all of `P`'s
//! qubits, not a single-qubit S on each. Rather than apply it at once, the
//! scheduler carries it and lets later operations deal with it.
//!
//! Write the circuit executed so far as `V` and the owed corrections as a
//! product `W`, so that the ideal circuit is `U = W V`. Then:
//!
//! - the next ideal rotation about `Q` is executed about `Q' = W^dag Q W`,
//!   which leaves `W` unchanged. `W = S_oldest ... S_newest`, so `Q` is
//!   conjugated by the oldest owed correction first;
//! - a failure of that rotation appends `S_Q'` as the newest factor, the
//!   first in time;
//! - emitting `S_E` as a gate removes it from `W`, which is valid only if
//!   every newer factor commutes with it;
//! - a Clifford `C` can run once every owed factor on its qubits commutes
//!   with it.
//!
//! Conjugating `Q` by an `S_P` it anticommutes with gives `i P Q`, whose
//! support is the union of both. Widening is always valid: conjugation keeps
//! commutation, so products on disjoint qubits still commute and may run in
//! any order. It only costs a wider product, which must not share a qubit with
//! anything else running in its cycle. So a correction is absorbed, widening
//! the product as needed, as long as the result stays within the weight limit
//! the circuit was transpiled with. A correction that would take it past the
//! limit is emitted first instead. Signs are dropped throughout, as elsewhere in
//! the scheduler: a sign only decides which injection outcome counts as a
//! failure.

use crate::pauliproduct::Operator;

/// One owed correction `S_P^power`, with `P`'s operators sorted by qubit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Correction {
    pub(crate) ops: Vec<Operator>,
    /// 1 for S, 3 for S-dagger. A net power of 2 is the Pauli `P` and leaves.
    pub(crate) power: u8,
    seq: u64,
}

/// What a failed T gate did to the frame.
#[derive(Debug, PartialEq)]
pub(crate) enum Added {
    /// A new owed correction.
    New,
    /// Combined with the newest owed correction on the same `P`, and the net
    /// power is 2 or 0: a Pauli or nothing, which the classical frame takes.
    Cancelled,
}

/// The result of passing an operation through the frame.
#[derive(Debug, PartialEq)]
pub(crate) enum Pass {
    /// It runs as given, possibly about conjugated operators.
    Run(Vec<Operator>),
    /// These corrections must be emitted first, in this order.
    Blocked(Vec<usize>),
}

/// The owed corrections, in creation order, indexed by qubit.
pub(crate) struct CorrectionFrame {
    slots: Vec<Option<Correction>>,
    free: Vec<usize>,
    /// For each qubit, the slots of the corrections on it, oldest first.
    on_qubit: Vec<Vec<usize>>,
    next_seq: u64,
    open: usize,
}

/// Label of `ops` on `q`, or 'I'.
fn label(ops: &[Operator], q: u16) -> char {
    match ops.binary_search_by_key(&q, |o| o.qubit) {
        Ok(i) => ops[i].basis,
        Err(_) => 'I',
    }
}

/// Whether two Paulis (operators sorted by qubit) commute.
pub(crate) fn commutes(a: &[Operator], b: &[Operator]) -> bool {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].qubit.cmp(&b[j].qubit) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += (a[i].basis != b[j].basis) as usize;
                i += 1;
                j += 1;
            }
        }
    }
    n % 2 == 0
}

/// Product of two single-qubit labels, sign dropped.
fn mul_label(a: char, b: char) -> char {
    match (a, b) {
        ('I', x) | (x, 'I') => x,
        (x, y) if x == y => 'I',
        ('X', 'Y') | ('Y', 'X') => 'Z',
        ('Y', 'Z') | ('Z', 'Y') => 'X',
        _ => 'Y',
    }
}

/// `P Q` with the sign dropped, both sorted by qubit, over the union of supports.
fn mul(p: &[Operator], q: &[Operator]) -> Vec<Operator> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(p.len() + q.len());
    while i < p.len() || j < q.len() {
        let (qubit, b) = if j == q.len() || (i < p.len() && p[i].qubit < q[j].qubit) {
            i += 1;
            (p[i - 1].qubit, p[i - 1].basis)
        } else if i == p.len() || q[j].qubit < p[i].qubit {
            j += 1;
            (q[j - 1].qubit, q[j - 1].basis)
        } else {
            i += 1;
            j += 1;
            (p[i - 1].qubit, mul_label(p[i - 1].basis, q[j - 1].basis))
        };
        if b != 'I' {
            out.push(Operator { qubit, basis: b });
        }
    }
    out
}

/// A single-qubit Clifford modulo Paulis, as its action on the labels X, Y, Z
/// (indices 0, 1, 2) by conjugation, signs dropped.
type Perm = [u8; 3];

const IDENTITY: Perm = [0, 1, 2];

fn index(a: char) -> usize {
    match a {
        'X' => 0,
        'Y' => 1,
        _ => 2,
    }
}

/// S about `axis` fixes it and swaps the other two labels.
fn tau(axis: char) -> Perm {
    match axis {
        'X' => [0, 2, 1],
        'Y' => [2, 1, 0],
        _ => [1, 0, 2],
    }
}

/// `first`, then `second`.
fn then(first: Perm, second: Perm) -> Perm {
    [second[first[0] as usize], second[first[1] as usize], second[first[2] as usize]]
}

/// The fewest S gates, oldest first, whose product acts as `pi`: none for the
/// identity, one for a swap, two for a three-cycle, avoiding Y where possible
/// since an S about Y needs both edges of the patch.
fn decompose(pi: Perm) -> Vec<char> {
    if pi == IDENTITY {
        return vec![];
    }
    for a in ['Z', 'X', 'Y'] {
        if pi == tau(a) {
            return vec![a];
        }
    }
    for (a, b) in [('X', 'Z'), ('Z', 'X'), ('X', 'Y'), ('Y', 'X'), ('Z', 'Y'), ('Y', 'Z')] {
        if then(tau(a), tau(b)) == pi {
            return vec![a, b];
        }
    }
    unreachable!("every single-qubit Clifford is at most two S gates")
}

#[allow(dead_code)]
fn label_index(a: char) -> usize {
    index(a)
}

/// Operators sorted by qubit, as the frame keeps them.
pub(crate) fn sorted(ops: &[Operator]) -> Vec<Operator> {
    let mut v = ops.to_vec();
    v.sort_by_key(|o| o.qubit);
    v
}

impl CorrectionFrame {
    pub(crate) fn new(n_qubits: usize) -> Self {
        CorrectionFrame {
            slots: Vec::new(),
            free: Vec::new(),
            on_qubit: vec![Vec::new(); n_qubits],
            next_seq: 0,
            open: 0,
        }
    }

    pub(crate) fn get(&self, slot: usize) -> &Correction {
        self.slots[slot].as_ref().expect("live correction slot")
    }

    /// Corrections still owed.
    pub(crate) fn open(&self) -> usize {
        self.open
    }

    /// Whether nothing is owed on any of these qubits: the common case, checked
    /// first so a clean qubit costs one lookup.
    pub(crate) fn clear_on(&self, ops: &[Operator]) -> bool {
        ops.iter().all(|o| self.on_qubit[o.qubit as usize].is_empty())
    }

    /// Slots of the corrections on these qubits, oldest first, without repeats.
    fn on(&self, ops: &[Operator]) -> Vec<usize> {
        let mut v: Vec<usize> =
            ops.iter().flat_map(|o| self.on_qubit[o.qubit as usize].iter().copied()).collect();
        v.sort_by_key(|&s| self.get(s).seq);
        v.dedup();
        v
    }

    fn remove(&mut self, slot: usize) {
        let c = self.slots[slot].take().expect("live correction slot");
        for o in &c.ops {
            self.on_qubit[o.qubit as usize].retain(|&s| s != slot);
        }
        self.free.push(slot);
        self.open -= 1;
    }

    /// Records the correction a failed T gate executed about `ops` owes. It
    /// combines with the newest correction on these qubits if that one is about
    /// the same Pauli, which is the newest factor of `W`, so the two are adjacent.
    pub(crate) fn add_failure(&mut self, ops: &[Operator]) -> Added {
        let p = sorted(ops);
        if p.len() == 1 {
            return self.add_single(p[0].qubit, p[0].basis);
        }
        if let Some(&newest) = self.on(&p).last() {
            if self.get(newest).ops == p {
                let c = self.slots[newest].as_mut().unwrap();
                c.power = (c.power + 1) % 4;
                if c.power % 2 == 0 {
                    self.remove(newest);
                    return Added::Cancelled;
                }
                return Added::New;
            }
        }
        self.push(p);
        Added::New
    }

    /// Adds `S_P` as the newest owed correction.
    fn push(&mut self, ops: Vec<Operator>) {
        let c = Correction { ops, power: 1, seq: self.next_seq };
        self.next_seq += 1;
        let slot = match self.free.pop() {
            Some(s) => {
                self.slots[s] = Some(c);
                s
            }
            None => {
                self.slots.push(Some(c));
                self.slots.len() - 1
            }
        };
        for o in &self.get(slot).ops.clone() {
            self.on_qubit[o.qubit as usize].push(slot);
        }
        self.open += 1;
    }

    /// A single-qubit failure. Single-qubit Cliffords on one qubit compose, and
    /// modulo the Paulis the classical frame takes there are six of them, each at
    /// most two S gates about different axes. So the newest owed corrections on
    /// the qubit that are single-qubit, those after any multi-qubit one there, are
    /// composed with the new one and kept as that canonical form: none (a Pauli),
    /// one S, or two. Without this, a qubit's corrections would pile up and each
    /// later cost a gate.
    fn add_single(&mut self, qubit: u16, axis: char) -> Added {
        let on = &self.on_qubit[qubit as usize];
        let mut suffix: Vec<usize> = Vec::new();
        let mut by_seq = on.clone();
        by_seq.sort_by_key(|&s| self.get(s).seq);
        for &s in by_seq.iter().rev() {
            if self.get(s).ops.len() != 1 {
                break;
            }
            suffix.push(s);
        }
        suffix.reverse(); // oldest first
        // Conjugation by W = S_a1 ... S_ak applies a1 first: pi = tau_ak ... tau_a1.
        let mut pi = IDENTITY;
        for &s in &suffix {
            pi = then(pi, tau(self.get(s).ops[0].basis));
        }
        pi = then(pi, tau(axis));
        for s in suffix {
            self.remove(s);
        }
        let factors = decompose(pi);
        if factors.is_empty() {
            return Added::Cancelled;
        }
        for a in factors {
            self.push(vec![Operator { qubit, basis: a }]);
        }
        Added::New
    }

    /// The corrections that must be emitted, in order, before `slot` can be:
    /// every newer one on its qubits that does not commute with it, each
    /// preceded by its own, and then `slot` itself.
    pub(crate) fn emission_order(&self, slot: usize) -> Vec<usize> {
        let mut out = Vec::new();
        self.push_emission(slot, &mut out);
        out
    }

    fn push_emission(&self, slot: usize, out: &mut Vec<usize>) {
        if out.contains(&slot) {
            return;
        }
        let c = self.get(slot);
        // Newest first: the newest factor is the first in time.
        for &s in self.on(&c.ops).iter().rev() {
            let d = self.get(s);
            if d.seq > c.seq && !commutes(&d.ops, &c.ops) {
                self.push_emission(s, out);
            }
        }
        out.push(slot);
    }

    fn blocked(&self, first: usize) -> Pass {
        Pass::Blocked(self.emission_order(first))
    }

    /// A T gate or measurement about `ops`: the operators to run it about, or
    /// the corrections to emit first. The owed corrections are applied oldest
    /// first, each only if it overlaps the product as it stands by then, so one
    /// that widens it brings in the newer corrections on the qubits it gained.
    /// A correction that would take it past `max_weight` (0: no limit) is
    /// emitted instead.
    pub(crate) fn pass_rotation(&self, ops: &[Operator], max_weight: usize) -> Pass {
        let mut q = sorted(ops);
        let mut cand = self.on(&q);
        let mut i = 0;
        while i < cand.len() {
            let s = cand[i];
            i += 1;
            let p = &self.get(s).ops;
            if commutes(p, &q) {
                continue;
            }
            let r = mul(p, &q);
            if max_weight > 0 && r.len() > max_weight {
                return self.blocked(s);
            }
            let seq = self.get(s).seq;
            let mut grew = false;
            for o in &r {
                if q.binary_search_by_key(&o.qubit, |x| x.qubit).is_err() {
                    for &t in &self.on_qubit[o.qubit as usize] {
                        if self.get(t).seq > seq && !cand[i..].contains(&t) {
                            cand.push(t);
                            grew = true;
                        }
                    }
                }
            }
            if grew {
                cand[i..].sort_by_key(|&t| self.get(t).seq);
            }
            q = r;
        }
        Pass::Run(q)
    }

    /// A Clifford whose operators are `ops`, one per qubit, each naming the
    /// axis the Clifford fixes on that qubit: Z for S, X for SX, and for a CX
    /// Z on its control and X on its target. `S_P` commutes with it exactly
    /// when, on each of its qubits, `P` is the identity or that axis. Returns
    /// the corrections to emit first, or none.
    pub(crate) fn pass_clifford(&self, ops: &[Operator]) -> Option<Vec<usize>> {
        let ops = sorted(ops);
        self.on(&ops)
            .into_iter()
            .find(|&s| {
                let p = &self.get(s).ops;
                ops.iter().any(|o| {
                    let l = label(p, o.qubit);
                    l != 'I' && l != o.basis
                })
            })
            .map(|s| self.emission_order(s))
    }

    /// An S or SX on one qubit about `axis` meets owed corrections there. If the
    /// only one is single-qubit about the same axis, the two merge: returns its
    /// slot and the net power. The caller takes it out with `emitted` once the
    /// merged gate is placed.
    pub(crate) fn merge_candidate(&self, qubit: u16, axis: char) -> Option<(usize, u8)> {
        let on = &self.on_qubit[qubit as usize];
        if on.len() != 1 {
            return None;
        }
        let s = on[0];
        let c = self.get(s);
        if c.ops.len() != 1 || c.ops[0].basis != axis {
            return None;
        }
        Some((s, (c.power + 1) % 4))
    }

    /// Takes a correction out of the frame once its gate has been placed.
    pub(crate) fn emitted(&mut self, slot: usize) {
        self.remove(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn ops(s: &str) -> Vec<Operator> {
        s.chars()
            .enumerate()
            .filter(|(_, c)| *c != '_')
            .map(|(q, c)| Operator { qubit: q as u16, basis: c })
            .collect()
    }

    #[test]
    fn a_commuting_product_passes_unchanged() {
        // The per-qubit model turned XX into YY under a failed ZZ.
        let mut f = CorrectionFrame::new(2);
        f.add_failure(&ops("ZZ"));
        assert_eq!(f.pass_rotation(&ops("XX"), 0), Pass::Run(ops("XX")));
    }

    #[test]
    fn an_anticommuting_product_on_the_same_qubits_absorbs_it() {
        let mut f = CorrectionFrame::new(2);
        f.add_failure(&ops("ZZ"));
        // ZX anticommutes with ZZ: (ZZ)(ZX) = I.Y up to sign.
        assert_eq!(f.pass_rotation(&ops("ZX"), 2), Pass::Run(ops("_Y")));
        assert_eq!(f.open(), 1, "absorbing leaves the correction owed");
    }

    #[test]
    fn a_product_widens_within_the_weight_limit() {
        let mut f = CorrectionFrame::new(3);
        f.add_failure(&ops("ZZ_"));
        // X on qubit 0 anticommutes with ZZ: (ZZ)(X_) = YZ up to sign.
        assert_eq!(f.pass_rotation(&ops("X__"), 0), Pass::Run(ops("YZ_")));
        assert_eq!(f.pass_rotation(&ops("X__"), 2), Pass::Run(ops("YZ_")));
        assert_eq!(f.open(), 1, "absorbing leaves the correction owed");
    }

    #[test]
    fn widening_past_the_limit_emits_the_correction() {
        let mut f = CorrectionFrame::new(3);
        f.add_failure(&ops("ZZ_"));
        match f.pass_rotation(&ops("X__"), 1) {
            Pass::Blocked(v) => assert_eq!(f.get(v[0]).ops, ops("ZZ_")),
            p => panic!("expected a block, got {:?}", p),
        }
    }

    #[test]
    fn a_widened_product_meets_newer_corrections_on_its_new_qubits() {
        let mut f = CorrectionFrame::new(3);
        f.add_failure(&ops("__X")); // older, on qubit 2: not met, X__ never touches it then
        f.add_failure(&ops("ZZ_")); // widens X__ to YZ_
        f.add_failure(&ops("_XX")); // newer, on qubit 1 the product gained: anticommutes
        // X__ -> YZ_ under ZZ_, then (_XX)(YZ_) = YYX.
        assert_eq!(f.pass_rotation(&ops("X__"), 0), Pass::Run(ops("YYX")));
    }

    #[test]
    fn mixed_axes_on_one_qubit_are_not_a_pauli() {
        // The per-qubit model counted S_X then S_Y as power 2 and dropped it.
        let mut f = CorrectionFrame::new(1);
        assert_eq!(f.add_failure(&ops("X")), Added::New);
        assert_eq!(f.add_failure(&ops("Y")), Added::New);
        assert_eq!(f.open(), 2);
        // S_X then S_Y acts as a three-cycle on the labels. It is kept as the
        // equivalent S_Z then S_X, avoiding Y. An S (fixes Z) needs the newer,
        // S_X, emitted. The older S_Z commutes with it and then merges into it.
        let order = f.pass_clifford(&ops("Z")).unwrap();
        assert_eq!(order.len(), 1);
        assert_eq!(f.get(order[0]).ops, ops("X"));
        f.emitted(order[0]);
        assert_eq!(f.pass_clifford(&ops("Z")), None);
        assert!(f.merge_candidate(0, 'Z').is_some());
    }

    #[test]
    fn the_same_pauli_twice_is_a_pauli() {
        let mut f = CorrectionFrame::new(2);
        f.add_failure(&ops("ZX"));
        assert_eq!(f.add_failure(&ops("ZX")), Added::Cancelled);
        assert_eq!(f.open(), 0);
    }

    #[test]
    fn a_cx_passes_corrections_it_fixes() {
        let mut f = CorrectionFrame::new(2);
        f.add_failure(&ops("Z_"));
        f.add_failure(&ops("_X"));
        assert_eq!(f.pass_clifford(&ops("ZX")), None);
        f.add_failure(&ops("XX"));
        assert!(f.pass_clifford(&ops("ZX")).is_some());
    }

    #[test]
    fn a_same_axis_s_merges() {
        let mut f = CorrectionFrame::new(1);
        f.add_failure(&ops("Z"));
        let (slot, net) = f.merge_candidate(0, 'Z').unwrap();
        assert_eq!(net, 2);
        f.emitted(slot);
        assert_eq!(f.open(), 0);
        f.add_failure(&ops("X"));
        assert_eq!(f.merge_candidate(0, 'Z'), None);
    }

    // --- dense check of the algebra on a few qubits -------------------------

    type C = (f64, f64);
    type M = Vec<Vec<C>>;

    fn cmul(a: C, b: C) -> C {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    fn mm(a: &M, b: &M) -> M {
        let n = a.len();
        let mut r = vec![vec![(0.0, 0.0); n]; n];
        for i in 0..n {
            for k in 0..n {
                if a[i][k] == (0.0, 0.0) {
                    continue;
                }
                for j in 0..n {
                    let p = cmul(a[i][k], b[k][j]);
                    r[i][j].0 += p.0;
                    r[i][j].1 += p.1;
                }
            }
        }
        r
    }

    fn dag(a: &M) -> M {
        let n = a.len();
        (0..n).map(|i| (0..n).map(|j| (a[j][i].0, -a[j][i].1)).collect()).collect()
    }

    fn pauli(n: usize, p: &[Operator]) -> M {
        let one: [[C; 2]; 2] = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)]];
        let x: [[C; 2]; 2] = [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]];
        let y: [[C; 2]; 2] = [[(0.0, 0.0), (0.0, -1.0)], [(0.0, 1.0), (0.0, 0.0)]];
        let z: [[C; 2]; 2] = [[(1.0, 0.0), (0.0, 0.0)], [(0.0, 0.0), (-1.0, 0.0)]];
        let dim = 1 << n;
        let mut r = vec![vec![(0.0, 0.0); dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut v = (1.0, 0.0);
                for q in 0..n {
                    let m = match label(p, q as u16) {
                        'X' => &x,
                        'Y' => &y,
                        'Z' => &z,
                        _ => &one,
                    };
                    v = cmul(v, m[(i >> q) & 1][(j >> q) & 1]);
                }
                r[i][j] = v;
            }
        }
        r
    }

    /// exp(-i theta P) = cos theta I - i sin theta P.
    fn rot(n: usize, p: &[Operator], theta: f64) -> M {
        let pm = pauli(n, p);
        let dim = 1 << n;
        (0..dim)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let id = if i == j { theta.cos() } else { 0.0 };
                        (id + theta.sin() * pm[i][j].1, -theta.sin() * pm[i][j].0)
                    })
                    .collect()
            })
            .collect()
    }

    /// a equals b up to a global phase.
    fn same_up_to_phase(a: &M, b: &M) -> bool {
        let n = a.len();
        let (mut bi, mut bj) = (0, 0);
        for i in 0..n {
            for j in 0..n {
                if b[i][j].0.hypot(b[i][j].1) > b[bi][bj].0.hypot(b[bi][bj].1) {
                    (bi, bj) = (i, j);
                }
            }
        }
        let d = b[bi][bj].0 * b[bi][bj].0 + b[bi][bj].1 * b[bi][bj].1;
        let ph = cmul(a[bi][bj], (b[bi][bj].0 / d, -b[bi][bj].1 / d));
        (0..n).all(|i| {
            (0..n).all(|j| {
                let p = cmul(ph, b[i][j]);
                (a[i][j].0 - p.0).abs() < 1e-9 && (a[i][j].1 - p.1).abs() < 1e-9
            })
        })
    }

    fn random_pauli(n: usize, rng: &mut StdRng) -> Vec<Operator> {
        loop {
            let p: Vec<Operator> = (0..n)
                .filter_map(|q| {
                    let b = ['I', 'X', 'Y', 'Z'][rng.gen_range(0..4)];
                    (b != 'I').then(|| Operator { qubit: q as u16, basis: b })
                })
                .collect();
            if !p.is_empty() {
                return p;
            }
        }
    }

    /// Whether a and b are equal up to sign (both Hermitian Paulis or Cliffords
    /// with real global phase handled by `same_up_to_phase`).
    fn commute_up_to_phase(a: &M, b: &M) -> bool {
        same_up_to_phase(&mm(a, b), &mm(b, a))
    }

    /// Runs random sequences of T rotations, single-qubit S gates and CX
    /// gates through the frame, with random failures, and checks the algebra
    /// the scheduler relies on against dense matrices. W is the product of the
    /// owed corrections. Signs are dropped, as in the frame: an owed S may
    /// really be S-dagger, and a cancelled pair may leave a Pauli, but neither
    /// changes an axis up to sign or whether two operators commute up to sign,
    /// which is all the scheduler's cost model uses.
    ///   1. every executed axis is +/- W^dag P W;
    ///   2. a Clifford is let through only if it commutes with W;
    ///   3. a correction is emitted only when every newer one commutes with it,
    ///      so it can be moved to the front of W.
    #[test]
    fn the_frame_matches_dense_conjugation() {
        let n = 4;
        let quarter = std::f64::consts::PI / 4.0;
        for seed in 0..600 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut f = CorrectionFrame::new(n);
            // 0: no weight limit.
            let limit = rng.gen_range(0..n);
            // W from the history alone: every failure appended, every emission
            // or merge taken off. Compression must not change its action.
            let dim = 1 << n;
            let mut hist: M = (0..dim)
                .map(|i| (0..dim).map(|j| if i == j { (1.0, 0.0) } else { (0.0, 0.0) }).collect())
                .collect();
            let emit = |f: &mut CorrectionFrame, hist: &mut M, order: Vec<usize>| {
                for s in order {
                    *hist = mm(hist, &dag(&rot(n, &f.get(s).ops, quarter)));
                    // 3. every newer correction commutes with this one.
                    let c = f.get(s).clone();
                    let cm = rot(n, &c.ops, quarter);
                    for d in f.slots.iter().flatten() {
                        if d.seq > c.seq {
                            assert!(
                                commute_up_to_phase(&cm, &rot(n, &d.ops, quarter)),
                                "seed {}: emitted {:?} past a newer {:?}",
                                seed,
                                c.ops,
                                d.ops
                            );
                        }
                    }
                    f.emitted(s);
                }
            };
            for _ in 0..16 {
                let r: f64 = rng.gen_range(0.0..1.0);
                if r < 0.3 {
                    // A Clifford: S, SX, or a CX written as Z on control, X on target.
                    let g: Vec<Operator> = if r < 0.2 {
                        let q = rng.gen_range(0..n) as u16;
                        vec![Operator { qubit: q, basis: if rng.gen_bool(0.5) { 'Z' } else { 'X' } }]
                    } else {
                        let c = rng.gen_range(0..n) as u16;
                        let t = (c + 1 + rng.gen_range(0..n as u16 - 1)) % n as u16;
                        sorted(&[Operator { qubit: c, basis: 'Z' }, Operator { qubit: t, basis: 'X' }])
                    };
                    let cliff = if g.len() == 1 {
                        rot(n, &g, quarter)
                    } else {
                        // CX = (Z_c X_t)_{pi/4} (Z_c)_{-pi/4} (X_t)_{-pi/4}
                        let zc: Vec<Operator> = g.iter().filter(|o| o.basis == 'Z').cloned().collect();
                        let xt: Vec<Operator> = g.iter().filter(|o| o.basis == 'X').cloned().collect();
                        mm(&mm(&rot(n, &g, quarter), &rot(n, &zc, -quarter)), &rot(n, &xt, -quarter))
                    };
                    while let Some(order) = f.pass_clifford(&g) {
                        emit(&mut f, &mut hist, order);
                    }
                    if g.len() == 1 {
                        if let Some((slot, _)) = f.merge_candidate(g[0].qubit, g[0].basis) {
                            hist = mm(&hist, &dag(&rot(n, &f.get(slot).ops, quarter)));
                            f.emitted(slot);
                            continue;
                        }
                    }
                    // 2. it now commutes with every owed correction.
                    assert!(f.pass_clifford(&g).is_none());
                    let w = hist.clone();
                    assert!(
                        commute_up_to_phase(&cliff, &w)
                            || same_up_to_phase(&mm(&mm(&cliff, &w), &dag(&cliff)), &w)
                            || pauli_ratio(&mm(&mm(&cliff, &w), &dag(&cliff)), &w, n),
                        "seed {}: Clifford {:?} let through a correction it moves",
                        seed,
                        g
                    );
                    continue;
                }
                let p = loop {
                    let p = random_pauli(n, &mut rng);
                    if limit == 0 || p.len() <= limit {
                        break p;
                    }
                };
                let q = loop {
                    match f.pass_rotation(&p, limit) {
                        Pass::Run(q) => break q,
                        Pass::Blocked(order) => emit(&mut f, &mut hist, order),
                    }
                };
                // 1. the executed axis is +/- W^dag P W, within the weight limit,
                // with W from the history, not the frame's compressed entries.
                let w = hist.clone();
                let target = mm(&mm(&dag(&w), &pauli(n, &p)), &w);
                assert!(
                    same_up_to_phase(&target, &pauli(n, &q)),
                    "seed {}: {:?} ran as {:?}, not W^dag P W",
                    seed,
                    p,
                    q
                );
                assert!(limit == 0 || q.len() <= limit, "seed {}: {:?} over the limit", seed, q);
                if rng.gen_bool(0.5) {
                    f.add_failure(&q);
                    hist = mm(&hist, &rot(n, &q, quarter));
                }
            }
        }
    }

    /// Whether a = b P for some Pauli P (up to phase).
    fn pauli_ratio(a: &M, b: &M, n: usize) -> bool {
        let r = mm(&dag(b), a);
        (0..1usize << (2 * n)).any(|code| {
            let p: Vec<Operator> = (0..n)
                .filter_map(|q| {
                    let c = ['I', 'X', 'Y', 'Z'][(code >> (2 * q)) & 3];
                    (c != 'I').then(|| Operator { qubit: q as u16, basis: c })
                })
                .collect();
            same_up_to_phase(&r, &pauli(n, &p))
        })
    }

    /// W as a matrix, the product of owed corrections, oldest leftmost.
    fn owed(f: &CorrectionFrame, n: usize) -> M {
        let mut live: Vec<&Correction> = f.slots.iter().flatten().collect();
        live.sort_by_key(|c| c.seq);
        let dim = 1 << n;
        let mut w: M = (0..dim)
            .map(|i| (0..dim).map(|j| if i == j { (1.0, 0.0) } else { (0.0, 0.0) }).collect())
            .collect();
        for c in live {
            w = mm(&w, &rot(n, &c.ops, std::f64::consts::PI / 4.0 * c.power as f64));
        }
        w
    }
}
