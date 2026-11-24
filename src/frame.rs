//! The Clifford corrections that failed T injections owe, tracked exactly.
//!
//! A T gate injected about the Pauli `P` fails half the time, and then the
//! correction `S_P = exp(-i pi/4 P)` is owed: one Clifford on all of `P`'s
//! qubits, not a single-qubit S on each. Rather than apply it at once, the
//! scheduler carries it and lets later operations deal with it.
//!
//! Write the circuit executed so far as `V` and the owed corrections as a
//! product `W`, so that the ideal state is `W` times the physical one and the
//! ideal circuit is `U = W V`. Each factor is labelled by the operator the
//! failed injection physically ran about, and new factors go on the right, so
//! `W = S_oldest ... S_newest`. (Labelling factors in ideal coordinates instead
//! puts new ones on the left and reverses every order below.) Then:
//!
//! - the next ideal rotation about `Q` is executed about `Q' = W^dag Q W`,
//!   which leaves `W` unchanged. `Q` is conjugated by the oldest owed
//!   correction first, since it is innermost;
//! - a failure of that rotation, which ran `T_Q'^dag = S_Q'^dag T_Q'`, makes
//!   `W` into `W S_Q'`: `S_Q'`, about the physical `Q'`, is the newest factor;
//! - to apply the corrections physically, the rightmost goes first: emitting
//!   `S_E` as a gate removes it from `W` only if every newer factor commutes
//!   with it, so newer ones that do not are emitted before it. For example, a
//!   T about Z fails (W = S_Z), the next T about X runs about Y and fails
//!   (W = S_Z S_Y): S_Y is applied, then S_Z;
//! - a Clifford `C` runs as it is and is pushed through the frame: the ideal
//!   state after it is `C W` times the physical one before it, which is
//!   `(C W C^dag)` times the physical one after it, so `W` becomes `C W C^dag`
//!   and every owed `S_P` becomes `S_{C P C^dag}`. An S or SX keeps every
//!   correction on the same qubits, so it always runs at once. A CX could
//!   spread a correction onto its other qubit; such corrections are emitted
//!   first, so corrections never grow through Cliffords, and at weight 1 every
//!   correction stays on one qubit.
//!
//! Conjugating `Q` by an `S_P` it anticommutes with gives the Hermitian Pauli
//! `i P Q`, whose weight is that of the product, at most the size of the union
//! of both supports (labels that agree cancel). Widening is always valid: conjugation keeps
//! commutation, so products on disjoint qubits still commute and may run in
//! any order. It only costs a wider product, which must not share a qubit with
//! anything else running in its cycle. So a correction is absorbed, widening
//! the product as needed, as long as the result stays within the weight limit
//! the circuit was transpiled with. A correction that would take it past the
//! limit is emitted first instead.
//!
//! Signs are dropped throughout, so the frame is tracked modulo Paulis. That is
//! exact for the cost model, not for a controller: a sign turns a T into a T
//! dagger, an S into an S dagger, or flips a measurement outcome, each of which
//! costs the same as the unsigned operation. A real controller must also track
//! the signs, a Pauli frame alongside this one.

use crate::pauliproduct::Operator;

/// The most corrections [`CorrectionFrame::emission_order`] returns at once.
const EMISSION_PREFIX: usize = 32;

/// One owed correction `S_P^power`, with `P`'s operators sorted by qubit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Correction {
    pub(crate) ops: Vec<Operator>,
    /// 1 for S, 3 for S-dagger. A net power of 2 is the Pauli `P` and leaves.
    pub(crate) power: u8,
    /// Creation order, counting every correction the frame has made.
    pub(crate) seq: u64,
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
    /// This correction must be emitted first (see
    /// [`CorrectionFrame::emission_plan`]).
    Blocked(usize),
}

/// The owed corrections, in creation order, indexed by qubit.
pub(crate) struct CorrectionFrame {
    slots: Vec<Option<Correction>>,
    free: Vec<usize>,
    /// For each qubit, the slots of the corrections on it, oldest first.
    on_qubit: Vec<Vec<usize>>,
    next_seq: u64,
    open: usize,
    /// `W` itself, modulo Paulis, so that conjugating by it costs the same however
    /// many corrections are owed. The list above is kept for choosing which to emit.
    tab: Tableau,
    /// For each qubit, a count bumped whenever its list or a label on it changes.
    ver: Vec<u64>,
    /// [`CorrectionFrame::cx_blockers`]'s last answer for a CX, by its qubits and axes,
    /// with both qubits' versions then: a pending CX asks every cycle, and the
    /// answer changes only with those lists.
    cx_cache: std::collections::HashMap<(u16, char, u16, char), (u64, u64, Option<usize>)>,
}

/// A Clifford modulo Paulis, as the images of the generators under conjugation:
/// row `q` is `W^dag X_q W` and row `n + q` is `W^dag Z_q W`, each a Pauli as bits,
/// X part in the first `words` words and Z part in the next `words`.
struct Tableau {
    n: usize,
    words: usize,
    rows: Vec<u64>,
}

impl Tableau {
    fn new(n: usize) -> Self {
        let words = n.div_ceil(64).max(1);
        let mut t = Tableau { n, words, rows: vec![0; 2 * n * 2 * words] };
        for q in 0..n {
            t.row_mut(q)[q / 64] |= 1 << (q % 64);
            t.row_mut(n + q)[words + q / 64] |= 1 << (q % 64);
        }
        t
    }

    fn row_mut(&mut self, r: usize) -> &mut [u64] {
        let w = 2 * self.words;
        &mut self.rows[r * w..(r + 1) * w]
    }

    fn bits(&self, ops: &[Operator]) -> Vec<u64> {
        let mut v = vec![0u64; 2 * self.words];
        for o in ops {
            let (i, b) = (o.qubit as usize / 64, 1u64 << (o.qubit % 64));
            if o.basis != 'Z' {
                v[i] |= b;
            }
            if o.basis != 'X' {
                v[self.words + i] |= b;
            }
        }
        v
    }

    /// `W <- W S_L` (or `W S_L^dag`, the same modulo Paulis): every image `v`
    /// that anticommutes with `L` becomes `v L`.
    fn apply_s(&mut self, ops: &[Operator]) {
        let l = self.bits(ops);
        let w = self.words;
        for r in 0..2 * self.n {
            let row = self.row_mut(r);
            let mut par = 0u32;
            for i in 0..w {
                par ^= ((row[i] & l[w + i]) ^ (row[w + i] & l[i])).count_ones();
            }
            if par & 1 == 1 {
                for (x, y) in row.iter_mut().zip(&l) {
                    *x ^= y;
                }
            }
        }
    }

    /// `v -> C v C^dag` on one Pauli's bits, signs dropped, for the Clifford
    /// fixing `axes` (see [`CorrectionFrame::push_clifford`]). On one qubit, an S
    /// about the axis `A`: `v` becomes `v A` if it anticommutes with `A`. On two,
    /// `exp(i pi/4 (I - A)(I - B))`, a CX for `A = Z` on its control and `B = X`
    /// on its target: `v` picks up `B` if it anticommutes with `A`, and `A` if it
    /// anticommutes with `B`. Each is its own inverse modulo Paulis, so this is
    /// also `C^dag v C`.
    fn clifford_bits(words: usize, axes: &[Operator], v: &mut [u64]) {
        let one = |o: &Operator| {
            let mut a = vec![0u64; 2 * words];
            let (i, b) = (o.qubit as usize / 64, 1u64 << (o.qubit % 64));
            if o.basis != 'Z' {
                a[i] |= b;
            }
            if o.basis != 'X' {
                a[words + i] |= b;
            }
            a
        };
        let anti = |v: &[u64], a: &[u64]| {
            (0..words).map(|i| ((v[i] & a[words + i]) ^ (v[words + i] & a[i])).count_ones()).sum::<u32>()
                & 1
                == 1
        };
        let xor = |v: &mut [u64], a: &[u64]| v.iter_mut().zip(a).for_each(|(x, y)| *x ^= y);
        match axes {
            [o] => {
                let a = one(o);
                if anti(v, &a) {
                    xor(v, &a);
                }
            }
            [o, p] => {
                let (a, b) = (one(o), one(p));
                let (va, vb) = (anti(v, &a), anti(v, &b));
                if va {
                    xor(v, &b);
                }
                if vb {
                    xor(v, &a);
                }
            }
            _ => unreachable!("a Clifford acts on one or two qubits"),
        }
    }

    /// `W <- C W C^dag`: the images become `G(W^dag G(g) W)` with `G` the
    /// conjugation by `C`.
    fn push_clifford(&mut self, axes: &[Operator]) {
        let (w, n) = (2 * self.words, self.n);
        let old = self.rows.clone();
        let mut g = vec![0u64; w];
        for r in 0..2 * n {
            g.iter_mut().for_each(|x| *x = 0);
            let (z, q) = (r >= n, (r % n) as u16);
            g[(z as usize) * self.words + q as usize / 64] |= 1 << (q % 64);
            Self::clifford_bits(self.words, axes, &mut g);
            // W^dag G(g) W, from the old images
            let mut acc = vec![0u64; w];
            for gz in [false, true] {
                for i in 0..self.words {
                    let mut m = g[(gz as usize) * self.words + i];
                    while m != 0 {
                        let b = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let src = (gz as usize) * n + i * 64 + b;
                        for (a, x) in acc.iter_mut().zip(&old[src * w..(src + 1) * w]) {
                            *a ^= x;
                        }
                    }
                }
            }
            Self::clifford_bits(self.words, axes, &mut acc);
            self.rows[r * w..(r + 1) * w].copy_from_slice(&acc);
        }
    }

    /// `W^dag Q W` with the sign dropped, sorted by qubit.
    fn conjugate(&self, ops: &[Operator]) -> Vec<Operator> {
        let w = 2 * self.words;
        let mut acc = vec![0u64; w];
        let mut add = |r: usize| {
            for (a, b) in acc.iter_mut().zip(&self.rows[r * w..(r + 1) * w]) {
                *a ^= b;
            }
        };
        for o in ops {
            let q = o.qubit as usize;
            if o.basis != 'Z' {
                add(q);
            }
            if o.basis != 'X' {
                add(self.n + q);
            }
        }
        self.decode(&acc)
    }

    /// A Pauli's bits as operators sorted by qubit.
    fn decode(&self, acc: &[u64]) -> Vec<Operator> {
        let mut out = Vec::new();
        for i in 0..self.words {
            let mut m = acc[i] | acc[self.words + i];
            while m != 0 {
                let b = m.trailing_zeros();
                m &= m - 1;
                let (x, z) = ((acc[i] >> b) & 1, (acc[self.words + i] >> b) & 1);
                let basis = match (x, z) {
                    (1, 1) => 'Y',
                    (1, 0) => 'X',
                    _ => 'Z',
                };
                out.push(Operator { qubit: (i * 64 + b as usize) as u16, basis });
            }
        }
        out
    }
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
            tab: Tableau::new(n_qubits),
            ver: vec![0; n_qubits],
            cx_cache: std::collections::HashMap::new(),
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
            self.ver[o.qubit as usize] += 1;
            let list = &mut self.on_qubit[o.qubit as usize];
            // Usually the newest there, so this is cheap however long the list.
            if list.last() == Some(&slot) {
                list.pop();
            } else {
                list.retain(|&s| s != slot);
            }
        }
        self.free.push(slot);
        self.open -= 1;
    }

    /// Records the correction a failed T gate executed about `ops` owes. It
    /// combines with the newest correction on these qubits if that one is about
    /// the same Pauli, which is the newest factor of `W`, so the two are adjacent.
    pub(crate) fn add_failure(&mut self, ops: &[Operator]) -> Added {
        let p = sorted(ops);
        // The list below may recombine factors, but always to the same W modulo
        // Paulis, which is W S_P.
        self.tab.apply_s(&p);
        if p.len() == 1 {
            return self.add_single(p[0].qubit, p[0].basis);
        }
        // Each qubit's list is oldest first, so the newest on P's qubits is the
        // newest of their last entries.
        let newest =
            p.iter().filter_map(|o| self.on_qubit[o.qubit as usize].last().copied()).max_by_key(
                |&s| self.get(s).seq,
            );
        if let Some(newest) = newest {
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
            self.ver[o.qubit as usize] += 1;
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
        let mut suffix: Vec<usize> = Vec::new();
        // The qubit's list is oldest first.
        for &s in self.on_qubit[qubit as usize].iter().rev() {
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
    ///
    /// Only the first [`EMISSION_PREFIX`] are returned. Any prefix is a valid
    /// order, since each correction's prerequisites come before it, and a cycle
    /// places far fewer: each emission holds its qubits, which the next in the
    /// order shares. Corrections can pile up into chains thousands long, and
    /// computing all of them every cycle was most of the scheduling time.
    pub(crate) fn emission_order(&self, slot: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.slots.len()];
        self.push_emission(slot, &mut out, &mut seen);
        out
    }

    /// Emits the newer corrections `slot` does not commute with first, newest
    /// first, then `slot`. Newer only, so the recursion cannot come back to a
    /// slot still being processed, and `seen` keeps each to one visit.
    fn push_emission(&self, slot: usize, out: &mut Vec<usize>, seen: &mut [bool]) {
        if seen[slot] || out.len() >= EMISSION_PREFIX {
            return;
        }
        seen[slot] = true;
        let c = self.get(slot);
        // The newer corrections on its qubits it does not commute with. Each
        // qubit's list is oldest first, so only its tail is read.
        let mut newer: Vec<usize> = Vec::new();
        for o in &c.ops {
            for &s in self.on_qubit[o.qubit as usize].iter().rev() {
                let d = self.get(s);
                if d.seq <= c.seq {
                    break;
                }
                if !seen[s] && !commutes(&d.ops, &c.ops) {
                    newer.push(s);
                }
            }
        }
        // Newest first: the newest factor is the first in time.
        newer.sort_by_key(|&s| std::cmp::Reverse(self.get(s).seq));
        newer.dedup();
        for s in newer {
            self.push_emission(s, out, seen);
        }
        // A prerequisite may have been cut off by the prefix; then so is this.
        if out.len() < EMISSION_PREFIX {
            out.push(slot);
        }
    }

    fn blocked(&self, first: usize) -> Pass {
        Pass::Blocked(first)
    }

    /// How to take the owed correction in `slot` out of `W`: the corrections to
    /// emit, in order, each with the operators to emit it about. Every newer one
    /// it does not commute with goes first, each about its own operators (see
    /// [`CorrectionFrame::emission_order`]), so no emitted gate is wider than a
    /// correction already owed.
    ///
    /// A single gate about `E' = R^dag E R`, `R` the newer corrections, would
    /// remove `E` too, but computing `E'` walks most of the frame when many are
    /// owed, and at intermediate weights it is usually past the limit anyway.
    pub(crate) fn emission_plan(&self, slot: usize) -> Vec<(usize, Vec<Operator>)> {
        self.emission_order(slot).into_iter().map(|s| (s, self.get(s).ops.clone())).collect()
    }

    /// A T gate or measurement about `ops`: the operators to run it about, or
    /// the corrections to emit first. The owed corrections are applied oldest
    /// first, each only if it overlaps the product as it stands by then, so one
    /// that widens it brings in the newer corrections on the qubits it gained.
    /// A correction that would take it past `max_weight` (0: no limit) is
    /// emitted instead.
    pub(crate) fn pass_rotation(&self, ops: &[Operator], max_weight: usize) -> Pass {
        let q = self.tab.conjugate(ops);
        debug_assert_eq!(q, self.conjugate_by_list(ops));
        if max_weight == 0 || q.len() <= max_weight {
            return Pass::Run(q);
        }
        // Only the operator it runs about counts against the limit. Emitting a
        // correction E changes that operator only if E anticommutes with it, to
        // E Q'. So emit the one that shrinks it most.
        let mut best: Option<(usize, u64, usize)> = None; // (weight after, seq, slot)
        for (k, o) in q.iter().enumerate() {
            for &s in &self.on_qubit[o.qubit as usize] {
                let e = &self.get(s).ops;
                // Count each correction once, on its first qubit within Q'.
                if q[..k].iter().any(|x| e.binary_search_by_key(&x.qubit, |y| y.qubit).is_ok()) {
                    continue;
                }
                if commutes(e, &q) {
                    continue;
                }
                let w = mul(e, &q).len();
                let key = (w, u64::MAX - self.get(s).seq, s);
                if best.is_none_or(|b| key < b) {
                    best = Some(key);
                }
            }
        }
        let (_, _, s) = best.expect("a correction widened it, so one anticommutes with it");
        self.blocked(s)
    }

    /// Adds the corrections newer than `s` on qubits `r` gained over `q`.
    fn extend_candidates(
        &self, q: &[Operator], r: &[Operator], s: usize, cand: &mut Vec<usize>, i: usize,
    ) {
        let seq = self.get(s).seq;
        let mut grew = false;
        for o in r {
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
    }

    /// `W^dag Q W` from the list of owed corrections, applied oldest first, each
    /// only if it overlaps the product as it stands by then. Slow: the tableau
    /// gives the same, and this checks it.
    fn conjugate_by_list(&self, ops: &[Operator]) -> Vec<Operator> {
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
            self.extend_candidates(&q, &r, s, &mut cand, i);
            q = r;
        }
        q
    }

    /// A Clifford `C` runs as it is, its operators `axes` naming the axis it
    /// fixes on each qubit: Z for S, X for SX, and for a CX Z on its control and
    /// X on its target. It is pushed through the frame instead of waiting for
    /// corrections: `W <- C W C^dag`, every owed `S_P` becoming `S_{C P C^dag}`.
    /// A CX can move a correction onto its other qubit, or off one. Returns how
    /// many owed corrections it changed.
    pub(crate) fn push_clifford(&mut self, axes: &[Operator]) -> usize {
        let axes = sorted(axes);
        let mut on: Vec<usize> = Vec::new();
        for o in &axes {
            on.extend(self.on_qubit[o.qubit as usize].iter().copied());
        }
        if on.is_empty() {
            return 0;
        }
        on.sort_unstable();
        on.dedup();
        self.tab.push_clifford(&axes);
        let words = self.tab.words;
        let mut changed = 0;
        for s in on {
            let old = self.get(s).ops.clone();
            let mut v = self.tab_bits(&old);
            Tableau::clifford_bits(words, &axes, &mut v);
            let new = self.tab.decode(&v);
            if new == old {
                continue;
            }
            changed += 1;
            for o in old.iter().chain(&new) {
                self.ver[o.qubit as usize] += 1;
            }
            // Qubits it left or gained, keeping each qubit's list oldest first.
            let seq = self.get(s).seq;
            for o in &old {
                if new.binary_search_by_key(&o.qubit, |x| x.qubit).is_err() {
                    self.on_qubit[o.qubit as usize].retain(|&t| t != s);
                }
            }
            for o in &new {
                if old.binary_search_by_key(&o.qubit, |x| x.qubit).is_err() {
                    let list = &self.on_qubit[o.qubit as usize];
                    let at = list.partition_point(|&t| self.get(t).seq < seq);
                    self.on_qubit[o.qubit as usize].insert(at, s);
                }
            }
            self.slots[s].as_mut().unwrap().ops = new;
        }
        changed
    }

    /// A CX pushes through the owed corrections it leaves on the same qubits,
    /// but one it would spread onto its other qubit must be emitted first, so
    /// that corrections never grow through Cliffords. Returns the oldest such
    /// correction, or none.
    pub(crate) fn cx_blockers(&mut self, axes: &[Operator]) -> Option<usize> {
        let axes = sorted(axes);
        let (a, b) = (&axes[0], &axes[1]);
        let key = (a.qubit, a.basis, b.qubit, b.basis);
        let vers = (self.ver[a.qubit as usize], self.ver[b.qubit as usize]);
        if let Some(&(va, vb, hit)) = self.cx_cache.get(&key) {
            if (va, vb) == vers {
                return hit;
            }
        }
        // It picks up b's qubit if it anticommutes with a there, and a's if it
        // anticommutes with b: it spreads exactly when it is on one of the two
        // qubits only, and anticommutes with that one's axis.
        let spreads = |s: usize| {
            let p = &self.get(s).ops;
            let (la, lb) = (label(p, a.qubit), label(p, b.qubit));
            (lb == 'I' && la != 'I' && la != a.basis) || (la == 'I' && lb != 'I' && lb != b.basis)
        };
        // Each qubit's list is oldest first: the first that spreads on each,
        // then the oldest of those.
        let hit = axes
            .iter()
            .filter_map(|o| self.on_qubit[o.qubit as usize].iter().copied().find(|&s| spreads(s)))
            .min_by_key(|&s| self.get(s).seq);
        self.cx_cache.insert(key, (vers.0, vers.1, hit));
        hit
    }

    fn tab_bits(&self, ops: &[Operator]) -> Vec<u64> {
        self.tab.bits(ops)
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

    /// Takes a correction out of the frame once its gate has been placed about
    /// its own operators: after a merge, or when every newer one commutes with it.
    pub(crate) fn emitted(&mut self, slot: usize) {
        let ops = self.get(slot).ops.clone();
        self.emitted_as(slot, &ops);
    }

    /// Takes the correction in `slot` out of the frame once an S about `ops`,
    /// from [`CorrectionFrame::emission_plan`], has been placed: `W <- W S^dag`.
    pub(crate) fn emitted_as(&mut self, slot: usize, ops: &[Operator]) {
        self.tab.apply_s(ops);
        self.remove(slot);
    }
}

/// One operation of a circuit, for [`predicted_depth_with_failures`]. Cliffords
/// carry the axes they fix: Z for S, X for SX, Z on a CX's control and X on its
/// target, as in the circuit files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Kind {
    T,
    M,
    S,
    SX,
    CX,
}

/// Logical cycles each operation holds its qubits, as the scheduler charges them.
fn cost(kind: Kind) -> usize {
    match kind {
        Kind::T | Kind::M => 1,
        Kind::CX => 2,
        Kind::S | Kind::SX => 3,
    }
}

/// Predicted depth of a circuit run with T-injection failures: each product is
/// placed at once after the last one on each of its qubits, as in the layering
/// downstream, but T gates fail with probability `p_fail` (seeded), and every operation goes
/// through the correction frame exactly as the scheduler does it: Cliffords
/// push through it, absorbed corrections widen products up to `max_weight`
/// (0: none), and corrections that would widen one past that are emitted first,
/// holding their qubits three cycles. Routing and magic
/// supply are left out, so this is a depth, not a schedule.
pub(crate) fn predicted_depth_with_failures<'a>(
    ops: impl Iterator<Item = (Kind, &'a [Operator])>, n_qubits: usize, max_weight: usize,
    p_fail: f64, seed: u64,
) -> usize {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let mut frame = CorrectionFrame::new(n_qubits);
    let mut depth = vec![0usize; n_qubits];
    let place = |depth: &mut Vec<usize>, ops: &[Operator], cycles: usize| {
        let start = ops.iter().map(|o| depth[o.qubit as usize]).max().unwrap_or(0);
        for o in ops {
            depth[o.qubit as usize] = start + cycles;
        }
    };
    let emit = |frame: &mut CorrectionFrame, depth: &mut Vec<usize>, slot: usize| {
        for (s, ops) in frame.emission_plan(slot) {
            place(depth, &ops, 3);
            frame.emitted_as(s, &ops);
        }
    };
    for (kind, ops) in ops {
        match kind {
            Kind::T | Kind::M => {
                let orig = sorted(ops);
                let limit = if max_weight == 0 { 0 } else { max_weight.max(orig.len()) };
                loop {
                    match frame.pass_rotation(&orig, limit) {
                        Pass::Run(q) => {
                            place(&mut depth, &q, cost(kind));
                            if kind == Kind::T && p_fail > 0.0 && rng.gen_bool(p_fail) {
                                frame.add_failure(&q);
                            }
                            break;
                        }
                        Pass::Blocked(slot) => emit(&mut frame, &mut depth, slot),
                    }
                }
            }
            Kind::S | Kind::SX | Kind::CX => {
                if kind != Kind::CX && !frame.clear_on(ops) {
                    if let Some((slot, net)) = frame.merge_candidate(ops[0].qubit, ops[0].basis) {
                        frame.emitted(slot);
                        if net % 2 == 1 {
                            place(&mut depth, ops, cost(kind));
                        }
                        continue;
                    }
                }
                if kind == Kind::CX {
                    while let Some(slot) = frame.cx_blockers(ops) {
                        emit(&mut frame, &mut depth, slot);
                    }
                }
                frame.push_clifford(ops);
                place(&mut depth, ops, cost(kind));
            }
        }
    }
    depth.into_iter().max().unwrap_or(0)
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
            Pass::Blocked(s) => assert_eq!(f.get(s).ops, ops("ZZ_")),
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
        // S_X then S_Y acts as a three-cycle on the labels: Z goes to Y under S_X
        // and stays there under S_Y; X stays under S_X and goes to Z under S_Y.
        assert_eq!(f.pass_rotation(&ops("Z"), 0), Pass::Run(ops("Y")));
        assert_eq!(f.pass_rotation(&ops("X"), 0), Pass::Run(ops("Z")));
    }

    #[test]
    fn the_same_pauli_twice_is_a_pauli() {
        let mut f = CorrectionFrame::new(2);
        f.add_failure(&ops("ZX"));
        assert_eq!(f.add_failure(&ops("ZX")), Added::Cancelled);
        assert_eq!(f.open(), 0);
    }

    #[test]
    fn a_cx_pushes_corrections_through_it() {
        let mut f = CorrectionFrame::new(3);
        f.add_failure(&ops("Z__"));
        f.add_failure(&ops("_X_"));
        // Z on the control and X on the target commute with the CX.
        assert_eq!(f.push_clifford(&ops("ZX_")), 0);
        f.add_failure(&ops("X_Z"));
        // X on the control picks up X on the target: X_Z becomes XXZ.
        assert_eq!(f.push_clifford(&ops("ZX_")), 1);
        assert!(f.slots.iter().flatten().any(|c| c.ops == ops("XXZ")));
        // Qubit 1 now carries it, oldest first in its list.
        let on1: Vec<Vec<Operator>> =
            f.on_qubit[1].iter().map(|&s| f.get(s).ops.clone()).collect();
        assert_eq!(on1, vec![ops("_X_"), ops("XXZ")]);
        // And conjugation follows: Z on the target becomes Y under _X_, then
        // XXZ times _Y_ is XZZ.
        assert_eq!(f.pass_rotation(&ops("_Z_"), 0), Pass::Run(ops("XZZ")));
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
    ///   2. a Clifford pushed through turns W into C W C^dag;
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
            let emit = |f: &mut CorrectionFrame, hist: &mut M, slot: usize| {
                let cap = if limit == 0 { usize::MAX } else { limit.max(f.get(slot).ops.len()) };
                for (s, e) in f.emission_plan(slot) {
                    // 3. no emitted gate is wider than the limit, and taking it
                    // off leaves the remaining corrections equal to the history.
                    assert!(e.len() <= cap, "seed {}: emitted {:?} past the limit", seed, e);
                    *hist = mm(hist, &dag(&rot(n, &e, quarter)));
                    f.emitted_as(s, &e);
                    assert!(pauli_ratio(&owed(f, n), hist, n), "seed {}: emitting {:?} broke W", seed, e);
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
                    if g.len() == 1 {
                        if let Some((slot, _)) = f.merge_candidate(g[0].qubit, g[0].basis) {
                            // The S runs merged with the correction: W loses it.
                            hist = mm(&hist, &dag(&rot(n, &f.get(slot).ops, quarter)));
                            f.emitted(slot);
                            continue;
                        }
                    }
                    // A CX first sheds what it would spread, then pushes through.
                    if g.len() == 2 {
                        while let Some(slot) = f.cx_blockers(&g) {
                            emit(&mut f, &mut hist, slot);
                        }
                    }
                    // 2. pushed through: W becomes C W C^dag.
                    f.push_clifford(&g);
                    hist = mm(&mm(&cliff, &hist), &dag(&cliff));
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
                        Pass::Blocked(slot) => emit(&mut f, &mut hist, slot),
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
