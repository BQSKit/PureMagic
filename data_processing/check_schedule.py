#!/usr/bin/env python3
"""Check a PureMagic schedule trace against its circuit, without the scheduler.

    puremagic -c circuit.trans ... --trace       # writes <stem>.trace
    python3 check_schedule.py <stem>.trace [--max-errors N]

The checker reads only the trace (the layout and each product's patches per
logical cycle) and the circuit file it names, and rebuilds the dependencies
itself. It checks:

  1. exclusivity: no patch serves two products in one logical cycle
  2. terminals: every product touches exactly the data-patch edges its
     operators name (X edge for X, Z edge for Z, both for Y) and no other
     data edge. T gates and measurements run about their operators
     conjugated by owed corrections, possibly widened onto other qubits up to
     the weight limit, which check 9 verifies
  3. connectivity: each product's patches form one connected piece of the
     layout
  4. duration: T and M take one cycle, CX two consecutive cycles, S and SX
     three, on the same patches throughout, except S and SX products a
     same-axis correction cancelled, which take one cycle and no patches
  5. dependencies: every product starts after each earlier product sharing
     a qubit has finished, in the circuit file's order (X and Z gates are
     frame updates and skipped)
  6. magic states: every T gate takes its state from a magic patch at the
     root of its path, and that patch was used by no path during the
     cultivation attempt that produced the state, so no state is consumed
     twice or produced while the patch was busy
  7. completeness: every product in the circuit is scheduled, T gates once
  8. qubits: no two products that touch a qubit overlap in time. Circuit
     products are ordered by check 5, but injected corrections and products
     the frame widened have no place in the circuit's dependencies there, and
     hold only the patch edges they route through
  9. frame: replaying the corrections failed T gates owe (the trace's f
     lines), each S about all of the failed gate's operators, independently
     of the scheduler: every T gate and measurement runs about its operators
     conjugated by the owed corrections, widened no further than the weight
     limit the trace records, every Clifford is pushed through the owed
     corrections, turning each S_P into S_(C P C^dag) (a lone single-qubit
     correction about an S's own axis merges into it instead) without a CX
     spreading any onto its other qubit, and every injected correction is an S
     about some owed correction conjugated past the newer ones, which removes
     that one exactly, and no wider than the weight limit or that correction

Exit status is 0 when every check passes.
"""

import argparse
import bisect
import os
import sys
from collections import defaultdict

# CHECK_FRAME_BOTH=1: conjugate every T gate and measurement both by the
# tableau and by the owed list, and fail on any difference (slow).
BOTH = os.environ.get("CHECK_FRAME_BOTH") == "1"

DURATION = {"T": 1, "M": 1, "CX": 2, "S": 3, "Sdg": 3, "SX": 3, "SXdg": 3}
S_AXIS = {"S": "Z", "Sdg": "Z", "SX": "X", "SXdg": "X"}  # the axis each fixes


def load_circuit(path):
    """-> list of (gate, [(qubit, basis)]) in file order, X and Z skipped.

    Mirrors the .trans format: a sign, one character per qubit (_ for
    identity), then <GATE>."""
    prods = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lt = s.index("<")
        gate = s[lt + 1 : s.index(">", lt)]
        if gate in ("X", "Z"):
            continue
        body = s[1:lt]
        ops = [(q, c) for q, c in enumerate(body) if c != "_"]
        prods.append((gate, ops))
    return prods


def parse_ops(text):
    """'3:Z,5:X' -> [(3, 'Z'), (5, 'X')], sorted by qubit."""
    return sorted((int(q), b) for q, b in (t.split(":") for t in text.split(",")))


def executed_ops(terms, node_qubit):
    """The Pauli a product's terminal edges name: X edge X, Z edge Z, both Y."""
    edges = defaultdict(set)
    for n in terms:
        if n not in node_qubit:
            return None
        q, e = node_qubit[n]
        edges[q].add(e)
    return sorted((q, "Y" if len(e) == 2 else next(iter(e))) for q, e in edges.items())


# --- the correction frame, reimplemented from its definition -----------------
# W, the product of owed S_P, satisfies U = W V for the ideal circuit U and the
# executed one V. A rotation about Q then runs about W^dag Q W, so Q is
# conjugated by the oldest owed correction first. Signs are dropped.


def commutes(a, b):
    db = dict(b)
    return sum(1 for q, x in a if q in db and db[q] != x) % 2 == 0


def mul(a, b):
    """P Q over the union of their qubits, sign dropped, sorted by qubit."""
    prod = {("X", "Y"): "Z", ("Y", "X"): "Z", ("Y", "Z"): "X",
            ("Z", "Y"): "X", ("Z", "X"): "Y", ("X", "Z"): "Y"}
    da, db = dict(a), dict(b)
    out = []
    for q in sorted(set(da) | set(db)):
        x, y = da.get(q), db.get(q)
        c = x if y is None else y if x is None else (None if x == y else prod[(x, y)])
        if c:
            out.append((q, c))
    return out


class Tableau:
    """W modulo Paulis as the images of the generators: img[q] = W^dag X_q W and
    img[n + q] = W^dag Z_q W, each a Pauli as an int (bit q its X part, bit
    n + q its Z part). Conjugating a Pauli by W is then an XOR of rows. The
    owed list below defines W; this only answers W^dag Q W quickly, and
    CHECK_FRAME_BOTH=1 checks every answer against the list.

    While no owed correction acts on more than one qubit, W is a product of
    single-qubit Cliffords, the image of X_q and Z_q lies on q alone, and an
    update about some qubits changes only their rows (`local`)."""

    def __init__(self, n):
        self.n = n
        self.img = [1 << g for g in range(2 * n)]

    def enc(self, ops):
        v = 0
        for q, b in ops:
            if b in "XY":
                v |= 1 << q
            if b in "ZY":
                v |= 1 << (self.n + q)
        return v

    def dec(self, v):
        n, out = self.n, []
        for q in range(n):
            x, z = (v >> q) & 1, (v >> (n + q)) & 1
            if x or z:
                out.append((q, "Y" if x and z else ("X" if x else "Z")))
        return out

    def swap(self, v):  # X and Z parts exchanged: symp(a, b) = parity(a & swap(b))
        m = (1 << self.n) - 1
        return ((v & m) << self.n) | (v >> self.n)

    def rows(self, ops, local):
        if local:
            return [r for q, _ in ops for r in (q, self.n + q)]
        return range(2 * self.n)

    def conjugate(self, ops):  # W^dag Q W, sorted by qubit
        v = 0
        for q, b in ops:
            if b in "XY":
                v ^= self.img[q]
            if b in "ZY":
                v ^= self.img[self.n + q]
        return self.dec(v)

    def apply_s(self, ops, local):  # W <- W S_L (or S_L^dag, the same mod Paulis)
        L = self.enc(ops)
        Ls = self.swap(L)
        img = self.img
        for r in self.rows(ops, local):
            if bin(img[r] & Ls).count("1") & 1:
                img[r] ^= L

    def g(self, axes, v):
        """C^dag v C for the Clifford fixing `axes` (S about Z, SX about X, a CX
        as Z on its control and X on its target), signs dropped; an involution."""
        if len(axes) == 1:
            a = self.enc(axes)
            return v ^ a if bin(v & self.swap(a)).count("1") & 1 else v
        a, b = self.enc(axes[:1]), self.enc(axes[1:])
        out = v
        if bin(v & self.swap(a)).count("1") & 1:
            out ^= b
        if bin(v & self.swap(b)).count("1") & 1:
            out ^= a
        return out

    def push(self, axes, local):  # W <- C W C^dag: img'(g) = G(W^dag G(g) W)
        n, img = self.n, self.img
        on = {q for q, _ in axes}
        old = {}
        for q in on:
            old[q], old[n + q] = img[q], img[n + q]
        rows = [r for q in on for r in (q, n + q)] if local else range(2 * n)
        new = {}
        for r in rows:
            if r % n in on:
                gg, acc = self.g(axes, 1 << r), 0
                for k in range(2 * n):
                    if (gg >> k) & 1:
                        acc ^= old[k] if k in old else img[k]
                new[r] = self.g(axes, acc)
            else:
                new[r] = self.g(axes, img[r])
        for r, v in new.items():
            img[r] = v


class Frame:
    def __init__(self, n):
        self.owed = []  # [ops, power, seq], oldest first
        self.seq = 0
        self.tab = Tableau(n)
        self.n_multi = 0  # owed corrections on more than one qubit

    def local(self):
        return self.n_multi == 0

    def _append(self, ops):
        self.owed.append([ops, 1, self.seq])
        self.seq += 1
        self.n_multi += len(ops) > 1

    def _remove(self, c):
        self.owed.remove(c)
        self.n_multi -= len(c[0]) > 1

    def on(self, qs):
        return [c for c in self.owed if any(q in qs for q, _ in c[0])]

    def newest_on(self, qs):
        """The newest owed correction on any of these qubits, or None."""
        for c in reversed(self.owed):
            if any(q in qs for q, _ in c[0]):
                return c
        return None

    def fail(self, ops):
        self.tab.apply_s(ops, self.local())
        if len(ops) == 1:
            return self.fail_single(*ops[0])
        c = self.newest_on({q for q, _ in ops})
        if c is not None and c[0] == ops:
            c[1] = (c[1] + 1) % 4
            if c[1] % 2 == 0:
                self._remove(c)
            return
        self._append(ops)

    # Single-qubit Cliffords on one qubit compose. Modulo Paulis each is a
    # permutation of the labels X, Y, Z (conjugation, signs dropped), and is at
    # most two S gates. The newest single-qubit corrections on a qubit, those
    # after any multi-qubit one there, are kept composed in that form.
    TAU = {"X": "XZY", "Y": "ZYX", "Z": "YXZ"}  # image of X, Y, Z under S_axis

    @staticmethod
    def act(perm, label):
        return perm["XYZ".index(label)]

    def fail_single(self, q, axis):
        suffix = []
        for c in reversed(self.owed):
            if all(x != q for x, _ in c[0]):
                continue
            if len(c[0]) != 1:
                break
            suffix.insert(0, c)
        perm = "XYZ"
        for c in suffix + [[[(q, axis)], 1, None]]:
            t = self.TAU[c[0][0][1]]
            perm = "".join(self.act(t, x) for x in perm)  # this factor after the earlier ones
        for c in suffix:
            self._remove(c)
        for a in self.factors(perm):
            self._append([(q, a)])

    def factors(self, perm):
        if perm == "XYZ":
            return []
        for a in "ZXY":
            if perm == self.TAU[a]:
                return [a]
        for a, b in (("X", "Z"), ("Z", "X"), ("X", "Y"), ("Y", "X"), ("Z", "Y"), ("Y", "Z")):
            if "".join(self.act(self.TAU[b], self.act(self.TAU[a], x)) for x in "XYZ") == perm:
                return [a, b]
        raise AssertionError(perm)

    def rotation(self, ops, max_weight):
        """-> operators to run about: W^dag Q W. None if the result passes the
        weight limit (0: none), when a correction must go first."""
        q = self.tab.conjugate(ops)
        if BOTH:
            assert q == sorted(self.rotation_by_list(ops)), (ops, q, self.rotation_by_list(ops))
        if max_weight and len(q) > max_weight:
            return None
        return q

    def rotation_by_list(self, ops):
        """W^dag Q W from its definition: every owed correction, oldest first."""
        q = list(ops)
        for c in self.owed:
            if not commutes(c[0], q):
                q = mul(c[0], q)
        return q

    def emitted(self, i, ops):
        """An S about `ops` = R^dag E R removed the i-th owed correction E."""
        self.tab.apply_s(ops, self.local())
        self._remove(self.owed[i])

    def merged(self, c):
        """A lone single-qubit correction merged into an S or SX about its axis."""
        self.tab.apply_s(c[0], self.local())
        self._remove(c)

    def conjugate_past_newer(self, i):
        """R^dag E R for E the i-th owed correction and R the newer ones."""
        q = self.owed[i][0]
        for d in self.owed[i + 1:]:
            if not commutes(d[0], q):
                q = mul(d[0], q)
        return q

    def push_clifford(self, gate, ops):
        """A Clifford runs as it is and W becomes C W C^dag: each owed S_P
        becomes S_(C P C^dag). On one qubit an S about the axis A takes P to
        P A if they anticommute. On two, exp(i pi/4 (I - A)(I - B)), a CX for
        A = Z on its control and B = X on its target, takes P to P B if it
        anticommutes with A and to P A if it anticommutes with B."""
        self.tab.push(sorted(ops), self.local())
        axes = [[(q, b)] for q, b in ops]
        for c in self.on({q for q, _ in ops}):
            p = c[0]
            if len(axes) == 1:
                if not commutes(p, axes[0]):
                    p = mul(axes[0], p)
            else:
                a, b = axes
                anti_a, anti_b = not commutes(p, a), not commutes(p, b)
                if anti_a:
                    p = mul(b, p)
                if anti_b:
                    p = mul(a, p)
            self.n_multi += (len(p) > 1) - (len(c[0]) > 1)
            c[0] = p


def replay_frame(runs, start, circuit, n_circ, corr, corr_seq, fails, node_qubit, err,
                 max_weight):
    n = 1 + max([q for _, ops in circuit for q, _ in ops]
                + [q for ops in corr.values() for q, _ in ops] + [0])
    frame = Frame(n)
    by_cycle = defaultdict(list)
    for pid, rs in runs.items():
        by_cycle[start[pid]].append(pid)
    for lc in sorted(set(by_cycle) | set(fails)):
        starts = by_cycle.get(lc, [])
        # Corrections first: none overlaps a product starting in the same cycle
        # (check 8), so the order among disjoint events does not matter.
        for pid in sorted(starts, key=lambda p: p < n_circ):
            if pid >= n_circ:
                # An S about E' = R^dag E R removes the owed E, R being the newer
                # ones: W = A S_E R = A R S_E'. It is never wider than the weight
                # limit or E itself.
                ops = corr[pid]
                seq = corr_seq.get(pid)
                hit = next((i for i, c in enumerate(frame.owed) if c[2] == seq), None)
                if hit is None:
                    err("frame", f"cycle {lc}: correction {pid} names no owed correction")
                    continue
                if frame.conjugate_past_newer(hit) != ops:
                    err("frame", f"cycle {lc}: correction {pid} {ops} is not owed correction "
                        f"{seq} {frame.owed[hit][0]} past the newer ones, "
                        f"{frame.conjugate_past_newer(hit)}")
                    continue
                cap = max(max_weight, len(frame.owed[hit][0])) if max_weight else None
                if cap is not None and len(ops) > cap:
                    err("frame", f"cycle {lc}: correction {pid} {ops} is wider than {cap}")
                frame.emitted(hit, ops)
                continue
            gate, ops = circuit[pid]
            if gate in ("T", "M"):
                # a product may run as wide as written: the transpiler's limit
                # does not bound measurements
                want = frame.rotation(sorted(ops), max_weight and max(max_weight, len(ops)))
                ran = executed_ops(runs[pid][0][4], node_qubit)
                if want is None:
                    err("frame", f"cycle {lc}: {gate} {pid} ran while the owed corrections "
                        f"would take it past weight {max_weight}")
                elif ran != sorted(want):
                    err("frame", f"cycle {lc}: {gate} {pid} ran about {ran}, frame says {want}")
            elif gate in S_AXIS:
                q, axis = ops[0][0], S_AXIS[gate]
                on = frame.on({q})
                if len(on) == 1 and on[0][0] == [(q, axis)]:
                    frame.merged(on[0])
                else:
                    frame.push_clifford(gate, [(q, axis)])
            elif gate == "CX":
                qs = {q for q, _ in ops}
                before = {id(c): {q for q, _ in c[0]} for c in frame.on(qs)}
                frame.push_clifford(gate, sorted(ops))
                for c in frame.on(qs):
                    if id(c) in before and not {q for q, _ in c[0]} <= before[id(c)]:
                        err("frame", f"cycle {lc}: CX {pid} spread an owed correction to {c[0]}")
        for pid, ops in fails.get(lc, []):
            ran = executed_ops(runs[pid][0][4], node_qubit) if pid in runs else None
            if ran != ops:
                err("frame", f"cycle {lc}: failed T {pid} recorded {ops}, ran about {ran}")
            frame.fail(ops)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("trace")
    ap.add_argument("--circuit", help="override the circuit path recorded in the trace")
    ap.add_argument("--max-errors", type=int, default=20)
    args = ap.parse_args()

    ntype, label, adj = {}, {}, {}
    data_node = {}
    occ = defaultdict(dict)  # lcycle -> node -> product
    runs = defaultdict(list)  # product -> [(lcycle, gate, magic, length, terms, nodes)]
    corr = {}
    corr_seq = {}  # injected correction -> creation number of the owed one it removed
    fails = defaultdict(list)  # lcycle -> [(T product, operators it ran about)]
    circuit_path = n_circ = None
    max_weight = 0
    for line in open(args.trace):
        f = line.split()
        if not f:
            continue
        if f[0] == "circuit":
            circuit_path = f[1]
        elif f[0] == "products":
            n_circ = int(f[1])
        elif f[0] == "max-weight":
            max_weight = int(f[1])
        elif f[0] == "node":
            i = int(f[1])
            ntype[i], label[i] = f[2], f[3]
            if f[2] == "D":
                data_node[(int(f[3][1:-1]), f[3][-1])] = i
        elif f[0] == "adj":
            adj[int(f[1])] = (
                set(int(x) for x in f[2].split(",")) if len(f) > 2 and f[2] else set()
            )
        elif f[0] == "s":
            lc, pid, gate = int(f[1]), int(f[2]), f[3]
            magic = None if f[4] == "-" else int(f[4])
            length = None if f[5] == "-" else int(f[5])
            terms = () if f[6] == "-" else tuple(int(x) for x in f[6].split(","))
            nodes = () if f[7] == "-" else tuple(int(x) for x in f[7].split(","))
            runs[pid].append((lc, gate, magic, length, terms, nodes))
        elif f[0] == "c":
            corr[int(f[1])] = parse_ops(f[2])
            if len(f) > 3:
                corr_seq[int(f[1])] = int(f[3])
        elif f[0] == "f":
            fails[int(f[1])].append((int(f[2]), parse_ops(f[3])))

    circuit = load_circuit(args.circuit or circuit_path)
    if len(circuit) != n_circ:
        sys.exit(f"circuit has {len(circuit)} products, trace says {n_circ}")

    errors = defaultdict(list)

    def err(kind, msg):
        errors[kind].append(msg)

    # The layout's edges in both directions, for the connectivity check.
    both = defaultdict(set)
    for u, vs in adj.items():
        both[u] |= vs
        for v in vs:
            both[v].add(u)

    # 1. exclusivity, and per-node use cycles for the magic check
    used_at = defaultdict(list)
    for pid, rs in runs.items():
        for lc, gate, magic, length, terms, nodes in rs:
            for n in nodes:
                other = occ[lc].get(n)
                if other is not None and other != pid:
                    err(
                        "exclusivity",
                        f"cycle {lc}: {label[n]} used by products {other} and {pid}",
                    )
                occ[lc][n] = pid
                used_at[n].append(lc)
    for n in used_at:
        used_at[n] = sorted(set(used_at[n]))
    node_qubit = {i: qe for qe, i in data_node.items()}

    start, end = {}, {}
    for pid, rs in runs.items():
        rs.sort()
        lcs = [r[0] for r in rs]
        gate = rs[0][1]
        start[pid], end[pid] = lcs[0], lcs[-1]
        empty = all(not r[5] for r in rs)
        # 4. duration and consistent footprint
        want = DURATION.get(gate)
        # an S or SX that a same-axis correction cancelled: one cycle, no patches
        merged = gate in S_AXIS and len(rs) == 1 and empty
        if want is not None and not merged:
            if lcs != list(range(lcs[0], lcs[0] + want)):
                err(
                    "duration",
                    f"product {pid} ({gate}) in cycles {lcs}, expected {want} consecutive",
                )
            if len({r[5] for r in rs}) != 1 or empty:
                err(
                    "duration",
                    f"product {pid} ({gate}) does not hold the same patches every cycle: "
                    f"{[len(r[5]) for r in rs]}",
                )
        # 2. terminals
        if pid < n_circ:
            ops = circuit[pid][1]
        else:
            ops = corr[pid]
        if gate in ("T", "M"):
            # Run about frame-conjugated operators, possibly widened onto other
            # qubits, checked in 9: here only that its edges name one Pauli
            # per qubit, within the weight limit.
            ops = executed_ops(rs[0][4], node_qubit)
            limit = max_weight and max(max_weight, len(circuit[pid][1]))
            if ops is None or (limit and len(ops) > limit):
                err("terminals", f"product {pid}: terminals off the layout or past "
                    f"weight {max_weight}")
                ops = circuit[pid][1]
        allowed = [{b} for _, b in ops]
        for lc, g, magic, length, terms, nodes in rs:
            if not nodes:
                continue
            data = {n for n in nodes if ntype[n] == "D"}
            if not set(terms) <= set(nodes):
                err("terminals", f"product {pid} cycle {lc}: terminals not in its path")
            if data - set(terms):
                err(
                    "terminals",
                    f"product {pid} cycle {lc}: touches data edges "
                    f"{sorted(label[n] for n in data - set(terms))} beyond its terminals",
                )
            ok = True
            covered = set()
            for (q, b), alts in zip(ops, allowed):
                on_q = {
                    n for n in terms if n in (data_node[(q, "X")], data_node[(q, "Z")])
                }
                covered |= on_q
                fits = [
                    (
                        {data_node[(q, "X")], data_node[(q, "Z")]}
                        if a == "Y"
                        else {data_node[(q, a)]}
                    )
                    for a in alts
                ]
                ok &= on_q in fits
            if not ok or covered != set(terms):
                err(
                    "terminals",
                    f"product {pid} cycle {lc}: terminals "
                    f"{sorted(label[n] for n in terms)} do not match operators {ops}",
                )
            # 3. connectivity
            ns = set(nodes)
            seen, stack = set(), [next(iter(ns))]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                stack.extend(both[u] & ns - seen)
            if seen != ns:
                err(
                    "connectivity",
                    f"product {pid} cycle {lc}: {len(ns) - len(seen)} of {len(ns)} "
                    f"patches not connected",
                )
        # 6. magic states
        if gate == "T":
            lc, _, magic, length, _, nodes = rs[0]
            if magic is None or ntype.get(magic) != "M" or magic not in nodes:
                err("magic", f"T product {pid}: no magic patch at its root")
            elif length is None or length < 1:
                err(
                    "magic",
                    f"T product {pid}: consumed a state with attempt length {length}",
                )
            else:
                # the last use before this cycle (the list is sorted)
                uses = used_at[magic]
                i = bisect.bisect_left(uses, lc)
                if i and uses[i - 1] >= lc - length:
                    err(
                        "magic",
                        f"T product {pid} at cycle {lc}: {label[magic]} was used at cycle "
                        f"{uses[i - 1]}, inside the {length}-cycle attempt that made its state",
                    )

    # 5. dependencies, rebuilt from the circuit: each product depends on the
    # previous product touching each of its qubits
    last = {}
    n_dep = 0
    for pid, (gate, ops) in enumerate(circuit):
        for q, _ in ops:
            p = last.get(q)
            if p is not None and p in end and pid in start:
                n_dep += 1
                if start[pid] <= end[p]:
                    err(
                        "dependencies",
                        f"product {pid} starts at cycle {start[pid]} before product "
                        f"{p} on qubit {q} finishes at {end[p]}",
                    )
            last[q] = pid

    # 8. qubits: per qubit, the products touching it never overlap in time
    on_qubit = defaultdict(list)
    for pid in runs:
        if pid >= n_circ:
            qs = corr[pid]
        elif circuit[pid][0] in ("T", "M"):
            qs = executed_ops(runs[pid][0][4], node_qubit) or circuit[pid][1]
        else:
            qs = circuit[pid][1]
        for q, _ in qs:
            on_qubit[q].append((start[pid], end[pid], pid))
    for q, iv in on_qubit.items():
        iv.sort()
        for (s0, e0, p0), (s1, e1, p1) in zip(iv, iv[1:]):
            if s1 <= e0:
                err("qubits", f"qubit {q}: products {p0} ({s0}-{e0}) and {p1} ({s1}-{e1}) overlap")

    # 9. frame
    replay_frame(runs, start, circuit, n_circ, corr, corr_seq, fails, node_qubit, err,
                 max_weight)

    # 7. completeness
    missing = [p for p in range(n_circ) if p not in runs]
    if missing:
        err(
            "completeness",
            f"{len(missing)} products never scheduled, e.g. {missing[:5]}",
        )
    for pid in range(n_circ):
        if circuit[pid][0] == "T" and pid in runs and len(runs[pid]) != 1:
            err("completeness", f"T product {pid} scheduled {len(runs[pid])} times")

    n_t = sum(1 for g, _ in circuit if g == "T")
    stem = os.path.basename(args.trace)
    print(
        f"{stem}: {n_circ} products ({n_t} T), {len(corr)} corrections, "
        f"{len(occ)} cycles, {n_dep} dependencies checked"
    )
    kinds = [
        "exclusivity",
        "terminals",
        "connectivity",
        "duration",
        "dependencies",
        "magic",
        "completeness",
        "qubits",
        "frame",
    ]
    for k in kinds:
        e = errors.get(k, [])
        print(f"  {k:13s} {'ok' if not e else f'{len(e)} violations'}")
        for m in e[: args.max_errors]:
            print(f"      {m}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
