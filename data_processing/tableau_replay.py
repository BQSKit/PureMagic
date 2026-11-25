#!/usr/bin/env python3
"""Independent end-to-end check of the T-failure corrections in a PureMagic trace.

    puremagic -c circuit.trans ... --trace       # writes <stem>.trace
    python3 tableau_replay.py <stem>.trace [...]  # exit status 1 on any mismatch

Unlike frame.rs and check_schedule.py, this keeps NO list of owed S_P factors
and makes no blocking/absorbing decisions. It tracks the whole owed Clifford W
(ideal = W . executed) as a symplectic tableau, mod Paulis (signs dropped):
img[g] = label(W^dag g W) for each generator g in X_0..X_{n-1}, Z_0..Z_{n-1}.

Events, in schedule order (starts within a cycle, then that cycle's failures):
  T or M about ideal Q, run about L    -> require L == label(W^dag Q W)
  T failure, run about L               -> W <- W S_L
  injected correction about L          -> W <- W S_L^dag   (== W S_L mod Pauli)
  circuit Clifford C run as-is         -> W <- C W C^dag    (push-through; exact)
  circuit S/SX cancelled (no patches)  -> W <- C W          (nothing executed)
Any wrong T/M operator, any correction emitted out of order, any Clifford run
past a correction it does not commute with, shows up as a label mismatch at
a later T or M (or as a residual W at the end).
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_schedule import load_circuit, executed_ops, parse_ops  # noqa: E402  (parsing only)


class Tableau:
    def __init__(self, n):
        self.n = n
        self.img = [1 << g for g in range(2 * n)]  # identity

    # Pauli label <-> int: bit q = x_q, bit n+q = z_q
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

    def symp(self, a, b):
        n, m = self.n, (1 << self.n) - 1
        ax, az, bx, bz = a & m, a >> n, b & m, b >> n
        return bin((ax & bz) ^ (az & bx)).count("1") & 1

    def apply(self, v):  # label(W^dag Q W)
        r, g = 0, 0
        while v:
            if v & 1:
                r ^= self.img[g]
            v >>= 1
            g += 1
        return r

    def transvect(self, v, L):
        return v ^ L if self.symp(v, L) else v

    def left_S(self, L):  # W <- W S_L : img <- T_L(img)
        self.img = [self.transvect(v, L) for v in self.img]

    # G_C(Q) = label(C^dag Q C), an involution mod signs for S, SX, CX
    def G(self, gate, qs, v):
        n = self.n
        if gate in ("S", "Sdg"):
            return self.transvect(v, 1 << (n + qs[0]))
        if gate in ("SX", "SXdg"):
            return self.transvect(v, 1 << qs[0])
        if gate == "CX":
            c, t = qs
            if (v >> c) & 1:  # X_c -> X_c X_t
                v ^= 1 << t
            if (v >> (n + t)) & 1:  # Z_t -> Z_c Z_t
                v ^= 1 << (n + c)
            return v
        raise ValueError(gate)

    def clifford_push(self, gate, qs):  # W <- C W C^dag : F <- G F G
        old = self.img
        new = []
        for g in range(2 * self.n):
            new.append(self.G(gate, qs, self.apply_with(old, self.G(gate, qs, 1 << g))))
        self.img = new

    def clifford_absorb(self, gate, qs):  # W <- C W : F <- F G
        old = self.img
        self.img = [self.apply_with(old, self.G(gate, qs, 1 << g)) for g in range(2 * self.n)]

    @staticmethod
    def apply_with(img, v):
        r, g = 0, 0
        while v:
            if v & 1:
                r ^= img[g]
            v >>= 1
            g += 1
        return r

    def moves(self, gate, qs):  # does C fail to commute with W (mod Pauli)?
        for g in range(2 * self.n):
            a = self.G(gate, qs, self.apply(self.G(gate, qs, 1 << g)))
            if a != self.img[g]:
                return True
        return False

    def residual(self):
        return sum(1 for g in range(2 * self.n) if self.img[g] != 1 << g)


def clifford_qubits(gate, ops):
    if gate == "CX":
        d = dict((b, q) for q, b in ops)
        return (d["Z"], d["X"])
    return (ops[0][0],)


def replay(trace_path):
    data_node, runs, corr, fails = {}, defaultdict(list), {}, defaultdict(list)
    circ_path = n_circ = None
    for line in open(trace_path):
        f = line.split()
        if not f:
            continue
        k = f[0]
        if k == "circuit":
            circ_path = f[1]
        elif k == "products":
            n_circ = int(f[1])
        elif k == "node" and f[2] == "D":
            data_node[(int(f[3][1:-1]), f[3][-1])] = int(f[1])
        elif k == "s":
            terms = () if f[6] == "-" else tuple(int(x) for x in f[6].split(","))
            nodes = () if f[7] == "-" else f[7]
            runs[int(f[2])].append((int(f[1]), f[3], terms, nodes))
        elif k == "c":
            corr[int(f[1])] = parse_ops(f[2])
        elif k == "f":
            fails[int(f[1])].append((int(f[2]), parse_ops(f[3])))
    circuit = load_circuit(circ_path)
    assert len(circuit) == n_circ
    node_qubit = {i: qe for qe, i in data_node.items()}
    n = 1 + max(q for _, ops in circuit for q, _ in ops)
    for ops in corr.values():
        n = max(n, 1 + max(q for q, _ in ops))
    tab = Tableau(n)

    starts = defaultdict(list)
    for pid, rs in runs.items():
        rs.sort()
        starts[rs[0][0]].append(pid)
    cycles = sorted(set(starts) | set(fails))
    stats = defaultdict(int)
    bad = []
    for lc in cycles:
        for pid in starts.get(lc, []):
            rs = runs[pid]
            if pid >= n_circ:
                tab.left_S(tab.enc(corr[pid]))
                stats["corrections"] += 1
                continue
            gate, ops = circuit[pid]
            if gate in ("T", "M"):
                run_about = executed_ops(rs[0][2], node_qubit)
                want = tab.dec(tab.apply(tab.enc(ops)))
                stats[gate] += 1
                if want != sorted(ops):
                    stats[gate + "_conjugated"] += 1
                if run_about != want:
                    bad.append((lc, pid, gate, ops, run_about, want))
            else:
                qs = clifford_qubits(gate, ops)
                cancelled = gate in ("S", "Sdg", "SX", "SXdg") and len(rs) == 1 and not rs[0][3]
                if cancelled:
                    tab.clifford_absorb(gate, qs)
                    stats["clifford_cancelled"] += 1
                else:
                    if tab.moves(gate, qs):
                        stats["clifford_pushed_through_W"] += 1  # changed owed corrections
                    tab.clifford_push(gate, qs)
                    stats["clifford"] += 1
        for pid, L in fails.get(lc, []):
            run_about = executed_ops(runs[pid][0][2], node_qubit)
            if run_about != L:
                bad.append((lc, pid, "f-line", L, run_about, None))
            tab.left_S(tab.enc(L))
            stats["failures"] += 1
    stats["residual_generators_at_end"] = tab.residual()
    return dict(stats), bad


if __name__ == "__main__":
    failed = False
    for p in sys.argv[1:]:
        stats, bad = replay(p)
        print(os.path.basename(p), "MISMATCHES", len(bad), stats)
        for b in bad[:5]:
            print("   ", b)
        failed |= bool(bad)
    sys.exit(1 if failed else 0)
