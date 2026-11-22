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
     data edge. A T gate's or single-qubit measurement's basis may have been
     conjugated by a pending S correction about any axis, so for those only
     the qubits are checked, each with a single Pauli
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

Exit status is 0 when every check passes.
"""

import argparse
import os
import sys
from collections import defaultdict

DURATION = {"T": 1, "M": 1, "CX": 2, "S": 3, "SX": 3}


def load_circuit(path):
    """-> list of (gate, [(qubit, basis)]) in file order, X and Z skipped.

    Mirrors the .trans format: a sign, one character per qubit (_ for
    identity), then <GATE>."""
    prods = []
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        lt = s.index("<")
        gate = s[lt + 1 : s.index(">", lt)]
        if gate in ("X", "Z"):
            continue
        body = s[1:lt]
        ops = [(q, c) for q, c in enumerate(body) if c != "_"]
        prods.append((gate, ops))
    return prods


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
    circuit_path = n_circ = None
    for line in open(args.trace):
        f = line.split()
        if not f:
            continue
        if f[0] == "circuit":
            circuit_path = f[1]
        elif f[0] == "products":
            n_circ = int(f[1])
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
            corr[int(f[1])] = (int(f[2]), f[3])

    circuit = load_circuit(args.circuit or circuit_path)
    if len(circuit) != n_circ:
        sys.exit(f"circuit has {len(circuit)} products, trace says {n_circ}")

    errors = defaultdict(list)

    def err(kind, msg):
        errors[kind].append(msg)

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
        merged = gate in ("S", "SX") and len(rs) == 1 and empty
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
            q, b = corr[pid]
            ops = [(q, b)]
        allowed = []
        for q, b in ops:
            alts = {b}
            if gate == "T" or (gate == "M" and len(ops) == 1):
                # A pending S correction about the failed T's own axis (X, Y
                # or Z) conjugates a T gate's or a single-qubit measurement's
                # basis, so any Pauli may stand here.
                alts = {"X", "Y", "Z"}
            allowed.append(alts)
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
                stack.extend((adj[u] | {v for v in ns if u in adj[v]}) & ns - seen)
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
                prev = [u for u in used_at[magic] if u < lc]
                if prev and prev[-1] >= lc - length:
                    err(
                        "magic",
                        f"T product {pid} at cycle {lc}: {label[magic]} was used at cycle "
                        f"{prev[-1]}, inside the {length}-cycle attempt that made its state",
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
    ]
    for k in kinds:
        e = errors.get(k, [])
        print(f"  {k:13s} {'ok' if not e else f'{len(e)} violations'}")
        for m in e[: args.max_errors]:
            print(f"      {m}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
