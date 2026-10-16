#!/usr/bin/env python3
"""Derive cultivation parameters from Gidney et al.'s published simulation data.

Supersedes the hand-transcribed survival curves in mcmc_cultivation.py for
everything those curves were being used to estimate. Reads `stats.csv` from the
cultivation paper's Zenodo record (doi:10.5281/zenodo.13777072, CC-BY-4.0),
which is standard Sinter output:

    shots, errors, discards, seconds, decoder, strong_id, json_metadata,
    custom_counts

Three things come out of it directly.

*The floor.* `json_metadata` gives `r`, the number of rounds in one attempt of
the end-to-end protocol. A cultivation attempt that fails post-selection
restarts from zero, so the cheapest possible outcome is one attempt that
survives every check: `r` rounds. That is the irreducible cost a restart pays
again, i.e. the floor for `puremagic --cultivation-floor`.

*The gap/fidelity curve.* `custom_counts` holds `C<g>` and `E<g>`: the number
of accepted shots whose complementary gap landed in bin `g`, and how many of
those carried a logical error. Suffix-summing from a cutoff `G` gives both the
acceptance rate and the logical error rate of the states a scheduler using that
cutoff would consume.

*The rate/fidelity trade.* Mean cultivation time scales as 1/p_accept(G),
because the cutoff is applied once at the end and does not change the
per-round discarding that precedes it. Anchoring that at one known operating
point gives mean cultivation time at any cutoff. The anchor is cross-checked
against independently published timings below.

What this script does NOT derive is the distribution's *shape* between the
floor and the mean, which needs the per-round survival curve behind Figure 15.
That curve is not in stats.csv; `make_lifetime_plot.py` in the record's
code.zip regenerates it, at the cost of re-running their Stim simulation.
mcmc_cultivation.py's transcribed approximation of it remains the only
estimate we have, and `--check-shape` reports what it implies.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATS = os.path.join(HERE, os.pardir, "data", "cultivation", "stats.csv")

# Independently published mean cultivation times, in logical cycles, for the
# d1=5 d2=15 p=1e-3 configuration at three gap cutoffs. Used only to check the
# 1/p_accept scaling, never as an input.
PUBLISHED_TIMINGS = {53: 6.2, 104: 26.0, 116: 59.7}


def load_row(path, construction, d1, d2, p):
    with open(path) as f:
        # The header is space-padded, so the field names need stripping.
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        m = json.loads(r["json_metadata"])
        if (m.get("c") == construction and m.get("d1") == d1
                and m.get("d2") == d2 and m.get("p") == p):
            return r, m
    sys.exit(f"no row for c={construction} d1={d1} d2={d2} p={p} in {path}")


def gap_table(row):
    """-> (shots, {gap: (accepted_at_or_above, errors_at_or_above)})"""
    cc = json.loads(row["custom_counts"])
    shots = int(row["shots"])
    C = {int(k[1:]): v for k, v in cc.items() if k.startswith("C")}
    E = {int(k[1:]): v for k, v in cc.items() if k.startswith("E")}
    suffix, acc_c, acc_e = {}, 0, 0
    for g in sorted(C, reverse=True):
        acc_c += C.get(g, 0)
        acc_e += E.get(g, 0)
        suffix[g] = (acc_c, acc_e)
    return shots, suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--stats", default=DEFAULT_STATS)
    ap.add_argument("-d", "--distance", type=int, default=17,
                    help="main code distance: one logical cycle is d rounds")
    ap.add_argument("--d1", type=int, default=5, help="cultivation distance")
    ap.add_argument("--d2", type=int, default=15, help="escape distance")
    ap.add_argument("-p", "--noise", type=float, default=0.001)
    ap.add_argument("-G", "--cutoff", type=int, default=104,
                    help="operating gap cutoff; 104 is the value used by "
                         "Gidney et al. and by this paper")
    ap.add_argument("--anchor-cycles", type=float, default=26.0,
                    help="mean cultivation time in logical cycles at --cutoff, "
                         "used to anchor the 1/p_accept scaling")
    ap.add_argument("--construction", default="end2end-inplace-distillation")
    ap.add_argument("--check-shape", action="store_true",
                    help="also report what mcmc_cultivation.py's transcribed "
                         "per-round curve implies, as a cross-check")
    args = ap.parse_args()

    row, meta = load_row(args.stats, args.construction, args.d1, args.d2, args.noise)
    shots, suffix = gap_table(row)
    d, R = args.distance, meta["r"]
    discards = int(row["discards"])

    def p_accept(G):
        ge = [g for g in suffix if g >= G]
        return suffix[min(ge)][0] / shots if ge else 0.0

    def ler(G):
        ge = [g for g in suffix if g >= G]
        if not ge:
            return float("nan")
        c, e = suffix[min(ge)]
        return e / c if c else float("nan")

    G = args.cutoff
    print(f"source   {os.path.relpath(args.stats)}  (Zenodo 10.5281/zenodo.13777072)")
    print(f"config   {json.dumps(meta)}")
    print(f"shots    {shots:,}   discards {discards:,}   "
          f"survive-to-gap {100 * (shots - discards) / shots:.2f}%")
    print()
    print(f"at the operating cutoff G={G}:")
    print(f"  acceptance rate      {p_accept(G):>12.4%}   "
          f"(discard rate {1 - p_accept(G):.3%})")
    print(f"  logical error rate   {ler(G):>12.3e}")
    print(f"  attempts per state   {1 / p_accept(G):>12.1f}")
    print()
    print(f"  floor: one attempt   {R:>12} rounds")
    print(f"                       {R / d:>12.2f} logical cycles at d={d}")
    print(f"  mean  (anchor)       {args.anchor_cycles:>12.2f} logical cycles")
    print(f"  floor / mean         {R / d / args.anchor_cycles:>12.1%}"
          f"   <- F/<t> for --cultivation-floor")
    print(f"  --cultivation-floor  {R / d:>12.2f}")
    print()

    print("rate/fidelity trade, mean scaled as 1/p_accept(G) and anchored at "
          f"G={G}:")
    hdr = f"  {'G':>4} {'p_accept':>11} {'LER':>11} {'mean lcyc':>10}"
    if PUBLISHED_TIMINGS:
        hdr += f" {'published':>10} {'diff':>6}"
    print(hdr)
    anchor_p = p_accept(G)
    for g in sorted(set(list(range(0, max(suffix) + 1, 10)) + list(PUBLISHED_TIMINGS) + [G])):
        pa = p_accept(g)
        if pa <= 0:
            continue
        mean = args.anchor_cycles * anchor_p / pa
        line = f"  {g:>4} {pa:>11.3e} {ler(g):>11.3e} {mean:>10.1f}"
        if g in PUBLISHED_TIMINGS:
            pub = PUBLISHED_TIMINGS[g]
            line += f" {pub:>10.1f} {abs(mean - pub) / pub:>5.0%}"
        print(line)
    print()
    print("  The 'published' column is an independent timing model; agreement")
    print("  across it validates both the anchor and the 1/p_accept scaling.")

    if args.check_shape:
        print()
        print("shape cross-check from mcmc_cultivation.py's transcribed curve")
        print("  (approximate: those arrays are read off Figure 15 by hand)")
        src = open(os.path.join(HERE, "mcmc_cultivation.py")).read()
        ns = {}
        exec(src[: src.index("# Report end-to-end survival")], ns)  # noqa: S102
        cum = ns[f"per_cycle_survival_d{args.d1}"]
        print(f"  transcribed attempt length   {len(cum):>6} rounds  (data: {R})")
        print(f"  transcribed end-to-end surv. {cum[-1]:>6.4f}        "
              f"(data: {p_accept(G):.4f} at G={G})")


if __name__ == "__main__":
    main()
