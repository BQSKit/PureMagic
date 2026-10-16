#!/usr/bin/env python3
"""Derive the fixed per-attempt cultivation cost (the "floor") from the
survival-curve model in mcmc_cultivation.py.

`puremagic --cultivation-floor F` models cultivation as a shifted exponential
to test how much of PureMagic's advantage depends on the distribution being
memoryless. That sweep is only interpretable once we know where the *real*
cultivation protocol sits on the F axis, and this script computes it.

The retry structure makes the answer immediate. A cultivation attempt runs
round by round; post-selection failure discards it and a fresh attempt starts
from zero. The cheapest possible outcome is therefore one attempt that
survives every round, so the minimum of the time distribution is exactly the
length of one full attempt -- which is what a restart has to pay again, i.e.
the floor. Everything above that minimum is retries, which are memoryless to a
good approximation.

Reported in units of logical cycles at code distance `-d`, matching
`--cultivation-floor`.
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_survival_curves():
    """Lift the survival arrays out of mcmc_cultivation.py without importing
    it -- importing would run its own simulation and write a figure."""
    src = open(os.path.join(HERE, "mcmc_cultivation.py")).read()
    ns = {}
    # Everything up to the reporting section is pure data plus the
    # cumulative->conditional helper.
    exec(src[: src.index("# Report end-to-end survival")], ns)  # noqa: S102
    return ns


def simulate(conditional, n, rng):
    """Total rounds until one attempt survives every round, n times over."""
    probs = np.asarray(conditional)
    times = np.empty(n, dtype=np.int64)
    for i in range(n):
        total = 0
        while True:
            survived = True
            for p in probs:
                total += 1
                if rng.random() > p:
                    survived = False
                    break
            if survived:
                break
        times[i] = total
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--distance", type=int, default=17,
                    help="code distance: one logical cycle is d rounds")
    ap.add_argument("-n", "--samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--curve", default="d5", choices=["d3", "d5"],
                    help="which survival curve from mcmc_cultivation.py")
    args = ap.parse_args()

    ns = load_survival_curves()
    cumulative = ns[f"per_cycle_survival_{args.curve}"]
    conditional = ns["cumulative_to_conditional"](cumulative)
    attempt_rounds = len(conditional)

    times = simulate(conditional, args.samples, np.random.default_rng(args.seed))
    d = args.distance
    floor, mean, median = times.min(), times.mean(), float(np.median(times))

    print(f"survival curve {args.curve}, {args.samples:,} samples, d={d}")
    print(f"  one attempt          {attempt_rounds:>9} rounds")
    print(f"  end-to-end survival  {cumulative[-1]:>9.4f}")
    print(f"  attempts per state   {mean / attempt_rounds:>9.1f}")
    print()
    print(f"  {'':<14} {'rounds':>10} {'lcycles':>10}")
    print(f"  {'floor (min)':<14} {floor:>10} {floor / d:>10.2f}")
    print(f"  {'mean':<14} {mean:>10.1f} {mean / d:>10.2f}")
    print(f"  {'median':<14} {median:>10.1f} {median / d:>10.2f}")
    print()
    print(f"  floor / mean         {floor / mean:>9.1%}   <- F/<t> for --cultivation-floor")
    print(f"  --cultivation-floor  {floor / d:>9.2f}")
    print()
    # An exponential of the same mean has median = mean*ln2. Agreement here is
    # what justifies the memoryless model in the first place, so report it
    # rather than asserting the fit is good.
    exp_median = mean * np.log(2.0)
    print(f"  exponential check: median {median:.1f} vs {exp_median:.1f} predicted "
          f"for an exponential of this mean ({abs(median - exp_median) / exp_median:.1%} off)")

    print()
    print("  NOTE: the survival arrays in mcmc_cultivation.py are transcribed by")
    print("  hand from Figure 15 of Gidney et al. and are acknowledged there as")
    print("  approximate. Re-deriving them from that paper's published end-to-end")
    print("  simulation data would make this number citable rather than indicative.")


if __name__ == "__main__":
    main()
