#!/usr/bin/env python3
"""Plot the exact cultivation time distribution against the exponential models.

The scheduler draws cultivation times from a (shifted) exponential. That is a
modelling convenience, not a derived result, so this figure shows what it is
approximating.

The exact distribution needs no simulation. Cultivation is a renewal process:
an attempt runs round by round, a postselected detector failure discards it,
and a fresh attempt starts from zero. Writing S_k for the probability of
surviving through tick k (`data/cultivation/survival_d5.csv`, with the final
entry replaced by the complementary-gap acceptance from `stats.csv`), an
attempt dies during tick k with probability S_{k-1} - S_k and costs exactly k
rounds, while a surviving attempt costs the full N. The total time to deliver
one state is therefore compound geometric, and its PMF satisfies

    f(t) = S_N [t = N] + sum_k (S_{k-1} - S_k) f(t - k)

which this script evaluates directly. Gidney et al. compute the *expectation*
of this process -- the Figure 15 caption defines expected qubit-rounds as the
integral of survival times activated qubits, divided by the final survival
rate -- so the mean is theirs; the distribution around it is what the data
release does not contain, and what the exponential assumption supplies.
"""

import argparse
import csv
import math
import os
import sys

import matplotlib

# Default to the non-interactive Agg backend; see the note in plot_puremagic.py.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import COLUMN_IN, PALETTE, REFERENCE, use_paper_style  # noqa: E402

use_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CURVE = os.path.join(HERE, os.pardir, "data", "cultivation", "survival_d5.csv")

# The exact curve is deliberately achromatic: it is the reference the two
# models are measured against, not a third category.
C_EXACT = REFERENCE
C_EXP = PALETTE[0]
C_SHIFT = PALETTE[1]


def exact_pmf(survival, p_accept, tmax):
    """PMF of total rounds per delivered state, by the renewal recurrence."""
    S = list(survival)
    S[-1] = p_accept
    n = len(S)

    prev, death = 1.0, []
    for s in S:
        death.append(prev - s)  # dies during this tick, having cost that tick
        prev = s

    f = [0.0] * (tmax + 1)
    for t in range(1, tmax + 1):
        acc = p_accept if t == n else 0.0
        for k in range(1, n + 1):
            if t - k >= 0 and death[k - 1]:
                acc += death[k - 1] * f[t - k]
        f[t] = acc
    return f, n


def cdf_from_pmf(f):
    out, c = [], 0.0
    for v in f:
        c += v
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve", default=DEFAULT_CURVE)
    ap.add_argument("-p", "--p-accept", type=float, default=0.011304,
                    help="complementary-gap acceptance at the operating cutoff")
    ap.add_argument("-d", "--distance", type=int, default=17,
                    help="code distance: one logical cycle is d rounds")
    ap.add_argument("--tmax-rounds", type=int, default=20000,
                    help="PMF support; the default captures the mass to 1e-6")
    ap.add_argument("--xmax", type=float, default=120.0, help="x limit, lcycles")
    ap.add_argument("--figsize", default=f"{COLUMN_IN},3.4")
    ap.add_argument("-o", "--output",
                    default=os.path.join(HERE, "cultivation_cdf.png"))
    args = ap.parse_args()

    with open(args.curve) as fh:
        survival = [float(r["survival"]) for r in csv.DictReader(fh)]

    f, n_attempt = exact_pmf(survival, args.p_accept, args.tmax_rounds)
    F = cdf_from_pmf(f)
    mass = F[-1]
    if mass < 0.999:
        sys.exit(f"PMF captured only {mass:.4f} of the mass; raise --tmax-rounds")

    d = args.distance
    mean = sum(t * f[t] for t in range(len(f))) / mass
    floor = n_attempt / d                      # one full attempt, in lcycles
    mean_l = mean / d

    def quantile(q):
        c = 0.0
        for t, v in enumerate(f):
            c += v
            if c >= q * mass:
                return t
        return len(f) - 1

    # Models: a plain exponential of the same mean, and one shifted by the
    # attempt length with its scale reduced so the total mean is preserved.
    def exp_cdf(t):
        return 1.0 - math.exp(-t / mean_l)

    def shift_cdf(t):
        if t <= floor:
            return 0.0
        return 1.0 - math.exp(-(t - floor) / (mean_l - floor))

    # Evaluate on the integer-round grid the exact CDF is actually defined on,
    # so no interpolation or rounding jitter enters the residual panel.
    t_hi = min(int(args.xmax * d), len(F) - 1)
    xs = [t / d for t in range(t_hi + 1)]
    ex = [F[t] / mass for t in range(t_hi + 1)]
    ey = [exp_cdf(x) for x in xs]
    sy = [shift_cdf(x) for x in xs]

    w, h = (float(v) for v in args.figsize.split(","))
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(w, h), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})

    # (a) the distributions themselves. They agree closely, which is the point:
    # the residual panel below is what resolves the difference.
    ax.plot(xs, ex, color=C_EXACT, lw=1.8, zorder=3, label="Exact (renewal)")
    ax.plot(xs, ey, color=C_EXP, lw=1.4, ls="--", zorder=2, label="Exponential")
    ax.plot(xs, sy, color=C_SHIFT, lw=1.4, ls=":", zorder=2,
            label=f"Shifted exp. ($F={floor:.2f}$)")

    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4,
              borderaxespad=0.6)

    # (b) model minus exact, in percentage points of cumulative probability.
    axr.axhline(0.0, color=C_EXACT, lw=1.2, zorder=3)
    axr.plot(xs, [(m - e) * 100 for m, e in zip(ey, ex)],
             color=C_EXP, lw=1.4, ls="--", zorder=2)
    axr.plot(xs, [(s - e) * 100 for s, e in zip(sy, ex)],
             color=C_SHIFT, lw=1.4, ls=":", zorder=2)

    axr.set_xlabel("Cultivation time (logical cycles)")
    axr.set_ylabel("Model $-$ exact (pp)")
    axr.set_xlim(0, args.xmax)
    axr.grid(alpha=0.25, lw=0.5)
    axr.set_axisbelow(True)
    for side in ("top", "right"):
        axr.spines[side].set_visible(False)

    fig.align_ylabels((ax, axr))
    fig.savefig(args.output)
    print(f"wrote {args.output}")

    print()
    print(f"attempt length {n_attempt} rounds = {floor:.4f} lcycles")
    print(f"PMF mass to t={args.tmax_rounds}: {mass:.6f}")
    print()
    print(f"{'':<10} {'exact':>9} {'exp':>9} {'diff':>8} {'shifted':>9} {'diff':>8}")
    for name, q in (("median", 0.5), ("p90", 0.9), ("p99", 0.99)):
        e = quantile(q) / d
        m = -mean_l * math.log(1 - q)
        s = floor - (mean_l - floor) * math.log(1 - q)
        print(f"{name:<10} {e:>9.2f} {m:>9.2f} {(m - e) / e:>7.1%} "
              f"{s:>9.2f} {(s - e) / e:>7.1%}")
    print(f"{'mean':<10} {mean_l:>9.2f} {mean_l:>9.2f} {0.0:>7.1%} "
          f"{mean_l:>9.2f} {0.0:>7.1%}")

    if os.environ.get("PUREMAGIC_SHOW_PLOTS"):
        plt.show()


if __name__ == "__main__":
    main()
