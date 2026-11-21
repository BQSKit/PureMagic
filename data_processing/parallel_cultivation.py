#!/usr/bin/env python3
"""Cultivation time when a patch runs several starts in parallel.

cultivation_model.py gives the cost of one attempt at a time with a full
restart after every discard. Gidney et al. note that "multiple attempts at
stage 1 and stage 2 would be run in parallel" (Figure 4 caption), and that
packing the early stages into the final code's footprint is the constraint
(Section 3.3). The paper says neither how many starts fit in a patch nor what
happens to the others when one escapes, so both are inputs here.

The model runs k slots in lockstep, one step per surface code round, on the
curve in data/cultivation/survival_d<d1>.csv with the acceptance from
cultivation_model.acceptance(). A slot that dies restarts on the next step. A
slot that survives the last round is accepted with probability
acceptance / S_last, as the gap decision takes no time. The escape stage
starts at the round where the activated qubit count jumps (round 10 at d1=5,
462 qubits, most of a d=17 patch), and its rules differ by mode:

- indep:     the slots never interact. An upper bound on the gain.
- exclusive: only one slot may be in escape at a time. The others keep
             running the early rounds, and a slot ready to escape waits.
- freeze:    as exclusive, but the others also stop while one escapes, and
             resume where they were. The patch is full during escape.
- discard:   as freeze, but the others lose their progress when one enters
             escape and restart afterwards.

The reported time is from a cold start to the first accepted state, which is
what the scheduler draws for every new cultivation. The result is still well
described by floor + exponential (floor = one attempt), so the scheduler needs
no change, only `--magic-state-lambda` set to 1/mean. With one start the
result agrees with cultivation_model.derive().

--check-qce reconstructs the cultivation time the QCE tag used. Its
simulation started four attempts in a patch, took the per-round survival from
a transcription of the paper's figure that sat one round late, and used an
acceptance of about 1% (the "99% discard" of Figure 2) in place of the
release's 1.11%. The single-start script that came with QCE then used the same
curve with its acceptance tuned to 1.8%, so that one start reproduced that
four-start mean, and that is the value the QCE tag shipped.
"""

import argparse
import csv
import os

import numpy as np

import cultivation_model as cm

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("indep", "exclusive", "freeze", "discard")

#: The per-round survival the QCE tag's mcmc_cultivation.py transcribed, d1=5,
#: kept to reproduce that model. One round late against survival_d5.csv, and
#: its last entry (1.8%) is the acceptance it used.
QCE_CURVE = [1.0, 1.0, 0.83, 0.83, 0.72, 0.53, 0.4, 0.3, 0.3, 0.19, 0.12, 0.1,
             0.08, 0.075, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06,
             0.06, 0.018]


def escape_round(d1=5):
    """First round whose activated qubit count jumps by more than 4x."""
    with open(os.path.join(cm.DATA, f"survival_d{d1}.csv")) as f:
        qubits = [int(r["active_qubits"]) for r in csv.DictReader(f)]
    for k in range(1, len(qubits)):
        if qubits[k] > 4 * qubits[k - 1]:
            return k + 1
    raise ValueError("no escape stage found in the activated qubit counts")


def conditional(survival):
    """Per-round survival given the attempt reached the round."""
    s = np.array(survival, dtype=float)
    prev = np.concatenate([[1.0], s[:-1]])
    return np.where(prev > 0, s / prev, 0.0)


def step(r, cond, p_gap, rng, k, mode, esc):
    """Advance every slot of every sample one round. -> (new r, accepted)."""
    m = r.shape[0]
    n = len(cond)
    if mode == "indep":
        go = np.ones((m, k), bool)
    else:
        in_slot = (r >= esc) & (r <= n - 1)
        in_escape = in_slot.any(axis=1)
        wants = r == esc - 1
        keys = rng.random((m, k))
        keys[~wants] = -1.0
        chosen = np.zeros((m, k), bool)
        chosen[np.arange(m), keys.argmax(axis=1)] = wants.any(axis=1)
        enter = chosen & ~in_escape[:, None]
        if mode == "exclusive":
            go = ~wants | enter
        else:
            escaping = in_escape | enter.any(axis=1)
            go = np.where(escaping[:, None], in_slot | enter, ~wants | enter)
    survive = rng.random((m, k)) < cond[r]
    new = np.where(go, np.where(survive, r + 1, 0), r)
    if mode == "discard":
        escaping_now = (in_slot | enter).any(axis=1)
        new = np.where(escaping_now[:, None] & ~(in_slot | enter), 0, new)
    done = new == n
    accepted = done & (rng.random((m, k)) < p_gap)
    return np.where(done, 0, new), accepted


def cold_start(cond, p_gap, esc, k, mode, samples, seed):
    """Rounds from a cold start to the first accepted state, per sample."""
    rng = np.random.default_rng(seed)
    r = np.zeros((samples, k), np.int64)
    t_done = np.zeros(samples)
    alive = np.ones(samples, bool)
    t = 0
    while alive.any():
        t += 1
        r, accepted = step(r, cond, p_gap, rng, k, mode, esc)
        hit = accepted.any(axis=1) & alive
        t_done[hit] = t
        alive &= ~hit
    return t_done


def mean_rounds(cond, p_gap, esc, k, mode, samples, seed):
    return cold_start(cond, p_gap, esc, k, mode, samples, seed).mean()


def check_qce(samples, seed, distance):
    """Old curve at two acceptances, one start and four starts."""
    s = np.array(QCE_CURVE[:24])
    cond = conditional(s)
    cond = np.append(cond, 1.0)  # round 25 culls nothing, the gap decides
    esc = escape_round()
    print(f"QCE reconstruction (transcribed curve, {distance=}):")
    print(f"  {'acceptance':>10} {'1 start':>17} {'4 starts, discard':>20}")
    for a in (0.010, cm.acceptance(25)[0], 0.018):
        p_gap = a / s[-1]
        one = mean_rounds(cond, p_gap, esc, 1, "indep", samples, seed)
        four = mean_rounds(cond, p_gap, esc, 4, "discard", samples, seed)
        print(f"  {a:>10.4%} {one:>8.0f} ({one / distance:5.1f} lc) "
              f"{four:>10.0f} ({four / distance:5.1f} lc)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--distance", type=int, default=17,
                    help="main code distance: one logical cycle is d rounds")
    ap.add_argument("-k", "--starts", type=int, default=4, help="parallel starts per patch")
    ap.add_argument("-n", "--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check-qce", action="store_true",
                    help="reproduce the QCE cultivation time from its own inputs")
    args = ap.parse_args()
    d = args.distance

    survival, _ = cm.load_curve()
    acc, _, source = cm.acceptance(len(survival))
    cond, p_gap, esc = conditional(survival), acc / survival[-1], escape_round()
    ref = cm.derive(d)
    print(f"{len(survival)}-round attempt, escape from round {esc}, acceptance "
          f"{acc:.4%} ({source}), d={d}")
    print(f"cultivation_model.derive(): mean {ref['mean_lcycles']:.2f} lcycles, "
          f"floor {ref['cultivation_floor']:.4f}")
    print(f"{'starts':>6} {'mode':<10} {'rounds':>7} {'lcycles':>8} {'speedup':>8} "
          f"{'--magic-state-lambda':>21}")
    base = None
    rows = [(1, "indep")] + [(args.starts, m) for m in MODES]
    for k, mode in rows:
        mean = mean_rounds(cond, p_gap, esc, k, mode, args.samples, args.seed)
        base = base or mean
        lc = mean / d
        print(f"{k:>6} {mode:<10} {mean:>7.0f} {lc:>8.2f} {base / mean:>7.2f}x "
              f"{1 / lc:>21.4f}")
    print(f"the floor stays {ref['cultivation_floor']:.4f}: no start finishes "
          "before one full attempt")
    if args.check_qce:
        print()
        check_qce(args.samples, args.seed, d)


if __name__ == "__main__":
    main()
