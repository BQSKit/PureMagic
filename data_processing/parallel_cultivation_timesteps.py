#!/usr/bin/env python3
"""Cultivation time with parallel starts per patch, in circuit timesteps.

parallel_cultivation.py advances every start one round per step, which
treats every round as the same length. In circuit timesteps the rounds are
15, 10, 7, 7 and then 10 timesteps long, and a start that dies restarts on
the next timestep, so starts drift out of step and a round-based simulation
cannot follow them. This one advances one timestep per step.

The rules are parallel_cultivation.py's "discard" mode, the case the escape
geometry allows (four starts, one per patch corner):

- each start runs the attempt's rounds in order and, if it dies at the end
  of a round, restarts on the next timestep;
- a start that completes the rounds before the escape stage enters it, and
  the others are discarded and wait, since the escape fills the patch;
- when the escape ends, rejected at the gap or dead, every start restarts;
- a start that completes the last round is accepted with probability
  acceptance / S_last, the gap decision taking no time.

By default the attempt is the 20-round r2=5 protocol, whose acceptance and
error the release reports, as cultivation_model.py --attempt-rounds 20 uses.
With one start the mean is cultivation_model's per-attempt timesteps over
the acceptance, which --check confirms. The floor is unchanged: no start
finishes before one whole attempt.

    python3 parallel_cultivation_timesteps.py --check
    python3 parallel_cultivation_timesteps.py --write-table
"""

import argparse
import csv
import multiprocessing
import os

import numpy as np

import cultivation_model as cm

TABLE = os.path.join(cm.DATA, "parallel_starts_timesteps.csv")
FIELDS = ("d1", "attempt_rounds", "gap_cutoff", "starts", "mode", "samples",
          "seed", "mean_timesteps", "stderr_timesteps", "median_timesteps")
CUTOFFS = (53, 70, 90, 104, 116)


def protocol(attempt_rounds, d1=5):
    """-> (per-round survival given the round is reached, round timesteps,
    rounds completed before the escape, survival after the last round)."""
    survival, timesteps = cm.load_curve(d1)
    qubits = cm.load_active_qubits(d1)
    surv = np.array(survival[:attempt_rounds])
    ts = np.array(timesteps[:attempt_rounds])
    esc = next(k for k in range(1, len(qubits)) if qubits[k] > 4 * qubits[k - 1])
    prev = np.concatenate([[1.0], surv[:-1]])
    return surv / prev, ts, esc, surv[-1]


def cold_start(k, gap_cutoff, attempt_rounds, samples, seed, d1=5):
    """-> timesteps from a cold start to the first accepted state, per sample."""
    cond, ts, esc, s_last = protocol(attempt_rounds, d1)
    n_rounds = len(ts)
    p_gap = cm.acceptance(n_rounds, gap_cutoff, d1)[0] / s_last
    rng = np.random.default_rng(seed)
    r = np.zeros((samples, k), np.int64)     # the round each start is in
    rem = np.tile(ts[0], (samples, k))       # timesteps left in that round
    escaping = np.full(samples, -1)          # the start in escape, or -1
    idx = np.arange(samples)                 # samples still running
    out = np.zeros(samples)
    slots = np.arange(k)[None, :]
    t = 0
    while idx.size:
        t += 1
        m = idx.size
        rows = np.arange(m)
        e = escaping >= 0
        active = np.ones((m, k), bool)
        active[e] = False
        active[rows[e], escaping[e]] = True
        rem = rem - active
        fin = active & (rem == 0)
        live = fin & (rng.random((m, k)) < cond[r])
        died = fin & ~live
        r = np.where(live, r + 1, np.where(died, 0, r))
        touched = fin.copy()
        last = live & (r == n_rounds)
        accepted = last & (rng.random((m, k)) < p_gap)
        rejected = last & ~accepted
        r = np.where(rejected, 0, r)
        # An escape that died or was rejected frees the patch: all restart.
        end = e & (died | rejected)[rows, np.maximum(escaping, 0)]
        r[end] = 0
        touched[end] = True
        escaping[end] = -1
        # A start ready for the escape takes the patch and the others are
        # discarded. Simultaneous ones are settled at random.
        want = live & (r == esc) & (escaping < 0)[:, None]
        if want.any():
            keys = np.where(want, rng.random((m, k)), -1.0)
            pick = keys.argmax(axis=1)
            go = want.any(axis=1)
            others = go[:, None] & (slots != pick[:, None])
            r = np.where(others, 0, r)
            touched |= others
            escaping[go] = pick[go]
        rem = np.where(touched, ts[np.minimum(r, n_rounds - 1)], rem)
        hit = accepted.any(axis=1)
        out[idx[hit]] = t
        keep = ~hit
        idx, r, rem, escaping = idx[keep], r[keep], rem[keep], escaping[keep]
    return out


def _row(task):
    g, k, attempt_rounds, samples, seed = task
    t = cold_start(k, g, attempt_rounds, samples, seed)
    return {"d1": 5, "attempt_rounds": attempt_rounds, "gap_cutoff": g,
            "starts": k, "mode": "discard", "samples": samples, "seed": seed,
            "mean_timesteps": f"{t.mean():.2f}",
            "stderr_timesteps": f"{t.std() / len(t) ** 0.5:.2f}",
            "median_timesteps": f"{np.median(t):.1f}"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--starts", type=int, default=4)
    ap.add_argument("--attempt-rounds", type=int, default=20)
    ap.add_argument("-n", "--samples", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-j", "--jobs", type=int, default=5)
    ap.add_argument("--check", action="store_true",
                    help="one start, against cultivation_model's closed form")
    ap.add_argument("--write-table", action="store_true",
                    help="write data/cultivation/parallel_starts_timesteps.csv")
    args = ap.parse_args()
    k = 1 if args.check else args.starts
    tasks = [(g, k, args.attempt_rounds, args.samples, args.seed) for g in CUTOFFS]
    with multiprocessing.Pool(min(args.jobs, len(tasks))) as pool:
        rows = pool.map(_row, tasks)
    for row in rows:
        g = row["gap_cutoff"]
        ref = cm.derive(17, g, "timesteps", attempt_rounds=args.attempt_rounds)
        one = ref["per_attempt_timesteps"] / ref["acceptance"]
        print(f"G={g:>3} {k} start(s): {float(row['mean_timesteps']):9.1f} "
              f"+- {row['stderr_timesteps']:>6} timesteps; one start, closed form "
              f"{one:8.1f}; speedup {one / float(row['mean_timesteps']):.3f}")
    if args.write_table and not args.check:
        with open(TABLE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {TABLE}")


if __name__ == "__main__":
    main()
