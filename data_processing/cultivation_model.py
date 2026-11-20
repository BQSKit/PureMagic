#!/usr/bin/env python3
"""Cultivation time per delivered magic state, in either round convention.

The single derivation of `puremagic --magic-state-lambda` and
`--cultivation-floor`, for any code distance and gap cutoff. Other code that
needs cultivation parameters (the journal paper's params.sh, the BB vs
PureMagic comparison) should take them from here rather than re-derive them.

Inputs, all in data/cultivation/:

- survival_d<d1>.csv: per-round survival of one attempt and each round's
  length in circuit timesteps, from the cultivation release's own sampler on
  the 25-round protocol (r1=d1, r2=10). The last ten rounds are the patch
  holding an unconfirmed state while the complementary gap is computed, as
  Figure 15 of Gidney et al. describes, so a patch is occupied for them.
- stats_r2.csv: the release's construction and decoder, sampled at r2=5 and
  r2=10, 2e7 shots each. Its r2=5 row reproduces stats.csv's 1.1304%
  acceptance at G=104 to within sampling error. stats.csv itself only covers
  r2=5, so the 25-round curve takes its acceptance from the r2=10 row.

The model is a full restart after every discard. An attempt alive at the
start of round k pays round k, so one attempt costs sum_k S_{k-1} w_k, where
S_0 = 1 and w_k is round k's length, and a delivered state costs that over
the acceptance rate. The floor is one whole attempt, sum_k w_k, since no
delivery can undercut a single attempt that survives everything.

The convention fixes w_k, in surface code rounds:

- "rounds": every round counts as one surface code round, as Gidney et al.
  count them. Their reason is that reset and measurement dominate a round's
  duration on superconducting hardware, so a round with eight CNOT layers is
  not much longer than one with four. PureMagic's defaults use this.
- "timesteps": every circuit layer takes one timestep, as in Tour de Gross,
  and a surface code round is SURFACE_ROUND_TIMESTEPS of them. Cultivation
  rounds are mostly 10 timesteps, so this charges about 1.67 surface code
  rounds each. Use it where the comparison counts time in uniform timesteps.
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data", "cultivation")

#: Reset, four CNOT layers and measurement (Tour de Gross's convention).
SURFACE_ROUND_TIMESTEPS = 6
CONVENTIONS = ("rounds", "timesteps")
GAP_CUTOFF = 104


def load_curve(d1=5):
    """-> (survival per round, timesteps per round) for the 25-round protocol."""
    with open(os.path.join(DATA, f"survival_d{d1}.csv")) as f:
        rows = list(csv.DictReader(f))
    return [float(r["survival"]) for r in rows], [int(r["timesteps"]) for r in rows]


def acceptance(rounds, gap_cutoff=GAP_CUTOFF, d1=5, p=0.001, required=True):
    """-> (acceptance, LER per accepted state or nan, source file) for the
    end-to-end protocol of `rounds` rounds. Prefers the larger sample. With
    `required=False`, returns None when no data covers that protocol."""
    best = None
    for name in ("stats.csv", "stats_r2.csv"):
        with open(os.path.join(DATA, name)) as f:
            for r in csv.DictReader(f):
                r = {k.strip(): v for k, v in r.items()}
                m = json.loads(r["json_metadata"])
                if (m.get("c") != "end2end-inplace-distillation" or m.get("d1") != d1
                        or m.get("p") != p or m.get("r") != rounds):
                    continue
                cc = json.loads(r["custom_counts"])
                shots = int(r["shots"])
                acc = sum(v for k, v in cc.items() if k[0] == "C" and int(k[1:]) >= gap_cutoff)
                err = sum(v for k, v in cc.items() if k[0] == "E" and int(k[1:]) >= gap_cutoff)
                if best is None or shots > best[3]:
                    best = (acc / shots, err / acc if err else float("nan"), name, shots)
    if best is None:
        if not required:
            return None
        sys.exit(f"no d1={d1} p={p} acceptance for a {rounds}-round attempt in {DATA}")
    return best[:3]


def derive(d=17, gap_cutoff=GAP_CUTOFF, convention="rounds", d1=5,
           surface_round_timesteps=SURFACE_ROUND_TIMESTEPS):
    """Mean cultivation time, lambda and floor at code distance d."""
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    survival, timesteps = load_curve(d1)
    weights = ([1.0] * len(survival) if convention == "rounds"
               else [t / surface_round_timesteps for t in timesteps])
    reach = [1.0] + survival[:-1]
    per_attempt = sum(r * w for r, w in zip(reach, weights))
    p_accept, ler, source = acceptance(len(survival), gap_cutoff, d1)
    mean = per_attempt / p_accept / d
    return {
        "convention": convention, "d": d, "gap_cutoff": gap_cutoff, "d1": d1,
        "attempt_rounds": len(survival), "attempt_timesteps": sum(timesteps),
        "per_attempt_surface_rounds": per_attempt,
        "acceptance": p_accept, "ler_per_accepted": ler, "acceptance_source": source,
        "mean_lcycles": mean, "magic_state_lambda": 1 / mean,
        "cultivation_floor": sum(weights) / d,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--distance", type=int, default=17)
    ap.add_argument("-G", "--cutoff", type=int, default=GAP_CUTOFF)
    ap.add_argument("--convention", choices=CONVENTIONS + ("both",), default="both")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()
    convs = CONVENTIONS if args.convention == "both" else (args.convention,)
    res = [derive(args.distance, args.cutoff, c) for c in convs]
    if args.json:
        print(json.dumps(res if len(res) > 1 else res[0], indent=1))
        return
    r0 = res[0]
    print(f"d1={r0['d1']}, {r0['attempt_rounds']}-round attempt "
          f"({r0['attempt_timesteps']} timesteps), G={r0['gap_cutoff']}: acceptance "
          f"{r0['acceptance']:.4%} from {r0['acceptance_source']}, d={r0['d']}")
    for r in res:
        print(f"  {r['convention']:<10} mean {r['mean_lcycles']:7.3f} lcycles   "
              f"--magic-state-lambda {r['magic_state_lambda']:.7f}   "
              f"--cultivation-floor {r['cultivation_floor']:.4f}")


if __name__ == "__main__":
    main()
