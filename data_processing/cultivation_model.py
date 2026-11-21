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

It also gives the cultivation volume the FLASQ cost model (Huggins et al.,
arXiv:2511.08508) charges per T state: Gidney et al.'s expected volume,
sum_k S_{k-1} q_k w_k over the acceptance rate, where q_k is the physical
qubits the attempt holds in round k, in blocks of 2(d+1)^2 qubits for d
rounds. That counts only the qubits in use, whereas the scheduler holds a
whole patch for the full mean time, which is <t> blocks per state.

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


def load_active_qubits(d1=5):
    """-> physical qubits the attempt holds in each round of the 25-round protocol."""
    with open(os.path.join(DATA, f"survival_d{d1}.csv")) as f:
        return [int(r["active_qubits"]) for r in csv.DictReader(f)]


def block_qubit_rounds(d):
    """Physical qubit-rounds in one FLASQ block: a logical qubit, 2(d+1)^2
    physical qubits, for one logical timestep of d rounds."""
    return 2 * (d + 1) ** 2 * d


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
                if (
                    m.get("c") != "end2end-inplace-distillation"
                    or m.get("d1") != d1
                    or m.get("p") != p
                    or m.get("r") != rounds
                ):
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


def derive(
    d=17,
    gap_cutoff=GAP_CUTOFF,
    convention="rounds",
    d1=5,
    surface_round_timesteps=SURFACE_ROUND_TIMESTEPS,
    attempt_rounds=None,
):
    """Mean cultivation time, lambda and floor at code distance d.

    `attempt_rounds` shortens the attempt to its first rounds, e.g. 20 for
    the r2=5 circuit stats.csv simulated and Tour de Gross time (199
    timesteps), which ends five rounds before the gap decision. That is
    exact only where no post-selection happens in the rounds cut, which is
    checked, and it takes that protocol's own acceptance.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    survival, timesteps = load_curve(d1)
    active = load_active_qubits(d1)
    if attempt_rounds is not None:
        if survival[attempt_rounds - 1 :] != [survival[-1]] * (len(survival) - attempt_rounds + 1):
            raise ValueError(
                f"post-selection continues past round {attempt_rounds}; "
                f"a {attempt_rounds}-round attempt is not a prefix of this curve"
            )
        survival, timesteps = survival[:attempt_rounds], timesteps[:attempt_rounds]
        active = active[:attempt_rounds]
    weights = (
        [1.0] * len(survival)
        if convention == "rounds"
        else [t / surface_round_timesteps for t in timesteps]
    )
    reach = [1.0] + survival[:-1]
    per_attempt = sum(r * w for r, w in zip(reach, weights))
    per_attempt_timesteps = sum(r * t for r, t in zip(reach, timesteps))
    p_accept, ler, source = acceptance(len(survival), gap_cutoff, d1)
    mean = per_attempt / p_accept / d
    # Gidney et al.'s expected volume: the qubits the attempt actually holds,
    # integrated against survival, per accepted state. FLASQ's Vol(Cultivate).
    qubit_rounds = sum(r * q * w for r, q, w in zip(reach, active, weights)) / p_accept
    return {
        "convention": convention,
        "d": d,
        "gap_cutoff": gap_cutoff,
        "d1": d1,
        "attempt_rounds": len(survival),
        "attempt_timesteps": sum(timesteps),
        "per_attempt_surface_rounds": per_attempt,
        "per_attempt_timesteps": per_attempt_timesteps,
        "timesteps_per_state": per_attempt_timesteps / p_accept,
        "acceptance": p_accept,
        "ler_per_accepted": ler,
        "acceptance_source": source,
        "mean_lcycles": mean,
        "magic_state_lambda": 1 / mean,
        "cultivation_floor": sum(weights) / d,
        "qubit_rounds_per_state": qubit_rounds,
        "flasq_vcult_blocks": qubit_rounds / block_qubit_rounds(d),
        "whole_patch_blocks": mean,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-d", "--distance", type=int, default=17)
    ap.add_argument("-G", "--cutoff", type=int, default=GAP_CUTOFF)
    ap.add_argument("--convention", choices=CONVENTIONS + ("both",), default="both")
    ap.add_argument(
        "--attempt-rounds",
        type=int,
        default=None,
        help="shorten the attempt, e.g. 20 for the r2=5 protocol "
        "without the gap wait; default: the whole 25 rounds",
    )
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()
    convs = CONVENTIONS if args.convention == "both" else (args.convention,)
    res = [derive(args.distance, args.cutoff, c, attempt_rounds=args.attempt_rounds) for c in convs]
    if args.json:
        print(json.dumps(res if len(res) > 1 else res[0], indent=1))
        return
    r0 = res[0]
    print(
        f"d1={r0['d1']}, {r0['attempt_rounds']}-round attempt "
        f"({r0['attempt_timesteps']} timesteps), G={r0['gap_cutoff']}: acceptance "
        f"{r0['acceptance']:.4%} from {r0['acceptance_source']}, d={r0['d']}"
    )
    for r in res:
        print(
            f"  {r['convention']:<10} mean {r['mean_lcycles']:7.3f} lcycles   "
            f"--magic-state-lambda {r['magic_state_lambda']:.7f}   "
            f"--cultivation-floor {r['cultivation_floor']:.4f}   "
            f"FLASQ v_cult {r['flasq_vcult_blocks']:.2f} blocks"
        )


if __name__ == "__main__":
    main()
