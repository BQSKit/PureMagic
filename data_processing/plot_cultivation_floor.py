#!/usr/bin/env python3
"""Plot how the PureMagic advantage depends on the cultivation floor.

PureMagic cuts the long tail of the cultivation time distribution by
interrupting slow attempts and restarting them. That mechanism benefits
specifically from the distribution being *memoryless*: a restart costs
nothing already spent. Real cultivation is not memoryless -- injection and
escape are near-deterministic, so a restarted attempt pays them again.

`puremagic --cultivation-floor F` models that by drawing from a shifted
exponential: a fixed F lcycles plus an exponential whose mean is reduced by F,
so the total mean stays 1/lambda. Sweeping F is therefore a pure *shape*
change at fixed mean -- F=0 is the memoryless model and F -> 1/lambda
approaches deterministic cultivation. This script plots what that costs.

Inputs are two `puremagic` run logs covering the same (circuit, floor) grid,
one for the bus layout and one for PureMagic.
"""

import argparse
import os
import re
import sys

import matplotlib

# Default to the non-interactive Agg backend; see the note in plot_puremagic.py.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_puremagic import _COLOURS, prettify_circuit_name  # noqa: E402

BANNER = re.compile(r"PureMagic - Git branch")
CIRCUIT = re.compile(r'circuit_fname: "[^"]*/([^/"]+?)\.trans[^"]*"')
FLOOR = re.compile(r"cultivation_floor:\s*([0-9.eE+\-]+),?")
VOLUME = re.compile(r"Scheduled \d+ in (\d+) logical cycles, volume (\d+)")
CULT_AVG = re.compile(r"average:\s*([0-9.eE+\-]+)")


def parse(path):
    """-> {circuit: {floor: (cycles, volume, observed_cultivation_avg)}}"""
    out = {}
    with open(path) as f:
        text = f.read()
    for run in re.split(r"(?=PureMagic - Git branch)", text):
        if "volume" not in run:
            continue
        c, fl, v = CIRCUIT.search(run), FLOOR.search(run), VOLUME.search(run)
        if not (c and v):
            continue
        # A log predating --cultivation-floor has no such line; treat it as 0.
        floor = float(fl.group(1)) if fl else 0.0
        ca = CULT_AVG.search(run)
        out.setdefault(c.group(1), {})[floor] = (
            int(v.group(1)),
            int(v.group(2)),
            float(ca.group(1)) if ca else float("nan"),
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", required=True, help="puremagic log for the bus layout")
    ap.add_argument("--pm", required=True, help="puremagic log for the PureMagic layout")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--figsize", default="7.0,3.2")
    ap.add_argument(
        "--mark-floor",
        type=float,
        default=None,
        help="Draw a vertical marker at this floor, for the physically derived "
        "operating point (see derive_cultivation_floor.py). Everything to the "
        "right of it is hypothetical.",
    )
    args = ap.parse_args()

    bus, pm = parse(args.bus), parse(args.pm)
    circuits = [c for c in pm if c in bus]
    if not circuits:
        sys.exit("no circuits common to both logs")

    w, h = (float(x) for x in args.figsize.split(","))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h))

    print(f"{'circuit':<26} {'floor':>5} {'bus vol':>12} {'pm vol':>12} {'impr %':>8} {'obs avg':>8}")
    for i, c in enumerate(sorted(circuits)):
        floors = sorted(set(bus[c]) & set(pm[c]))
        impr, obs = [], []
        for fl in floors:
            bv, pv = bus[c][fl][1], pm[c][fl][1]
            impr.append((bv - pv) / pv * 100.0)
            obs.append(pm[c][fl][2])
            print(f"{prettify_circuit_name(c):<26} {fl:>5.0f} {bv:>12,} {pv:>12,} "
                  f"{impr[-1]:>8.1f} {obs[-1]:>8.2f}")
        col = _COLOURS[i % len(_COLOURS)]
        label = prettify_circuit_name(c)
        ax1.plot(floors, impr, "o-", color=col, label=label)
        ax2.plot(floors, obs, "o-", color=col, label=label)

    ax1.set_xlabel("Cultivation floor (lcycles)")
    ax1.set_ylabel("PureMagic advantage over bus (%)")
    ax1.axhline(0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("Cultivation floor (lcycles)")
    ax2.set_ylabel("Observed cultivation time (lcycles)")
    ax2.legend(fontsize=7)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        if args.mark_floor is not None:
            ax.axvline(args.mark_floor, color="black", lw=1.2, ls="--")
            ax.annotate(
                f"actual\n$F\\approx{args.mark_floor:g}$",
                xy=(args.mark_floor, 0.97),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                ha="left",
                va="top",
                xytext=(3, 0),
                textcoords="offset points",
            )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nwrote {args.output}")

    # Only pop up a window when explicitly asked; the figure is already saved.
    if os.environ.get("PUREMAGIC_SHOW_PLOTS"):
        plt.show()


if __name__ == "__main__":
    main()
