#!/usr/bin/env python3
"""Plot scheduled volume against the ancilla-to-data patch ratio.

More ancilla means more area charged against volume, but also more patches
cultivating and less routing contention. Where the optimum sits is therefore a
property of the circuit (how magic-state-starved it is) and of the cultivation
rate (how much a cultivating patch is worth), not a constant -- which is what
this sweep measures.

Input is one `puremagic` log covering an `--ancilla-rows` sweep, optionally
crossed with `--magic-state-lambda`. Each lambda becomes a line style, each
circuit a colour.
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

CIRCUIT = re.compile(r'circuit_fname: "[^"]*/([^/"]+?)\.trans[^"]*"')
ROWS = re.compile(r"ancilla_rows:\s*(\d+),?")
LAMBDA = re.compile(r"magic_state_lambda:\s*([0-9.eE+\-]+),?")
DATA_Q = re.compile(r"data:\s+(\d+) \(")
MAGIC_Q = re.compile(r"magic:\s+(\d+) \(")
VOLUME = re.compile(r"Scheduled \d+ in (\d+) logical cycles, volume (\d+)")


def parse(path):
    rows = []
    with open(path) as f:
        text = f.read()
    for run in re.split(r"(?=PureMagic - Git branch)", text):
        if "volume" not in run:
            continue
        c, r, lam = CIRCUIT.search(run), ROWS.search(run), LAMBDA.search(run)
        d, m, v = DATA_Q.search(run), MAGIC_Q.search(run), VOLUME.search(run)
        if not (c and r and d and m and v):
            continue
        rows.append(
            dict(
                circuit=c.group(1),
                rows=int(r.group(1)),
                lam=float(lam.group(1)) if lam else float("nan"),
                data=int(d.group(1)),
                ancilla=int(m.group(1)),
                volume=int(v.group(2)),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--file", required=True, help="puremagic log of an --ancilla-rows sweep")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--figsize", default="7.0,3.2")
    ap.add_argument("--normalize", action="store_true",
                    help="plot volume relative to each circuit's own best, not absolute")
    args = ap.parse_args()

    rows = parse(args.file)
    if not rows:
        sys.exit(f"no runs parsed from {args.file}")

    lambdas = sorted({r["lam"] for r in rows}, reverse=True)
    circuits = sorted({r["circuit"] for r in rows})
    styles = ["o-", "s--", "^:", "d-."]

    w, h = (float(x) for x in args.figsize.split(","))
    fig, axes = plt.subplots(1, len(lambdas), figsize=(w, h), squeeze=False)

    hdr = f"{'circuit':<26} {'rows':>4} {'anc/data':>8} {'volume':>14}"
    print(hdr)
    for ax, lam in zip(axes[0], lambdas):
        for i, c in enumerate(circuits):
            pts = sorted((r for r in rows if r["circuit"] == c and r["lam"] == lam),
                         key=lambda r: r["rows"])
            if not pts:
                continue
            ratio = [r["ancilla"] / r["data"] for r in pts]
            vol = [r["volume"] for r in pts]
            for r, q, v in zip(pts, ratio, vol):
                print(f"{prettify_circuit_name(c):<26} {r['rows']:>4} {q:>8.2f} {v:>14,}")
            if args.normalize:
                best = min(vol)
                vol = [v / best for v in vol]
            ax.plot(ratio, vol, styles[i % len(styles)],
                    color=_COLOURS[i % len(_COLOURS)], label=prettify_circuit_name(c))
        # 1/lambda is the mean cultivation time in lcycles.
        ax.set_title(f"mean cultivation {1.0 / lam:.0f} lcycles", fontsize=9)
        ax.set_xlabel("Ancilla / data patches")
        ax.grid(alpha=0.3)
        if not args.normalize:
            ax.set_yscale("log")
    axes[0][0].set_ylabel("Volume relative to best" if args.normalize else "Volume")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nwrote {args.output}")

    # Only pop up a window when explicitly asked; the figure is already saved.
    if os.environ.get("PUREMAGIC_SHOW_PLOTS"):
        plt.show()


if __name__ == "__main__":
    main()
