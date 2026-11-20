#!/usr/bin/env python3
"""Extract the per-tick survival curve behind Figure 15 of Gidney et al.

Reuses the cultivation release's own circuit construction and its
`sample_times` sampler (src/cultiv/make_lifetime_plot.py), so the curve is
theirs rather than a transcription of their plot. Writes a CSV of
tick -> survival fraction, which data_processing/mcmc_cultivation.py can
consume in place of its hand-typed arrays.

Note `sample_times` returns raw survivor counts per tick, where a tick is one
layer of the flattened circuit, not one of Figure 15's labelled cycles. The
mapping between the two is recovered here by reporting the tick count
alongside the construction's own round parameters.
"""

import argparse
import csv
import sys

sys.path.insert(0, "src")

import stim  # noqa: E402

import cultiv  # noqa: E402
import gen  # noqa: E402
from cultiv.make_lifetime_plot import sample_times  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcolor", type=int, default=5)
    ap.add_argument("--dsurface", type=int, default=15)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, default=1024 * 100)
    ap.add_argument("-o", "--out", default="survival.csv")
    args = ap.parse_args()

    # Same parameters make_lifetime_plot.main() uses for its d1=5 panel.
    circuit = cultiv.make_end2end_cultivation_circuit(
        dcolor=args.dcolor,
        dsurface=args.dsurface,
        basis="Y",
        r_growing=args.dcolor,
        r_end=10,
        inject_style="unitary",
    )
    circuit = gen.NoiseModel.uniform_depolarizing(
        args.noise
    ).noisy_circuit_skipping_mpp_boundaries(circuit)
    print(f"circuit: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors")

    counts, qubit_counts = sample_times(circuit, args.shots)
    ticks = sorted(t for t in counts if t > 0)
    total = counts[0]
    print(f"sampled {total:,} shots over {len(ticks)} ticks")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "survivors", "survival", "active_qubits"])
        for t in ticks:
            q = qubit_counts[t - 1] if t - 1 < len(qubit_counts) else qubit_counts[-1]
            w.writerow([t, counts[t], counts[t] / total, q])
    print(f"wrote {args.out}")

    final = counts[ticks[-1]] / total
    print(f"survival to end of postselection: {final:.5f}")
    print("  (the release grafts the gap-cutoff acceptance on separately;")
    print("   stats.csv gives 1.13% at the G=104 cutoff)")


if __name__ == "__main__":
    main()
