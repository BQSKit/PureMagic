#!/usr/bin/env python3
"""Extract the per-tick survival curve behind Figure 15 of Gidney et al.

Reuses the cultivation release's own circuit construction and its
`sample_times` sampler (src/cultiv/make_lifetime_plot.py), so the curve is
theirs rather than a transcription of their plot. Writes a CSV of
tick -> survival fraction, which data_processing/cultivation_model.py and
mcmc_cultivation.py read.

`sample_times` returns survivor counts after each tick, where a tick is one
round as `circuit_to_layers` groups them (resets, two-qubit gate layers,
measurements), one of Figure 15's labelled cycles. Each row also records the
round's length in circuit timesteps (TICKs), mostly 10 against a surface
code round's 6. The circuit is Figure 15's 25-round protocol (r2=10), whose
last ten rounds hold the patch while the complementary gap is computed.
"""

import argparse
import csv
import sys

sys.path.insert(0, "src")

import stim  # noqa: E402

import cultiv  # noqa: E402
import gen  # noqa: E402
from cultiv.make_lifetime_plot import circuit_to_layers, sample_times  # noqa: E402


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
    # Each tick's length in circuit timesteps, for cultivation_model.py's
    # "timesteps" convention. The noise model adds no TICKs.
    timesteps = [sum(1 for i in layer.flattened() if i.name == "TICK")
                 for layer in circuit_to_layers(circuit)]

    counts, qubit_counts = sample_times(circuit, args.shots)
    ticks = sorted(t for t in counts if t > 0)
    total = counts[0]
    print(f"sampled {total:,} shots over {len(ticks)} ticks")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "survivors", "survival", "active_qubits", "timesteps"])
        for t in ticks:
            q = qubit_counts[t - 1] if t - 1 < len(qubit_counts) else qubit_counts[-1]
            w.writerow([t, counts[t], counts[t] / total, q, timesteps[t - 1]])
    print(f"wrote {args.out}")

    final = counts[ticks[-1]] / total
    print(f"survival to end of postselection: {final:.5f}")
    print("  (the release grafts the gap-cutoff acceptance on separately;")
    print("   stats.csv gives 1.13% at the G=104 cutoff)")


if __name__ == "__main__":
    main()
