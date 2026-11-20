#!/usr/bin/env python3
"""Sample cultivation acceptance at r2=5 and r2=10 with the release's own code.

Writes data/cultivation/stats_r2.csv, which cultivation_model.py reads. The
published stats.csv only covers r2=5 (20 rounds), but an attempt also holds
its patch for the rounds spent waiting for the complementary gap, which
Figure 15 of Gidney et al. draws as r2=10 (25 rounds). Both are sampled here
so the r2=5 row can be checked against stats.csv before the r2=10 row is
trusted.

Builds the end-to-end d1=5, d2=15, p=1e-3 circuits with the release's
tools/make_circuits and samples them with its desaturation decoder through
sinter. 2e7 shots each took about 20 minutes on 10 cores.

    python3 data_processing/simulate_cultivation_acceptance.py \\
        --release /path/to/unpacked/code.zip --work /path/to/work
"""

import argparse
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "cultivation" / "stats_r2.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", type=pathlib.Path, required=True,
                    help="the cultivation release's code.zip, unpacked")
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--shots", type=int, default=20_000_000)
    ap.add_argument("--processes", type=int, default=10)
    args = ap.parse_args()
    release, work = args.release.resolve(), args.work.resolve()
    circuits = work / "circuits"
    for r2 in (5, 10):
        subprocess.run(
            [sys.executable, str(release / "tools/make_circuits"),
             "--circuit_type", "end2end-inplace-distillation",
             "--noise_strength", "1e-3", "--gateset", "css", "--basis", "Y",
             "--out_dir", str(circuits), "--d1", "5", "--d2", "15",
             "--r1", "d1", "--r2", str(r2)], check=True, cwd=release)
    resume = work / "stats.csv"
    subprocess.run(
        ["sinter", "collect", "--metadata_func", "auto",
         "--circuits", *map(str, sorted(circuits.glob("*.stim"))),
         "--decoders", "desaturation", "--custom_decoders", "cultiv:sinter_samplers",
         "--processes", str(args.processes), "--max_shots", str(args.shots),
         "--save_resume_filepath", str(resume)],
        check=True, env={**os.environ, "PYTHONPATH": str(release / "src")})

    import sinter
    merged = {}
    for s in sinter.read_stats_from_csv_files(str(resume)):
        k = s.json_metadata["r2"]
        merged[k] = s if k not in merged else merged[k] + s
    OUT.write_text(sinter.CSV_HEADER + "\n"
                   + "\n".join(merged[k].to_csv_line() for k in sorted(merged)) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
