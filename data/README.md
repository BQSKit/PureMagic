# Data

Benchmark circuits and circuit-generation scripts. Only the paths below are tracked in git;
everything else you may see locally under `data/` (`benchpress/`, `qasmbench/`, `mqt/`,
`pm_paper_circuits/`, `grover_benchmarks/`, `lssp-data-silva-et-al/`, `circuits-ablations/`,
`all_compiled_bqskit/`, etc.) is gitignored -- local caches of externally-sourced circuits or
scratch output from ablation runs, not part of the repo.

## `circuits/`

The curated benchmark suite used to stress-test the Clifford+T compiler (`compile_cliffordt`)
and the lattice-surgery scheduler (`puremagic`). See [`circuits/README.md`](circuits/README.md).

## `circuit-generators/`

This repo's own circuit generators, built on Qiskit. `gen_qft.py` builds a QFT size-scaling family
(`qft_N{size}_qiskit.qasm`) via Qiskit's `QFT` library construction; `gen_grover.py` builds Grover
search circuits (oracle + diffusion iterations) for a given qubit count and iteration count.
`gen_common.py` holds the shared OpenQASM-writing helper both scripts use. Run either with `-h`
for its options.
