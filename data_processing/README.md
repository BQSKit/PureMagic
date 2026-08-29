# Data processing

Python scripts for generating, converting, and analyzing the circuits and results used around
the Rust `puremagic`/`transpile`/`compile_cliffordt` pipeline. Everything here is a standalone
script (run with `-h`/`--help` for its options); shared logic lives in the `*_common.py` modules
they import. Generated `*.png`/`*.out` files in this directory are gitignored, not tracked.

## Clifford+T compilation baseline

- **`compile_cliffordt.py`** -- Compiles an arbitrary OpenQASM circuit to the Clifford+T gate set
  via Qiskit or bqskit (`--backend`). Deliberately naive (no exact-rewrite optimization), kept as
  a baseline to compare against the Rust `compile_cliffordt` binary described in the top-level
  [README](../README.md).

## Circuit generation and conversion

- **`generate_rnd_layout.py`** -- Generates random sequences of Pauli products directly in
  `.trans` format, for scheduler testing without a full compile pipeline.
- **`convert_lss_to_qasm.py`** / **`convert_lss_to_trans.py`** -- Convert the verbose "LSS"
  rotation/measurement format (e.g. from `data/lssp-data-silva-et-al/`) to OpenQASM 2 or directly
  to `.trans`. Shared parsing lives in **`lss_common.py`**.
- **`remove_y_gates.py`** -- Rewrites `y` gates out of a Clifford+T circuit, needed because wisq's
  GUOQ optimizer backend can't process them.

## Analysis and lower bounds

- **`flasq_lower_bound.py`** -- Computes the FLASQ lower bound on T-count/volume for a Clifford+T
  circuit in OpenQASM 2 format.
- **`dascot_qubit_count.py`** -- Computes the total logical qubits required by the DASCOT Square
  Sparse and Compact architectures for a given number of data qubits.
- **`mcmc_cultivation.py`** -- MCMC simulation of the magic-state cultivation time distribution
  (survival fractions from Gidney et al.).
- **`analyze_wisq.py`** -- Counts steps and IDs (total and per-step average) in wisq JSON output.

## Plotting and tables

- **`plot_puremagic.py`** -- The unified `puremagic` results plotter; reads `puremagic` stdout logs
  and plots scheduling metrics (efficiency, volume, lcycles, ...) against a chosen x-axis.
  `plot_all.sh` drives it across the standard set of comparison plots.
- **`plot_cultivation_dist.py`** -- Plots cultivation-time distributions from one or more
  directories' `*.cultivation_dist` files.
- **`plot_volume_vs_layers.py`** -- Plots scheduled volume vs. pre-scheduling DAG layer count
  across a max-weight sweep (`out-pm-trans0`..`out-pm-transN`), one point per (circuit, weight).
- **`circuit_table.py`** -- Generates a LaTeX table of circuit statistics from QASM files.
- **`scheduling_table.py`** -- Generates a LaTeX table comparing two scheduling results.
- **`table_common.py`** -- Shared LaTeX table helpers for `circuit_table.py`/`scheduling_table.py`.
- **`puremagic_log.py`** -- Regex patterns shared by the scripts that parse `puremagic` stdout
  (`plot_puremagic.py`, `scheduling_table.py`, `circuit_table.py`).
