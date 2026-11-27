# Cultivation data

`stats.csv` from the data release accompanying Gidney, Shutty & Jones,
*Magic state cultivation: growing T states as cheap as CNOT gates*
([Zenodo 10.5281/zenodo.13777072](https://doi.org/10.5281/zenodo.13777072),
CC-BY-4.0). Vendored unmodified.

Standard Sinter output. Two parts of it are paper inputs, both read by
[`data_processing/cultivation_params.py`](../../data_processing/cultivation_params.py):

- **`json_metadata`** gives each configuration's round counts. For the
  `end2end-inplace-distillation` construction at `d1=5, d2=15`, `r=20` is the
  length of the circuit simulated (`r2=5` rounds after the escape). An attempt
  in operation is 25 rounds, as Figure 15 draws it (`r2=10`): the extra rounds
  are the patch waiting for the complementary gap to be computed.
- **`custom_counts`** holds `C<g>`/`E<g>`: accepted shots whose complementary
  gap fell in bin `g`, and how many of those carried a logical error.
  Suffix-summing from a cutoff gives the acceptance rate and logical error rate
  at that cutoff. At `G=104` (the value Gidney et al. use, and this paper's
  operating point) that is 1.13% acceptance at a 1.06e-9 error rate.

What it does *not* contain is the per-round survival curve, or acceptance
for the 25-round protocol. Two files here supply them, both produced with the
release's own code from `code.zip`:

- **`survival_d3.csv`, `survival_d5.csv`**: survival after each round of the
  25-round protocol (18 rounds at `d1=3`), from the release's `sample_times`
  sampler at 102,400 shots, and each round's length in circuit timesteps
  (mostly 10, 249 for a whole `d1=5` attempt). Written by
  [`extract_survival_curve.py`](../../data_processing/extract_survival_curve.py).
- **`stats_r2.csv`**: the release's circuit generator and desaturation decoder
  at `r2=5` and `r2=10`, 2e7 shots each. The `r2=5` row reproduces
  `stats.csv`'s 1.1304% acceptance at `G=104` (1.1306%). The `r2=10` row gives
  1.1140%. Written by
  [`simulate_cultivation_acceptance.py`](../../data_processing/simulate_cultivation_acceptance.py).

[`cultivation_model.py`](../../data_processing/cultivation_model.py) combines
them into `puremagic`'s `--magic-state-lambda` and `--cultivation-floor`: at
`d=17`, 37.24 logical cycles and a floor of 25/17 when a cultivation round
counts as one surface code round (the default), or 62.12 and 2.44 when every
circuit timestep counts and a surface code round is 6 of them.

- **`parallel_starts.csv`**: the mean rounds per delivered state when a patch
  runs four starts, one at each corner, and an escape discards the other
  three. The escape grows the color code into a `d2=15` grafted code from one
  of its corners, which fills most of a `d=17` patch, so a start can escape
  only from near a patch corner. Simulated per gap cutoff in lockstep rounds
  on the curve and acceptance above, 2e5 cold starts each. Written by
  [`parallel_cultivation.py`](../../data_processing/parallel_cultivation.py)
  `--write-table`, and read by `cultivation_model.py --starts 4`: 18.76
  logical cycles at `d=17`, `G=104`, with the floor unchanged.

- **`parallel_starts_timesteps.csv`**: the same four-start discard case on
  the 20-round attempt in circuit timesteps, each round its own length (15,
  10, 7, 7, then 10), simulated one timestep at a time, since starts that
  die and restart drift out of step with rounds of unequal length. 4e5 cold
  starts per gap cutoff, acceptance from `cultivation_model.acceptance(20,
  G)`. Written by
  [`parallel_cultivation_timesteps.py`](../../data_processing/parallel_cultivation_timesteps.py)
  `--write-table`. Four starts are 2.07x faster than one at every cutoff,
  2903 timesteps against 6007 at `G=104`. With one start the simulation
  reproduces `cultivation_model.py --attempt-rounds 20 --convention
  timesteps` (`--check`).

The release's `code.zip` and `circuits.zip` are gitignored. The two scripts
above run from inside the unpacked `code.zip`. Re-fetch it from the DOI if
needed.
