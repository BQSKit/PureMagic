# Cultivation data

`stats.csv` from the data release accompanying Gidney, Shutty & Jones,
*Magic state cultivation: growing T states as cheap as CNOT gates*
([Zenodo 10.5281/zenodo.13777072](https://doi.org/10.5281/zenodo.13777072),
CC-BY-4.0). Vendored unmodified.

Standard Sinter output. Two parts of it are paper inputs, both read by
[`data_processing/cultivation_params.py`](../../data_processing/cultivation_params.py):

- **`json_metadata`** gives each configuration's round counts. For the
  `end2end-inplace-distillation` construction at `d1=5, d2=15`, `r=20` is the
  length of one cultivation attempt -- the cost a post-selection failure makes
  a patch pay again, and therefore the floor for `puremagic
  --cultivation-floor`.
- **`custom_counts`** holds `C<g>`/`E<g>`: accepted shots whose complementary
  gap fell in bin `g`, and how many of those carried a logical error.
  Suffix-summing from a cutoff gives the acceptance rate and logical error rate
  at that cutoff. At `G=104` (the value Gidney et al. use, and this paper's
  operating point) that is 1.13% acceptance at a 1.06e-9 error rate.

What it does *not* contain is the per-round survival curve. `survival_d3.csv`
and `survival_d5.csv` supply it. They were produced by
[`data_processing/extract_survival_curve.py`](../../data_processing/extract_survival_curve.py),
which runs the release's own `sample_times` sampler (from `code.zip`) for
409,600 shots on the same circuits `stats.csv` measured: `r1=d1`, `r2=5`, so
13 rounds at `d1=3` and 20 at `d1=5`. One tick is one round of syndrome
extraction. The paper's Figure 15 draws a longer variant with `r2=10`, whose
five extra post-escape rounds have no post-selection in them and no
acceptance data in the release, so it is not used.

At `d1=5` and `d=17` this gives a mean of 35.18 logical cycles per delivered
state and a floor of 20/17 = 1.1765, the defaults of `puremagic
--magic-state-lambda` (0.0284238) and `--cultivation-floor`.

The release's `code.zip` and `circuits.zip` are gitignored. Only
`extract_survival_curve.py` needs `code.zip`, run from inside the unpacked
release. Re-fetch it from the DOI if needed.
