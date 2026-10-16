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

What it does *not* contain is the per-round survival curve behind the paper's
Figure 15, which only has end-to-end aggregates here. `make_lifetime_plot.py`
in the release's `code.zip` regenerates that curve by re-running the Stim
simulation; `data_processing/mcmc_cultivation.py` instead approximates it from
values transcribed off the figure by hand, which is adequate for the
distribution's *shape* but is not a citable source.

The release's `code.zip` and `circuits.zip` are gitignored -- nothing here
reads them. Re-fetch from the DOI if needed.
