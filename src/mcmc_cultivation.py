#!/usr/bin/env python3

"""
MCMC simulation of magic state cultivation time distribution.

Survival fractions are read from Figure 15 of Gidney et al.
(arXiv:2409.17595). Each array entry is the proportion of shots surviving
to that round (i.e. the y-value of the survival curve at that x position).
Values must be non-increasing and start at 1.0.

The x-axis of Figure 15 has:
  - d1=3 case: 18 labelled cycles
  - d1=5 case: 25 labelled cycles

The final ~10 cycles are the escape stage gap-wait, during which survival
holds flat (no postselection). The final survival rates from Figure 2 are:
  - d1=3: ~20% (discard rate 80%)
  - d1=5: ~1%  (discard rate 99%)
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Cumulative survival fractions from Figure 15 of Gidney et al.
# Each entry is the proportion of shots surviving TO that round
# (the y-value of the survival curve at that x position).
# Values must be non-increasing; final value = end-to-end survival rate.
#
# Structure for d1=5 (25 cycles total):
#   - ~3 injection cycles (survival drops sharply)
#   - ~12 cultivation cycles (check-grow-stabilize, survival drops each)
#   - ~10 escape/gap-wait cycles (survival flat - no postselection)
#
# Final values from Figure 2:
#   d1=3: ~0.20  (discard rate 80%)
#   d1=5: ~0.01  (discard rate 99%)
# ---------------------------------------------------------------------------

# d1=3 case: 18 cycles, final survival ~20%
# Rough structure: 3 injection + 5 cultivation + 10 gap-wait
# We distribute the 80% total loss across the first 8 cycles.
# Adjust these values to match the actual curve shape in Figure 15.
per_cycle_survival_d3 = [
    1.0,  # encode T
    1.0,  # stabilize
    0.83, # check T
    0.83, # check T
    0.74, # stabilize
    0.64, # stabilize
    0.6,  # stabilize
    0.57, # escaped
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.55, # wait
    0.25 # ready
]
assert len(per_cycle_survival_d3) == 18, f"Expected 18 cycles, got {len(per_cycle_survival_d3)}"

# d1=5 case: 25 cycles, final survival ~1%
# Rough structure: 3 injection + 12 cultivation + 10 gap-wait
# We distribute the 99% total loss across the first 15 cycles.
# Adjust these values to match the actual curve shape in Figure 15.
per_cycle_survival_d5 = [
    1.0,  # encode T
    1.0,  # stabilize
    0.83, # check T
    0.83, # check T
    0.72, # stabilze
    0.53, # stabilize
    0.4,  # stabilize
    0.3,  # check T
    0.3,  # check T
    0.19, # stabilize
    0.12, # stabilize
    0.1,  # stabilize
    0.08, # stabilize
    0.075, # stabilize
    0.07, # escaped
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.06, # wait
    0.018, # ready
]
assert len(per_cycle_survival_d5) == 25, f"Expected 25 cycles, got {len(per_cycle_survival_d5)}"

# Convert cumulative survival fractions to conditional per-cycle probabilities
# cond[i] = P(survive cycle i | survived all previous cycles)
#          = cumulative[i] / cumulative[i-1]
def cumulative_to_conditional(cumulative):
    arr = np.array(cumulative, dtype=float)
    cond = np.empty_like(arr)
    cond[0] = arr[0]
    for i in range(1, len(arr)):
        cond[i] = arr[i] / arr[i - 1] if arr[i - 1] > 0 else 0.0
    return cond

# Report end-to-end survival (final value of each array)
cum_d3 = per_cycle_survival_d3[-1]
cum_d5 = per_cycle_survival_d5[-1]
print(f"d1=3 end-to-end survival: {cum_d3:.4f} (target ~0.20)")
print(f"d1=5 end-to-end survival: {cum_d5:.4f} (target ~0.01)")

cond_d3 = cumulative_to_conditional(per_cycle_survival_d3)
cond_d5 = cumulative_to_conditional(per_cycle_survival_d5)


def run_mcmc(conditional_probs, n_samples=1_000_000):
    """
    Run MCMC simulation of cultivation time.

    conditional_probs[i] = probability of surviving cycle i given all
    previous cycles survived (derived from cumulative survival fractions).

    Each attempt proceeds cycle by cycle. If it fails at any cycle, the
    attempt is discarded and a new one starts from cycle 0. If it survives
    all cycles, the magic state is successfully produced.

    Returns an array of total cycle counts until success (including all
    failed attempts).
    """
    probs = np.array(conditional_probs)
    success_times = []
    report_every = max(1, n_samples // 20)

    for i in range(n_samples):
        if i % report_every == 0:
            pct = 100.0 * i / n_samples
            print(f"  progress: {i:>{len(str(n_samples))}}/{n_samples} ({pct:5.1f}%)", end="\r", flush=True)
        total_cycles = 0
        while True:
            survived = True
            for p in probs:
                total_cycles += 1
                if rng.random() > p:
                    survived = False
                    break
            if survived:
                success_times.append(total_cycles)
                break

    print(f"  progress: {n_samples}/{n_samples} (100.0%)", flush=True)
    return np.array(success_times)


def fit_exponential(times):
    """Fit an exponential distribution to the cultivation times."""
    # MLE for exponential: lambda = 1/mean
    mean_time = np.mean(times)
    lam = 1.0 / mean_time
    return lam, mean_time


def plot_results(times_d3, times_d5, lam_d3, lam_d5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, times, lam, label, color in zip(
        axes,
        [times_d3, times_d5],
        [lam_d3, lam_d5],
        ["d1=3", "d1=5"],
        ["steelblue", "darkorange"],
    ):
        mean_t = 1.0 / lam
        # CDF
        sorted_t = np.sort(times)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax.plot(sorted_t, cdf, color=color, lw=1.5, label="Empirical CDF")

        # Fitted exponential CDF
        t_range = np.linspace(0, sorted_t[-1], 1000)
        ax.plot(t_range, 1 - np.exp(-lam * t_range), "k--", lw=1.5,
                label=f"Exp fit: λ={lam:.5f}\nmean={mean_t:.1f} cycles")

        ax.set_xlabel("Circuit cycles until success")
        ax.set_ylabel("CDF")
        ax.set_title(f"Cultivation time distribution ({label})")
        ax.legend()
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("cultivation_time_distribution.png", dpi=150)
    plt.show()
    print("Saved cultivation_time_distribution.png")


if __name__ == "__main__":
    N_SAMPLES = 100_000

    n_cycles_d3 = len(per_cycle_survival_d3)
    n_cycles_d5 = len(per_cycle_survival_d5)

    print(f"\nRunning MCMC for d1=3 ({N_SAMPLES:,} samples)...")
    times_d3 = run_mcmc(cond_d3, n_samples=N_SAMPLES)
    lam_d3, mean_d3 = fit_exponential(times_d3)
    print(f"  Mean cultivation time: {mean_d3:.1f} circuit cycles")
    print(f"  Fitted λ: {lam_d3:.6f} per circuit cycle")
    print(f"  Median: {np.median(times_d3):.1f} cycles")
    print(f"  90th percentile: {np.percentile(times_d3, 90):.1f} cycles")

    print(f"\nRunning MCMC for d1=5 ({N_SAMPLES:,} samples)...")
    times_d5 = run_mcmc(cond_d5, n_samples=N_SAMPLES)
    lam_d5, mean_d5 = fit_exponential(times_d5)
    print(f"  Mean cultivation time: {mean_d5:.1f} circuit cycles")
    print(f"  Fitted λ: {lam_d5:.6f} per circuit cycle")
    print(f"  Median: {np.median(times_d5):.1f} cycles")
    print(f"  90th percentile: {np.percentile(times_d5, 90):.1f} cycles")

    # Compare to paper's claimed value
    paper_lam = 0.00227
    paper_mean_cycles = 1.0 / paper_lam
    paper_d = 17
    paper_mean_logical = paper_mean_cycles / paper_d
    print(f"\nPaper's claimed λ={paper_lam}, mean={paper_mean_cycles:.1f} circuit cycles")
    print(f"  = {paper_mean_logical:.1f} logical cycles (at d={paper_d})")

    print(f"\nAnalytic estimate (N_cycles / end-to-end survival):")
    print(f"  d1=3: {n_cycles_d3} / {cum_d3:.4f} = {n_cycles_d3 / cum_d3:.1f} circuit cycles"
          f" = {n_cycles_d3 / cum_d3 / paper_d:.1f} logical cycles")
    print(f"  d1=5: {n_cycles_d5} / {cum_d5:.4f} = {n_cycles_d5 / cum_d5:.1f} circuit cycles"
          f" = {n_cycles_d5 / cum_d5 / paper_d:.1f} logical cycles")

    plot_results(times_d3, times_d5, lam_d3, lam_d5)
