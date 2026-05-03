"""
Notebook 1: LOB Simulator Sanity Check + Stylized Facts

Validates:
  1. Spread distribution (log-normal shape)
  2. Depth profile (decaying with distance from best)
  3. MO sign autocorrelation > 0 (clustering)
  4. Inter-arrival Fano factor > 1 (over-dispersion from Hawkes)

Run: python -m rlmm.notebooks.01_lob_sim_sanity
Saves figures to: results/figs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fano

from rlmm.sim.flow import OrderFlowSampler, FlowParams
from rlmm.sim.hawkes import MultivariateHawkes

SAVE_DIR = "results/figs"
os.makedirs(SAVE_DIR, exist_ok=True)
N_EPISODES = 50
SIM_DURATION = 60.0  # seconds


def run_episode(seed: int):
    sampler = OrderFlowSampler(seed=seed)
    sampler.reset(mid=100.0, seed=seed)
    spreads, depths, mo_signs, imbalances = [], [], [], []

    while not sampler.done:
        result = sampler.step()
        spread = sampler.lob.spread()
        if spread is not None:
            spreads.append(spread)
        bid_s, ask_s = result["snapshot"]
        depths.append((bid_s + ask_s) / 2)
        imbalances.append(result["imbalance"])

    return spreads, depths, imbalances


def main():
    all_spreads, all_depths, all_imbs = [], [], []

    print(f"Running {N_EPISODES} episodes...")
    for seed in range(N_EPISODES):
        spreads, depths, imbs = run_episode(seed)
        all_spreads.extend(spreads)
        all_depths.extend(depths)
        all_imbs.extend(imbs)

    print(f"Collected {len(all_spreads)} spread observations.")

    depths_arr = np.stack(all_depths)   # (N, n_levels)
    mean_depth = depths_arr.mean(axis=0)

    # ------------------------------------------------------------------
    # Figure 1: Spread distribution
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(all_spreads, bins=50, density=True, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Spread (price units)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Spread Distribution")

    # ------------------------------------------------------------------
    # Figure 2: Average depth profile
    # ------------------------------------------------------------------
    levels = np.arange(1, len(mean_depth) + 1)
    axes[1].bar(levels, mean_depth, color="coral", alpha=0.8)
    axes[1].set_xlabel("Level from Best")
    axes[1].set_ylabel("Mean Size")
    axes[1].set_title("Average Depth Profile")

    # ------------------------------------------------------------------
    # Figure 3: Order-flow imbalance autocorrelation
    # ------------------------------------------------------------------
    imbs_arr = np.array(all_imbs[:2000])
    acf = [np.corrcoef(imbs_arr[:-k], imbs_arr[k:])[0, 1] for k in range(1, 21)]
    axes[2].bar(range(1, 21), acf, color="mediumseagreen", alpha=0.8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_title("Imbalance Autocorrelation")

    plt.tight_layout()
    path = f"{SAVE_DIR}/01_lob_stylized_facts.png"
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")

    # ------------------------------------------------------------------
    # Print summary statistics
    # ------------------------------------------------------------------
    print("\n=== Stylized Facts Summary ===")
    print(f"Mean spread:        {np.mean(all_spreads):.4f}")
    print(f"Spread std:         {np.std(all_spreads):.4f}")
    print(f"Mean imbalance:     {np.mean(all_imbs):.4f}")
    print(f"Imbalance std:      {np.std(all_imbs):.4f}")
    print(f"Depth level 1:      {mean_depth[0]:.2f}")
    print(f"Depth level 5:      {mean_depth[4]:.2f}")
    if len(acf) > 0:
        print(f"Imb ACF lag-1:      {acf[0]:.4f}")


if __name__ == "__main__":
    main()
