"""
Notebook 2: Analytic Baselines on Simulated LOB

Evaluates A-S, GLT, CJ on 200 episodes and reports:
  - Mean PnL, Sharpe, CVaR-5%, inventory L2, fill rate, realized spread

Run: python -m rlmm.notebooks.02_baselines
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from rlmm.envs.mm_env import MarketMakingEnv
from rlmm.baselines.avellaneda_stoikov import AvellanedaStoikov
from rlmm.baselines.gueant_lehalle_tapia import GueantLehalleTapia
from rlmm.baselines.cartea_jaimungal import CarteaJaimungal
from rlmm.train.eval import evaluate_policy, print_comparison_table

SAVE_DIR = "results/figs"
os.makedirs(SAVE_DIR, exist_ok=True)
N_EPISODES = 200
N_LEVELS = 10
T_WINDOW = 20


def env_factory():
    return MarketMakingEnv(n_levels=N_LEVELS, T_window=T_WINDOW)


def main():
    import torch
    import torch.nn as nn

    baselines = {
        "Avellaneda-Stoikov": AvellanedaStoikov(gamma=0.1, k=1.5),
        "Gueant-Lehalle-Tapia": GueantLehalleTapia(gamma=0.1, k=1.5, q_max=50),
        "Cartea-Jaimungal": CarteaJaimungal(gamma=0.1, k=1.5, alpha=0.3),
    }

    print(f"Evaluating {len(baselines)} baselines over {N_EPISODES} episodes each...\n")

    results = {}
    pnl_distributions = {}

    for name, baseline in baselines.items():
        print(f"--- {name} ---")
        metrics = evaluate_policy(
            policy=nn.Module(),   # dummy, baseline overrides
            env_factory=env_factory,
            n_episodes=N_EPISODES,
            baseline=baseline,
        )
        results[name] = metrics
        print(f"  Sharpe:   {metrics['sharpe']:.4f}")
        print(f"  CVaR-5%:  {metrics['cvar_5']:.4f}")
        print(f"  Mean PnL: {metrics['mean_pnl']:.4f}")
        print()

    print("\n=== Comparison Table ===")
    print_comparison_table(results)

    # ------------------------------------------------------------------
    # Bar chart comparison
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    names = list(results.keys())
    sharpes = [results[n]["sharpe"] for n in names]
    cvars = [results[n]["cvar_5"] for n in names]
    inv_l2 = [results[n]["inventory_l2"] for n in names]

    axes[0].bar(names, sharpes, color=["steelblue", "coral", "mediumseagreen"])
    axes[0].set_title("Sharpe Ratio")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(names, cvars, color=["steelblue", "coral", "mediumseagreen"])
    axes[1].set_title("CVaR-5% (higher = better tail risk)")
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(names, inv_l2, color=["steelblue", "coral", "mediumseagreen"])
    axes[2].set_title("Inventory L2 (lower = better)")
    axes[2].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = f"{SAVE_DIR}/02_baselines_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
