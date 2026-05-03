"""
Notebook 3: Main RL Experiments

4 experiments + ablations:
  1. MLP + scalar critic + mean objective (vanilla PPO baseline)
  2. MLP + IQN critic + CVaR objective
  3. Transformer + IQN + CVaR (no warm-start, no curriculum)
  4. Transformer + IQN + CVaR + A-S warm-start + curriculum  [main result]

All compared against A-S, GLT, CJ baselines.

Run: python -m rlmm.notebooks.03_ppo_iqn_cvar
Expects checkpoints to exist at checkpoints/*.pt (run train.py first).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from rlmm.envs.mm_env import MarketMakingEnv
from rlmm.agents.mlp_policy import MLPPolicy
from rlmm.agents.transformer_policy import TransformerPolicy
from rlmm.agents.iqn_critic import IQNCritic
from rlmm.baselines.avellaneda_stoikov import AvellanedaStoikov
from rlmm.baselines.gueant_lehalle_tapia import GueantLehalleTapia
from rlmm.baselines.cartea_jaimungal import CarteaJaimungal
from rlmm.train.eval import evaluate_policy, print_comparison_table

SAVE_DIR = "results/figs"
CKPT_DIR = "checkpoints"
N_EPISODES = 200
N_LEVELS = 10
T_WINDOW = 20


def env_factory(regime="high"):
    return MarketMakingEnv(n_levels=N_LEVELS, T_window=T_WINDOW, vol_regime=regime)


def load_policy(path, policy_type="transformer", device=torch.device("cpu")):
    env_probe = env_factory()
    feature_dim = env_probe.feature_builder.feature_dim
    obs_dim = env_probe.obs_dim

    if policy_type == "transformer":
        policy = TransformerPolicy(input_dim=feature_dim, action_dim=4)
    else:
        policy = MLPPolicy(obs_dim=obs_dim, action_dim=4)

    state = torch.load(path, map_location=device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cpu")

    env_probe = env_factory()
    feature_dim = env_probe.feature_builder.feature_dim
    obs_dim = env_probe.obs_dim

    results = {}

    # ------------------------------------------------------------------
    # Load trained agents (if checkpoints exist)
    # ------------------------------------------------------------------
    ckpt_map = {
        "MLP+Scalar+Mean": ("mlpolicy_mean_best.pt", "mlp"),
        "MLP+IQN+CVaR": ("mlpolicy_cvar_best.pt", "mlp"),
        "Transformer+IQN+CVaR": ("transformerpolicy_cvar_best.pt", "transformer"),
        "Transformer+IQN+CVaR+WarmStart+Curriculum": ("transformer_cvar_final.pt", "transformer"),
    }

    import torch.nn as nn
    for name, (ckpt_file, ptype) in ckpt_map.items():
        ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue
        print(f"Loading {name}...")
        policy = load_policy(ckpt_path, ptype, device)
        transformer_mode = (ptype == "transformer")
        metrics = evaluate_policy(
            policy, lambda: env_factory("high"),
            n_episodes=N_EPISODES, device=device,
            transformer_mode=transformer_mode,
            T_window=T_WINDOW, feature_dim=feature_dim,
        )
        results[name] = metrics

    # ------------------------------------------------------------------
    # Analytic baselines
    # ------------------------------------------------------------------
    baselines = {
        "A-S": AvellanedaStoikov(),
        "GLT": GueantLehalleTapia(),
        "CJ": CarteaJaimungal(),
    }
    for name, baseline in baselines.items():
        print(f"Evaluating baseline {name}...")
        metrics = evaluate_policy(
            nn.Module(), lambda: env_factory("high"),
            n_episodes=N_EPISODES, device=device, baseline=baseline,
        )
        results[name] = metrics

    if not results:
        print("No checkpoints found. Run training first:\n  python -m rlmm.train.train --policy transformer ...")
        return

    print("\n=== Full Comparison ===")
    print_comparison_table(results)

    # Save results JSON
    json_path = f"{SAVE_DIR}/03_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    names = list(results.keys())
    metrics_to_plot = ["sharpe", "cvar_5", "inventory_l2"]
    titles = ["Sharpe Ratio", "CVaR-5%", "Inventory L2"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        vals = [results[n].get(metric, 0.0) for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)

    plt.tight_layout()
    path = f"{SAVE_DIR}/03_main_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
