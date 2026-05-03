"""
Head-to-head comparison: trained agent vs 3 analytic baselines.

Usage:
    python scripts/compare.py --ckpt checkpoints/transformer_cvar_best.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlmm.envs.mm_env import MarketMakingEnv
from rlmm.agents.mlp_policy import MLPPolicy
from rlmm.agents.transformer_policy import TransformerPolicy
from rlmm.baselines.avellaneda_stoikov import AvellanedaStoikov
from rlmm.baselines.gueant_lehalle_tapia import GueantLehalleTapia
from rlmm.baselines.cartea_jaimungal import CarteaJaimungal
from rlmm.train.eval import evaluate_policy, print_comparison_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    p.add_argument("--policy", choices=["transformer", "mlp"], default="transformer")
    p.add_argument("--n-episodes", type=int, default=200, dest="n_episodes")
    p.add_argument("--n-levels", type=int, default=10, dest="n_levels")
    p.add_argument("--T-window", type=int, default=20, dest="T_window")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    probe = MarketMakingEnv(n_levels=args.n_levels, T_window=args.T_window)
    feature_dim = probe.feature_builder.feature_dim
    obs_dim = probe.obs_dim

    if args.policy == "transformer":
        policy = TransformerPolicy(input_dim=feature_dim)
    else:
        policy = MLPPolicy(obs_dim=obs_dim)

    state = torch.load(args.ckpt, map_location=device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    transformer_mode = (args.policy == "transformer")

    def env_factory():
        return MarketMakingEnv(n_levels=args.n_levels, T_window=args.T_window, vol_regime="high")

    results = {}

    # RL agent
    print(f"Evaluating RL agent ({args.policy})...")
    results["RL Agent"] = evaluate_policy(
        policy, env_factory, n_episodes=args.n_episodes, device=device,
        transformer_mode=transformer_mode, T_window=args.T_window, feature_dim=feature_dim,
    )

    # Baselines
    for name, baseline in [
        ("A-S", AvellanedaStoikov()),
        ("GLT", GueantLehalleTapia()),
        ("CJ", CarteaJaimungal()),
    ]:
        print(f"Evaluating {name}...")
        results[name] = evaluate_policy(
            nn.Module(), env_factory, n_episodes=args.n_episodes,
            device=device, baseline=baseline,
        )

    print("\n=== Head-to-Head Comparison ===")
    print_comparison_table(results)


if __name__ == "__main__":
    main()
