"""
Evaluation harness: runs policy/baseline over N episodes and computes metrics.

Metrics reported:
    mean_pnl            : average episode PnL (cash + mark-to-market)
    sharpe              : mean_pnl / std_pnl (per-episode)
    cvar_5              : CVaR at 5% (average of worst 5% episodes)
    inventory_l2        : mean L2 norm of inventory path per episode
    fill_rate           : fraction of submitted quotes that filled
    realized_spread     : mean realised half-spread per filled unit
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


def evaluate_policy(
    policy: nn.Module,
    env_factory,
    n_episodes: int = 100,
    device: torch.device = torch.device("cpu"),
    transformer_mode: bool = False,
    T_window: int = 20,
    feature_dim: int = 24,
    deterministic: bool = True,
    baseline=None,   # optional analytic baseline; overrides policy if set
) -> dict[str, float]:
    """
    Run `n_episodes` episodes and return dict of metrics.

    env_factory : callable() → gym.Env
    """
    episode_pnls = []
    inventory_traces = []
    fill_sizes = []
    spread_captured = []
    turnover_totals = []

    policy.eval()

    for ep in range(n_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=ep)
        done = False
        ep_pnl = 0.0
        ep_inventory_sq = 0.0
        ep_steps = 0
        ep_turnover = 0.0
        ep_spread = 0.0

        while not done:
            if baseline is not None:
                mid = env._sampler.mid
                inv = env.inventory
                tau = 1.0 - env._step_count / env._max_steps
                sigma = 0.02
                action = baseline.action(mid, inv, tau, sigma, env.tick_size)
            else:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    if transformer_mode:
                        obs_t = obs_t.view(1, T_window, feature_dim)
                    if deterministic:
                        mean, _, _ = policy(obs_t)
                        action = mean.squeeze(0).cpu().numpy()
                    else:
                        action, _, _ = policy.act(obs_t)
                        action = action.squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_pnl += info.get("pnl_realised", 0.0) + info.get("mtm_pnl", 0.0)
            ep_inventory_sq += info.get("inventory", 0.0) ** 2
            ep_turnover += info.get("turnover", 0.0)
            ep_spread += info.get("pnl_realised", 0.0)   # spread captured per step
            ep_steps += 1

        episode_pnls.append(ep_pnl)
        inventory_traces.append(np.sqrt(ep_inventory_sq / max(ep_steps, 1)))
        fill_sizes.append(ep_turnover)
        spread_captured.append(ep_spread)

    pnls = np.array(episode_pnls)
    mean_pnl = float(pnls.mean())
    std_pnl = float(pnls.std() + 1e-8)
    sharpe = mean_pnl / std_pnl

    # CVaR-5%: mean of worst 5% episodes
    k = max(1, int(len(pnls) * 0.05))
    sorted_pnl = np.sort(pnls)
    cvar_5 = float(sorted_pnl[:k].mean())

    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "sharpe": sharpe,
        "cvar_5": cvar_5,
        "inventory_l2": float(np.mean(inventory_traces)),
        "fill_rate": float(np.mean(fill_sizes)),
        "realized_spread": float(np.mean(spread_captured)),
    }


def compare_policies(
    policies: dict,          # name -> (policy_or_baseline, is_baseline)
    env_factory,
    n_episodes: int = 200,
    device: torch.device = torch.device("cpu"),
    transformer_mode: bool = False,
    T_window: int = 20,
    feature_dim: int = 24,
) -> dict[str, dict]:
    """
    Evaluate multiple policies and return results dict.
    """
    results = {}
    for name, (pol, is_baseline) in policies.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_policy(
            pol if not is_baseline else nn.Module(),
            env_factory,
            n_episodes=n_episodes,
            device=device,
            transformer_mode=transformer_mode,
            T_window=T_window,
            feature_dim=feature_dim,
            baseline=pol if is_baseline else None,
        )
        results[name] = metrics
        print(f"  {name}: sharpe={metrics['sharpe']:.3f}  cvar_5={metrics['cvar_5']:.4f}")
    return results


def print_comparison_table(results: dict[str, dict]) -> None:
    keys = ["mean_pnl", "sharpe", "cvar_5", "inventory_l2", "fill_rate", "realized_spread"]
    header = f"{'Policy':<25}" + "".join(f"{k:>18}" for k in keys)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<25}" + "".join(f"{m.get(k, float('nan')):>18.4f}" for k in keys)
        print(row)
