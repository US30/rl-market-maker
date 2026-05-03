"""
Main training script — mirrors deephedge/train/train.py CLI style.

Usage:
    python -m rlmm.train.train \
        --policy transformer \
        --critic iqn \
        --objective cvar \
        --warm-start as \
        --curriculum on \
        --n-envs 16 \
        --total-steps 2000000 \
        --save-dir checkpoints/
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from ..envs.mm_env import MarketMakingEnv
from ..agents.mlp_policy import MLPPolicy
from ..agents.transformer_policy import TransformerPolicy
from ..agents.iqn_critic import IQNCritic
from ..agents.ppo import PPOTrainer, PPOConfig
from ..agents.as_imitator import ASImitator
from ..train.curriculum import CurriculumScheduler
from ..train.eval import evaluate_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mlp", "transformer"], default="transformer")
    p.add_argument("--critic", choices=["scalar", "iqn"], default="iqn")
    p.add_argument("--objective", choices=["mean", "cvar"], default="cvar")
    p.add_argument("--warm-start", choices=["none", "as"], default="as", dest="warm_start")
    p.add_argument("--curriculum", choices=["on", "off"], default="on")
    p.add_argument("--n-envs", type=int, default=8, dest="n_envs")
    p.add_argument("--total-steps", type=int, default=2_000_000, dest="total_steps")
    p.add_argument("--save-dir", default="checkpoints", dest="save_dir")
    p.add_argument("--n-levels", type=int, default=10, dest="n_levels")
    p.add_argument("--T-window", type=int, default=20, dest="T_window")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def make_env(vol_regime: str = "low", n_levels: int = 10, T_window: int = 20, seed: int = 0):
    def _init():
        env = MarketMakingEnv(
            n_levels=n_levels,
            T_window=T_window,
            vol_regime=vol_regime,
            seed=seed,
        )
        return env
    return _init


def make_vec_envs(vol_regime: str, n_envs: int, n_levels: int, T_window: int, base_seed: int):
    fns = [make_env(vol_regime, n_levels, T_window, base_seed + i) for i in range(n_envs)]
    return gym.vector.SyncVectorEnv(fns)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Build a single env to get dims
    probe_env = MarketMakingEnv(n_levels=args.n_levels, T_window=args.T_window)
    obs_dim = probe_env.obs_dim
    action_dim = 4
    feature_dim = probe_env.feature_builder.feature_dim
    transformer_mode = (args.policy == "transformer")

    # Policy
    if args.policy == "transformer":
        policy = TransformerPolicy(
            input_dim=feature_dim,
            action_dim=action_dim,
            d_model=128,
            n_heads=4,
            n_layers=3,
        )
    else:
        policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden=256, n_layers=3)

    # IQN critic (optional)
    iqn_critic = None
    if args.critic == "iqn":
        iqn_critic = IQNCritic(input_dim=feature_dim * args.T_window if transformer_mode else obs_dim)

    # A-S warm-start
    if args.warm_start == "as":
        print("Warm-starting policy from A-S behaviour cloning...")
        warm_env = probe_env
        imitator = ASImitator(policy, lr=1e-3)
        imitator.pretrain(warm_env, n_steps=5000, device=device, verbose=True)
        print("Warm-start done.")

    # Curriculum
    sched = CurriculumScheduler(args.total_steps) if args.curriculum == "on" else None
    vol_regime = sched.current_regime if sched else "low"

    envs = make_vec_envs(vol_regime, args.n_envs, args.n_levels, args.T_window, args.seed)

    cfg = PPOConfig(
        n_envs=args.n_envs,
        total_steps=args.total_steps,
        objective=args.objective,
        critic_type=args.critic,
        lr=args.lr,
        save_dir=args.save_dir,
    )

    trainer = PPOTrainer(
        policy=policy,
        envs=envs,
        cfg=cfg,
        device=device,
        iqn_critic=iqn_critic,
        transformer_mode=transformer_mode,
        T_window=args.T_window,
        feature_dim=feature_dim,
    )

    # Curriculum-aware training with manual rollout control
    if sched is None:
        # Simple train
        def eval_fn(pol, dev):
            return evaluate_policy(
                pol, lambda: MarketMakingEnv(n_levels=args.n_levels, T_window=args.T_window, vol_regime="high"),
                n_episodes=50, device=dev, transformer_mode=transformer_mode,
                T_window=args.T_window, feature_dim=feature_dim,
            )
        trainer.train(eval_fn=eval_fn)
    else:
        # Curriculum: rebuild envs at stage transitions
        obs_np, _ = envs.reset()
        dones_np = np.zeros(args.n_envs, dtype=np.float32)

        while trainer._global_step < args.total_steps:
            regime, changed = sched.step(trainer._global_step)
            if changed:
                print(f"  [Curriculum] {sched.stage_label()}")
                envs.close()
                envs = make_vec_envs(regime, args.n_envs, args.n_levels, args.T_window, args.seed)
                trainer.envs = envs
                obs_np, _ = envs.reset()
                dones_np = np.zeros(args.n_envs, dtype=np.float32)

            obs_np, dones_np = trainer._collect_rollout(obs_np, dones_np)
            tensors = trainer.buffer.get_tensors(device)
            metrics = trainer._update_policy(tensors)

            if (trainer._global_step // (cfg.n_steps * args.n_envs)) % cfg.log_interval == 0:
                print(f"step={trainer._global_step}  stage={sched.stage_label()}  "
                      + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        print("Training complete.")
        tag = f"{args.policy}_{args.objective}_final"
        trainer._save(tag)
        print(f"Saved {tag}.pt")


if __name__ == "__main__":
    main()
