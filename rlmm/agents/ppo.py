"""
Clean PPO trainer with:
  - GAE(λ) advantage estimation
  - PPO-clip objective (or CVaR variant via cvar_loss.py)
  - IQN critic support
  - Vectorized env support
  - Checkpointing (mirrors deephedge pattern)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .cvar_loss import ppo_mean_loss, ppo_cvar_loss, make_rl_loss
from .iqn_critic import IQNCritic


@dataclass
class PPOConfig:
    # Rollout
    n_envs: int = 8
    n_steps: int = 256          # steps per env per rollout
    total_steps: int = 2_000_000

    # PPO
    clip_eps: float = 0.2
    n_epochs: int = 4
    batch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    vf_coeff: float = 0.5
    max_grad_norm: float = 0.5

    # Optimizer
    lr: float = 3e-4

    # Loss variant
    objective: str = "cvar"     # 'mean' or 'cvar'
    cvar_alpha: float = 0.05
    cvar_coeff: float = 0.5

    # Critic variant
    critic_type: str = "iqn"    # 'scalar' or 'iqn'
    iqn_n_samples: int = 32

    # Logging / saving
    log_interval: int = 10      # rollouts
    save_dir: str = "checkpoints"
    save_best_by: str = "cvar_5"  # metric to save best checkpoint


class RolloutBuffer:
    """Stores one rollout of (obs, action, log_prob, reward, done, value) per env."""

    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple, action_dim: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        self._ptr = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self.values[self._ptr] = value
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr >= self.n_steps

    def reset(self):
        self._ptr = 0

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray, gamma: float, lam: float):
        """Compute GAE advantages and discounted returns in-place."""
        last_adv = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_adv = delta + gamma * lam * next_non_terminal * last_adv
            self.advantages[t] = last_adv
        self.returns = self.advantages + self.values

    def get_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Flatten (n_steps, n_envs) → (n_steps*n_envs,) and move to device."""
        N = self.n_steps * self.n_envs
        return {
            "obs": torch.FloatTensor(self.obs.reshape(N, *self.obs.shape[2:])).to(device),
            "actions": torch.FloatTensor(self.actions.reshape(N, -1)).to(device),
            "log_probs": torch.FloatTensor(self.log_probs.reshape(N)).to(device),
            "advantages": torch.FloatTensor(self.advantages.reshape(N)).to(device),
            "returns": torch.FloatTensor(self.returns.reshape(N)).to(device),
        }


class PPOTrainer:
    """
    PPO training loop supporting both MLPPolicy and TransformerPolicy,
    and both scalar and IQN critics.

    For TransformerPolicy, policy acts as actor+critic.
    For MLPPolicy + IQN, the IQN critic is a separate module.
    """

    def __init__(
        self,
        policy: nn.Module,
        envs,                       # gymnasium VectorEnv or list of envs
        cfg: PPOConfig,
        device: torch.device,
        iqn_critic: Optional[IQNCritic] = None,
        transformer_mode: bool = False,  # obs shape (T, D) vs flat
        T_window: int = 20,
        feature_dim: int = 24,
    ):
        self.policy = policy.to(device)
        self.envs = envs
        self.cfg = cfg
        self.device = device
        self.iqn_critic = iqn_critic.to(device) if iqn_critic else None
        self.transformer_mode = transformer_mode
        self.T_window = T_window
        self.feature_dim = feature_dim

        # Infer obs shape
        if hasattr(envs, "single_observation_space"):
            obs_shape = envs.single_observation_space.shape
            action_dim = envs.single_action_space.shape[0]
            n_envs = envs.num_envs
        else:
            obs_shape = envs[0].observation_space.shape
            action_dim = envs[0].action_space.shape[0]
            n_envs = len(envs)

        self.buffer = RolloutBuffer(cfg.n_steps, n_envs, obs_shape, action_dim)
        self.loss_fn = make_rl_loss(cfg.objective, alpha=cfg.cvar_alpha, cvar_coeff=cfg.cvar_coeff)

        params = list(policy.parameters())
        if iqn_critic:
            params += list(iqn_critic.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.lr)

        os.makedirs(cfg.save_dir, exist_ok=True)
        self._best_metric = -np.inf
        self._global_step = 0

    def _to_policy_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Reshape flat obs to (B, T, D) if transformer mode."""
        if self.transformer_mode:
            return obs.view(obs.shape[0], self.T_window, self.feature_dim)
        return obs

    @torch.no_grad()
    def _get_value(self, obs: torch.Tensor) -> np.ndarray:
        obs_p = self._to_policy_obs(obs)
        if self.iqn_critic is not None:
            v = self.iqn_critic.mean_value(obs_p if not self.transformer_mode else obs_p.flatten(1))
        else:
            _, _, v = self.policy(obs_p)
        return v.cpu().numpy()

    @torch.no_grad()
    def _collect_rollout(self, obs_np: np.ndarray, dones_np: np.ndarray) -> tuple:
        self.buffer.reset()
        for _ in range(self.cfg.n_steps):
            obs_t = torch.FloatTensor(obs_np).to(self.device)
            obs_p = self._to_policy_obs(obs_t)
            action, log_prob, value = self.policy.act(obs_p)
            if self.iqn_critic is not None:
                flat_obs = obs_p.flatten(1) if self.transformer_mode else obs_p
                value = self.iqn_critic.mean_value(flat_obs)

            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = self.envs.step(action_np)
            done = terminated | truncated

            self.buffer.add(obs_np, action_np, log_prob_np, rewards, done.astype(np.float32), value_np)
            obs_np = next_obs
            dones_np = done.astype(np.float32)
            self._global_step += obs_np.shape[0]

        last_obs_t = torch.FloatTensor(obs_np).to(self.device)
        last_values = self._get_value(last_obs_t)
        self.buffer.compute_gae(last_values, dones_np, self.cfg.gamma, self.cfg.gae_lambda)
        return obs_np, dones_np

    def _update_policy(self, tensors: dict) -> dict:
        obs = tensors["obs"]
        actions = tensors["actions"]
        old_log_probs = tensors["log_probs"]
        advantages = tensors["advantages"]
        returns = tensors["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"pg_loss": [], "vf_loss": [], "entropy": []}
        N = obs.shape[0]

        for _ in range(self.cfg.n_epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.cfg.batch_size):
                mb_idx = idx[start: start + self.cfg.batch_size]
                mb_obs = self._to_policy_obs(obs[mb_idx])
                mb_actions = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                log_prob, entropy, value = self.policy.evaluate(mb_obs, mb_actions)

                # Critic loss
                if self.iqn_critic is not None:
                    flat = mb_obs.flatten(1) if self.transformer_mode else mb_obs
                    tau = torch.rand(flat.shape[0], self.cfg.iqn_n_samples, device=self.device)
                    pred_q = self.iqn_critic(flat, tau)
                    target_q = mb_ret.unsqueeze(1).expand_as(pred_q).detach()
                    tau_target = torch.rand_like(tau)
                    vf_loss = self.iqn_critic.quantile_huber_loss(pred_q, target_q, tau)
                else:
                    vf_loss = 0.5 * (value - mb_ret).pow(2).mean()

                pg_loss = self.loss_fn(log_prob, mb_old_lp, mb_adv, mb_ret)
                ent_loss = -entropy.mean()
                loss = pg_loss + self.cfg.vf_coeff * vf_loss + self.cfg.entropy_coeff * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                if self.iqn_critic:
                    nn.utils.clip_grad_norm_(self.iqn_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["pg_loss"].append(pg_loss.item())
                metrics["vf_loss"].append(float(vf_loss.item() if hasattr(vf_loss, "item") else vf_loss))
                metrics["entropy"].append(-ent_loss.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    def train(self, eval_fn=None) -> dict:
        """
        Main training loop.

        eval_fn: callable(policy, device) → dict of metrics; called every log_interval rollouts.
        """
        obs_np, _ = self.envs.reset()
        dones_np = np.zeros(self.buffer.n_envs, dtype=np.float32)

        rollout_idx = 0
        all_metrics = []

        while self._global_step < self.cfg.total_steps:
            obs_np, dones_np = self._collect_rollout(obs_np, dones_np)
            tensors = self.buffer.get_tensors(self.device)
            update_metrics = self._update_policy(tensors)

            rollout_idx += 1
            if rollout_idx % self.cfg.log_interval == 0:
                msg = f"step={self._global_step}  " + "  ".join(f"{k}={v:.4f}" for k, v in update_metrics.items())
                print(msg)

                if eval_fn is not None:
                    eval_metrics = eval_fn(self.policy, self.device)
                    all_metrics.append(eval_metrics)
                    print("  eval:", {k: f"{v:.4f}" for k, v in eval_metrics.items()})

                    metric_val = eval_metrics.get(self.cfg.save_best_by, eval_metrics.get("sharpe", 0.0))
                    if metric_val > self._best_metric:
                        self._best_metric = metric_val
                        tag = f"{self.policy.__class__.__name__.lower()}_{self.cfg.objective}_best"
                        self._save(tag)
                        print(f"  → new best {self.cfg.save_best_by}={metric_val:.4f}, saved {tag}.pt")

        return {"metrics": all_metrics}

    def _save(self, tag: str):
        path = os.path.join(self.cfg.save_dir, f"{tag}.pt")
        state = {"policy": self.policy.state_dict()}
        if self.iqn_critic:
            state["iqn_critic"] = self.iqn_critic.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        if self.iqn_critic and "iqn_critic" in state:
            self.iqn_critic.load_state_dict(state["iqn_critic"])
