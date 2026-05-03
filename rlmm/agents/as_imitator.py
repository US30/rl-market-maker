"""
Imitation warm-start: pre-train the policy to mimic Avellaneda-Stoikov quotes.

Phase 1 (behaviour cloning, ~5k steps): supervised on A-S target actions.
Phase 2 (PPO fine-tune): unfreeze and train with RL objective.

A-S quotes at each state:
    r* = mid - q*gamma*sigma^2*(T-t)          [indifference price]
    s* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)  [optimal spread]
    bid* = r* - s*/2,  ask* = r* + s*/2

We convert A-S (bid_price, ask_price) → (bid_offset_ticks, ask_offset_ticks, skew=0, size=default)
and normalise to [-1, 1] action space.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .avellaneda_stoikov_helper import as_action_from_state


class ASImitator:
    """
    Wraps a policy (MLP or Transformer) and provides behaviour-cloning pretraining.

    Usage:
        imitator = ASImitator(policy)
        imitator.pretrain(env, n_steps=5000, device=device)
        # Now use policy in PPO as usual
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma_as: float = 0.1,
        k_as: float = 1.5,
    ):
        self.policy = policy
        self.lr = lr
        self.gamma_as = gamma_as
        self.k_as = k_as

    def pretrain(
        self,
        env,
        n_steps: int = 5000,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list[float]:
        """
        Collect (obs, a_star) pairs from A-S policy and minimise MSE.

        Returns list of per-batch losses.
        """
        from ..baselines.avellaneda_stoikov import AvellanedaStoikov
        as_policy = AvellanedaStoikov(gamma=self.gamma_as, k=self.k_as)

        optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        losses = []

        obs_buf, act_buf = [], []
        obs_np, _ = env.reset()

        for step in range(n_steps):
            # Get A-S target action from env state
            mid = env._sampler.mid
            inventory = env.inventory
            tau = 1.0 - env._step_count / env._max_steps
            sigma_hat = getattr(env, "_last_sigma", 0.01)
            a_star = as_policy.action(mid, inventory, tau, sigma_hat, env.tick_size)

            obs_buf.append(obs_np.copy())
            act_buf.append(a_star)

            # Step env with A-S action (normalised)
            obs_np, _, done, _, info = env.step(a_star)
            if done:
                obs_np, _ = env.reset()

            if len(obs_buf) >= batch_size:
                obs_t = torch.FloatTensor(np.stack(obs_buf)).to(device)
                act_t = torch.FloatTensor(np.stack(act_buf)).to(device)

                # Handle 2D obs for transformer
                if hasattr(self.policy, "input_proj"):
                    obs_t = self._reshape_for_transformer(obs_t, env)

                mean, _, _ = self.policy(obs_t) if hasattr(self.policy, "input_proj") else (
                    *self.policy(obs_t)[:1], None, None
                )
                # For MLPPolicy, forward returns (mean, log_std, value)
                if hasattr(self.policy, "actor_head"):
                    mean, _, _ = self.policy(obs_t)

                loss = loss_fn(mean, act_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())
                obs_buf.clear()
                act_buf.clear()

                if verbose and len(losses) % 10 == 0:
                    print(f"[ASImitator] step {step}/{n_steps}  BC loss={losses[-1]:.4f}")

        return losses

    def _reshape_for_transformer(self, obs_flat: torch.Tensor, env) -> torch.Tensor:
        """Reshape flat obs to (B, T_window, feature_dim) for TransformerPolicy."""
        T = env.T_window
        D = env.feature_builder.feature_dim
        return obs_flat.view(obs_flat.shape[0], T, D)
