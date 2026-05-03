"""MLP actor-critic policy — baseline for ablation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLPPolicy(nn.Module):
    """
    Gaussian actor + scalar-value critic.

    Input: flat observation (obs_dim,)
    Output: action mean (4,), log_std (4,), value (1,)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        hidden: int = 256,
        n_layers: int = 3,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        dims = [obs_dim] + [hidden] * n_layers
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        self.shared = nn.Sequential(*layers)

        self.actor_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.critic_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action_mean, log_std, value). All (B, *)."""
        h = self.shared(obs)
        mean = torch.tanh(self.actor_head(h))
        log_std = self.log_std.expand_as(mean)
        value = self.critic_head(h).squeeze(-1)
        return mean, log_std, value

    def get_dist(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        return dist, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, return (action, log_prob, value)."""
        dist, value = self.get_dist(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1.0, 1.0), log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_prob, entropy, value) for stored (obs, action) pairs."""
        dist, value = self.get_dist(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
