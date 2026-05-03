"""
Implicit Quantile Network (IQN) critic.

Models the full return distribution Z(s) instead of a scalar value E[Z(s)].
At inference: CVaR_alpha = E[Z | Z < VaR_alpha] estimated from quantile samples.

Same parameter count as a scalar critic at matched hidden dim.
Reference: Dabney et al., "Implicit Quantile Networks for Distributional Reinforcement Learning" (2018).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNCritic(nn.Module):
    """
    IQN value network.

    State encoder: MLP or transformer-provided embedding.
    Quantile embedding: cosine features (Mnih et al., IQN).

    For a given state embedding h (B, d_model) and quantile samples tau (B, N):
      - Embed tau: phi(tau) = ReLU(sum_i cos(pi*i*tau) * w_i)  → (B, N, d_embed)
      - Element-wise multiply: h (B, 1, d_model) * phi (B, N, d_model) → (B, N, d_model)
      - Linear head → Z(s, tau) (B, N)
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 256,
        n_cos: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_cos = n_cos

        # State encoder
        enc_layers = []
        dims = [input_dim] + [hidden] * n_layers
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)

        # Quantile embedding
        self.phi = nn.Sequential(
            nn.Linear(n_cos, hidden),
            nn.ReLU(),
        )

        # Value head
        self.value_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _cos_embed(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: (B, N) in [0, 1]
        Returns: (B, N, hidden) quantile embeddings
        """
        B, N = tau.shape
        # i = 1..n_cos
        i = torch.arange(1, self.n_cos + 1, device=tau.device, dtype=tau.dtype)  # (n_cos,)
        # cos(pi * i * tau): (B, N, n_cos)
        cos_features = torch.cos(math.pi * i.view(1, 1, -1) * tau.unsqueeze(-1))
        return self.phi(cos_features)   # (B, N, hidden)

    def forward(self, obs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, obs_dim) or (B, T, feature_dim) — auto-flattened if 3D
        tau : (B, N) quantile levels in [0, 1]
        Returns: (B, N) quantile values
        """
        if obs.dim() == 3:
            obs = obs.flatten(1)
        h = self.encoder(obs)           # (B, hidden)
        phi = self._cos_embed(tau)      # (B, N, hidden)
        merged = h.unsqueeze(1) * phi   # (B, N, hidden)
        return self.value_head(merged).squeeze(-1)   # (B, N)

    def cvar(
        self,
        obs: torch.Tensor,
        alpha: float = 0.05,
        n_samples: int = 64,
    ) -> torch.Tensor:
        """
        Monte-Carlo CVaR_alpha estimate.
        tau ~ U[0, alpha] → average of lowest quantile values.
        Returns (B,) CVaR values.
        """
        B = obs.shape[0]
        tau = torch.rand(B, n_samples, device=obs.device) * alpha   # (B, N) in [0, alpha]
        quantiles = self(obs, tau)   # (B, N)
        return quantiles.mean(dim=-1)   # (B,)

    def mean_value(self, obs: torch.Tensor, n_samples: int = 32) -> torch.Tensor:
        """Expected value estimate via uniform tau sampling."""
        B = obs.shape[0]
        tau = torch.rand(B, n_samples, device=obs.device)
        return self(obs, tau).mean(dim=-1)

    def quantile_huber_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        tau: torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """
        QR-Huber loss for distributional TD updates.

        predicted : (B, N) quantile estimates at tau
        target    : (B, M) target quantile samples (stop-grad)
        tau       : (B, N) quantile levels used for predicted
        """
        # (B, N, M) pairwise errors
        err = target.unsqueeze(1) - predicted.unsqueeze(2)   # (B, N, M)
        huber = torch.where(err.abs() <= kappa, 0.5 * err ** 2, kappa * (err.abs() - 0.5 * kappa))
        indicator = (err < 0).float()
        weight = (tau.unsqueeze(2) - indicator).abs()   # (B, N, M)
        return (weight * huber).mean()
