"""
Transformer actor-critic policy — main contribution, ported from deep_hedging_transformer.

Architecture:
  - Input projection: feature_dim → d_model
  - N causal transformer blocks with RoPE (from sibling project)
  - Pool last token → actor head (4-dim Gaussian) + critic head (scalar or IQN)

Input shape: (B, T_window, feature_dim)
Action: (B, 4) clipped to [-1, 1] via tanh
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ------------------------------------------------------------------
# RoPE helpers (verbatim from deep_hedging_transformer)
# ------------------------------------------------------------------

def _rotary_emb(dim: int, seq_len: int, device: torch.device):
    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.sin(), freqs.cos()


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, sin, cos):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, sin, cos):
        x = x + self.attn(self.norm1(x), sin, cos)
        x = x + self.ffn(self.norm2(x))
        return x


# ------------------------------------------------------------------
# Main policy
# ------------------------------------------------------------------

class TransformerPolicy(nn.Module):
    """
    Causal transformer acting as actor-critic for the MM environment.

    Unlike TransformerHedger (which produces per-step outputs), this policy
    pools the last-token representation to produce a single action per episode step.
    The full T_window of book snapshots provides temporal context.

    input_dim   : feature_dim = 2*n_levels + 4 scalars
    action_dim  : 4 (bid_off, ask_off, skew, size)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 64,
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.actor_head = nn.Linear(d_model, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.critic_head = nn.Linear(d_model, 1)

        sin, cos = _rotary_emb(d_model // n_heads, max_seq_len, torch.device("cpu"))
        self.register_buffer("rope_sin", sin)
        self.register_buffer("rope_cos", cos)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)

    def _get_rope(self, T: int, device: torch.device):
        if T > self.rope_sin.shape[0]:
            sin, cos = _rotary_emb(self.d_model // self.n_heads, T, device)
            self.rope_sin = sin
            self.rope_cos = cos
        return self.rope_sin[:T].to(device), self.rope_cos[:T].to(device)

    def _encode(self, obs_2d: torch.Tensor) -> torch.Tensor:
        """obs_2d: (B, T, input_dim) → last-token repr (B, d_model)."""
        B, T, _ = obs_2d.shape
        x = self.input_proj(obs_2d)
        sin, cos = self._get_rope(T, obs_2d.device)
        for block in self.blocks:
            x = block(x, sin, cos)
        x = self.norm(x)
        return x[:, -1, :]   # last token

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: (B, T, input_dim) 2D, or (B, T*input_dim) flat — auto-detected.
        Returns (action_mean, log_std, value).
        """
        if obs.dim() == 2:
            # Flat: infer T from self.max_seq_len — caller must set input_dim correctly
            raise ValueError("Pass 2D obs (B, T, input_dim) to TransformerPolicy.")
        h = self._encode(obs)
        mean = torch.tanh(self.actor_head(h))
        log_std = self.log_std.expand_as(mean)
        value = self.critic_head(h).squeeze(-1)
        return mean, log_std, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1.0, 1.0), log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
