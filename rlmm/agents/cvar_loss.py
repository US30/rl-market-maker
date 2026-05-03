"""
CVaR-based loss functions for RL — ported from deephedge/train/losses.py.

Used to replace the standard mean-reward PPO objective with a tail-risk objective.
"""

import torch


def cvar_loss(rewards: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR_alpha(-rewards): average of worst (1-alpha) fraction of episodes.
    Differentiable sorted approximation.
    """
    losses = -rewards
    n = losses.shape[0]
    k = max(1, int(n * alpha))
    sorted_loss, _ = losses.sort(descending=True)
    return sorted_loss[:k].mean()


def cvar_rockafellar(rewards: torch.Tensor, z: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Rockafellar-Uryasev dual: CVaR = min_z { z + E[max(-r - z, 0)] / alpha }
    z is a trainable scalar for joint optimisation.
    """
    excess = (-rewards - z).clamp(min=0.0)
    return z + excess.mean() / alpha


def ppo_mean_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Standard PPO-clip policy gradient (mean-advantage baseline)."""
    ratio = (log_probs - old_log_probs).exp()
    clip = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.min(ratio * advantages, clip * advantages).mean()


def ppo_cvar_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    episode_returns: torch.Tensor,
    clip_eps: float = 0.2,
    alpha: float = 0.05,
    cvar_coeff: float = 0.5,
) -> torch.Tensor:
    """
    PPO loss with CVaR regularisation on episode returns.

    Total = PPO_clip_loss + cvar_coeff * CVaR_alpha(-returns)

    The CVaR term penalises tail episodes, pushing the policy away from
    catastrophic inventory drawdowns even when mean PnL is acceptable.
    """
    pg_loss = ppo_mean_loss(log_probs, old_log_probs, advantages, clip_eps)
    cvar_reg = cvar_loss(episode_returns, alpha)
    return pg_loss + cvar_coeff * cvar_reg


def make_rl_loss(name: str, **kwargs):
    """Factory matching deephedge convention."""
    if name == "mean":
        return lambda lp, olp, adv, ret: ppo_mean_loss(lp, olp, adv, kwargs.get("clip_eps", 0.2))
    if name == "cvar":
        return lambda lp, olp, adv, ret: ppo_cvar_loss(
            lp, olp, adv, ret,
            clip_eps=kwargs.get("clip_eps", 0.2),
            alpha=kwargs.get("alpha", 0.05),
            cvar_coeff=kwargs.get("cvar_coeff", 0.5),
        )
    raise ValueError(f"Unknown RL loss: {name}")
