"""
Cartea-Jaimungal (2015) market-making policy.

Extension incorporating:
  - Adverse selection cost: fills are more likely when mid-price moves against us
  - Running inventory penalty (same as A-S risk-aversion term)
  - Explicit alpha parameter capturing probability that an MO is informed

From Cartea, Jaimungal, Penalva (2015), "Algorithmic and High-Frequency Trading", Ch. 10:

    h*(q, t) ≈ (1/k) * ln(1 + gamma/k)
              + gamma * sigma^2 * (T-t)
              - alpha * sigma * sqrt(T-t) * q / q_max   [adverse selection term]

    Reservation price: r* = S - (2q+1)/2 * gamma * sigma^2 * (T-t)

This is a simplified closed-form approximation; the exact solution requires PDE numerics.
"""

from __future__ import annotations

import math
import numpy as np


class CarteaJaimungal:
    """
    Parameters
    ----------
    gamma   : risk-aversion (inventory penalty)
    k       : order-arrival shape
    alpha   : adverse-selection coefficient
    sigma   : volatility (optional override)
    T       : horizon
    q_max   : normalisation for inventory term
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        alpha: float = 0.3,
        sigma: float | None = None,
        T: float = 1.0,
        q_max: float = 50.0,
    ):
        self.gamma = gamma
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
        self.T = T
        self.q_max = q_max

    def quote(
        self,
        mid: float,
        inventory: float,
        tau: float,
        sigma: float | None = None,
    ) -> tuple[float, float]:
        """Returns (bid_price, ask_price)."""
        sig = sigma if sigma is not None else (self.sigma or 0.02)
        t_rem = max(tau * self.T, 1e-6)
        q = float(np.clip(inventory, -self.q_max, self.q_max))

        # Reservation price
        reservation = mid - (2 * q + 1) / 2.0 * self.gamma * sig ** 2 * t_rem

        # Half-spread with adverse selection correction
        half_spread_base = (1.0 / self.k) * math.log(1.0 + self.gamma / self.k)
        adv_sel_correction = self.alpha * sig * math.sqrt(t_rem) * abs(q) / (self.q_max + 1e-6)
        half_spread = max(half_spread_base + adv_sel_correction, 1e-4)

        bid = reservation - half_spread
        ask = reservation + half_spread
        return bid, ask

    def action(
        self,
        mid: float,
        inventory: float,
        tau: float,
        sigma: float | None,
        tick_size: float = 0.01,
        max_offset_ticks: int = 10,
    ) -> np.ndarray:
        """Return normalised action in [-1, 1]^4."""
        bid, ask = self.quote(mid, inventory, tau, sigma)
        bid_ticks = np.clip((mid - bid) / tick_size, 1, max_offset_ticks)
        ask_ticks = np.clip((ask - mid) / tick_size, 1, max_offset_ticks)
        bid_norm = float(np.clip((bid_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))
        ask_norm = float(np.clip((ask_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))
        skew_norm = float(np.clip(inventory / self.q_max, -1.0, 1.0))
        return np.array([bid_norm, ask_norm, skew_norm, 0.0], dtype=np.float32)
