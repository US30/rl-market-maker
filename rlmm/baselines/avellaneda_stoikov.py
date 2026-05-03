"""
Avellaneda-Stoikov (2008) closed-form market-making policy.

Optimal bid/ask quotes under:
  - BM mid-price with constant sigma
  - Poisson order arrivals with intensity lambda(delta) = A * exp(-k * delta)
  - Exponential utility with risk-aversion gamma
  - Finite horizon T

Closed-form solution:
    r*(t) = S(t) - q * gamma * sigma^2 * (T - t)       [indifference price]
    s*(t) = gamma * sigma^2 * (T - t) + (2/gamma)*ln(1 + gamma/k)   [optimal spread]
    bid*(t) = r*(t) - s*(t)/2
    ask*(t) = r*(t) + s*(t)/2
"""

from __future__ import annotations

import math
import numpy as np


class AvellanedaStoikov:
    """
    Parameters
    ----------
    gamma   : risk-aversion coefficient (inventory penalty scale)
    k       : order-arrival decay parameter (higher k = steeper dropoff)
    sigma   : mid-price volatility (if None, uses sigma passed at call time)
    T       : episode horizon in same units as tau
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float | None = None,
        T: float = 1.0,
    ):
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T

    def quote(
        self,
        mid: float,
        inventory: float,
        tau: float,
        sigma: float | None = None,
    ) -> tuple[float, float]:
        """
        Returns (bid_price, ask_price).

        tau : time remaining normalised to [0, 1] (1=start, 0=end)
        """
        sig = sigma if sigma is not None else (self.sigma or 0.02)
        t_rem = tau * self.T

        reservation = mid - inventory * self.gamma * sig ** 2 * t_rem
        spread = self.gamma * sig ** 2 * t_rem + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        spread = max(spread, 2e-4)   # floor at 2 ticks equivalent

        bid = reservation - spread / 2.0
        ask = reservation + spread / 2.0
        return bid, ask

    def action(
        self,
        mid: float,
        inventory: float,
        tau: float,
        sigma: float | None,
        tick_size: float = 0.01,
        max_offset_ticks: int = 10,
        default_size_norm: float = 0.0,   # -1..1 → decoded to size in env
    ) -> np.ndarray:
        """
        Return action in the env's normalised [-1, 1]^4 space.
        (bid_offset_norm, ask_offset_norm, skew_norm=0, size_norm)
        """
        bid_price, ask_price = self.quote(mid, inventory, tau, sigma)
        bid_offset = max(mid - bid_price, tick_size)
        ask_offset = max(ask_price - mid, tick_size)

        # Convert ticks to [-1, 1]: offset_ticks ∈ [1, 10] → norm ∈ [-1, 1]
        # inverse of: offset_ticks = (norm+1)*2.5 + 0.5  ⟹  norm = (ticks - 0.5)/2.5 - 1
        bid_ticks = np.clip(bid_offset / tick_size, 1, max_offset_ticks)
        ask_ticks = np.clip(ask_offset / tick_size, 1, max_offset_ticks)
        bid_norm = float(np.clip((bid_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))
        ask_norm = float(np.clip((ask_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))

        # Inventory skew: positive inventory → positive skew (widen bid)
        skew_norm = float(np.clip(inventory / 50.0, -1.0, 1.0))

        return np.array([bid_norm, ask_norm, skew_norm, default_size_norm], dtype=np.float32)
