"""
Guéant-Lehalle-Tapia (2012) market-making policy.

Extension of A-S with:
  - Finite inventory bounds [-q_max, q_max]
  - Exponential utility with exact PDE solution
  - Closed-form approximation (Appendix A of Guéant, Lehalle, Tapia 2012)

Approximate optimal spread (Eq. 20 in GLT):
    s*(q, t) ≈ gamma * sigma^2 * (T-t) * (1 - q^2 / q_max^2)
              + (2/gamma) * ln(1 + gamma/k)
    with inventory-adjusted reservation price r* as in A-S.
"""

from __future__ import annotations

import math
import numpy as np


class GueantLehalleTapia:
    """
    Parameters
    ----------
    gamma   : risk-aversion
    k       : order-arrival decay
    sigma   : volatility (optional override per call)
    T       : horizon
    q_max   : max inventory (hard clamp)
    """

    def __init__(
        self,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float | None = None,
        T: float = 1.0,
        q_max: float = 50.0,
    ):
        self.gamma = gamma
        self.k = k
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
        t_rem = tau * self.T
        q = np.clip(inventory, -self.q_max, self.q_max)

        reservation = mid - q * self.gamma * sig ** 2 * t_rem

        # GLT spread: tighter when inventory near bounds
        inv_factor = max(0.0, 1.0 - (q / self.q_max) ** 2)
        spread = (
            self.gamma * sig ** 2 * t_rem * inv_factor
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )
        spread = max(spread, 2e-4)

        # One-sided quote suppression near inventory limits
        if q >= self.q_max * 0.9:
            # Very long: suppress bid (don't buy more)
            bid = reservation - spread * 2
            ask = reservation + spread / 2
        elif q <= -self.q_max * 0.9:
            # Very short: suppress ask (don't sell more)
            bid = reservation - spread / 2
            ask = reservation + spread * 2
        else:
            bid = reservation - spread / 2
            ask = reservation + spread / 2

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
