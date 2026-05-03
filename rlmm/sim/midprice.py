"""
Mid-price dynamics: Brownian motion with imbalance-driven drift + jump component.

The mid-price is NOT an exogenous GBM; it moves because of order-flow imbalance.
Large buy MOs push mid up; large sell MOs push mid down.
Small BM noise captures residual uncertainty.
"""

from __future__ import annotations

import numpy as np


class MidPriceProcess:
    """
    Discrete-time mid-price update rule:

        dS = sigma * sqrt(dt) * dW  +  kappa * imbalance * dt  +  jump * N(jump_prob)

    Parameters
    ----------
    sigma           : diffusion coefficient (price units / sqrt(sec))
    kappa           : drift coefficient from order-flow imbalance
    jump_scale      : size of jumps (in ticks)
    jump_prob       : probability of a jump per unit time
    tick_size       : price grid
    """

    def __init__(
        self,
        sigma: float = 0.02,
        kappa: float = 0.005,
        jump_scale: float = 0.03,
        jump_prob: float = 0.1,
        tick_size: float = 0.01,
    ):
        self.sigma = sigma
        self.kappa = kappa
        self.jump_scale = jump_scale
        self.jump_prob = jump_prob
        self.tick_size = tick_size

    def step(
        self,
        mid: float,
        imbalance: float,
        dt: float,
        rng: np.random.Generator,
    ) -> float:
        """Update mid-price by one time step dt."""
        diffusion = self.sigma * np.sqrt(dt) * rng.standard_normal()
        drift = self.kappa * imbalance * dt
        jump = 0.0
        if rng.random() < self.jump_prob * dt:
            jump = self.jump_scale * rng.choice([-1.0, 1.0])
        new_mid = mid + diffusion + drift + jump
        return round(new_mid / self.tick_size) * self.tick_size

    def simulate(
        self,
        S0: float,
        imbalances: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Simulate full path given imbalance series.

        Parameters
        ----------
        S0          : initial mid-price
        imbalances  : (T,) imbalance at each step
        dt          : time step size (seconds)

        Returns
        -------
        mids : (T+1,) mid-price path
        """
        T = len(imbalances)
        mids = np.empty(T + 1, dtype=np.float32)
        mids[0] = S0
        for i in range(T):
            mids[i + 1] = self.step(mids[i], imbalances[i], dt, rng)
        return mids
