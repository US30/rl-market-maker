"""
Multivariate Hawkes process with exponential decay kernels.

4 event types: 0=MO-buy, 1=MO-sell, 2=LO-buy, 3=LO-sell.

Intensity of type m at time t:
    lambda_m(t) = mu_m + sum_{t_i < t} alpha_{m, type_i} * exp(-beta_m * (t - t_i))

Simulation via Ogata's thinning algorithm.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple


# Event types
MO_BUY  = 0
MO_SELL = 1
LO_BUY  = 2
LO_SELL = 3
N_TYPES = 4


class Event(NamedTuple):
    time: float
    etype: int   # 0-3


class MultivariateHawkes:
    """
    Parameters
    ----------
    mu    : (4,) baseline intensities [events/sec]
    alpha : (4, 4) excitation matrix; alpha[m, j] = excitation of type m by type j
    beta  : (4,) decay rates [1/sec]
    """

    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
    ):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.beta = np.asarray(beta, dtype=np.float64)
        assert self.mu.shape == (N_TYPES,)
        assert self.alpha.shape == (N_TYPES, N_TYPES)
        assert self.beta.shape == (N_TYPES,)

    # ------------------------------------------------------------------
    # Default calibration-free params (reasonable for equities, ~1s tick)
    # ------------------------------------------------------------------

    @classmethod
    def default_params(cls) -> "MultivariateHawkes":
        mu = np.array([2.0, 2.0, 5.0, 5.0])         # LOs arrive more often
        # Symmetric self-excitation; cross-excitation MO<->MO (adverse sel)
        alpha = np.array([
            [0.3, 0.1, 0.05, 0.05],  # MO-buy excited by MO-buy, MO-sell, LO-buy, LO-sell
            [0.1, 0.3, 0.05, 0.05],
            [0.05, 0.05, 0.3, 0.1],
            [0.05, 0.05, 0.1, 0.3],
        ])
        beta = np.array([5.0, 5.0, 3.0, 3.0])
        return cls(mu, alpha, beta)

    @classmethod
    def high_excitation_params(cls) -> "MultivariateHawkes":
        """High-vol regime with stronger self-excitation."""
        mu = np.array([3.0, 3.0, 6.0, 6.0])
        alpha = np.array([
            [0.6, 0.2, 0.1, 0.1],
            [0.2, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.2],
            [0.1, 0.1, 0.2, 0.5],
        ])
        beta = np.array([5.0, 5.0, 3.0, 3.0])
        return cls(mu, alpha, beta)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, T_max: float, seed: int | None = None) -> list[Event]:
        """
        Ogata's thinning algorithm.

        Returns list of Events sorted by time, up to T_max.
        """
        rng = np.random.default_rng(seed)
        events: list[Event] = []

        # Running sum of exp-weighted history per (m, past_event)
        # Maintained as R[m] = sum_{i} alpha[m, type_i] * exp(-beta[m]*(t - t_i))
        R = np.zeros(N_TYPES)
        t = 0.0

        while t < T_max:
            # Upper bound on total intensity
            lam_bar = (self.mu + R).sum()
            if lam_bar <= 0:
                break

            # Draw next candidate event time
            dt = rng.exponential(1.0 / lam_bar)
            t_cand = t + dt

            if t_cand > T_max:
                break

            # Decay R to t_cand
            decay = np.exp(-self.beta * dt)
            R = R * decay

            # Compute intensities at t_cand
            lam = self.mu + R
            lam = np.maximum(lam, 0.0)

            # Thinning: accept with prob sum(lam)/lam_bar
            if rng.random() < lam.sum() / lam_bar:
                # Choose event type proportional to lam
                probs = lam / lam.sum()
                etype = int(rng.choice(N_TYPES, p=probs))

                events.append(Event(t_cand, etype))
                # Update R: add excitation from this event
                R += self.alpha[:, etype]

            t = t_cand

        return events

    # ------------------------------------------------------------------
    # Intensity query (for calibration / notebooks)
    # ------------------------------------------------------------------

    def intensity_at(self, t: float, history: list[Event]) -> np.ndarray:
        """Compute lambda(t) given event history before t."""
        R = np.zeros(N_TYPES)
        for ev in history:
            if ev.time >= t:
                break
            decay = np.exp(-self.beta * (t - ev.time))
            R += self.alpha[:, ev.etype] * decay
        return self.mu + R

    # ------------------------------------------------------------------
    # Branching ratio (stationarity check: spectral radius < 1)
    # ------------------------------------------------------------------

    def branching_ratio(self) -> float:
        """Spectral radius of alpha / beta elementwise. < 1 for stationarity."""
        G = self.alpha / self.beta[:, None]
        return float(np.linalg.eigvals(G).real.max())

    def is_stationary(self) -> bool:
        return self.branching_ratio() < 1.0
