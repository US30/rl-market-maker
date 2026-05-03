"""
Feature builder: LOB snapshot history + scalars → observation tensor.

Observation layout:
  (T_window, 2*n_levels)  — normalized bid/ask sizes at each historical step
  + scalar features appended as extra columns at each time step:
      inventory q (clipped, normalized)
      time-to-horizon tau (normalized 0..1)
      realized vol sigma_hat (normalized)
      order-flow imbalance (in [-1, 1])

Final obs shape: (T_window, 2*n_levels + 4)  — flattened to (T_window * (2*n_levels + 4),)
for MLP policy, or kept as 2D for transformer.
"""

from __future__ import annotations

from collections import deque

import numpy as np


N_SCALARS = 4


class FeatureBuilder:
    """
    Rolling window of book snapshots + scalar state features.

    Call update() each env step, then get_obs() for the current observation.
    """

    def __init__(
        self,
        n_levels: int = 10,
        T_window: int = 20,
        max_inventory: float = 100.0,
        max_sigma: float = 0.2,
    ):
        self.n_levels = n_levels
        self.T_window = T_window
        self.max_inventory = max_inventory
        self.max_sigma = max_sigma

        self.feature_dim = 2 * n_levels + N_SCALARS
        self.obs_dim = T_window * self.feature_dim

        # Ring buffer of (2*n_levels + 4,) rows
        self._window: deque[np.ndarray] = deque(maxlen=T_window)

    def reset(self) -> None:
        self._window.clear()

    def update(
        self,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
        inventory: float,
        tau: float,
        sigma_hat: float,
        imbalance: float,
    ) -> None:
        """Append one time step of features to rolling window."""
        # Normalise book sizes by total visible depth
        total = bid_sizes.sum() + ask_sizes.sum()
        if total > 0:
            bids_norm = bid_sizes / total
            asks_norm = ask_sizes / total
        else:
            bids_norm = bid_sizes
            asks_norm = ask_sizes

        # Normalise scalars to roughly [-1, 1]
        q_norm = np.clip(inventory / self.max_inventory, -1.0, 1.0)
        tau_norm = np.clip(tau, 0.0, 1.0)
        sig_norm = np.clip(sigma_hat / self.max_sigma, 0.0, 1.0)
        imb = np.clip(imbalance, -1.0, 1.0)

        row = np.concatenate([
            bids_norm.astype(np.float32),
            asks_norm.astype(np.float32),
            np.array([q_norm, tau_norm, sig_norm, imb], dtype=np.float32),
        ])
        self._window.append(row)

    def get_obs_2d(self) -> np.ndarray:
        """Returns (T_window, feature_dim) — padded with zeros if window not full."""
        pad = self.T_window - len(self._window)
        rows = list(self._window)
        if pad > 0:
            zeros = [np.zeros(self.feature_dim, dtype=np.float32)] * pad
            rows = zeros + rows
        return np.stack(rows, axis=0)   # (T_window, feature_dim)

    def get_obs_flat(self) -> np.ndarray:
        """Returns (T_window * feature_dim,) flattened observation."""
        return self.get_obs_2d().reshape(-1)
