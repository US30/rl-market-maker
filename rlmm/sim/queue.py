"""
Queue-position dynamics and fill/cancellation probability model.

Fill probability depends on:
  - How much volume is ahead of the agent in the queue
  - Book imbalance (correlated with direction of next MO)
  - Remaining time horizon

Cancellation hazard depends on:
  - Queue position (deep-in-queue orders are more likely to cancel before fill)
  - Book imbalance (imbalance toward same side → less likely to cancel)
"""

from __future__ import annotations

import numpy as np


class QueueModel:
    """
    Simple parametric model for fill/cancel probabilities.

    Parameters
    ----------
    base_fill_rate    : base fills per unit time when at front of queue
    queue_decay       : exponential decay in fill prob per unit of queue ahead
    imbalance_fill    : imbalance contribution to fill rate (positive = favorable)
    base_cancel_rate  : base cancellation hazard per unit time
    queue_cancel_scale: cancellation hazard scales up with queue depth
    """

    def __init__(
        self,
        base_fill_rate: float = 2.0,
        queue_decay: float = 0.1,
        imbalance_fill: float = 1.0,
        base_cancel_rate: float = 0.5,
        queue_cancel_scale: float = 0.05,
    ):
        self.base_fill_rate = base_fill_rate
        self.queue_decay = queue_decay
        self.imbalance_fill = imbalance_fill
        self.base_cancel_rate = base_cancel_rate
        self.queue_cancel_scale = queue_cancel_scale

    def fill_intensity(
        self,
        queue_ahead: float,
        imbalance: float,
        side: int,
    ) -> float:
        """
        Instantaneous fill rate (events/sec) for an order with `queue_ahead` volume ahead.

        imbalance in [-1, 1]: positive = more bid volume.
        For a bid order (side=0), positive imbalance is unfavorable (MOs less likely to sell).
        For an ask order (side=1), positive imbalance is favorable (MOs more likely to sell/hit ask).
        """
        # Signed imbalance from the perspective of the passive side
        signed_imb = imbalance if side == 1 else -imbalance
        rate = (
            self.base_fill_rate
            * np.exp(-self.queue_decay * max(queue_ahead, 0.0))
            * (1.0 + self.imbalance_fill * np.clip(signed_imb, -1.0, 1.0))
        )
        return max(rate, 0.0)

    def cancel_intensity(self, queue_ahead: float) -> float:
        """Cancellation hazard rate (events/sec)."""
        return self.base_cancel_rate + self.queue_cancel_scale * max(queue_ahead, 0.0)

    def fill_probability(
        self,
        queue_ahead: float,
        imbalance: float,
        side: int,
        dt: float,
    ) -> float:
        """P(fill before cancel in dt) using competing-hazard approximation."""
        lam_fill = self.fill_intensity(queue_ahead, imbalance, side)
        lam_cancel = self.cancel_intensity(queue_ahead)
        total = lam_fill + lam_cancel
        if total <= 0:
            return 0.0
        # P(next event is fill) * P(some event in dt)
        p_fill_wins = lam_fill / total
        p_event = 1.0 - np.exp(-total * dt)
        return float(p_fill_wins * p_event)

    def sample_fill(
        self,
        queue_ahead: float,
        imbalance: float,
        side: int,
        dt: float,
        rng: np.random.Generator,
    ) -> bool:
        """Stochastic fill outcome for agent order over interval dt."""
        return rng.random() < self.fill_probability(queue_ahead, imbalance, side, dt)
