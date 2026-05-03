"""
Order-flow sampler: combines Hawkes arrivals + LOB matching + queue dynamics.

Drives a complete episode of market activity that the MM agent operates in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .hawkes import MultivariateHawkes, Event, MO_BUY, MO_SELL, LO_BUY, LO_SELL
from .lob import LOB, Fill
from .queue import QueueModel
from .midprice import MidPriceProcess


@dataclass
class FlowParams:
    """Hyperparameters for the order-flow simulation."""
    # LOB init
    n_levels: int = 10
    tick_size: float = 0.01
    initial_spread_ticks: int = 2       # initial book spread in ticks
    initial_depth_per_level: float = 10.0  # shares per level at init

    # Market order sizes
    mo_size_mean: float = 5.0
    mo_size_std: float = 2.0

    # Limit order sizes and placement
    lo_size_mean: float = 8.0
    lo_size_std: float = 3.0
    lo_offset_mean_ticks: int = 1      # mean depth at which LOs arrive
    lo_offset_std_ticks: float = 1.0

    # Time step
    dt: float = 0.1   # seconds per RL env step

    # Episode
    T_episode: float = 60.0   # seconds


class OrderFlowSampler:
    """
    Drives a full episode of LOB activity using Hawkes + QueueModel + MidPriceProcess.

    At each time step (dt), processes all events that arrived in [t, t+dt]:
      - LO events add resting orders to the book
      - MO events match against the book and fill resting orders
      - Imbalance is computed and fed to mid-price updater

    The RL environment calls step() once per time step.
    """

    def __init__(
        self,
        hawkes: Optional[MultivariateHawkes] = None,
        queue_model: Optional[QueueModel] = None,
        mid_process: Optional[MidPriceProcess] = None,
        params: Optional[FlowParams] = None,
        seed: int = 0,
    ):
        self.hawkes = hawkes or MultivariateHawkes.default_params()
        self.queue_model = queue_model or QueueModel()
        self.mid_process = mid_process or MidPriceProcess()
        self.params = params or FlowParams()
        self.rng = np.random.default_rng(seed)

        self.lob = LOB(tick_size=self.params.tick_size, n_levels=self.params.n_levels)
        self.mid: float = 100.0
        self._events: list[Event] = []
        self._event_idx: int = 0
        self._t: float = 0.0

    def reset(self, mid: float = 100.0, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mid = mid
        self.lob.reset()
        self._t = 0.0

        # Pre-populate book with symmetric depth
        p = self.params
        for k in range(1, p.n_levels + 1):
            bid_price = mid - k * p.tick_size
            ask_price = mid + k * p.tick_size
            size = max(1.0, p.initial_depth_per_level + self.rng.normal(0, 2.0))
            self.lob.submit_limit(0, bid_price, size)
            self.lob.submit_limit(1, ask_price, size)

        # Pre-simulate Hawkes events for the full episode
        self._events = self.hawkes.simulate(self.params.T_episode, seed=int(self.rng.integers(1 << 31)))
        self._event_idx = 0

    def step(self, agent_bid_id: Optional[int] = None, agent_ask_id: Optional[int] = None) -> dict:
        """
        Advance simulation by one dt.

        Returns
        -------
        dict with keys:
          mid        : float, current mid-price
          imbalance  : float in [-1, 1]
          agent_fills: list[Fill] — fills on agent's resting orders
          snapshot   : (bid_sizes, ask_sizes) each np.ndarray(n_levels,)
          t          : current simulation time
        """
        p = self.params
        t_end = self._t + p.dt
        self.lob.clear_fills()

        imbalance_acc = []

        while self._event_idx < len(self._events):
            ev = self._events[self._event_idx]
            if ev.time > t_end:
                break
            self._event_idx += 1

            if ev.etype in (MO_BUY, MO_SELL):
                mo_size = max(1.0, self.rng.normal(p.mo_size_mean, p.mo_size_std))
                mo_side = 0 if ev.etype == MO_BUY else 1
                self.lob.submit_market(mo_side, mo_size)

            elif ev.etype in (LO_BUY, LO_SELL):
                lo_size = max(1.0, self.rng.normal(p.lo_size_mean, p.lo_size_std))
                lo_side = 0 if ev.etype == LO_BUY else 1
                offset_ticks = max(1, int(round(self.rng.normal(p.lo_offset_mean_ticks, p.lo_offset_std_ticks))))
                best = self.lob.best_bid() if lo_side == 0 else self.lob.best_ask()
                if best is not None:
                    lo_price = best + offset_ticks * p.tick_size if lo_side == 0 else best - offset_ticks * p.tick_size
                else:
                    lo_price = self.mid + (offset_ticks if lo_side == 1 else -offset_ticks) * p.tick_size
                self.lob.submit_limit(lo_side, lo_price, lo_size)

            imbalance_acc.append(self.lob.order_imbalance())

        imbalance = float(np.mean(imbalance_acc)) if imbalance_acc else self.lob.order_imbalance()

        # Update mid-price
        self.mid = self.mid_process.step(self.mid, imbalance, p.dt, self.rng)

        # Collect fills on agent's resting orders
        agent_fills = [f for f in self.lob.fills if f.is_agent_passive]

        # Replenish book levels if depleted (maintain depth)
        self._replenish()

        self._t = t_end

        return {
            "mid": self.mid,
            "imbalance": imbalance,
            "agent_fills": agent_fills,
            "snapshot": self.lob.get_snapshot(),
            "t": self._t,
        }

    def _replenish(self) -> None:
        """Add passive depth if top N levels become empty."""
        p = self.params
        for side in (0, 1):
            for k in range(1, 4):
                if side == 0:
                    price = self.mid - k * p.tick_size
                    has_orders = price in self.lob._bids
                else:
                    price = self.mid + k * p.tick_size
                    has_orders = price in self.lob._asks
                if not has_orders:
                    size = max(1.0, p.initial_depth_per_level * 0.5 + self.rng.normal(0, 1.0))
                    self.lob.submit_limit(side, price, size)

    @property
    def done(self) -> bool:
        return self._t >= self.params.T_episode
