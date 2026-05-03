"""
Market Making gymnasium environment.

Action space (continuous Box):
    [bid_offset, ask_offset, inventory_skew, size]
    bid_offset, ask_offset in [0, 10] ticks from mid
    inventory_skew in [-1, 1]: positive = widen bid more (reduce buy pressure when long)
    size in [1, 20] shares

Observation:
    Flattened LOB snapshot window + scalars — see features.py.

Reward:
    ΔPnL (mark-to-market + realised fills) − gamma * q^2 − lambda_adv * adverse_sel_proxy − fee * turnover
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..sim.flow import OrderFlowSampler, FlowParams
from ..sim.hawkes import MultivariateHawkes
from ..sim.queue import QueueModel
from ..sim.midprice import MidPriceProcess
from .features import FeatureBuilder


class MarketMakingEnv(gym.Env):
    """
    Gymnasium environment for RL market-making.

    Parameters
    ----------
    n_levels        : LOB depth visible to agent
    T_window        : history window length (obs has T_window time steps)
    tick_size       : price grid
    T_episode_sec   : episode length in seconds
    dt              : time step in seconds
    gamma_inv       : inventory penalty coefficient
    lambda_adv      : adverse-selection penalty coefficient (on imbalance × inventory)
    fee_rate        : transaction cost per unit traded
    max_inventory   : hard inventory clamp (shares)
    vol_regime      : 'low' | 'high' | 'hawkes_high' — controls sim params
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_levels: int = 10,
        T_window: int = 20,
        tick_size: float = 0.01,
        T_episode_sec: float = 60.0,
        dt: float = 0.1,
        gamma_inv: float = 0.001,
        lambda_adv: float = 0.0005,
        fee_rate: float = 0.0001,
        max_inventory: float = 100.0,
        vol_regime: str = "low",
        seed: int = 0,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.T_window = T_window
        self.tick_size = tick_size
        self.T_episode_sec = T_episode_sec
        self.dt = dt
        self.gamma_inv = gamma_inv
        self.lambda_adv = lambda_adv
        self.fee_rate = fee_rate
        self.max_inventory = max_inventory
        self.vol_regime = vol_regime
        self._seed = seed

        self.feature_builder = FeatureBuilder(n_levels, T_window, max_inventory)
        self.obs_dim = self.feature_builder.obs_dim

        # Action: (bid_offset, ask_offset, skew, size) — normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._sampler: Optional[OrderFlowSampler] = None
        self._init_sampler(seed)

        # Episode state
        self.inventory: float = 0.0
        self.cash: float = 0.0
        self.prev_mid: float = 100.0
        self.agent_bid_id: Optional[int] = None
        self.agent_ask_id: Optional[int] = None
        self._step_count: int = 0
        self._max_steps: int = int(T_episode_sec / dt)

        # Realized-spread tracking
        self._total_filled_bid_value: float = 0.0
        self._total_filled_ask_value: float = 0.0
        self._total_filled_size: float = 0.0

    # ------------------------------------------------------------------
    # Sampler factory
    # ------------------------------------------------------------------

    def _init_sampler(self, seed: int) -> None:
        if self.vol_regime == "hawkes_high":
            hawkes = MultivariateHawkes.high_excitation_params()
        else:
            hawkes = MultivariateHawkes.default_params()

        if self.vol_regime == "low":
            mid_proc = MidPriceProcess(sigma=0.01, kappa=0.003)
        elif self.vol_regime == "high":
            mid_proc = MidPriceProcess(sigma=0.03, kappa=0.008, jump_prob=0.2)
        else:
            mid_proc = MidPriceProcess(sigma=0.04, kappa=0.01, jump_prob=0.3)

        fp = FlowParams(
            n_levels=self.n_levels,
            tick_size=self.tick_size,
            dt=self.dt,
            T_episode=self.T_episode_sec,
        )
        self._sampler = OrderFlowSampler(
            hawkes=hawkes,
            queue_model=QueueModel(),
            mid_process=mid_proc,
            params=fp,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._seed = seed

        ep_seed = int(np.random.default_rng(self._seed).integers(1 << 31))
        self._sampler.reset(mid=100.0, seed=ep_seed)
        self._seed += 1   # vary each episode

        self.inventory = 0.0
        self.cash = 0.0
        self.prev_mid = self._sampler.mid
        self.agent_bid_id = None
        self.agent_ask_id = None
        self._step_count = 0
        self._total_filled_bid_value = 0.0
        self._total_filled_ask_value = 0.0
        self._total_filled_size = 0.0
        self.feature_builder.reset()

        # Warm-up: fill observation window with no-agent steps
        for _ in range(self.T_window):
            result = self._sampler.step()
            bid_s, ask_s = result["snapshot"]
            self.feature_builder.update(
                bid_s, ask_s,
                inventory=0.0,
                tau=1.0,
                sigma_hat=0.0,
                imbalance=result["imbalance"],
            )

        return self.feature_builder.get_obs_flat(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        action: np.ndarray(4,) in [-1, 1]:
            [bid_offset_norm, ask_offset_norm, inv_skew_norm, size_norm]
        """
        # Cancel previous agent orders
        if self.agent_bid_id is not None:
            self._sampler.lob.cancel(self.agent_bid_id)
            self.agent_bid_id = None
        if self.agent_ask_id is not None:
            self._sampler.lob.cancel(self.agent_ask_id)
            self.agent_ask_id = None

        # Decode action
        bid_offset_ticks = int(np.clip((action[0] + 1.0) * 2.5 + 0.5, 1, 10))
        ask_offset_ticks = int(np.clip((action[1] + 1.0) * 2.5 + 0.5, 1, 10))
        inv_skew = float(np.clip(action[2], -1.0, 1.0))
        size = float(np.clip((action[3] + 1.0) * 9.5 + 1.0, 1.0, 20.0))

        # Adjust offsets by inventory skew (positive inventory → widen bid, tighten ask)
        bid_off_adj = max(1, bid_offset_ticks + int(inv_skew * 2))
        ask_off_adj = max(1, ask_offset_ticks - int(inv_skew * 2))

        mid = self._sampler.mid
        bid_price = mid - bid_off_adj * self.tick_size
        ask_price = mid + ask_off_adj * self.tick_size

        # Submit agent orders (if within inventory limits)
        if self.inventory > -self.max_inventory:
            self.agent_bid_id = self._sampler.lob.submit_limit(0, bid_price, size, is_agent=True)
        if self.inventory < self.max_inventory:
            self.agent_ask_id = self._sampler.lob.submit_limit(1, ask_price, size, is_agent=True)

        # Advance simulation
        result = self._sampler.step(self.agent_bid_id, self.agent_ask_id)
        new_mid = result["mid"]
        imbalance = result["imbalance"]
        bid_s, ask_s = result["snapshot"]

        # Process agent fills
        pnl_realised = 0.0
        turnover = 0.0
        for fill in result["agent_fills"]:
            if fill.side == 0:  # buy MO hit our ask → we sold
                self.cash += fill.price * fill.size
                self.inventory -= fill.size
                self._total_filled_ask_value += fill.price * fill.size
            else:               # sell MO hit our bid → we bought
                self.cash -= fill.price * fill.size
                self.inventory += fill.size
                self._total_filled_bid_value += fill.price * fill.size
            pnl_realised += fill.size * abs(fill.price - mid)  # spread captured
            turnover += fill.size

        # Mark-to-market PnL change
        mtm_pnl = self.inventory * (new_mid - self.prev_mid)
        self.prev_mid = new_mid

        # Reward
        total_pnl = pnl_realised + mtm_pnl
        inventory_penalty = self.gamma_inv * (self.inventory ** 2)
        adverse_sel_penalty = self.lambda_adv * abs(imbalance * self.inventory)
        fee_cost = self.fee_rate * turnover
        reward = float(total_pnl - inventory_penalty - adverse_sel_penalty - fee_cost)

        # Update observation
        self._step_count += 1
        tau = 1.0 - self._step_count / self._max_steps
        sigma_hat = abs(new_mid - mid) / (mid + 1e-8)
        self.feature_builder.update(bid_s, ask_s, self.inventory, tau, sigma_hat, imbalance)
        obs = self.feature_builder.get_obs_flat()

        terminated = self._sampler.done or self._step_count >= self._max_steps
        info = {
            "mid": new_mid,
            "inventory": self.inventory,
            "pnl_realised": pnl_realised,
            "mtm_pnl": mtm_pnl,
            "turnover": turnover,
            "imbalance": imbalance,
        }
        return obs, reward, terminated, False, info

    def render(self):
        pass
