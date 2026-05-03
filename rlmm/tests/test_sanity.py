"""
Sanity tests for the RL market-maker project.
Run: pytest rlmm/tests/test_sanity.py -v
"""

import numpy as np
import pytest

from rlmm.sim.lob import LOB
from rlmm.sim.hawkes import MultivariateHawkes, N_TYPES
from rlmm.sim.flow import OrderFlowSampler, FlowParams
from rlmm.envs.mm_env import MarketMakingEnv
from rlmm.baselines.avellaneda_stoikov import AvellanedaStoikov
from rlmm.baselines.gueant_lehalle_tapia import GueantLehalleTapia
from rlmm.baselines.cartea_jaimungal import CarteaJaimungal


# ------------------------------------------------------------------
# LOB invariants
# ------------------------------------------------------------------

class TestLOB:
    def setup_method(self):
        self.lob = LOB(tick_size=0.01, n_levels=10)

    def test_no_crossed_book_after_limit_orders(self):
        self.lob.submit_limit(0, 99.95, 10)  # bid
        self.lob.submit_limit(1, 100.05, 10) # ask
        assert not self.lob.is_crossed()
        assert self.lob.best_bid() == pytest.approx(99.95)
        assert self.lob.best_ask() == pytest.approx(100.05)

    def test_market_order_fills_correctly(self):
        self.lob.submit_limit(1, 100.00, 5.0)  # ask at 100
        fills = self.lob.submit_market(0, 3.0)  # buy 3
        assert len(fills) == 1
        assert fills[0].size == pytest.approx(3.0)
        assert fills[0].price == pytest.approx(100.00)
        # Remaining ask should be 2
        bid_s, ask_s = self.lob.get_snapshot()
        assert ask_s[0] == pytest.approx(2.0)

    def test_fifo_queue_priority(self):
        # Two orders at same price — first should fill first
        id1 = self.lob.submit_limit(1, 100.00, 5.0)
        id2 = self.lob.submit_limit(1, 100.00, 5.0)
        # Queue position of id2 should be 5 (id1 is ahead)
        assert self.lob.queue_position(id2) == pytest.approx(5.0)
        assert self.lob.queue_position(id1) == pytest.approx(0.0)

    def test_cancel_removes_order(self):
        oid = self.lob.submit_limit(0, 99.90, 10.0)
        assert self.lob.queue_position(oid) == pytest.approx(0.0)
        self.lob.cancel(oid)
        assert self.lob.queue_position(oid) is None

    def test_snapshot_shape(self):
        self.lob.submit_limit(0, 99.95, 10)
        self.lob.submit_limit(1, 100.05, 10)
        bid_s, ask_s = self.lob.get_snapshot(10)
        assert bid_s.shape == (10,)
        assert ask_s.shape == (10,)
        assert bid_s[0] == pytest.approx(10.0)
        assert ask_s[0] == pytest.approx(10.0)

    def test_no_cross_after_mo_partial_fill(self):
        self.lob.submit_limit(0, 99.95, 10)
        self.lob.submit_limit(1, 100.05, 10)
        self.lob.submit_market(0, 5.0)
        assert not self.lob.is_crossed()

    def test_order_imbalance_in_bounds(self):
        self.lob.submit_limit(0, 99.95, 8)
        self.lob.submit_limit(1, 100.05, 4)
        imb = self.lob.order_imbalance()
        assert -1.0 <= imb <= 1.0
        assert imb > 0   # more bid volume


# ------------------------------------------------------------------
# Hawkes process
# ------------------------------------------------------------------

class TestHawkes:
    def test_positive_intensity(self):
        h = MultivariateHawkes.default_params()
        events = h.simulate(T_max=1.0, seed=0)
        for ev in events:
            lam = h.intensity_at(ev.time, [e for e in events if e.time < ev.time])
            assert np.all(lam >= 0)

    def test_stationarity_default(self):
        h = MultivariateHawkes.default_params()
        assert h.is_stationary(), f"branching_ratio={h.branching_ratio():.3f} should be < 1"

    def test_event_types_valid(self):
        h = MultivariateHawkes.default_params()
        events = h.simulate(T_max=5.0, seed=1)
        assert len(events) > 0
        for ev in events:
            assert ev.etype in range(N_TYPES)
            assert ev.time >= 0

    def test_all_event_types_present(self):
        h = MultivariateHawkes.default_params()
        events = h.simulate(T_max=20.0, seed=42)
        etypes = {ev.etype for ev in events}
        assert etypes == {0, 1, 2, 3}, f"Missing event types: {set(range(4)) - etypes}"

    def test_high_excitation_more_events(self):
        h_low = MultivariateHawkes.default_params()
        h_high = MultivariateHawkes.high_excitation_params()
        ev_low = h_low.simulate(T_max=30.0, seed=7)
        ev_high = h_high.simulate(T_max=30.0, seed=7)
        assert len(ev_high) > len(ev_low)


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

class TestEnv:
    def test_reset_obs_shape(self):
        env = MarketMakingEnv(n_levels=10, T_window=20)
        obs, info = env.reset(seed=0)
        assert obs.shape == (env.obs_dim,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_shapes(self):
        env = MarketMakingEnv(n_levels=10, T_window=20)
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert "inventory" in info
        assert "mid" in info

    def test_episode_terminates(self):
        env = MarketMakingEnv(T_episode_sec=10.0, dt=1.0)
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
            assert steps < 1000, "Episode did not terminate"
        assert steps <= 15   # T=10, dt=1 → ~10 steps

    def test_action_space(self):
        env = MarketMakingEnv()
        assert env.action_space.shape == (4,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------

class TestBaselines:
    def _run_episode(self, policy, env, n_steps: int = 50) -> float:
        obs, _ = env.reset(seed=0)
        total_pnl = 0.0
        for _ in range(n_steps):
            mid = env._sampler.mid
            inv = env.inventory
            tau = 1.0 - env._step_count / env._max_steps
            action = policy.action(mid, inv, tau, sigma=0.02, tick_size=env.tick_size)
            _, reward, terminated, truncated, _ = env.step(action)
            total_pnl += reward
            if terminated or truncated:
                break
        return total_pnl

    def test_as_runs(self):
        env = MarketMakingEnv(T_episode_sec=30.0)
        policy = AvellanedaStoikov()
        pnl = self._run_episode(policy, env)
        assert np.isfinite(pnl)

    def test_glt_runs(self):
        env = MarketMakingEnv(T_episode_sec=30.0)
        policy = GueantLehalleTapia()
        pnl = self._run_episode(policy, env)
        assert np.isfinite(pnl)

    def test_cj_runs(self):
        env = MarketMakingEnv(T_episode_sec=30.0)
        policy = CarteaJaimungal()
        pnl = self._run_episode(policy, env)
        assert np.isfinite(pnl)

    def test_as_quote_properties(self):
        pol = AvellanedaStoikov(gamma=0.1, k=1.5)
        bid, ask = pol.quote(mid=100.0, inventory=0.0, tau=1.0, sigma=0.02)
        assert bid < 100.0 < ask, "Spread should straddle mid"
        # Higher inventory → reservation price lower → bid lower
        bid_long, ask_long = pol.quote(100.0, inventory=10.0, tau=1.0, sigma=0.02)
        assert bid_long < bid

    def test_cj_spread_widens_with_inventory(self):
        pol = CarteaJaimungal()
        _, _ = pol.quote(100.0, 0.0, 1.0, 0.02)
        b0, a0 = pol.quote(100.0, 0.0, 1.0, 0.02)
        b10, a10 = pol.quote(100.0, 10.0, 1.0, 0.02)
        spread_0 = a0 - b0
        spread_10 = a10 - b10
        assert spread_10 >= spread_0   # CJ widens spread with adverse selection
