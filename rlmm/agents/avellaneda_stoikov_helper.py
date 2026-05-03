"""Helper to get A-S action from raw env state (used by as_imitator.py)."""

import numpy as np


def as_action_from_state(
    mid: float,
    inventory: float,
    tau: float,
    sigma: float,
    tick_size: float = 0.01,
    gamma: float = 0.1,
    k: float = 1.5,
    max_offset_ticks: int = 10,
) -> np.ndarray:
    """Returns normalised action [-1,1]^4 matching A-S quotes."""
    import math
    t_rem = max(tau, 1e-6)
    reservation = mid - inventory * gamma * sigma ** 2 * t_rem
    spread = gamma * sigma ** 2 * t_rem + (2.0 / gamma) * math.log(1.0 + gamma / k)
    spread = max(spread, 2e-4)
    bid = reservation - spread / 2.0
    ask = reservation + spread / 2.0

    bid_ticks = np.clip((mid - bid) / tick_size, 1, max_offset_ticks)
    ask_ticks = np.clip((ask - mid) / tick_size, 1, max_offset_ticks)
    bid_norm = float(np.clip((bid_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))
    ask_norm = float(np.clip((ask_ticks - 0.5) / 2.5 - 1.0, -1.0, 1.0))
    skew_norm = float(np.clip(inventory / 50.0, -1.0, 1.0))
    return np.array([bid_norm, ask_norm, skew_norm, 0.0], dtype=np.float32)
