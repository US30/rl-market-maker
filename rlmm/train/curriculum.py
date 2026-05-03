"""
Curriculum scheduler: progressively harder vol regimes during training.

Stage 1 (0%–40% of total steps):
    vol_regime='low'    — BM sigma=0.01, default Hawkes
Stage 2 (40%–75%):
    vol_regime='high'   — BM sigma=0.03, more frequent jumps
Stage 3 (75%–100%):
    vol_regime='hawkes_high' — high-excitation Hawkes, sigma=0.04

At each stage transition, new envs are created with updated params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CurriculumStage:
    vol_regime: str
    start_frac: float   # fraction of total_steps when this stage begins


STAGES = [
    CurriculumStage("low", 0.0),
    CurriculumStage("high", 0.4),
    CurriculumStage("hawkes_high", 0.75),
]


class CurriculumScheduler:
    """
    Tracks current training stage and tells caller when to rebuild envs.

    Usage:
        sched = CurriculumScheduler(total_steps)
        ...
        regime, changed = sched.step(global_step)
        if changed:
            envs = make_envs(vol_regime=regime)
    """

    def __init__(self, total_steps: int, stages: list[CurriculumStage] | None = None):
        self.total_steps = total_steps
        self.stages = stages or STAGES
        self._current_stage_idx = 0

    @property
    def current_regime(self) -> str:
        return self.stages[self._current_stage_idx].vol_regime

    def step(self, global_step: int) -> tuple[str, bool]:
        """
        Returns (regime, changed).
        changed=True triggers env rebuild by caller.
        """
        frac = global_step / max(self.total_steps, 1)
        new_idx = 0
        for i, stage in enumerate(self.stages):
            if frac >= stage.start_frac:
                new_idx = i
        changed = new_idx != self._current_stage_idx
        self._current_stage_idx = new_idx
        return self.current_regime, changed

    def stage_label(self) -> str:
        return f"stage{self._current_stage_idx+1}/{len(self.stages)}:{self.current_regime}"
