"""
curriculum.py — Tier-based curriculum scheduler for EduForge GRPO training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from src.environment.student_fsm import MisconceptionType


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class Tier(IntEnum):
    T1 = 1   # Fixed misconception, high attention
    T2 = 2   # Randomized misconception, high attention
    T3 = 3   # Randomized misconception, randomized attention
    T4 = 4   # Multiple hidden misconceptions, degraded attention


WIN_RATE_THRESHOLD = 0.6   # advance when win_rate >= this


@dataclass
class TierParams:
    tier:                   Tier
    fixed_misconception:    Optional[MisconceptionType]  # None = randomize
    attention_init:         float                         # starting attention
    attention_jitter:       float                         # ± uniform noise added to attention_init
    num_misconceptions:     int                           # 1 for T1-T3, 2 for T4
    hidden_misconceptions:  bool                          # T4: second misconception not revealed
    max_turns:              int
    description:            str


# ---------------------------------------------------------------------------
# Tier catalogue
# ---------------------------------------------------------------------------

TIER_CATALOGUE: dict[Tier, TierParams] = {
    Tier.T1: TierParams(
        tier=Tier.T1,
        fixed_misconception=MisconceptionType.PROCEDURAL,
        attention_init=8.0,
        attention_jitter=0.0,
        num_misconceptions=1,
        hidden_misconceptions=False,
        max_turns=12,
        description="Fixed procedural misconception, high stable attention.",
    ),
    Tier.T2: TierParams(
        tier=Tier.T2,
        fixed_misconception=None,      # randomized each episode
        attention_init=8.0,
        attention_jitter=0.5,
        num_misconceptions=1,
        hidden_misconceptions=False,
        max_turns=12,
        description="Random misconception, high attention with slight jitter.",
    ),
    Tier.T3: TierParams(
        tier=Tier.T3,
        fixed_misconception=None,
        attention_init=5.0,
        attention_jitter=2.0,          # attention in [3.0, 7.0]
        num_misconceptions=1,
        hidden_misconceptions=False,
        max_turns=12,
        description="Random misconception, randomized attention.",
    ),
    Tier.T4: TierParams(
        tier=Tier.T4,
        fixed_misconception=None,
        attention_init=4.0,
        attention_jitter=2.0,          # attention in [2.0, 6.0]
        num_misconceptions=2,
        hidden_misconceptions=True,    # second misconception not in observation
        max_turns=15,                  # slightly longer to compensate complexity
        description="Multiple hidden misconceptions, degraded attention.",
    ),
}


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Manages tier progression during GRPO training.

    Usage
    -----
    scheduler = CurriculumScheduler()
    params = scheduler.get_params(win_rate=0.0)   # returns T1 params

    # After training window:
    params = scheduler.get_params(win_rate=0.65)  # advances to T2, returns T2
    """

    def __init__(self, start_tier: Tier = Tier.T1) -> None:
        self._tier: Tier = start_tier
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_tier(self) -> Tier:
        return self._tier

    def get_params(self, win_rate: float) -> TierParams:
        """
        Evaluate win_rate against threshold.
        Advance tier if win_rate >= WIN_RATE_THRESHOLD and not already at T4.
        Returns TierParams for the (possibly updated) current tier.
        """
        advanced = False
        if win_rate >= WIN_RATE_THRESHOLD and self._tier < Tier.T4:
            prev = self._tier
            self._tier = Tier(self._tier + 1)
            advanced = True
            self._history.append({
                "event":    "advance",
                "from":     prev,
                "to":       self._tier,
                "win_rate": round(win_rate, 4),
            })
        else:
            self._history.append({
                "event":    "hold",
                "tier":     self._tier,
                "win_rate": round(win_rate, 4),
            })

        return TIER_CATALOGUE[self._tier]

    def force_tier(self, tier: Tier) -> TierParams:
        """Override tier — useful for eval / ablation."""
        self._tier = tier
        return TIER_CATALOGUE[tier]

    def is_final_tier(self) -> bool:
        return self._tier == Tier.T4

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        return {
            "current_tier": self._tier.name,
            "description":  TIER_CATALOGUE[self._tier].description,
            "advances":     sum(1 for e in self._history if e["event"] == "advance"),
        }
