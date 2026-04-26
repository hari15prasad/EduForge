"""
engine.py — Multi-component reward engine for EduForge.
REVISED: Aggressive penalty for information dumping to favor Query over Example.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from src.environment.student_fsm import MisconceptionType, TutorAction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_OUTCOME_RESOLUTION = 10.0
R_OUTCOME_FAILURE    = -3.0
CONFUSION_RESOLUTION = 2.0

SCAFFOLD_ACTIONS = {"question", "hint", "analogize"}
DIRECT_TELL_ACTIONS = {"worked_example", "correct_fact"}

# STRENGTHENED: Minimum steps before any resolution bonus is granted.
MIN_DIAGNOSTIC_DEPTH = 5

# STRENGTHENED: How long the tutor is FORBIDDEN from dumping information.
EARLY_SESSION_THRESHOLD = 5 

# ── REBALANCED DOMAIN PRIORITY ──────────────────────────────────────────────
DOMAIN_PRIORITY = {
    MisconceptionType.PROCEDURAL:  ["hint", "question", "worked_example"],
    MisconceptionType.FACTUAL:     ["question", "correct_fact", "explain"],
    MisconceptionType.TRANSFER:    ["analogize", "question", "worked_example"],
    MisconceptionType.CONCEPTUAL:  ["question", "analogize", "hint"],
}

# STRENGTHENED: Significant gap between first-priority (Query) and others.
ALIGNMENT_REWARDS = [4.0, 1.5, 0.5] 


@dataclass
class RewardComponents:
    r_outcome:      float = 0.0
    r_process:      float = 0.0
    r_alignment:    float = 0.0
    r_scaffolding:  float = 0.0
    r_recovery:     float = 0.0
    p_penalty:      float = 0.0
    total:          float = 0.0
    breakdown:      dict  = field(default_factory=dict)


class RewardEngine:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._action_streak   = {}
        self._prev_confusion  = None
        self._prev_attention  = None
        self._prev_action     = None
        self._scaffold_streak = 0

    def compute(
        self,
        *,
        confusion_before:  float,
        confusion_after:   float,
        attention_after:   float,
        action_text:       str,
        format_valid:      bool,
        done:              bool,
        done_reason:       Optional[str],
        attention_before:  float                       = 5.0,
        action:            Optional[TutorAction]       = None,
        action_history:    Optional[List[str]]         = None,
        misconception:     Optional[MisconceptionType] = None,
        episode_length:    int                         = 1,
    ) -> tuple[float, RewardComponents]:

        breakdown: dict[str, float] = {}
        action_val = action.value if action else None
        confusion_delta = confusion_before - confusion_after

        # ------------------------------------------------------------------
        # 1. CONFUSION PROGRESS
        # ------------------------------------------------------------------
        if action_val in SCAFFOLD_ACTIONS:
            w_method = 1.5  # Increased from 1.4
        elif action_val in DIRECT_TELL_ACTIONS:
            w_method = 0.4  # Decreased from 0.6
        else:
            w_method = 1.0

        r_process = 2.0 * w_method * confusion_delta
        
        if action_val in SCAFFOLD_ACTIONS:
            self._scaffold_streak += 1
        elif action_val in DIRECT_TELL_ACTIONS:
            self._scaffold_streak = 0

        # ------------------------------------------------------------------
        # 2. AGGRESSIVE INFORMATION DUMPING PENALTY
        # ------------------------------------------------------------------
        p_info_dump = 0.0
        if action_val in DIRECT_TELL_ACTIONS:
            early_steps_remaining = EARLY_SESSION_THRESHOLD - episode_length + 1
            if early_steps_remaining > 0:
                # Increased multiplier to -10.0 for step 1
                p_info_dump = 2.5 * early_steps_remaining 

        # ------------------------------------------------------------------
        # 3. PEDAGOGICAL ALIGNMENT (High weight on Query/Question)
        # ------------------------------------------------------------------
        bonus_alignment = 0.0
        if action and misconception:
            priority_list = DOMAIN_PRIORITY.get(misconception, [])
            if action_val in priority_list:
                rank = priority_list.index(action_val)
                bonus_alignment = ALIGNMENT_REWARDS[min(rank, len(ALIGNMENT_REWARDS) - 1)]

        # ------------------------------------------------------------------
        # 4. SCAFFOLDING SEQUENCE & STATE IMPROVEMENT
        # ------------------------------------------------------------------
        bonus_scaffolding = 0.0
        r_state_improvement = 0.0
        if action_val in SCAFFOLD_ACTIONS and confusion_delta > 0:
            r_state_improvement = min(2.0 * confusion_delta, 4.0)
        
        bonus_scaffolding += r_state_improvement

        # ------------------------------------------------------------------
        # 5. OUTCOME: MIN_DIAGNOSTIC_DEPTH GATE
        # ------------------------------------------------------------------
        r_outcome = 0.0
        r_scaffolded_resolution = 0.0

        if done and done_reason == "success":
            # If the session ends TOO FAST, penalize even if "successful"
            if episode_length < MIN_DIAGNOSTIC_DEPTH:
                r_outcome = -5.0 
            else:
                r_outcome = R_OUTCOME_RESOLUTION
                if self._scaffold_streak >= 3:
                    r_scaffolded_resolution = 10.0 # High reward for deep scaffolding

        # ------------------------------------------------------------------
        # 6. FINAL AGGREGATION
        # ------------------------------------------------------------------
        total = float(np.clip(
            r_process + bonus_alignment + bonus_scaffolding + r_outcome 
            - p_info_dump, -15.0, 15.0
        ))

        components = RewardComponents(
            r_outcome     = r_outcome + r_scaffolded_resolution,
            r_alignment   = bonus_alignment,
            r_scaffolding = bonus_scaffolding,
            p_penalty     = p_info_dump,
            total         = total,
        )

        self._prev_confusion = confusion_before
        self._prev_action    = action
        return total, components