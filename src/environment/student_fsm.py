"""
student_fsm.py — Simulated student with FSM-based state transitions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MisconceptionType(str, Enum):
    PROCEDURAL   = "procedural"
    CONCEPTUAL   = "conceptual"
    FACTUAL      = "factual"
    TRANSFER     = "transfer"


class StudentType(str, Enum):
    FAST       = "fast"
    AVERAGE    = "average"
    SLOW       = "slow"
    DISENGAGED = "disengaged"


class TutorAction(str, Enum):
    EXPLAIN        = "explain"
    WORKED_EXAMPLE = "worked_example"
    HINT           = "hint"
    QUESTION       = "question"
    CORRECT_FACT   = "correct_fact"
    ANALOGIZE      = "analogize"
    REPEAT         = "repeat"


# ---------------------------------------------------------------------------
# Index maps for neural encoding
# ---------------------------------------------------------------------------

MISCONCEPTION_IDX = {m.value: i for i, m in enumerate(MisconceptionType)}
STUDENT_TYPE_IDX  = {s.value: i for i, s in enumerate(StudentType)}
ACTION_IDX        = {a.value: i for i, a in enumerate(TutorAction)}
ACTION_IDX[None]  = -1
NUM_ACTIONS       = len(TutorAction)


# ---------------------------------------------------------------------------
# Strategy-misconception effectiveness map
# (worked_example added to TRANSFER based on empirical run data)
# ---------------------------------------------------------------------------

_EFFECTIVE_STRATEGIES: dict[MisconceptionType, set[TutorAction]] = {
    MisconceptionType.PROCEDURAL: {TutorAction.WORKED_EXAMPLE, TutorAction.HINT},
    MisconceptionType.CONCEPTUAL: {TutorAction.EXPLAIN, TutorAction.ANALOGIZE, TutorAction.QUESTION},
    MisconceptionType.FACTUAL:    {TutorAction.CORRECT_FACT, TutorAction.EXPLAIN},
    MisconceptionType.TRANSFER:   {
        TutorAction.ANALOGIZE,
        TutorAction.QUESTION,
        TutorAction.WORKED_EXAMPLE,
        TutorAction.EXPLAIN,
    },
}


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

_RESPONSE_TEMPLATES: dict[str, list[str]] = {
    "high": [
        "I don't really get it… can you explain again?",
        "I'm completely lost here.",
        "This doesn't make sense to me at all.",
    ],
    "medium": [
        "I think I sort of understand, but I'm not sure.",
        "Okay… maybe? Can you show me an example?",
        "I get parts of it, but something still feels off.",
    ],
    "low": [
        "That's starting to make sense.",
        "I think I see what you mean now.",
    ],
    "resolved": [
        "Oh! I get it now.",
        "That makes complete sense, thank you!",
    ],
}


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudentState:
    misconception_id: MisconceptionType
    student_type:     StudentType
    confusion:        float
    attention:        float
    learning_trend:   float
    turn:             int
    last_action:      Optional[TutorAction]       = None
    action_history:   List[TutorAction]           = field(default_factory=list)
    last_response:    str                         = ""


# ---------------------------------------------------------------------------
# Per-action confusion and attention deltas
# ---------------------------------------------------------------------------

_ACTION_CONFUSION: dict[str, tuple[float, float]] = {
    "explain":        (0.8, 1.8),   # tightened from (0.8,1.2) to reduce oscillation
    "worked_example": (1.0, 2.0),   # tightened from (1.0,1.5)
    "question":       (0.3, 0.7),
    "correct_fact":   (1.0, 1.8),
    "analogize":      (0.6, 1.4),
    "hint":           (0.4, 0.8),
    "repeat":         (0.0, 0.1),
}

_ACTION_ATTENTION: dict[str, tuple[float, float]] = {
    "explain":        (-0.4,  0.0),   # softened — was (-0.6,-0.1)
    "worked_example": (-0.5,  0.0),   # softened — was (-0.8,-0.2)
    "question":       (+0.3, +0.8),
    "correct_fact":   (-0.2, +0.2),
    "analogize":      (+0.2, +0.7),
    "hint":           (-0.1, +0.3),
    "repeat":         (-1.2, -0.5),
}


# ---------------------------------------------------------------------------
# StudentSimulator
# ---------------------------------------------------------------------------

class StudentSimulator:
    CONFUSION_MIN = 0.0
    CONFUSION_MAX = 10.0
    ATTENTION_MIN = 0.0
    ATTENTION_MAX = 10.0

    def __init__(
        self,
        seed:               Optional[int]             = None,
        confusion_init:     Optional[float]           = None,
        attention_init:     Optional[float]           = None,
        misconception_init: Optional[MisconceptionType] = None,
    ) -> None:
        rng = random.Random(seed)
        self._rng = rng

        self.misconception_id = (
            misconception_init if misconception_init is not None
            else rng.choice(list(MisconceptionType))
        )
        self.student_type = rng.choice(list(StudentType))

        self.confusion = (
            confusion_init if confusion_init is not None
            else rng.uniform(5.0, 9.0)
        )
        self.attention = (
            attention_init if attention_init is not None
            else rng.uniform(4.0, 8.0)   # raised floor from 3.0 to 4.0
        )

        self._turn:             int                  = 0
        self._prev_action:      Optional[TutorAction] = None
        self._action_history:   list[TutorAction]    = []
        self._action_counts:    dict[TutorAction, int] = {}
        self._last_response:    str                  = ""
        self._confusion_history: list[float]         = [self.confusion]

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, action: TutorAction) -> None:
        effective   = _EFFECTIVE_STRATEGIES[self.misconception_id]
        action_name = action.value

        # Track history
        self._action_history.append(action)
        if len(self._action_history) > 3:
            self._action_history.pop(0)

        # Track total uses for diminishing returns
        times_used = self._action_counts.get(action, 0)
        self._action_counts[action] = times_used + 1

        # ------------------------------------------------------------------
        # Stochastic events (reduced probability — was 0.15/0.25/0.35)
        # ------------------------------------------------------------------
        r = self._rng.random()
        force_confusion_up   = r < 0.10    # 10% chance (was 15%)
        force_attention_down = 0.10 <= r < 0.18  # 8% (was 10%)
        force_plateau        = 0.18 <= r < 0.25  # 7% (was 10%)

        # ------------------------------------------------------------------
        # Confusion update
        # ------------------------------------------------------------------
        lo, hi = _ACTION_CONFUSION.get(action_name, (0.0, 0.2))

        if force_plateau:
            confusion_change = self._rng.uniform(-0.1, 0.1)
        elif force_confusion_up:
            # Spike size reduced for transfer — transfer needs tolerance
            spike_max = 1.2 if self.misconception_id == MisconceptionType.TRANSFER else 1.8
            confusion_change = self._rng.uniform(0.3, spike_max)
        else:
            if action in effective:
                base_drop = self._rng.uniform(lo, hi)
                confusion_change = -base_drop
                # Transfer still harder but less punishing
                if self.misconception_id == MisconceptionType.TRANSFER:
                    confusion_change *= 0.70   # was 0.55
            else:
                confusion_change = -self._rng.uniform(0.0, 0.2)
                if self.misconception_id == MisconceptionType.TRANSFER:
                    confusion_change = self._rng.uniform(0.1, 0.8)   # was (0.3,1.2)
                if (action == TutorAction.CORRECT_FACT
                        and self.misconception_id != MisconceptionType.FACTUAL):
                    confusion_change = self._rng.uniform(-0.1, 0.4)

        # Student type modifier
        if self.student_type == StudentType.FAST:
            if confusion_change < 0:
                confusion_change *= 1.4
        elif self.student_type == StudentType.SLOW:
            if confusion_change < 0:
                confusion_change *= 0.6

        # Diminishing returns (20% reduction per prior use)
        if confusion_change < 0:
            confusion_change *= (0.8 ** times_used)

        # Noise (tightened)
        noise_scale = abs(confusion_change) * 0.08 if confusion_change != 0 else 0.1
        confusion_change += self._rng.uniform(-noise_scale, noise_scale)

        self.confusion = max(
            self.CONFUSION_MIN,
            min(self.CONFUSION_MAX, self.confusion + confusion_change),
        )
        self._confusion_history.append(self.confusion)
        if len(self._confusion_history) > 3:
            self._confusion_history.pop(0)

        # ------------------------------------------------------------------
        # Attention update
        # ------------------------------------------------------------------
        att_lo, att_hi = _ACTION_ATTENTION.get(action_name, (-0.2, 0.2))

        if force_attention_down:
            attention_change = self._rng.uniform(-1.8, -0.8)   # was (-2.5,-1.0)
        else:
            attention_change = self._rng.uniform(att_lo, att_hi)
            # Task 5: Tactical Attention Floor (Procedural recovery boost)
            if (self.misconception_id == MisconceptionType.PROCEDURAL 
                and self.attention < 4.0 
                and attention_change > 0 
                and action in [TutorAction.ANALOGIZE, TutorAction.QUESTION]):
                attention_change *= 3.0

        # Repetition penalty — softer at low attention
        if action == TutorAction.REPEAT or action == self._prev_action:
            if self.attention > 3.0:
                attention_change -= 0.8    # was -1.0
            else:
                attention_change -= 0.2    # was -0.3

        # Student type modifier
        if self.student_type == StudentType.DISENGAGED:
            attention_change -= 0.3        # was -0.5

        # Noise
        attention_change += self._rng.uniform(-0.3, 0.3)

        self.attention = max(
            self.ATTENTION_MIN,
            min(self.ATTENTION_MAX, self.attention + attention_change),
        )

        self._prev_action = action
        self._turn += 1
        self._last_response = self.generate_response()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def learning_trend(self) -> float:
        if len(self._confusion_history) < 2:
            return 0.0
        return self._confusion_history[-1] - self._confusion_history[-2]

    def generate_response(self) -> str:
        if self.confusion <= 2.0:
            pool = _RESPONSE_TEMPLATES["resolved"]
        elif self.confusion <= 4.0:
            pool = _RESPONSE_TEMPLATES["low"]
        elif self.confusion <= 6.0:
            pool = _RESPONSE_TEMPLATES["medium"]
        else:
            pool = _RESPONSE_TEMPLATES["high"]
        return self._rng.choice(pool)

    def state_snapshot(self) -> StudentState:
        return StudentState(
            misconception_id = self.misconception_id,
            student_type     = self.student_type,
            confusion        = round(self.confusion, 3),
            attention        = round(self.attention, 3),
            learning_trend   = round(self.learning_trend, 3),
            turn             = self._turn,
            last_action      = self._prev_action,
            action_history   = list(self._action_history),
            last_response    = self._last_response,
        )


# ---------------------------------------------------------------------------
# State encoding for DQN
# ---------------------------------------------------------------------------

def encode_state(state: StudentState) -> np.ndarray:
    """
    Dense numerical vector for PyTorch / neural nets.

    Layout (39 dims):
    [0]     confusion        (normalised /10)
    [1]     attention        (normalised /10)
    [2]     learning_trend   (normalised /10)
    [3:7]   misconception    one-hot (4)
    [7:11]  student_type     one-hot (4)
    [11:18] last_action      one-hot (7)
    [18:25] history t-1      one-hot (7)
    [25:32] history t-2      one-hot (7)
    [32:39] history t-3      one-hot (7)
    """
    vec: list[float] = []

    vec.append(state.confusion / 10.0)
    vec.append(state.attention / 10.0)
    vec.append(state.learning_trend / 10.0)

    misc_idx = MISCONCEPTION_IDX[state.misconception_id.value]
    vec.extend(1.0 if i == misc_idx else 0.0 for i in range(len(MisconceptionType)))

    type_idx = STUDENT_TYPE_IDX[state.student_type.value]
    vec.extend(1.0 if i == type_idx else 0.0 for i in range(len(StudentType)))

    def action_onehot(a_val: Optional[str]) -> list[float]:
        idx = ACTION_IDX.get(a_val, -1)
        return [1.0 if i == idx else 0.0 for i in range(NUM_ACTIONS)]

    last_val = state.last_action.value if state.last_action else None
    vec.extend(action_onehot(last_val))

    hist_vals = [a.value for a in state.action_history]
    padded = [None] * (3 - len(hist_vals)) + hist_vals
    for hv in padded:
        vec.extend(action_onehot(hv))

    return np.array(vec, dtype=np.float32)