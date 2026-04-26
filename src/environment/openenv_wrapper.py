"""
openenv_wrapper.py — OpenEnv-compatible environment wrapping StudentSimulator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import openenv  # type: ignore
    if not hasattr(openenv, 'Environment'):
        raise ImportError
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False
    class _EnvBase:
        def reset(self): raise NotImplementedError
        def step(self, action): raise NotImplementedError
    openenv = type("openenv", (), {"Environment": _EnvBase})()

from .student_fsm import (
    MisconceptionType, StudentType, StudentSimulator,
    TutorAction, StudentState, encode_state,
)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    student_response: str
    confusion:        float
    attention:        float
    learning_trend:   float
    turn:             int
    misconception_id: MisconceptionType
    student_type:     StudentType
    last_action:      Optional[TutorAction]    = None
    steps_taken:      int                      = 0
    recent_actions:   tuple[str, ...]          = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_response": self.student_response,
            "confusion":        self.confusion,
            "attention":        self.attention,
            "learning_trend":   self.learning_trend,
            "turn":             self.turn,
            "misconception_id": self.misconception_id.value,
            "student_type":     self.student_type.value,
            "last_action":      self.last_action.value if self.last_action else None,
            "steps_taken":      self.steps_taken,
            "recent_actions":   list(self.recent_actions),
        }

    def to_numpy(self):
        from .student_fsm import StudentState, encode_state
        snap = StudentState(
            misconception_id = self.misconception_id,
            student_type     = self.student_type,
            confusion        = self.confusion,
            attention        = self.attention,
            learning_trend   = self.learning_trend,
            turn             = self.turn,
            last_action      = self.last_action,
        )
        return encode_state(snap)


# ---------------------------------------------------------------------------
# StepInfo
# ---------------------------------------------------------------------------

@dataclass
class StepInfo:
    done_reason:      Optional[str]
    raw_action:       str
    parsed_action:    Optional[TutorAction]
    misconception_id: MisconceptionType


# ---------------------------------------------------------------------------
# Strategy tag parser
# ---------------------------------------------------------------------------

_STRATEGY_PATTERN = re.compile(
    r"<STRATEGY>\s*([a-z_]+)\s*</STRATEGY>",
    re.IGNORECASE,
)

def _parse_action(action_str: str) -> Optional[TutorAction]:
    match = _STRATEGY_PATTERN.search(action_str)
    if not match:
        return None
    try:
        return TutorAction(match.group(1).lower())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Teaching phase shaping
# ---------------------------------------------------------------------------

_PHASE_BONUS = 0.2   # small — guide without dominating

_PHASE_PREFERRED: dict[int, set[TutorAction]] = {
    0: {TutorAction.QUESTION, TutorAction.EXPLAIN},          # diagnose  (confusion > 6)
    1: {TutorAction.WORKED_EXAMPLE, TutorAction.ANALOGIZE},  # intervene (confusion > 3.5)
    2: {TutorAction.QUESTION, TutorAction.CORRECT_FACT},     # consolidate (confusion <= 3.5)
}

def _phase_index(confusion: float) -> int:
    if confusion > 6.0:  return 0
    if confusion > 3.5:  return 1
    return 2


# ---------------------------------------------------------------------------
# Episode constants
# ---------------------------------------------------------------------------

MAX_TURNS          = 15
CONFUSION_SUCCESS  = 2.0
ATTENTION_FLOOR    = 0.5   # Lowered — soft penalty only above, terminal only at near-zero


# ---------------------------------------------------------------------------
# EduForgeEnv
# ---------------------------------------------------------------------------

class EduForgeEnv(openenv.Environment):
    """
    OpenEnv-compatible tutoring environment.

    Done conditions
    ---------------
    success     : confusion <= 2.0
    timeout     : turn_count >= 15
    disengaged  : attention <= 0.5  (near-zero only — no premature kills)
    """

    def __init__(
        self,
        seed:               Optional[int]   = None,
        confusion_init:     Optional[float] = None,
        attention_init:     Optional[float] = None,
        misconception_init: Optional[str]   = None,
    ) -> None:
        self._seed               = seed
        self._confusion_init     = confusion_init
        self._attention_init     = attention_init
        self._misconception_init = misconception_init
        self._sim:               Optional[StudentSimulator] = None
        from ..rewards.engine import RewardEngine
        self._reward_engine      = RewardEngine()
        self._turn_count:        int                        = 0
        self._recent_actions:    list[TutorAction]          = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        m_init: Optional[MisconceptionType] = None
        if self._misconception_init is not None:
            try:
                m_init = MisconceptionType(self._misconception_init)
            except ValueError:
                pass

        self._sim = StudentSimulator(
            seed               = self._seed,
            confusion_init     = self._confusion_init,
            attention_init     = self._attention_init,
            misconception_init = m_init,
        )
        self._turn_count    = 0
        self._recent_actions = []
        self._reward_engine.reset()

        snap             = self._sim.state_snapshot()
        initial_response = self._sim.generate_response()

        return Observation(
            student_response = initial_response,
            confusion        = snap.confusion,
            attention        = snap.attention,
            learning_trend   = snap.learning_trend,
            turn             = self._turn_count,
            misconception_id = snap.misconception_id,
            student_type     = snap.student_type,
            last_action      = None,
            steps_taken      = 0,
            recent_actions   = tuple(),
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action_str: str
    ) -> tuple[Observation, float, bool, StepInfo]:
        if self._sim is None:
            raise RuntimeError("Call reset() before step().")

        # 1. Parse
        parsed_action = _parse_action(action_str)
        format_valid  = parsed_action is not None
        if parsed_action is None:
            parsed_action = TutorAction.REPEAT

        # 2. Pre-transition snapshot
        confusion_before  = self._sim.confusion
        attention_before  = self._sim.attention
        misconception_now = self._sim.misconception_id

        # 3. Phase bonus
        phase      = _phase_index(confusion_before)
        phase_bonus = _PHASE_BONUS if parsed_action in _PHASE_PREFERRED[phase] else 0.0

        # 4. Transition
        self._sim.transition(parsed_action)
        self._turn_count += 1

        snap = self._sim.state_snapshot()

        # 5. Done conditions
        done        = False
        done_reason: Optional[str] = None

        if snap.confusion <= CONFUSION_SUCCESS:
            done        = True
            done_reason = "success"
        elif snap.attention <= ATTENTION_FLOOR:
            done        = True
            done_reason = "disengaged"
        elif self._turn_count >= MAX_TURNS:
            done        = True
            done_reason = "timeout"

        # 6. Reward
        reward, _components = self._reward_engine.compute(
            confusion_before = confusion_before,
            confusion_after  = snap.confusion,
            attention_before = attention_before,
            attention_after  = snap.attention,
            action_text      = action_str,
            format_valid     = format_valid,
            learning_trend   = snap.learning_trend,
            action           = parsed_action,
            action_history   = [a.value for a in self._recent_actions[-3:]],
            misconception    = misconception_now,
            done             = done,
            done_reason      = done_reason,
            phase_bonus      = phase_bonus,
            episode_length   = self._turn_count,
        )

        # 7. Update history
        self._recent_actions.append(parsed_action)
        if len(self._recent_actions) > 5:
            self._recent_actions.pop(0)

        obs = Observation(
            student_response = snap.last_response,
            confusion        = snap.confusion,
            attention        = snap.attention,
            learning_trend   = snap.learning_trend,
            turn             = self._turn_count,
            misconception_id = snap.misconception_id,
            student_type     = snap.student_type,
            last_action      = snap.last_action,
            steps_taken      = self._turn_count,
            recent_actions   = tuple(a.value for a in self._recent_actions[-3:]),
        )

        info = StepInfo(
            done_reason      = done_reason,
            raw_action       = action_str,
            parsed_action    = parsed_action,
            misconception_id = snap.misconception_id,
        )

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def __repr__(self) -> str:
        if self._sim is None:
            return "EduForgeEnv(not started)"
        snap = self._sim.state_snapshot()
        return (
            f"EduForgeEnv(turn={self._turn_count}, "
            f"confusion={snap.confusion:.2f}, "
            f"attention={snap.attention:.2f}, "
            f"misconception={snap.misconception_id.value})"
        )