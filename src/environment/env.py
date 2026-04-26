"""
env.py — EduForgeEnv: Gym-compatible adaptive tutoring environment.

Domain priority rules are enforced via action masking at every step so the
agent can never accidentally "drift" into the explain-heavy failure mode seen
in episodes 34-46.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from src.environment.student_fsm import MisconceptionType, TutorAction, StudentSimulator
from src.rewards.engine import RewardEngine


# ---------------------------------------------------------------------------
# Action index ↔ TutorAction mapping (order is the contract with agent.py)
# ---------------------------------------------------------------------------
ACTION_INDEX: Dict[int, TutorAction] = {
    0: TutorAction.EXPLAIN,
    1: TutorAction.WORKED_EXAMPLE,
    2: TutorAction.ANALOGIZE,
    3: TutorAction.CORRECT_FACT,
    4: TutorAction.QUESTION,
}
ACTION_TEXT: Dict[int, str] = {
    0: "explain",
    1: "worked_example",
    2: "analogize",
    3: "correct_fact",
    4: "question",
}

# ---------------------------------------------------------------------------
# Per-domain step limits (fix for "stalls near confusion 3-5, runs out of steps")
# ---------------------------------------------------------------------------
STEP_LIMITS: Dict[MisconceptionType, int] = {
    MisconceptionType.PROCEDURAL:  18,   # procedural needs more scaffolding
    MisconceptionType.FACTUAL:     10,
    MisconceptionType.TRANSFER:    10,
    MisconceptionType.CONCEPTUAL:  10,
}

# ---------------------------------------------------------------------------
# Domain priority tables — used for forced masking (not just reward shaping)
# PROCEDURAL  → worked_example > correct_fact > explain
# FACTUAL     → correct_fact   > explain      > analogize
# TRANSFER    → analogize      > worked_example > question
# CONCEPTUAL  → question       > analogize    > explain
# ---------------------------------------------------------------------------
DOMAIN_PRIORITY_ACTIONS: Dict[MisconceptionType, int] = {
    MisconceptionType.PROCEDURAL:  1,   # worked_example
    MisconceptionType.FACTUAL:     3,   # correct_fact
    MisconceptionType.TRANSFER:    2,   # analogize
    MisconceptionType.CONCEPTUAL:  4,   # question
}

# Attention threshold below which we restrict to engagement-recovery actions
ATTENTION_GUARD_THRESHOLD = 3.0
# Actions allowed when attention is critically low
ATTENTION_GUARD_ACTIONS = {1, 4}  # worked_example, question


class EduForgeEnv(gym.Env):
    """
    Gym environment for training RL tutors on a simulated student.

    Observation (4 floats, all in [0, 1]):
        confusion_norm  — current confusion / 10.0
        attention_norm  — current attention / 10.0
        step_norm       — current step / max_steps
        domain_norm     — misconception_id / 3.0   (0=CONCEPTUAL … 3=TRANSFER)

    Action space: Discrete(5)
        0=explain  1=worked_example  2=analogize  3=correct_fact  4=question

    Termination conditions:
        SUCCESS  — confusion < 2.0
        FAIL     — attention < 0.5  (disengaged student)
        TIMEOUT  — step count exceeds domain step limit
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        misconception_init: Optional[MisconceptionType | str] = None,
        confusion_init: Optional[float] = None,
        attention_init: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Handle string input for misconception_init (common in pipeline scripts)
        if isinstance(misconception_init, str):
            misconception_init = MisconceptionType(misconception_init)

        self._seed = seed
        self._init_misconception = misconception_init
        self._init_confusion = confusion_init
        self._init_attention = attention_init

        # 4-feature continuous observation
        self.observation_space = spaces.Box(
            low=np.zeros(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        # 5 discrete tutor actions
        self.action_space = spaces.Discrete(5)

        self.reward_engine = RewardEngine()

        # Episode state
        self.misconception: Optional[MisconceptionType] = None
        self.fsm: Optional[StudentSimulator] = None
        self.confusion: float = 0.0
        self.attention: float = 0.0
        self.step_count: int = 0
        self.max_steps: int = 10
        self.action_history: list = []       # list[str] for RewardEngine
        self.consecutive_explain: int = 0    # tracks consecutive explain uses
        self.is_overtime: bool = False      # tracks if in overtime
        self.done: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Initialise or reset the episode state.
        Uses values from __init__ if provided, else randomises.
        """
        if seed is not None:
            self._seed = seed
        
        if self._seed is not None:
            np.random.seed(self._seed)

        self.misconception = self._init_misconception or np.random.choice(list(MisconceptionType))
        self.confusion  = self._init_confusion if self._init_confusion is not None else np.random.uniform(4.0, 9.0)
        self.attention  = self._init_attention if self._init_attention is not None else 8.0
        
        self.step_count = 0
        self.max_steps  = STEP_LIMITS[self.misconception]
        self.action_history = []
        self.consecutive_explain = 0
        self.is_overtime = False
        self.done = False
        self.reward_engine.reset()

        # StudentSimulator manages how actions alter confusion/attention
        self.fsm = StudentSimulator(
            misconception_init=self.misconception,
            confusion_init=self.confusion,
            attention_init=self.attention,
        )

        return self._get_obs()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one tutoring action and return (obs, reward, done, info).

        Action masking is enforced BEFORE the FSM transition so that illegal
        actions are remapped to the best legal substitute rather than silently
        ignored (prevents reward hacking via illegal action indices).
        """
        assert not self.done, "Environment must be reset before calling step() again."

        action = self._apply_action_mask(action)

        confusion_before = self.confusion
        attention_before = self.attention

        tutor_action = ACTION_INDEX[action]
        action_text  = ACTION_TEXT[action]

        # Track consecutive explain calls (failure mode fix #3)
        if action == 0:  # explain
            self.consecutive_explain += 1
        else:
            self.consecutive_explain = 0

        self.action_history.append(action_text)
        self.step_count += 1

        # ---- FSM transition: student reacts to the tutor action ----
        self.fsm.transition(tutor_action)
        self.confusion = self.fsm.confusion
        self.attention = self.fsm.attention

        # ---- Termination logic ----
        done_reason: Optional[str] = None
        done = False

        if self.confusion < 2.0:
            done = True
            done_reason = "success"
        elif self.attention < 0.5:
            done = True
            done_reason = "disengaged"
        elif self.step_count >= self.max_steps:
            # Task 3: Overtime Elasticity
            if not self.is_overtime and self.confusion < 2.5:
                self.is_overtime = True
                self.max_steps += 3
                # Continue episode
            else:
                done = True
                done_reason = "timeout"

        self.done = done

        # ---- Format validity: action was legal (not masked away) ----
        format_valid = self._is_format_valid(action, confusion_before)

        # ---- Reward from engine ----
        reward_val, _components = self.reward_engine.compute(
            confusion_before=confusion_before,
            confusion_after=self.confusion,
            attention_before=attention_before,
            attention_after=self.attention,
            action_text=action_text,
            action=tutor_action,
            format_valid=format_valid,
            done=done,
            done_reason=done_reason,
            action_history=self.action_history,
            misconception=self.misconception,
            episode_length=self.step_count,
        )

        # Task 3: Overtime Penalty
        if self.is_overtime and self.step_count > (self.max_steps - 3):
            reward_val -= 0.5  # Small penalty for every Overtime step

        obs = self._get_obs()
        info = {
            "confusion":    self.confusion,
            "attention":    self.attention,
            "step":         self.step_count,
            "misconception": self.misconception.name,
            "done_reason":  done_reason,
            "action_taken": action_text,
        }
        return obs, reward_val, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the normalised 4-float observation vector."""
        confusion_norm  = np.clip(self.confusion / 10.0,  0.0, 1.0)
        attention_norm  = np.clip(self.attention / 10.0,  0.0, 1.0)
        step_norm       = self.step_count / self.max_steps
        # Map MisconceptionType to 0-3 and normalise
        domain_id       = list(MisconceptionType).index(self.misconception)
        domain_norm     = domain_id / 3.0
        return np.array(
            [confusion_norm, attention_norm, step_norm, domain_norm],
            dtype=np.float32,
        )

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a float mask: 0.0 = allowed, -inf = blocked.
        Based on the current environment state.
        """
        mask = np.zeros(self.action_space.n, dtype=np.float32)

        # Rule 1 — force worked_example for procedural early steps
        if (
            self.misconception == MisconceptionType.PROCEDURAL
            and self.step_count < 3
        ):
            # Block everything EXCEPT worked_example (idx 1)
            for i in range(self.action_space.n):
                if i != 1:
                    mask[i] = -np.inf
            return mask

        # Rule 2 — attention guard: only engagement-recovery actions allowed
        if self.attention < ATTENTION_GUARD_THRESHOLD:
            for i in range(self.action_space.n):
                if i not in ATTENTION_GUARD_ACTIONS:
                    mask[i] = -np.inf

        # Rule 3 — consecutive explain masking
        if self.consecutive_explain >= 3:
            mask[0] = -np.inf  # block explain

        # Fallback: if all masked, allow explain
        if np.all(mask == -np.inf):
            mask[0] = 0.0

        return mask

    def _apply_action_mask(self, action: int) -> int:
        """
        Remap an illegal action to a legal one if necessary.
        Used for robustness if an agent ignores the mask.
        """
        mask = self.get_action_mask()
        if mask[action] == 0.0:
            return action

        # Remap logic
        if self.misconception == MisconceptionType.PROCEDURAL and self.step_count < 3:
            return 1  # worked_example

        if self.attention < ATTENTION_GUARD_THRESHOLD:
            return 4  # question

        if action == 0 and self.consecutive_explain >= 3:
            return DOMAIN_PRIORITY_ACTIONS[self.misconception]

        # Final fallback: just pick first allowed action
        allowed = np.where(mask == 0.0)[0]
        return int(allowed[0]) if len(allowed) > 0 else 0

    def _is_format_valid(self, action: int, confusion_before: float) -> bool:
        """
        Heuristic for whether the action was contextually appropriate.
        Used by RewardEngine to scale format bonuses/penalties.
        """
        # An explain when confusion is already low is a weak move
        if action == 0 and confusion_before < 3.0:
            return False
        # Correct_fact on a non-factual domain is a mismatch
        if action == 3 and self.misconception not in (
            MisconceptionType.FACTUAL, MisconceptionType.PROCEDURAL
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # render / close (minimal stubs for Gym compatibility)
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> None:
        print(
            f"[EduForgeEnv] step={self.step_count:2d} | "
            f"domain={self.misconception.name:<12} | "
            f"confusion={self.confusion:.2f} | "
            f"attention={self.attention:.2f}"
        )

    def close(self) -> None:
        pass