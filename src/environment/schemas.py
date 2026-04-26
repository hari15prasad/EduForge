"""
Pydantic schemas for EduForge-Env data contracts.

All data flowing between the environment, reward engine, and trainer
is typed through these models so serialisation / validation is free.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import time


# ── Enumerations ──────────────────────────────────────────────────────────────

class SubjectDomain(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    CODING = "coding"
    READING = "reading"


class DifficultyLevel(int, Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3


class CurriculumStrategy(str, Enum):
    ADAPTIVE = "adaptive"
    FIXED = "fixed"
    RANDOM = "random"


# ── Core schemas ──────────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Persistent record of a student agent's performance history."""

    student_id: str
    domain_scores: Dict[SubjectDomain, float] = Field(
        default_factory=lambda: {d: 0.5 for d in SubjectDomain}
    )
    total_episodes: int = 0
    total_correct: int = 0
    average_attempts: float = 0.0
    current_difficulty: DifficultyLevel = DifficultyLevel.EASY

    @property
    def accuracy(self) -> float:
        if self.total_episodes == 0:
            return 0.0
        return self.total_correct / self.total_episodes


class Problem(BaseModel):
    """A single educational problem presented to the student agent."""

    problem_id: str
    domain: SubjectDomain
    difficulty: DifficultyLevel
    question: str
    answer: str
    hints: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """Action produced by the student agent at each environment step."""

    raw_text: str
    action_type: str = "answer"          # "answer" | "hint_request" | "give_up"
    parsed_answer: Optional[str] = None
    thinking: Optional[str] = None       # chain-of-thought scratchpad
    token_count: int = 0
    latency_ms: float = 0.0


class StepResult(BaseModel):
    """Result returned by OpenEnvWrapper.step()."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    fsm_state: str = "UNKNOWN"
    step_index: int = 0
    timestamp: float = Field(default_factory=time.time)


class EpisodeResult(BaseModel):
    """Aggregated result for a complete episode."""

    episode_id: str
    student_id: str
    problem: Problem
    steps: List[StepResult] = Field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    num_attempts: int = 0
    num_hints: int = 0
    duration_seconds: float = 0.0
    final_fsm_state: str = "UNKNOWN"

    @property
    def mean_reward_per_step(self) -> float:
        if not self.steps:
            return 0.0
        return self.total_reward / len(self.steps)


class EnvironmentConfig(BaseModel):
    """Runtime configuration passed to OpenEnvWrapper."""

    max_steps: int = 20
    timeout_seconds: int = 30
    subject_domains: List[SubjectDomain] = Field(
        default_factory=lambda: list(SubjectDomain)
    )
    curriculum_strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE
    mastery_threshold: float = 0.75
    difficulty_levels: int = 3
    render_mode: Optional[str] = None

    @field_validator("mastery_threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("mastery_threshold must be in (0, 1]")
        return v


class RolloutBatch(BaseModel):
    """A batch of episode results collected during a rollout phase."""

    batch_id: str
    episodes: List[EpisodeResult]
    mean_reward: float = 0.0
    success_rate: float = 0.0
    timestamp: float = Field(default_factory=time.time)

    def compute_stats(self) -> "RolloutBatch":
        if not self.episodes:
            return self
        rewards = [e.total_reward for e in self.episodes]
        self.mean_reward = sum(rewards) / len(rewards)
        self.success_rate = sum(1 for e in self.episodes if e.success) / len(self.episodes)
        return self
