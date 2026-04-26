"""
benchmark_monitor.py — Production-grade Watchdog for EduForge.

Tracks a rolling performance window and emits a KILL_AGENT signal when
the agent's win-rate or confusion de-escalation drops below threshold.
Supports self-healing restart via configurable restart callbacks.
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, List, Optional


# ---------------------------------------------------------------------------
# Signal Enum
# ---------------------------------------------------------------------------

class WatchdogSignal(str, Enum):
    OK           = "OK"
    WARN         = "WARN"           # approaching threshold
    KILL_AGENT   = "KILL_AGENT"     # threshold breached — restart required
    RESTARTING   = "RESTARTING"     # restart in progress
    HEALTHY      = "HEALTHY"        # post-restart clean bill of health


# ---------------------------------------------------------------------------
# Episode Record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode_id:           int
    outcome:              str           # "success" | "timeout" | "disengaged"
    confusion_start:      float
    confusion_end:        float
    steps:                int
    timestamp:            float = field(default_factory=time.time)

    @property
    def win(self) -> bool:
        return self.outcome == "success"

    @property
    def confusion_delta(self) -> float:
        """Positive means confusion was reduced (good)."""
        return self.confusion_start - self.confusion_end


# ---------------------------------------------------------------------------
# BenchmarkMonitor
# ---------------------------------------------------------------------------

class BenchmarkMonitor:
    """
    Watchdog monitoring agent performance against a production benchmark.

    Production Benchmark (defaults):
      - Target win rate           : ≥ 70%
      - Avg confusion de-escalation: > 4.0 per episode
      - Kill threshold (rolling)  : < 50% win rate over `window` episodes
    """

    def __init__(
        self,
        *,
        target_win_rate:         float = 0.70,
        min_confusion_delta:     float = 4.0,
        kill_threshold:          float = 0.50,
        window:                  int   = 5,
        warn_threshold:          float = 0.60,
        on_kill:  Optional[Callable[[], None]] = None,
        on_restart_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self.target_win_rate        = target_win_rate
        self.min_confusion_delta    = min_confusion_delta
        self.kill_threshold         = kill_threshold
        self.warn_threshold         = warn_threshold
        self.window                 = window

        # Callbacks
        self._on_kill               = on_kill
        self._on_restart_complete   = on_restart_complete

        # State
        self._history: Deque[EpisodeRecord] = deque(maxlen=window)
        self._all_episodes: List[EpisodeRecord] = []
        self._episode_counter       = 0
        self._signal                = WatchdogSignal.OK
        self._kill_count            = 0
        self._last_kill_time: Optional[float] = None
        self._lock                  = threading.Lock()
        self._killed                = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_episode(
        self,
        *,
        outcome:         str,
        confusion_start: float,
        confusion_end:   float,
        steps:           int,
    ) -> WatchdogSignal:
        """
        Register a completed episode and evaluate the watchdog signal.
        Returns the current WatchdogSignal.
        """
        with self._lock:
            self._episode_counter += 1
            rec = EpisodeRecord(
                episode_id      = self._episode_counter,
                outcome         = outcome,
                confusion_start = confusion_start,
                confusion_end   = confusion_end,
                steps           = steps,
            )
            self._history.append(rec)
            self._all_episodes.append(rec)
            self._signal = self._evaluate()

            if self._signal == WatchdogSignal.KILL_AGENT and not self._killed:
                self._killed = True
                self._kill_count += 1
                self._last_kill_time = time.time()
                if self._on_kill:
                    threading.Thread(target=self._on_kill, daemon=True).start()

            return self._signal

    def acknowledge_restart(self) -> None:
        """
        Call after the agent has been re-initialized.
        Clears the rolling window and resets kill state.
        """
        with self._lock:
            self._history.clear()
            self._killed  = False
            self._signal  = WatchdogSignal.HEALTHY
            if self._on_restart_complete:
                self._on_restart_complete()

    def force_kill(self) -> WatchdogSignal:
        """Manually trigger a KILL_AGENT signal (fail-safe override)."""
        with self._lock:
            self._signal = WatchdogSignal.KILL_AGENT
            self._killed = True
            self._kill_count += 1
            self._last_kill_time = time.time()
            if self._on_kill:
                threading.Thread(target=self._on_kill, daemon=True).start()
            return self._signal

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def signal(self) -> WatchdogSignal:
        return self._signal

    @property
    def rolling_win_rate(self) -> float:
        if not self._history:
            return 1.0   # benefit of the doubt at start
        return sum(r.win for r in self._history) / len(self._history)

    @property
    def rolling_avg_delta(self) -> float:
        if not self._history:
            return self.min_confusion_delta  # assume OK at start
        return sum(r.confusion_delta for r in self._history) / len(self._history)

    @property
    def total_episodes(self) -> int:
        return self._episode_counter

    @property
    def total_wins(self) -> int:
        return sum(1 for r in self._all_episodes if r.win)

    @property
    def global_win_rate(self) -> float:
        if not self._all_episodes:
            return 0.0
        return self.total_wins / len(self._all_episodes)

    @property
    def kill_count(self) -> int:
        return self._kill_count

    def status_dict(self) -> dict:
        """Serialisable snapshot for Gradio / logging."""
        return {
            "signal":               self._signal.value,
            "rolling_win_rate":     round(self.rolling_win_rate, 3),
            "rolling_avg_delta":    round(self.rolling_avg_delta, 3),
            "global_win_rate":      round(self.global_win_rate, 3),
            "total_episodes":       self.total_episodes,
            "total_wins":           self.total_wins,
            "kill_count":           self.kill_count,
            "window":               self.window,
            "kill_threshold":       self.kill_threshold,
            "target_win_rate":      self.target_win_rate,
            "min_confusion_delta":  self.min_confusion_delta,
            "last_kill_time":       self._last_kill_time,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate(self) -> WatchdogSignal:
        """Compute signal from current rolling window. Called under lock."""
        if len(self._history) < self.window:
            # Not enough data yet — check partial window leniently
            if len(self._history) == 0:
                return WatchdogSignal.OK
            win_rate = sum(r.win for r in self._history) / len(self._history)
            if win_rate < self.kill_threshold:
                return WatchdogSignal.KILL_AGENT
            return WatchdogSignal.OK

        win_rate   = self.rolling_win_rate
        avg_delta  = self.rolling_avg_delta

        # Hard kill: win-rate below kill threshold
        if win_rate < self.kill_threshold:
            return WatchdogSignal.KILL_AGENT

        # Confusion not de-escalating at all + mediocre win-rate
        if avg_delta < (self.min_confusion_delta * 0.4) and win_rate < self.warn_threshold:
            return WatchdogSignal.KILL_AGENT

        # Soft warn: approaching threshold
        if win_rate < self.warn_threshold:
            return WatchdogSignal.WARN

        if avg_delta < self.min_confusion_delta * 0.6:
            return WatchdogSignal.WARN

        return WatchdogSignal.OK
