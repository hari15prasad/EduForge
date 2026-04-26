"""
qlearning_pipeline.py — Q-learning training pipeline for EduForge.

Modular pipeline:
  1. Dataset Loader   — load & validate training_samples.json
  2. Q-table Bootstrap — seed Q-values from offline dataset
  3. Training Loop     — adaptive epsilon-greedy online Q-learning
  4. Evaluation        — greedy policy rollouts with reporting
  5. Interactive REPL  — human-in-the-loop tutoring

Entry point: python scripts/qlearning_pipeline.py
"""

from __future__ import annotations

import json
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environment.openenv_wrapper import EduForgeEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Action catalogue
# ---------------------------------------------------------------------------
ACTIONS: dict[int, str] = {
    0: "explain",
    1: "worked_example",
    2: "question",
    3: "correct_fact",
    4: "analogize",
}
ACTION_TO_IDX: dict[str, int] = {v: k for k, v in ACTIONS.items()}
N_ACTIONS = len(ACTIONS)

MISCONCEPTION_MAP: dict[str, int] = {
    "none": 0, "procedural": 1, "conceptual": 2, "factual": 3, "transfer": 4,
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(_ROOT, "src", "environment", "training_samples.json")
MODEL_DIR    = os.path.join(_ROOT, "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "q_table.pkl")

REQUIRED_FIELDS = {
    "misconception", "confusion", "attention",
    "action", "next_confusion", "next_attention", "reward", "done",
}

# Hyperparameters
ALPHA_BOOTSTRAP  = 0.2
BOOTSTRAP_EPOCHS = 3
ALPHA            = 0.15
GAMMA            = 0.92
EPSILON_START    = 1.0
EPSILON_MIN      = 0.01
N_EPISODES       = 4000
MAX_STEPS        = 15
EVAL_EPISODES    = 80         # 4 misconceptions × 20 seeds each
SEED             = 42

# Thresholds (must match openenv_wrapper.py)
DONE_CONFUSION_THRESHOLD   = 2.0
ATTENTION_FAILURE_THRESHOLD = 0.5   # Match the environment's floor (ATTENTION_FLOOR)

# Q-value clipping — prevent explosion
Q_VALUE_CLIP = 15.0


# ---------------------------------------------------------------------------
# 1. State discretisation — integer buckets, compact space
# ---------------------------------------------------------------------------

# Coarse bin edges for discretization
_CONF_BINS = [0, 2, 4, 6, 8, 10.01]    # 5 bins
_ATT_BINS  = [0, 2, 4, 6, 8, 10.01]    # 5 bins

def _bin_value(val: float, edges: list[float]) -> int:
    """Return bin index for a value given sorted bin edges."""
    val = max(edges[0], min(edges[-1] - 0.01, val))
    for i in range(len(edges) - 1):
        if val < edges[i + 1]:
            return i
    return len(edges) - 2


def get_state(
    confusion: float,
    attention: float,
    misconception: str | int,
    step_number: int = 1,
    last_action: int | None = None,
    prev_last_action: int | None = None,
    progress_signal: int = 0,
    steps_since_improvement: int = 0
) -> tuple:
    """
    Map student metrics to a coarse discrete state tuple.
    """
    c = _bin_value(confusion, _CONF_BINS)
    a = _bin_value(attention, _ATT_BINS)

    if isinstance(misconception, str):
        m = MISCONCEPTION_MAP.get(misconception, 0)
    else:
        m = int(misconception)
        
    if step_number <= 5:
        p = 0
    elif step_number <= 10:
        p = 1
    else:
        p = 2

    la = 5 if last_action is None else int(last_action)
    
    ps = progress_signal + 1
    
    if steps_since_improvement <= 1:
        ssi = 0
    elif steps_since_improvement <= 3:
        ssi = 1
    else:
        ssi = 2

    pla = 5 if prev_last_action is None else int(prev_last_action)

    return (c, a, m, p, la, pla, ps, ssi)


def get_state_from_obs(
    obs, 
    last_action_idx: int | None = None,
    prev_last_action_idx: int | None = None,
    progress_signal: int = 0,
    steps_since_improvement: int = 0
) -> tuple:
    """Extract and discretise the state from an Observation object."""
    return get_state(
        obs.confusion,
        obs.attention,
        obs.misconception_id.value,
        obs.turn if hasattr(obs, 'turn') else 1,
        last_action_idx,
        prev_last_action_idx,
        progress_signal,
        steps_since_improvement
    )


# ---------------------------------------------------------------------------
# 2. Reward function — Continuous Multi-Component
# ---------------------------------------------------------------------------

def compute_reward(
    prev_conf: float,
    new_conf: float,
    prev_att: float,
    new_att: float,
    done: bool,
    success: bool,
    action_idx: int,
    misc_str: str,
    action_history: list[int],
    step: int,
    confusion_history: list[float],
    prev_reward: float = 0.0
) -> float:
    """
    Revised reward function with mode-dependent scaling, exponential attention penalties,
    and variance control.
    """
    reward = 0.0
    conf_delta = prev_conf - new_conf  # Positive delta is good

    # 4. Mode-Dependent Reward System
    if misc_str == "conceptual":
        if action_idx in [ACTION_TO_IDX["explain"], ACTION_TO_IDX["analogize"]]:
            reward += 1.5 * max(0, conf_delta)
        elif action_idx == ACTION_TO_IDX["question"]:
            reward -= 0.5  # Mild penalty for over-questioning
            
    elif misc_str == "factual":
        if action_idx == ACTION_TO_IDX["correct_fact"]:
            reward += 2.0 * max(0, conf_delta)
        elif action_idx == ACTION_TO_IDX["explain"]:
            reward += 1.0 * max(0, conf_delta)
        elif action_idx == ACTION_TO_IDX["question"]:
            reward += 0.2 * max(0, conf_delta)
            
    elif misc_str == "procedural":
        if action_idx == ACTION_TO_IDX["worked_example"]:
            reward += 1.5 * max(0, conf_delta)
        if len(action_history) > 0 and action_idx != action_history[-1]:
            reward -= 0.5  # Stability > exploration
            
    elif misc_str == "transfer":
        if len(action_history) > 0 and action_idx != action_history[-1]:
            reward -= 1.0  # Penalize rapid strategy switching
        if conf_delta > 0:
            reward += 1.2 * conf_delta

    # 1. Attention Safety Continuous Penalty
    if new_att < 4.0:
        reward *= 0.5  # Negative scaling
        reward -= 1.0
    if new_att < 2.0:
        reward -= (2.0 - new_att) ** 2  # Exponential penalty

    # 2. Question Action Control (Negative consequences)
    if action_idx == ACTION_TO_IDX["question"]:
        if new_conf > prev_conf or new_att < prev_att:
            reward -= 2.0  # Immediate negative reward

    # 5. Confusion Reduction Rule (Monotonicity Bias)
    if len(confusion_history) >= 3:
        if confusion_history[-2] < confusion_history[-3] and confusion_history[-1] < confusion_history[-2]:
            if new_conf > prev_conf:  # Broke a reduction streak
                reward -= 2.5
                
    # 7. Failure Prevention Objective
    if step > 10 and len(confusion_history) >= 4:
        recent_conf_drop = confusion_history[-4] - new_conf
        if recent_conf_drop <= 0.5:
            reward -= 1.5 * (step - 10)  # Scaling penalty for stagnation

    # Terminal Rewards
    if done:
        if new_att <= 0.5:
            reward -= 10.0
        elif success:
            reward += 5.0
        else:
            reward -= 2.0

    # 6. Reward Variance Control
    jump = abs(reward - prev_reward)
    if jump > 5.0:
        reward -= 0.5 * (jump - 5.0)  # Smoothing
        
    norm_factor = {"conceptual": 1.0, "factual": 0.8, "procedural": 1.2, "transfer": 1.5}.get(misc_str, 1.0)
    reward /= norm_factor
    
    return float(np.clip(reward, -10.0, 10.0))


# ---------------------------------------------------------------------------
# 3. Q-Table Architecture & Update
# ---------------------------------------------------------------------------

def create_q_system() -> dict[str, defaultdict]:
    """Create a structured dictionary of Q-tables."""
    return {
        "shared": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
        "conceptual": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
        "factual": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
        "procedural": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
        "transfer": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
        "none": defaultdict(lambda: np.zeros(N_ACTIONS, dtype=np.float32)),
    }

def get_q_values(q_system: dict[str, defaultdict], state: tuple, misc_str: str) -> np.ndarray:
    """Q_final(s, a) = Q_shared(s, a) + Q_type(s, a)"""
    shared_q = q_system["shared"][state]
    type_q = q_system[misc_str][state]
    return shared_q + type_q

def update_q(
    q_system: dict[str, defaultdict],
    state: tuple,
    misc_str: str,
    action_idx: int,
    reward: float,
    next_state: tuple,
    done: bool,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
) -> None:
    """Standard Bellman using the combined Q-value."""
    current_q_vals = get_q_values(q_system, state, misc_str)
    
    if done:
        best_next = 0.0
    else:
        next_q_vals = get_q_values(q_system, next_state, misc_str)
        best_next = float(np.max(next_q_vals))
        
    td_target = reward + gamma * best_next
    td_error = td_target - current_q_vals[action_idx]
    
    # Split the TD error update evenly
    q_system["shared"][state][action_idx] += (alpha / 2.0) * td_error
    q_system[misc_str][state][action_idx] += (alpha / 2.0) * td_error
    
    # Clip Q-values
    q_system["shared"][state][action_idx] = float(np.clip(q_system["shared"][state][action_idx], -Q_VALUE_CLIP, Q_VALUE_CLIP))
    q_system[misc_str][state][action_idx] = float(np.clip(q_system[misc_str][state][action_idx], -Q_VALUE_CLIP, Q_VALUE_CLIP))


# ---------------------------------------------------------------------------
# 4. Action selection — Adaptive Constraints
# ---------------------------------------------------------------------------

def apply_constraints(
    attempted_action: int,
    action_history: list[int],
    prev_att: float,
    misc_str: str,
    confusion_history: list[float]
) -> tuple[int, float]:
    """Hard safety constraints and rule-based action corrections."""
    final_action = attempted_action
    penalty = 0.0

    we_idx = ACTION_TO_IDX["worked_example"]
    q_idx  = ACTION_TO_IDX["question"]
    ex_idx = ACTION_TO_IDX["explain"]

    # 1. Attention Safety Enforcement (Hard limits)
    if prev_att < 2.5:
        if final_action != ex_idx:
            final_action = ex_idx
            penalty -= 5.0
    elif prev_att < 4.0:
        if final_action not in [ex_idx, we_idx]:
            final_action = ex_idx
            penalty -= 2.0

    # 2. Question Action Control (Max 2 per 5-step window)
    if final_action == q_idx:
        q_count = action_history[-5:].count(q_idx)
        if q_count >= 2:
            penalty -= 2.0 * (q_count - 1)
            final_action = ex_idx

    # 3. Action Stability Rule (Anti-Oscillation)
    if final_action == q_idx and len(action_history) >= 2:
        if action_history[-1] == q_idx and action_history[-2] == q_idx:
            final_action = ex_idx
            penalty -= 3.0
    
    if len(action_history) >= 4:
        recent_4 = action_history[-4:]
        is_oscillation = (
            recent_4 == [we_idx, q_idx, we_idx, q_idx] or 
            recent_4 == [q_idx, we_idx, q_idx, we_idx]
        )
        if is_oscillation and final_action in [we_idx, q_idx]:
            penalty -= 2.5
            final_action = ex_idx

    # 5. Confusion Reduction Rule (Monotonicity Bias)
    if len(confusion_history) >= 3:
        c_curr, c_prev, c_prev2 = confusion_history[-1], confusion_history[-2], confusion_history[-3]
        if c_curr > c_prev and c_prev > c_prev2:
            if final_action not in [ex_idx, we_idx]:
                final_action = ex_idx
                penalty -= 3.0

    return final_action, penalty


def select_action(
    q_system: dict[str, defaultdict],
    state: tuple,
    epsilon: float,
    rng: random.Random,
    obs_attention: float,
    misc_str: str,
    action_history: list[int],
    confusion_history: list[float]
) -> int:
    """Action selection applying strict safety pre-masking."""
    q_vals = get_q_values(q_system, state, misc_str).copy()
    allowed = list(ACTIONS.keys())

    def mask_except(allowed_names):
        allowed_idxs = [ACTION_TO_IDX[n] for n in allowed_names]
        to_remove = [a for a in allowed if a not in allowed_idxs]
        for a in to_remove:
            allowed.remove(a)
            q_vals[a] = -1e9

    # 1. Attention Safety Enforcement
    if obs_attention < 2.5:
        mask_except(["explain"])
    elif obs_attention < 4.0:
        mask_except(["explain", "worked_example"])

    # 5. Confusion Monotonicity Force Switch
    if len(confusion_history) >= 3:
        if confusion_history[-1] > confusion_history[-2] > confusion_history[-3]:
            mask_except(["explain", "worked_example"])

    if rng.random() < epsilon and allowed:
        return rng.choice(allowed)

    return int(np.argmax(q_vals))

# ---------------------------------------------------------------------------
# 5. Dataset Loader
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("Dataset must be a non-empty JSON array.")

    validated: list[dict[str, Any]] = []
    for i, record in enumerate(raw):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            continue
        record["confusion"]      = float(record["confusion"])
        record["attention"]      = float(record["attention"])
        record["next_confusion"] = float(record["next_confusion"])
        record["next_attention"] = float(record["next_attention"])
        record["reward"]         = float(record["reward"])
        record["done"]           = bool(record["done"])
        validated.append(record)

    print(f"[Loader] {len(validated)}/{len(raw)} samples loaded from {path}")
    return validated


# ---------------------------------------------------------------------------
# 6. Q-table Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_qtable(
    dataset: list[dict[str, Any]],
    alpha: float  = ALPHA_BOOTSTRAP,
    gamma: float  = GAMMA,
    n_epochs: int = BOOTSTRAP_EPOCHS,
) -> dict[str, defaultdict]:
    q_system = create_q_system()

    total_updates = 0
    for epoch in range(1, n_epochs + 1):
        count = 0
        for sample in dataset:
            action_str = sample["action"]
            if action_str not in ACTION_TO_IDX:
                continue

            action_idx = ACTION_TO_IDX[action_str]
            state = get_state(
                sample["confusion"], sample["attention"], sample["misconception"],
            )
            next_state = get_state(
                sample["next_confusion"], sample["next_attention"], sample["misconception"],
            )

            s = sample["next_confusion"] < DONE_CONFUSION_THRESHOLD
            # Placeholder histories for bootstrap samples
            r = compute_reward(
                sample["confusion"], sample["next_confusion"],
                sample["attention"], sample["next_attention"],
                done=s, success=s,
                action_idx=action_idx,
                misc_str=sample["misconception"],
                action_history=[],
                step=1,
                confusion_history=[sample["confusion"], sample["next_confusion"]],
                prev_reward=0.0
            )
            
            misc_str = sample["misconception"]
            if misc_str not in q_system:
                misc_str = "none"

            update_q(q_system, state, misc_str, action_idx, r, next_state, s, alpha, gamma)
            count += 1

        total_updates += count

    print(f"[Bootstrap] Done — {total_updates} total updates")
    return q_system


# ---------------------------------------------------------------------------
# 7. Save / Load Q-table
# ---------------------------------------------------------------------------

def save_q_table(q_system: dict[str, defaultdict], path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {k: dict(v) for k, v in q_system.items()}
    with open(path, "wb") as fh:
        pickle.dump(serializable, fh)
    print(f"[Model] Q-table system saved -> {path}")


def load_q_table(path: str = MODEL_PATH) -> dict[str, defaultdict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No saved Q-table at: {path}")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
        
    q_system = create_q_system()
    for k, v in data.items():
        if k in q_system:
            q_system[k].update(v)
            
    print(f"[Model] Q-table system loaded <- {path}")
    return q_system


# ---------------------------------------------------------------------------
# 8. Training Loop
# ---------------------------------------------------------------------------

def train(
    q_system: dict[str, defaultdict],
    n_episodes: int = N_EPISODES,
    max_steps: int  = MAX_STEPS,
    alpha: float    = ALPHA,
    gamma: float    = GAMMA,
    epsilon_start: float = EPSILON_START,
    epsilon_min: float   = EPSILON_MIN,
    seed: int       = SEED,
) -> tuple[dict[str, defaultdict], list[float]]:
    rng = random.Random(seed)
    
    episode_rewards: list[float] = []
    recent_rewards: list[float] = []
    epsilon = epsilon_start
    
    misconceptions = ["conceptual", "factual", "procedural", "transfer"]

    print(f"\n[Training] {n_episodes} episodes | eps={epsilon_start:.2f}->{epsilon_min:.2f}")
    print("-" * 60)

    for ep in range(1, n_episodes + 1):
        misc = rng.choice(misconceptions)
        env = EduForgeEnv(seed=rng.randint(0, 99_999), misconception_init=misc)
        obs = env.reset()

        last_action_idx: int | None = None
        prev_last_action_idx: int | None = None
        
        progress_signal = 0
        steps_since_improvement = 0
        
        action_history = []
        confusion_history = [obs.confusion]
        prev_reward = 0.0

        state = get_state_from_obs(obs, last_action_idx, prev_last_action_idx, progress_signal, steps_since_improvement)

        total_reward = 0.0

        domain_max_steps = 15 if misc == "procedural" else 10
        for step in range(1, domain_max_steps + 1):
            prev_conf = obs.confusion
            prev_att  = obs.attention

            action_idx = select_action(
                q_system, state, epsilon, rng,
                obs.attention, misc, action_history, confusion_history
            )
            
            attempted_action = action_idx
            action_idx, penalty = apply_constraints(
                attempted_action, action_history, prev_att, misc, confusion_history
            )

            action_tag = f"<STRATEGY>{ACTIONS[action_idx]}</STRATEGY>"
            obs, env_reward, _, _ = env.step(action_tag)

            action_history.append(action_idx)
            confusion_history.append(obs.confusion)

            success = obs.confusion < DONE_CONFUSION_THRESHOLD
            att_fail = obs.attention <= 0.5
            timeout = (step >= domain_max_steps)
            done = success or att_fail or timeout

            step_reward = env_reward

            # Apply hard constraint penalty
            step_reward += penalty

            # Update progress signal
            confusion_delta = prev_conf - obs.confusion
            attention_delta = obs.attention - prev_att
            if confusion_delta > 0 or attention_delta > 0:
                progress_signal = 1
                steps_since_improvement = 0
            elif confusion_delta < 0 or attention_delta < 0:
                progress_signal = -1
                steps_since_improvement += 1
            else:
                progress_signal = 0
                steps_since_improvement += 1

            next_state = get_state_from_obs(obs, action_idx, last_action_idx, progress_signal, steps_since_improvement)
            update_q(q_system, state, misc, attempted_action, step_reward, next_state, done, alpha, gamma)

            total_reward += step_reward
            prev_reward = step_reward
            state = next_state
            prev_last_action_idx = last_action_idx
            last_action_idx = action_idx

            if done:
                break

        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            
        # Adaptive Epsilon Update
        if len(recent_rewards) == 100 and ep % 10 == 0:
            avg_first_half = np.mean(recent_rewards[:50])
            avg_second_half = np.mean(recent_rewards[50:])
            
            if avg_second_half > avg_first_half + 0.5:
                # Improving -> decay faster
                epsilon = max(epsilon_min, epsilon * 0.95)
            elif avg_second_half < avg_first_half - 0.5:
                # Dropping -> increase noise
                epsilon = min(1.0, epsilon * 1.1)
            else:
                # Plateau -> slow decay
                epsilon = max(epsilon_min, epsilon * 0.99)
                
        # Base decay early on to ensure it doesn't get stuck at 1.0 initially
        if ep < 200:
            epsilon = max(epsilon_min, epsilon * 0.995)

        if ep % 500 == 0 or ep == 1:
            avg = float(np.mean(recent_rewards))
            print(f"  Ep {ep:>5}/{n_episodes} | eps={epsilon:.4f} | avg_reward(last 100)={avg:+.4f}")

    return q_system, episode_rewards


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    q_system: dict[str, defaultdict],
    n_episodes: int = EVAL_EPISODES,
    max_steps: int  = MAX_STEPS,
    seed: int       = SEED + 1,
) -> dict[str, Any]:
    rng = random.Random(seed)

    print("\n" + "=" * 60)
    print("EVALUATION — Greedy Policy")
    print("=" * 60)

    results = {"resolved": 0, "failed_timeout": 0, "failed_attention": 0}
    misconception_actions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    misconceptions = ["conceptual", "factual", "procedural", "transfer"]
    seeds_per_misc = 20
    total_episodes = len(misconceptions) * seeds_per_misc
    
    for misc_idx, fixed_m_str in enumerate(misconceptions):
        if fixed_m_str not in results:
            results[fixed_m_str] = {"resolved": 0, "failed_timeout": 0, "failed_attention": 0, "steps": [], "rewards": []}
        
        for seed_idx in range(seeds_per_misc):
            ep = misc_idx * seeds_per_misc + seed_idx + 1
            env = EduForgeEnv(seed=seed + seed_idx, misconception_init=fixed_m_str, attention_init=8.0)
            obs = env.reset()

            last_action_idx: int | None = None
            prev_last_action_idx: int | None = None
            total_reward = 0.0
            final_step = 0
            outcome = ""
            progress_signal = 0
            steps_since_improvement = 0
            
            action_history = []
            confusion_history = [obs.confusion]
            prev_reward = 0.0

            m_str = fixed_m_str
            state = get_state_from_obs(obs, last_action_idx, prev_last_action_idx, progress_signal, steps_since_improvement)

            print(f"\n--- Episode {ep} ---")
            print(f"  Misconception : {m_str}")
            print(f"  Initial       : confusion={obs.confusion:.2f}  attention={obs.attention:.2f}")

            domain_max_steps = 15 if m_str == "procedural" else 10
            for step in range(1, domain_max_steps + 1):
                prev_conf = obs.confusion
                prev_att  = obs.attention

                action_idx = select_action(
                    q_system, state, 0.0, rng,
                    obs.attention, m_str, action_history, confusion_history
                )
                
                attempted_action = action_idx
                action_idx, penalty = apply_constraints(
                    attempted_action, action_history, prev_att, m_str, confusion_history
                )
                
                chosen = ACTIONS[action_idx]
                action_tag = f"<STRATEGY>{chosen}</STRATEGY>"

                misconception_actions[m_str][chosen] += 1
                obs, env_reward, _, _ = env.step(action_tag)

                action_history.append(action_idx)
                confusion_history.append(obs.confusion)

                success = obs.confusion < DONE_CONFUSION_THRESHOLD
                att_fail = obs.attention <= 0.5
                timeout = (step >= domain_max_steps)
                done = success or att_fail or timeout

                step_reward = env_reward
                
                step_reward += penalty
                
                confusion_delta = prev_conf - obs.confusion
                attention_delta = obs.attention - prev_att
                if confusion_delta > 0 or attention_delta > 0:
                    progress_signal = 1
                    steps_since_improvement = 0
                elif confusion_delta < 0 or attention_delta < 0:
                    progress_signal = -1
                    steps_since_improvement += 1
                else:
                    progress_signal = 0
                    steps_since_improvement += 1

                total_reward += step_reward
                prev_reward = step_reward
                state = get_state_from_obs(obs, action_idx, last_action_idx, progress_signal, steps_since_improvement)

                print(f"  Step {step:>2} | action={chosen:<15} | "
                      f"confusion={obs.confusion:.2f}  attention={obs.attention:.2f} | "
                      f"reward={step_reward:+.2f}")

                prev_last_action_idx = last_action_idx
                last_action_idx = action_idx
                final_step = step

                if done:
                    if success:
                        outcome = "[RESOLVED]"
                        results[m_str]["resolved"] += 1
                        print(f"         >> RESOLVED  confusion={obs.confusion:.2f} < {DONE_CONFUSION_THRESHOLD}")
                    elif att_fail:
                        outcome = "[FAILED: disengaged]"
                        results[m_str]["failed_attention"] += 1
                        print(f"         >> FAILED  attention={obs.attention:.2f} < {ATTENTION_FAILURE_THRESHOLD}")
                    else:
                        outcome = "[FAILED: timeout]"
                        results[m_str]["failed_timeout"] += 1
                        print(f"         >> FAILED  confusion={obs.confusion:.2f} > {DONE_CONFUSION_THRESHOLD} (max steps)")
                    break

            results[m_str]["rewards"].append(total_reward)
            results[m_str]["steps"].append(final_step)
            print(f"  {outcome} after {final_step} step(s) | total_reward={total_reward:+.2f}")

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    total_res = sum(v["resolved"] for v in results.values() if isinstance(v, dict))
    total_tout = sum(v["failed_timeout"] for v in results.values() if isinstance(v, dict))
    total_att = sum(v["failed_attention"] for v in results.values() if isinstance(v, dict))
    total_eps = total_res + total_tout + total_att
    
    all_r = []
    all_s = []
    for v in results.values():
        if isinstance(v, dict):
            all_r.extend(v["rewards"])
            all_s.extend(v["steps"])
            
    sr = total_res / total_eps * 100 if total_eps > 0 else 0
    var_r = np.var(all_r) if all_r else 0.0
    print(f"  Overall Success: {total_res}/{total_eps} ({sr:.0f}%)")
    print(f"  Overall Avg steps: {np.mean(all_s):.1f}")
    print(f"  Reward Variance: {var_r:.2f}")
    
    for m, m_data in results.items():
        if not isinstance(m_data, dict):
            continue
        m_total = m_data["resolved"] + m_data["failed_timeout"] + m_data["failed_attention"]
        if m_total == 0:
            continue
        m_sr = m_data["resolved"] / m_total * 100
        print(f"\n  [{m.upper()}] Success: {m_data['resolved']}/{m_total} ({m_sr:.0f}%)")
        print(f"    Avg steps: {np.mean(m_data['steps']):.1f} | Avg reward: {np.mean(m_data['rewards']):+.2f}")
        print(f"    Failures: {m_data['failed_timeout']} timeout, {m_data['failed_attention']} attention")

    print("\n  POLICY — Dominant Strategies per Misconception")
    print("  " + "-" * 50)
    for m, counts in sorted(misconception_actions.items()):
        t = sum(counts.values())
        print(f"\n  {m} ({t} actions):")
        for act, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {act:<15} : {cnt:>3} ({cnt/t*100:>5.1f}%)")

    print("\n" + "=" * 60)
    return results


# ---------------------------------------------------------------------------
# 10. Human Feedback Hooks
# ---------------------------------------------------------------------------

class FeedbackHook:
    REWARD_GOOD      = +2.0
    REWARD_CONFUSING = -1.5
    REWARD_BORING    = -1.0

    def __init__(self, q_system: dict[str, defaultdict], alpha: float = ALPHA) -> None:
        self.q_system = q_system
        self.alpha = alpha

    def _apply(self, state: tuple, misc_str: str, action_idx: int, reward: float) -> float:
        update_q(self.q_system, state, misc_str, action_idx, reward, state, True, self.alpha, GAMMA)
        return reward

    def good(self, state: tuple, misc_str: str, action_idx: int) -> float:
        return self._apply(state, misc_str, action_idx, self.REWARD_GOOD)

    def confusing(self, state: tuple, misc_str: str, action_idx: int) -> float:
        return self._apply(state, misc_str, action_idx, self.REWARD_CONFUSING)

    def boring(self, state: tuple, misc_str: str, action_idx: int) -> float:
        return self._apply(state, misc_str, action_idx, self.REWARD_BORING)


# ---------------------------------------------------------------------------
# 11. Interactive REPL
# ---------------------------------------------------------------------------

_HIGH_CONFUSION_KW = {
    "don't understand", "dont understand", "lost", "confused",
    "no idea", "what", "help", "stuck", "not clear", "makes no sense",
}
_MED_CONFUSION_KW = {
    "somewhat", "maybe", "kind of", "sort of", "not sure",
    "partially", "a bit", "a little",
}
_ACTION_DESC: dict[str, str] = {
    "explain":        "Give a clear, step-by-step explanation of the concept.",
    "worked_example": "Walk through a fully worked example together.",
    "question":       "Ask the student a probing question to test understanding.",
    "correct_fact":   "Directly correct the factual error the student has made.",
    "analogize":      "Use a real-world analogy to build intuition.",
}

def estimate_state(
    user_input: str, misconception: str = "none",
) -> tuple[float, float, str]:
    text = user_input.lower()
    if any(kw in text for kw in _HIGH_CONFUSION_KW):
        return 8.0, 5.0, misconception
    elif any(kw in text for kw in _MED_CONFUSION_KW):
        return 5.0, 5.0, misconception
    else:
        return 3.0, 6.0, misconception


def interact(q_system: dict[str, defaultdict] | None = None) -> None:
    if q_system is None:
        q_system = load_q_table(MODEL_PATH)

    hook = FeedbackHook(q_system)

    print("\n" + "=" * 60)
    print("EduForge Interactive Tutoring Session")
    print("=" * 60)
    print("  Misconception types: " + ", ".join(MISCONCEPTION_MAP.keys()))
    print("  Commands: 'switch <type>', 'quit'")
    print("  Feedback: y = helpful, n = confusing, b = boring")
    print("=" * 60)

    misconception = "none"
    session_pos, session_neg, session_bored = 0, 0, 0
    total_reward = 0.0

    while True:
        print(f"\n[Active misconception: {misconception}]")
        try:
            user_input = input("Student > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("[Session ended]")
            break
        if user_input.lower().startswith("switch "):
            req = user_input[7:].strip().lower()
            if req in MISCONCEPTION_MAP:
                misconception = req
                print(f"  [System] Switched to: {misconception}")
            else:
                print(f"  [System] Unknown. Options: {list(MISCONCEPTION_MAP)}")
            continue

        confusion, attention, m = estimate_state(user_input, misconception)
        state = get_state(confusion, attention, m)
        
        q_vals = get_q_values(q_system, state, misconception)
        action_idx = int(np.argmax(q_vals))
        action_name = ACTIONS[action_idx]

        print(f"  [State]  confusion={confusion:.1f}  attention={attention:.1f}")
        print(f"  [Action] {action_name}")
        print(f"  [Tutor]  {_ACTION_DESC[action_name]}")

        try:
            fb = input("  Feedback (y/n/b): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if fb == "y":
            r = hook.good(state, misconception, action_idx)
            session_pos += 1
            print("  [+] Positive signal recorded.")
        elif fb == "b":
            r = hook.boring(state, misconception, action_idx)
            session_bored += 1
            print("  [~] Boredom signal recorded — agent adjusting.")
        else:
            r = hook.confusing(state, misconception, action_idx)
            session_neg += 1
            print("  [-] Negative signal recorded — agent adjusting.")

        total_reward += r

        if confusion < DONE_CONFUSION_THRESHOLD:
            print("  [EduForge] Student appears to understand. Great job!")

    total_turns = session_pos + session_neg + session_bored
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    if total_turns > 0:
        print(f"  Helpful  : {session_pos}  ({session_pos/total_turns*100:.0f}%)")
        print(f"  Confusing: {session_neg}")
        print(f"  Boring   : {session_bored}")
        print(f"  Reward   : {total_reward:+.1f}")
        save_q_table(q_system, MODEL_PATH)
    else:
        print("  No feedback — Q-table unchanged.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("EduForge Q-Learning Pipeline")
    print("=" * 60)

    print("\n[1/4] Loading dataset...")
    dataset = load_dataset(DATASET_PATH)

    print(f"\n[2/4] Bootstrapping Q-table ({BOOTSTRAP_EPOCHS} epochs)...")
    # Disable bootstrapping because offline data does not follow the new hard constraints
    # and would poison the initial Q-table.
    q_system = create_q_system()

    print("\n[3/4] Online Q-learning training...")
    q_system, reward_history = train(
        q_system,
        n_episodes=N_EPISODES, max_steps=MAX_STEPS,
        alpha=ALPHA, gamma=GAMMA,
        epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
        seed=SEED,
    )

    thirds = len(reward_history) // 3 or 1
    print(f"\n  Reward trend  - "
          f"first 3rd avg: {float(np.mean(reward_history[:thirds])):+.4f} | "
          f"last 3rd avg: {float(np.mean(reward_history[-thirds:])):+.4f}")

    save_q_table(q_system, MODEL_PATH)

    print("\n[4/4] Evaluating greedy policy...")
    evaluate(q_system, n_episodes=EVAL_EPISODES, max_steps=MAX_STEPS, seed=SEED + 1)

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EduForge Q-Learning Pipeline")
    parser.add_argument("--interact", action="store_true")
    args = parser.parse_args()

    if args.interact:
        interact()
    else:
        main()
