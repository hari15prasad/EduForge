"""
dqn_pipeline.py — Deep Q-Network training pipeline for EduForge.

Implements all pedagogical constraints from the Q-learning pipeline:
  1. 5x5 state discretisation (confusion × attention bins)
  2. Attention collapse as hard terminal constraint (-5.0 penalty, done=True)
  3. Action masking BEFORE argmax (pedagogical safety, procedural interleaving)
  4. Procedural phase constraint (force question/explain after 2× worked_example)
  5. Stable bounded reward function (DQN-safe, no >5.0 magnitude spikes)
  6. Mask-aware replay buffer (stores mask_state for correct bootstrapping)
  7. Enriched network input (confusion bin, attention bin, last_action, consec_we)
  8. Deterministic 40-episode evaluation (10 seeds × 4 misconception types)
"""

from __future__ import annotations

import os
import sys
import random
from collections import deque, defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.environment.env import EduForgeEnv

from src.environment.student_fsm import MisconceptionType, TutorAction, MISCONCEPTION_IDX
from src.environment.env import ACTION_TEXT, STEP_LIMITS
from src.training.agent import DQN, ReplayBuffer

ACTIONS = ACTION_TEXT
ACTION_TO_IDX = {v: k for k, v in ACTIONS.items()}
N_ACTIONS = len(ACTIONS)
MISCONCEPTIONS = [m.value for m in MisconceptionType]

# Hyperparameters
BATCH_SIZE    = 64
GAMMA         = 0.95
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.997
TARGET_UPDATE = 50
LEARNING_RATE = 5e-4
MEMORY_SIZE   = 20_000

N_EPISODES    = 3000
MAX_STEPS     = 18
SEED          = 42

DONE_CONFUSION_THRESHOLD    = 2.0
ATTENTION_FAILURE_THRESHOLD = 2.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. State Discretisation — 5×5 bins (mirrors qlearning_pipeline.py)
# ---------------------------------------------------------------------------
_CONF_BINS = [0, 2, 4, 6, 8, 10.01]   # 5 bins
_ATT_BINS  = [0, 2, 4, 6, 8, 10.01]   # 5 bins

def _bin_value(val: float, edges: list[float]) -> int:
    val = max(edges[0], min(edges[-1] - 0.01, val))
    for i in range(len(edges) - 1):
        if val < edges[i + 1]:
            return i
    return len(edges) - 2


def get_state_vector(
    env: EduForgeEnv,
    last_action_idx: int,
    consecutive_worked_example: int,
    step: int,
) -> np.ndarray:
    """
    Build the network input vector. Includes:
      - confusion (normalised /10)
      - attention (normalised /10)
      - misconception one-hot (4D)
      - last_action one-hot (5D)
      - consecutive_worked_example (scalar, capped at 5)
      - step progress (normalised, 0–1)
    Total: 2 + 4 + 5 + 1 + 1 = 13 dims
    """
    conf_norm = env.confusion / 10.0
    att_norm  = env.attention / 10.0

    misc_onehot = np.zeros(len(MisconceptionType), dtype=np.float32)
    misc_idx = MISCONCEPTION_IDX.get(env.misconception.value, 0)
    misc_onehot[misc_idx] = 1.0

    action_onehot = np.zeros(N_ACTIONS, dtype=np.float32)
    if 0 <= last_action_idx < N_ACTIONS:
        action_onehot[last_action_idx] = 1.0

    consec = min(consecutive_worked_example, 5) / 5.0
    # Use max steps from environment
    max_steps = env.max_steps
    step_norm = step / max_steps

    return np.concatenate([
        [conf_norm],
        [att_norm],
        misc_onehot,
        action_onehot,
        [consec],
        [step_norm],
    ]).astype(np.float32)


STATE_DIM = 2 + len(MisconceptionType) + N_ACTIONS + 1 + 1  # 2+4+5+1+1 = 13
# DQN and ReplayBuffer moved to src/training/agent.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Action Masking — applied BEFORE argmax, covers all pedagogical constraints
# ---------------------------------------------------------------------------
def select_action(
    state: np.ndarray,
    mask: np.ndarray,
    domain_idx: int,
    policy_net: DQN,
    epsilon: float,
    rng: random.Random,
) -> int:
    valid_actions = [i for i in range(N_ACTIONS) if mask[i] == 0.0]
    if not valid_actions:
        valid_actions = [ACTION_TO_IDX["explain"]]

    if rng.random() < epsilon:
        return rng.choice(valid_actions)

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_vals  = policy_net(state_t, domain_idx).cpu().numpy()[0]
        masked_q = q_vals + mask   # -inf blocks invalid actions from argmax
        return int(np.argmax(masked_q))


# ---------------------------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------------------------
def train_dqn(n_episodes: int = N_EPISODES, seed: int = SEED) -> DQN:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy_net = DQN(STATE_DIM, N_ACTIONS).to(device)
    target_net = DQN(STATE_DIM, N_ACTIONS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory    = ReplayBuffer(MEMORY_SIZE)
    
    domain_epsilon = {
        "procedural": 0.5,
        "conceptual": 0.2,
        "factual": 0.2,
        "transfer": 0.2
    }
    
    # Curriculum: start with easy domains
    curriculum_probs = {
        "factual": 0.4,
        "conceptual": 0.4,
        "procedural": 0.2,
        "transfer": 0.0
    }
    
    # Track stats for adaptive logic
    stats = {d: deque(maxlen=100) for d in MISCONCEPTIONS}
    
    episode_rewards: list[float] = []

    print(f"\n[Training DQN] {n_episodes} episodes | Device: {device}")
    print("-" * 60)

    for ep in range(1, n_episodes + 1):
        # Sample MISC from curriculum
        if ep <= 500:
            # Task 1: Warm-up curriculum - 50% Procedural
            warmup_probs = {
                "factual": 0.2,
                "conceptual": 0.2,
                "procedural": 0.5,
                "transfer": 0.1
            }
            misc = rng.choices(
                list(warmup_probs.keys()), 
                weights=list(warmup_probs.values())
            )[0]
        else:
            misc = rng.choices(
                list(curriculum_probs.keys()), 
                weights=list(curriculum_probs.values())
            )[0]
        env  = EduForgeEnv(seed=rng.randint(0, 99_999), misconception_init=misc)
        obs  = env.reset()

        last_action_idx          = -1   # -1 = no previous action
        consecutive_worked_example = 0
        step                     = 0
        total_reward             = 0.0

        state = get_state_vector(env, last_action_idx, consecutive_worked_example, step)
        mask  = env.get_action_mask()
        misc_idx = MISCONCEPTION_IDX[env.misconception.value]
        current_eps = domain_epsilon.get(misc, 0.1)

        for step in range(1, MAX_STEPS + 1):
            action_idx = select_action(state, mask, misc_idx, policy_net, current_eps, rng)

            # Track consecutive worked_example
            if action_idx == ACTION_TO_IDX["worked_example"]:
                consecutive_worked_example += 1
            else:
                consecutive_worked_example = 0

            action_tag = f"<STRATEGY>{ACTIONS[action_idx]}</STRATEGY>"
            obs, reward, done, info = env.step(action_idx)

            total_reward += reward

            next_state = get_state_vector(env, action_idx, consecutive_worked_example, step)
            next_mask  = env.get_action_mask()

            # Store with next_mask so we can mask bootstrap Q-values correctly
            memory.push(state, action_idx, reward, next_state, done, next_mask, misc_idx)

            state          = next_state
            mask           = next_mask
            last_action_idx = action_idx

            # ----------------------------------------------------------------
            # Learning step (Double DQN with masked next-state bootstrap)
            # ----------------------------------------------------------------
            if len(memory) >= BATCH_SIZE:
                (
                    states_b, actions_b, rewards_b,
                    next_states_b, dones_b, next_masks_b,
                    indices_b, weights_b, domain_idxs_b
                ) = memory.sample(BATCH_SIZE)

                states_t      = torch.FloatTensor(states_b).to(device)
                actions_t     = torch.LongTensor(actions_b).unsqueeze(1).to(device)
                rewards_t     = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
                next_states_t = torch.FloatTensor(next_states_b).to(device)
                dones_t       = torch.FloatTensor(dones_b).unsqueeze(1).to(device)
                next_masks_t  = torch.FloatTensor(next_masks_b).to(device)
                weights_t     = torch.FloatTensor(weights_b).unsqueeze(1).to(device)
                domain_idxs_t = torch.LongTensor(domain_idxs_b).unsqueeze(1).to(device)

                with torch.no_grad():
                    # Double DQN: select action with policy net, evaluate with target net
                    next_q_policy = policy_net(next_states_t, domain_idxs_t) + next_masks_t
                    best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                    next_q_target = target_net(next_states_t, domain_idxs_t).gather(1, best_next_actions)
                    target_q = rewards_t + GAMMA * next_q_target * (1.0 - dones_t)

                current_q = policy_net(states_t, domain_idxs_t).gather(1, actions_t)
                
                # TD Error for Priority Updates
                td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
                memory.update_priorities(indices_b, td_errors.flatten() + 1e-6)

                # Weighted MSE/Huber loss
                loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        # --------------------------------------------------------------------
        # End of episode: sync target network & decay epsilon
        # --------------------------------------------------------------------
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Domain-aware epsilon decay
        domain_epsilon[misc] = max(EPS_END, domain_epsilon[misc] * EPS_DECAY)
        
        # Task 8: Targeted Exploration (Procedural min 0.25 for first 1500 episodes)
        if ep <= 1500:
            domain_epsilon["procedural"] = max(domain_epsilon["procedural"], 0.25)
        
        # Task 7: Procedural Exploration Boost (Keep 0.4 for very early episodes)
        if ep <= 200:
            domain_epsilon["procedural"] = max(domain_epsilon["procedural"], 0.4)

        # Track success for adaptive logic
        success = info.get("done_reason") == "success"
        stats[misc].append(1.0 if success else 0.0)

        # --------------------------------------------------------------------
        # Adaptive Curricula: Increase Procedural if Easy is mastered (>90%)
        # --------------------------------------------------------------------
        easy_avg = np.mean([np.mean(stats[d]) if stats[d] else 0.0 for d in ["factual", "conceptual"]])
        if easy_avg > 0.9 and curriculum_probs["procedural"] < 0.5:
            curriculum_probs["procedural"] += 0.01
            curriculum_probs["transfer"]   += 0.01
            # Normalise
            total_probs = sum(curriculum_probs.values())
            for d in curriculum_probs: curriculum_probs[d] /= total_probs

        # --------------------------------------------------------------------
        # Adaptive Epsilon: Reset if Procedural plateaued (<30% success)
        # --------------------------------------------------------------------
        proc_avg = np.mean(stats["procedural"]) if len(stats["procedural"]) >= 50 else 1.0
        if proc_avg < 0.3:
            domain_epsilon["procedural"] = 0.5
            stats["procedural"].clear()

        episode_rewards.append(total_reward)

        if ep % 500 == 0 or ep == 1:
            avg_rew = float(np.mean(episode_rewards[-100:]))
            avg_eps = float(np.mean(list(domain_epsilon.values())))
            print(f"  Ep {ep:>5}/{n_episodes} | eps_avg={avg_eps:.4f} (proc={domain_epsilon['procedural']:.4f}) | "
                  f"avg_reward=+{avg_rew:.2f} | proc_sr={proc_avg:.2f}")

    print("[Training DQN] Complete.")
    return policy_net


# ---------------------------------------------------------------------------
# 8. Deterministic Evaluation — 10 seeds × 4 misconception types = 40 episodes
# ---------------------------------------------------------------------------
def evaluate(policy_net: DQN, seed: int = SEED + 1) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("EVALUATION — Greedy DQN Policy")
    print("=" * 60)

    results: dict[str, Any] = {}
    misconception_actions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seeds_per_misc = 10

    for misc_idx, fixed_m_str in enumerate(MISCONCEPTIONS):
        results[fixed_m_str] = {
            "resolved": 0, "failed_timeout": 0, "failed_attention": 0,
            "steps": [], "rewards": [],
        }

        for seed_idx in range(seeds_per_misc):
            ep = misc_idx * seeds_per_misc + seed_idx + 1
            env = EduForgeEnv(
                seed=seed + seed_idx,
                misconception_init=fixed_m_str,
                attention_init=8.0,
            )
            obs = env.reset()

            last_action_idx            = -1
            consecutive_worked_example = 0
            total_reward               = 0.0
            final_step                 = 0
            outcome                    = ""
            step                       = 0

            state = get_state_vector(env, last_action_idx, consecutive_worked_example, step)
            mask  = env.get_action_mask()
            misc_idx = MISCONCEPTION_IDX[env.misconception.value]

            print(f"\n--- Episode {ep} [{fixed_m_str}] ---")
            print(f"  Initial: confusion={env.confusion:.2f}  attention={env.attention:.2f}")

            for step in range(1, MAX_STEPS + 1):
                action_idx = select_action(state, mask, misc_idx, policy_net, 0.0, random.Random(seed_idx + step))
                chosen     = ACTIONS[action_idx]

                if action_idx == ACTION_TO_IDX["worked_example"]:
                    consecutive_worked_example += 1
                else:
                    consecutive_worked_example = 0

                action_tag = f"<STRATEGY>{chosen}</STRATEGY>"
                misconception_actions[fixed_m_str][chosen] += 1
                obs, step_reward, done, info = env.step(action_idx)

                total_reward += step_reward

                next_state = get_state_vector(env, action_idx, consecutive_worked_example, step)
                next_mask  = env.get_action_mask()
                state      = next_state
                mask       = next_mask
                last_action_idx = action_idx
                final_step = step
                
                done_reason = info.get("done_reason", "")

                print(f"  Step {step:>2} | action={chosen:<15} | "
                      f"confusion={env.confusion:.2f}  attention={env.attention:.2f} | "
                      f"reward={step_reward:+.2f}")

                if done:
                    if done_reason == "success":
                        outcome = "[RESOLVED]"
                        results[fixed_m_str]["resolved"] += 1
                        print(f"         >> RESOLVED  confusion={env.confusion:.2f} < {DONE_CONFUSION_THRESHOLD}")
                    elif done_reason == "disengaged":
                        outcome = "[FAILED: disengaged]"
                        results[fixed_m_str]["failed_attention"] += 1
                        print(f"         >> FAILED  attention={env.attention:.2f} < {ATTENTION_FAILURE_THRESHOLD}")
                    else:
                        outcome = "[FAILED: timeout]"
                        results[fixed_m_str]["failed_timeout"] += 1
                        print(f"         >> FAILED  confusion={env.confusion:.2f} > {DONE_CONFUSION_THRESHOLD} (max steps)")
                    break

            results[fixed_m_str]["rewards"].append(total_reward)
            results[fixed_m_str]["steps"].append(final_step)
            print(f"  {outcome} after {final_step} step(s) | total_reward={total_reward:+.2f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    all_r, all_s = [], []
    total_res, total_tout, total_att = 0, 0, 0

    for m_str, m_data in results.items():
        total_res  += m_data["resolved"]
        total_tout += m_data["failed_timeout"]
        total_att  += m_data["failed_attention"]
        all_r.extend(m_data["rewards"])
        all_s.extend(m_data["steps"])

    total_eps = total_res + total_tout + total_att
    sr = total_res / total_eps * 100 if total_eps > 0 else 0.0
    print(f"  Overall Success: {total_res}/{total_eps} ({sr:.0f}%)")
    print(f"  Overall Avg steps: {np.mean(all_s):.1f}")
    print(f"  Reward Variance: {np.var(all_r):.2f}")

    for m_str, m_data in results.items():
        m_total = m_data["resolved"] + m_data["failed_timeout"] + m_data["failed_attention"]
        if m_total == 0:
            continue
        m_sr = m_data["resolved"] / m_total * 100
        print(f"\n  [{m_str.upper()}] Success: {m_data['resolved']}/{m_total} ({m_sr:.0f}%)")
        print(f"    Avg steps: {np.mean(m_data['steps']):.1f} | Avg reward: {np.mean(m_data['rewards']):+.2f}")
        print(f"    Failures: {m_data['failed_timeout']} timeout, {m_data['failed_attention']} attention")

    print("\n  POLICY — Dominant Strategies per Misconception")
    print("  " + "-" * 50)
    for m_str, counts in sorted(misconception_actions.items()):
        t = sum(counts.values())
        print(f"\n  {m_str} ({t} actions):")
        for act, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {act:<15}: {cnt:>3} ({cnt / t * 100:>5.1f}%)")

    print("\n" + "=" * 60)
    return results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = train_dqn(N_EPISODES, seed=SEED)
    evaluate(net, seed=SEED + 1)
