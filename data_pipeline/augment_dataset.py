"""
augment_dataset.py
──────────────────
Expands the ASSISTments RL transition dataset by 3–5x via:
  • Gaussian state noise + occasional spike events
  • Behavioural variation (suboptimal actions, repetition)
  • Reward noise with sequence-quality bonuses / penalties
  • Temporally consistent next-state generation
  • Hard student archetypes (10% difficult, 5% disengaged)

Output schema is identical to training_samples.json.
"""

import json
import random
import math
import numpy as np

# ──────────────────────────────
# Config
# ──────────────────────────────

INPUT_PATH  = "src/environment/training_samples.json"
OUTPUT_PATH = "src/environment/training_samples_augmented.json"

AUGMENT_FACTOR  = 4       # Each original sample → this many augmented copies
RANDOM_SEED     = 42

# Probability gates
P_SPIKE_CONF    = 0.08    # P(sudden confusion spike on a sample)
P_SPIKE_ATTN    = 0.08    # P(sudden attention drop on a sample)
P_SUBOPTIMAL    = 0.25    # P(replace optimal action with suboptimal)
P_REPEAT_ACTION = 0.12    # P(force same action as previous in batch)
P_DIFFICULT     = 0.10    # P(tag as difficult student)
P_DISENGAGED    = 0.05    # P(tag as disengaged student)

# Pedagogically ideal action per misconception (the "correct" teacher choice)
IDEAL_ACTION = {
    "procedural":  "worked_example",
    "conceptual":  "analogize",
    "factual":     "correct_fact",
    "transfer":    "explain",
}

# All valid actions
ALL_ACTIONS = ["explain", "analogize", "worked_example", "question", "correct_fact"]

# Expected confusion delta per (misconception, action) — positive = helps
# Values are approximate pedagogical priors; noise is added on top.
EFFICACY = {
    # (misconception, action) → mean confusion reduction
    ("procedural",  "worked_example"): +1.4,
    ("procedural",  "question"):       +0.5,
    ("procedural",  "explain"):        +0.2,
    ("procedural",  "analogize"):      -0.3,   # slightly off-topic
    ("procedural",  "correct_fact"):   -0.1,
    ("conceptual",  "analogize"):      +1.5,
    ("conceptual",  "explain"):        +0.8,
    ("conceptual",  "question"):       +0.3,
    ("conceptual",  "worked_example"): +0.0,
    ("conceptual",  "correct_fact"):   -0.2,
    ("factual",     "correct_fact"):   +2.0,
    ("factual",     "explain"):        +0.5,
    ("factual",     "analogize"):      +0.1,
    ("factual",     "worked_example"): -0.1,
    ("factual",     "question"):       +0.3,
    ("transfer",    "explain"):        +1.2,
    ("transfer",    "analogize"):      +0.9,
    ("transfer",    "question"):       +0.5,
    ("transfer",    "worked_example"): +0.3,
    ("transfer",    "correct_fact"):   -0.1,
}


# ──────────────────────────────
# Helpers
# ──────────────────────────────

def clamp(val, lo=0.0, hi=10.0):
    return float(max(lo, min(hi, val)))


def gaussian(mu=0.0, sigma=1.0):
    return np.random.normal(mu, sigma)


def pick_suboptimal(misconception, current_action):
    """Return a pedagogically suboptimal action for this misconception."""
    ideal = IDEAL_ACTION.get(misconception, "explain")
    alternatives = [a for a in ALL_ACTIONS if a != ideal]
    # Prefer actions that are "close but wrong" vs. random
    weighted = [a for a in alternatives if a != current_action]
    return random.choice(weighted) if weighted else random.choice(alternatives)


def compute_reward(conf_before, conf_after, attention, action, misconception,
                   repeated=False):
    """
    Reward function mirroring the training pipeline's intent:
      delta  → main signal
      attention bonus → engagement quality
      repetition penalty → discourages spamming one action
    """
    delta  = conf_before - conf_after          # positive = confusion fell = good
    reward = delta + 0.2 * (attention - 5.0)
    if attention < 2.0:
        reward -= 3.0                           # disengagement penalty
    if repeated:
        reward -= 0.8                           # repetition discouraged
    return float(np.clip(reward + gaussian(0, 0.3), -10, 10))


def next_confusion(conf, misconception, action, difficult=False, disengaged=False):
    """
    Compute next confusion using pedagogical efficacy table + noise.
    Difficult students improve slower; disengaged students ignore instruction.
    """
    base_delta = EFFICACY.get((misconception, action), 0.0)

    if difficult:
        base_delta *= 0.4        # slow improvement
    if disengaged:
        base_delta *= 0.1        # barely responds

    noise = gaussian(0, 0.5)
    return clamp(conf - base_delta + noise)


def next_attention(attn, action, difficult=False, disengaged=False):
    """
    Attention evolves with slight drift + volatility.
    Disengaged students tend downward; normal students fluctuate.
    """
    drift = -0.2 if disengaged else gaussian(0, 0.3)
    if difficult:
        drift -= 0.1
    return clamp(attn + drift, 0, 10)


def add_state_noise(confusion, attention):
    """
    Apply Gaussian noise to state values, plus occasional spike events.
    Returns (noisy_confusion, noisy_attention, had_conf_spike, had_attn_drop).
    """
    conf_noise = gaussian(0, random.uniform(0.3, 1.0))
    attn_noise = gaussian(0, random.uniform(0.2, 0.8))

    conf_spike = 0.0
    attn_drop  = 0.0

    if random.random() < P_SPIKE_CONF:
        conf_spike = random.uniform(1.0, 2.0)   # sudden confusion surge

    if random.random() < P_SPIKE_ATTN:
        attn_drop  = random.uniform(1.0, 2.0)   # sudden attention drop

    noisy_conf = clamp(confusion + conf_noise + conf_spike)
    noisy_attn = clamp(attention + attn_noise - attn_drop)
    return noisy_conf, noisy_attn


# ──────────────────────────────
# Archetype Flags
# ──────────────────────────────

def sample_archetype():
    """Returns (difficult, disengaged) booleans."""
    r = random.random()
    if r < P_DISENGAGED:
        return False, True
    elif r < P_DISENGAGED + P_DIFFICULT:
        return True, False
    return False, False


# ──────────────────────────────
# Core Augmentation
# ──────────────────────────────

def augment_sample(sample, prev_action=None):
    """
    Produce one augmented transition from an original sample.

    Steps:
      1. Noisify state
      2. Decide archetype
      3. Possibly swap action (behavioural variation)
      4. Compute next state from pedagogical model
      5. Compute reward with quality adjustments
    """
    misconception = sample["misconception"]
    action        = sample["action"]

    # 1. Noisify state
    conf, attn = add_state_noise(sample["confusion"], sample["attention"])

    # 2. Archetype
    difficult, disengaged = sample_archetype()

    # 3. Behavioural variation
    repeated = False
    if random.random() < P_SUBOPTIMAL:
        action = pick_suboptimal(misconception, action)
    elif prev_action and random.random() < P_REPEAT_ACTION:
        action   = prev_action   # teacher repeats previous move
        repeated = True

    # 4. Temporally consistent next state
    next_conf = next_confusion(conf, misconception, action, difficult, disengaged)
    next_attn = next_attention(attn, action, difficult, disengaged)

    # Disengaged: force low attention in next state
    if disengaged and random.random() < 0.6:
        next_attn = clamp(next_attn - random.uniform(0.5, 1.5), 0, 10)

    # 5. Reward
    reward = compute_reward(conf, next_conf, next_attn, action, misconception,
                            repeated=repeated)

    # ── Sequence quality reward shaping ──
    ideal = IDEAL_ACTION.get(misconception, "explain")
    if action == ideal:
        reward = float(np.clip(reward + gaussian(0.1, 0.15), -10, 10))   # slight boost
    elif repeated:
        pass  # already penalised above

    return {
        "confusion":      round(conf,      6),
        "attention":      round(attn,      6),
        "misconception":  misconception,
        "action":         action,
        "reward":         round(reward,    6),
        "next_confusion": round(next_conf, 6),
        "next_attention": round(next_attn, 6),
        "done":           bool(next_conf < 2.0),
    }


# ──────────────────────────────
# Pipeline
# ──────────────────────────────

def augment():
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Load original dataset
    print(f"[INFO] Loading {INPUT_PATH} ...")
    with open(INPUT_PATH, "r") as f:
        originals = json.load(f)

    print(f"[INFO] Original transitions: {len(originals)}")
    print(f"[INFO] Target augmentation factor: {AUGMENT_FACTOR}x")

    augmented = []
    prev_action = None   # track last action for repetition simulation

    for sample in originals:
        for copy_idx in range(AUGMENT_FACTOR):
            aug = augment_sample(sample, prev_action=prev_action if copy_idx > 0 else None)
            augmented.append(aug)
            prev_action = aug["action"]

    # Shuffle so archetypes / noise patterns don't cluster
    random.shuffle(augmented)

    # ── Debug metrics ──
    rewards      = [t["reward"]      for t in augmented]
    confusions   = [t["confusion"]   for t in augmented]
    attentions   = [t["attention"]   for t in augmented]
    done_count   = sum(1 for t in augmented if t["done"])
    action_dist  = {}
    for t in augmented:
        action_dist[t["action"]] = action_dist.get(t["action"], 0) + 1

    sep = "-" * 52
    print(f"\n[STATS] Augmented Dataset")
    print(sep)
    print(f"  Total transitions      : {len(augmented)}")
    print(f"  Expansion factor       : {len(augmented) / len(originals):.2f}x")
    print(f"  Avg reward             : {np.mean(rewards):.4f}")
    print(f"  Reward std             : {np.std(rewards):.4f}")
    print(f"  Avg confusion (state)  : {np.mean(confusions):.4f}")
    print(f"  Avg attention (state)  : {np.mean(attentions):.4f}")
    print(f"  Done (confusion < 2)   : {done_count} ({100*done_count/len(augmented):.1f}%)")
    print(f"  Action distribution:")
    total = len(augmented)
    for act, cnt in sorted(action_dist.items(), key=lambda x: -x[1]):
        print(f"    {act:<20} {cnt:>5}  ({100*cnt/total:.1f}%)")
    print(sep + "\n")

    # Save
    print(f"[INFO] Saving to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"[DONE] Augmented dataset saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    augment()
