import pandas as pd
import json
import numpy as np
from collections import defaultdict

INPUT_PATH = "data/assignments_2012.csv"
OUTPUT_PATH = "src/environment/training_samples.json"

CHUNK_SIZE = 50000    # Number of rows per CSV chunk (tune to available RAM)
MAX_TRANSITIONS = 10000  # Cap on total output transitions


# ---------------------------
# Utility: Sequence Sorting
# ---------------------------

def sort_key(row):
    """
    Return a sortable value for ordering interactions within a sequence.
    Priority: start_time → end_time → problem_log_id → _order (fallback index).
    'order_id' is intentionally excluded.
    """
    for col in ["start_time", "end_time", "problem_log_id"]:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    # Ultimate fallback: incremental index assigned during loading
    return float(row.get("_order", 0))


# ---------------------------
# Utility: Feature Extraction
# ---------------------------

def map_action(row):
    """Map ASSISTments interaction → pedagogical action."""
    if row.get("hint_count", 0) > 0:
        return "worked_example"
    elif row.get("attempt_count", 1) > 1:
        return "question"
    elif row.get("correct", 0) == 0:
        return "explain"
    else:
        return "analogize"


def compute_confusion(row):
    """Higher attempts + incorrect → more confusion (0–10 scale)."""
    base = 5.0
    attempts = row.get("attempt_count", 1)
    correct = row.get("correct", 0)
    confusion = base + attempts * 0.8 - correct * 2.0
    return np.clip(confusion + np.random.normal(0, 0.3), 0, 10)


def compute_attention(row):
    """Simple proxy for attention based on time spent on problem."""
    time_spent = row.get("ms_first_response", 30000)
    attention = 5 + (time_spent / 60000)   # ~1 min → neutral attention
    return np.clip(attention + np.random.normal(0, 0.5), 0, 10)


def compute_reward(prev_conf, new_conf, attention):
    """Reward = confusion reduction + attention bonus; penalise disengagement."""
    delta = prev_conf - new_conf
    reward = delta + 0.2 * (attention - 5)
    if attention < 2:
        reward -= 3   # strong penalty for disengagement
    return float(np.clip(reward, -10, 10))


def classify_misconception(row):
    """Rough heuristic classifier based on skill name."""
    skill = str(row.get("skill", "")).lower()
    if "concept" in skill:
        return "conceptual"
    elif "fact" in skill:
        return "factual"
    elif "transfer" in skill:
        return "transfer"
    else:
        return "procedural"


def smooth_confusion(history, new_val):
    """
    Moving average over the last 2 confusion values (including current).
    Reduces noise caused by random jitter in single steps.
    """
    if len(history) >= 1:
        return (history[-1] + new_val) / 2.0
    return new_val


# ---------------------------
# Main Pipeline
# ---------------------------

def process():
    # ── Step 1: Load CSV in chunks, group by composite key (user_id, skill_id) ──
    # Using a composite key captures per-skill learning trajectories, which are
    # more meaningful than raw per-user sequences.

    sequences = defaultdict(list)   # key: (user_id, skill_or_problem_id)

    print("[INFO] Reading CSV in chunks...")

    row_count = 0
    for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            user_id = row.get("user_id")
            if pd.isna(user_id):
                continue

            # Composite grouping key: prefer skill_id, fallback to problem_id
            skill_id = row.get("skill_id")
            if pd.isna(skill_id) or skill_id is None:
                skill_id = row.get("problem_id", "unknown")

            group_key = (user_id, skill_id)

            # Convert to dict and assign incremental index as last-resort sort key
            row_dict = row.to_dict()
            row_dict["_order"] = row_count   # fallback ordering (no order_id)

            sequences[group_key].append(row_dict)
            row_count += 1

            if row_count >= MAX_TRANSITIONS * 3:   # load extra to account for filtering
                break
        if row_count >= MAX_TRANSITIONS * 3:
            break

    print(f"[INFO] Sequences loaded (before filter): {len(sequences)}")

    # ── Step 2: Debug metrics before filtering ──
    raw_lengths = [len(seq) for seq in sequences.values()]
    if raw_lengths:
        print(f"[DEBUG] Average interactions per sequence (raw): {np.mean(raw_lengths):.2f}")
        print(f"[DEBUG] Max interactions per sequence (raw):     {max(raw_lengths)}")

    # ── Step 3: Filter – require at least 2 interactions per sequence ──
    # Sequences with fewer than 2 steps cannot produce any state transition.
    sequences = {key: seq for key, seq in sequences.items() if len(seq) >= 2}

    filtered_lengths = [len(seq) for seq in sequences.values()]
    print(f"[INFO] Total sequences after filtering (< 2 interactions): {len(sequences)}")
    if filtered_lengths:
        print(f"[DEBUG] Average interactions per sequence (filtered): {np.mean(filtered_lengths):.2f}")
        print(f"[DEBUG] Max interactions per sequence (filtered):     {max(filtered_lengths)}")

    # ── Step 4: Build RL transitions ──
    transitions = []

    print("[INFO] Building RL transitions...")

    for group_key, interactions in sequences.items():
        # Sort by temporal priority; no order_id used
        interactions = sorted(interactions, key=sort_key)

        # ── Per-sequence state reset (no stale state bleeds across sequences) ──
        prev_conf = None
        conf_history = []   # tracks recent confusion values for smoothing

        for i, row in enumerate(interactions):
            raw_confusion = compute_confusion(row)

            # Apply 2-step moving average to smooth noise
            smoothed_confusion = smooth_confusion(conf_history, raw_confusion)
            conf_history.append(smoothed_confusion)   # store smoothed value

            attention = compute_attention(row)
            misconception = classify_misconception(row)
            action = map_action(row)

            # First step in sequence → seed prev_conf and skip (no prior state)
            if prev_conf is None:
                prev_conf = smoothed_confusion
                continue

            reward = compute_reward(prev_conf, smoothed_confusion, attention)

            # Append transition (consecutive steps only, per-sequence)
            transitions.append({
                "confusion":      float(prev_conf),
                "attention":      float(attention),
                "misconception":  misconception,
                "action":         action,
                "reward":         reward,
                "next_confusion": float(smoothed_confusion),
                "next_attention": float(attention),
                "done":           bool(smoothed_confusion < 2.0)
            })

            prev_conf = smoothed_confusion   # advance state correctly

            if len(transitions) >= MAX_TRANSITIONS:
                break   # honour output cap

        if len(transitions) >= MAX_TRANSITIONS:
            break

    print(f"[INFO] Transitions generated: {len(transitions)}")

    # ── Step 5: Save ──
    print("[INFO] Saving JSON...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(transitions, f, indent=2)

    print(f"[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    process()