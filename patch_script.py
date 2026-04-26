import sys
from collections import defaultdict

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Add attention_history to train_phase
    old_train_init = """        action_history = []
        consecutive_conf_increases = 0
        consecutive_att_drops = 0
        locked_phase_timer = 0
        last_reward = 0.0"""
    new_train_init = """        action_history = []
        attention_history = []
        consecutive_conf_increases = 0
        consecutive_att_drops = 0
        locked_phase_timer = 0
        last_reward = 0.0"""
    content = content.replace(old_train_init, new_train_init)
    
    # 2. Add attention_history update in train_phase loop
    old_train_loop = """            # Update tracking variables
            action_history.append(action_idx)
            if len(action_history) > 3:
                action_history.pop(0)

            if obs.confusion > prev_conf:
                consecutive_conf_increases += 1
            else:
                consecutive_conf_increases = 0

            if obs.attention < prev_att:
                consecutive_att_drops += 1
            else:
                consecutive_att_drops = 0"""
    new_train_loop = """            # Update tracking variables
            action_history.append(action_idx)
            if len(action_history) > 4:
                action_history.pop(0)
                
            attention_history.append(obs.attention)
            if len(attention_history) > 4:
                attention_history.pop(0)

            if obs.confusion >= prev_conf:
                consecutive_conf_increases += 1
            else:
                consecutive_conf_increases = 0

            if obs.attention < prev_att:
                consecutive_att_drops += 1
            else:
                consecutive_att_drops = 0"""
    content = content.replace(old_train_loop, new_train_loop)

    # 3. Add attention_history to evaluate
    old_eval_init = """        action_history = []
        consecutive_conf_increases = 0
        consecutive_att_drops = 0
        locked_phase_timer = 0
        last_reward = 0.0"""
    new_eval_init = """        action_history = []
        attention_history = []
        consecutive_conf_increases = 0
        consecutive_att_drops = 0
        locked_phase_timer = 0
        last_reward = 0.0"""
    content = content.replace(old_eval_init, new_eval_init)
    
    # 4. Add attention_history update in evaluate loop
    old_eval_loop = """            # Update tracking variables
            action_history.append(action_idx)
            if len(action_history) > 3:
                action_history.pop(0)

            if obs.confusion > prev_conf:
                consecutive_conf_increases += 1
            else:
                consecutive_conf_increases = 0

            if obs.attention < prev_att:
                consecutive_att_drops += 1
            else:
                consecutive_att_drops = 0"""
    new_eval_loop = """            # Update tracking variables
            action_history.append(action_idx)
            if len(action_history) > 4:
                action_history.pop(0)
                
            attention_history.append(obs.attention)
            if len(attention_history) > 4:
                attention_history.pop(0)

            if obs.confusion >= prev_conf:
                consecutive_conf_increases += 1
            else:
                consecutive_conf_increases = 0

            if obs.attention < prev_att:
                consecutive_att_drops += 1
            else:
                consecutive_att_drops = 0"""
    content = content.replace(old_eval_loop, new_eval_loop)

    # 5. Replace select_action function
    # Because of how complex it is, we will locate its start and end
    start_str = "def select_action("
    end_str = "# 5. Dataset Loader"
    
    start_idx = content.find(start_str)
    end_idx = content.find(end_str)
    
    new_select_action = '''def select_action(
    q_table: defaultdict,
    state: tuple,
    epsilon: float,
    rng: random.Random,
    obs_attention: float,
    obs_confusion: float,
    consecutive_conf_increases: int,
    consecutive_att_drops: int,
    locked_phase_timer: int,
    action_history: list[int],
    attention_history: list[float],
    last_reward: float
) -> int:
    """Strict hierarchical action selection for Guided RL following policy constraints."""
    c, a, m, p, la, ps, ssi = state
    
    misc_str = "none"
    for k, v in MISCONCEPTION_MAP.items():
        if v == m:
            misc_str = k
            break

    allowed = list(ACTIONS.keys())

    # Helper function to mask out actions
    def mask_action(action_name: str):
        idx = ACTION_TO_IDX.get(action_name, -1)
        if idx in allowed:
            allowed.remove(idx)

    def mask_except(action_names: list[str]):
        idxs = [ACTION_TO_IDX[name] for name in action_names]
        allowed[:] = [act for act in allowed if act in idxs]
        
    def force_action(action_name: str):
        idx = ACTION_TO_IDX[action_name]
        allowed[:] = [idx]

    # --- Step 1: Calculate Action Diversity & History ---
    action_counts = defaultdict(int)
    for act in action_history:
        action_counts[act] += 1
        
    worked_example_idx = ACTION_TO_IDX["worked_example"]
    we_used_consecutively = len(action_history) >= 2 and action_history[-1] == worked_example_idx and action_history[-2] == worked_example_idx

    # --- Step 2: "Engagement Floor Rule" (from feedback) ---
    att_below_3_count = sum(1 for att in attention_history if att < 3.0)
    engagement_floor_broken = (consecutive_att_drops >= 3) or (att_below_3_count >= 2)
    if engagement_floor_broken:
        # Switch away from dominant action
        if action_history:
            dominant_action = max(action_counts, key=action_counts.get)
            if dominant_action in allowed:
                allowed.remove(dominant_action)
            # To ensure switch, we force exploration if it was worked_example
            if dominant_action == worked_example_idx:
                mask_except(["explain", "question", "analogize"])

    # --- Step 3: Hard Constraint 1 - Attention Safety ---
    # If attention < 2.5: NO worked_example, prefer explain/question
    if obs_attention < 2.5:
        mask_action("worked_example")
        if not engagement_floor_broken:
            mask_except(["explain", "question"])

    # If attention < 3.0: immediate strategy shift required
    if obs_attention < 3.0:
        if action_history and action_history[-1] in allowed:
            allowed.remove(action_history[-1])

    # If attention is falling for 2 consecutive steps -> switch strategy class entirely
    if consecutive_att_drops >= 2:
        if action_history and action_history[-1] in allowed:
            allowed.remove(action_history[-1])

    # --- Step 4: Hard Constraint 2 - Action Diversity ---
    # No action type may exceed 60% frequency (in window of 4, max 2 occurrences)
    for act, count in action_counts.items():
        if count >= 2:
            if act in allowed:
                allowed.remove(act)
                
    # No repeated worked_example more than 2 consecutive times. 
    if we_used_consecutively:
        mask_action("worked_example")
        
    # If allowed is empty because of diversity, force exploration action
    if not allowed:
        allowed = list(ACTIONS.keys())
        mask_except(["question", "analogize"])
        if not allowed: # If those were masked, fallback
            allowed = [ACTION_TO_IDX["question"], ACTION_TO_IDX["analogize"]]

    # --- Step 5: Hard Constraint 3 - Confusion Reduction Rule ---
    # If confusion is NOT decreasing for 2 consecutive steps
    if consecutive_conf_increases >= 2:
        if misc_str == "conceptual":
            mask_except(["analogize", "question"])
        elif misc_str == "factual":
            force_action("correct_fact")
        elif misc_str == "procedural":
            if not we_used_consecutively and ACTION_TO_IDX["worked_example"] in allowed:
                force_action("worked_example")
            else:
                mask_except(["explain", "question"])
        elif misc_str == "transfer":
            mask_except(["explain", "analogize"])

    # --- Step 6: Hard Constraint 4 - Factual Misconception Rule ---
    if misc_str == "factual":
        # Primary actions: correct_fact (must lead), explain (support)
        # worked_example only if confusion < 6
        if obs_confusion >= 6.0:
            mask_action("worked_example")
        if len(action_history) == 0:
            force_action("correct_fact") # Must lead

    # --- Step 7: Domain Selection Policies (Priors if not overridden) ---
    if len(allowed) > 1 and not (consecutive_conf_increases >= 2 or engagement_floor_broken or obs_attention < 3.0):
        if misc_str == "conceptual":
            mask_except(["explain", "analogize", "question"])
        elif misc_str == "procedural":
            mask_except(["worked_example", "explain", "question"])
        elif misc_str == "factual":
            mask_except(["correct_fact", "explain", "worked_example"])
        elif misc_str == "transfer":
            mask_except(["explain", "analogize", "worked_example", "question"])

    if not allowed:
        allowed = list(ACTIONS.keys()) # Safe fallback

    if rng.random() < epsilon:
        return rng.choice(allowed)
        
    # Argmax over ALLOWED actions only
    q_vals = [q_table[state][act] if act in allowed else -1e9 for act in range(N_ACTIONS)]
    return int(np.argmax(q_vals))


# ---------------------------------------------------------------------------
'''
    
    content = content[:start_idx] + new_select_action + content[end_idx:]
    
    # 6. We must also fix the function call signatures for select_action
    old_call_train = """            action_idx = select_action(
                q_table, state, current_epsilon, rng,
                obs.attention, obs.confusion,
                consecutive_conf_increases, consecutive_att_drops,
                locked_phase_timer, action_history, last_reward
            )"""
    new_call_train = """            action_idx = select_action(
                q_table, state, current_epsilon, rng,
                obs.attention, obs.confusion,
                consecutive_conf_increases, consecutive_att_drops,
                locked_phase_timer, action_history, attention_history, last_reward
            )"""
    content = content.replace(old_call_train, new_call_train)
    
    old_call_eval = """            action_idx = select_action(
                q_table, state, 0.0, rng, # epsilon=0 for evaluation
                obs.attention, obs.confusion,
                consecutive_conf_increases, consecutive_att_drops,
                locked_phase_timer, action_history, last_reward
            )"""
    new_call_eval = """            action_idx = select_action(
                q_table, state, 0.0, rng, # epsilon=0 for evaluation
                obs.attention, obs.confusion,
                consecutive_conf_increases, consecutive_att_drops,
                locked_phase_timer, action_history, attention_history, last_reward
            )"""
    content = content.replace(old_call_eval, new_call_eval)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
patch_file("scripts/qlearning_pipeline.py")
