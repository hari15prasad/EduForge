import sys
import re

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. State Enrichment (get_state_from_obs)
    content = re.sub(
        r'def get_state_from_obs\(obs, last_action_idx, progress_signal, steps_since_improvement\):',
        r'def get_state_from_obs(obs, last_action_idx, prev_last_action_idx, progress_signal, steps_since_improvement):',
        content
    )
    content = re.sub(
        r'return \(c_bin, a_bin, m, progress_signal, last_act, ssi\)',
        r'prev_last_act = prev_last_action_idx if prev_last_action_idx is not None else -1\n    return (c_bin, a_bin, m, progress_signal, last_act, prev_last_act, ssi)',
        content
    )
    
    # 2. Reward Shaping (compute_reward)
    content = re.sub(
        r'def compute_reward\([\s\S]*?done: bool,[\s\S]*?success: bool,[\s\S]*?\) -> float:',
        r'def compute_reward(\n    prev_conf: float,\n    new_conf: float,\n    prev_att: float,\n    new_att: float,\n    done: bool,\n    success: bool,\n    step_number: int,\n    action_idx: int,\n    last_action_idx: int | None,\n    prev_last_action_idx: int | None,\n    steps_since_improvement: int,\n    prev_prev_conf: float = -1.0\n) -> float:',
        content
    )
    
    # Inside compute_reward, replace the old confusion logic
    reward_logic = """    if success:
        return 5.0

    if done and not success:
        return -5.0

    import numpy as np
    # Base: Directional Clamping
    delta = prev_conf - new_conf
    reward = float(np.clip(np.sign(delta) * 1.5, -1.5, 1.5))
    
    # Smoothness & Reversal
    if prev_prev_conf != -1.0:
        if new_conf < prev_conf < prev_prev_conf:
            reward += 1.0  # Smoothness bonus
        elif prev_conf < prev_prev_conf and new_conf > prev_conf:
            reward -= 3.0  # Reversal penalty
            
    # Early Attention Floor
    if new_att < 4.0:
        reward -= 1.5
"""
    content = re.sub(
        r'    if success:[\s\S]*?reward = float\(\(prev_conf - new_conf\) \* 1\.5\)',
        reward_logic,
        content
    )
    
    # 3. Sequence Masking (select_action)
    mask_logic = """    # Action Masking based on Sequence
    for a in range(N_ACTIONS):
        if len(action_history) >= 3 and action_history[-3:] == [a, a, a]:
            mask[a] = -np.inf
            
    if len(action_history) >= 2:
        we_idx = ACTION_TO_IDX["worked_example"]
        if action_history[-2:] == [we_idx, we_idx]:
            mask[we_idx] = -np.inf"""
            
    content = content.replace('    # Safe fallback', mask_logic + '\n\n    # Safe fallback')
    
    # 4. Decoupled Epsilon (train_phase)
    # Replaces the piecewise decay loop in train_phase
    epsilon_decay_logic = """        # Decoupled Domain Epsilon
        decay_rate = 0.999 if misconception in ["procedural", "factual"] else 0.995
        if ep < 500:
            epsilon = 1.0
        else:
            epsilon = max(epsilon_min, epsilon * decay_rate)"""
            
    content = re.sub(
        r'        # Piecewise decay[\s\S]*?epsilon = max\(epsilon_min, epsilon \* 0\.999\)',
        epsilon_decay_logic,
        content
    )
    
    # 5. Domain Confidence Merge (train)
    merge_logic = """    for s in all_states:
        m_val = s[2] # misconception_int
        m_str = ""
        for name, val in MISCONCEPTION_MAP.items():
            if val == m_val:
                m_str = name
                break
                
        vals = []
        weights = []
        for i, misc in enumerate(misconceptions):
            if s in q_tables[i]:
                vals.append(q_tables[i][s])
                weights.append(0.9 if misc == m_str else 0.1)
                
        if vals:
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            merged_q[s] = np.average(vals, axis=0, weights=weights)"""
            
    content = re.sub(
        r'    for s in all_states:[\s\S]*?merged_q\[s\] = 0\.7 \* np\.max\(vals, axis=0\) \+ 0\.3 \* np\.mean\(vals, axis=0\)',
        merge_logic,
        content
    )
    
    # 6. Fix references to get_state_from_obs and compute_reward
    content = re.sub(
        r'state = get_state_from_obs\(obs, last_action_idx, progress_signal, steps_since_improvement\)',
        r'state = get_state_from_obs(obs, last_action_idx, prev_last_action_idx, progress_signal, steps_since_improvement)',
        content
    )
    
    content = re.sub(
        r'next_state = get_state_from_obs\(obs, action_idx, progress_signal, steps_since_improvement\)',
        r'next_state = get_state_from_obs(obs, action_idx, last_action_idx, progress_signal, steps_since_improvement)',
        content
    )
    
    # For compute_reward, we need to pass prev_prev_conf. We must track it in the main loops.
    # In train_phase:
    content = re.sub(
        r'            prev_conf = obs.confusion\n            prev_att  = obs.attention',
        r'            prev_prev_conf = prev_conf if step > 1 else -1.0\n            prev_conf = obs.confusion\n            prev_att  = obs.attention',
        content
    )
    
    content = re.sub(
        r'                steps_since_improvement=steps_since_improvement,\n            \)',
        r'                steps_since_improvement=steps_since_improvement,\n                prev_prev_conf=prev_prev_conf\n            )',
        content
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
if __name__ == "__main__":
    patch_file("scripts/qlearning_pipeline.py")
    print("Patch applied.")
