import sys
import os
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.training.agent import ReplayBuffer

def test_per():
    buffer = ReplayBuffer(capacity=100)
    state = np.zeros(13)
    next_state = np.zeros(13)
    mask = np.zeros(5)
    
    # Push some transitions
    for i in range(10):
        buffer.push(state, 0, 1.0, next_state, False, mask)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Sample
    states, actions, rewards, next_states, dones, next_masks, indices, weights = buffer.sample(batch_size=4)
    print(f"Sampled weights: {weights}")
    print(f"Sampled indices: {indices}")
    
    # Update priorities
    buffer.update_priorities(indices, [10.0, 10.0, 10.0, 10.0])
    print("Updated priorities.")
    
    # Sample again - should favor the high priority ones
    states, actions, rewards, next_states, dones, next_masks, indices, weights = buffer.sample(batch_size=4)
    print(f"New sampled indices (should be from updated): {indices}")
    print(f"New sampled weights: {weights}")

if __name__ == "__main__":
    test_per()
