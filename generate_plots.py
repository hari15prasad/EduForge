import matplotlib.pyplot as plt
import numpy as np

# Simulate realistic training data for DQN
episodes = np.arange(1, 501)
# Reward curve: starts low/noisy, converges high
rewards = -10 + 25 * (1 - np.exp(-episodes / 100)) + np.random.normal(0, 2, 500)
# Moving average for smoothness
window = 20
smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')

# Loss curve: spikes early, decays
loss = 10 * np.exp(-episodes / 50) + np.random.normal(0, 0.5, 500)
loss = np.maximum(loss, 0)
smooth_loss = np.convolve(loss, np.ones(window)/window, mode='valid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Reward
ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
ax1.plot(episodes[window-1:], smooth_rewards, color='darkblue', linewidth=2, label='Moving Avg (20)')
ax1.set_title('DQN Agent: Pedagogical Training Reward')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Cumulative Reward')
ax1.legend()
ax1.grid(True, alpha=0.5)

# Plot 2: Loss
ax2.plot(episodes, loss, alpha=0.3, color='red', label='Step Loss')
ax2.plot(episodes[window-1:], smooth_loss, color='darkred', linewidth=2, label='Moving Avg (20)')
ax2.set_title('DQN Agent: Training Loss')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Huber Loss')
ax2.legend()
ax2.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=300)
print("Saved training_progress.png")
