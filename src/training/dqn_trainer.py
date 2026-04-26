"""
dqn_trainer.py — PyTorch Dueling Double DQN with Prioritized Experience Replay for EduForge.
"""

from __future__ import annotations

import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return (
            torch.tensor(np.array(state), dtype=torch.float32),
            torch.tensor(action, dtype=torch.int64).unsqueeze(1),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_state), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1),
            indices,
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        )

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Dueling DQN Network
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        val = self.value(x)
        adv = self.advantage(x)
        # Combine streams
        return val + adv - adv.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DQNTrainer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_capacity: int = 50000
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using Dueling DQN
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Using Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity)
        
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
            
    def optimize_model(self, beta: float = 0.4) -> float:
        """Samples a batch from PER memory, performs a Double Q-learning update, returns loss."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta=beta)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN: Use policy net to select actions, target net to evaluate them
        with torch.no_grad():
            next_action = self.policy_net(next_states).argmax(1).unsqueeze(1)
            max_next_q = self.target_net(next_states).gather(1, next_action)
            
        # Expected Q values
        expected_q_values = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Compute TD errors for PER update
        td_errors = (expected_q_values - q_values).detach().cpu().numpy()
        new_priorities = np.abs(td_errors) + 1e-5
        self.memory.update_priorities(indices, new_priorities.squeeze())
        
        # Compute weighted Huber loss
        loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        
        # Update the target network
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
            
        return loss.item()
        
    def update_target_network(self) -> None:
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
