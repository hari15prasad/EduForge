import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any
from collections import deque

# ---------------------------------------------------------------------------
# DQN Architecture
# Standard dense architecture for the EduForge pedagogical framework.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Multi-Head DQN Architecture (Task 6)
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Domain-Specific Heads: 0=procedural, 1=conceptual, 2=factual, 3=transfer
        self.heads = nn.ModuleList([
            nn.Linear(128, output_dim) for _ in range(4)
        ])

    def forward(self, x: torch.Tensor, domain_idx: Any = None) -> torch.Tensor:
        trunk_out = self.trunk(x)
        
        if domain_idx is None:
            # Default to first head if not specified (for safety/eval)
            return self.heads[0](trunk_out)
            
        if isinstance(domain_idx, (int, np.integer)):
            return self.heads[domain_idx](trunk_out)
            
        # Batch processing
        if torch.is_tensor(domain_idx):
            # Compute all heads
            all_q = torch.stack([head(trunk_out) for head in self.heads], dim=1) # (batch, 4, output_dim)
            # domain_idx is (batch, 1)
            # We need to select the head for each sample in the batch
            # Reshape domain_idx for gathering: (batch, 1, output_dim)
            idx = domain_idx.view(-1, 1, 1).expand(-1, 1, all_q.size(-1))
            selected_q = all_q.gather(1, idx).squeeze(1)
            return selected_q
            
        return self.heads[0](trunk_out)

# ---------------------------------------------------------------------------
# Double DQN Agent
# Decouples selection from evaluation to stabilize Q-value estimation.
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99, buffer_size=20000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks: Policy (Live) and Target (Lagging)
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def select_action(self, state, epsilon, domain_idx, mask=None):
        if random.random() < epsilon:
            # Exploration
            available_actions = [i for i, m in enumerate(mask) if m == 0] if mask is not None else range(self.output_dim)
            return random.choice(available_actions)
        
        # Exploitation
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t, domain_idx)
            if mask is not None:
                # Apply mask by heavily penalizing forbidden actions
                # mask in dqn_pipeline is 0.0=allowed, -inf=blocked
                mask_t = torch.FloatTensor(mask).to(self.device)
                q_values = q_values + mask_t
            return q_values.max(1)[1].item()

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        # Sample transitions
        states, actions, rewards, next_states, dones, next_masks, indices, weights, domain_idxs = self.memory.sample(batch_size)

        # To Tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        next_masks_t = torch.FloatTensor(next_masks).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        domain_idxs_t = torch.LongTensor(domain_idxs).unsqueeze(1).to(self.device)

        # 1. Get current Q-values from Policy Net (Domain-specific head)
        current_q = self.policy_net(states_t, domain_idxs_t).gather(1, actions_t)

        # 2. DOUBLE DQN LOGIC:
        # Step A: Policy Net selects the best action for the next state
        with torch.no_grad():
            # Apply masks to next state predictions so we don't bootstrap from invalid actions
            next_q_policy = self.policy_net(next_states_t, domain_idxs_t)
            next_q_policy = next_q_policy + next_masks_t
            best_actions = next_q_policy.max(1)[1].unsqueeze(1)

            # Step B: Target Net evaluates the value of that specific action
            next_q_target = self.target_net(next_states_t, domain_idxs_t).gather(1, best_actions).squeeze(1)
            
            # Step C: Reward Clipping (+10/-10)
            clipped_rewards = torch.clamp(rewards_t, -10.0, 10.0)
            target_q = clipped_rewards + (self.gamma * next_q_target * (1 - dones_t))

        # 3. Compute Loss
        # Use Importance Sampling weights if available
        td_errors = torch.abs(current_q.squeeze() - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        loss = (weights_t * F.smooth_l1_loss(current_q.squeeze(), target_q, reduction='none')).mean()

        # 4. Backprop with Gradient Clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------------------------------------------------------------------
# Prioritized Replay Buffer (Task 4)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_mask, domain_idx):
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done, next_mask, domain_idx))
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done, next_mask, domain_idx)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.memory) == len(self.priorities):
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = zip(*samples)
        states, actions, rewards, next_states, dones, next_masks, domain_idxs = batch

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones),
            np.stack(next_masks),
            indices,
            weights,
            np.array(domain_idxs)
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)