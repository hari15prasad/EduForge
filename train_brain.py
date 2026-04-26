import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.training.agent import DQN
from tqdm import tqdm

def train():
    # 1. Load Data
    print("Loading 27,000+ training samples...")
    try:
        with open("src/environment/training_samples.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Setup Brain
    # input_dim=13 based on your existing agent architecture
    # output_dim=5 (Explain, Correct, Example, Analogy, Query)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(input_dim=13, output_dim=5).to(device)
    
    # Load existing if available to fine-tune
    if os.path.exists("latest_model.pt"):
        print("Loading existing weights for fine-tuning...")
        model.load_state_dict(torch.load("latest_model.pt", map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Mapping for Misconceptions
    m_map = {"conceptual": 0, "factual": 1, "procedural": 2, "transfer": 3}
    # Mapping for Actions
    a_map = {"explain": 0, "correct_fact": 1, "worked_example": 2, "analogize": 3, "question": 4}

    print(f"Training on {device}...")
    
    # 3. Training Loop (Supervised Learning on RL transitions)
    epochs = 5
    batch_size = 64
    
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle data
        np.random.shuffle(data)
        
        for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = data[i:i+batch_size]
            
            states = []
            targets = []
            
            for item in batch:
                # Construct state vector (matching your environment's 13-dim vector)
                # [confusion, attention, misconception_onehot(4), last_action_onehot(5), step_count, reward_total]
                m_idx = m_map.get(item['misconception'].lower(), 0)
                m_onehot = [0]*4
                m_onehot[m_idx] = 1
                
                # Mocking remaining dims for compatibility with 13-dim input
                state = [item['confusion'], item['attention']] + m_onehot + [0]*5 + [0, 0]
                states.append(state)
                
                # Target: We want the Q-value for the chosen action to match the observed reward
                action_idx = a_map.get(item['action'].lower(), 0)
                
                # We start with the current prediction and only update the chosen action's head
                with torch.no_grad():
                    current_target = model(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze().tolist()
                
                current_target[action_idx] = item['reward']
                targets.append(current_target)

            states_t = torch.FloatTensor(states).to(device)
            targets_t = torch.FloatTensor(targets).to(device)

            optimizer.zero_grad()
            outputs = model(states_t)
            loss = criterion(outputs, targets_t)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / (len(data)/batch_size):.4f}")

    # 4. Save
    torch.save(model.state_dict(), "latest_model.pt")
    print("\nTraining finished! Updated 'latest_model.pt' with new knowledge.")

if __name__ == "__main__":
    train()
