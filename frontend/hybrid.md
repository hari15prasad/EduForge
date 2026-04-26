# EduForge Hybrid Implementation Guide: Brain + Voice

This guide outlines the steps to integrate the **DQN Multi-Head Agent** (Strategic Logic) with a **Generative LLM** (Natural Language Output).

---

## 1. Architecture Overview

The system operates in a three-stage pipeline:
1. **State Encoding**: Convert current student metrics (Confusion, Attention, History) into a 13-feature vector.
2. **Pedagogical Selection**: The DQN Agent selects the mathematically optimal `TutorAction`.
3. **Response Generation**: An LLM receives the chosen action and the student's message to generate a high-fidelity response.



---

## 2. Prerequisites
- Python 3.9+
- PyTorch
- OpenAI API Key (or access to Gemini/Claude API)
- Your trained `latest_model.pt`

---

## 3. Implementation Code: `hybrid_bridge.py`

Create this file in your root directory. It wraps your RL agent and connects it to the generative API.

```python
import torch
import openai
from src.training.agent import DQN  # Based on your fixed import path

class EduForgeHybrid:
    def __init__(self, model_path, api_key):
        # 1. Load the Pedagogical Brain (DQN)
        self.state_dim = 13
        self.n_actions = 5
        self.agent = DQN(input_dim=self.state_dim, output_dim=self.n_actions)
        self.agent.load_state_dict(torch.load(model_path))
        self.agent.eval()

        # 2. Initialize the Voice (LLM)
        self.client = openai.OpenAI(api_key=api_key)
        self.action_map = {
            0: "EXPLAIN",
            1: "WORKED_EXAMPLE",
            2: "ANALOGIZE",
            3: "QUESTION",
            4: "CORRECT_FACT"
        }

    def generate_response(self, student_state_vector, student_message):
        """
        Processes student input through the RL-LLM pipeline.
        """
        # Step 1: RL Strategy Selection
        state_tensor = torch.FloatTensor(student_state_vector).unsqueeze(0)
        with torch.no_grad():
            action_idx = torch.argmax(self.agent(state_tensor)).item()
        
        chosen_strategy = self.action_map[action_idx]

        # Step 2: LLM Prompt Construction (The "System Override")
        system_prompt = f"""
        You are the EduForge AI Tutor, an expert ISE Department assistant.
        The Pedagogical Engine has mandated that you MUST use the strategy: {chosen_strategy}.
        
        Constraints:
        - If the strategy is WORKED_EXAMPLE, provide a step-by-step solution.
        - If the strategy is ANALOGIZE, use a real-world metaphor.
        - Keep the tone encouraging but academically rigorous.
        """

        # Step 3: API Call
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": student_message}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content, chosen_strategy

# --- Usage Example ---
if __name__ == "__main__":
    tutor = EduForgeHybrid("latest_model.pt", "your-api-key-here")
    
    # Mock state: High Confusion (8.0), Low Attention (3.0)
    mock_state = [8.0, 3.0, 0, 0, 1, 0, 0, 0.5, 0.2, 1, 0, 0, 0] 
    
    msg = "I don't understand how Inter-Process Communication works in UNIX."
    reply, strategy = tutor.generate_response(mock_state, msg)
    
    print(f"Pedagogical Brain selected: {strategy}")
    print(f"EduForge Response: {reply}")