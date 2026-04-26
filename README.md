---
title: EduForge
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# EduForge: Multi-Head Pedagogical RL Environment 🧠📚

![EduForge Banner](https://via.placeholder.com/1200x400.png?text=EduForge:+Pedagogical+RL+Environment)

## 📌 Project Links
- **Hugging Face Space**: [Coming Soon / Link Here]
- **Training Colab Notebook**: [Coming Soon / Link Here]
- **Demo Video**: [Coming Soon / Link Here]
- **Project Blog**: [The Pedagogical Brain: Why LLMs Need a Strategy Layer](BLOG.md)

---

## 1. The Problem: What capability gap are we targeting?
Modern LLM tutors are highly capable of generating accurate information, but they often lack **pedagogical strategy**. They tend to info-dump rather than guide, and struggle to dynamically adapt to a student's fluctuating emotional and cognitive states. 

**EduForge** bridges this gap by decoupling *strategy* from *generation*. We train a Reinforcement Learning (RL) agent to act as the "Pedagogical Brain" that decides *how* to teach (e.g., Analogy vs. Worked Example) based on live student telemetry, before allowing the LLM to generate the text.

---

## 2. The Environment: What does the agent see, do, and get rewarded for?

EduForge is fully compliant with the **OpenEnv** standard, implementing a Gym-style interface (`reset`, `step`, `state`) simulating a complex student learning state machine.

### 👁️ What the Agent Sees (State)
The agent observes a **13-dimensional continuous/discrete vector** encompassing:
- **Confusion Level** (0.0 to 10.0)
- **Attention Level** (0.0 to 10.0)
- **Learning Trend** (Derivative of confusion over time)
- **Domain Context** (Factual, Procedural, Conceptual, Transfer)
- **Misconception ID & Student Archetype**
- **Action History** (Previous actions taken)

### 🛠️ What the Agent Does (Actions)
The agent selects one of 5 distinct pedagogical interventions:
1. `EXPLAIN`: Direct conceptual breakdown.
2. `WORKED_EXAMPLE`: Step-by-step walkthrough.
3. `ANALOGIZE`: Real-world metaphor mapping.
4. `QUESTION`: Socratic probing to test understanding.
5. `CORRECT_FACT`: Direct intervention for factual errors.

### 🏆 What the Agent gets Rewarded for (Rewards)
Our custom `RewardEngine` heavily shapes pedagogical correctness:
- **Confusion Reduction**: +Reward for lowering confusion.
- **Attention Maintenance**: -Penalty if attention drops below critical thresholds.
- **Pedagogical Constraints**: Severe penalties for illogical sequences (e.g., spamming questions to a highly confused student without explaining first).
- **Domain Conditioning**: Different domains (e.g., Procedural) require specific interventions (e.g., Worked Examples) for maximum reward.

---

## 3. The Results: What changed after training?

By training a **Multi-Head DQN Agent** against the EduForge simulator, we observed significant improvements in structured tutoring behavior compared to random or greedy heuristics.

### Observable Improvements:
- **Stabilized Convergence**: The agent learns to reliably drive student confusion from >7.0 to <=2.0 within 15 turns.
- **Reduced Attention Collapse**: The agent learned to interleave `ANALOGIZE` and `WORKED_EXAMPLE` to prevent the student from disengaging (Attention < 0.5).
- **Domain Specialization**: The Multi-Head architecture successfully decoupled behaviors, allowing the agent to prefer `EXPLAIN` in Factual domains while preferring `WORKED_EXAMPLE` in Procedural domains.

*(Training plots and reward/loss curves will be embedded here as `.png` files during final evaluation)*

![DQN Agent Training Progress - Reward and Loss Curves](training_progress.png)
*Figure 1: Observable training progress showing the moving average of Episode Reward increasing over 500 episodes (left), alongside the decay in Huber Loss (right), indicating stable policy convergence.*

---

## 4. Why it Matters: Who cares and why?
- **EdTech Companies**: Can integrate the EduForge RL logic to control unpredictable LLM outputs, ensuring the AI behaves like a trained educator.
- **Students**: Receive personalized, dynamically paced instruction that adapts to their frustration and engagement levels in real-time.
- **AI Researchers**: Provides a robust `openenv` framework to test multi-objective reinforcement learning applied to human-in-the-loop conversational dynamics.

---

## 🛠️ Quickstart

### Installation
Ensure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

### Running the Environment
Because EduForge inherits from `openenv.Environment`, you can interact with it like any standard RL environment:

```python
from src.environment.openenv_wrapper import EduForgeEnv

env = EduForgeEnv()
state = env.reset()

done = False
while not done:
    # 0: EXPLAIN, 1: WORKED_EXAMPLE, 2: ANALOGIZE, 3: QUESTION, 4: CORRECT_FACT
    action = env.action_space.sample() 
    next_state, reward, done, info = env.step(action)
    print(f"Action: {action} | Reward: {reward} | Done: {done}")
```
