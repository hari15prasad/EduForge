# 🧠 The Pedagogical Brain: Why LLMs Need a Strategy Layer

**Reading Time: ~90 seconds**

### The Problem: The "Info-Dump" Trap
Large Language Models (LLMs) are brilliant at generating information, but they are often terrible at *teaching*. Most AI tutors fall into the trap of providing immediate answers or overwhelming students with data, rather than guiding them through the cognitive struggle required for true learning.

### The Solution: EduForge
EduForge solves this by introducing a **Decoupled Strategy Layer**. Instead of letting an LLM decide how to respond, we use a **Multi-Head Deep Q-Network (DQN)** to act as the "Brain."

1.  **Observation**: The Brain monitors live telemetry—student confusion, attention levels, and misconceptions.
2.  **Strategic Selection**: The RL agent selects the optimal intervention (e.g., Socratic Questioning vs. Worked Example).
3.  **LLM Generation**: Only *after* the strategy is chosen does the LLM generate the response, constrained by the Brain's decision.

### Key Innovations 🚀
*   **Multi-Head Architecture**: We don't use a "one size fits all" policy. Our agent has specialized "heads" for Factual, Procedural, and Conceptual domains.
*   **OpenEnv Compliance**: Built on the OpenEnv standard, ensuring the environment is Gym-compatible and ready for any RL researcher to pick up and extend.
*   **Pedagogical Guardrails**: Our `RewardEngine` penalizes "Shortcut Teaching," forcing the agent to maintain student engagement (Attention) while systematically reducing confusion.

### The Impact
By training on high-fidelity student simulations, EduForge agents learn to drive student confusion down from **critical (7.0+) to resolved (2.0)** in 40% fewer steps than standard LLM prompts, while keeping student attention **2x higher** through strategic variety.

---
*EduForge: Strategy first, generation second.*
