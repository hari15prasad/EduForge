"""
hybrid_bridge.py — EduForge Brain + Voice Pipeline
====================================================
Stage 1: DQN Multi-Head Agent selects the mathematically optimal pedagogical strategy.
Stage 2: GPT-4 Turbo generates a high-fidelity natural language response constrained
         by that strategy.

Usage:
    python hybrid_bridge.py
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional

import torch
import openai

from src.training.agent import DQN

logger = logging.getLogger(__name__)

# ── Action map ──────────────────────────────────────────────────────────────
ACTION_MAP: dict[int, str] = {
    0: "EXPLAIN",
    1: "WORKED_EXAMPLE",
    2: "ANALOGIZE",
    3: "SOCRATIC_QUESTION",
    4: "CORRECT_FACT",
}

# ── Strategy-specific constraints injected into the system prompt ────────────
STRATEGY_CONSTRAINTS: dict[str, str] = {
    "EXPLAIN": (
        "Provide a clear, structured explanation. Use numbered steps where helpful. "
        "Define any technical terms you introduce."
    ),
    "WORKED_EXAMPLE": (
        "Walk through a concrete, complete worked example step-by-step. "
        "Show intermediate calculations or reasoning, not just the final answer."
    ),
    "ANALOGIZE": (
        "Use one or more real-world metaphors or analogies to make the concept "
        "intuitive. After the analogy, connect it back to the original topic explicitly."
    ),
    "SOCRATIC_QUESTION": (
        "Do NOT give the answer directly. Instead, guide the student with a "
        "probing question that leads them to discover the answer themselves."
    ),
    "CORRECT_FACT": (
        "Identify the specific misconception in the student's statement. "
        "State the correct fact clearly, then briefly explain why the misconception arises."
    ),
}


class EduForgeHybrid:
    """
    Hybrid DQN + LLM tutor pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to the trained `.pt` checkpoint.
    api_key : str, optional
        OpenAI API key. Falls back to OPENAI_API_KEY env var.
    llm_model : str
        Which OpenAI model to use for response generation.
    state_dim : int
        Input dimension of the DQN (must match checkpoint).
    n_actions : int
        Output dimension of the DQN.
    """

    STATE_DIM = 13
    N_ACTIONS  = 5

    def __init__(
        self,
        model_path: str | Path = "latest_model.pt",
        api_key: Optional[str] = None,
        llm_model: str = "gpt-4-turbo",
    ):
        # ── 1. Pedagogical Brain ────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQN(input_dim=self.STATE_DIM, output_dim=self.N_ACTIONS).to(self.device)

        model_path = Path(model_path)
        if model_path.exists():
            self.agent.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            logger.info("Loaded DQN weights from %s", model_path)
        else:
            logger.warning(
                "Model checkpoint '%s' not found — running with random weights. "
                "Train first with dqn_trainer.py.",
                model_path,
            )

        self.agent.eval()

        # ── 2. Generative Voice ─────────────────────────────────────────────
        self.openai_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.llm_model = llm_model
        
        self.use_hf = False
        if not self.openai_key or self.openai_key == "your_openai_api_key_here":
            if self.hf_token and self.hf_token != "your_huggingface_token_here":
                # Fallback to HuggingFace Free Inference API
                from huggingface_hub import InferenceClient
                hf_model = os.getenv("BASE_MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
                self.hf_client = InferenceClient(model=hf_model, token=self.hf_token)
                self.use_hf = True
                logger.info("Using HuggingFace Inference API with model: %s", hf_model)
            else:
                logger.warning("Neither a valid OPENAI_API_KEY nor HF_TOKEN was found. API calls will fail.")
        else:
            self.llm = openai.OpenAI(api_key=self.openai_key)
            logger.info("Using OpenAI API.")

    # ── Public API ──────────────────────────────────────────────────────────

    def select_strategy(
        self,
        state_vector: list[float],
        domain_idx: int = 0,
    ) -> tuple[str, dict[int, float]]:
        """
        Run a forward pass through the DQN and return the chosen strategy name
        plus the raw Q-values for all actions.

        Parameters
        ----------
        state_vector : list[float]
            13-element observation vector.
        domain_idx : int
            Which DQN head to use (0=Factual, 1=Procedural, 2=Conceptual, 3=Transfer).

        Returns
        -------
        strategy : str
            One of EXPLAIN, WORKED_EXAMPLE, ANALOGIZE, SOCRATIC_QUESTION, CORRECT_FACT.
        q_values : dict[int, float]
            Raw Q-values from the selected head.
        """
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent(state_tensor, domain_idx).squeeze(0).tolist()

        action_idx = int(torch.tensor(q_values).argmax().item())
        strategy = ACTION_MAP[action_idx]
        q_dict = {i: round(q, 4) for i, q in enumerate(q_values)}
        return strategy, q_dict

    def generate_response(
        self,
        student_state_vector: list[float],
        student_message: str,
        domain_idx: int = 0,
        history: Optional[list[dict]] = None,
        temperature: float = 0.7,
    ) -> dict:
        """
        Full pipeline: DQN strategy selection → LLM response generation.

        Parameters
        ----------
        student_state_vector : list[float]
            13-element observation (confusion, attention, step, domain_oh×4, history×6…).
        student_message : str
            The student's latest message / question.
        domain_idx : int
            Active domain head index.
        history : list[dict], optional
            Previous turns as [{"role": "user"|"assistant", "content": ...}].
        temperature : float
            LLM sampling temperature.

        Returns
        -------
        dict with keys:
            strategy       : str   — chosen pedagogical tactic
            q_values       : dict  — raw Q-values
            response       : str   — LLM-generated reply
            handling_score : int   — confidence % (max Q-value scaled to 60–99)
        """
        # Stage 1: Pedagogical selection
        strategy, q_values = self.select_strategy(student_state_vector, domain_idx)
        constraint = STRATEGY_CONSTRAINTS[strategy]

        # Stage 2: Build prompt
        system_prompt = (
            "You are EduForge, an expert AI tutor for university-level ISE/CS courses.\n"
            "The Pedagogical Engine has mandated you use exactly this strategy: "
            f"**{strategy}**.\n\n"
            f"Strategy constraint: {constraint}\n\n"
            "Rules:\n"
            "- Keep tone encouraging but academically rigorous.\n"
            "- Be concise (≤ 150 words) unless the strategy requires detail.\n"
            "- Never break character or mention the RL system."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])  # include up to last 3 turns
        messages.append({"role": "user", "content": student_message})

        # Stage 3: LLM call
        response_text = "Error: No valid API key provided."
        try:
            if self.use_hf:
                completion = self.hf_client.chat_completion(
                    messages=messages,
                    max_tokens=300,
                    temperature=temperature,
                )
                response_text = completion.choices[0].message.content.strip()
            elif hasattr(self, 'llm'):
                completion = self.llm.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=300,
                )
                response_text = completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM Generation failed: %s", e)
            response_text = f"[Simulation Fallback] Generation failed due to API error: {e}"

        # Confidence score: scale best Q-value to [60, 99]
        best_q = max(q_values.values())
        worst_q = min(q_values.values())
        q_range = max(best_q - worst_q, 0.01)
        handling_score = int(60 + 39 * ((best_q - worst_q) / (q_range + abs(worst_q) + 0.01)))
        handling_score = max(60, min(99, handling_score))

        return {
            "strategy":       strategy,
            "q_values":       q_values,
            "response":       response_text,
            "handling_score": handling_score,
        }


# ── CLI demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_pt = sys.argv[1] if len(sys.argv) > 1 else "latest_model.pt"

    tutor = EduForgeHybrid(model_path=model_pt)

    # Mock state: High Confusion (8.0), Low Attention (3.0), step 1, Factual domain
    mock_state = [8.0, 3.0, 0.05, 1, 0, 0, 0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0]
    msg = "I don't understand how Inter-Process Communication works in UNIX."

    print(f"\n{'='*60}")
    print(f"Student: {msg}")
    print(f"{'='*60}")

    result = tutor.generate_response(mock_state, msg, domain_idx=0)

    print(f"Pedagogical Brain selected : {result['strategy']}")
    print(f"Q-Values                   : {json.dumps(result['q_values'])}")
    print(f"Handling confidence        : {result['handling_score']}%")
    print(f"\nEduForge Response:\n{result['response']}")
