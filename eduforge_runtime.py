import torch
from huggingface_hub import InferenceClient
from groq import Groq
import os
from src.training.agent import DQN

class EduForgeRuntime:
    """
    The Master Controller for EduForge.
    Bridges the Strategic Brain (DQN) with the Generative Voice (LLM).
    """
    def __init__(self, rl_checkpoint="latest_model.pt", hf_endpoint=None, hf_token=None, groq_api_key=None):
        # 1. Initialize the RL Brain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.brain = DQN(input_dim=13, output_dim=5).to(self.device)
        
        # Load weights if available
        if os.path.exists(rl_checkpoint):
            try:
                self.brain.load_state_dict(torch.load(rl_checkpoint, map_location=self.device))
                self.brain.eval()
            except Exception as e:
                print(f"Warning: Could not load checkpoint {rl_checkpoint}: {e}")
        
        # 2. LLM Providers
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token
        
        self.groq_client = None
        if groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_api_key)
            except ImportError:
                print("Warning: 'groq' package not found.")

        # 3. Memory (Chat History)
        self.histories: dict[str, list[dict]] = {}
        self.MAX_HISTORY = 10

    def get_strategic_action(self, state_vector: list[float]) -> tuple[str, list[float]]:
        """Query the DQN for the optimal pedagogical strategy."""
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.brain(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            q_values_list = q_values.squeeze().tolist()
        
        strategies = ["EXPLAIN", "CORRECT_FACT", "WORKED_EXAMPLE", "ANALOGIZE", "QUESTION"]
        return strategies[action_idx], q_values_list

    def generate_tutor_response(self, user_message, state_vector, forced_strategy=None, session_id="default"):
        """Main hybrid pipeline: Strategic selection -> Generative response."""
        
        # Step 1: Brain (Strategy)
        if forced_strategy:
            chosen_strategy = forced_strategy.upper()
            _, q_values = self.get_strategic_action(state_vector)
        else:
            chosen_strategy, q_values = self.get_strategic_action(state_vector)
        
        # Initialize or retrieve history
        if session_id not in self.histories:
            self.histories[session_id] = []
        
        history = self.histories[session_id]
        
        system_prompt = (
            "You are a highly efficient, focused Academic Tutor. "
            "You must execute the requested pedagogical strategy directly and concisely. "
            "CRITICAL RULES:\n"
            "1. NO introductory pleasantries or filler (e.g., skip 'Hello!', 'Sure!', 'I can help with that').\n"
            "2. DO NOT narrate your strategy. Do NOT say 'Here is an analogy' or 'Let me explain'. Just provide the analogy or explanation directly.\n"
            "3. Stay strictly within the context of the user's specific problem or question.\n"
            "4. Be extremely concise. Maximum 2-3 sentences per response.\n"
            f"\nCURRENT STRATEGY TO EXECUTE: {chosen_strategy}\n"
            "- EXPLAIN: Deliver a direct, clear definition or conceptual breakdown.\n"
            "- WORKED_EXAMPLE: Walk through a concrete step-by-step example.\n"
            "- ANALOGIZE: Provide a brief, relatable real-world analogy to clarify the topic.\n"
            "- QUESTION: Ask a single targeted Socratic question to test understanding.\n"
            "- CORRECT_FACT: Directly point out the factual error and provide the correct fact."
        )

        # Build messages with history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-self.MAX_HISTORY:])
        messages.append({"role": "user", "content": user_message})

        # Step 2: Try Groq First
        reply = None
        if self.groq_client:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=250,
                )
                reply = chat_completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"Groq API Error: {e}. Falling back to HF...")

        # Step 3: Fallback to Hugging Face
        if reply is None:
            if not self.hf_token:
                reply = f"[MOCK MODE] Strategy: {chosen_strategy}. (Missing API Keys)"
            else:
                try:
                    client = InferenceClient(
                        model=self.hf_endpoint if self.hf_endpoint else "meta-llama/Meta-Llama-3-8B-Instruct",
                        token=self.hf_token
                    )
                    model_id = None if self.hf_endpoint else "meta-llama/Meta-Llama-3-8B-Instruct"
                    response = client.chat_completion(
                        model=model_id,
                        messages=messages,
                        max_tokens=250,
                        temperature=0.7
                    )
                    reply = response.choices[0].message.content.strip()
                except Exception as e:
                    reply = f"[API ERROR] {str(e)}"

        # Update History
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        
        return reply, chosen_strategy, q_values

# Example Integration for your app.py:
# tutor = EduForgeRuntime(hf_endpoint="YOUR_URL", hf_token="YOUR_TOKEN")
# reply, strategy = tutor.generate_tutor_response("How do I code a pipe?", state_vector)