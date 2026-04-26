import os
import csv
import json
from dotenv import load_dotenv
from groq import Groq
from eduforge_runtime import EduForgeRuntime

# Load environment variables
load_dotenv()

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize EduForge Tutor (Ensure HF_TOKEN is in your .env)
tutor = EduForgeRuntime(
    rl_checkpoint=os.getenv("MODEL_PATH", "latest_model.pt"),
    hf_token=os.getenv("HF_TOKEN")
)

# A standard starting state vector (13 dims: 4 domain one-hot, 5 history, 4 metrics)
# Representing a moderately confused student.
MOCK_STATE_VECTOR = [1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0,  0.5, 5.0, 5.0, 3.0]

def evaluate_with_gpt(question, tutor_response):
    """Uses GPT-4o to grade the response based on the rubric."""
    
    prompt = f"""You are a Senior Computer Science Professor and Pedagogical Expert. Your task is to evaluate the response of a student-built AI Tutor named "EduForge-v33" based on a strict evaluation rubric.

Evaluation Rubric (Score 1-5 for each):
Technical Accuracy: Is the computer science concept explained correctly? Are there any syntax errors in code or conceptual hallucinations?
Pedagogical Style: Does the tutor use analogies or real-world examples? Is the tone encouraging and helpful rather than just robotic?
Guidance (Scaffolding): Does the tutor avoid "giving the answer away" immediately? Does it guide the student toward the solution?
Check for Understanding: Does the response end with a relevant follow-up question to test the student's knowledge?

Input Data:
Student Question: {question}
EduForge-v33 Response: {tutor_response}

You MUST return your response as a valid JSON object matching the exact structure below:
{{
  "scores": {{
    "accuracy": 0,
    "style": 0,
    "scaffolding": 0,
    "cfu": 0
  }},
  "feedback": "Detailed explanation of why these scores were given.",
  "suggested_improvement": "One specific sentence the model could have added to be better."
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling Groq Judge: {e}")
        return None

def main():
    test_questions = [
        "I don't understand how recursion works. It just looks like an infinite loop.",
        "How does a binary search work? I keep getting index out of bounds.",
        "What is the difference between an Array and a Linked List?",
        "Why is QuickSort considered faster than Bubble Sort if both use loops?",
        "I'm confused by Big O notation. Does O(n) mean it takes n seconds?"
    ]

    results_log = "evaluation_results.csv"
    
    with open(results_log, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Chosen Strategy", "Tutor Response", "Accuracy", "Style", "Scaffolding", "CFU", "Feedback", "Suggested Improvement"])

        for idx, question in enumerate(test_questions, 1):
            print(f"\n[{idx}/{len(test_questions)}] Evaluating Question: {question}")
            
            # 1. Get answer from the fine-tuned model (via the RL Bridge)
            print("  Generating Tutor Response...")
            answer, chosen_strategy, _ = tutor.generate_tutor_response(question, MOCK_STATE_VECTOR)
            print(f"  > Strategy Chosen: {chosen_strategy}")
            
            if "[API ERROR]" in answer:
                print(f"  > Failed to generate response: {answer}")
                continue

            # 2. Send answer + Rubric Prompt to Groq
            print("  Calling Judge LLM (Groq: Llama 3 70B)...")
            evaluation = evaluate_with_gpt(question, answer)
            
            if evaluation and "scores" in evaluation:
                scores = evaluation["scores"]
                writer.writerow([
                    question, 
                    chosen_strategy,
                    answer, 
                    scores.get("accuracy"), 
                    scores.get("style"), 
                    scores.get("scaffolding"), 
                    scores.get("cfu"), 
                    evaluation.get("feedback"), 
                    evaluation.get("suggested_improvement")
                ])
                print("  > Evaluation saved.")
            else:
                print("  > Evaluation failed for this question.")

    print(f"\nEvaluation complete! Results saved to {results_log}")

if __name__ == "__main__":
    main()
