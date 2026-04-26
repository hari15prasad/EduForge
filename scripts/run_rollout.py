"""
run_rollout.py — Load a trained EduForge model and run a 12-turn demo episode.
Prints per-turn state transitions: Confusion | Strategy | Reward.

Usage
-----
# With a trained model:
python scripts/run_rollout.py --model_dir outputs/eduforge_grpo/final_model

# Without a model (rule-based fallback for demo):
python scripts/run_rollout.py --no_model
"""

from __future__ import annotations

import argparse
import re
import sys
import os

# Ensure project root is on path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.environment.openenv_wrapper import EduForgeEnv
from src.environment.student_fsm import MisconceptionType, TutorAction
from src.rewards.engine import RewardEngine


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"
BLUE   = "\033[94m"


def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


def _bar(value: float, max_val: float = 10.0, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    filled = int(round((value / max_val) * width))
    return fill * filled + empty * (width - filled)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_dir: str):
    """Load Unsloth/HF model + tokenizer. Returns (model, tokenizer)."""
    try:
        from unsloth import FastLanguageModel  # type: ignore
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        # Fallback: plain HF transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        return model, tokenizer


# ---------------------------------------------------------------------------
# Action generator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert AI tutor. Always respond with:\n"
    "<STRATEGY>{strategy}</STRATEGY>\n"
    "<RESPONSE>{tutoring response}</RESPONSE>\n"
    "Valid strategies: explain, worked_example, hint, question, correct_fact, analogize, repeat."
)

_EFFECTIVE_MAP: dict[MisconceptionType, TutorAction] = {
    MisconceptionType.PROCEDURAL:  TutorAction.WORKED_EXAMPLE,
    MisconceptionType.CONCEPTUAL:  TutorAction.EXPLAIN,
    MisconceptionType.FACTUAL:     TutorAction.CORRECT_FACT,
    MisconceptionType.TRANSFER:    TutorAction.ANALOGIZE,
}

_RESPONSE_BANK: dict[TutorAction, list[str]] = {
    TutorAction.EXPLAIN:        [
        "Let me walk you through the core concept step by step.",
        "The key idea here is that the rule applies because…",
    ],
    TutorAction.WORKED_EXAMPLE: [
        "Here's a concrete example — watch each step carefully.",
        "Let's solve a simpler version first so you can see the pattern.",
    ],
    TutorAction.HINT:           [
        "Think about what happens if you apply the rule backwards.",
        "What do you notice about the first term?",
    ],
    TutorAction.QUESTION:       [
        "Can you tell me what you think the first step should be?",
        "Why do you think that step comes before the other?",
    ],
    TutorAction.CORRECT_FACT:   [
        "Actually, the correct fact is: the value is always positive here.",
        "That's a common mix-up — the actual definition is slightly different.",
    ],
    TutorAction.ANALOGIZE:      [
        "Think of it like water flowing through pipes — same rules apply.",
        "It's similar to how a map scale works; the ratio stays constant.",
    ],
    TutorAction.REPEAT:         [
        "As I mentioned, the key step is…",
    ],
}

import random as _random

def _rule_based_action(misconception: MisconceptionType, turn: int, prev_action: str | None) -> str:
    """Deterministic rule-based fallback agent for demo without a trained model."""
    strategy = _EFFECTIVE_MAP[misconception]
    # Occasionally vary to avoid repetition penalty
    if turn % 3 == 2:
        strategy = TutorAction.QUESTION
    response = _random.choice(_RESPONSE_BANK[strategy])
    return f"<STRATEGY>{strategy.value}</STRATEGY>\n<RESPONSE>{response}</RESPONSE>"


def _model_action(model, tokenizer, prompt: str) -> str:
    import torch
    inputs = tokenizer(
        [f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>"],
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_header(misconception: MisconceptionType) -> None:
    print()
    print(_c("═" * 66, BOLD))
    print(_c("  EduForge — Demo Rollout", BOLD + CYAN))
    print(_c(f"  Student misconception type: {misconception.value.upper()}", BOLD))
    print(_c("═" * 66, BOLD))
    print()


def _print_turn(
    turn:        int,
    confusion:   float,
    attention:   float,
    student_text:str,
    action_str:  str,
    strategy:    str | None,
    reward:      float,
    done:        bool,
    done_reason: str | None,
) -> None:
    conf_colour = RED if confusion > 6 else (YELLOW if confusion > 3 else GREEN)
    attn_colour = RED if attention < 3 else (YELLOW if attention < 6 else GREEN)

    print(_c(f"┌── Turn {turn:>2} {'─' * 52}", GREY))
    print(f"│  {_c('Student :', CYAN)} {student_text}")
    print(f"│  {_c('Strategy:', BLUE)} {strategy or '(none parsed)'}")

    # Tutor response excerpt
    response_match = re.search(r"<RESPONSE>(.*?)</RESPONSE>", action_str, re.DOTALL | re.I)
    if response_match:
        excerpt = response_match.group(1).strip()[:80]
        print(f"│  {_c('Tutor   :', BLUE)} {excerpt}")

    # Metrics row
    conf_bar  = _bar(confusion,  10.0, width=14)
    attn_bar  = _bar(attention,  10.0, width=14)
    r_colour  = GREEN if reward > 0 else RED
    print(
        f"│  {_c('Confusion', conf_colour)} {conf_bar} {_c(f'{confusion:.2f}', conf_colour)}  "
        f"{_c('Attention', attn_colour)} {attn_bar} {_c(f'{attention:.2f}', attn_colour)}  "
        f"{_c('Reward', r_colour)} {_c(f'{reward:+.4f}', r_colour)}"
    )

    if done:
        symbol = {"success": "✓ RESOLVED", "timeout": "✗ TIMEOUT", "disengaged": "✗ DISENGAGED"}.get(
            done_reason or "", "⊘ DONE"
        )
        colour = GREEN if done_reason == "success" else RED
        print(f"│  {_c(symbol, colour + BOLD)}")

    print(_c("└" + "─" * 64, GREY))


def _print_summary(results: list[dict]) -> None:
    total_r   = sum(r["reward"] for r in results)
    c_start   = results[0]["confusion_before"]
    c_end     = results[-1]["confusion"]
    delta     = c_start - c_end
    done_r    = results[-1].get("done_reason")
    outcome   = _c("✓ SUCCESS", GREEN + BOLD) if done_r == "success" else _c(f"✗ {(done_r or 'unknown').upper()}", RED + BOLD)

    avg_confusion = sum(r["confusion"] for r in results) / len(results)

    action_counts: dict[str, int] = {}
    for r in results:
        s = r["strategy"] or "(none)"
        action_counts[s] = action_counts.get(s, 0) + 1
    total_actions = sum(action_counts.values())

    print()
    print(_c("═" * 66, BOLD))
    print(_c("  Episode Summary", BOLD + CYAN))
    print(_c("═" * 66, BOLD))
    print(f"  Outcome        : {outcome}")
    print(f"  Turns used     : {len(results)}")
    print(f"  Confusion Δ    : {c_start:.2f} → {c_end:.2f}  ({_c(f'-{delta:.2f}', GREEN)})")
    print(f"  Avg confusion  : {_c(f'{avg_confusion:.2f}', YELLOW)}")
    print(f"  Total reward   : {_c(f'{total_r:+.4f}', GREEN if total_r > 0 else RED)}")
    print(_c("  ─" * 33, GREY))
    print(f"  {'Action':<18} {'Count':>5}  {'%':>6}")
    print(_c("  ─" * 33, GREY))
    for act, cnt in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        pct = cnt / total_actions * 100
        print(f"  {act:<18} {cnt:>5}  {pct:>5.1f}%")
    print(_c("═" * 66, BOLD))
    print()

# ---------------------------------------------------------------------------
# Main rollout runner
# ---------------------------------------------------------------------------

def run_rollout(
    model=None,
    tokenizer=None,
    seed: int = 0,
    max_turns: int = 12,
) -> list[dict]:
    env           = EduForgeEnv(seed=seed)
    reward_engine = RewardEngine()
    obs           = env.reset()
    reward_engine.reset()

    _print_header(obs.misconception_id)

    results:     list[dict] = []
    prev_action: str | None = None

    for _ in range(max_turns):
        confusion_before = obs.confusion

        # Generate action
        if model is not None and tokenizer is not None:
            prompt     = f"[Turn {obs.turn}] Student: \"{obs.student_response}\"\nMisconception: {obs.misconception_id.value}"
            action_str = _model_action(model, tokenizer, prompt)
        else:
            action_str = _rule_based_action(obs.misconception_id, obs.turn, prev_action)

        # Environment step
        obs, _, done, info = env.step(action_str)

        # Reward
        format_valid = bool(re.search(r"<STRATEGY>\s*\w+\s*</STRATEGY>", action_str, re.I))
        _, components = reward_engine.compute(
            confusion_before=confusion_before,
            confusion_after=obs.confusion,
            attention_after=obs.attention,
            action_text=action_str,
            format_valid=format_valid,
            done=done,
            done_reason=info.done_reason,
            episode_length=obs.turn,
        )

        strategy = info.parsed_action.value if info.parsed_action else None

        _print_turn(
            turn         = obs.turn,
            confusion    = obs.confusion,
            attention    = obs.attention,
            student_text = obs.student_response,
            action_str   = action_str,
            strategy     = strategy,
            reward       = components.total,
            done         = done,
            done_reason  = info.done_reason,
        )

        results.append({
            "turn":             obs.turn,
            "confusion_before": confusion_before,
            "confusion":        obs.confusion,
            "attention":        obs.attention,
            "strategy":         strategy,
            "reward":           components.total,
            "done_reason":      info.done_reason,
        })

        prev_action = action_str
        if done:
            break

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="EduForge rollout demo")
    parser.add_argument("--model_dir", default=None,  help="Path to trained model directory")
    parser.add_argument("--no_model",  action="store_true", help="Use rule-based fallback (no GPU needed)")
    parser.add_argument("--seed",      type=int, default=0,  help="Episode seed")
    parser.add_argument("--max_turns", type=int, default=12, help="Maximum turns")
    args = parser.parse_args()

    model = tokenizer = None

    if not args.no_model:
        if args.model_dir is None:
            print(_c("No --model_dir given. Using rule-based fallback.", YELLOW))
        else:
            print(_c(f"Loading model from {args.model_dir} …", CYAN))
            model, tokenizer = load_model(args.model_dir)
            print(_c("Model loaded.", GREEN))

    run_rollout(model=model, tokenizer=tokenizer, seed=args.seed, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
