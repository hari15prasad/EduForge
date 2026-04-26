"""
grpo_trainer.py — GRPO training pipeline for EduForge using Unsloth + TRL.
"""

from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
except ImportError:
    _UNSLOTH_AVAILABLE = False
    FastLanguageModel = None

try:
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    GRPOConfig = GRPOTrainer = None

from datasets import Dataset  # type: ignore

from src.environment.openenv_wrapper import EduForgeEnv
from src.environment.student_fsm import MisconceptionType, TutorAction
from src.rewards.engine import RewardEngine
from src.training.curriculum import CurriculumScheduler, Tier, TierParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class EduForgeTrainConfig:
    model_name:            str   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    max_seq_length:        int   = 2048
    lora_rank:             int   = 16
    lora_alpha:            int   = 32
    lora_dropout:          float = 0.05
    target_modules:        list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate:         float = 5e-6
    per_device_batch:      int   = 4
    gradient_accumulation: int   = 4
    num_generations:       int   = 6
    max_new_tokens:        int   = 512
    temperature:           float = 0.9
    max_steps:             int   = 1000
    win_rate_window:       int   = 50
    win_rate_threshold:    float = 0.6
    output_dir:            str   = "outputs/eduforge_grpo"
    logging_steps:         int   = 10
    save_steps:            int   = 100
    seed:                  int   = 42


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI tutor. Your goal is to resolve the student's misconception
within the allowed turns by choosing the most effective tutoring strategy.

Always respond in the following format:
<STRATEGY>{strategy}</STRATEGY>
<RESPONSE>{your tutoring response to the student}</RESPONSE>

Valid strategies: explain, worked_example, hint, question, correct_fact, analogize, repeat.
Choose based on the student's confusion level and what has been tried before.\
"""

def build_prompt(obs_text: str, turn: int, misconception_hint: str) -> str:
    return (
        f"[Turn {turn}] Student says: \"{obs_text}\"\n"
        f"Misconception type (visible): {misconception_hint}\n"
        "Select the best strategy and respond."
    )


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    total_reward:    float
    win:             bool
    confusion_delta: float
    turns_used:      int
    done_reason:     Optional[str]
    component_log:   list[dict]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    model,
    tokenizer,
    tier_params: TierParams,
    reward_engine: RewardEngine,
    seed: Optional[int] = None,
) -> EpisodeResult:
    env = EduForgeEnv(seed=seed)
    obs = env.reset()
    reward_engine.reset()

    _apply_tier_attention(env, tier_params, seed)

    initial_confusion = obs.confusion
    total_reward      = 0.0
    component_log     = []
    done              = False
    info              = None

    misconception_hint = (
        obs.misconception_id.value
        if not tier_params.hidden_misconceptions
        else "hidden"
    )

    while not done:
        prompt = build_prompt(obs.student_response, obs.turn, misconception_hint)
        action_str = _generate_action(model, tokenizer, prompt)

        confusion_before  = obs.confusion
        attention_before  = obs.attention
        misconception_now = obs.misconception_id

        obs, step_reward, done, info = env.step(action_str)

        format_valid  = bool(re.search(r"<STRATEGY>\s*\w+\s*</STRATEGY>", action_str, re.I))
        parsed_action = info.parsed_action  # already resolved by env

        # learning_trend: confusion delta over this step (negative = improving)
        learning_trend = obs.confusion - confusion_before

        _, components = reward_engine.compute(
            confusion_before  = confusion_before,
            confusion_after   = obs.confusion,
            attention_before  = attention_before,
            attention_after   = obs.attention,
            action_text       = action_str,
            format_valid      = format_valid,
            done              = done,
            done_reason       = info.done_reason,
            action            = parsed_action,
            misconception     = misconception_now,
            learning_trend    = learning_trend,
            episode_length    = obs.turn,
        )
        total_reward += components.total
        component_log.append(components.to_dict())

    confusion_delta = initial_confusion - obs.confusion
    win = info.done_reason == "success" if info else False

    return EpisodeResult(
        total_reward    = round(total_reward, 4),
        win             = win,
        confusion_delta = round(confusion_delta, 4),
        turns_used      = env.turn_count,
        done_reason     = info.done_reason if info else None,
        component_log   = component_log,
    )


def _apply_tier_attention(env: EduForgeEnv, params: TierParams, seed) -> None:
    if env._sim is None:
        return
    rng = random.Random(seed)
    jitter   = rng.uniform(-params.attention_jitter, params.attention_jitter)
    new_attn = max(0.0, min(10.0, params.attention_init + jitter))
    env._sim.attention = new_attn


def _generate_action(model, tokenizer, prompt: str) -> str:
    if model is None:
        return "<STRATEGY>explain</STRATEGY>\n<RESPONSE>Let me clarify that for you.</RESPONSE>"

    inputs = tokenizer(
        [f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>"],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens      = 256,
            temperature         = 0.9,
            do_sample           = True,
            pad_token_id        = tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return decoded.strip()


# ---------------------------------------------------------------------------
# Reward functions for GRPOTrainer
# ---------------------------------------------------------------------------

def make_reward_fns(cfg: EduForgeTrainConfig):
    reward_engine = RewardEngine()

    def env_step_reward(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        rewards = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            confusion_before = kwargs.get("confusion_before", 8.0)
            if isinstance(confusion_before, list):
                confusion_before = confusion_before[i] if i < len(confusion_before) else 8.0

            misconception_val = kwargs.get("misconception", None)
            if isinstance(misconception_val, list):
                misconception_val = misconception_val[i] if i < len(misconception_val) else None

            misconception: Optional[MisconceptionType] = None
            if misconception_val is not None:
                try:
                    misconception = MisconceptionType(misconception_val)
                except ValueError:
                    misconception = None

            env = EduForgeEnv(seed=cfg.seed)
            env.reset()

            format_valid = bool(re.search(r"<STRATEGY>\s*\w+\s*</STRATEGY>", completion, re.I))
            obs, _, done, info = env.step(completion)

            parsed_action  = info.parsed_action if info else None
            learning_trend = obs.confusion - confusion_before

            reward_engine.reset()
            total, _ = reward_engine.compute(
                confusion_before  = confusion_before,
                confusion_after   = obs.confusion,
                attention_before  = 8.0,           # default — single-step, no prior
                attention_after   = obs.attention,
                action_text       = completion,
                format_valid      = format_valid,
                done              = done,
                done_reason       = info.done_reason if info else None,
                action            = parsed_action,
                misconception     = misconception,
                learning_trend    = learning_trend,
                episode_length    = obs.turn,
            )
            rewards.append(total)
        return rewards

    def format_reward(
        prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        results = []
        for completion in completions:
            has_strategy = bool(re.search(r"<STRATEGY>\s*\w+\s*</STRATEGY>", completion, re.I))
            has_response = bool(re.search(r"<RESPONSE>.+</RESPONSE>", completion, re.DOTALL | re.I))
            results.append(0.1 if (has_strategy and has_response) else -1.0)
        return results

    return [env_step_reward, format_reward]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

_STUDENT_STARTERS = [
    "I don't understand this at all.",
    "I keep making the same mistake.",
    "Can you explain it a different way?",
    "I thought I understood but I don't.",
    "This step doesn't make sense to me.",
]

def build_dataset(tier_params: TierParams, n_samples: int = 500) -> Dataset:
    rng = random.Random(42)
    records = []
    for _ in range(n_samples):
        misconception = (
            tier_params.fixed_misconception.value
            if tier_params.fixed_misconception
            else rng.choice(list(MisconceptionType)).value
        )
        student_text = rng.choice(_STUDENT_STARTERS)
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n[Turn 1] Student says: \"{student_text}\"\n"
            f"Misconception type (visible): {misconception}\n"
            "Select the best strategy and respond.\n"
            "<|assistant|>"
        )
        records.append({"prompt": prompt, "misconception": misconception})
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Metrics logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    def __init__(self, log_dir: str = "logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "training_metrics.jsonl")
        self._window: list[EpisodeResult] = []

    def record(self, result: EpisodeResult, step: int, tier: Tier) -> None:
        self._window.append(result)

    def flush(self, step: int, tier: Tier) -> dict[str, float]:
        if not self._window:
            return {}

        win_rate       = sum(r.win for r in self._window) / len(self._window)
        avg_reward     = sum(r.total_reward for r in self._window) / len(self._window)
        avg_conf_delta = sum(r.confusion_delta for r in self._window) / len(self._window)
        avg_turns      = sum(r.turns_used for r in self._window) / len(self._window)

        metrics = {
            "step":           step,
            "tier":           tier.name,
            "win_rate":       round(win_rate, 4),
            "avg_reward":     round(avg_reward, 4),
            "avg_conf_delta": round(avg_conf_delta, 4),
            "avg_turns":      round(avg_turns, 2),
            "n_episodes":     len(self._window),
        }

        import json
        with open(self._log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        logger.info(
            "[Step %d | %s] win_rate=%.3f  avg_reward=%.3f  conf_delta=%.3f",
            step, tier.name, win_rate, avg_reward, avg_conf_delta,
        )
        self._window.clear()
        return metrics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: EduForgeTrainConfig | None = None) -> None:
    if cfg is None:
        cfg = EduForgeTrainConfig()

    if not _UNSLOTH_AVAILABLE:
        raise ImportError("unsloth is required. pip install unsloth")
    if not _TRL_AVAILABLE:
        raise ImportError("trl>=0.12 is required. pip install trl")

    logging.basicConfig(level=logging.INFO)
    logger.info("Initialising EduForge GRPO training — model: %s", cfg.model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.model_name,
        max_seq_length = cfg.max_seq_length,
        load_in_4bit   = True,
        fast_inference = False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r                        = cfg.lora_rank,
        lora_alpha               = cfg.lora_alpha,
        lora_dropout             = cfg.lora_dropout,
        target_modules           = cfg.target_modules,
        use_gradient_checkpointing = "unsloth",
        random_state             = cfg.seed,
    )

    scheduler     = CurriculumScheduler(start_tier=Tier.T1)
    metrics_log   = MetricsLogger(log_dir=os.path.join(cfg.output_dir, "logs"))
    reward_fns    = make_reward_fns(cfg)
    reward_engine = RewardEngine()

    tier_params = scheduler.get_params(win_rate=0.0)
    dataset     = build_dataset(tier_params, n_samples=2000)

    grpo_cfg = GRPOConfig(
        learning_rate               = cfg.learning_rate,
        per_device_train_batch_size = cfg.per_device_batch,
        gradient_accumulation_steps = cfg.gradient_accumulation,
        num_generations             = cfg.num_generations,
        max_new_tokens              = cfg.max_new_tokens,
        temperature                 = cfg.temperature,
        max_steps                   = cfg.max_steps,
        output_dir                  = cfg.output_dir,
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        seed                        = cfg.seed,
        report_to                   = "none",
    )

    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        config        = grpo_cfg,
        train_dataset = dataset,
        reward_funcs  = reward_fns,
    )

    step = 0
    while step < cfg.max_steps:
        for _ in range(cfg.win_rate_window):
            result = run_episode(
                model         = model,
                tokenizer     = tokenizer,
                tier_params   = tier_params,
                reward_engine = reward_engine,
                seed          = cfg.seed + step,
            )
            metrics_log.record(result, step, scheduler.current_tier)
            step += 1

        window_metrics = metrics_log.flush(step, scheduler.current_tier)
        win_rate       = window_metrics.get("win_rate", 0.0)
        tier_params    = scheduler.get_params(win_rate=win_rate)

        if scheduler.history()[-1]["event"] == "advance":
            dataset = build_dataset(tier_params, n_samples=2000)
            trainer.train_dataset = dataset
            logger.info("Curriculum advanced to %s — rebuilt dataset.", scheduler.current_tier.name)

        trainer.train()

    model.save_pretrained(os.path.join(cfg.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(cfg.output_dir, "final_model"))
    logger.info("Training complete. Model saved to %s/final_model", cfg.output_dir)
    logger.info("Curriculum summary: %s", scheduler.summary())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EduForge GRPO Trainer")
    parser.add_argument("--model",      default="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    parser.add_argument("--max_steps",  type=int,   default=1000)
    parser.add_argument("--batch",      type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--output_dir", default="outputs/eduforge_grpo")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    train(EduForgeTrainConfig(
        model_name       = args.model,
        max_steps        = args.max_steps,
        per_device_batch = args.batch,
        learning_rate    = args.lr,
        output_dir       = args.output_dir,
        seed             = args.seed,
    ))