import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model with Unsloth (Optimized for L4)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-it", # or "unsloth/llama-3-8b-instruct"
    max_seq_length = 1024,
    load_in_4bit = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Load your 27k lines (Ensure it's the .jsonl version)
dataset = load_dataset("json", data_files="training_samples.jsonl", split="train")

# 4. Training Arguments (The 500-Error Killers)
training_args = TrainingArguments(
    per_device_train_batch_size = 1,      # Keep this at 1 for L4 stability
    gradient_accumulation_steps = 16,     # Your effective batch size is 16
    warmup_steps = 5,
    max_steps = 1000,                     # Adjust based on your 27k lines
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 1,
    output_dir = "outputs",
    save_strategy = "steps",
    save_steps = 50,
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text", # This MUST match the key in your .jsonl
    max_seq_length = 1024,
    args = training_args,
)

# 6. Train!
trainer.train()