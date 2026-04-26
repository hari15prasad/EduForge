import argparse
import json
import torch
import os
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

# Optional Unsloth import
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

def format_conversations(examples):
    texts = []
    for conv in examples["messages"]:
        # Simple format for Gemma
        text = ""
        for msg in conv:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += f"<bos><start_of_turn>model\n{content}<end_of_turn>\n"
            elif role == "user":
                text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                text += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        text += "<eos>"
        texts.append(text)
    return {"text": texts}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2b-it")
    parser.add_argument("--output_dir", type=str, default="./eduforge_checkpoints")
    parser.add_argument("--use_unsloth", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=True)
    args = parser.parse_args()

    # Determine if we can actually use Unsloth
    use_unsloth = args.use_unsloth and HAS_UNSLOTH and torch.cuda.is_available()
    
    if args.use_unsloth and not HAS_UNSLOTH:
        print("⚠️ Unsloth not found. Falling back to standard transformers...")
    elif args.use_unsloth and not torch.cuda.is_available():
        print("⚠️ CUDA not available. Unsloth requires NVIDIA GPU. Falling back to CPU/Standard...")

    max_seq_length = 2048

    if use_unsloth:
        print(f"🚀 Loading {args.model_name} with Unsloth (CUDA Optimized)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print(f"📦 Loading {args.model_name} with standard Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    print("Loading training data...")
    with open("src/environment/training_samples.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Restructure into standard message format
    conversations = []
    for sample in data:
        # We simulate the prompt/response based on the trajectory state
        prompt = f"Student State: Confusion {sample['state']['confusion']:.1f}, Attention {sample['state']['attention']:.1f}, Misconception: {sample['state']['misconception']}\n"
        prompt += f"Selected Action: {sample['action']}"
        
        # In a real setup, response would be the generated LLM output for that action. 
        # Here we just construct a proxy for demonstration.
        response = f"Proceeding with {sample['action']} to address {sample['state']['misconception']}."
        
        conversations.append({
            "messages": [
                {"role": "system", "content": "You are EduForge, an intelligent tutoring agent."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })
        
    dataset = Dataset.from_list(conversations)
    dataset = dataset.map(format_conversations, batched=True)

    print("Starting SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not args.bf16,
            bf16=args.bf16,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
        ),
    )

    # Train the model
    trainer_stats = trainer.train()
    
    print(f"Training completed. Saving to {args.output_dir}...")
    model.save_pretrained(f"{args.output_dir}/lora_model")
    tokenizer.save_pretrained(f"{args.output_dir}/lora_model")
    
    print("Done!")

if __name__ == "__main__":
    main()
