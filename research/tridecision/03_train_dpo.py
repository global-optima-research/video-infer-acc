"""
Step 3: TriDecision DPO training on Qwen2.5-7B-Instruct.

Uses TRL's DPOTrainer with LoRA for parameter-efficient fine-tuning.
Risk-Level-Aware weighting is achieved via weight duplication in Step 2.

Usage (on 5090):
    HF_ENDPOINT=https://hf-mirror.com python3 research/tridecision/03_train_dpo.py

Multi-GPU (if available):
    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --num_processes 8 research/tridecision/03_train_dpo.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PAIRS_PATH = Path(__file__).parent / "preference_pairs.json"
OUTPUT_DIR = Path(__file__).parent / "checkpoints" / "tridecision-pilot"

# Training hyperparameters
LEARNING_RATE = 5e-7
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # effective batch = 8 * 1 = 8 (single GPU)
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
BETA = 0.1  # DPO beta

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_preference_data():
    with open(PAIRS_PATH) as f:
        pairs = json.load(f)

    records = []
    for pair in pairs:
        records.append({
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} preference pairs")
    return dataset


def main():
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # Load data
    dataset = load_preference_data()

    # Split: 80% train, 20% eval (for such small data, mainly for sanity check)
    split = dataset.train_test_split(test_size=0.2, seed=42)

    # Training config
    output_dir = str(OUTPUT_DIR)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"\nTraining config:")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Beta: {BETA}")
    print(f"  Train samples: {len(split['train'])}")
    print(f"  Eval samples: {len(split['test'])}")
    print(f"  Output: {output_dir}")

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_dir = str(OUTPUT_DIR / "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # Log final metrics
    metrics = trainer.evaluate()
    print(f"\nFinal eval metrics: {metrics}")

    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
