import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add current path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configuration_qu_ssm import QUSSMConfig
from modeling_qu_ssm import QUSSMForCausalLM

def run_math_training(
    model_id: str = "Prannesshkva/QU-SSM-130M-MoE",
    output_dir: str = "./qu_ssm_math_checkpoint",
    num_steps: int = 500,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 70)
    print("🧮 QU-SSM-MoE MATHEMATICS FINE-TUNING PIPELINE")
    print(f"Device: {device.upper()} | Target Steps: {num_steps} | Output: {output_dir}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    print(f"\n[*] Loading Base Model & Tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] Model Loaded: {total_params / 1e6:.2f}M Parameters")

    # 2. Stream GSM8K Math Dataset
    print("\n[*] Streaming GSM8K Mathematical Reasoning Dataset from Hugging Face...")
    dataset = load_dataset("gsm8k", "main", split="train", streaming=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 3. Training Loop
    print("\n" + "-" * 70)
    print(f"{'Step':<8} | {'Math Loss':<12} | {'Tokens Processed':<18} | {'Step Time':<10}")
    print("-" * 70)

    model.train()
    step = 0
    running_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for item in dataset:
        formatted_prompt = (
            f"Question: {item['question']}\n"
            f"Solution:\n{item['answer']}\n<|endoftext|>"
        )

        inputs = tokenizer(
            formatted_prompt,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        input_ids = inputs.input_ids
        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        total_tokens += (input_ids != tokenizer.pad_token_id).sum().item()
        step += 1

        if step % 20 == 0:
            avg_loss = running_loss / 20
            dt = time.time() - t0
            step_time_ms = (dt / 20) * 1000
            print(f"{step:<8} | {avg_loss:<12.4f} | {total_tokens:<18} | {step_time_ms:<8.1f} ms")
            running_loss = 0.0
            t0 = time.time()

        if step >= num_steps:
            break

    # 4. Save Fine-Tuned Checkpoint
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[*] Saving Fine-Tuned Math Weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[✓] Checkpoint saved successfully!")

    # 5. Math Reasoning Evaluation
    print("\n" + "=" * 70)
    print("🧪 POST-TRAINING MATHEMATICAL REASONING TEST")
    print("=" * 70)
    test_question = "Question: Maya has 12 apples. She gives 4 to Tom and buys 6 more. How many apples does Maya have?\nSolution:"
    
    model.eval()
    test_inputs = tokenizer(test_question, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(test_inputs.input_ids, max_new_tokens=60, temperature=0.7)
    
    print("\nPrompt:\n", test_question)
    print("\nQU-SSM-Math Generation:\n", tokenizer.decode(out_ids[0], skip_special_tokens=True))
    print("=" * 70)

if __name__ == "__main__":
    run_math_training(num_steps=100)
