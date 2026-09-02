import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import random
import copy
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

QUOSSM_MODEL_ID = "Prannesshkva/QU-SSM-130M-MoE"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNE_STEPS  = 500
LR              = 3e-4
PASSKEY_SAMPLES = 20
PASSKEY_LENGTHS = [256, 512, 1024, 2048]


print(f"Device: {DEVICE}")
print(f"Loading QU-SSM-130M-MoE base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    QUOSSM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(QUOSSM_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Base model loaded.\n")


# ==============================================================================
#  VARIANT CREATION
# ==============================================================================

def make_variant_a(base):
    """
    Variant A: Pure Unitary (Closed Quantum System)
    Freezes gamma to 1.0 at all times. Model CANNOT forget.
    """
    model = copy.deepcopy(base)
    for name, module in model.named_modules():
        if hasattr(module, 'gamma_proj'):
            module.gamma_proj.weight.data.zero_()
            if module.gamma_proj.bias is not None:
                module.gamma_proj.bias.data.fill_(15.0)
            module.gamma_proj.weight.requires_grad_(False)
            if module.gamma_proj.bias is not None:
                module.gamma_proj.bias.requires_grad_(False)
    return model


def make_variant_b(base):
    """
    Variant B: Full QU-SSM (Open Quantum System - Learned gamma_t)
    This is the original architecture unchanged.
    """
    return copy.deepcopy(base)


def make_variant_c(base):
    """
    Variant C: Fixed Exponential Decay (Classical Mamba-style dissipation)
    Replaces learned gamma with a fixed constant decay (gamma = 0.95 always).
    """
    model = copy.deepcopy(base)
    for name, module in model.named_modules():
        if hasattr(module, 'gamma_proj'):
            module.gamma_proj.weight.data.zero_()
            if module.gamma_proj.bias is not None:
                fixed_logit = torch.log(torch.tensor(0.95 / (1.0 - 0.95)))
                module.gamma_proj.bias.data.fill_(fixed_logit.item())
            module.gamma_proj.weight.requires_grad_(False)
            if module.gamma_proj.bias is not None:
                module.gamma_proj.bias.requires_grad_(False)
    return model


# ==============================================================================
#  QUICK FINE-TUNING
# ==============================================================================

def make_finetune_batch(tokenizer, batch_size=4, seq_len=256):
    """
    Creates simple next-token-prediction batches using
    key-value pair memorization sentences.
    """
    texts = []
    for _ in range(batch_size):
        k = random.randint(10000, 99999)
        v = random.randint(10000, 99999)
        texts.append(
            f"Remember: code {k} maps to value {v}. "
            f"The code {k} corresponds to {v}. "
            f"When you see {k}, recall {v}."
        )
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len
    )
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    labels[enc["attention_mask"] == 0] = -100
    return input_ids, labels


def finetune_model(model, tokenizer, steps, lr, label):
    print(f"  Fine-tuning {label} for {steps} steps...")
    model = model.to(DEVICE).train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=0.01)

    total_loss = 0.0
    for step in range(1, steps + 1):
        input_ids, labels = make_finetune_batch(tokenizer)
        input_ids = input_ids.to(DEVICE)
        labels    = labels.to(DEVICE)

        out  = model(input_ids=input_ids, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            avg = total_loss / step
            print(f"    Step {step:>4}/{steps} | Avg Loss: {avg:.4f}")

    model.eval()
    print(f"  Fine-tuning complete. Final Avg Loss: {total_loss/steps:.4f}\n")
    return model


# ==============================================================================
#  PASSKEY RETRIEVAL EVALUATION
# ==============================================================================

def make_padding(tokenizer, num_tokens):
    filler = "the "
    single = len(tokenizer(filler)["input_ids"])
    return filler * max(1, num_tokens // single)


def generate_answer(model, tokenizer, prompt, max_new_tokens=8):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4500).to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


def run_passkey_eval(model, tokenizer, context_lengths, n_samples):
    results = {}
    for ctx_len in context_lengths:
        correct = 0
        for _ in range(n_samples):
            passkey  = str(random.randint(10000, 99999))
            prefix   = f"Remember this secret passkey: {passkey}. "
            question = f"What is the secret passkey? Answer with only the 5-digit number: "

            prefix_toks   = len(tokenizer(prefix)["input_ids"])
            question_toks = len(tokenizer(question)["input_ids"])
            pad_budget    = ctx_len - prefix_toks - question_toks - 5
            padding       = make_padding(tokenizer, max(0, pad_budget))

            prompt = prefix + padding + question
            answer = generate_answer(model, tokenizer, prompt)

            if passkey in answer:
                correct += 1

        acc = correct / n_samples * 100
        results[ctx_len] = acc
        print(f"    Context {ctx_len:>6} tokens: {correct:>2}/{n_samples} = {acc:.1f}%")
    return results


# ==============================================================================
#  MAIN: BUILD, TRAIN, EVALUATE ALL 3 VARIANTS
# ==============================================================================

print("="*72)
print("  BUILDING VARIANTS")
print("="*72)
print("  Creating Variant A (Pure Unitary: gamma frozen to 1.0)...")
model_a = make_variant_a(base_model)
print("  Creating Variant B (Full QU-SSM: learned gamma_t)...")
model_b = make_variant_b(base_model)
print("  Creating Variant C (Fixed Decay: gamma frozen to 0.95)...")
model_c = make_variant_c(base_model)
print("  All variants created.\n")

print("="*72)
print("  FINE-TUNING ALL VARIANTS (identical data, identical steps)")
print("="*72)
model_a = finetune_model(model_a, tokenizer, FINETUNE_STEPS, LR, "Variant A (Pure Unitary, gamma=1.0)")
model_b = finetune_model(model_b, tokenizer, FINETUNE_STEPS, LR, "Variant B (Full QU-SSM, learned gamma)")
model_c = finetune_model(model_c, tokenizer, FINETUNE_STEPS, LR, "Variant C (Fixed Decay, gamma=0.95)")

print("="*72)
print("  PASSKEY RETRIEVAL EVALUATION")
print("="*72)

print("\n  [Variant A: Pure Unitary — gamma=1.0, CANNOT forget]")
res_a = run_passkey_eval(model_a, tokenizer, PASSKEY_LENGTHS, PASSKEY_SAMPLES)

print("\n  [Variant B: Full QU-SSM — Learned gamma_t, CAN selectively forget]")
res_b = run_passkey_eval(model_b, tokenizer, PASSKEY_LENGTHS, PASSKEY_SAMPLES)

print("\n  [Variant C: Fixed Decay — gamma=0.95, ALWAYS decays]")
res_c = run_passkey_eval(model_c, tokenizer, PASSKEY_LENGTHS, PASSKEY_SAMPLES)

print(f"\n{'='*72}")
print("  ABLATION STUDY RESULTS: GAMMA DECOHERENCE GATE COMPARISON")
print(f"{'='*72}")
print(f"  {'Context':>10} | {'A: gamma=1.0':>14} | {'B: Learned':>14} | {'C: gamma=0.95':>14}")
print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
for ctx in PASSKEY_LENGTHS:
    a_acc = res_a.get(ctx, 0)
    b_acc = res_b.get(ctx, 0)
    c_acc = res_c.get(ctx, 0)
    print(f"  {ctx:>10} | {a_acc:>13.1f}% | {b_acc:>13.1f}% | {c_acc:>13.1f}%")

print(f"{'='*72}")
print("""
  INTERPRETATION GUIDE:
  - If B > C at 1k+ tokens: Learned gamma outperforms fixed decay (quantum
    decoherence gate is better than classical Mamba-style dissipation).
  - If B > A at 1k+ tokens: Learnable forgetting is better than pure
    unitary memory (confirming Open Quantum System design is correct).
  - If B wins both: gamma_t acts as a genuine quantum decoherence controller.
""")
